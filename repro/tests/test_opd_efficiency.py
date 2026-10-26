import unittest
import torch
from opd_official_kl.efficiency import global_denominator, denominator_record, SlotGradientSum
from shared.master_optim import MasterWeightAdamW


class Contracts(unittest.TestCase):
    def test_denominator_slot_order_and_failures(self):
        traces = [dict(ids=torch.zeros(12+i), prompt_len=3,
                       rounds=[dict(rejected_offsets=[1, 3, 15])]) for i in range(4)]
        rows = [denominator_record(t, i, str(i)) for i,t in enumerate(traces)]
        expected = sum(float.fromhex(r['denominator_hex']) for r in rows)
        self.assertEqual(global_denominator(rows[::-1], list('0123')).hex(), expected.hex())
        for bad in [rows[:3], rows[:3]+[rows[0]], [dict(rows[0], task='wrong')]+rows[1:],
                    [dict(rows[0], denominator_hex=(1.0).hex())]+rows[1:]]:
            with self.assertRaises(ValueError): global_denominator(bad, list('0123'))

    def test_order_presence_and_external_optimizer(self):
        model = torch.nn.Linear(2, 1).to(torch.bfloat16)
        opt = MasterWeightAdamW(model, lr=1e-5, betas=(.9,.95), weight_decay=0)
        agg = SlotGradientSum(['weight','bias'])
        for slot in range(4):
            agg.add(slot, {'weight':torch.full((1,2), float(slot)), 'bias':None})
        grads = agg.finish()
        self.assertTrue(torch.equal(grads['weight'], torch.full((1,2),6.0)))
        self.assertIsNone(grads['bias'])
        # Stale module gradients must never replace the external sum or None.
        for p in model.parameters(): p.grad = torch.ones_like(p)
        opt.set_external_master_grads(grads)
        self.assertEqual(opt.step(collect_update_norms=False), {})
        self.assertIsNone(opt.master['bias'].grad)
        self.assertNotIn(opt.master['bias'], opt._opt.state)
        self.assertTrue(torch.equal(opt.master['weight'].grad, grads['weight']))
        with self.assertRaises(ValueError): agg.add(4, grads)

    def test_none_presence_and_invalid_external_are_atomic(self):
        agg=SlotGradientSum(['a','b'], slots=2)
        agg.add(0, {'a':None,'b':torch.tensor([2.])})
        agg.add(1, {'a':torch.tensor([3.]),'b':None})
        got=agg.finish()
        self.assertEqual(got['a'].item(),3.)
        self.assertEqual(got['b'].item(),2.)
        model=torch.nn.Linear(2,1).to(torch.bfloat16)
        opt=MasterWeightAdamW(model,lr=1e-5,betas=(.9,.95),weight_decay=0)
        with self.assertRaises(ValueError):
            opt.set_external_master_grads({'weight':torch.ones(1,2),'bias':torch.ones(2)})
        self.assertTrue(all(p.grad is None for p in opt.master.values()))
        self.assertFalse(opt._external_grads_ready)

    def test_gradient_gate_detects_per_tensor_failure(self):
        from opd_official_kl.efficiency import compare_gradients
        a={'large':torch.ones(1000),'small':torch.tensor([.01]),'unused':None}
        b={'large':torch.ones(1000),'small':torch.tensor([.011]),'unused':None}
        report=compare_gradients(a,b)
        self.assertGreater(report['cosine'],.9999)
        self.assertLess(report['relative_l2'],.01)
        self.assertFalse(report['passed'])
        self.assertFalse(next(x for x in report['tensors'] if x['name']=='small')['passed'])

    def test_slot_sum_not_associative(self):
        agg=SlotGradientSum(['x'], slots=3)
        for i,x in enumerate([1e20,-1e20,1.]): agg.add(i, {'x':torch.tensor([x])})
        self.assertEqual(agg.finish()['x'].item(),1.)
        with self.assertRaises(ValueError): SlotGradientSum(['x']).add(1,{'x':None})

class SerialSwitches(unittest.TestCase):
    def test_rollout_and_replay_exact(self):
        from tests.test_opd_verified_rollout import ToyTarget
        from transformers import DynamicCache
        from glance.vl import VisionPositions
        from opd_official_kl import runtime
        from opd_official_kl.efficiency import bind_trace, validate_trace
        from opd_official_kl.trainer import move_trace
        target=ToyTarget();target.training=False
        head=torch.nn.Linear(1,1).eval();head.block_size=16
        def draft(h,t,fused,**kw):
            return torch.nn.functional.one_hot(torch.zeros(15,dtype=torch.long),12).float()*2+h.weight.reshape(1,1)*torch.arange(12)*.001
        for cap in (1,2,17):
            traces=[]; gradients=[]
            for optimized in (False,True):
                tr=runtime.rollout(target,head,torch.tensor([[2,3]]),VisionPositions(),
                    draft_fn=draft,feature_fn=lambda hs:hs[0],cache_factory=DynamicCache,
                    eos={9},max_new_tokens=cap,max_seq_len=48,
                    batch_teacher_ids=optimized,preallocate_context=optimized)
                identity=dict(checkpoint_sha256='head',task='qa',slot=0,prompt_sha256='prompt',config_sha256='cfg')
                bind_trace(tr,identity);validate_trace(move_trace(tr,'cpu'),identity)
                with self.assertRaises(ValueError):validate_trace(tr,dict(identity,slot=1))
                head.zero_grad(set_to_none=True)
                for stat in runtime.score(target,head,tr,draft_fn=draft,deferred_checks=optimized):
                    (stat.numerator/runtime.denominator(tr)).backward()
                gradients.append(None if head.weight.grad is None else head.weight.grad.clone())
                traces.append(tr)
            for key in ('ids','hidden','fused'):self.assertTrue(torch.equal(traces[0][key],traces[1][key]))
            for key in ('plan','boundaries'):self.assertEqual(traces[0][key],traces[1][key])
            for a,b in zip(traces[0]['rounds'],traces[1]['rounds']):
                for key in a:
                    if torch.is_tensor(a[key]):self.assertTrue(torch.equal(a[key],b[key]))
                    else:self.assertEqual(a[key],b[key])
            self.assertTrue(all(g is None for g in gradients) if gradients[0] is None else torch.equal(*gradients))
            if tr['rounds']:
                tr['rounds'][0]['prop'][0]=7
                with self.assertRaises(RuntimeError):
                    list(runtime.score(target,head,tr,draft_fn=draft,deferred_checks=True))

class ProfileSummary(unittest.TestCase):
    def test_streaming_nested_and_expired_scopes(self):
        import json,tempfile
        from pathlib import Path
        from diagnostics.summarize_opd_profile_stream import summarize
        events=[dict(cat='user_annotation',ph='X',name='outer',pid=1,tid=1,ts=0,dur=20),
                dict(cat='user_annotation',ph='X',name='inner',pid=1,tid=1,ts=2,dur=3),
                dict(cat='cuda_runtime',name='cudaLaunchKernel',pid=1,tid=1,ts=3,args={'correlation':10}),
                dict(cat='cuda_runtime',name='cudaLaunchKernel',pid=1,tid=1,ts=7,args={'correlation':11}),
                dict(cat='kernel',dur=100,args={'correlation':10}),
                dict(cat='kernel',dur=200,args={'correlation':11}),
                dict(cat='user_annotation',ph='X',name='anchor_backward',pid=1,tid=1,ts=8,dur=3),
                dict(cat='cuda_runtime',name='cudaLaunchKernel',pid=1,tid=2,ts=9,args={'correlation':12}),
                dict(cat='kernel',dur=300,args={'correlation':12})]
        with tempfile.TemporaryDirectory(dir='/mnt/local3/mcy/GLANCE/results') as tmp:
            p=Path(tmp)/'profile.json';p.write_text(json.dumps({'traceEvents':events}))
            got=summarize(p)
        self.assertEqual(got['kernel_count'],3)
        self.assertEqual(got['gpu_kernel_seconds_by_innermost_launch_scope'],{'inner':.0001,'outer':.0002,'anchor_backward_cross_thread':.0003})

if __name__ == '__main__': unittest.main()
