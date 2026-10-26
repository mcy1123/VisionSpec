"""Serial/block state-machine equality; no real model or updates."""
import unittest
from unittest.mock import patch
import torch
from tests.test_opd_verified_rollout import ToyTarget
from transformers import DynamicCache
from glance.vl import VisionPositions
from opd_official_kl import runtime, adapters


class BlockVerificationTests(unittest.TestCase):
    def test_state_and_gradient_matrix(self):
        for proposal in ([0, 0, 0], [5, 3, 6], [5, 0, 0], [5, 3, 9], [9, 5, 3]):
            for eos in ({9}, {6}, {5}, {3}, {7}):
                for cap in (1, 2, 4, 9):
                    with self.subTest(proposal=proposal, eos=eos, cap=cap):
                        target = ToyTarget(); target.training = False
                        head = torch.nn.Linear(1, 1).eval(); head.block_size = 4
                        def draft(h, t, fused, **kw):
                            return torch.nn.functional.one_hot(torch.tensor(proposal), 12).float()*2 + h.weight.reshape(1,1)*torch.arange(12)*.001
                        traces=[]; gradients=[]; observations=[]
                        for impl in ('serial_reference', 'block'):
                            obs=[]
                            tr=runtime.rollout(target,head,torch.tensor([[2,3]]),VisionPositions(),draft_fn=draft,feature_fn=lambda hs:hs[0],cache_factory=DynamicCache,eos=eos,max_new_tokens=cap,max_seq_len=40,teacher_verify_impl=impl,verification_observer=lambda p,t,l:obs.append((p,t,l.clone())))
                            traces.append(tr); observations.append(obs)
                            head.zero_grad()
                            for stat in runtime.score(target,head,tr,draft_fn=draft):
                                (stat.numerator/runtime.denominator(tr)).backward()
                            gradients.append(None if head.weight.grad is None else head.weight.grad.clone())
                        a,b=traces
                        for key in ('ids','fused','hidden'):
                            self.assertTrue(torch.equal(a[key],b[key]),key)
                        for key in ('plan','boundaries','final_cache_length'):
                            self.assertEqual(a[key],b[key])
                        self.assertEqual(len(a['rounds']),len(b['rounds']))
                        for x,y in zip(a['rounds'],b['rounds']):
                            for k in x:
                                if torch.is_tensor(x[k]): self.assertTrue(torch.equal(x[k],y[k]))
                                else: self.assertEqual(x[k],y[k])
                        self.assertTrue(all(g is None for g in gradients) if gradients[0] is None else torch.equal(*gradients))
                        for x,y in zip(*observations):
                            self.assertEqual(x[:2],y[:2]); self.assertTrue(torch.equal(x[2],y[2]))

    def test_unvalidated_block_cannot_train(self):
        import contextlib, io
        from opd_official_kl.cli import main
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                main(["--teacher-verify-impl", "block"])
            self.assertEqual(cm.exception.code, 2)

    def test_adapter_opt_in(self):
        head=torch.nn.Linear(1,1)
        with patch('shared.draft_geometry.eos_ids', return_value={9}), patch.object(runtime,'rollout',return_value='ok') as fn:
            self.assertEqual(adapters.collect(None,head,{'input_ids':torch.tensor([[1]])},max_new_tokens=4,max_seq_len=20,teacher_verify_impl='block'),'ok')
            self.assertEqual(fn.call_args.kwargs['teacher_verify_impl'],'block')

if __name__ == '__main__': unittest.main()
