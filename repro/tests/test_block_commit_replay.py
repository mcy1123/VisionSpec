"""Block shape roundoff must not propagate into next-round Draft context."""
import unittest
import torch
from transformers import DynamicCache
from glance.vl import VisionPositions
from tests.test_opd_verified_rollout import ToyTarget
from opd_official_kl.runtime import rollout

class ShapePerturbedTarget(ToyTarget):
    def __call__(self, ids, *, past_key_values, **kwargs):
        verify = past_key_values.get_seq_length() > 0 and ids.shape[1] > 1
        out = super().__call__(ids, past_key_values=past_key_values, **kwargs)
        if verify:
            # Model the observed shape-dependent fp32 difference, preserving top1.
            out.last_hidden_state = out.last_hidden_state + 1e-4
            out.hidden_states = tuple(h + 1e-4 for h in out.hidden_states)
        return out

class CommitReplayTest(unittest.TestCase):
    def test_block_roundoff_does_not_enter_committed_features(self):
        target=ShapePerturbedTarget();target.training=False
        head=torch.nn.Linear(1,1).eval();head.block_size=4
        def draft(*args, **kwargs):
            return torch.nn.functional.one_hot(torch.tensor([5,3,6]),12).float()
        traces=[]
        for impl in ('serial_reference','block'):
            traces.append(rollout(target,head,torch.tensor([[2,3]]),VisionPositions(),draft_fn=draft,feature_fn=lambda hs:hs[0],cache_factory=DynamicCache,eos={9},max_new_tokens=12,max_seq_len=40,teacher_verify_impl=impl))
        a,b=traces
        for k in ('ids','hidden','fused'):self.assertTrue(torch.equal(a[k],b[k]),k)
        self.assertEqual(a['plan'],b['plan'])
        for x,y in zip(a['rounds'],b['rounds']):
            for k in ('pos','prop','n_acc','committed','rejected_offsets'):self.assertEqual(x[k],y[k])

if __name__=='__main__':unittest.main()
