"""Native block KV ownership/call count; serial exactness is not a GPU gate."""
import unittest
from unittest.mock import patch
from _bootstrap import configure
configure()
import torch
from transformers import DynamicCache
from glance.vl import VisionPositions
from tests.test_opd_verified_rollout import ToyTarget
from opd_official_kl import runtime

class CountTarget(ToyTarget):
    def __init__(self):
        super().__init__();self.training=False;self.calls=[]
    def __call__(self,ids,**kw):
        self.calls.append(ids.shape[1]);return super().__call__(ids,**kw)

class NativeTests(unittest.TestCase):
    def test_causal_kv_eos_cap_and_one_block_call_per_round(self):
        for prop in [[0,0,0],[5,3,6],[5,0,0],[5,3,9]]:
            for eos in [{9},{6},{5}]:
                for cap in [1,2,4,9]:
                    traces=[]
                    for mode in ['serial_reference','pure_block_native']:
                        target=CountTarget();head=torch.nn.Linear(1,1).eval();head.block_size=4
                        def draft(h,t,fused,**kw):
                            return torch.nn.functional.one_hot(torch.tensor(prop),12).float()*2
                        tr=runtime.rollout(target,head,torch.tensor([[2,3]]),VisionPositions(),draft_fn=draft,
                            feature_fn=lambda hs:hs[0],cache_factory=DynamicCache,eos=eos,
                            max_new_tokens=cap,max_seq_len=40,teacher_verify_impl=mode)
                        traces.append(tr)
                        if mode=='pure_block_native':
                            self.assertEqual(target.calls,[2]+[4]*len(tr['rounds'])+[1])
                            self.assertEqual(tr['teacher_verify_impl'],mode)
                    a,b=traces
                    for k in ['ids','hidden','fused']:self.assertTrue(torch.equal(a[k],b[k]),k)
                    for k in ['plan','boundaries','final_cache_length']:self.assertEqual(a[k],b[k])

    def test_serial_cli_still_rejects_old_block_training(self):
        from opd_official_kl.cli import main
        import io,contextlib
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):main(['--teacher-verify-impl','block'])
            with self.assertRaises(SystemExit):main(['--teacher-verify-impl','pure_block_native'])

if __name__=='__main__':unittest.main()
