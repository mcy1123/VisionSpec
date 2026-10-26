"""Transport, policy-version, detached-Target and irreversible budget contracts."""
import datetime
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
import torch
import torch.distributed as dist
from opd_official_kl.rollout_parallel import (pack_trace, unpack_trace, raw_digest,
    send_trace, receive_trace, validate_envelope, check_versions, budget_event)


def fixture():
    policy = dict(head_sha256='a'*64, optimizer_step=120, schedule_cursor=119)
    identity = dict(checkpoint_sha256='a'*64, task='qa', slot=0,
                    prompt_sha256='prompt', config_sha256='config')
    trace = dict(ids=torch.arange(4), prompt_len=2, hidden=torch.randn(1,4,2),
        fused=torch.randn(1,4,6), rounds=[dict(prop=list(range(15)),
        teacher_ids=list(range(16)), teacher_logp=torch.linspace(-4,0,15),
        rejected_offsets=[1,15])], plan=[(2,1)], boundaries=[1],
        final_cache_length=4, efficiency_identity=identity, policy_identity=policy)
    trace['hidden'][0,0,0] = -0.
    return trace, identity, policy


def transport_worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method='file://'+path, timeout=datetime.timedelta(seconds=30))
    if rank:
        tr, identity, policy = fixture()
        length = send_trace(tr)
        received, n = receive_trace(0)
        assert length == n
        assert raw_digest(tr['hidden']) == raw_digest(received['hidden'])
        validate_envelope(received, identity, policy)
    else:
        received, n = receive_trace(1)
        validate_envelope(received, received['efficiency_identity'], received['policy_identity'])
        assert send_trace(received, 1) == n
    dist.destroy_process_group()


class RolloutParallelTest(unittest.TestCase):
    def test_trace_roundtrip_bits_and_teacher(self):
        tr, identity, policy = fixture()
        b, sha = pack_trace(tr)
        got = unpack_trace(b, sha)
        validate_envelope(got, identity, policy)
        self.assertEqual(raw_digest(tr['hidden']), raw_digest(got['hidden']))
        self.assertEqual(raw_digest(tr['rounds'][0]['teacher_logp']),
                         raw_digest(got['rounds'][0]['teacher_logp']))
        with self.assertRaises(ValueError):unpack_trace(b[:-1]+b'x', sha)

    def test_stale_head_and_version_barrier(self):
        tr, identity, policy = fixture()
        check_versions([policy, dict(policy)])
        stale = dict(policy, optimizer_step=119)
        with self.assertRaises(ValueError):check_versions([policy, stale])
        with self.assertRaises(ValueError):validate_envelope(tr, identity, stale)
        with self.assertRaises(ValueError):check_versions([{}])

    def test_student_graph_cache_forbidden(self):
        tr, identity, policy = fixture()
        tr['student_kv'] = torch.zeros(2)
        with self.assertRaises(ValueError):validate_envelope(tr, identity, policy)
        del tr['student_kv']
        tr['fused'].requires_grad_(True)
        with self.assertRaises(ValueError):validate_envelope(tr, identity, policy)

    def test_teacher_precision_and_alignment(self):
        tr, identity, policy = fixture()
        tr['rounds'][0]['teacher_logp'] = tr['rounds'][0]['teacher_logp'].bfloat16()
        with self.assertRaises(ValueError):validate_envelope(tr, identity, policy)
        tr, identity, policy = fixture()
        tr['hidden'] = tr['hidden'][:,:3]
        with self.assertRaises(ValueError):validate_envelope(tr, identity, policy)

    def test_no_refund_duplicate_and_hard_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'budget.json'
            p.write_text(json.dumps(dict(max_attempts=2,cases={'a':{'updates':3}},attempts=[],completed=[])))
            budget_event(p,'a','reserve',0)
            with self.assertRaises(ValueError):budget_event(p,'a','reserve',0)
            budget_event(p,'a','complete',0)
            with self.assertRaises(ValueError):budget_event(p,'a','complete',0)
            budget_event(p,'a','reserve',1)
            with self.assertRaises(ValueError):budget_event(p,'a','reserve',2)
            self.assertEqual(len(json.loads(p.read_text())['attempts']),2)

    def test_live_gloo_wire_roundtrip(self):
        context=multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as tmp:
            ps=[context.Process(target=transport_worker,args=(rank,str(Path(tmp)/'init'))) for rank in range(2)]
            for p in ps:p.start()
            for p in ps:p.join(40)
            try:self.assertTrue(all(not p.is_alive() and p.exitcode==0 for p in ps))
            finally:
                for p in ps:
                    if p.is_alive():p.terminate();p.join()

if __name__=='__main__':unittest.main()
