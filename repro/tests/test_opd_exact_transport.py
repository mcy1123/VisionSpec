"""CPU integration test for the actual point-to-point presence protocol."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'diagnostics'))
import datetime
import multiprocessing
import tempfile
import unittest
import torch
import torch.distributed as dist
from opd_exact_gradient import send_cpu,recv_cpu
from opd_official_kl.ordered_parallel import OrderedAnchorGradients


def worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+path,timeout=datetime.timedelta(seconds=30))
    names=['a','b','c'];params={n:torch.empty(3) for n in names}
    if rank==0:
        send_cpu(dict(a=torch.tensor([1.,-2.,3.]),b=None,c=torch.zeros(3)),names,1)
        got=recv_cpu(params,names,1)
        assert got['a'] is None and torch.equal(got['b'],torch.tensor([4.,5.,6.]))
        assert got['c'] is None
    else:
        got=recv_cpu(params,names,0)
        assert got['b'] is None and torch.equal(got['a'],torch.tensor([1.,-2.,3.]))
        assert got['c'] is not None and torch.equal(got['c'],torch.zeros(3))
        send_cpu(dict(a=None,b=torch.tensor([4.,5.,6.]),c=None),names,0)
    dist.destroy_process_group()


def ordered_worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+path,timeout=datetime.timedelta(seconds=30))
    head=torch.nn.Module()
    for n in ['a','b','c']:head.register_parameter(n,torch.nn.Parameter(torch.zeros(3,dtype=torch.bfloat16)))
    reducer=OrderedAnchorGradients(head,dist.group.WORLD,rank,2)
    if rank==0:
        head.b.grad=torch.tensor([1.,2.,3.],dtype=torch.bfloat16)
        reducer.contribute(0,0);reducer.receive_slot(1,1);got=reducer.finish()
        assert torch.equal(got['a'],torch.tensor([4.,5.,6.]))
        assert torch.equal(got['b'],torch.tensor([1.,2.,3.])) and got['c'] is None
    else:
        head.a.grad=torch.tensor([4.,5.,6.],dtype=torch.bfloat16)
        reducer.contribute(1,0)
    dist.destroy_process_group()


class ExactTransportTest(unittest.TestCase):
    def test_ordered_anchor_absence(self):
        self.run_pair(ordered_worker)

    def test_two_rank_presence_roundtrip(self):
        self.run_pair(worker)

    def run_pair(self,fn):
        context=multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as d:
            ps=[context.Process(target=fn,args=(rank,str(Path(d)/'rendezvous'))) for rank in range(2)]
            for p in ps:p.start()
            for p in ps:p.join(40)
            try:
                self.assertTrue(all(not p.is_alive() and p.exitcode==0 for p in ps))
            finally:
                for p in ps:
                    if p.is_alive():p.terminate();p.join()

if __name__=='__main__':unittest.main()
