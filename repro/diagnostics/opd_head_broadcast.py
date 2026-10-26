"""Standalone Head broadcast timing. NO Target, gradients or optimizer.
Launch with torchrun on the explicitly selected two UUIDs. Not a QA-parallel gate.
"""
import argparse
import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--expected-world-size',type=int,choices=[2,4],default=2)
    ap.add_argument('--source-kind',choices=['diagnostic','training'],default='diagnostic')
    args=ap.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    world=int(os.environ['WORLD_SIZE'])
    if world!=args.expected_world_size:raise ValueError('world size differs from registered probe')
    if not str(args.out.resolve()).startswith(('/mnt/local/','/mnt/local2/','/mnt/local3/')):raise ValueError('invalid output mount')
    torch.set_num_threads(8);torch.cuda.set_device(local)
    dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    if rank==0:args.out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    state=torch.load(args.state,map_location='cpu',weights_only=False,mmap=True)
    if args.source_kind=='diagnostic':
        if not state['diagnostic_only'] or state['optimizer_steps_executed']!=1:raise ValueError('wrong isolated state')
    elif state.get('update_step')!=120 or state.get('diagnostic_only',False):
        raise ValueError('expected original full step120')
    head=state['head'];names=sorted(head)
    if any(p.dtype!=torch.bfloat16 for p in head.values()):raise ValueError('Head dtype changed')
    # Fixed name order, <=64 MiB chunks, no reduction or conversion.
    buffers=[(head[n].to('cuda') if rank==0 else torch.empty_like(head[n],device='cuda')) for n in names]
    chunks=[c for b in buffers for c in b.reshape(-1).split(32*1024*1024)]
    def broadcast():
        for chunk in chunks:dist.broadcast(chunk,src=0)
    broadcast();torch.cuda.synchronize();dist.barrier()
    times=[]
    for _ in range(3):
        dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
        broadcast();torch.cuda.synchronize();times.append(time.perf_counter()-start)
    equal=all(torch.equal(b.cpu(),head[n]) for n,b in zip(names,buffers))
    records=[None]*world
    dist.all_gather_object(records,dict(rank=rank,times=times,bitwise_equal=equal))
    if rank==0:
        value=dict(status='PASS' if all(r['bitwise_equal'] for r in records) else 'FAIL',
            world_size=world,source_kind=args.source_kind,optimizer_steps=0,target_loaded=False,gradient_computation=False,
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],bytes=sum(b.numel()*b.element_size() for b in buffers),
            chunks=len(chunks),chunk_cap_bytes=64*1024*1024,warmup_broadcasts=1,records=records,
            paired_max_seconds=[max(r['times'][i] for r in records) for i in range(3)],
            scope='standalone transport only; not QA-parallel update or speedup')
        (args.out/'RESULT.json').write_text(json.dumps(value,indent=2))
    dist.destroy_process_group()
    if not equal:raise RuntimeError('broadcast altered Head bits')

if __name__=='__main__':main()
