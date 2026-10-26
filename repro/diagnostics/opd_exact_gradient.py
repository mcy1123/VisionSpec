"""Deterministic frozen-trace gradient-only protocols, NEVER constructs optimizer.
reference: independent legacy + per-QA fp32 inputs.
distributed: original exact fp32 transport, then ordered-anchor bf16 diagnostic.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import datetime
import gc
import json
import os
import time
import torch
import torch.distributed as dist
from shared import common,training_resume
from opd_official_kl import runtime,adapters
from opd_official_kl.trainer import move_trace
from opd_official_kl.efficiency import (capture_gradients,compare_gradients,validate_trace,
    denominator_record,global_denominator,SlotGradientSum)
from opd_efficiency_gate import BASE,write

CAP=64*1024*1024


def identical(a,b):
    return a.keys()==b.keys() and all((x is None and b[n] is None) or
        (x is not None and b[n] is not None and torch.equal(x,b[n])) for n,x in a.items())


def chunks(t):
    return t.reshape(-1).split(CAP//t.element_size())


def load(args,device):
    manifest=json.loads((args.baseline/'manifest_before.json').read_text())
    # Rank0 verifies complete inputs; every rank verifies trace identity below.
    if int(os.environ.get('RANK',0))==0:
        for p,sha in manifest['hashes'].items():
            if training_resume.file_hash(p)!=sha:raise RuntimeError('input changed: '+p)
    target,_=common.load_target(device=device,dtype=torch.float32)
    target.eval().requires_grad_(False)
    head=common.load_head(str(BASE/'step120'),device=device,dtype=torch.bfloat16).eval().requires_grad_(True)
    state=torch.load(BASE/'step120/training_state.pt',map_location='cpu',weights_only=False,mmap=True)
    head.load_state_dict(state['head'])
    rng=dict(state['rng']);cuda_rng=rng.pop('torch_cuda');rng['torch_cuda']=[]
    training_resume.restore_rng(rng)
    if len(cuda_rng)!=1:raise ValueError('expected one-device original RNG state')
    torch.cuda.set_rng_state(cuda_rng[0],device=device)
    traces=torch.load(args.baseline/'fixed_traces.pt',map_location='cpu',weights_only=False)
    for slot,tr in zip(manifest['slots'],traces):
        validate_trace(tr,tr['efficiency_identity'])
        if slot['task']!=tr['efficiency_identity']['task'] or slot['slot']!=tr['efficiency_identity']['slot']:
            raise RuntimeError('slot identity mismatch')
    return manifest,target,head,traces


def score(target,head,tr,device,denom,on_anchor=None):
    gpu=move_trace(tr,device)
    for a,b in zip(tr['rounds'],gpu['rounds']):
        if not torch.equal(a['teacher_logp'],b['teacher_logp'].cpu()):raise RuntimeError('teacher logp transfer changed bits')
    total=0.;count=0
    for i,stat in enumerate(runtime.score(target,head,gpu,draft_fn=adapters.draft)):
        (stat.numerator/denom).backward();total+=float(stat.numerator.detach());count+=1
        if on_anchor:on_anchor(i)
    del gpu
    return total,count


def compare_save(path,ref,got,**extra):
    r=compare_gradients(ref,got);r.update(bitwise=identical(ref,got),**extra);write(path,r);return r


def reference(args):
    manifest,target,head,traces=load(args,'cuda:0')
    slots=manifest['slots'];records=[denominator_record(t,s['slot'],s['task']) for t,s in zip(traces,slots)]
    denom=global_denominator(records,[s['task'] for s in slots])
    head.zero_grad(set_to_none=True);loss=0.
    for tr in traces:loss+=score(target,head,tr,'cuda:0',denom)[0]
    legacy=capture_gradients(head);torch.save(legacy,args.out/'legacy.pt')
    prior=torch.load(args.prior_legacy,map_location='cpu',weights_only=False)
    repeat=compare_save(args.out/'legacy_repeat.json',prior,legacy,loss=loss/denom)
    if not repeat['bitwise']:raise RuntimeError('legacy repeat is not bitwise')
    del prior
    agg=SlotGradientSum([n for n,_ in head.named_parameters()]);qa=[]
    for s,tr in enumerate(traces):
        head.zero_grad(set_to_none=True)
        num,count=score(target,head,tr,'cuda:0',denom)
        g=capture_gradients(head);torch.save(g,args.out/f'slot{s}.pt');agg.add(s,g)
        qa.append(dict(slot=s,numerator=num,anchors=count));del g
    total=agg.finish();torch.save(total,args.out/'qa_fp32.pt')
    compat=compare_save(args.out/'qa_compatibility.json',legacy,total)
    write(args.out/'RESULT.json',dict(status='REFERENCE_READY',denominator_hex=denom.hex(),records=records,
        qa=qa,legacy_repeat_bitwise=True,qa_compatible=compat['passed'],optimizer_steps=0))


def send_cpu(g,names,dst):
    mask=torch.tensor([g[n] is not None for n in names],dtype=torch.uint8);dist.send(mask,dst)
    for n in names:
        if g[n] is not None:
            for c in chunks(g[n]):dist.send(c,dst)


def recv_cpu(params,names,src):
    mask=torch.empty(len(names),dtype=torch.uint8);dist.recv(mask,src)
    g={}
    for n,present in zip(names,mask.tolist()):
        g[n]=torch.empty(params[n].shape,dtype=torch.float32) if present else None
        if present:
            for c in chunks(g[n]):dist.recv(c,src)
    return g


def distributed(args):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world not in (2,4):raise ValueError('only registered 2/4 rank topologies')
    device=f'cuda:{local}';torch.cuda.set_device(local)
    dist.init_process_group('gloo',timeout=datetime.timedelta(minutes=20))
    manifest,target,head,traces=load(args,device)
    slots=manifest['slots'];owned=[s for s in range(4) if s%world==rank]
    records=[denominator_record(traces[s],s,slots[s]['task']) for s in owned]
    gathered=[None]*world;dist.all_gather_object(gathered,records)
    allrecords=[r for group in gathered for r in group]
    denom=global_denominator(allrecords,[s['task'] for s in slots])
    hexvalue=[denom.hex() if rank==0 else None];dist.broadcast_object_list(hexvalue,src=0)
    if denom.hex()!=hexvalue[0]:raise RuntimeError('global denominator bit mismatch')
    names=sorted(n for n,_ in head.named_parameters());params=dict(head.named_parameters())
    # Protocol A: preserve QA bf16 accumulation, transport CPU fp32 contributions exactly.
    localg={};local_reports=[];start=time.perf_counter()
    for s in owned:
        head.zero_grad(set_to_none=True);num,count=score(target,head,traces[s],device,denom)
        localg[s]=capture_gradients(head)
        ref=torch.load(args.reference/f'slot{s}.pt',map_location='cpu',weights_only=False)
        r=compare_gradients(ref,localg[s]);r.update(slot=s,bitwise=identical(ref,localg[s]),numerator=num,anchors=count)
        local_reports.append(r);del ref
    head.zero_grad(set_to_none=True)
    dist.barrier()
    if rank==0:
        agg=SlotGradientSum(names)
        for s in range(4):
            owner=s%world;g=localg[s] if owner==0 else recv_cpu(params,names,owner)
            agg.add(s,g)
        summed=agg.finish()
        ref=torch.load(args.reference/'qa_fp32.pt',map_location='cpu',weights_only=False)
        transport=compare_save(args.out/'qa_transport.json',ref,summed,seconds=time.perf_counter()-start)
        del ref
        legacy=torch.load(args.reference/'legacy.pt',map_location='cpu',weights_only=False)
        compare_save(args.out/'qa_vs_legacy.json',legacy,summed)
        del summed,agg
    else:
        for s in owned:send_cpu(localg[s],names,0)
        transport=None
    allqa=[None]*world;dist.all_gather_object(allqa,local_reports)
    ok=[transport['bitwise'] and all(r['bitwise'] for group in allqa for r in group) if rank==0 else None]
    dist.broadcast_object_list(ok,src=0)
    write(args.out/f'rank{rank}_qa.json',dict(owned=owned,qa=local_reports,denominator_hex=denom.hex()))
    if not ok[0]:raise RuntimeError('exact fp32 transport/per-QA implementation failed')
    del localg;gc.collect()
    # Protocol B: original slot/anchor BF16 addition order, zero leaf grads every anchor.
    nccl=dist.new_group(backend='nccl',timeout=datetime.timedelta(minutes=20))
    init=torch.zeros(1,device=device);dist.broadcast(init,src=0,group=nccl);torch.cuda.synchronize()
    accumulator=dict.fromkeys(names) if rank==0 else None
    receive={n:torch.empty_like(params[n]) for n in names} if rank==0 else None
    events=[];nums={};start=time.perf_counter()
    def contribute(slot,index):
        mask=torch.tensor([params[n].grad is not None for n in names],dtype=torch.uint8)
        if rank:
            dist.send(torch.tensor([slot,index],dtype=torch.int64),0);dist.send(mask,0)
            for n,present in zip(names,mask.tolist()):
                if present:
                    for c in chunks(params[n].grad):dist.send(c,0,group=nccl)
        else:
            for n,present in zip(names,mask.tolist()):
                if present:
                    g=params[n].grad.detach()
                    if accumulator[n] is None:accumulator[n]=g.clone()
                    else:accumulator[n].add_(g)
        head.zero_grad(set_to_none=True)
    for s in range(4):
        owner=s%world
        if rank==owner:
            head.zero_grad(set_to_none=True)
            nums[s]=score(target,head,traces[s],device,denom,lambda i,s=s:contribute(s,i))
        elif rank==0:
            head.zero_grad(set_to_none=True)
            for i in range(len(traces[s]['plan'])):
                header=torch.empty(2,dtype=torch.int64);dist.recv(header,owner)
                if header.tolist()!=[s,i]:raise RuntimeError('slot/anchor sequence mismatch')
                mask=torch.empty(len(names),dtype=torch.uint8);dist.recv(mask,owner)
                for n,present in zip(names,mask.tolist()):
                    if present:
                        for c in chunks(receive[n]):dist.recv(c,owner,group=nccl)
                        if accumulator[n] is None:accumulator[n]=receive[n].clone()
                        else:accumulator[n].add_(receive[n])
        if rank==0:events.append(dict(slot=s,anchors=len(traces[s]['plan']),owner=owner))
    torch.cuda.synchronize();dist.barrier();elapsed=time.perf_counter()-start
    if rank==0:
        exact={n:None if g is None else g.cpu().float() for n,g in accumulator.items()}
        result=compare_save(args.out/'anchor_vs_legacy.json',legacy,exact,seconds=elapsed,events=events)
        torch.save(exact,args.out/'anchor_ordered_gradients.pt')
        write(args.out/'RESULT.json',dict(status='PASS' if result['bitwise'] else 'HOLD',world_size=world,
            qa_fp32_transport_exact=True,anchor_ordered_legacy_bitwise=result['bitwise'],optimizer_steps=0,
            denominator_hex=denom.hex(),target_backbone_forwards=0,trace_source='frozen',live_rollout_validated=False,
            timing_scope='diagnostic includes synchronization, transfers and gradient comparisons; NOT full update'))
    write(args.out/f'rank{rank}_state.json',dict(rank=rank,owned=owned,numerators=nums,
        cuda_rng_hex=bytes(torch.cuda.get_rng_state(local).tolist()).hex(),optimizer_steps=0))
    dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['reference','distributed'],required=True)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',type=Path);ap.add_argument('--prior-legacy',type=Path)
    args=ap.parse_args()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic startup environment required')
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    training_resume.validate_output(args.out)
    if args.mode=='reference':args.out.mkdir(parents=True,exist_ok=False)
    elif int(os.environ['RANK'])==0:args.out.mkdir(parents=True,exist_ok=False)
    # rank0 publishes the directory before other ranks write anything.
    if args.mode=='distributed':
        while not args.out.exists():time.sleep(.1)
    write(args.out/f'RUNNING_rank{os.environ.get("RANK","0")}.json',dict(mode=args.mode,optimizer_steps=0))
    if args.mode=='reference':reference(args)
    else:distributed(args)

if __name__=='__main__':main()
