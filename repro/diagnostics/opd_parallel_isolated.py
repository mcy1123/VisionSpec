"""Experimental ONE isolated QA-parallel update. Default is live no-update check.
No long training/resume entry. diagnostic_only outputs cannot resume production.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import datetime
import hashlib
import json
import math
import os
import time
import torch
import torch.distributed as dist
from shared import common,training_resume
from shared.master_optim import MasterWeightAdamW
from opd_official_kl import adapters,runtime
from opd_official_kl.trainer import move_trace
from opd_official_kl.efficiency import denominator_record,global_denominator,compare_gradients
from opd_official_kl.ordered_parallel import OrderedAnchorGradients,broadcast_head
from opd_serial_candidate import compare_trace
from opd_exact_gradient import score,identical
from opd_efficiency_gate import BASE,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--kernel-mode',choices=['deterministic','default'],default='deterministic')
    ap.add_argument('--authorize-one-isolated-step',action='store_true')
    args=ap.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world not in (2,4):raise ValueError('2/4 ranks only')
    det=args.kernel_mode=='deterministic'
    if det!=(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'):raise ValueError('kernel environment mismatch')
    training_resume.validate_output(args.out)
    torch.set_num_threads(8);torch.use_deterministic_algorithms(det);torch.cuda.set_device(local);device=f'cuda:{local}'
    dist.init_process_group('gloo',timeout=datetime.timedelta(minutes=30))
    if rank==0:args.out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    manifest=json.loads((args.baseline/'manifest_before.json').read_text());cfg=manifest['config'];slots=manifest['slots']
    if rank==0:
        for p,sha in manifest['hashes'].items():
            if training_resume.file_hash(p)!=sha:raise RuntimeError('input changed')
    import pandas as pd
    rows=pd.read_parquet(cfg['pool']).set_index('task',verify_integrity=True)
    target,processor=common.load_target(device=device,dtype=torch.float32);target.eval().requires_grad_(False)
    head=common.load_head(str(BASE/'step120'),device=device,dtype=torch.bfloat16).eval().requires_grad_(True)
    state=torch.load(BASE/'step120/training_state.pt',map_location='cpu',weights_only=False,mmap=not args.authorize_one_isolated_step)
    head.load_state_dict(state['head']);rng=dict(state['rng']);saved_cuda=rng['torch_cuda'];rng['torch_cuda']=[]
    optimizer=None
    if rank==0 and args.authorize_one_isolated_step:
        optimizer=MasterWeightAdamW(head,lr=cfg['lr'],betas=common.BETAS,weight_decay=common.WEIGHT_DECAY,where='cpu',clip=cfg['clip'])
        optimizer.load_state_dict(state['master_optimizer'])
    del state
    refs=torch.load(args.baseline/'fixed_traces.pt',map_location='cpu',weights_only=False)
    nccl=dist.new_group(backend='nccl',timeout=datetime.timedelta(minutes=30))
    init=torch.zeros(1,device=device);dist.broadcast(init,0,group=nccl)
    reducer=OrderedAnchorGradients(head,nccl,rank,world);owned=[s for s in range(4) if s%world==rank]
    forward_counts={'target_model':0,'lm_head':0,'draft_head':0}
    def counter(key):
        def hook(module,inputs):forward_counts[key]+=1
        return hook
    target.model.register_forward_pre_hook(counter('target_model'))
    target.lm_head.register_forward_pre_hook(counter('lm_head'))
    head.register_forward_pre_hook(counter('draft_head'))
    training_resume.restore_rng(rng);torch.cuda.set_rng_state(saved_cuda[0],local)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();dist.barrier();start=time.perf_counter();traces={}
    for s in owned:
        inputs=adapters.encode(processor,rows.loc[slots[s]['task']].to_dict(),'image',max_pixels=cfg['max_pixels'])
        tr=adapters.collect(target,head,inputs,max_new_tokens=cfg['max_new_tokens'],max_seq_len=cfg['max_seq_len'])
        tr['efficiency_identity']=refs[s]['efficiency_identity'];traces[s]=move_trace(tr,'cpu');del tr,inputs
        # Before any step: full discrete + teacher logp + committed context contract.
        compare_trace(refs[s],traces[s])
    dist.barrier();rollout_end=time.perf_counter()
    local_records=[denominator_record(traces[s],s,slots[s]['task']) for s in owned]
    gathered=[None]*world;dist.all_gather_object(gathered,local_records)
    records=[r for group in gathered for r in group];denom=global_denominator(records,[s['task'] for s in slots])
    hexvalue=[denom.hex() if rank==0 else None];dist.broadcast_object_list(hexvalue,0)
    if denom.hex()!=hexvalue[0]:raise RuntimeError('denominator bit mismatch')
    nums={}
    for s in range(4):
        owner=s%world
        if rank==owner:
            head.zero_grad(set_to_none=True)
            nums[s]=score(target,head,traces[s],device,denom,lambda index,s=s:reducer.contribute(s,index))
        elif rank==0:reducer.receive_slot(s,len(refs[s]['plan']))
    torch.cuda.synchronize();dist.barrier();backward_end=time.perf_counter()
    grad=None;comparison=None
    if rank==0:
        grad=reducer.finish()
        if det:
            legacy=torch.load(args.reference/'legacy.pt',map_location='cpu',weights_only=False)
            comparison=compare_gradients(legacy,grad);comparison['bitwise']=identical(legacy,grad)
            write(args.out/'GRADIENT_CHECK.json',comparison)
            if not comparison['bitwise']:raise RuntimeError('live deterministic legacy gradient differs')
    # Ensure every rank completed all owned QA before committing the single step.
    dist.barrier();before_step=time.perf_counter()
    if rank==0 and optimizer is not None:
        write(args.out/'STEP_STARTED.json',dict(optimizer_steps_started=1,do_not_retry=True))
        optimizer.set_external_master_grads(grad);preclip=optimizer.grad_norm()
        if not math.isfinite(preclip):raise RuntimeError('nonfinite preclip norm, step aborted')
        optimizer.step()
    else:preclip=None
    dist.barrier();step_end=time.perf_counter()
    if args.authorize_one_isolated_step:broadcast_head(head,nccl)
    torch.cuda.synchronize();dist.barrier();end=time.perf_counter()
    digest=hashlib.sha256()
    for name,p in sorted(head.named_parameters()):
        digest.update(name.encode());digest.update(p.detach().cpu().view(torch.uint8).numpy().tobytes())
    hashes=[None]*world;dist.all_gather_object(hashes,digest.hexdigest())
    if len(set(hashes))!=1:raise RuntimeError('Head broadcast mismatch')
    # Capture each process's local RNG only; do not initialize unrelated devices.
    local_rng=dict(torch=torch.get_rng_state(),torch_cuda=[torch.cuda.get_rng_state(local)])
    import random,numpy as np
    local_rng.update(python=random.getstate(),numpy=np.random.get_state(),local_cuda_device=local,owned=owned)
    allrng=[None]*world;dist.all_gather_object(allrng,local_rng)
    reports=[None]*world;dist.all_gather_object(reports,dict(rank=rank,owned=owned,numerators=nums,rollout=rollout_end-start,backward=backward_end-rollout_end,total=end-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),forward_counts=forward_counts))
    torch.save([traces[s] for s in owned],args.out/f'rank{rank}_traces.pt')
    if rank==0:
        if optimizer is not None:
            torch.save(dict(diagnostic_only=True,parent_step=120,reused_schedule_update=119,world_size=world,slot_assignment={r:[s for s in range(4) if s%world==r] for r in range(world)},
                head={n:p.detach().cpu() for n,p in head.state_dict().items()},master_optimizer=optimizer.state_dict(),per_rank_rng=allrng,optimizer_steps_executed=1),args.out/'isolated_state.pt')
        else:torch.save(grad,args.out/'live_gradients.pt')
        write(args.out/'RESULT.json',dict(status='ISOLATED_STEP_COMPLETE' if optimizer is not None else 'LIVE_NO_UPDATE_PASS',
            optimizer_steps=1 if optimizer else 0,diagnostic_only=True,world_size=world,kernel_mode=args.kernel_mode,
            denominator_hex=denom.hex(),records=records,token_and_teacher_trace_exact=True,head_hashes=hashes,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            gradient_bitwise=comparison['bitwise'] if comparison else None,preclip_norm=preclip,
            phase_seconds=dict(rollout_max=max(r['rollout'] for r in reports),backward_and_transport=backward_end-rollout_end,
                gradient_validation_and_transfer=before_step-backward_end,optimizer=step_end-before_step,
                head_broadcast=end-step_end,total=max(r['total'] for r in reports)),rank_reports=reports,
            clean_repeated_benchmark=False,production_checkpoint=False))
    dist.destroy_process_group()

if __name__=='__main__':main()
