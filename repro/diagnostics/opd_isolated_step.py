"""Exactly ONE explicitly authorized optimizer step on a replayed step120 schedule.
The output is diagnostic-only, never a production resume checkpoint.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import json
import os
import math
import time
import torch
from shared import common,training_resume
from shared.master_optim import MasterWeightAdamW
from opd_official_kl import adapters,runtime
from opd_official_kl.trainer import move_trace
from opd_serial_candidate import compare_trace
from opd_efficiency_gate import BASE,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--authorized-single-step',action='store_true',required=True)
    ap.add_argument('--kernel-mode',choices=['default','deterministic'],default='default')
    args=ap.parse_args();training_resume.validate_output(args.out);args.out.mkdir(parents=True,exist_ok=False)
    write(args.out/'RUN_STATUS.json',dict(status='RUNNING',optimizer_steps=0))
    deterministic=args.kernel_mode=='deterministic'
    if deterministic!=(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'):raise ValueError('kernel environment mismatch')
    torch.set_num_threads(8);torch.use_deterministic_algorithms(deterministic)
    manifest=json.loads((args.baseline/'manifest_before.json').read_text());cfg=manifest['config'];slots=manifest['slots']
    for p,sha in manifest['hashes'].items():
        if training_resume.file_hash(p)!=sha:raise RuntimeError('input SHA mismatch: '+p)
    import pandas as pd
    rows=pd.read_parquet(cfg['pool']).set_index('task',verify_integrity=True)
    target,processor=common.load_target(device='cuda:0',dtype=torch.float32);target.eval().requires_grad_(False)
    head=common.load_head(str(BASE/'step120'),device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(True)
    state=torch.load(BASE/'step120/training_state.pt',map_location='cpu',weights_only=False)
    optimizer=MasterWeightAdamW(head,lr=cfg['lr'],betas=(.9,.95),weight_decay=0.,where='cpu',clip=cfg['clip'])
    training_resume.restore(state,head,optimizer,state['resume_identity'],requested_step=120)
    parent=dict(step=120,schedule_update=119,input_hashes=manifest['hashes'],resume_identity=state['resume_identity'])
    del state
    refs=torch.load(args.baseline/'fixed_traces.pt',map_location='cpu',weights_only=False)
    forward_counts={'target_model':0,'lm_head':0,'draft_head':0}
    def counter(key):
        def hook(module,inputs):forward_counts[key]+=1
        return hook
    target.model.register_forward_pre_hook(counter('target_model'))
    target.lm_head.register_forward_pre_hook(counter('lm_head'))
    head.register_forward_pre_hook(counter('draft_head'))
    # One no-update warmup, then restore complete checkpoint state before measured run.
    def collect(slot):
        inputs=adapters.encode(processor,rows.loc[slot['task']].to_dict(),'image',max_pixels=cfg['max_pixels'])
        return adapters.collect(target,head,inputs,max_new_tokens=cfg['max_new_tokens'],max_seq_len=cfg['max_seq_len'])
    warm=collect(slots[0]);del warm
    state=torch.load(BASE/'step120/training_state.pt',map_location='cpu',weights_only=False)
    training_resume.restore(state,head,optimizer,state['resume_identity'],requested_step=120);del state
    optimizer.zero_grad(set_to_none=True)
    forward_counts={k:0 for k in forward_counts};torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();start=time.perf_counter();traces=[];rollout_times=[]
    for slot,ref in zip(slots,refs):
        t=time.perf_counter();trace=collect(slot);trace['efficiency_identity']=ref['efficiency_identity']
        traces.append(move_trace(trace,'cpu'));torch.cuda.synchronize()
        rollout_times.append(time.perf_counter()-t);del trace
    rollout_end=time.perf_counter();denom=sum(runtime.denominator(t) for t in traces);num=0.
    for trace in traces:
        gpu=move_trace(trace,'cuda:0')
        for stat in runtime.score(target,head,gpu,draft_fn=adapters.draft):
            (stat.numerator/denom).backward();num+=float(stat.numerator.detach())
        del gpu,stat
    torch.cuda.synchronize();backward_end=time.perf_counter()
    optimizer.sync_grads();torch.cuda.synchronize();sync_end=time.perf_counter()
    norm=optimizer.grad_norm();norm_end=time.perf_counter()
    if not math.isfinite(norm) or not math.isfinite(num/denom):raise RuntimeError('nonfinite update')
    for trace,ref in zip(traces,refs):compare_trace(trace,ref)
    # Time the production step itself; wrappers do not alter its math or control flow.
    original_clip=torch.nn.utils.clip_grad_norm_;original_adam=optimizer._opt.step;parts={}
    def clip(*a,**kw):
        t=time.perf_counter();v=original_clip(*a,**kw);parts['clip']=time.perf_counter()-t;return v
    def adam(*a,**kw):
        t=time.perf_counter();v=original_adam(*a,**kw);parts['adam']=time.perf_counter()-t;return v
    torch.nn.utils.clip_grad_norm_=clip;optimizer._opt.step=adam
    # Persist the intent before the irreversible in-process update. Never retry automatically.
    write(args.out/'RUN_STATUS.json',dict(status='STEP_ENTERED_DO_NOT_RETRY',optimizer_steps_started=1))
    t=time.perf_counter()
    try:optimizer.step()
    finally:torch.nn.utils.clip_grad_norm_=original_clip;optimizer._opt.step=original_adam
    torch.cuda.synchronize();finish=time.perf_counter();parts['step_total']=finish-t
    parts['write_back_and_legacy_update_norms']=parts['step_total']-parts['clip']-parts['adam']
    parts.update(rollout_and_trace_offload=rollout_end-start,scoring_backward=backward_end-rollout_end,
        master_sync=sync_end-backward_end,preclip_norm=norm_end-sync_end,total=finish-start,
        head_broadcast=None,head_broadcast_reason='single-rank update; no collective broadcast; separate 2/4-rank probes are reported',
        per_slot_rollout=rollout_times)
    write(args.out/'TIMING.json',dict(seconds=parts,loss=num/denom,preclip_gradient_norm=norm,
        denominator_hex=denom.hex(),kernel_mode=args.kernel_mode,forward_counts=forward_counts,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),optimizer_steps=1,clean_repeated_benchmark=False,
        caveat='one isolated full update; warmup one QA; shared host/GPU conditions recorded separately'))
    for trace,ref in zip(traces,refs):compare_trace(trace,ref)
    torch.save(dict(diagnostic_only=True,parent=parent,head={n:p.detach().cpu() for n,p in head.state_dict().items()},
        master_optimizer=optimizer.state_dict(),rng=training_resume.capture_rng(),optimizer_steps_executed=1),args.out/'isolated_state.pt')
    after={p:training_resume.file_hash(p) for p in manifest['hashes']}
    write(args.out/'inputs_after.json',dict(hashes=after,unchanged=after==manifest['hashes']))
    if after!=manifest['hashes']:raise RuntimeError('input SHA changed')
    write(args.out/'RUN_STATUS.json',dict(status='COMPLETE',optimizer_steps=1,trace_bitwise_exact=True,production_checkpoint=False))

if __name__=='__main__':main()
