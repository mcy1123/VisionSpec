"""No-update serial-switch validation against saved fixed traces and gradients.
Each variant is a correctness diagnostic, not a clean speed benchmark.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import gc
import hashlib
import json
import time
import torch
from shared import common,training_resume
from opd_official_kl import runtime,adapters
from opd_official_kl.trainer import move_trace
from opd_official_kl.efficiency import (validate_trace,denominator_record,global_denominator,
                                      capture_gradients,compare_gradients)
from opd_efficiency_gate import BASE, write


def compare_trace(a,b):
    for key in ('ids','hidden','fused'):
        if not torch.equal(a[key].cpu(),b[key].cpu()):raise RuntimeError('trace tensor mismatch: '+key)
    for key in ('prompt_len','plan','boundaries','final_cache_length','efficiency_identity'):
        if a[key]!=b[key]:raise RuntimeError('trace metadata mismatch: '+key)
    if len(a['rounds'])!=len(b['rounds']):raise RuntimeError('round count mismatch')
    for x,y in zip(a['rounds'],b['rounds']):
        if x.keys()!=y.keys():raise RuntimeError('round fields differ')
        for k in x:
            if torch.is_tensor(x[k]):
                if not torch.equal(x[k].cpu(),y[k].cpu()):raise RuntimeError('round tensor mismatch: '+k)
            elif x[k]!=y[k]:raise RuntimeError('round metadata mismatch: '+k)


RUN_OUT = None

def main():
    global RUN_OUT
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--variants',nargs='+',choices=['baseline_repeat','baseline_repeat_b','baseline_repeat_c','batch_ids','deferred_checks','preallocate','resident','all'],default=['baseline_repeat','baseline_repeat_b','baseline_repeat_c','batch_ids','deferred_checks','preallocate','resident','all'])
    ap.add_argument('--resident-mib',type=int,default=2048)
    args=ap.parse_args()
    if len(args.variants)!=len(set(args.variants)):raise ValueError('duplicate variants would overwrite evidence')
    training_resume.validate_output(args.out);out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
    RUN_OUT = out
    write(out/'RUN_STATUS.json',dict(status='RUNNING',optimizer_steps=0))
    torch.set_num_threads(8);base=Path(args.baseline)
    cfg=json.loads((BASE/'config.json').read_text());manifest=json.loads((base/'manifest_before.json').read_text())
    for p,sha in manifest['hashes'].items():
        if training_resume.file_hash(p)!=sha:raise RuntimeError('input hash changed: '+p)
    import pandas as pd
    rows=pd.read_parquet(cfg['pool']).set_index('task',verify_integrity=True);slots=manifest['slots']
    for slot in slots:
        actual=hashlib.sha256(rows.loc[slot['task'],'prompt'].encode()).hexdigest()
        if actual!=slot['prompt_sha256']:raise RuntimeError('pool/schedule prompt mismatch')
    checkpoint=BASE/'step120'
    target,processor=common.load_target(device='cuda:0',dtype=torch.float32);target.eval().requires_grad_(False)
    head=common.load_head(str(checkpoint),device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(True)
    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
    initial,rng=state['head'],state['rng'];del state
    write(out/'environment.json',dict(cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,target_dtype=str(next(target.parameters()).dtype),head_dtype=str(next(head.parameters()).dtype),allow_tf32=torch.backends.cuda.matmul.allow_tf32,deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    reference=torch.load(base/'fixed_traces.pt',map_location='cpu',weights_only=False)
    refgrads=torch.load(base/'legacy_gradients.pt',map_location='cpu',weights_only=False)
    results=[]
    for variant in args.variants:
        frozen = variant.startswith('baseline_repeat') or variant=='deferred_checks'
        head.load_state_dict(initial);head.zero_grad(set_to_none=True);training_resume.restore_rng(rng)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        traces=[];resident_bytes=0;offloaded=[];started=time.perf_counter()
        flags=dict(batch_teacher_ids=variant in ('batch_ids','all'),preallocate_context=variant in ('preallocate','all'))
        for slot,ref in zip(slots,reference):
            if frozen:
                tr=move_trace(ref,'cpu')
                validate_trace(tr,ref['efficiency_identity'])
                traces.append(tr);offloaded.append(slot['slot'])
                continue
            inputs=adapters.encode(processor,rows.loc[slot['task']].to_dict(),'image',max_pixels=cfg['max_pixels'])
            tr=adapters.collect(target,head,inputs,max_new_tokens=cfg['max_new_tokens'],max_seq_len=cfg['max_seq_len'],**flags)
            tr['efficiency_identity']=dict(ref['efficiency_identity'])
            validate_trace(tr,ref['efficiency_identity'])
            try:
                compare_trace(ref,tr)
            except RuntimeError:
                torch.save(move_trace(tr,'cpu'),out/f'{variant}_slot{slot["slot"]}_failed_trace.pt')
                raise
            size=sum(tr[k].untyped_storage().nbytes() for k in ('ids','fused','hidden'))+sum(r['teacher_logp'].numel()*4 for r in tr['rounds'])
            keep=(variant in ('resident','all') and resident_bytes+size<=args.resident_mib*1024**2 and torch.cuda.mem_get_info()[0]>=4*1024**3)
            if keep:resident_bytes+=size
            else:tr=move_trace(tr,'cpu');offloaded.append(slot['slot'])
            traces.append(tr);del inputs
        records=[denominator_record(t,s['slot'],s['task']) for t,s in zip(traces,slots)]
        denom=global_denominator(records,[s['task'] for s in slots])
        if denom.hex()!=sum(runtime.denominator(t) for t in reference).hex():raise RuntimeError('denominator changed')
        numerator=0.
        for tr in traces:
            gpu=move_trace(tr,'cuda:0')
            for stat in runtime.score(target,head,gpu,draft_fn=adapters.draft,deferred_checks=variant in ('deferred_checks','all')):
                (stat.numerator/denom).backward();numerator+=float(stat.numerator.detach())
            del gpu,stat
        grads=capture_gradients(head);report=compare_gradients(refgrads,grads)
        timing_files=[p for p in base.glob('timing_*.json') if int(p.stem.split('_')[1])>=0]
        latest=max(timing_files,key=lambda p:int(p.stem.split('_')[1]))
        baseline_loss=json.loads(latest.read_text())['loss']
        report['gradient_bitwise_exact']=all((a is None and grads[n] is None) or (a is not None and grads[n] is not None and torch.equal(a,grads[n])) for n,a in refgrads.items())
        report['loss']=numerator/denom
        report['loss_passed']=abs(report['loss']-baseline_loss)<=1e-4+1e-3*abs(baseline_loss)
        report['passed'] &= report['loss_passed']
        report.update(variant=variant,live_trace_checked=not frozen,trace_source='frozen' if frozen else 'live',trace_bitwise_exact=True,teacher_logp_bitwise_exact=True,denominator_hex=denom.hex(),
            optimizer_steps=0,diagnostic_seconds=time.perf_counter()-started,clean_timing=False,
            peak_allocated=torch.cuda.max_memory_allocated(),resident_bytes=resident_bytes,offloaded_slots=offloaded)
        write(out/(variant+'.json'),report);results.append(dict(variant=variant,passed=report['passed']))
        print(json.dumps(results[-1]),flush=True)
        del traces,tr,grads;head.zero_grad(set_to_none=True);gc.collect()
        if not report['passed']:break
    after={p:training_resume.file_hash(p) for p in manifest['hashes']}
    write(out/'manifest_after.json',dict(hashes=after,unchanged=after==manifest['hashes']))
    if after!=manifest['hashes']:raise RuntimeError('input hash changed')
    write(out/'RUN_STATUS.json',dict(status='PASS' if all(r['passed'] for r in results) else 'HOLD',results=results,optimizer_steps=0))

if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if RUN_OUT is not None:
            write(RUN_OUT/'RUN_STATUS.json',dict(status='ERROR',error=repr(exc),optimizer_steps=0))
        raise
