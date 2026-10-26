"""Isolated no-update profiling and accumulation compatibility gate.
Never constructs or steps an optimizer. All artifacts must resolve under /mnt.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import contextlib
import gc
import hashlib
import json
import time
import subprocess
import torch
from shared import common, training_resume
from opd_official_kl import adapters, runtime
from opd_official_kl.trainer import move_trace
from opd_official_kl.efficiency import (denominator_record, global_denominator,
    bind_trace, validate_trace, SlotGradientSum, capture_gradients, compare_gradients)

BASE = Path('/mnt/local2/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/train/opd')

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

class Measures:
    def __init__(self, target, head):
        self.handles=[]; self.counts={}; self.rows={}; self.events=[]; self.active=False
        self.stack={}
        modules={'target_model':target.model, 'lm_head':target.lm_head, 'draft_head':head}
        if hasattr(target.model,'visual'): modules['visual_encoder']=target.model.visual
        for name, module in modules.items():
            def before(mod,args,kwargs,name=name):
                if not self.active: return
                tensor=next((x for x in args if torch.is_tensor(x)),None)
                if tensor is None: tensor=kwargs.get('input_ids', kwargs.get('noise_embedding'))
                rows=0 if tensor is None else (tensor.numel() if tensor.ndim==2 and tensor.dtype==torch.long else tensor.numel()//tensor.shape[-1])
                label=name
                if name=='target_model': label += '_prefill' if 'pixel_values' in kwargs else '_verification_or_final_root'
                if name=='draft_head': label += '_replay' if torch.is_grad_enabled() else '_proposal'
                if name=='lm_head': label += '_student' if torch.is_grad_enabled() else '_teacher'
                self.counts[label]=self.counts.get(label,0)+1; self.rows[label]=self.rows.get(label,0)+rows
                scope=torch.profiler.record_function(label); scope.__enter__()
                self.stack.setdefault(name,[]).append((scope,label))
            def after(mod,args,kwargs,out,name=name):
                if self.active:
                    scope,label=self.stack[name].pop(); scope.__exit__(None,None,None)
            self.handles += [module.register_forward_pre_hook(before,with_kwargs=True), module.register_forward_hook(after,with_kwargs=True)]

    def reset(self): self.counts={};self.rows={}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--profile',action='store_true')
    ap.add_argument('--batch-teacher-ids',action='store_true')
    ap.add_argument('--deferred-checks',action='store_true')
    ap.add_argument('--preallocate-context',action='store_true')
    args=ap.parse_args()
    if args.repeats < 1: raise ValueError('positive repeat count required')
    training_resume.validate_output(args.out); out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'RUN_STATUS.json',dict(status='RUNNING',optimizer_steps=0))
    try: run(args,out)
    except BaseException as exc:
        write(out/'RUN_STATUS.json',dict(status='ERROR',error=repr(exc),optimizer_steps=0))
        raise


def run(args,out):
    import pandas as pd
    torch.set_num_threads(8)
    cfg=json.loads((BASE/'config.json').read_text()); checkpoint=BASE/'step120'
    schedule=[json.loads(l) for l in Path(cfg['schedule']).read_text().splitlines() if l.strip()]
    slots=sorted([r for r in schedule if r['update']==119],key=lambda r:r['slot'])
    assert [r['slot'] for r in slots]==list(range(4))
    rows=pd.read_parquet(cfg['pool']).set_index('task',verify_integrity=True)
    hashes={str(p):training_resume.file_hash(p) for p in [checkpoint/'model.safetensors', checkpoint/'training_state.pt', Path(cfg['pool']), Path(cfg['schedule'])]}
    target_files=sorted(Path(common.TARGET_PATH).glob('*.safetensors'))
    hashes.update({str(p):training_resume.file_hash(p) for p in target_files})
    write(out/'manifest_before.json',dict(hashes=hashes,config=cfg,slots=slots,args=vars(args),
        torch=torch.__version__,cuda=torch.version.cuda,
        optimizer='NOT_EXECUTED: no-update gate',
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv'],text=True)))
    start=time.perf_counter()
    target,processor=common.load_target(device='cuda:0',dtype=torch.float32)
    target.eval().requires_grad_(False)
    head=common.load_head(str(checkpoint),device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(True)
    state=torch.load(checkpoint/'training_state.pt',weights_only=False,map_location='cpu')
    if state.get('resume_schema') != 1 or state['update'] != 120: raise ValueError('checkpoint identity')
    if state['resume_identity'] != cfg['resume_identity']: raise ValueError('checkpoint/config mismatch')
    rng=state['rng']; initial=state['head']; del state;gc.collect()
    # Only one visible GPU. Original training also used local cuda:0.
    if len(rng['torch_cuda']) != 1: raise ValueError('unexpected checkpoint CUDA RNG topology')
    write(out/'load.json',dict(seconds=time.perf_counter()-start))
    measures=Measures(target,head)
    checkpoint_hash=hashes[str(checkpoint/'model.safetensors')]
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()

    def restore():
        head.load_state_dict(initial);head.zero_grad(set_to_none=True)
        training_resume.restore_rng(rng)
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()

    def collect():
        traces=[]; stage=[]
        for slot in slots:
            tick=time.perf_counter()
            with torch.profiler.record_function('image_preprocessing'):
                inputs=adapters.encode(processor,rows.loc[slot['task']].to_dict(),'image',max_pixels=cfg['max_pixels'])
            prep=time.perf_counter()-tick
            with torch.profiler.record_function('collect_rollout'):
                trace=adapters.collect(target,head,inputs,max_new_tokens=cfg['max_new_tokens'],max_seq_len=cfg['max_seq_len'],
                    batch_teacher_ids=args.batch_teacher_ids,preallocate_context=args.preallocate_context)
            identity=dict(checkpoint_sha256=checkpoint_hash,task=slot['task'],slot=slot['slot'],prompt_sha256=slot['prompt_sha256'],config_sha256=config_hash)
            trace['efficiency_identity']=identity
            with torch.profiler.record_function('trace_D2H'): cpu=move_trace(trace,'cpu')
            traces.append(cpu);stage.append(dict(slot=slot['slot'],preprocess_seconds=prep,rounds=len(cpu['rounds']),anchors=len(cpu['plan'])))
            del trace,inputs
        records=[denominator_record(t,s['slot'],s['task']) for t,s in zip(traces,slots)]
        denom=global_denominator(records,[s['task'] for s in slots])
        if denom.hex()!=sum(runtime.denominator(t) for t in traces).hex(): raise RuntimeError('denominator mismatch')
        return traces,denom,records,stage

    def backward(traces,denom,qa_reference=False):
        aggregate=SlotGradientSum([n for n,_ in head.named_parameters()]) if qa_reference else None
        head.zero_grad(set_to_none=True);numerator=0.
        for slot,cpu in enumerate(traces):
            if qa_reference:head.zero_grad(set_to_none=True)
            with torch.profiler.record_function('trace_H2D'): trace=move_trace(cpu,'cuda:0')
            iterator=runtime.score(target,head,trace,draft_fn=adapters.draft,deferred_checks=args.deferred_checks)
            while True:
                with torch.profiler.record_function('anchor_scoring'):
                    try: stat=next(iterator)
                    except StopIteration: break
                with torch.profiler.record_function('anchor_backward'): (stat.numerator/denom).backward()
                numerator+=float(stat.numerator.detach())
            del trace,iterator,stat
            if qa_reference:
                with torch.profiler.record_function('qa_gradient_D2H_and_sum'):
                    grads=capture_gradients(head);aggregate.add(slot,grads);del grads
        return numerator/denom, aggregate.finish() if qa_reference else None

    saved_traces=None;last_grads=None
    for repeat in range(-1,args.repeats):
        restore();measures.reset();measures.active=False
        torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        traces,denom,records,stage=collect();loss,_=backward(traces,denom)
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        # Correctness instrumentation is outside the clean timed window.
        for tr in traces:
            validate_trace(tr,tr['efficiency_identity'])
            for r in tr['rounds']:
                if not torch.equal(r['teacher_logp'],r['teacher_logp'].to('cuda:0').cpu()):
                    raise RuntimeError('teacher logp roundtrip')
        result=dict(repeat=repeat,warmup=repeat==-1,seconds=elapsed,loss=loss,denominator_hex=denom.hex(),
                    slots=stage,denominator_records=records,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),optimizer_steps=0)
        write(out/f'timing_{repeat}.json',result);print(json.dumps(result),flush=True)
        if repeat==args.repeats-1:
            saved_traces=traces;last_grads=capture_gradients(head)
        else:del traces
    torch.save(last_grads,out/'legacy_gradients.pt')
    # Compatibility first: same frozen traces, identical Head, same anchor order.
    restore();tick=time.perf_counter()
    ref_loss,reference=backward(saved_traces,denom,qa_reference=True)
    comparison=compare_gradients(last_grads,reference)
    comparison.update(legacy_loss=loss,reference_loss=ref_loss,seconds=time.perf_counter()-tick,
                      classification='PASS' if comparison['passed'] else 'ACCUMULATION_COMPATIBILITY_HOLD')
    comparison['loss_passed']=abs(ref_loss-loss)<=1e-4+1e-3*abs(loss)
    comparison['passed'] &= comparison['loss_passed']
    write(out/'gradient_compatibility.json',comparison)
    torch.save(reference,out/'qa_fp32_gradients.pt')
    torch.save(saved_traces,out/'fixed_traces.pt')
    del reference,last_grads,saved_traces;head.zero_grad(set_to_none=True);gc.collect()
    if args.profile:
        restore();measures.reset();measures.active=True
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
            traces,pdenom,_,_=collect();backward(traces,pdenom)
            torch.cuda.synchronize()
        measures.active=False
        prof.export_chrome_trace(str(out/'profile.json'))
        write(out/'profile_counts.json',dict(calls=measures.counts,token_rows=measures.rows,
            note='hooks count actual module invocations; counts exclude optimizer (not executed)'))
        (out/'profile_table.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=100))
        del traces,prof
    after={p:training_resume.file_hash(p) for p in hashes}
    write(out/'manifest_after.json',dict(hashes=after,unchanged=after==hashes))
    if after!=hashes: raise RuntimeError('source weights/data changed')
    write(out/'RUN_STATUS.json',dict(status='NO_UPDATE_GATE_PASS' if comparison['passed'] else 'ACCUMULATION_COMPATIBILITY_HOLD',optimizer_steps=0,
        distributed_implemented=False,full_update_verified=False))
    print('FINAL',json.dumps(comparison),flush=True)

if __name__=='__main__':main()
