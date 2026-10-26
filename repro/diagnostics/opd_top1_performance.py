"""Default-kernel paired four-QA timing. NO optimizer, NO gradient gate.
Timing includes rollout/offload and scoring/backward. Full-update speedup is NOT measured.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# Coordinate with an explicitly authorized full-update queue, before any CUDA import.
if __name__=='__main__' and '--out' in sys.argv:
    import time as _coord_time
    _coord_root=Path(sys.argv[sys.argv.index('--out')+1]).parent
    if (_coord_root/'FULL_UPDATE_AUTHORIZATION.json').exists():
        (_coord_root/'TOP1_WAITING_FOR_FULL_UPDATES').write_text('waiting before CUDA initialization\n')
        while not (_coord_root/'FULL_UPDATE_QUEUE_DONE.json').exists():_coord_time.sleep(2)
from _bootstrap import configure
configure()
import argparse
import gc
import json
import os
import statistics
import subprocess
import time
import torch
from shared import common,training_resume
from opd_official_kl import adapters,runtime
from opd_official_kl.trainer import move_trace
from opd_serial_candidate import compare_trace
from opd_efficiency_gate import BASE,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();training_resume.validate_output(args.out);args.out.mkdir(parents=True,exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):raise ValueError('default performance must not inherit deterministic CUBLAS config')
    torch.set_num_threads(8);torch.use_deterministic_algorithms(False)
    manifest=json.loads((args.baseline/'manifest_before.json').read_text());cfg=manifest['config'];slots=manifest['slots']
    for p,sha in manifest['hashes'].items():
        if training_resume.file_hash(p)!=sha:raise RuntimeError('input changed')
    import pandas as pd
    rows=pd.read_parquet(cfg['pool']).set_index('task',verify_integrity=True)
    target,processor=common.load_target(device='cuda:0',dtype=torch.float32);target.eval().requires_grad_(False)
    head=common.load_head(str(BASE/'step120'),device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(True)
    state=torch.load(BASE/'step120/training_state.pt',map_location='cpu',weights_only=False,mmap=True)
    refs=torch.load(args.baseline/'fixed_traces.pt',map_location='cpu',weights_only=False)
    records=[];counts={}
    def count(label):
        def hook(module,inputs):counts[label]=counts.get(label,0)+1
        return hook
    target.model.register_forward_pre_hook(count('target_model'))
    target.lm_head.register_forward_pre_hook(count('lm_head'))
    head.register_forward_pre_hook(count('draft_head'))
    order=[(-1,'baseline')]+[(pair,name) for pair in range(3) for name in (('baseline','batch_ids') if pair%2==0 else ('batch_ids','baseline'))]
    for pair,name in order:
        head.load_state_dict(state['head']);head.zero_grad(set_to_none=True);training_resume.restore_rng(state['rng'])
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        counts={};torch.cuda.synchronize();start=time.perf_counter();traces=[]
        for slot in slots:
            inputs=adapters.encode(processor,rows.loc[slot['task']].to_dict(),'image',max_pixels=cfg['max_pixels'])
            tr=adapters.collect(target,head,inputs,max_new_tokens=cfg['max_new_tokens'],max_seq_len=cfg['max_seq_len'],batch_teacher_ids=name=='batch_ids')
            tr['efficiency_identity']=refs[slot['slot']]['efficiency_identity'];traces.append(move_trace(tr,'cpu'));del tr,inputs
        torch.cuda.synchronize();rollout_end=time.perf_counter()
        denom=sum(runtime.denominator(t) for t in traces);num=0.
        for tr in traces:
            gpu=move_trace(tr,'cuda:0')
            for stat in runtime.score(target,head,gpu,draft_fn=adapters.draft):
                (stat.numerator/denom).backward();num+=float(stat.numerator.detach())
            del gpu,stat
        torch.cuda.synchronize();end=time.perf_counter()
        # Exact forward/label verification happens OUTSIDE timed window. No default gradient gate.
        for a,b in zip(refs,traces):compare_trace(a,b)
        if denom.hex()!=sum(runtime.denominator(t) for t in refs).hex():raise RuntimeError('denominator changed')
        result=dict(pair=pair,variant=name,seconds=end-start,rollout_seconds=rollout_end-start,
            scoring_backward_seconds=end-rollout_end,loss=num/denom,denominator_hex=denom.hex(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            forward_counts=dict(counts),token_trace_exact=True,teacher_logp_exact=True,optimizer_steps=0,default_gradient_gate=False,gpu_before=gpu_before)
        write(args.out/f'pair{pair}_{name}.json',result);records.append(result)
        print(json.dumps({k:result[k] for k in ['pair','variant','seconds','rollout_seconds','loss']}),flush=True)
        del traces;head.zero_grad(set_to_none=True)
    paired=[]
    for pair in range(3):
        a=next(r['seconds'] for r in records if r['pair']==pair and r['variant']=='baseline')
        b=next(r['seconds'] for r in records if r['pair']==pair and r['variant']=='batch_ids')
        paired.append(a/b)
    median=statistics.median(paired);passed=median>1.05 and all(x>1 for x in paired)
    after={p:training_resume.file_hash(p) for p in manifest['hashes']}
    if after!=manifest['hashes']:raise RuntimeError('input changed')
    write(args.out/'RESULT.json',dict(status='BENCHMARK_COMPLETE',paired_speedups=paired,median_speedup=median,
        threshold=1.05,all_pairs_faster=all(x>1 for x in paired),performance_passed=passed,
        recommendation='experimental only; full-update validation pending' if passed else 'keep default OFF',
        optimizer_steps=0,full_update_speedup_measured=False,source_sha_unchanged=True,default_gradient_gate=False))

if __name__=='__main__':main()
