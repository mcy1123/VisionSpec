"""Bounded no-update profiler: all four QA, separate rollout/backward captures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import gc
import json
import torch
from shared import common,training_resume
from opd_official_kl import adapters,runtime
from opd_official_kl.trainer import move_trace
from opd_official_kl.efficiency import global_denominator,denominator_record
from opd_efficiency_gate import BASE,Measures,write


RUN_OUT = None

def main():
    global RUN_OUT
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--out',required=True);ap.add_argument('--resume-slot0');args=ap.parse_args()
    training_resume.validate_output(args.out);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    RUN_OUT = out
    write(out/'RUN_STATUS.json',dict(status='RUNNING',optimizer_steps=0))
    if args.resume_slot0 and not Path(args.resume_slot0).is_file():raise ValueError('missing resumed raw trace')
    torch.set_num_threads(8)
    base=Path(args.baseline);manifest=json.loads((base/'manifest_before.json').read_text());cfg=manifest['config'];slots=manifest['slots']
    for p,h in manifest['hashes'].items():
        if training_resume.file_hash(p)!=h:raise RuntimeError('input hash mismatch')
    import pandas as pd
    rows=pd.read_parquet(cfg['pool']).set_index('task',verify_integrity=True)
    target,processor=common.load_target(device='cuda:0',dtype=torch.float32);target.eval().requires_grad_(False)
    head=common.load_head(str(BASE/'step120'),device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(True)
    def collect(slot):
        with torch.profiler.record_function('image_preprocessing'):
            inputs=adapters.encode(processor,rows.loc[slot['task']].to_dict(),'image',max_pixels=cfg['max_pixels'])
        with torch.profiler.record_function('collect_rollout'):
            tr=adapters.collect(target,head,inputs,max_new_tokens=cfg['max_new_tokens'],max_seq_len=cfg['max_seq_len'])
        with torch.profiler.record_function('trace_D2H'):tr=move_trace(tr,'cpu')
        return tr
    # Warm up without retaining a trace or changing weights.
    warm=collect(slots[0]);del warm;gc.collect();torch.cuda.empty_cache()
    measures=Measures(target,head);measures.active=True
    saved=torch.load(base/'fixed_traces.pt',map_location='cpu',weights_only=False)
    traces=[];head.zero_grad(set_to_none=True);summary=[]
    for phase in ('rollout','scoring_backward'):
        if phase=='scoring_backward':
            records=[denominator_record(t,s['slot'],s['task']) for t,s in zip(traces,slots)]
            denom=global_denominator(records,[s['task'] for s in slots]);numerator=0.
        for slot in slots:
            if phase=='rollout' and slot['slot']==0 and args.resume_slot0:
                traces.append(saved[0])
                summary.append(dict(phase=phase,slot=0,raw_trace=str(Path(args.resume_slot0).resolve())))
                continue
            chunk=out/f"{phase}_slot{slot['slot']}";chunk.mkdir()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as prof:
                if phase=='rollout':traces.append(collect(slot))
                else:
                    with torch.profiler.record_function('trace_H2D'):tr=move_trace(traces[slot['slot']],'cuda:0')
                    iterator=runtime.score(target,head,tr,draft_fn=adapters.draft)
                    while True:
                        with torch.profiler.record_function('anchor_scoring'):
                            try:stat=next(iterator)
                            except StopIteration:break
                        with torch.profiler.record_function('anchor_backward'):(stat.numerator/denom).backward()
                        numerator+=float(stat.numerator.detach())
                    del tr,stat,iterator
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(chunk/'profile.json'))
            summary.append(dict(phase=phase,slot=slot['slot'],raw_trace=str(chunk/'profile.json')))
            print('PROFILE_CHUNK_COMPLETE',phase,slot['slot'],flush=True)
            del prof;gc.collect()
    write(out/'profile_counts.json',dict(note='hook counts exclude resumed rollout slot0; use raw trace for its counts' if args.resume_slot0 else 'all four QA captured',calls=measures.counts,token_rows=measures.rows,optimizer_steps=0,loss=numerator/denom,denominator_hex=denom.hex()))
    write(out/'ranges.json',summary)
    after={p:training_resume.file_hash(p) for p in manifest['hashes']}
    write(out/'manifest_after.json',dict(hashes=after,unchanged=after==manifest['hashes']))
    if after!=manifest['hashes']:raise RuntimeError('input hashes changed')
    write(out/'RUN_STATUS.json',dict(status='COMPLETE',optimizer_steps=0,profile_memory=False,record_shapes=False,clean_timing=False))

if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if RUN_OUT is not None:
            write(RUN_OUT/'RUN_STATUS.json',dict(status='ERROR',error=repr(exc),optimizer_steps=0))
        raise
