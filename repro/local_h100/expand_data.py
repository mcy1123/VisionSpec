#!/usr/bin/env python3
"""Expand local Video-MME training data while freezing an unused holdout.

CPU only. Existing bounded-summary train/dev rows and frame caches are reused
verbatim. New sources are byte-deduplicated before any split is assigned.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPRO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPRO))
from video_caption import build_real_video_pilot as b
from local_h100.prepare_data import process, atomic_json, require_local
from local_h100.build_prompt_view import summary_prompt

ROOT=Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')


def write_rows(path,rows):
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def probe_and_hash(source):
    command=['ffprobe','-v','error','-select_streams','v:0','-show_entries',
             'stream=width,height,avg_frame_rate,nb_frames,duration:format=duration',
             '-of','json',source['source_path']]
    result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=120)
    value=json.loads(result.stdout);stream=value['streams'][0]
    raw=stream.get('duration')
    if raw in (None,'N/A'):raw=value.get('format',{}).get('duration')
    duration=float(raw);fps=float(Fraction(stream['avg_frame_rate']))
    if not math.isfinite(duration) or not 45<=duration<=600 or fps<=0:
        return {'video_id':source['video_id'],'eligible':False,'reason':'duration_outside_45_600_or_invalid_fps',
                'duration_s':duration}
    count=stream.get('nb_frames')
    count_method='container_nb_frames'
    if count in (None,'N/A','0'):
        fallback=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                                 '-show_entries','stream=nb_read_frames','-of','json',source['source_path']],
                                check=True,capture_output=True,text=True,timeout=600)
        count=json.loads(fallback.stdout)['streams'][0]['nb_read_frames']
        count_method='ffprobe_count_frames'
    count=int(count)
    if count<128:
        return {'video_id':source['video_id'],'eligible':False,'reason':'fewer_than_128_source_frames','source_frame_count':count}
    return dict(source,eligible=True,video_duration=duration,source_fps=fps,
                source_width=int(stream['width']),source_height=int(stream['height']),
                probed_source_frame_count=count,frame_count_probe_method=count_method,
                source_sha256=b.sha256_file(source['source_path']))


def freeze_selection(candidates,seed,holdout_per_category,additional_train):
    groups={category:sorted([row for row in candidates if row['duration_category']==category],
                            key=lambda row:row['video_id']) for category in ('short','medium')}
    for category,rows in groups.items():
        random.Random(f'{seed}:{category}:expanded320').shuffle(rows)
        if len(rows)<holdout_per_category:
            raise ValueError(f'cannot reserve {holdout_per_category} {category} holdout sources; only {len(rows)} eligible unique sources')
    holdout=[dict(row,split='future_holdout',materialized=False,model_evaluated=False)
             for category in ('short','medium') for row in groups[category][:holdout_per_category]]
    remaining={category:rows[holdout_per_category:] for category,rows in groups.items()}
    target=min(additional_train,sum(map(len,remaining.values())))
    # Balanced allocation where possible, then fill only from available strata.
    allocation={category:min(target//2,len(rows)) for category,rows in remaining.items()}
    while sum(allocation.values())<target:
        choices=[category for category in remaining if allocation[category]<len(remaining[category])]
        category=min(choices,key=lambda c:(allocation[c],c))
        allocation[category]+=1
    selected=[dict(row,split='train') for category in ('short','medium')
              for row in remaining[category][:allocation[category]]]
    reserve=[dict(row,split='train_candidate_reserve') for category in ('short','medium')
             for row in remaining[category][allocation[category]:]]
    random.Random(f'{seed}:training-order:expanded320').shuffle(selected)
    random.Random(f'{seed}:replacement-order:expanded320').shuffle(reserve)
    return holdout,selected,reserve,allocation


def materialize(source,cache,prompt):
    expected=source['source_sha256']
    try:
        records=process(dict(source,split='train'),cache,384)
    except ValueError as exc:
        return {'ok':False,'video_id':source['video_id'],'reason':str(exc),'kind':'strict_input_validation'}
    if any(row['source_sha256']!=expected for row in records):
        raise ValueError('source changed after split-freezing SHA audit')
    if any(row['source_frame_count']!=source['probed_source_frame_count'] for row in records):
        raise ValueError('decoded source frame count disagrees with pre-split metadata')
    for row in records:
        row['prompt']=row['question']=prompt
    return {'ok':True,'rows':records,'video_id':source['video_id']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-data',default=str(ROOT/'data/bounded_summary_v2'))
    parser.add_argument('--source-jsonl',default='/czsun/mcy/video_opd_pilot/Video-MME/full/videomme_900videos_2700qa.jsonl')
    parser.add_argument('--source-root',default='/czsun/mcy/video_opd_pilot/Video-MME/full/videos')
    parser.add_argument('--out',default=str(ROOT/'data/expanded320'))
    parser.add_argument('--cache',default=str(ROOT/'frames_expanded320'))
    parser.add_argument('--seed',type=int,default=20261112)
    parser.add_argument('--additional-train',type=int,default=240)
    parser.add_argument('--holdout-per-category',type=int,default=40)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.workers<=8 or args.additional_train<1 or args.holdout_per_category<1:
        raise ValueError('invalid worker/count settings')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    out,cache=require_local(args.out),require_local(args.cache)
    parent=require_local(args.parent_data)
    if out.exists() or cache.exists():
        raise FileExistsError('new expanded output/cache required; never overwrite or silently restart')
    out.mkdir(parents=True);cache.mkdir(parents=True)
    (out/'completed_sources').mkdir()
    start=time.time()
    state={'phase':'validating_parent','requested_additional_train':args.additional_train,
           'completed_additional_train':0,'started_unix':start,'workers':args.workers}
    def status(**changes):
        state.update(changes,elapsed_s=time.time()-start)
        atomic_json(out/'STATUS.json',state)
        print(json.dumps(state),flush=True)
    status()
    try:
        prompt=summary_prompt(REPRO/'local_h100/audit_native_generation.py')
        prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()
        old={n:[json.loads(line) for line in (parent/f'manifest_{n}.jsonl').read_text().splitlines()]
             for n in (64,128)}
        for n,rows in old.items():
            if len(rows)!=120 or sum(r['split']=='train' for r in rows)!=80 or sum(r['split']=='dev' for r in rows)!=40:
                raise ValueError('parent must be the existing 80 train / 40 dev split')
            if any(row['prompt']!=prompt or row['question']!=prompt for row in rows):
                raise ValueError('parent prompt differs from audited bounded summary; preserve parent rather than change it')
        original={r['video_id']:r for r in old[128]}
        if len(original)!=120 or len({r['source_sha256'] for r in original.values()})!=120:
            raise ValueError('parent contains duplicate source IDs or byte hashes')
        if {r['video_id'] for r in old[64]}!=set(original):
            raise ValueError('parent frame tiers differ')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(b.sha256_file,row['source_path']):row for row in original.values()}
            for future in as_completed(futures):
                row=futures[future]
                if future.result()!=row['source_sha256']:
                    raise ValueError(f"parent source changed: {row['video_id']}")
        qa=[json.loads(line) for line in Path(args.source_jsonl).read_text().splitlines() if line.strip()]
        sources=b.unique_sources(qa,set(),args.source_root)
        unused=[s for s in sources if s['video_id'] not in original and s['duration_category'] in ('short','medium')]
        status(phase='probing_and_hashing',candidate_source_count=len(unused),probed_sources=0)
        probes=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(probe_and_hash,row):row for row in unused}
            for future in as_completed(futures):
                probes.append(future.result())
                if len(probes)%20==0 or len(probes)==len(unused):
                    status(probed_sources=len(probes))
        write_rows(out/'source_probe.jsonl',sorted(probes,key=lambda r:r['video_id']))
        seen={row['source_sha256']:row['video_id'] for row in original.values()}
        candidates=[];aliases=[]
        for row in sorted([row for row in probes if row['eligible']],key=lambda r:r['video_id']):
            digest=row['source_sha256']
            if digest in seen:
                aliases.append({'video_id':row['video_id'],'canonical_video_id':seen[digest],'source_sha256':digest})
            else:
                seen[digest]=row['video_id'];candidates.append(row)
        write_rows(out/'excluded_source_aliases.jsonl',aliases)
        holdout,selected,reserve,allocation=freeze_selection(candidates,args.seed,args.holdout_per_category,args.additional_train)
        write_rows(out/'future_holdout.jsonl',holdout)
        write_rows(out/'initial_additional_train_selection.jsonl',selected)
        write_rows(out/'unused_train_candidate_reserve.jsonl',reserve)
        holdout_hashes={r['source_sha256'] for r in holdout}
        old_hashes={r['source_sha256'] for r in original.values()}
        if holdout_hashes & old_hashes or len(holdout_hashes)!=len(holdout):
            raise ValueError('holdout overlaps old train/dev by source bytes')
        planning={'schema':'expanded_video_training_plan_v1','parent_data':str(parent),
                  'parent_manifest_sha256':{str(n):b.sha256_file(parent/f'manifest_{n}.jsonl') for n in (64,128)},
                  'source_jsonl_sha256':b.sha256_file(args.source_jsonl),
                  'seed':args.seed,'requested_additional_train':args.additional_train,
                  'available_unique_candidates_before_holdout':len(candidates),
                  'available_by_category':{c:sum(r['duration_category']==c for r in candidates) for c in ('short','medium')},
                  'future_holdout_n':len(holdout),'future_holdout_per_category':args.holdout_per_category,
                  'future_holdout_sha256':b.sha256_file(out/'future_holdout.jsonl'),
                  'selected_additional_train':len(selected),'selected_allocation':allocation,
                  'reserve_candidates':len(reserve),'source_aliases_excluded':len(aliases),
                  'eligibility':{'categories':['short','medium'],'duration_s':[45,600],'minimum_source_frames':128},
                  'prompt_sha256':prompt_sha,'prompt':prompt,'long_side':384,'frame_counts':[64,128],
                  'future_holdout_model_evaluated':False,'future_holdout_materialized':False,
                  'source_sha_dedup_before_split':True,'existing_cache_reused':True,
                  'script_sha256':b.sha256_file(__file__)}
        atomic_json(out/'PLAN.json',planning)
        status(phase='decoding_new_training_sources',selected_additional_train=len(selected),
               selected_allocation=allocation,future_holdout_n=len(holdout),source_aliases_excluded=len(aliases))
        good=[];failed=[];pending=list(selected);reserve=list(reserve)
        target=len(selected)
        while pending:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(materialize,row,str(cache),prompt):row for row in pending}
                for future in as_completed(futures):
                    source=futures[future];result=future.result()
                    if result['ok']:
                        good.append(result['rows'])
                        atomic_json(out/'completed_sources'/f"{source['source_sha256']}.json",result)
                    else:
                        failed.append(dict(result,source_sha256=source['source_sha256'],duration_category=source['duration_category']))
                        atomic_json(out/'REJECTED_INPUTS.json',failed)
                    status(completed_additional_train=len(good),rejected_new_inputs=len(failed))
            missing=target-len(good)
            pending=[]
            if missing>0 and reserve:
                # Repair only preprocessing exclusions, without touching the frozen holdout.
                pending=[dict(row,split='train') for row in reserve[:missing]]
                reserve=reserve[missing:]
                status(phase='decoding_replacement_sources',replacement_candidates=len(pending))
        extra={n:[] for n in (64,128)}
        for records in good:
            for row in records:extra[row['n_frames']].append(row)
        for n in extra:extra[n].sort(key=lambda r:r['video_id'])
        new_ids={row['video_id'] for row in extra[128]}
        new_hashes={row['source_sha256'] for row in extra[128]}
        if len(new_hashes)!=len(new_ids) or new_hashes&(old_hashes|holdout_hashes):
            raise ValueError('final additional train sources are not byte-disjoint')
        if not extra[128]:
            raise ValueError('no new usable training source')
        import pandas as pd
        output_hashes={}
        for n in (64,128):
            records=old[n]+extra[n]
            # Existing rows are copied with no prompt, split, frame or metadata changes.
            if records[:120]!=old[n]:raise AssertionError('parent rows changed')
            for name,subset in ((f'manifest_{n}',records),
                                (f'train_{n}',[r for r in records if r['split']=='train']),
                                (f'dev_{n}',[r for r in records if r['split']=='dev']),
                                (f'additional_train_{n}',extra[n])):
                path=out/f'{name}.jsonl';write_rows(path,subset)
                pd.DataFrame(subset).to_parquet(out/f'{name}.parquet',index=False)
                output_hashes[path.name]=b.sha256_file(path)
                output_hashes[f'{name}.parquet']=b.sha256_file(out/f'{name}.parquet')
        write_rows(out/'additional_train_final_sources.jsonl',[
            {key:row[key] for key in ('video_id','source_path','source_sha256','duration_category','video_duration')}
            for row in extra[128]])
        final=dict(planning,schema_version='real_video_pilot_v1',dataset='Video-MME-derived',official_benchmark=False,
                   n_train=80+len(good),n_dev=40,n_additional_train=len(good),n_source_materialized=120+len(good),
                   final_additional_allocation={c:sum(r['duration_category']==c for r in extra[128]) for c in ('short','medium')},
                   rejected_inputs=len(failed),trainable=True,all_timestamps_exact=True,
                   source_byte_dedup_verified=True,cache_root=str(cache),parent_cache_reused=True,
                   qa_per_update=4,divisible_by_four=(80+len(good))%4==0,
                   view_files_sha256=output_hashes,completed_unix=time.time(),elapsed_s=time.time()-start,
                   limitations=['Expanded training pool; unchanged 40-video development set remains used for tuning.',
                                'Future holdout contains only frozen metadata and source SHA; no frames or model outputs were generated.',
                                'The new 80-source future holdout has not been used for model selection in this local run.',
                                'No claim of independence from unknown historical experiments on public Video-MME.',
                                'Bounded-summary task; no thousand-token long-output claim.',
                                'If the actual train count is not divisible by four, the trainer must not silently drop or pad examples.'])
        atomic_json(out/'PROTOCOL.json',final)
        status(phase='completed',n_train=final['n_train'],n_dev=40,n_future_holdout=len(holdout),
               final_additional_allocation=final['final_additional_allocation'],divisible_by_four=final['divisible_by_four'])
    except BaseException as exc:
        status(phase='failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
