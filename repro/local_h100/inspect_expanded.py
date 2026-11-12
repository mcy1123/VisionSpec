#!/usr/bin/env python3
"""CPU processor checks and fixed schedules for the expanded video pool.

Previously checked pixel files are not rehashed. Old visual/grid statistics are
reused and checked; old input lengths are re-encoded because their source report
predates the bounded-summary prompt.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPRO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPRO))
from video_caption.build_pilot_schedule import build_schedule
from local_h100.prepare_data import atomic_json,require_local

ROOT=Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
_PROCESSOR=None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_init(model):
    global _PROCESSOR
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _PROCESSOR=AutoProcessor.from_pretrained(model,local_files_only=True)


def encode_video(pair):
    from qa_opd_fair.videoencode import encode_inputs
    processor=_PROCESSOR
    pad=processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')
    records=[]
    for row in pair:
        inputs=encode_inputs(processor,row,'real')
        ids=inputs['input_ids']
        records.append({'video_id':row['video_id'],'task':row['task'],'split':row['split'],
                        'n_frames':row['n_frames'],'source_sha256':row['source_sha256'],
                        'cache_sha256':row['cache_hash'],'source_duration_s':row['video_duration'],
                        'input_ids_len':int(ids.shape[-1]),'visual_token_count':int((ids==pad).sum()),
                        'grid_thw':inputs['video_grid_thw'].tolist(),
                        'input_ids_sha256':hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                        'prompt_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest()})
    return records


def write_schedule(path,rows,seed,data,prompt_sha,reason):
    schedule=build_schedule(rows,epochs=1,qa_per_update=4,seed=seed)
    payload=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in schedule)
    meta_path=path.with_suffix('.jsonl.protocol.json')
    if path.exists() or meta_path.exists():raise FileExistsError(path)
    with path.open('x') as file:file.write(payload)
    metadata={'version':1,'rows':str(data),'rows_sha256':sha(data),'prompt_sha256':prompt_sha,
              'schedule_sha256':sha(path),'epochs':1,'qa_per_update':4,'seed':seed,
              'updates':len(schedule)//4,'qa_exposures':len(schedule),'unique_videos':len(rows),
              'dropped_or_padded':0,'selection_reason':reason,
              'selected_source_video_ids':[r['video_id'] for r in rows]}
    with meta_path.open('x') as file:json.dump(metadata,file,indent=2);file.write('\n')
    return schedule


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default=str(ROOT/'data/expanded320'))
    parser.add_argument('--model',default='/czsun/mcy/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--control',default=str(ROOT/'control'))
    parser.add_argument('--out',default=str(ROOT/'ENCODE_EXPANDED_VALIDATION.json'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-new-tokens',type=int,default=1024)
    parser.add_argument('--max-seq-len',type=int,default=16384)
    args=parser.parse_args()
    if not 1<=args.workers<=8:raise ValueError('workers must be 1..8')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    data,control,output=map(require_local,(args.data,args.control,args.out))
    if output.exists():raise FileExistsError(output)
    rows={n:[json.loads(line) for line in (data/f'manifest_{n}.jsonl').read_text().splitlines()]
          for n in (64,128)}
    by={n:{r['video_id']:r for r in records} for n,records in rows.items()}
    if set(by[64])!=set(by[128]) or len(by[64])!=360:raise ValueError('expected 360 same sources in both tiers')
    protocol=json.loads((data/'PROTOCOL.json').read_text());prompt_sha=protocol['prompt_sha256']
    if any(hashlib.sha256(row['prompt'].encode()).hexdigest()!=prompt_sha for rr in rows.values() for row in rr):
        raise ValueError('expanded data prompt identities differ')
    old_report=ROOT/'ENCODE_FULL_VALIDATION.json'
    old=json.loads(old_report.read_text())
    old_by={(r['video_id'],r['n_frames']):r for r in old['rows']}
    control.mkdir(parents=True,exist_ok=True)
    schedules={}
    for n in (64,128):
        train=sorted([r for r in rows[n] if r['split']=='train'],key=lambda r:r['video_id'])
        if len(train)!=320 or sum(r['split']=='dev' for r in rows[n])!=40:raise ValueError('incorrect train/dev counts')
        for seed in (20261112,20261113):
            key=f'{n}_seed{seed}'
            schedules[key]=write_schedule(control/f'schedule_expanded320_{key}.jsonl',train,seed,
                                           data/f'train_{n}.jsonl',prompt_sha,'all 320 train sources, once each')
    for seed in (20261112,20261113):
        if schedules[f'64_seed{seed}']!=schedules[f'128_seed{seed}']:
            raise AssertionError('frame-tier schedules must have identical source order')
    print('FULL_SCHEDULES_READY',flush=True)
    started=time.time();state_path=ROOT/'analysis/ENCODE_EXPANDED_STATUS.json'
    state={'phase':'encoding','completed_source_videos':0,'target_source_videos':360,'workers':args.workers}
    atomic_json(state_path,state)
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(args.model,)) as pool:
        futures={pool.submit(encode_video,[by[64][video],by[128][video]]):video for video in sorted(by[64])}
        for future in as_completed(futures):
            pair=future.result()
            for record in pair:
                prior=old_by.get((record['video_id'],record['n_frames']))
                record['newly_added_source']=prior is None
                if prior is not None:
                    for field in ('source_sha256','cache_sha256','visual_token_count','grid_thw'):
                        if record[field]!=prior[field]:raise ValueError(f'old source visual identity changed: {field}')
                    record['previous_prompt_input_ids_len']=prior['input_ids_len']
                    record['prompt_change_input_token_delta']=record['input_ids_len']-prior['input_ids_len']
                record['input_plus_max_new_tokens']=record['input_ids_len']+args.max_new_tokens
                record['fits_max_seq_len']=record['input_plus_max_new_tokens']<=args.max_seq_len
                if not record['fits_max_seq_len']:raise ValueError(f'input budget exceeded: {record}')
            records.extend(pair)
            completed=len(records)//2
            if completed%10==0 or completed==360:
                state.update(completed_source_videos=completed,elapsed_s=time.time()-started)
                atomic_json(state_path,state)
                print(f'ENCODE_EXPANDED {completed}/360 elapsed_s={time.time()-started:.1f}',flush=True)
    records.sort(key=lambda r:(r['n_frames'],r['video_id']))
    if sum(r['newly_added_source'] for r in records)!=480:raise ValueError('expected exactly 240 new sources times two tiers')
    import numpy as np
    summary={}
    for n in (64,128):
        summary[str(n)]={}
        for group in ('train','dev','all','new_train'):
            selected=[r for r in records if r['n_frames']==n and
                      (group=='all' or group=='new_train' and r['newly_added_source'] or r['split']==group)]
            stats={'n':len(selected)}
            for field in ('input_ids_len','visual_token_count'):
                values=[r[field] for r in selected]
                stats[field]={'mean':float(np.mean(values)),
                              'quantiles':{f'p{q}':float(np.percentile(values,q)) for q in (0,25,50,75,90,95,99,100)}}
            stats['longest_input']=max(selected,key=lambda r:(r['input_ids_len'],r['video_id']))
            summary[str(n)][group]=stats
    peak_ids=[]
    for n in (64,128):
        video=summary[str(n)]['train']['longest_input']['video_id']
        if video not in peak_ids:peak_ids.append(video)
    for row in sorted([r for r in records if r['split']=='train'],key=lambda r:(-r['input_ids_len'],r['video_id'])):
        if row['video_id'] not in peak_ids and len(peak_ids)<8:peak_ids.append(row['video_id'])
    peaks={}
    for n in (64,128):
        peaks[n]=write_schedule(control/f'schedule_expanded320_peak_probe{n}.jsonl',
                                [by[n][video] for video in peak_ids],20261112,data/f'train_{n}.jsonl',prompt_sha,
                                'eight train-only sources including the actual maximum input of each frame tier; GPU master/evidence probe')
    if peaks[64]!=peaks[128]:raise AssertionError('peak probe schedules differ across tiers')
    result={'schema':'expanded320_actual_processor_validation_v1','cpu_only':True,'data':str(data),
            'data_protocol_sha256':sha(data/'PROTOCOL.json'),'model':str(Path(args.model).resolve()),
            'model_config_sha256':sha(Path(args.model)/'config.json'),'prompt_sha256':prompt_sha,
            'source_script_sha256':sha(__file__),'old_validation_report_sha256':sha(old_report),
            'old_visual_grid_provenance_reused_and_verified':True,
            'old_input_lengths_reencoded_because_prompt_changed':True,
            'raw_source_or_cache_rehashed':False,'holdout_used':False,'n_sources':360,'n_train':320,'n_dev':40,
            'n_old_sources':120,'n_new_sources':240,'n_actual_encoded_conditions':len(records),
            'max_new_tokens':args.max_new_tokens,'max_seq_len':args.max_seq_len,'all_inputs_fit':True,
            'largest_input_plus_max_new_tokens':max(r['input_plus_max_new_tokens'] for r in records),
            'old_prompt_input_token_deltas':sorted({r['prompt_change_input_token_delta'] for r in records if not r['newly_added_source']}),
            'summary':summary,'rows':records,'peak_probe_source_ids':peak_ids,
            'frame_tier_schedules_identical':True,'wall_s':time.time()-started}
    with output.open('x') as file:json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
    state.update(phase='completed',output=str(output),elapsed_s=time.time()-started)
    atomic_json(state_path,state)
    print(json.dumps({'out':str(output),'max_input_plus_output':result['largest_input_plus_max_new_tokens'],
                      'peak_probe_source_ids':peak_ids,'old_prompt_input_token_deltas':result['old_prompt_input_token_deltas']},indent=2),flush=True)


if __name__=='__main__':
    main()
