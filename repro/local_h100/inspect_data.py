#!/usr/bin/env python3
"""Validate local real-frame data on CPU and freeze matched training schedules."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from video_caption.build_pilot_schedule import build_schedule
from video_caption.build_real_video_pilot import sha256_file
from qa_opd_fair.videoencode import encode_inputs, validate_real_metadata

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')


def write_schedule(path, rows, seed, source, *, selection_reason):
    order = build_schedule(rows, epochs=1, qa_per_update=4, seed=seed)
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in order)
    protocol_path = path.with_suffix(path.suffix + '.protocol.json')
    if path.exists() or protocol_path.exists():
        raise FileExistsError(f'refusing to replace schedule: {path}')
    with path.open('x') as file:
        file.write(payload)
    protocol = {'version':1, 'rows':str(source), 'rows_sha256':digest(source),
                'schedule_sha256':hashlib.sha256(payload.encode()).hexdigest(),
                'epochs':1, 'qa_per_update':4, 'seed':seed,
                'updates':len(order)//4, 'qa_exposures':len(order),
                'unique_videos':len(rows), 'dropped_or_padded':0,
                'selection_reason':selection_reason,
                'selected_source_video_ids':[row['video_id'] for row in rows],
                'shared_builder_sha256':digest(REPRO/'video_caption/build_pilot_schedule.py')}
    write_new(protocol_path, protocol)
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default=str(ROOT/'data/full'))
    parser.add_argument('--model',default='/czsun/mcy/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--control-dir',default=str(ROOT/'control'))
    parser.add_argument('--out',default=str(ROOT/'ENCODE_FULL_VALIDATION.json'))
    parser.add_argument('--wait-for-data',action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    data = Path(args.data_dir).resolve()
    control = Path(args.control_dir).resolve()
    output = Path(args.out).resolve()
    for path in (data, control, output):
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('all output/input data paths must stay under authorized local run root')
    if output.exists():
        raise FileExistsError(output)
    if args.wait_for_data:
        while not (data/'PROTOCOL.json').exists():
            state_path=data.parent/'STATUS.json'
            if state_path.exists() and json.loads(state_path.read_text()).get('phase')=='failed':
                raise RuntimeError('frame preparation failed; no final data consumed')
            time.sleep(15)
    protocol=json.loads((data/'PROTOCOL.json').read_text())
    if not protocol.get('trainable') or not protocol.get('all_timestamps_exact'):
        raise ValueError('full data needs a completed, exact-time trainable protocol')
    from transformers import AutoProcessor
    import numpy as np
    import torch
    torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    video_token=processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')
    rows={n:[json.loads(line) for line in (data/f'manifest_{n}.jsonl').read_text().splitlines()]
          for n in (64,128)}
    if {r['video_id'] for r in rows[64]}!={r['video_id'] for r in rows[128]}:
        raise ValueError('frame tiers must have exactly the same source identities')
    counts={s:sum(r['split']==s for r in rows[64]) for s in ('train','dev')}
    if counts != {'train':80,'dev':40}:
        raise ValueError(f'this fixed pilot requires 80 train and 40 dev: {counts}')
    by_tier={n:{r['video_id']:r for r in records} for n,records in rows.items()}
    records=[]
    started=time.time()
    for index,video_id in enumerate(sorted(by_tier[64])):
        short,long=by_tier[64][video_id],by_tier[128][video_id]
        if short['source_sha256']!=long['source_sha256'] or short['split']!=long['split']:
            raise ValueError('source/split changed across frame tiers')
        with np.load(short['cache_path'],allow_pickle=False) as a,np.load(long['cache_path'],allow_pickle=False) as b:
            mapping={frame_id:i for i,frame_id in enumerate(long['sample_frame_ids'])}
            if not np.array_equal(a['frames'],b['frames'][[mapping[i] for i in short['sample_frame_ids']]]):
                raise ValueError('64-frame pixels must be an exact nested subset of 128-frame pixels')
        for n,row in ((64,short),(128,long)):
            validate_real_metadata(row,require_cache=True,require_exact=True)
            if sha256_file(row['cache_path'])!=row['cache_hash']:
                raise ValueError(f'frame cache SHA mismatch: {video_id} {n}')
            inputs=encode_inputs(processor,row,'real')
            records.append({'video_id':video_id,'task':row['task'],'split':row['split'],'n_frames':n,
                            'input_ids_len':int(inputs['input_ids'].shape[-1]),
                            'visual_token_count':int((inputs['input_ids']==video_token).sum()),
                            'grid_thw':inputs['video_grid_thw'].tolist(),
                            'source_duration_s':row['video_duration'],
                            'source_sha256':row['source_sha256'],'cache_sha256':row['cache_hash']})
        if (index+1)%10==0:
            print(f'CPU_ENCODE {index+1}/120 elapsed_s={time.time()-started:.1f}',flush=True)
    summary={}
    for n in (64,128):
        tier=[r for r in records if r['n_frames']==n]
        summary[str(n)]={}
        for split in ('train','dev','all'):
            selected=[r for r in tier if split=='all' or r['split']==split]
            summary[str(n)][split]={field:{'min':min(r[field] for r in selected),
                                          'max':max(r[field] for r in selected),
                                          'mean':sum(r[field] for r in selected)/len(selected)}
                                    for field in ('input_ids_len','visual_token_count')}
            summary[str(n)][split]['longest_input']=max(selected,key=lambda r:(r['input_ids_len'],r['video_id']))
    # Same eight identities in both tiers; union includes each tier's input peak.
    peak_ids=[]
    for n in (64,128):
        peak=summary[str(n)]['train']['longest_input']['video_id']
        if peak not in peak_ids:
            peak_ids.append(peak)
    train_metrics=[r for r in records if r['split']=='train']
    ordered=sorted(train_metrics,key=lambda r:(-r['input_ids_len'],r['video_id']))
    for row in ordered:
        if row['video_id'] not in peak_ids and len(peak_ids)<8:
            peak_ids.append(row['video_id'])
    if len(peak_ids)!=8:
        raise ValueError('peak smoke needs eight distinct training sources')
    control.mkdir(parents=True,exist_ok=True)
    schedule_summary={}
    for n in (64,128):
        source=data/f'train_{n}.jsonl'
        peak_rows=[by_tier[n][video] for video in peak_ids]
        schedule_summary[f'peak{n}']=write_schedule(control/f'schedule_peak_smoke{n}.jsonl',peak_rows,20261112,source,
                                                    selection_reason='eight distinct full-train videos including both frame-tier input maxima')
        train_rows=sorted([r for r in rows[n] if r['split']=='train'],key=lambda r:r['video_id'])
        for seed in (20261112,20261113):
            key=f'{n}_seed{seed}'
            schedule_summary[key]=write_schedule(control/f'schedule_full{n}_seed{seed}.jsonl',train_rows,seed,source,
                                                  selection_reason='all 80 training videos, one exposure each')
    if schedule_summary['peak64']!=schedule_summary['peak128']:
        raise ValueError('peak frame tiers must have identical task order')
    for seed in (20261112,20261113):
        if schedule_summary[f'64_seed{seed}']!=schedule_summary[f'128_seed{seed}']:
            raise ValueError('full frame tiers must have identical task order')
    validation={'schema':'local_h100_full_cpu_encoding_v1','model':str(Path(args.model).resolve()),
                'model_config_sha256':digest(Path(args.model)/'config.json'),
                'data_protocol_sha256':digest(data/'PROTOCOL.json'),
                'inspection_source_sha256':digest(__file__),'cpu_only':True,
                'n_sources':120,'n_encoded_rows':len(records),'n_train':80,'n_dev':40,
                'cache_sha_verified':True,'nested_pixels_exact':True,'real_time_exact':True,
                'source_split_unchanged':True,'summary':summary,'rows':records,
                'peak_smoke_source_ids':peak_ids,'schedule_frame_tiers_identical':True,
                'control_dir':str(control),'wall_s':time.time()-started}
    write_new(output,validation)
    print(json.dumps({'output':str(output),'summary':summary,'peak_smoke_source_ids':peak_ids},indent=2),flush=True)


if __name__=='__main__':
    main()
