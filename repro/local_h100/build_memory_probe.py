#!/usr/bin/env python3
"""Freeze a two-update engineering probe with the longest input in both steps.

This deliberately repeats one training source. It is not an unbiased training
comparison schedule; the full 320-source schedules remain unchanged.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path=ROOT/'ENCODE_EXPANDED_VALIDATION.json'
    report=json.loads(report_path.read_text())
    longest={n:report['summary'][str(n)]['train']['longest_input']['video_id'] for n in (64,128)}
    if longest[64]!=longest[128]:
        raise ValueError('this paired probe requires a common measured maximum source in both tiers')
    peak=longest[64]
    others=[]
    for row in sorted([r for r in report['rows'] if r['split']=='train'],
                      key=lambda r:(-r['input_ids_len'],r['video_id'])):
        if row['video_id']!=peak and row['video_id'] not in others and len(others)<6:
            others.append(row['video_id'])
    if len(others)!=6:raise ValueError('need six additional distinct training sources')
    order=[peak]+others[:3]+[peak]+others[3:]
    schedule=[{'update':i//4,'slot':i%4,'task':video,'epoch':1} for i,video in enumerate(order)]
    payload=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in schedule)
    generated=[]
    for n in (64,128):
        source=ROOT/f'data/expanded320/train_{n}.jsonl'
        train={r['video_id']:r for r in map(json.loads,source.read_text().splitlines())}
        if not set(order)<=set(train):raise ValueError('engineering probe must be train-only')
        if any(train[video]['task']!=video for video in order):raise ValueError('task/video identity mismatch')
        path=ROOT/f'control/schedule_expanded_peak_memory_{n}.jsonl'
        protocol_path=path.with_suffix('.jsonl.protocol.json')
        if path.exists() or protocol_path.exists():raise FileExistsError(path)
        with path.open('x') as file:file.write(payload)
        meta={'version':1,'rows':str(source),'rows_sha256':sha(source),
              'prompt_sha256':report['prompt_sha256'],'schedule_sha256':sha(path),
              'encoding_report_sha256':sha(report_path),'epochs':1,'qa_per_update':4,
              'seed':20261112,'updates':2,'qa_exposures':8,'unique_videos':7,
              'engineering_memory_probe_only':True,'diagnostic_only':True,
              'not_a_training_comparison_schedule':True,
              'repeated_source_exposures':{peak:2},'source_order':order,
              'longest_source':peak,'longest_input_tokens':report['summary'][str(n)]['train']['longest_input']['input_ids_len'],
              'longest_positions':[{'update':0,'slot':0},{'update':1,'slot':0}],
              'construction':'deterministic measured maximum at both update roots, six other distinct training sources',
              'reason':'The second update must exercise the maximum input after first-step Adam moment allocation, with master weights and gradients.',
              'formal_320_source_schedules_unchanged':True,'holdout_used':False}
        with protocol_path.open('x') as file:json.dump(meta,file,indent=2);file.write('\n')
        generated.append(str(path))
    result={'schema':'expanded_two_update_memory_probe_v1','schedules':generated,
            'source_order':order,'longest_source':peak,'unique_videos':7,'qa_exposures':8,
            'longest_at_update0_slot0_and_update1_slot0':True,
            'frame_tier_schedule_bytes_identical':True,'engineering_only':True}
    out=ROOT/'analysis/expanded_memory_probe_validation.json'
    with out.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
