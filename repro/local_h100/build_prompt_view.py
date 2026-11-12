#!/usr/bin/env python3
"""Create a new bounded-summary text view without changing frames or splits."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

REPRO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPRO))
from video_caption.build_pilot_schedule import build_schedule

ROOT=Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive_json(path,value):
    with path.open('x') as file:
        json.dump(value,file,indent=2,allow_nan=False)
        file.write('\n')


def summary_prompt(source):
    # Read the exact audited constant without importing a GPU diagnostic module.
    tree=ast.parse(source.read_text())
    values=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='SUMMARY_PROMPT' for target in node.targets)]
    if len(values)!=1 or not isinstance(values[0],str) or not values[0].strip():
        raise ValueError('expected one literal audited SUMMARY_PROMPT')
    return values[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(ROOT/'data/full'))
    parser.add_argument('--out',default=str(ROOT/'data/bounded_summary_v2'))
    parser.add_argument('--control',default=str(ROOT/'control'))
    args=parser.parse_args()
    source,out,control=map(lambda p:Path(p).resolve(),(args.source,args.out,args.control))
    if any(not p.is_relative_to(ROOT.resolve()) for p in (source,out,control)):
        raise ValueError('local data view paths must remain under the authorized run root')
    if out.exists():
        raise FileExistsError(out)
    audited_source=REPRO/'local_h100/audit_native_generation.py'
    prompt=summary_prompt(audited_source)
    prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()
    parent_protocol=json.loads((source/'PROTOCOL.json').read_text())
    if not parent_protocol.get('trainable'):
        raise ValueError('parent data must be trainable')
    old={n:[json.loads(line) for line in (source/f'manifest_{n}.jsonl').read_text().splitlines()]
         for n in (64,128)}
    new={n:[dict(row,prompt=prompt,question=prompt) for row in rows] for n,rows in old.items()}
    for n in (64,128):
        if len(new[n])!=120 or sum(row['split']=='train' for row in new[n])!=80:
            raise ValueError('this view requires the same 80 train / 40 dev source pilot')
        for before,after in zip(old[n],new[n]):
            if {k:v for k,v in before.items() if k not in ('prompt','question')} != {k:v for k,v in after.items() if k not in ('prompt','question')}:
                raise ValueError('only prompt/question may change')
    import pandas as pd
    import pyarrow  # Require writer before publishing the new directory.
    out.mkdir(parents=True)
    hashes={}
    for n,rows in new.items():
        for name,subset in ((f'manifest_{n}',rows),
                            (f'train_{n}',[r for r in rows if r['split']=='train']),
                            (f'dev_{n}',[r for r in rows if r['split']=='dev'])):
            jsonl=out/f'{name}.jsonl'
            with jsonl.open('x') as file:
                for row in subset:file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            pd.DataFrame(subset).to_parquet(out/f'{name}.parquet',index=False)
            hashes[jsonl.name]=sha(jsonl)
            hashes[f'{name}.parquet']=sha(out/f'{name}.parquet')
    protocol=dict(parent_protocol,prompt_sha256=prompt_sha,
                  text_view_schema='bounded_summary_v2',parent_data=str(source),
                  parent_protocol_sha256=sha(source/'PROTOCOL.json'),
                  prompt_source=str(audited_source),prompt_source_sha256=sha(audited_source),
                  prompt_constant='SUMMARY_PROMPT',prompt=prompt,
                  changed_row_fields=['prompt','question'],frames_and_splits_unchanged=True,
                  source_view_sha256={str(n):sha(source/f'manifest_{n}.jsonl') for n in (64,128)},
                  view_files_sha256=hashes,
                  task_scope='bounded beginning/middle/end video summary; no thousand-token long-generation claim')
    protocol['limitations']=list(protocol.get('limitations',[]))+[
        'Prompt chosen from a development diagnostic; its 8/8 EOS observation is not a held-out performance claim.',
        'This view has no new pixels, sampling decisions, splits, or independent test examples.',
        'Generated output length is bounded by the task; this is not the long-output evaluation task.']
    exclusive_json(out/'PROTOCOL.json',protocol)
    schedules=[]
    control.mkdir(parents=True,exist_ok=True)
    for n in (64,128):
        training=sorted([r for r in new[n] if r['split']=='train'],key=lambda r:r['video_id'])
        by_id={r['video_id']:r for r in training}
        peak_parent=control/f'schedule_peak_smoke{n}.jsonl'
        peak_protocol=json.loads(peak_parent.with_suffix('.jsonl.protocol.json').read_text())
        peak=[by_id[video] for video in peak_protocol['selected_source_video_ids']]
        definitions=[(f'schedule_bounded_v2_{n}_seed{seed}',training,seed,
                      control/f'schedule_full{n}_seed{seed}.jsonl') for seed in (20261112,20261113)]
        definitions.append((f'schedule_bounded_v2_peak_smoke{n}',peak,20261112,peak_parent))
        for name,selected,seed,parent_schedule in definitions:
            rows=build_schedule(selected,epochs=1,qa_per_update=4,seed=seed)
            payload=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
            if payload!=parent_schedule.read_text():
                raise ValueError('new text view must preserve the original task schedule exactly')
            path=control/f'{name}.jsonl'
            ppath=path.with_suffix('.jsonl.protocol.json')
            if path.exists() or ppath.exists():
                raise FileExistsError(path)
            with path.open('x') as file:file.write(payload)
            meta={'version':1,'rows':str(out/f'train_{n}.jsonl'),
                  'rows_sha256':sha(out/f'train_{n}.jsonl'),'prompt_sha256':prompt_sha,
                  'text_view_schema':'bounded_summary_v2','schedule_sha256':sha(path),
                  'parent_schedule':str(parent_schedule),'parent_schedule_sha256':sha(parent_schedule),
                  'source_order_unchanged':True,'epochs':1,'qa_per_update':4,'seed':seed,
                  'updates':len(rows)//4,'qa_exposures':len(rows),'unique_videos':len(selected),
                  'dropped_or_padded':0,'selected_source_video_ids':[r['video_id'] for r in selected]}
            exclusive_json(ppath,meta)
            schedules.append(str(path))
    print(json.dumps({'out':str(out),'prompt_sha256':prompt_sha,'n_train':80,'n_dev':40,
                      'frames_and_splits_unchanged':True,'schedules':schedules},indent=2))


if __name__=='__main__':
    main()
