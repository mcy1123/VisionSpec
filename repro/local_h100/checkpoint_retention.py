#!/usr/bin/env python3
"""User-authorized bounded checkpoint retirement; no GPU/process control.

Retain metadata in place and archive its original bytes before removing only
model.safetensors/training_state.pt. Never touch pending evaluation inputs,
formal final heads, initialization dependencies, incomplete atomic saves or
the latest published checkpoint of an active lambda10 arm.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_h100.supervise import atomic_json

ROOT=Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
LARGE=('model.safetensors','training_state.pt')
TERMINAL={'SUCCEEDED','FAILED','BLOCKED'}
AUDIT_REL='analysis/checkpoint_cleanup_20261112'


def read(path):return json.loads(Path(path).read_text())


def fingerprint(path):
    stat=path.stat()
    return dict(device=stat.st_dev,inode=stat.st_ino,size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def reference_heads(queue,state,root):
    result=set()
    for job in queue['jobs']:
        if state['jobs'][job['id']]['status'] in TERMINAL:continue
        argv=job['argv']
        for flag in ('--head','--init-from','--resume-from'):
            if flag in argv:result.add(Path(argv[argv.index(flag)+1]).resolve())
        for item in job.get('inputs',[]):
            p=Path(item).resolve()
            if p.parent.name.startswith('step') and p.is_relative_to(root/'jobs'):
                result.add(p.parent)
    return result


def plan(root):
    root=Path(root).resolve()
    queue_path=root/'control/scheduler/queue.json';queue_bytes=queue_path.read_bytes()
    queue=json.loads(queue_bytes);state=read(root/'control/scheduler/STATE.json')
    protected=reference_heads(queue,state,root)
    protected.add(Path(read(root/'FACTORIAL80_PROTOCOL.json')['common_base']).resolve())
    protected.update(root/'jobs'/job/'run/step1' for job in (
        'long_resume_opd_C_visual_seed20261112_baseline',
        'long_sft_C_visual_resume_gate_continuous'))
    checkpoints=[]
    for p in sorted((root/'jobs').glob('*/run/step*')):
        if not p.is_dir() or p.is_symlink() or not p.name[4:].isdigit():continue
        if not (p/'SHA256.json').is_file():continue
        files=[p/name for name in LARGE if (p/name).is_file()]
        if not files:continue
        if any(f.is_symlink() for f in files):continue
        job=p.relative_to(root/'jobs').parts[0]
        if job not in state['jobs']:continue
        checkpoints.append(dict(path=p,job=job,step=int(p.name[4:]),files=files))
    newest={}
    for row in checkpoints:
        if row['job'].startswith('factorial80_tv10_'):
            newest[row['job']]=max(newest.get(row['job'],0),row['step'])
    retired=[];kept=[]
    for row in checkpoints:
        p,job,step=row['path'],row['job'],row['step']
        status=state['jobs'][job]['status']
        reason=None
        if p.resolve() in protected:reason='pending_or_running_dependency_or_explicit_recovery_point'
        elif job.startswith('factorial80_') and step==80:reason='formal_final_head'
        elif job.startswith('factorial80_tv10_') and step==newest[job]:reason='latest_lambda10_checkpoint'
        elif status!='SUCCEEDED' and not job.startswith('factorial80_tv10_'):reason='owner_not_successfully_completed'
        if reason:
            kept.append(dict(path=str(p),reason=reason));continue
        known=job.startswith(('factorial80_','mature_warm80_','bounded_v2_warm20_',
            'gpu_master_','sft_smoke_','opd_pipeline_smoke','evidence_','lambda10_',
            'long_resume_','long_sft_C_visual_resume_gate_'))
        if not known:
            kept.append(dict(path=str(p),reason='unknown_family_preserved'));continue
        manifest=read(p/'SHA256.json')
        retired.append(dict(path=str(p),job=job,step=step,
            files=[dict(name=f.name,stat=fingerprint(f),expected_creation_sha256=manifest.get(f.name)) for f in row['files']],
            manifest_sha256=hashlib.sha256((p/'SHA256.json').read_bytes()).hexdigest()))
    watched=[]
    for job in queue['jobs']:
        name=job['id'];argv=job['argv']
        if name.startswith('factorial80_tv10_') and '_eval' not in name:watched.append(name)
        elif name.startswith('factorial80_128f_') and any(f'_eval{n}_dev40' in name for n in (20,40)):
            watched.append(name)
    return dict(queue_bytes=queue_bytes,retire=retired,keep=kept,
        all_watched_terminal=all(state['jobs'][name]['status'] in TERMINAL for name in watched))


def apply_one(root,record,queue_bytes):
    root=Path(root).resolve();p=Path(record['path'])
    if not p.resolve().is_relative_to(root/'jobs') or p.is_symlink():raise ValueError('unsafe retirement path')
    if (root/'control/scheduler/queue.json').read_bytes()!=queue_bytes:
        raise RuntimeError('queue changed; rescan before retirement')
    if hashlib.sha256((p/'SHA256.json').read_bytes()).hexdigest()!=record['manifest_sha256']:
        raise RuntimeError('checkpoint manifest changed')
    for item in record['files']:
        f=p/item['name']
        if item['name'] not in LARGE or f.is_symlink() or fingerprint(f)!=item['stat']:
            raise RuntimeError('checkpoint file changed before retirement')
    archive=root/AUDIT_REL/'metadata'/record['job']/p.name
    archive.mkdir(parents=True,exist_ok=True)
    for f in p.iterdir():
        if f.is_file() and f.name not in LARGE and f.suffix=='.json':
            target=archive/f.name
            if not target.exists():shutil.copy2(f,target)
            elif f.name!='RETIRED.json' and target.read_bytes()!=f.read_bytes():
                raise RuntimeError('archived metadata mismatch')
    event=dict(record,utc=datetime.now(timezone.utc).isoformat(),
        user_authorization='保存的太多了，清理一些，以及不要这么频繁地保存',
        metadata_archive=str(archive),original_metadata_preserved=True,
        interpretation='Large checkpoint files explicitly retired; preserved SHA is historical evidence, not a promise all original files still exist.',
        files_deleted=[],bytes_deleted=0)
    # Publication before unlink preserves the intended exact file identities
    # even if interrupted; append success log only after each verified unlink.
    atomic_json(archive/'RETIRE_INTENT.json',event)
    for item in record['files']:
        f=p/item['name']
        if (root/'control/scheduler/queue.json').read_bytes()!=queue_bytes or fingerprint(f)!=item['stat']:
            raise RuntimeError('queue/file changed before unlink')
        f.unlink()
        event['files_deleted'].append(item['name']);event['bytes_deleted']+=item['stat']['size']
        atomic_json(p/'RETIRED.json',event)
        atomic_json(archive/'RETIRE_RESULT.json',event)
    audit=root/AUDIT_REL;audit.mkdir(parents=True,exist_ok=True)
    with (audit/'deleted.jsonl').open('a') as stream:
        stream.write(json.dumps(event)+'\n');stream.flush();os.fsync(stream.fileno())
    return event


def run_once(root,*,apply=False):
    value=plan(root)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),apply=apply,
        planned_checkpoints=len(value['retire']),planned_bytes=sum(f['stat']['size'] for r in value['retire'] for f in r['files']),
        kept=value['keep'],deleted_checkpoints=0,deleted_bytes=0,
        all_watched_terminal=value['all_watched_terminal'])
    if apply:
        for record in value['retire']:
            event=apply_one(root,record,value['queue_bytes'])
            result['deleted_checkpoints']+=1;result['deleted_bytes']+=event['bytes_deleted']
        audit=Path(root)/AUDIT_REL
        audit.mkdir(parents=True,exist_ok=True)
        history=[json.loads(line) for line in (audit/'deleted.jsonl').read_text().splitlines()] if (audit/'deleted.jsonl').exists() else []
        result['cumulative_deleted_bytes']=sum(x['bytes_deleted'] for x in history)
        result['cumulative_retired_checkpoints']=len(history)
        result['disk_free_bytes']=shutil.disk_usage(root).free
        atomic_json(audit/'STATUS.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--until-idle',action='store_true')
    parser.add_argument('--deadline-utc')
    parser.add_argument('--interval',type=int,default=60)
    args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline_utc) if args.deadline_utc else None
    if args.until_idle and (not args.apply or deadline is None or deadline.tzinfo is None):
        parser.error('bounded retention requires --apply and timezone-aware deadline')
    while True:
        result=run_once(ROOT,apply=args.apply)
        print(json.dumps(result),flush=True)
        if not args.until_idle or result['all_watched_terminal']:break
        if datetime.now(timezone.utc)>=deadline:break
        time.sleep(max(1,min(60,args.interval)))


if __name__=='__main__':main()
