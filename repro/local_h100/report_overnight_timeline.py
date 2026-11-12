#!/usr/bin/env python3
"""Bounded nightly result collection and authorized retirement of gate binaries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_h100.checkpoint_retention import ROOT, apply_one, fingerprint
from local_h100.supervise import atomic_json
from local_h100.summarize_warm_dev import pool_identity, load_evaluation, paired_bootstrap
FOLDER=ROOT/'overnight_timeline_20261113'


def summary():
    protocol=json.loads((FOLDER/'PROTOCOL.json').read_text())
    state=json.loads((ROOT/'control/scheduler/STATE.json').read_text())['jobs']
    out=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status='RUNNING',tasks={},evaluations={},comparisons={},errors=[],primary_metric='macro mean accepted+1 (tau), paired by source video',holdout_accessed=False)
    for name in protocol['jobs']:out['tasks'][name]=state.get(name,{}).get('status','NOT_ADOPTED')
    loaded={}
    for frames in (128,64):
        pool=pool_identity(ROOT/f'data/full/dev_{frames}.parquet',frames)
        names=([protocol['warm_evaluation']] if frames==128 else [])+[a['primary' if frames==128 else 'transfer64'] for a in protocol['arms']]
        for name in names:
            if out['tasks'][name]!='SUCCEEDED':continue
            try:
                value=load_evaluation(ROOT/'jobs'/name/'result.json',pool,frames)
                rows=value['rows']
                if set(rows)!=set(pool['sources']):raise ValueError('requires all40 sources')
                taus=[r['tau'] for r in rows.values()]
                accepted=[v for r in rows.values() for v in r['accepted']]
                from collections import Counter
                out['evaluations'][name]=dict(n=len(rows),tau=sum(taus)/len(taus),pure_draft_accepted_mean=sum(taus)/len(taus)-1,accepted_round_histogram=dict(Counter(accepted)),lossless=sum(bool(r['lossless']) for r in rows.values()),natural_eos=sum(bool(r.get('teacher_natural_eos',r.get('hit_eos',False))) for r in rows.values()))
                loaded[name]=rows
            except Exception as exc:out['errors'].append(dict(evaluation=name,error=str(exc)))
    def compare(label,left,right):
        if left not in loaded or right not in loaded:return
        a,b=loaded[left],loaded[right]
        if set(a)!=set(b):raise ValueError('different sources')
        deltas=[]
        for key in a:
            if a[key]['input_identity']!=b[key]['input_identity'] or a[key]['ar_ids']!=b[key]['ar_ids']:raise ValueError('unmatched input/independent AR')
            deltas.append(a[key]['tau']-b[key]['tau'])
        out['comparisons'][label]=paired_bootstrap(deltas,resamples=20000)
    for frames in (128,64):
        field='primary' if frames==128 else 'transfer64'
        lookup={(a['arm'],a['lambda_weight']):a[field] for a in protocol['arms']}
        for arm in ('A_plain','B_text','C_visual','D_visual_bins1'):
            compare(f'{frames}f/{arm}/lambda10-minus0',lookup[arm,10],lookup[arm,0])
        for lam in (0,10):
            for arm in ('C_visual','D_visual_bins1'):
                compare(f'{frames}f/{arm}-minus-B/lambda{lam}',lookup[arm,lam],lookup['B_text',lam])
    # Retire only finished gate inputs after their own AR/SPEC check succeeded.
    gates=[(protocol['warm_gate'],protocol['warm_gate_eval'])]+[(a['gate'],a['gate_eval']) for a in protocol['arms']]
    for name,ev in gates:
        if out['tasks'][name]!='SUCCEEDED' or out['tasks'][ev]!='SUCCEEDED':continue
        p=ROOT/'jobs'/name/'run/step2'
        files=[p/n for n in ('model.safetensors','training_state.pt') if (p/n).exists()]
        if not files:continue
        queue_bytes=(ROOT/'control/scheduler/queue.json').read_bytes()
        manifest=json.loads((p/'SHA256.json').read_text())
        record=dict(path=str(p),job=name,step=2,files=[dict(name=f.name,stat=fingerprint(f),expected_creation_sha256=manifest.get(f.name)) for f in files],manifest_sha256=hashlib.sha256((p/'SHA256.json').read_bytes()).hexdigest())
        apply_one(ROOT,record,queue_bytes)
    terminal={'SUCCEEDED','FAILED','BLOCKED','RECOVERY_REQUIRED'}
    if all(s in terminal for s in out['tasks'].values()):
        out['status']='COMPLETE' if all(s=='SUCCEEDED' for s in out['tasks'].values()) and not out['errors'] else 'FINISHED_WITH_FAILURES'
        atomic_json(FOLDER/'DONE.json',out)
    atomic_json(FOLDER/'STATUS.json',out)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--max-hours',type=float,default=14);args=p.parse_args()
    end=time.monotonic()+args.max_hours*3600
    while True:
        try:
            out=summary();print(json.dumps(dict(status=out['status'],evaluations=len(out['evaluations']),errors=out['errors'])),flush=True)
            if out['status']!='RUNNING' or not args.watch:break
        except Exception as exc:
            atomic_json(FOLDER/'REPORTER_ERROR.json',dict(utc=datetime.now(timezone.utc).isoformat(),error=repr(exc)))
            print(repr(exc),flush=True)
            if not args.watch:raise
        if time.monotonic()>=end:break
        time.sleep(60)

if __name__=='__main__':main()
