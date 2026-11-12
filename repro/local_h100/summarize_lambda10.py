#!/usr/bin/env python3
"""Strict final-step lambda10 versus lambda0 dev40 comparisons."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import summarize_factorial80 as factorial
from local_h100 import summarize_warm_dev as warm

SCHEMA = 'local_matched_lambda10_dev40_v1'


def validate_training_pair(baseline, candidate):
    """Changing the auxiliary weight is the only permitted protocol change."""
    expected, actual = copy.deepcopy(baseline), copy.deepcopy(candidate)
    if expected['config']['e2e_tv_weight'] != 0:
        raise ValueError('baseline is not lambda0')
    expected['config']['e2e_tv_weight'] = 10.
    expected.pop('protocol_sha256',None)
    actual.pop('protocol_sha256',None)
    if expected != actual:
        changed = [key for key in sorted(set(expected)|set(actual))
                   if expected.get(key) != actual.get(key)]
        raise ValueError('training differs beyond lambda: ' + ','.join(changed))


def load_candidate(root, record, jobs, pool):
    name = record['evaluation80']
    job = jobs[name]
    path = Path(job['result']['path'])
    state = warm.completion(path)
    if state == 'PENDING':
        raise factorial.PendingResult('evaluation not complete')
    if state != 'READY':
        raise ValueError(state)
    value = warm.load_evaluation(path,pool,128)
    factorial.validate_complete(value,pool)
    config, protocol = value['config'],value['protocol']
    expected_head = root/'jobs'/record['training']/'run/step80'
    expected_config = dict(ar=True,warmup=0,dtype='float32',max_new=1024,
        budget=63,branch=8,time_mode='real',require_exact_time=True,
        eos_stop='immediate',zero_visual=False,evidence_intervention='normal',
        evidence_mode=factorial.MODES[record['arm']],n=0,shards=1,shard=0)
    if any(config.get(k)!=v for k,v in expected_config.items()):
        raise ValueError('lambda10 deployment protocol differs from fixed dev40')
    args=factorial.options(job['argv'])
    if (path.resolve().parent != Path(job['job_dir']).resolve()
            or Path(config['head']).resolve()!=expected_head.resolve()
            or Path(args['--head']).resolve()!=expected_head.resolve()
            or config['target']!=args['--target']):
        raise ValueError('lambda10 result head/target/path differs from its queue definition')
    artifacts=protocol['artifacts']
    identity=factorial.read(expected_head/'CHECKPOINT_IDENTITY.json')
    if (identity!=artifacts['checkpoint_identity']
            or warm.file_sha(expected_head/'CHECKPOINT_IDENTITY.json')!=artifacts['checkpoint_identity_sha256']
            or identity['optimizer_step']!=80 or identity['qa_exposures']!=320):
        raise ValueError('lambda10 checkpoint identity/budget mismatch')
    saved=factorial.read(expected_head/'SHA256.json')
    if any(saved.get(k)!=v for k,v in artifacts['head_weights_sha256'].items()):
        raise ValueError('lambda10 evaluated weights differ from saved manifest')
    if artifacts['head_config_sha256']!=warm.file_sha(expected_head/'config.json'):
        raise ValueError('lambda10 head config mismatch')
    baseline=factorial.read(root/'jobs'/record['baseline_training']/'run/PROTOCOL.json')
    training=factorial.read(expected_head.parent/'PROTOCOL.json')
    validate_training_pair(baseline,training)
    if identity['protocol_sha256']!=training['protocol_sha256']:
        raise ValueError('lambda10 checkpoint is not bound to training protocol')
    for row in value['rows'].values():
        if row.get('first_mismatch') is not None:
            raise ValueError('mismatch despite claimed lossless result')
        if record['arm']!='A_plain':
            evidence=row['evidence']
            if (evidence['metadata']['encoded_sha256']!=row['input_identity']
                    or evidence['diagnostics']['mode']!=factorial.MODES[record['arm']]):
                raise ValueError('actual evidence identity/mode mismatch')
    return value


def build_summary(root,*,resamples=20000):
    root=Path(root).resolve()
    plan=factorial.read(root/'LAMBDA10_PROTOCOL.json')
    pool=warm.pool_identity(root/'data/bounded_summary_v2/dev_128.parquet',128)
    jobs={j['id']:j for j in factorial.read(root/'control/scheduler/queue.json')['jobs']}
    available,evaluations,pending,invalid={},{},[],[]
    for record in plan['formal_arms']:
        arm,seed=record['arm'],record['seed']
        for objective in ('sft','opd'):
            name=factorial.eval_id(arm,objective,seed,80)
            try:
                available[name]=factorial.load_arm(jobs[name],pool,arm,objective,seed,80,root)
            except factorial.PendingResult as exc:
                pending.append(dict(evaluation=name,reason=str(exc)))
            except (ValueError,KeyError,OSError,TypeError) as exc:
                invalid.append(dict(evaluation=name,error=str(exc)))
        name=record['evaluation80']
        try:
            value=load_candidate(root,record,jobs,pool)
            available[name]=value
            evaluations[name]=dict(arm=arm,seed=seed,summary=warm.aggregate(value['rows']),
                path=value['path'],result_sha256=value['result_sha256'],exit_sha256=value['exit_sha256'])
        except factorial.PendingResult as exc:
            pending.append(dict(evaluation=name,reason=str(exc)))
        except (ValueError,KeyError,OSError,TypeError) as exc:
            invalid.append(dict(evaluation=name,error=str(exc)))
    by_seed={}
    for seed in (20261112,20261113):
        records={v['arm']:v for v in plan['formal_arms'] if v['seed']==seed}
        def name(arm):return records[arm]['evaluation80']
        def old(arm,objective='opd'):return factorial.eval_id(arm,objective,seed,80)
        def contrast(terms):return factorial.contrast(terms,available,pool,resamples=resamples)
        by_seed[str(seed)]=dict(
            lambda10_minus_lambda0={a:contrast({name(a):1,old(a):-1}) for a in factorial.MODES},
            lambda10_minus_sft={a:contrast({name(a):1,old(a,'sft'):-1}) for a in factorial.MODES},
            within_lambda10={label:contrast({name(p):1,name(n):-1}) for label,p,n in
                [('C_minus_B','C_visual','B_text'),('C_minus_A','C_visual','A_plain'),('B_minus_A','B_text','A_plain')]},
            visual_by_lambda_interaction=contrast({name('C_visual'):1,name('B_text'):-1,
                old('C_visual'):-1,old('B_text'):1}))
    bad=[v for seed in by_seed.values() for group in seed.values()
         for v in ([group] if 'status' in group else group.values()) if v['status']=='VALIDATION_FAILED']
    return dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
        status='VALIDATION_FAILED' if invalid or bad else ('COMPLETE' if len(evaluations)==6 else 'PARTIAL'),
        plan_sha256=warm.file_sha(root/'LAMBDA10_PROTOCOL.json'),source_sha256=warm.file_sha(__file__),
        complete_final_evaluations=len(evaluations),planned_final_evaluations=6,
        evaluations=evaluations,by_seed=by_seed,pending=pending,invalid=invalid,
        invalid_contrasts=bad,holdout_accessed=False,training_dispatched=False,
        limitations=['Fixed step80, same initial head, pool, full schedule, original trainer and precision; only lambda changes.',
            'Complete40-video results only; actual input SHA and independent AR must match across every contrast.',
            'Reused development set and unadjusted paired CIs are exploratory; two seeds are never pooled as80 videos.',
            'Concurrent wall times are descriptive; effective tokens/round is not end-to-end speedup.',
            'This reproduces the auxiliary-loss comparison on the current bounded task, not the old checkpoint or1024-token task.',
            'Target weight identity relies on the existing externally verified local manifest.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=warm.ROOT)
    parser.add_argument('--resamples',type=int,default=20000)
    args=parser.parse_args()
    value=build_summary(args.run_root,resamples=args.resamples)
    path=args.run_root.resolve()/'analysis/lambda10_summary.json'
    if path.exists():
        if factorial.read(path).get('schema')!=SCHEMA:raise ValueError('refusing to overwrite another artifact')
        value['previous_summary_sha256']=warm.file_sha(path)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    print(json.dumps({k:value[k] for k in ('status','complete_final_evaluations','planned_final_evaluations')}))
    if invalid:=value['invalid'] or value['invalid_contrasts']:
        raise SystemExit(1)


if __name__=='__main__':main()
