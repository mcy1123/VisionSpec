#!/usr/bin/env python3
"""Strict paired dev40 factorial summaries; no training or holdout access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import summarize_warm_dev as warm

ROOT = warm.ROOT
SCHEMA = 'local_factorial80_dev40_paired_v1'
MODES = {'A_plain': 'none', 'B_text': 'text', 'C_visual': 'visual'}
COMMON_CONFIG = tuple(key for key in warm.CONFIG_FIELDS if key != 'evidence_mode') + (
    'evidence_intervention_seed', 'max_memory', 'evidence_donor_memory', 'visual_memory')


class PendingResult(ValueError):
    """Missing or incomplete evaluation; no training failure is inferred."""


def read(path):
    return json.loads(Path(path).read_text())


def options(argv):
    result = {}
    for index, value in enumerate(argv):
        if value.startswith('--'):
            if value in result:
                raise ValueError('duplicate queue CLI option: ' + value)
            result[value] = argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith('--') else True
    return result


def eval_id(arm, objective, seed, step):
    return f'factorial80_128f_{arm}_{objective}_seed{seed}_eval{step}_dev40'


def validate_complete(value, pool):
    """Do not compare a subset even when an evaluator wrote LOSSLESS_PASSED."""
    actual, expected = set(value['rows']), set(pool['sources'])
    if actual - expected:
        raise ValueError('foreign video outside the declared development pool')
    if actual != expected:
        raise PendingResult(f'incomplete dev40: {len(actual)}/40 unique expected videos')


def load_arm(job, pool, arm, objective, seed, step, root):
    path = Path(job['result']['path']).resolve()
    if path.parent != Path(job['job_dir']).resolve() or not path.is_relative_to(root / 'jobs'):
        raise ValueError('evaluation result must be inside its declared run job directory')
    state = warm.completion(path)
    if state != 'READY':
        raise PendingResult(state + '; evaluation unavailable, training failure not inferred')
    raw = read(path)
    # A valid but incomplete result remains pending. Duplicate/foreign rows are
    # validation errors, not acceptable partial data or a shorter benchmark.
    videos = [row.get('video') for row in raw.get('rows', [])]
    if len(videos) != len(set(videos)) or set(videos) - set(pool['sources']):
        raise ValueError('duplicate or foreign video in completed result')
    if set(videos) != set(pool['sources']):
        raise PendingResult(f'incomplete dev40: {len(videos)}/40 unique expected videos')
    value = warm.load_evaluation(path, pool, 128)
    validate_complete(value, pool)
    config, protocol = value['config'], value['protocol']
    expected = dict(ar=True, warmup=0, dtype='float32', budget=63, branch=8,
                    time_mode='real', require_exact_time=True, eos_stop='immediate',
                    zero_visual=False, evidence_intervention='normal', evidence_mode=MODES[arm],
                    shards=1, shard=0)
    for key, item in expected.items():
        if config.get(key) != item:
            raise ValueError(f'config.{key} differs from the registered normal fp32 evaluation')
    if config.get('n') not in (0, 40):
        raise ValueError('evaluation row limit is not the complete dev40')
    args = options(job['argv'])
    if Path(config['head']).resolve() != Path(args['--head']).resolve():
        raise ValueError('evaluated head differs from its queued head')
    expected_head = root / 'jobs' / eval_id(arm, objective, seed, step).split('_eval')[0] / 'run' / f'step{step}'
    if Path(config['head']).resolve() != expected_head:
        raise ValueError('head path differs from the expected architecture/objective/seed/step')
    if Path(args['--rows']).resolve() != Path(pool['path']).resolve():
        raise ValueError('queue does not request the declared development pool')
    if int(args['--max-new']) != config['max_new'] or args['--target'] != config['target']:
        raise ValueError('target or output cap differs from the queued protocol')
    if job['result']['status'] != 'LOSSLESS_PASSED':
        raise ValueError('queue does not require LOSSLESS_PASSED')
    artifacts = protocol['artifacts']
    identity = artifacts['checkpoint_identity']
    if identity['optimizer_step'] != step or identity['qa_exposures'] != 4 * step:
        raise ValueError('checkpoint optimizer step or exposure budget differs')
    actual_identity = read(expected_head / 'CHECKPOINT_IDENTITY.json')
    if identity != actual_identity or warm.file_sha(expected_head / 'CHECKPOINT_IDENTITY.json') != artifacts['checkpoint_identity_sha256']:
        raise ValueError('recorded checkpoint identity differs from the saved checkpoint')
    saved_sha = read(expected_head / 'SHA256.json')
    if any(saved_sha.get(name) != sha for name, sha in artifacts['head_weights_sha256'].items()):
        raise ValueError('evaluated head weights differ from checkpoint SHA manifest')
    if artifacts['head_config_sha256'] != warm.file_sha(expected_head / 'config.json'):
        raise ValueError('head config differs from evaluated artifact')
    training = read(expected_head.parent / 'PROTOCOL.json')
    if identity['protocol_sha256'] != training['protocol_sha256'] or training['config']['seed'] != seed:
        raise ValueError('checkpoint is not bound to the correct training protocol/seed')
    mode = protocol['evidence']['mode']
    if mode != MODES[arm]:
        raise ValueError('actual architecture differs from its factorial arm')
    if arm != 'A_plain':
        if training['evidence_architecture']['time_encoding'] != 'off':
            raise ValueError('evidence training does not have explicit time encoding off')
        evidence = protocol['evidence']
        if (evidence['dim'], evidence['max_tokens'], evidence['spatial_bins']) != (256, 128, 2):
            raise ValueError('evidence capacity differs from the matched factorial specification')
    for row in value['rows'].values():
        if row.get('first_mismatch') is not None:
            raise ValueError('nonempty mismatch record despite claimed lossless result')
        if arm != 'A_plain':
            request = row['evidence']
            if request['metadata']['encoded_sha256'] != row['input_identity']:
                raise ValueError('evidence request is not bound to actual encoded input SHA')
            if request['diagnostics']['mode'] != MODES[arm]:
                raise ValueError('actual evidence forward mode differs from arm')
    value['training_protocol_sha256'] = identity['protocol_sha256']
    return value


def assert_pairable(values, pool):
    """Architectures may differ; target/decoder/inputs/teacher outputs may not."""
    first = values[0]
    for value in values:
        validate_complete(value, pool)
        for key in COMMON_CONFIG:
            if value['config'].get(key) != first['config'].get(key):
                raise ValueError('comparison config differs at ' + key)
        for key in warm.ARTIFACT_FIELDS:
            if value['protocol']['artifacts'].get(key) != first['protocol']['artifacts'].get(key):
                raise ValueError('comparison artifact differs at ' + key)
        if value['protocol']['model_contract'] != first['protocol']['model_contract']:
            raise ValueError('target/draft dimension contract differs')
        for video in sorted(pool['sources']):
            left, right = first['rows'][video], value['rows'][video]
            if left['input_identity'] != right['input_identity']:
                raise ValueError(f'{video}: actual encoded input SHA differs')
            if left['ar_ids'] != right['ar_ids']:
                raise ValueError(f'{video}: independent AR token IDs differ')
            if (left['ar_hit_eos'], left['truncated']) != (right['ar_hit_eos'], right['truncated']):
                raise ValueError(f'{video}: independent AR EOS/cap flags differ')


def contrast(terms, available, pool, *, resamples=20000, bootstrap_seed=20261112):
    """One resampled unit is one video, also for the four-term interaction."""
    missing = [name for name in terms if name not in available]
    if missing:
        return dict(status='PENDING', missing_complete_valid_evaluations=missing)
    try:
        assert_pairable([available[name] for name in terms], pool)
    except (ValueError, KeyError) as exc:
        return dict(status='VALIDATION_FAILED', error=str(exc), terms=terms)
    paired = []
    for video in sorted(pool['sources']):
        source = next(iter(available[name]['rows'][video] for name in terms))
        components = {name: {key: available[name]['rows'][video][key] for key in (
            'tau', 'effective_tokens_per_round', 'spec_wall_s', 'spec_end_to_end_s', 'ar_wall_s')}
                      for name in terms}
        paired.append(dict(video=video, source_sha256=pool['sources'][video],
            input_identity=source['input_identity'], ar_ids_sha256=warm.digest(source['ar_ids']),
            teacher_output_tokens=source['gen_tokens'], natural_eos=source['ar_hit_eos'], components=components,
            delta_tau=sum(weight * components[name]['tau'] for name, weight in terms.items()),
            delta_effective_tokens_per_round=sum(weight * components[name]['effective_tokens_per_round'] for name, weight in terms.items())))
    result = dict(status='COMPLETE', n=40, terms=terms,
        all_input_SHA_and_independent_AR_ids_match=True, all_SPEC_exactly_AR=True,
        delta_tau=warm.paired_bootstrap([r['delta_tau'] for r in paired],seed=bootstrap_seed,resamples=resamples),
        delta_effective_tokens_per_round=warm.paired_bootstrap(
            [r['delta_effective_tokens_per_round'] for r in paired],seed=bootstrap_seed,resamples=resamples),
        paired_videos=paired)
    if len(terms)==2 and sorted(terms.values())==[-1,1]:
        positive = next(name for name, value in terms.items() if value==1)
        negative = next(name for name, value in terms.items() if value==-1)
        result['timing_descriptive_only'] = dict(
            direction='negative-term runtime / positive-term runtime; greater than one favors positive term',
            total_spec_wall_ratio=sum(r['components'][negative]['spec_wall_s'] for r in paired)/sum(r['components'][positive]['spec_wall_s'] for r in paired),
            total_spec_end_to_end_ratio=sum(r['components'][negative]['spec_end_to_end_s'] for r in paired)/sum(r['components'][positive]['spec_end_to_end_s'] for r in paired),
            per_video_spec_wall_ratio=warm.stats(r['components'][negative]['spec_wall_s']/r['components'][positive]['spec_wall_s'] for r in paired))
    return result


def build_summary(root, *, resamples=20000, bootstrap_seed=20261112):
    root=Path(root).resolve()
    plan=read(root/'FACTORIAL80_PROTOCOL.json')
    if plan['arms']!=list(MODES) or plan['frame_count']!=128 or plan['evaluation']['development_sources']!=40:
        raise ValueError('unsupported factorial architecture/frame/development protocol')
    seeds=plan['training_seeds']; steps=plan['budget']['evaluate_development_steps']
    if len(seeds)!=2 or len(set(seeds))!=2 or steps!=[20,40,80]:
        raise ValueError('requires two distinct seeds and registered steps20/40/80')
    pool=warm.pool_identity(root/'data/bounded_summary_v2/dev_128.parquet',128)
    queue=read(root/'control/scheduler/queue.json')['jobs']
    if len({job['id'] for job in queue})!=len(queue):
        raise ValueError('duplicate queue job identity')
    jobs={job['id']:job for job in queue}
    duplicate_runs=[]
    for seed in seeds:
        alias=f'mature_warm80_128f_seed{seed}'
        formal=f'factorial80_128f_A_plain_sft_seed{seed}'
        if alias in jobs and formal in jobs:
            left,right=options(jobs[alias]['argv']),options(jobs[formal]['argv'])
            left.pop('--out',None);right.pop('--out',None)
            duplicate_runs.append(dict(alias=alias,formal_factorial_run=formal,
                same_entry_and_CLI_except_output=(jobs[alias]['argv'][:3]==jobs[formal]['argv'][:3] and left==right),
                included_as_extra_evidence=False,
                note='A repeat of the same warm20-to80 training recipe is not an independent stronger baseline; only the formal factorial ID is summarized.'))
    available, evaluations, pending, invalid = {}, {}, [], []
    for seed in seeds:
        for step in steps:
            for objective in ('sft','opd'):
                for arm in MODES:
                    name=eval_id(arm,objective,seed,step)
                    try:
                        if name not in jobs:
                            raise PendingResult('evaluation is not yet present in the finite queue')
                        value=load_arm(jobs[name],pool,arm,objective,seed,step,root)
                        available[name]=value
                        evaluations[name]=dict(arm=arm,objective=objective,seed=seed,step=step,
                            summary=warm.aggregate(value['rows']),
                            artifacts={key:value[key] for key in ('path','result_sha256','exit_sha256')},
                            head=value['config']['head'], checkpoint_artifacts=value['protocol']['artifacts'],
                            training_protocol_sha256=value['training_protocol_sha256'])
                    except PendingResult as exc:
                        pending.append(dict(evaluation=name,status='PENDING',reason=str(exc),training_failure_inferred=False))
                    except (ValueError,KeyError,OSError,TypeError) as exc:
                        invalid.append(dict(evaluation=name,status='EVALUATION_VALIDATION_FAILED',error=str(exc),training_failure_inferred=False))
    by_seed={}
    for seed in seeds:
        by_step={}
        for step in steps:
            objective_results={}
            for objective in ('sft','opd'):
                comparisons={}
                for label,positive,negative in (('C_minus_B','C_visual','B_text'),('C_minus_A','C_visual','A_plain'),('B_minus_A','B_text','A_plain')):
                    terms={eval_id(positive,objective,seed,step):1,eval_id(negative,objective,seed,step):-1}
                    comparisons[label]=contrast(terms,available,pool,resamples=resamples,bootstrap_seed=bootstrap_seed)
                objective_results[objective]=comparisons
            terms={eval_id('C_visual','opd',seed,step):1,eval_id('B_text','opd',seed,step):-1,
                   eval_id('C_visual','sft',seed,step):-1,eval_id('B_text','sft',seed,step):1}
            by_step[str(step)]=dict(within_objective=objective_results,
                interaction_CminusB_OPD_minus_SFT=contrast(terms,available,pool,resamples=resamples,bootstrap_seed=bootstrap_seed))
        by_seed[str(seed)]=dict(video_units_per_comparison=40,by_step=by_step)
    all_contrasts=[value for seed in by_seed.values() for stage in seed['by_step'].values()
                   for value in [stage['interaction_CminusB_OPD_minus_SFT'],
                                 *stage['within_objective']['sft'].values(),*stage['within_objective']['opd'].values()]]
    invalid_contrasts=[c for c in all_contrasts if c['status']=='VALIDATION_FAILED']
    return dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
        status='VALIDATION_FAILED' if invalid or invalid_contrasts else ('COMPLETE' if len(available)==36 else 'PARTIAL'),
        plan_sha256=warm.file_sha(root/'FACTORIAL80_PROTOCOL.json'),queue_sha256=warm.file_sha(root/'control/scheduler/queue.json'),
        summary_source_sha256=warm.file_sha(__file__),row_validation_source_sha256=warm.file_sha(warm.__file__),
        complete_evaluations=len(available),planned_evaluations=36,
        complete_contrasts=sum(c['status']=='COMPLETE' for c in all_contrasts),planned_contrasts=42,
        pool=pool,evaluations=evaluations,by_seed=by_seed,pending=pending,invalid=invalid,
        invalid_contrasts=invalid_contrasts,duplicate_run_notices=duplicate_runs,
        holdout_accessed=False,training_dispatched=False,
        limitations=[
            'Only complete40-video evaluations with EXIT0, LOSSLESS_PASSED, exact SPEC=independentAR tokens and matched actual encoded-input SHA enter contrasts.',
            'Each seed is reported separately. Bootstrap resamples40 paired source videos; two seeds are never pooled as80 independent videos.',
            'The interaction is computed per video as(C_OPD-B_OPD)-(C_SFT-B_SFT), then bootstrapped; confidence-interval endpoints are never subtracted.',
            'Development40 is reused for checkpoint/hyperparameter selection; unadjusted percentile95% intervals are exploratory, not independent confirmation.',
            'Pending/incomplete evaluation does not establish training failure. No automatic winner selection or dispatch is performed.',
            'mature_warm80 runs repeat the corresponding A_plain SFT80 recipe and are not counted as independent evidence or as a stronger baseline.',
            'Concurrent GPU load and execution order make all wall-clock quantities descriptive, not formal speed evidence.',
            'Raw tau counts accepted+1 per round; effective tokens/round includes the separate initial prefill root and returned-output EOS/cap clipping.',
            'Target weight identity relies on the existing external verified manifest; this summary checks recorded target/config/code identities and exact independentAR tokens.',
            'Only declared dev40 files are read. Future holdout80 files and model execution are untouched.'])


def write_summary(path,value,root):
    path,root=Path(path).resolve(),Path(root).resolve()
    if not path.is_relative_to(root/'analysis'):
        raise ValueError('summary must remain under this run/analysis')
    if path.exists():
        if read(path).get('schema')!=SCHEMA:
            raise ValueError('refusing to overwrite a non-factorial-summary artifact')
        value=dict(value,previous_summary_sha256=warm.file_sha(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--resamples',type=int,default=20000)
    parser.add_argument('--bootstrap-seed',type=int,default=20261112)
    args=parser.parse_args(argv)
    value=build_summary(args.run_root,resamples=args.resamples,bootstrap_seed=args.bootstrap_seed)
    out=args.out or args.run_root/'analysis/factorial80_summary.json'
    write_summary(out,value,args.run_root)
    print(json.dumps({key:value[key] for key in ('status','complete_evaluations','planned_evaluations','complete_contrasts')}))


if __name__=='__main__':
    main()
