#!/usr/bin/env python3
"""Validate fixed-step long-input dev40 comparisons, never partial-row wins.

Only the already declared long-input jobs are read. No model execution, training
dispatch, new checkpoint selection or holdout access occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import summarize_factorial80 as factorial
from local_h100 import summarize_warm_dev as warm

SCHEMA = 'local_fixed_step_long_dev40_paired_v1'
READ = factorial.read


def validate_profile(value, profile_path, frames):
    profile = READ(profile_path)
    digest = warm.file_sha(profile_path)
    config, protocol = value['config'], value['protocol']
    audit = protocol['processor_profile']
    if (config.get('processor_profile_sha256') != digest or audit['sha256'] != digest
            or Path(config['processor_profile']).resolve() != profile_path.resolve()
            or audit['profile'] != profile or audit['actual_processor_instantiated'] is not True):
        raise ValueError('actual processor profile differs from the frozen profile')
    encoded = audit['actual_encodings']
    if len(encoded) != len(value['rows']) or len({r['video'] for r in encoded}) != len(encoded):
        raise ValueError('encoding audit has missing or duplicate source videos')
    by_video = {row['video']: row for row in encoded}
    if set(by_video) != set(value['rows']):
        raise ValueError('encoding audit sources differ from evaluated sources')
    for video, row in value['rows'].items():
        encoded_row = by_video[video]
        visual = (frames // 2) * 14 * 24 // 4
        if (row['processor_profile_sha256'] != digest
                or row['processor_size'] != profile['video_size']
                or encoded_row['processor_profile_sha256'] != digest
                or encoded_row['processor_size'] != profile['video_size']
                or encoded_row['n_frames'] != frames
                or encoded_row['video_grid_thw'] != [[frames // 2, 14, 24]]
                or encoded_row['visual_token_count'] != visual
                or row['n_visual_tokens'] != visual
                or row['prompt_len'] != encoded_row['input_ids_len']):
            raise ValueError(video + ': actual long-input encoding differs from frozen profile')
    memory = protocol['cuda_process_peak_memory']
    if (memory['status'] != 'COLLECTED' or memory['max_memory_allocated_bytes'] <= 0
            or memory['max_memory_reserved_bytes'] < memory['max_memory_allocated_bytes']):
        raise ValueError('missing or invalid process CUDA peak memory')


def same_head(left, right):
    if Path(left['config']['head']).resolve() != Path(right['config']['head']).resolve():
        raise ValueError('checkpoint path differs across its own gates')
    for name in ('head_weights_sha256', 'head_config_sha256',
                 'checkpoint_identity_sha256', 'checkpoint_identity'):
        if left['protocol']['artifacts'].get(name) != right['protocol']['artifacts'].get(name):
            raise ValueError('checkpoint identity differs across gates at ' + name)


def completed_result(job, pool, frames, root, profile_path):
    path = Path(job['result']['path']).resolve()
    if path.parent != Path(job['job_dir']).resolve() or not path.is_relative_to(root / 'jobs'):
        raise ValueError('result path is outside its declared job')
    state = warm.completion(path)
    if state == 'PENDING':
        raise factorial.PendingResult('evaluation has not exited')
    if state != 'READY':
        raise ValueError(state)
    result = warm.load_evaluation(path, pool, frames)
    if set(result['rows']) != set(pool['sources']):
        raise factorial.PendingResult('incomplete source set; no partial-row comparison')
    args = factorial.options(job['argv'])
    if (Path(args['--rows']).resolve() != Path(pool['path']).resolve()
            or Path(args['--head']).resolve() != Path(result['config']['head']).resolve()):
        raise ValueError('evaluated inputs/head differ from queue')
    for key, expected in dict(ar=True, warmup=0, dtype='float32', max_new=1024,
            budget=63, branch=8, time_mode='real', require_exact_time=True,
            eos_stop='immediate', zero_visual=False, evidence_intervention='normal',
            shards=1, shard=0, n=0).items():
        if result['config'].get(key) != expected:
            raise ValueError('evaluation protocol differs at ' + key)
    validate_profile(result, profile_path, frames)
    return result


def build_summary(root, *, resamples=20000):
    root = Path(root).resolve()
    data = root / 'data/dev40_long_input_fixed_spatial'
    profile_path = data / 'PROCESSOR_PROFILE.json'
    ready = READ(data / 'READY.json')
    if ready['status'] != 'READY_ACTUAL80_ENCODINGS_VALIDATED':
        raise ValueError('long-input development data is not ready')
    for path, digest in ready['view_files_sha256'].items():
        if warm.file_sha(data / path) != digest:
            raise ValueError('frozen data file changed: ' + path)
    if warm.file_sha(profile_path) != ready['processor_profile_sha256']:
        raise ValueError('processor profile changed after data validation')
    jobs = {j['id']: j for j in READ(root / 'control/scheduler/queue.json')['jobs']}
    pool128 = warm.pool_identity(root / 'data/bounded_summary_v2/dev_128.parquet', 128)
    pools = {frames: warm.pool_identity(data / f'dev_{frames}.parquet', frames) for frames in (256, 512)}
    if any(pool['sources'] != pool128['sources'] for pool in pools.values()):
        raise ValueError('long-input sources differ from the frozen dev40')
    # All expected final heads, fixed in the protocol before seeing outcomes.
    heads = [(arm, objective, seed) for arm in factorial.MODES
             for objective in ('sft', 'opd') for seed in (20261112, 20261113)]
    available, evaluations, gates, pending, invalid = {}, {}, {}, [], []
    for arm, objective, seed in heads:
        short_id = factorial.eval_id(arm, objective, seed, 80)
        prefix = f'long_dev40_{arm}_{objective}_seed{seed}'
        try:
            short = factorial.load_arm(jobs[short_id], pool128, arm, objective, seed, 80, root)
            gate_job = jobs[prefix + '_512f_peak1']
            gate_path = Path(factorial.options(gate_job['argv'])['--rows'])
            # Gate contains one predeclared longest-input source, not dev40.
            import pandas as pd
            rows = pd.read_parquet(gate_path).to_dict('records')
            if len(rows) != 1 or rows[0]['video_name'] not in pools[512]['sources']:
                raise ValueError('peak gate is not one declared development source')
            row = rows[0]
            gate_pool = dict(path=str(gate_path), sha256=warm.file_sha(gate_path),
                sources={row['video_name']: row['source_sha256']},
                caches={row['video_name']: row['cache_hash']})
            gate = completed_result(gate_job, gate_pool, 512, root, profile_path)
            same_head(short, gate)
            gates[prefix] = dict(path=gate['path'], result_sha256=gate['result_sha256'],
                process_memory=gate['protocol']['cuda_process_peak_memory'])
        except factorial.PendingResult as exc:
            pending.append(dict(head=prefix, stage='128_or_peak_gate', reason=str(exc)))
            continue
        except (ValueError, KeyError, OSError, TypeError) as exc:
            invalid.append(dict(head=prefix, stage='128_or_peak_gate', error=str(exc)))
            continue
        for frames, pool in pools.items():
            name = prefix + f'_{frames}f_full40'
            try:
                value = completed_result(jobs[name], pool, frames, root, profile_path)
                same_head(short, value)
                if value['config']['evidence_mode'] != factorial.MODES[arm]:
                    raise ValueError('evaluated architecture differs from its fixed head')
                for result_row in value['rows'].values():
                    if result_row.get('first_mismatch') is not None:
                        raise ValueError('mismatch detail is present')
                    if arm != 'A_plain':
                        evidence = result_row['evidence']
                        if (evidence['metadata']['encoded_sha256'] != result_row['input_identity']
                                or evidence['diagnostics']['mode'] != factorial.MODES[arm]):
                            raise ValueError('evidence request differs from actual input/mode')
                available[name] = value
                evaluations[name] = dict(arm=arm, objective=objective, seed=seed, frames=frames,
                    summary=warm.aggregate(value['rows']), path=value['path'],
                    result_sha256=value['result_sha256'], exit_sha256=value['exit_sha256'],
                    head_artifacts=value['protocol']['artifacts'],
                    process_memory=value['protocol']['cuda_process_peak_memory'])
            except factorial.PendingResult as exc:
                pending.append(dict(evaluation=name, reason=str(exc)))
            except (ValueError, KeyError, OSError, TypeError) as exc:
                invalid.append(dict(evaluation=name, error=str(exc)))
    comparisons = {}
    for frames, pool in pools.items():
        by_seed = {}
        for seed in (20261112, 20261113):
            def name(arm, objective):
                return f'long_dev40_{arm}_{objective}_seed{seed}_{frames}f_full40'
            def contrast(terms):
                return factorial.contrast(terms, available, pool, resamples=resamples)
            objectives = {}
            for objective in ('sft', 'opd'):
                objectives[objective] = {
                    label: contrast({name(positive, objective): 1, name(negative, objective): -1})
                    for label, positive, negative in (('C_minus_B', 'C_visual', 'B_text'),
                        ('C_minus_A', 'C_visual', 'A_plain'), ('B_minus_A', 'B_text', 'A_plain'))}
            by_seed[str(seed)] = dict(within_objective=objectives,
                interaction_CminusB_OPD_minus_SFT=contrast({name('C_visual', 'opd'): 1,
                    name('B_text', 'opd'): -1, name('C_visual', 'sft'): -1, name('B_text', 'sft'): 1}))
        comparisons[str(frames)] = by_seed
    bad_comparisons = [value for seeds in comparisons.values() for stage in seeds.values()
        for value in [stage['interaction_CminusB_OPD_minus_SFT'],
            *stage['within_objective']['sft'].values(), *stage['within_objective']['opd'].values()]
        if value['status'] == 'VALIDATION_FAILED']
    return dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        status='VALIDATION_FAILED' if invalid or bad_comparisons else ('COMPLETE' if len(available)==24 else 'PARTIAL'),
        source_sha256=warm.file_sha(__file__), complete_factorial_evaluations=len(available),
        planned_factorial_evaluations=24, evaluations=evaluations, peak_gates=gates,
        pending=pending, invalid=invalid, invalid_comparisons=bad_comparisons,
        by_frames=comparisons, holdout_accessed=False, training_dispatched=False,
        limitations=['Only complete40 paired videos enter each contrast; missing heads are not silently dropped.',
            'Two seeds remain separate; development reuse and unadjusted bootstrap intervals are exploratory.',
            'Comparisons are within the same frame tier, never across changed AR inputs/outputs.',
            'Concurrent wall times are descriptive only; peak memory covers the whole eval process.',
            'Warm20 reference is scheduled but excluded from this factorial-only summary.',
            'Long visual input does not imply thousand-token reasoning or independently held-out performance.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=warm.ROOT)
    parser.add_argument('--resamples', type=int, default=20000)
    args = parser.parse_args()
    value = build_summary(args.run_root, resamples=args.resamples)
    path = args.run_root.resolve() / 'analysis/long_dev40_summary.json'
    if path.exists():
        if READ(path).get('schema') != SCHEMA:
            raise ValueError('refusing to overwrite another artifact')
        value['previous_summary_sha256'] = warm.file_sha(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    print(json.dumps({k:value[k] for k in ('status', 'complete_factorial_evaluations', 'planned_factorial_evaluations')}))


if __name__ == '__main__':
    main()
