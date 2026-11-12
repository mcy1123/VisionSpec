#!/usr/bin/env python3
"""Strict paired dev40 diagnostics; original eval artifacts are read-only.

Only complete, successful, token-identical arms enter comparisons. Bootstrap
intervals resample paired videos and are exploratory on reused development
data, not an independent confirmation or a hyperparameter-selection rule.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100.summarize_baseline import digest, file_sha, stats

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
SCHEMA = 'local_warm20_dev40_paired_v1'
FRAMES, LRS, SEEDS = (64, 128), ('1e-05', '3e-06'), (20261112, 20261113)
CONFIG_FIELDS = ('target', 'max_new', 'budget', 'branch', 'ar', 'warmup', 'dtype',
                 'device_map', 'time_mode', 'require_exact_time', 'eos_stop',
                 'zero_visual', 'evidence_intervention', 'evidence_mode')
ARTIFACT_FIELDS = ('target_config_sha256', 'source_sha256')


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def pool_identity(path, frames):
    rows = pd.read_parquet(path).to_dict('records')
    if len(rows) != 40 or any(row['split'] != 'dev' or int(row['n_frames']) != frames for row in rows):
        raise ValueError('reference pool must be exactly 40 dev videos at the requested frame tier')
    videos = [str(row['video_name']) for row in rows]
    sources = [row['source_sha256'] for row in rows]
    if len(set(videos)) != 40 or len(set(sources)) != 40 or not all(map(valid_sha, sources)):
        raise ValueError('dev pool has duplicate/missing video or source identity')
    return dict(path=str(path), sha256=file_sha(path),
                sources={video: source for video, source in zip(videos, sources)},
                caches={str(row['video_name']): row['cache_hash'] for row in rows})


def completion(path):
    exit_path = path.parent / 'EXIT.json'
    if not exit_path.is_file():
        return 'PENDING'
    value = json.loads(exit_path.read_text())
    if value.get('exit_code') != 0:
        return 'EXIT_FAILED'
    return 'READY' if path.is_file() else 'RESULT_MISSING_AFTER_EXIT'


def validate_row(row):
    """Validate immediate-EOS traces, including the separate prefill root.

    Decoder pos is the already materialized root. A round writes its accepted
    path and bonus at pos+1 ... pos+accepted+1. The first root is emitted by
    prefill, so natural bonus EOS can give effective > raw tau by 1/rounds.
    """
    if row.get('lossless') is not True or row.get('spec_ids') != row.get('ar_ids'):
        raise ValueError('SPEC differs from independent AR tokens')
    n, rounds = row['gen_tokens'], row['rounds']
    if n < 1 or len(row['spec_ids']) != n or row['ar_tokens'] != n:
        raise ValueError('output token count mismatch')
    positions, accepted = row['anchor_rel'], row['accepted']
    if rounds < 1 or len(positions) != rounds or len(accepted) != rounds:
        raise ValueError('round trace count mismatch')
    if positions[0] != 0 or any(p < 0 or a < 0 for p, a in zip(positions, accepted)):
        raise ValueError('invalid round trace')
    if any(positions[i + 1] != positions[i] + accepted[i] + 1 for i in range(rounds - 1)):
        raise ValueError('round trace is not contiguous')
    newly_emitted = sum(min(a + 1, max(0, n - p - 1)) for p, a in zip(positions, accepted))
    if positions[-1] >= n or 1 + newly_emitted != n:
        raise ValueError('prefill root plus clipped round outputs do not cover returned tokens exactly')
    if not math.isclose(row['tau'], sum(a + 1 for a in accepted) / rounds, rel_tol=1e-9):
        raise ValueError('tau does not match raw round trace')
    if not math.isclose(row['effective_tokens_per_round'], n / rounds, rel_tol=1e-9):
        raise ValueError('effective tokens/round mismatch')
    for key in ('spec_wall_s', 'ar_wall_s', 'spec_end_to_end_s', 'ar_end_to_end_s'):
        if not math.isfinite(row[key]) or row[key] <= 0:
            raise ValueError('invalid wall time')


def load_evaluation(path, pool, frames):
    if completion(path) != 'READY':
        raise ValueError(f'{path.parent.name}: requires successful EXIT and result')
    blob = json.loads(path.read_text())
    if blob.get('status') != 'LOSSLESS_PASSED' or blob['protocol'].get('lossless_checked') is not True:
        raise ValueError('evaluation did not pass the explicit lossless gate')
    config, protocol = blob['config'], blob['protocol']
    if Path(config['rows']).resolve() != Path(pool['path']).resolve() or protocol['rows_sha256'] != pool['sha256']:
        raise ValueError('result pool path/SHA differs from the current dev pool')
    if config.get('ar') is not True or config.get('warmup') != 0 or config.get('eos_stop') != 'immediate':
        raise ValueError('requires independent AR, no excluded warmup rows and immediate EOS')
    result = {}
    for row in blob['rows']:
        validate_row(row)
        video = row['video']
        if video in result or video not in pool['sources'] or row['n_frames'] != frames or row['warmup']:
            raise ValueError('duplicate, foreign or wrong-frame video in evaluation')
        identity = row['input_identity']
        if any(not valid_sha(identity.get(name)) for name in ('input_ids', 'pixel_values_videos', 'video_grid_thw')):
            raise ValueError('actual encoded input SHA is absent or malformed')
        if row['timeline'].get('cache_sha256') != pool['caches'][video]:
            raise ValueError('evaluated frame cache differs from source pool')
        if row['hit_eos'] != row['ar_hit_eos'] or row['truncated'] != (not row['hit_eos'] and row['gen_tokens'] >= config['max_new']):
            raise ValueError('SPEC/AR EOS or cap flags disagree')
        result[video] = row
    if not result:
        raise ValueError('completed result contains no rows')
    launch = path.parent / 'LAUNCH.json'
    return dict(path=str(path), result_sha256=file_sha(path), exit_sha256=file_sha(path.parent / 'EXIT.json'),
                launch=json.loads(launch.read_text()) if launch.exists() else None,
                config=config, protocol=protocol, rows=result)


def comparable(left, right):
    for key in CONFIG_FIELDS:
        if left['config'].get(key) != right['config'].get(key):
            raise ValueError(f'evaluation protocol differs at config.{key}')
    for key in ARTIFACT_FIELDS:
        if left['protocol']['artifacts'].get(key) != right['protocol']['artifacts'].get(key):
            raise ValueError(f'evaluation artifacts differ at {key}')
    if left['protocol'].get('model_contract') != right['protocol'].get('model_contract'):
        raise ValueError('target/draft dimension contract differs')


def combine_baseline(parts, pool):
    if len(parts) != 4:
        raise ValueError('all four baseline shards are required')
    rows = {}
    for index, part in enumerate(parts):
        comparable(parts[0], part)
        if part['config'].get('shards') != 4 or part['config'].get('shard') != index:
            raise ValueError('baseline shard configuration differs from its expected index')
        for key in ('head_weights_sha256', 'head_config_sha256'):
            if part['protocol']['artifacts'].get(key) != parts[0]['protocol']['artifacts'].get(key):
                raise ValueError('baseline shards do not use the same public head')
        if set(rows) & set(part['rows']):
            raise ValueError('baseline shards overlap')
        rows.update(part['rows'])
    if set(rows) != set(pool['sources']):
        raise ValueError('combined baseline is not the complete 40-source dev set')
    return dict(config=parts[0]['config'], protocol=parts[0]['protocol'], rows=rows,
                artifacts=[{key: p[key] for key in ('path', 'result_sha256', 'exit_sha256', 'launch')} for p in parts])


def aggregate(rows):
    values = list(rows.values())
    rounds = sum(row['rounds'] for row in values)
    eos = [row['gen_tokens'] for row in values if row['ar_hit_eos']]
    return dict(n=len(values), lossless=len(values),
                macro_tau=stats(r['tau'] for r in values),
                macro_effective_tokens_per_round=stats(r['effective_tokens_per_round'] for r in values),
                pooled_tau=sum(sum(a + 1 for a in r['accepted']) for r in values) / rounds,
                pooled_effective_tokens_per_round=sum(r['gen_tokens'] for r in values) / rounds,
                teacher_output_tokens=stats(r['gen_tokens'] for r in values),
                teacher_natural_eos_count=len(eos), teacher_natural_eos_output_tokens=stats(eos),
                truncated_count=sum(r['truncated'] for r in values),
                prompt_tokens=stats(r['prompt_len'] for r in values),
                visual_tokens=stats(r['n_visual_tokens'] for r in values),
                timing_descriptive_only=dict(
                    spec_wall_total_s=sum(r['spec_wall_s'] for r in values),
                    ar_wall_total_s=sum(r['ar_wall_s'] for r in values),
                    ar_over_spec_total_wall=sum(r['ar_wall_s'] for r in values) / sum(r['spec_wall_s'] for r in values),
                    ar_over_spec_total_end_to_end=sum(r['ar_end_to_end_s'] for r in values) / sum(r['spec_end_to_end_s'] for r in values)))


def paired_bootstrap(values, seed=20261112, resamples=20000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or resamples < 100:
        raise ValueError('invalid paired bootstrap input')
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), ci95_percentile=[float(lo), float(hi)],
                unit='paired source video', resamples=resamples, seed=seed,
                wins=int((values > 0).sum()), ties=int((values == 0).sum()), losses=int((values < 0).sum()))


def compare_arm(baseline, arm, pool, *, resamples=20000):
    comparable(baseline, arm)
    if set(arm['rows']) != set(pool['sources']):
        raise ValueError('candidate is not the complete 40-source dev set; partial rows cannot be compared')
    paired = []
    for video in sorted(pool['sources']):
        left, right = baseline['rows'][video], arm['rows'][video]
        if left['input_identity'] != right['input_identity']:
            raise ValueError(f'{video}: actual input tensor SHA differs')
        if left['ar_ids'] != right['ar_ids']:
            raise ValueError(f'{video}: independent AR output differs')
        if left['ar_hit_eos'] != right['ar_hit_eos'] or left['truncated'] != right['truncated']:
            raise ValueError(f'{video}: teacher EOS/cap differs')
        paired.append(dict(video=video, source_sha256=pool['sources'][video],
                           input_identity=left['input_identity'], ar_ids_sha256=digest(left['ar_ids']),
                           teacher_output_tokens=left['gen_tokens'], natural_eos=left['ar_hit_eos'],
                           baseline_tau=left['tau'], warm_tau=right['tau'], delta_tau=right['tau'] - left['tau'],
                           baseline_effective=left['effective_tokens_per_round'], warm_effective=right['effective_tokens_per_round'],
                           delta_effective=right['effective_tokens_per_round'] - left['effective_tokens_per_round'],
                           baseline_spec_wall_s=left['spec_wall_s'], warm_spec_wall_s=right['spec_wall_s'],
                           baseline_over_warm_spec_wall=left['spec_wall_s'] / right['spec_wall_s'],
                           baseline_over_warm_end_to_end=left['spec_end_to_end_s'] / right['spec_end_to_end_s'],
                           baseline_over_warm_ar_wall=left['ar_wall_s'] / right['ar_wall_s']))
    return dict(n=40, all_input_tensor_sha_and_ar_ids_exact=True, all_spec_equal_ar=True,
                delta_tau=paired_bootstrap([r['delta_tau'] for r in paired], resamples=resamples),
                delta_effective_tokens_per_round=paired_bootstrap([r['delta_effective'] for r in paired], resamples=resamples),
                timing_descriptive_only=dict(
                    baseline_over_warm_total_spec_wall=sum(r['baseline_spec_wall_s'] for r in paired) / sum(r['warm_spec_wall_s'] for r in paired),
                    per_video_wall_ratio=stats(r['baseline_over_warm_spec_wall'] for r in paired),
                    per_video_end_to_end_ratio=stats(r['baseline_over_warm_end_to_end'] for r in paired),
                    per_video_ar_wall_ratio=stats(r['baseline_over_warm_ar_wall'] for r in paired)), rows=paired)


def build_summary(root, *, resamples=20000):
    root = Path(root)
    baselines, arms, pending, invalid, pools = {}, {}, [], [], {}
    for frames in FRAMES:
        try:
            pool = pool_identity(root / f'data/bounded_summary_v2/dev_{frames}.parquet', frames)
            pools[str(frames)] = pool
            paths = [root / 'jobs' / f'bounded_v2_public_dev40_{frames}f_shard{s}' / 'result.json' for s in range(4)]
            states = [completion(p) for p in paths]
            if any(s != 'READY' for s in states):
                target = invalid if any(s not in ('READY', 'PENDING') for s in states) else pending
                target.append(dict(kind='baseline', frames=frames, shards=states))
                continue
            baseline = combine_baseline([load_evaluation(p, pool, frames) for p in paths], pool)
            baselines[str(frames)] = dict(summary=aggregate(baseline['rows']), artifacts=baseline['artifacts'])
        except (ValueError, KeyError, OSError) as exc:
            invalid.append(dict(kind='baseline', frames=frames, error=str(exc)))
            continue
        for lr in LRS:
            for seed in SEEDS:
                name = f'bounded_v2_warm20_{frames}f_lr{lr}_seed{seed}_dev40'
                path = root / 'jobs' / name / 'result.json'
                state = completion(path)
                if state != 'READY':
                    (pending if state == 'PENDING' else invalid).append(dict(kind='warm', name=name, status=state))
                    continue
                try:
                    arm = load_evaluation(path, pool, frames)
                    paired = compare_arm(baseline, arm, pool, resamples=resamples)
                    arms[name] = dict(frames=frames, lr=float(lr), seed=seed,
                                      summary=aggregate(arm['rows']), paired_vs_public=paired,
                                      artifacts={key: arm[key] for key in ('path', 'result_sha256', 'exit_sha256', 'launch')},
                                      head=arm['config']['head'],
                                      head_identity=arm['protocol']['artifacts'])
                except (ValueError, KeyError, OSError) as exc:
                    invalid.append(dict(kind='warm', name=name, error=str(exc)))
    if len(pools) == 2 and pools['64']['sources'] != pools['128']['sources']:
        raise ValueError('frame tiers do not share exactly the same 40 source videos')
    return dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                status='VALIDATION_FAILED' if invalid else ('COMPLETE' if len(arms) == 8 else 'PARTIAL'),
                complete_arms=len(arms), planned_arms=8, pools=pools, baselines=baselines,
                arms=arms, pending=pending, invalid=invalid,
                limitations=[
                    'Development videos are reused for selecting learning rate/seed; these are not independent confirmation results.',
                    'Percentile 95% bootstrap intervals resample paired source videos, not tokens; unadjusted exploratory intervals across eight arms.',
                    'No automatic winner selection or training dispatch is performed.',
                    'Concurrent GPU runs and unbalanced execution order make wall-clock comparisons descriptive only.',
                    'Natural EOS denotes the evaluator AR termination flag; output length is not a grounding or quality score.',
                    'Raw tau counts accepted+1 per round. Effective uses returned tokens, including the initial prefill root and EOS/cap clipping; bonus-EOS can yield effective=raw_tau+1/rounds.',
                    'Target weight identity is externally verified by the run; this CPU summary checks recorded config/code/input identities and exact AR tokens.'])


def write_summary(path, value, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root / 'analysis'):
        raise ValueError('derived summary must remain under this run/analysis')
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('schema') != SCHEMA:
            raise ValueError('refusing to replace a non-summary artifact')
        value = dict(value, previous_summary_sha256=file_sha(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path)
    p.add_argument('--resamples', type=int, default=20000)
    a = p.parse_args(argv)
    if a.resamples < 100:
        p.error('resamples must be at least 100')
    result = build_summary(a.run_root, resamples=a.resamples)
    out = a.out or a.run_root / 'analysis/warm_dev_summary.json'
    write_summary(out, result, a.run_root)
    print(json.dumps({key: result[key] for key in ('status', 'complete_arms', 'pending', 'invalid')}), flush=True)
    return result


if __name__ == '__main__':
    main()
