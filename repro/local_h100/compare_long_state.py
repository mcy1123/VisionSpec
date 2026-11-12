#!/usr/bin/env python3
"""CPU audit of complete restoration and subsequent continuous/resumed states.

Loading fidelity is a hard gate recorded before the resumed update. Subsequent
CUDA training equality is reported separately, with no tolerance introduced
after seeing a difference. This tool never initializes CUDA or runs a model.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import long_training_state as state
from shared import training_resume

SCHEMA = 'local_long_continuation_comparison_v1'
IGNORED_STATE_FIELDS = {'parent_manifest_sha256'}
IGNORED_LOG_FIELDS = {'seconds', 'elapsed_seconds', 'segment_start_step',
                      'segment_end_step', 'teacher_cache_hit'}


def tensor_difference(left, right):
    """Bound temporary RAM while quantifying every differing tensor element."""
    x, y = left.detach().cpu().reshape(-1), right.detach().cpu().reshape(-1)
    unequal, total, squared, maximum = 0, 0., 0., 0.
    for start in range(0, x.numel(), 1_000_000):
        a, b = x[start:start + 1_000_000], y[start:start + 1_000_000]
        unequal += int((a != b).sum())
        delta = (a.double() - b.double()).abs()
        total += float(delta.sum())
        squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.max()) if delta.numel() else 0.)
    return dict(different_elements=unequal, elements=x.numel(), max_abs=maximum,
                mean_abs=total / max(1, x.numel()), l2=math.sqrt(squared))


def compare_tree(left, right, path='state'):
    result = dict(exact=True, tensors=0, tensor_elements=0, arrays=0,
                  scalars=0, differences=[])
    def mismatch(where, kind, **details):
        result['exact'] = False
        result['differences'].append(dict(path=where, kind=kind, **details))
    def visit(a, b, where):
        if type(a) is not type(b):
            mismatch(where, 'type', baseline=str(type(a)), resumed=str(type(b)))
        elif torch.is_tensor(a):
            result['tensors'] += 1; result['tensor_elements'] += a.numel()
            if a.device.type != 'cpu' or b.device.type != 'cpu':
                raise ValueError('comparison requires CPU tensors only')
            if a.shape != b.shape or a.dtype != b.dtype:
                mismatch(where, 'tensor_layout', baseline_shape=list(a.shape), resumed_shape=list(b.shape),
                         baseline_dtype=str(a.dtype), resumed_dtype=str(b.dtype))
            elif not torch.equal(a, b):
                mismatch(where, 'tensor_values', dtype=str(a.dtype), shape=list(a.shape),
                         **tensor_difference(a, b))
        elif isinstance(a, np.ndarray):
            result['arrays'] += 1
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                mismatch(where, 'array_values_or_layout')
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                mismatch(where, 'keys', baseline_only=sorted(map(str, a.keys() - b.keys())),
                         resumed_only=sorted(map(str, b.keys() - a.keys())))
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f'{where}.{key}')
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                mismatch(where, 'length', baseline=len(a), resumed=len(b))
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f'{where}[{index}]')
        else:
            result['scalars'] += 1
            if a != b:
                details = dict(baseline=repr(a)[:300], resumed=repr(b)[:300])
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    difference = abs(a - b)
                    if math.isfinite(difference):
                        details['abs_difference'] = difference
                mismatch(where, 'scalar_value', **details)
    visit(left, right, path)
    return result


def semantic_log(value):
    if isinstance(value, dict):
        return {k: semantic_log(v) for k, v in value.items() if k not in IGNORED_LOG_FIELDS}
    if isinstance(value, list):
        return [semantic_log(v) for v in value]
    return value


def log_record(path, step, protocol_sha):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row.get('update') == step]
    if len(matches) != 1 or matches[0].get('protocol_sha256') != protocol_sha:
        raise ValueError('comparison log must contain exactly one matching update/protocol')
    if 'loss' not in matches[0] or not ('qa' in matches[0] or 'tasks' in matches[0]):
        raise ValueError('comparison log lacks loss or QA/task order')
    return semantic_log(matches[0])


def compare_checkpoints(baseline, resumed, *, allow_cpu_test=False):
    baseline, resumed = state.safe_path(baseline), state.safe_path(resumed)
    if baseline == resumed or baseline.parent == resumed.parent:
        raise ValueError('continuous and restored checkpoints must be in distinct segments')
    left, lp, lh = state.read_checkpoint(baseline, allow_cpu_test=allow_cpu_test)
    right, rp, rh = state.read_checkpoint(resumed, allow_cpu_test=allow_cpu_test)
    if lp != rp or left['update_step'] != right['update_step']:
        raise ValueError('logical training protocol or absolute checkpoint step differs')
    restore_path = resumed.parent / 'RESTORE_AUDIT.json'
    restore = json.loads(restore_path.read_text())
    if (restore.get('status') != 'EXACT_RESTORE_VERIFIED'
            or restore.get('complete_state_exact') is not True
            or restore.get('next_update_started') is not False
            or restore.get('protocol_sha256') != rp['protocol_sha256']
            or restore.get('parent_manifest_sha256') != right.get('parent_manifest_sha256')
            or not 0 < restore.get('restored_step', 0) < right['update_step']
            or not {'head', 'master_optimizer', 'rng'} <= set(restore.get('checks', {}))):
        raise ValueError('resumed segment lacks its exact pre-update complete-state restore gate')
    parent = baseline.parent / f"step{restore['restored_step']}" / 'SHA256.json'
    if state.digest(json.loads(parent.read_text())) != restore['parent_manifest_sha256']:
        raise ValueError('restored parent is not the continuous run matching-step checkpoint')
    compared = compare_tree({k: v for k, v in left.items() if k not in IGNORED_STATE_FIELDS},
                            {k: v for k, v in right.items() if k not in IGNORED_STATE_FIELDS})
    logs = [p.parent / 'log.jsonl' for p in (baseline, resumed)]
    log_comparison = compare_tree(
        log_record(logs[0], left['update_step'], lp['protocol_sha256']),
        log_record(logs[1], right['update_step'], rp['protocol_sha256']), 'update_log')
    exact = compared['exact'] and log_comparison['exact']
    return dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                status='CONTINUATION_EXACT' if exact else 'CONTINUATION_DIFFERENT',
                complete_state_restoration_gate='PASSED', continuation_exact=exact,
                absolute_step=left['update_step'], protocol_sha256=lp['protocol_sha256'],
                baseline=dict(path=str(baseline), manifest_sha256=lh),
                resumed=dict(path=str(resumed), manifest_sha256=rh),
                restore_audit=dict(path=str(restore_path), sha256=training_resume.file_hash(restore_path), **restore),
                state_comparison=compared, log_comparison=log_comparison,
                log_sha256={str(p): training_resume.file_hash(p) for p in logs},
                ignored_state_fields=sorted(IGNORED_STATE_FIELDS), ignored_log_fields=sorted(IGNORED_LOG_FIELDS),
                gpu_used=False, tolerance_applied=False,
                interpretation='Exact loading and subsequent training equality are separate claims. '
                    'Any continuation differences are quantified without a post-hoc tolerance. '
                    'This comparison does not establish deployment SPEC=AR; that remains a separate gate.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_checkpoints(args.baseline, args.resumed)
    state.write_json(args.out, result)
    print(json.dumps({key: result[key] for key in (
        'status', 'complete_state_restoration_gate', 'absolute_step', 'continuation_exact')}))
    return result


if __name__ == '__main__':
    main()
