"""CPU counterexamples for continuation auditing, never model/GPU execution."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import compare_long_state as compare


def test_tensor_metrics_and_rng_change_are_not_hidden_by_metadata():
    left = {'head': {'w': torch.tensor([1., 2., 3.])},
            'rng': (torch.tensor([1, 2], dtype=torch.uint8), np.array([3, 4]))}
    right = copy.deepcopy(left)
    right['head']['w'][1] += .25
    right['rng'][1][0] = 7
    result = compare.compare_tree(left, right)
    assert not result['exact'] and result['tensors'] == 2
    tensor = next(d for d in result['differences'] if d['path'] == 'state.head.w')
    assert tensor['different_elements'] == 1
    assert tensor['max_abs'] == .25 and tensor['l2'] == .25
    assert tensor['mean_abs'] == .25 / 3
    assert any(d['path'] == 'state.rng[1]' for d in result['differences'])


def test_log_removes_only_declared_runtime_fields():
    row = {'update': 2, 'loss': 3., 'seconds': 99.,
           'qa': [{'task': 'x', 'teacher_cache_hit': False, 'teacher_cache_key': 'same', 'loss': 2.}]}
    other = copy.deepcopy(row); other['seconds'] = 1.
    other['qa'][0]['teacher_cache_hit'] = True
    assert compare.semantic_log(row) == compare.semantic_log(other)
    other['qa'][0]['loss'] += .1
    assert not compare.compare_tree(compare.semantic_log(row), compare.semantic_log(other))['exact']


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(compare.state, 'ROOT', tmp_path)
    a, b = tmp_path / 'continuous' / 'step2', tmp_path / 'resumed' / 'step2'
    a.mkdir(parents=True); b.mkdir(parents=True)
    parent = a.parent / 'step1'; parent.mkdir()
    manifest = {'training_state.pt': 'a' * 64}
    (parent / 'SHA256.json').write_text(json.dumps(manifest))
    parent_sha = compare.state.digest(manifest)
    protocol = {'protocol_sha256': 'p'}
    left = {'update_step': 2, 'head': {'w': torch.tensor([1., 2.])},
            'master_optimizer': {'master': {'w': torch.tensor([1., 2.])}},
            'rng': {'torch': torch.tensor([1], dtype=torch.uint8)}, 'parent_manifest_sha256': None}
    right = copy.deepcopy(left); right['parent_manifest_sha256'] = parent_sha
    def reader(path, **kwargs):
        return (left if path == a else right), protocol, 'manifest'
    monkeypatch.setattr(compare.state, 'read_checkpoint', reader)
    audit = dict(status='EXACT_RESTORE_VERIFIED', complete_state_exact=True,
        next_update_started=False, restored_step=1, protocol_sha256='p',
        parent_manifest_sha256=parent_sha, checks={'head': {}, 'master_optimizer': {}, 'rng': {}})
    (b.parent / 'RESTORE_AUDIT.json').write_text(json.dumps(audit))
    for path in (a, b):
        (path.parent / 'log.jsonl').write_text(json.dumps(
            {'update': 2, 'protocol_sha256': 'p', 'loss': 1., 'qa': [{'task': 'same'}]}) + '\n')
    return a, b, left, right


def test_loading_gate_and_training_equality_are_separate(runs):
    a, b, _, right = runs
    exact = compare.compare_checkpoints(a, b)
    assert exact['status'] == 'CONTINUATION_EXACT'
    right['master_optimizer']['master']['w'][0] += .01
    changed = compare.compare_checkpoints(a, b)
    assert changed['complete_state_restoration_gate'] == 'PASSED'
    assert changed['status'] == 'CONTINUATION_DIFFERENT'
    assert not changed['tolerance_applied']


def test_missing_restore_gate_and_wrong_parent_rejected(runs):
    a, b, _, _ = runs
    gate = b.parent / 'RESTORE_AUDIT.json'
    audit = json.loads(gate.read_text())
    audit['complete_state_exact'] = False
    gate.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='pre-update'):
        compare.compare_checkpoints(a, b)
    audit['complete_state_exact'] = True; gate.write_text(json.dumps(audit))
    (a.parent / 'step1/SHA256.json').write_text(json.dumps({'different': 'b' * 64}))
    with pytest.raises(ValueError, match='restored parent'):
        compare.compare_checkpoints(a, b)
