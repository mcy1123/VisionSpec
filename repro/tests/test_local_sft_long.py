"""CPU lifecycle proofs for the new SFT entry; no real video or CUDA work."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import configure
configure()
from local_h100 import train_video_sft_long as entry
from local_h100 import train_video_sft as base
from local_h100 import train_video_evidence_sft as evidence_entry
from local_h100 import evidence_sft
from shared.master_optim import MasterWeightAdamW
from shared import training_resume
from test_evidence_readout import make_head, request
from test_local_evidence_sft import Target


def tiny_head(mode):
    model = make_head(mode, nonzero=mode != 'none')
    if mode != 'none':
        model.config.dflash_config['evidence_time_encoding'] = 'off'
        model.evidence_readout.time.weight.data.zero_()
        evidence_sft.freeze_time_after_unfreeze(model)
    for name, parameter in model.named_parameters():
        if not name.startswith('evidence_readout.'):
            parameter.data = parameter.data.to(torch.bfloat16)
    return model


def optimizer(model):
    return MasterWeightAdamW(model, lr=1e-3, betas=(.9, .999),
                             weight_decay=.01, where='cpu', clip=1.)


def args():
    return SimpleNamespace(seed=137, qa_per_update=2, anchors_per_qa=3,
                           max_new_tokens=16, max_seq_len=64,
                           total_updates=2, save_every=20,
                           deadline_utc=None, stop_file=None)


def prepared(tmp_path, start=0, end=2):
    protocol = {'config': {'total_updates': 2, 'qa_per_update': 2}, 'test_only': True}
    protocol['protocol_sha256'] = entry.state._protocol_hash(protocol)
    return dict(start_step=start, end_step=end, out=tmp_path,
                rows=pd.DataFrame([{'task': 'a', 'id': 'a'}, {'task': 'b', 'id': 'b'}]).set_index('task'),
                schedule={i: [{'task': 'a', 'slot': 0}, {'task': 'b', 'slot': 1}] for i in range(2)},
                protocol=protocol, gpus=[], parent_manifest_sha256=None,
                resume_state=None, _test_allow_cpu=True)


def seed_all():
    random.seed(811); np.random.seed(811); torch.manual_seed(811)


def compare_tree(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare_tree(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('mode', ['none', 'text', 'visual'])
def test_real_sft_two_updates_match_save_restore_and_absolute_cursor(tmp_path, monkeypatch, mode):
    """Actual DFlash/coverage CE/FP32 Adam, not a mock scalar loss.

    I/O stand-ins encode five prompt tokens and replay a tiny frozen target
    context. That replay consumes Python/NumPy/Torch RNG to make a missing RNG
    restore observable. This also uses the new complete checkpoint serializer,
    checksum reader and restore path; CPU fixture states cannot resume on GPUs.
    """
    from qa_opd_fair import videoencode
    monkeypatch.setattr(entry.state, 'ROOT', tmp_path)
    torch.set_num_threads(1)
    monkeypatch.setattr(videoencode, 'encode_inputs', lambda *a, **k: {
        'input_ids': torch.arange(5)[None],
        'pixel_values_videos': torch.zeros(1), 'video_grid_thw': torch.tensor([[2, 2, 2]])})
    monkeypatch.setattr(evidence_entry, 'build_evidence_request', lambda inp, row, proc: request(row['id']))
    monkeypatch.setattr(base, 'get_teacher', lambda *a, **k: (list(range(5, 14)), True, 'tiny-cache'))
    monkeypatch.setattr(base.common, 'target_context_cached', lambda *a, **k:
                        torch.randn(1, 14, 64) + random.random() + float(np.random.random()))
    target = Target(); target.model = SimpleNamespace(rope_deltas=None)
    continuous = tiny_head(mode); segmented = copy.deepcopy(continuous)
    continuous_opt, segmented_opt = optimizer(continuous), optimizer(segmented)
    all_args = args(); prep = prepared(tmp_path)
    original_anchor = base.anchor_plan
    seeds = []

    def anchor(*values):
        seeds.append(values[-1])
        return original_anchor(*values)

    monkeypatch.setattr(base, 'anchor_plan', anchor)
    def update(head, opt, cursor):
        runtime = evidence_entry.evidence_runtime() if mode != 'none' else nullcontext()
        with runtime:
            return entry.train_update(prep, all_args, cursor, target, None, head, opt,
                                      4, 99, tmp_path, {99}, device='cpu')

    seed_all()
    expected_records = [update(continuous, continuous_opt, cursor) for cursor in range(2)]
    expected_rng = training_resume.capture_rng()
    seed_all()
    first = update(segmented, segmented_opt, 0)
    checkpoint = entry.state.save_checkpoint(prep, 1, segmented, segmented_opt)
    # Fresh process-like model construction disturbs all random streams.
    resumed = tiny_head(mode); resumed_opt = optimizer(resumed)
    random.random(); np.random.random(); torch.randn(7)
    saved, protocol, parent_hash = entry.state.read_checkpoint(checkpoint, allow_cpu_test=True)
    resume = dict(prep, start_step=1, resume_state=saved, protocol=protocol,
                  parent_manifest_sha256=parent_hash)
    cursor = entry.state.restore_if_needed(resume, resumed, resumed_opt)
    assert cursor == 1 and resume['resume_state'] is None
    second = update(resumed, resumed_opt, cursor)
    compare_tree(expected_records, [first, second])
    compare_tree(continuous.state_dict(), resumed.state_dict())
    compare_tree(continuous_opt.state_dict(), resumed_opt.state_dict())
    compare_tree(expected_rng, training_resume.capture_rng())
    assert seeds == [137, 138, 100140, 100141] * 2
    assert all(parameter.grad is None for parameter in target.parameters())
    if mode != 'none':
        evidence_sft.assert_time_off(resumed)
        assert resumed.evidence_readout.request is None
        assert resumed.evidence_readout.query.weight.grad.abs().sum() > 0


def test_segment_uses_absolute_schedule_and_saves_last_committed_update(tmp_path, monkeypatch):
    prep = prepared(tmp_path, start=1, end=2)
    progress = dict(completed=1, last_checkpoint=None, reason=None)
    calls = []
    def update(*values):
        calls.append(values[2]); return dict(loss=1., gradient_norm=1., written_update_norm=1., qa=[])
    monkeypatch.setattr(entry.state, 'save_checkpoint', lambda p, step, *rest:
                        calls.append(('save', step)) or (p['out'] / f'step{step}'))
    monkeypatch.setattr(entry.state, 'finish_segment', lambda p, *values: values)
    result = entry.run_segment(prep, args(), None, None, None, None, 4, 99,
                               tmp_path, {99}, progress, update_fn=update)
    assert calls == [1, ('save', 2)]
    assert result == (2, None, str(tmp_path / 'step2'))
    records = [json.loads(line) for line in (tmp_path / 'log.jsonl').read_text().splitlines()]
    assert [row['update'] for row in records] == [2]
    assert records[0]['segment_start_step'] == 1


def test_early_stop_saves_only_completed_update(tmp_path, monkeypatch):
    prep = prepared(tmp_path)
    progress = dict(completed=0, last_checkpoint=None, reason=None)
    reasons = iter([None, 'requested stop'])
    monkeypatch.setattr(base, 'stop_reason', lambda *values: next(reasons))
    saved = []
    monkeypatch.setattr(entry.state, 'save_checkpoint', lambda p, step, *rest:
                        saved.append(step) or (p['out'] / f'step{step}'))
    monkeypatch.setattr(entry.state, 'finish_segment', lambda p, *values: values)
    result = entry.run_segment(prep, args(), None, None, None, None, 4, 99,
                               tmp_path, {99}, progress,
                               update_fn=lambda *values: dict(loss=1., qa=[]))
    assert saved == [1]
    assert result == (1, 'requested stop', str(tmp_path / 'step1'))


def test_failed_update_is_not_saved_or_reported_completed(tmp_path, monkeypatch):
    prep = prepared(tmp_path)
    progress = dict(completed=0, last_checkpoint=None, reason=None)
    monkeypatch.setattr(entry.state, 'save_checkpoint', lambda *values: pytest.fail('failed update saved'))
    monkeypatch.setattr(entry.state, 'finish_segment', lambda *values: pytest.fail('failed segment completed'))
    def fail(*values):
        raise RuntimeError('injected backward failure')
    with pytest.raises(RuntimeError, match='backward failure'):
        entry.run_segment(prep, args(), None, None, None, None, 4, 99,
                           tmp_path, {99}, progress, update_fn=fail)
    assert progress['completed'] == 0 and progress['last_checkpoint'] is None
    assert not (tmp_path / 'log.jsonl').exists()


def test_out_of_segment_cursor_rejected_before_model_work(tmp_path):
    with pytest.raises(ValueError, match='outside this segment'):
        entry.train_update(prepared(tmp_path, 1, 2), args(), 0,
                           None, None, None, None, 4, 99, tmp_path, {99})


@pytest.mark.parametrize('mode', ['none', 'text', 'visual'])
def test_one_update_matches_independent_original_coverage_ce_reference(tmp_path, monkeypatch, mode):
    """Separate reference: original loss helper plus explicit mean-QA update."""
    from qa_opd_fair import videoencode
    torch.set_num_threads(1)
    monkeypatch.setattr(videoencode, 'encode_inputs', lambda *a, **k: {
        'input_ids': torch.arange(5)[None], 'pixel_values_videos': torch.zeros(1),
        'video_grid_thw': torch.tensor([[2, 2, 2]])})
    monkeypatch.setattr(evidence_entry, 'build_evidence_request', lambda inp, row, proc: request(row['id']))
    monkeypatch.setattr(base, 'get_teacher', lambda *a, **k: (list(range(5, 14)), True, 'cache'))
    actual = tiny_head(mode); reference = copy.deepcopy(actual)
    oa, ob = optimizer(actual), optimizer(reference)
    target = Target(); target.model = SimpleNamespace(rope_deltas=None)
    contexts = [torch.randn(1, 14, 64).to(torch.bfloat16) for _ in range(2)]
    iterator = iter(contexts)
    monkeypatch.setattr(base.common, 'target_context_cached', lambda *a, **k: next(iterator))
    runtime = evidence_entry.evidence_runtime() if mode != 'none' else nullcontext()
    with runtime:
        result = entry.train_update(prepared(tmp_path), args(), 0, target, None, actual,
                                    oa, 4, 99, tmp_path, {99}, device='cpu')
    ob.zero_grad(set_to_none=True)
    losses = []
    for slot, context in enumerate(contexts):
        anchors, reveals = base.anchor_plan(5, 14, 3, 4, 137 + slot)
        if mode == 'none':
            loss, _ = base.hidden_loss(reference, target, context, torch.arange(14), anchors, reveals, 99)
        else:
            loss, _ = evidence_sft.hidden_loss(base.hidden_loss, reference, target, context,
                torch.arange(14), anchors, reveals, 99, request(('a', 'b')[slot]))
        losses.append(float(loss.detach()))
        (loss / 2).backward()
    ob.sync_grads(); expected_norm = ob.grad_norm(); ob.step()
    assert result['loss'] == sum(losses) / 2
    assert result['gradient_norm'] == expected_norm
    compare_tree(actual.state_dict(), reference.state_dict())
    compare_tree(oa.state_dict(), ob.state_dict())


@pytest.mark.parametrize('restore_fails', [False, True])
def test_main_restore_after_initialization_and_before_update_without_cuda(tmp_path, monkeypatch, restore_fails):
    """Exercise main's order and failure marker with allocation-free stand-ins."""
    from shared import draft_geometry
    prep = prepared(tmp_path, 1, 2)
    prep['init_path'] = '/test/checkpoint'
    events = []
    monkeypatch.setattr(entry.state, 'prepare', lambda *a: prep)
    monkeypatch.setattr(base, 'visible_gpu_ids', lambda: ['GPU-test'])
    monkeypatch.setattr(entry, 'evidence_settings', lambda path: {'mode': 'none'})
    monkeypatch.setattr(base, 'validate_head', lambda path: (4, 99))
    monkeypatch.setattr(entry.state, 'safe_path', lambda path: Path(path))
    monkeypatch.setattr(entry.state, 'start_segment', lambda *a: events.append('start'))
    monkeypatch.setattr(torch.cuda, 'set_device', lambda value: events.append('cuda-select-stub'))
    target = Target(); target.model = SimpleNamespace(rope_deltas=None)
    def load_target(*a, **k):
        events.append('target'); return target, 'processor'
    monkeypatch.setattr(base.common, 'load_target', load_target)
    class Head:
        block_size = 4
        def eval(self): return self
        def requires_grad_(self, value):
            events.append('head-unfreeze'); return self
        def parameters(self):
            return iter([SimpleNamespace(device=torch.device('cuda:0'))])
    def load_head(*a, **k):
        events.append('head'); return Head()
    monkeypatch.setattr(base.common, 'load_head', load_head)
    def opt(*a, **k):
        assert k['where'] == 'gpu'
        events.append('optimizer'); return 'optimizer-object'
    monkeypatch.setattr(base, 'MasterWeightAdamW', opt)
    monkeypatch.setattr(draft_geometry, 'eos_ids', lambda value: events.append('eos') or {99})
    def restore(*a):
        events.append('restore')
        if restore_fails:
            raise ValueError('restore identity rejected')
        return 1
    monkeypatch.setattr(entry.state, 'restore_if_needed', restore)
    monkeypatch.setattr(entry.state, 'validate_runtime', lambda *a: events.append('validate'))
    monkeypatch.setattr(entry, 'run_segment', lambda *a: events.append('updates') or {'status': 'test'})
    markers = []
    monkeypatch.setattr(entry.state, 'write_json', lambda path, value: markers.append((path, value)))
    argv = ['--pool', 'pool', '--schedule', 'schedule', '--target', 'target',
            '--out', str(tmp_path), '--resume-from', '/test/checkpoint',
            '--total-updates', '2', '--run-updates', '1', '--seed', '137']
    if restore_fails:
        with pytest.raises(ValueError, match='restore identity'):
            entry.main(argv)
        assert 'updates' not in events
        assert len(markers) == 1 and markers[0][0].name == 'FAILED.json'
        assert markers[0][1]['completed_updates'] == 1
        assert markers[0][1]['last_checkpoint'] is None
    else:
        assert entry.main(argv) == {'status': 'test'}
        assert events[-3:] == ['restore', 'validate', 'updates']
        assert not markers
    assert events.index('optimizer') < events.index('restore')
    assert events.index('eos') < events.index('restore')
    assert all(not parameter.requires_grad for parameter in target.parameters())
