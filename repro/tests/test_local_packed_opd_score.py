"""Fixed-trace CPU proof: actual head, original OPD losses, gradients and Adam."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import configure
configure()
from local_h100 import packed_opd_score as packed
from local_h100 import evidence_sft as evidence
from local_h100.evidence_opd_prompt_boundary import known_visual_prefix
from opd_official_kl import adapters, runtime
from opd_official_kl.plan import response_plan
from shared.master_optim import MasterWeightAdamW
from test_evidence_readout import make_head, request
from test_local_evidence_sft import head as evidence_head, Target


def head(mode):
    return make_head('none') if mode == 'none' else evidence_head(mode, nonzero=True)


def trace(seed=19, *, uncovered=False):
    generator = torch.Generator().manual_seed(seed)
    rounds = [dict(pos=5, prop=[20, 21, 22], rejected_offsets=[1, 2, 3]),
              dict(pos=6, prop=[23, 24, 25], rejected_offsets=[2, 3]),
              dict(pos=8, prop=[26, 27, 28], rejected_offsets=[3]),
              dict(pos=11, prop=[29, 30, 31], rejected_offsets=[3])]
    for row in rounds:
        row['teacher_logp'] = -torch.rand(3, generator=generator) - 3.0
    rejected = [(row['pos'], offset) for row in rounds for offset in row['rejected_offsets']]
    plan = response_plan(5, 9, [1, 3, 6, 8], 4, rejected)
    if uncovered:
        plan = [(pos, count - 1 if pos == 11 else count) for pos, count in plan]
    return dict(prompt_len=5, ids=torch.arange(1, 15),
                fused=torch.randn(1, 14, 64, generator=generator),
                hidden=torch.randn(1, 14, 32, generator=generator),
                rounds=rounds, plan=plan, evidence_request=request(f'video-{seed}'))


def serial(target, model, record, logits):
    def draft(actual_head, actual_target, fused, *, anchor_pos, root_token):
        scope = (known_visual_prefix(model, record['prompt_len'], record['evidence_request'])
                 if model.evidence_mode == 'visual' and anchor_pos == record['prompt_len'] - 1
                 else packed.nullcontext())
        with scope:
            result = adapters.draft(actual_head, actual_target, fused,
                                    anchor_pos=anchor_pos, root_token=root_token)
        logits.append(result.detach().clone())
        return result
    scope = (evidence.bound_request(model, record['evidence_request'])
             if model.evidence_mode != 'none' else packed.nullcontext())
    with scope:
        yield from runtime.score(target, model, record, draft_fn=draft, e2e_tv_weight=0.)


def compare_gradients(a, b):
    maximum = 0.
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other and (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=8e-7, rtol=5e-5, msg=name)
            maximum = max(maximum, float((p.grad - q.grad).abs().max()))
    return maximum


@pytest.mark.parametrize('mode', ['none', 'text', 'visual'])
@pytest.mark.parametrize('pack_size', [1, 2, 4])
def test_per_anchor_logits_pool_all_gradients_and_one_master_update(mode, pack_size):
    a = head(mode); b = copy.deepcopy(a); target = Target().eval()
    records = [trace(), trace(29, uncovered=True)]
    denominator = sum(runtime.denominator(row) for row in records)
    oa = MasterWeightAdamW(a, lr=1e-5, betas=(.9, .95), weight_decay=0., clip=1., where='cpu')
    ob = MasterWeightAdamW(b, lr=1e-5, betas=(.9, .95), weight_decay=0., clip=1., where='cpu')
    calls = []
    hook = b.register_forward_hook(lambda *args: calls.append(1))
    total_a = total_b = 0.
    for original in records:
        reference, actual = copy.deepcopy(original), copy.deepcopy(original)
        ref_logits, ref_pools = [], []
        for stat in serial(target, a, reference, ref_logits):
            ref_pools.append((stat.numerator.detach().clone(), stat.denominator.detach().clone()))
            total_a += float(stat.numerator.detach()) / denominator
            (stat.numerator / denominator).backward()
        cursor = 0
        for group in packed.score(target, b, actual, pack_size=pack_size):
            assert len(group.anchors) <= pack_size
            for index, member in enumerate(group.anchor_pools):
                torch.testing.assert_close(group.logits[index], ref_logits[cursor], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(member.numerator, ref_pools[cursor][0], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(member.denominator, ref_pools[cursor][1], atol=0, rtol=0)
                cursor += 1
            total_b += float(group.numerator.detach()) / denominator
            (group.numerator / denominator).backward()
        assert cursor == len(original['plan'])
        assert actual['plan'] == original['plan']
        torch.testing.assert_close(actual['ids'], original['ids'], atol=0, rtol=0)
        assert torch.equal(actual['evidence_request']['prompt_visual_mask'], original['evidence_request']['prompt_visual_mask'])
        if mode != 'none':
            assert b.evidence_readout.request is None
            assert all(key not in b.evidence_readout.__dict__ for key in ('prepare', 'forward', '_visual'))
    hook.remove()
    assert len(calls) == sum((len(row['plan']) + pack_size - 1) // pack_size for row in records)
    assert abs(total_a - total_b) < 2e-6
    compare_gradients(a, b)
    assert all(p.grad is None for p in target.parameters())
    if mode != 'none':
        assert b.evidence_readout.time.weight.grad is None
        for name in ('query', 'key', 'value', 'output'):
            assert getattr(b.evidence_readout, name).weight.grad.abs().sum() > 0
    oa.step(); ob.step()
    for name in oa.master:
        torch.testing.assert_close(oa.master[name], ob.master[name], atol=5e-7, rtol=5e-6, msg=name)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, atol=5e-7, rtol=5e-6)
    for sa, sb in zip(oa._opt.state.values(), ob._opt.state.values()):
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(sa[key], sb[key], atol=2e-7, rtol=1e-4)


@pytest.mark.parametrize('mode', ['none', 'text', 'visual'])
def test_future_context_and_other_block_roots_cannot_change_first_anchor(mode):
    model = head(mode); target = Target().eval(); record = trace()
    anchors = [pos for pos, _ in record['plan'][:4]]
    first = anchors[0]
    assert first == record['prompt_len'] - 1
    changed = copy.deepcopy(record)
    changed['fused'][:, first:] = torch.randn_like(changed['fused'][:, first:]) * 100
    changed['ids'][first + 1:] = 70  # other packed root blocks must also remain invisible
    def call(row):
        scope = evidence.bound_request(model, row['evidence_request']) if mode != 'none' else packed.nullcontext()
        with scope:
            return packed.packed_logits(target, model, row, anchors)
    torch.testing.assert_close(call(record)[0], call(changed)[0], atol=0, rtol=0)
    # An early block's backbone gradient also cannot reach future context rows.
    source = record['fused'].clone().requires_grad_(True)
    noise = torch.randn(1, len(anchors) * model.block_size, model.config.hidden_size)
    scope = evidence.bound_request(model, copy.deepcopy(record['evidence_request'])) if mode != 'none' else packed.nullcontext()
    with scope:
        output = packed.forward_anchors(model, source, noise, anchors, record['prompt_len'])
    output[:, :model.block_size].square().mean().backward()
    assert source.grad[:, :first].abs().sum() > 0
    assert torch.count_nonzero(source.grad[:, first:]) == 0


def test_missing_visual_token_at_boundary_is_rejected_without_mutating_request():
    model = head('visual'); target = Target().eval(); record = trace()
    record['evidence_request']['prompt_visual_mask'] = torch.tensor([False, True, False, False, True])
    before = record['evidence_request']['prompt_visual_mask'].clone()
    with pytest.raises(ValueError, match='every visual position'):
        list(packed.score(target, model, record))
    assert torch.equal(record['evidence_request']['prompt_visual_mask'], before)
    assert model.evidence_readout.request is None
    assert all(key not in model.evidence_readout.__dict__ for key in ('prepare', 'forward', '_visual'))


def test_reject_nonzero_lambda_bad_trace_and_restore_after_forward_failure(monkeypatch):
    model = head('visual'); target = Target().eval(); record = trace()
    with pytest.raises(ValueError, match='e2e-tv-weight 0'):
        list(packed.score(target, model, record, e2e_tv_weight=10))
    invalid = copy.deepcopy(record); invalid['plan'][0] = (3, 2)
    with pytest.raises(ValueError, match='known prompt root'):
        list(packed.score(target, model, invalid))
    def fail(*args, **kwargs):
        raise RuntimeError('injected backbone failure')
    monkeypatch.setattr(model.layers[0], 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        list(packed.score(target, model, record))
    assert model.evidence_readout.request is None
    assert all(key not in model.evidence_readout.__dict__ for key in ('prepare', 'forward', '_visual'))


def test_generator_close_clears_request_and_runtime_only_changes_scoring_protocol():
    model = head('visual'); target = Target().eval(); record = trace()
    generator = packed.score(target, model, record)
    next(generator); generator.close()
    assert model.evidence_readout.request is None
    old_score = lambda *args: 'original'
    old_protocol = lambda *args: {'hashes': {}, 'protocol_sha256': 'old'}
    trainer = SimpleNamespace(score_training_trace=old_score, make_protocol=old_protocol,
                              backward_update=object(), adapters=object(), MasterWeightAdamW=object())
    untouched = (trainer.backward_update, trainer.adapters, trainer.MasterWeightAdamW)
    with pytest.raises(RuntimeError, match='injected'):
        with packed.packed_runtime(trainer):
            result = trainer.make_protocol(SimpleNamespace(e2e_tv_weight=0., loss_normalization='global_pooled'))
            assert result['packed_opd_score']['sft_labels_used'] is False
            assert result['packed_opd_score']['pack_size'] == 4
            assert result['protocol_sha256'] == hashlib.sha256(json.dumps(
                {k: v for k, v in result.items() if k != 'protocol_sha256'}, sort_keys=True).encode()).hexdigest()
            assert (trainer.backward_update, trainer.adapters, trainer.MasterWeightAdamW) == untouched
            with pytest.raises(ValueError, match='global_pooled'):
                trainer.make_protocol(SimpleNamespace(e2e_tv_weight=0., loss_normalization='legacy_per_qa'))
            raise RuntimeError('injected')
    assert trainer.score_training_trace is old_score and trainer.make_protocol is old_protocol


@pytest.mark.parametrize('mode', ['none', 'visual'])
def test_existing_backward_update_keeps_global_denominator_and_qa_accumulation(mode):
    from video_caption import train_video_9k as trainer
    a = head(mode); b = copy.deepcopy(a); target = Target().eval()
    records = [trace(), trace(31, uncovered=True)]
    before_rng = torch.get_rng_state().clone()
    reference = trainer.backward_update(iter(copy.deepcopy(records)),
        lambda row: serial(target, a, row, []), 'global_pooled')
    actual = trainer.backward_update(iter(copy.deepcopy(records)),
        lambda row: packed.score(target, b, row), 'global_pooled')
    assert torch.equal(before_rng, torch.get_rng_state())
    assert reference['denominator'] == actual['denominator'] == sum(runtime.denominator(row) for row in records)
    assert reference['qa_denominators'] == actual['qa_denominators']
    assert abs(reference['loss'] - actual['loss']) < 2e-6
    assert reference['loss_normalization'] == actual['loss_normalization'] == 'global_pooled'
    compare_gradients(a, b)


@pytest.mark.parametrize('mode', ['none', 'text', 'visual'])
def test_cpu_bf16_gradients_are_close_but_not_claimed_update_equivalent(mode):
    """Quantify mixed-precision grouping; near-zero gradients can change Adam sign."""
    a = head(mode)
    for name, parameter in a.named_parameters():
        if not name.startswith('evidence_readout.'):
            parameter.data = parameter.data.to(torch.bfloat16)
    b = copy.deepcopy(a); target = Target().eval(); record = trace()
    denominator = runtime.denominator(record)
    reference_logits, actual_logits = [], []
    for stat in serial(target, a, copy.deepcopy(record), reference_logits):
        (stat.numerator / denominator).backward()
    for stat in packed.score(target, b, copy.deepcopy(record)):
        actual_logits.extend(value.detach() for value in stat.logits)
        (stat.numerator / denominator).backward()
    delta = scale = 0.
    for p, q in zip(a.parameters(), b.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
            delta += float((p.grad.float() - q.grad.float()).square().sum())
            scale += float(p.grad.float().square().sum())
    assert (delta / scale) ** .5 < .01  # CPU fixture bound, not a CUDA or exact-update claim.
    for actual, reference in zip(actual_logits, reference_logits):
        torch.testing.assert_close(actual, reference, atol=.005, rtol=.005)
