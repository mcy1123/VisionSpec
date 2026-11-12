"""CPU proof for nonzero packed evidence forward, coverage CE and gradients."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import configure
configure()
from local_h100 import evidence_sft as local
from local_h100 import train_video_evidence_sft as entry
from local_h100 import train_video_sft as base
from local_h100 import train_video_evidence_opd as opd_entry
from shared import common, sft_blocks
from shared import training_resume
from test_evidence_readout import make_head, request


def head(mode, nonzero=True):
    value = make_head(mode, nonzero=nonzero)
    value.config.dflash_config['evidence_time_encoding'] = 'off'
    with torch.no_grad():
        value.evidence_readout.time.weight.zero_()
    local.freeze_time_after_unfreeze(value)
    return value


def data():
    torch.manual_seed(194)
    return torch.randn(1, 14, 64), torch.randn(1, 12, 32), [5, 8, 10]


def packed(model, ctx, noise, anchors):
    positions = (torch.tensor(anchors)[:, None] + torch.arange(4)).reshape(1, -1)
    with local.bound_request(model, request()):
        return local.forward_anchors(model, ctx, noise, torch.arange(ctx.shape[1])[None], positions,
                                     sft_blocks.build_anchor_mask(anchors, 4, ctx.shape[1], 'cpu', ctx.dtype))


def serial(model, ctx, noise, anchors):
    with local.bound_request(model, request()):
        return torch.cat([model(target_hidden=ctx[:, :a], noise_embedding=noise[:, i * 4:(i + 1) * 4],
                                position_ids=torch.arange(a + 4)[None], use_cache=False, is_causal=False)
                          for i, a in enumerate(anchors)], dim=1)


def compare_gradients(left, right):
    maximum = 0.0
    for (name, p), (other_name, q) in zip(left.named_parameters(), right.named_parameters()):
        assert name == other_name
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-5, msg=name)
            maximum = max(maximum, float((p.grad - q.grad).abs().max()))
    for name in ('query', 'key', 'value', 'output'):
        assert getattr(left.evidence_readout, name).weight.grad.abs().sum() > 0
    assert left.evidence_readout.time.weight.grad is None
    return maximum


@pytest.mark.parametrize('mode', ['text', 'visual'])
def test_actual_packed_head_matches_serial_nonzero_forward_all_gradients(mode):
    a = head(mode); b = copy.deepcopy(a)
    ctx, noise, anchors = data()
    output, reference = packed(a, ctx, noise, anchors), serial(b, ctx, noise, anchors)
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
    weights = torch.linspace(0.3, 1.3, output.numel()).reshape_as(output)
    (output * weights).square().mean().backward()
    (reference * weights).square().mean().backward()
    compare_gradients(a, b)
    assert a.evidence_readout.request is None
    assert 'prepare' not in a.evidence_readout.__dict__
    assert 'forward' not in a.evidence_readout.__dict__


class Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(100, 32)
        self.lm_head = torch.nn.Linear(32, 100, bias=False)
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.emb


@pytest.mark.parametrize('mode', ['text', 'visual'])
def test_baseline_variable_reveal_ce_and_all_gradients_match_serial(mode):
    a = head(mode); b = copy.deepcopy(a)
    ctx, _, anchors = data()
    target = Target()
    full, reveals = torch.arange(14), torch.tensor([1, 2, 3])
    actual, stats = local.hidden_loss(base.hidden_loss, a, target, ctx, full, anchors, reveals, 99, request())
    blocks, labels, supervised, weights = sft_blocks.build_blocks(full, anchors, reveals, 4, 99, 'cpu')
    noise = target.emb(blocks).reshape(1, -1, 32)
    output = serial(b, ctx, noise, anchors)
    keep = supervised.reshape(-1).bool()
    logits = target.lm_head(output[0][keep])
    nll = F.cross_entropy(logits.float(), labels.reshape(-1)[keep], reduction='none')
    w = weights.reshape(-1)[keep]
    reference = (nll * w).sum() / w.sum().clamp_min(1.0)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    assert stats['positions'] == int(keep.sum())
    actual.backward(); reference.backward()
    compare_gradients(a, b)
    assert all(p.grad is None for p in target.parameters())


@pytest.mark.parametrize('mode', ['text', 'visual'])
def test_future_prefix_cannot_change_early_anchor(mode):
    model = head(mode)
    ctx, noise, anchors = data()
    changed = ctx.clone(); changed[:, anchors[0]:] = torch.randn_like(changed[:, anchors[0]:]) * 300
    a, b = packed(model, ctx, noise, anchors), packed(model, changed, noise, anchors)
    torch.testing.assert_close(a[:, :4], b[:, :4], atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['text', 'visual'])
def test_zero_residual_equals_plain_baseline_sft(mode):
    model, plain = head(mode, nonzero=False), make_head('none')
    plain.load_state_dict({n: t for n, t in model.state_dict().items() if not n.startswith('evidence_readout.')})
    ctx, noise, anchors = data()
    positions = (torch.tensor(anchors)[:, None] + torch.arange(4)).reshape(1, -1)
    expected = sft_blocks.forward_anchors(plain, ctx, noise, torch.arange(14)[None], positions,
                                         sft_blocks.build_anchor_mask(anchors, 4, 14, 'cpu', ctx.dtype))
    torch.testing.assert_close(packed(model, ctx, noise, anchors), expected, atol=2e-6, rtol=2e-5)


def test_freeze_occurs_after_global_unfreeze_and_never_silently_zeroes():
    model = head('text'); model.requires_grad_(True)
    with pytest.raises(ValueError, match='unfrozen'):
        local.assert_time_off(model)
    local.freeze_time_after_unfreeze(model)
    assert not model.evidence_readout.time.weight.requires_grad
    model.evidence_readout.time.weight.data[0, 0] = 1
    with pytest.raises(ValueError, match='nonzero'):
        local.freeze_time_after_unfreeze(model)


@pytest.mark.parametrize('mode', ['text', 'visual'])
def test_mixed_dtype_save_load_keeps_evidence_exact_fp32(tmp_path, mode):
    model = head(mode)
    for name, param in model.named_parameters():
        if not name.startswith('evidence_readout.'):
            param.data = param.data.to(torch.bfloat16)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith('evidence_readout.')}
    model.save_pretrained(tmp_path)
    loaded = common.load_head(tmp_path, dtype=torch.bfloat16, device='cpu')
    loaded.requires_grad_(True)
    local.freeze_time_after_unfreeze(loaded)
    for name, value in before.items():
        actual = loaded.state_dict()[name]
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, value, atol=0, rtol=0)


def test_request_and_instance_substitutions_clear_on_failure():
    model = head('text')
    ctx, noise, anchors = data()
    original_forward = model.forward
    def fail(**kwargs):
        raise RuntimeError('injected forward failure')
    model.forward = fail
    with pytest.raises(RuntimeError, match='injected'):
        packed(model, ctx, noise, anchors)
    assert model.evidence_readout.request is None
    assert 'prepare' not in model.evidence_readout.__dict__
    assert 'forward' not in model.evidence_readout.__dict__
    model.forward = original_forward
    assert torch.isfinite(packed(model, ctx, noise, anchors)).all()


def test_wrapper_consumes_one_bound_request_and_restores_every_substitution(monkeypatch):
    from qa_opd_fair import videoencode
    model, target = head('text'), Target()
    ctx, _, anchors = data()
    monkeypatch.setattr(videoencode, 'encode_inputs', lambda *args, **kwargs: {'encoded': 'same'})
    monkeypatch.setattr(entry, 'build_evidence_request', lambda inputs, row, proc: request(row['id']))
    before = (base.validate_head, base.hidden_loss, base.MasterWeightAdamW, base.build_protocol,
              base.checkpoint, videoencode.encode_inputs, sft_blocks.forward_anchors)
    with pytest.raises(RuntimeError, match='exit test'):
        with entry.evidence_runtime():
            with pytest.raises(RuntimeError, match='no encoded-input'):
                base.hidden_loss(model, target, ctx, torch.arange(14), anchors, torch.ones(3, dtype=torch.long), 99)
            for identity in ('qa-a', 'qa-b'):
                videoencode.encode_inputs(None, {'id': identity}, 'real')
                loss, _ = base.hidden_loss(model, target, ctx, torch.arange(14), anchors, torch.ones(3, dtype=torch.long), 99)
                assert torch.isfinite(loss) and model.evidence_readout.request is None
            raise RuntimeError('exit test')
    after = (base.validate_head, base.hidden_loss, base.MasterWeightAdamW, base.build_protocol,
             base.checkpoint, videoencode.encode_inputs, sft_blocks.forward_anchors)
    assert before == after


def test_plain_guard_still_refuses_evidence_head(tmp_path):
    model = head('text'); model.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match='plain SFT'):
        base.validate_head(tmp_path)
    # Production config requires last retained layer 33; tiny geometry is safe
    # for CPU tests but must not bypass the real entry's separate check.
    with pytest.raises(ValueError, match='33'):
        entry.validate_head(tmp_path)


def test_opd_same_time_policy_request_cleanup_and_original_math_restored(monkeypatch):
    model = head('visual')
    model.requires_grad_(True)
    def optimizer(value, **kwargs):
        local.assert_time_off(value)
        return 'original optimizer'
    def collect(target, value, inputs, **kwargs):
        assert value.evidence_readout.request['identity'] == inputs
        return {'original': inputs}
    def score(target, value, trace, **kwargs):
        value.set_evidence_request(request(trace['original']))
        yield 'original scoring graph'
        if kwargs.get('fail'):
            raise RuntimeError('scoring failed')
    trainer = SimpleNamespace(
        validate_training_head=lambda *args: 'validated', MasterWeightAdamW=optimizer,
        make_protocol=lambda *args: {}, adapters=SimpleNamespace(collect=collect),
        score_training_trace=score, evidence_gradient_diagnostics=lambda value: {'original': True})
    original_checkpoint = lambda *args, **kwargs: 'saved'
    monkeypatch.setattr(training_resume, 'save_atomic_checkpoint', original_checkpoint)
    before = (trainer.validate_training_head, trainer.MasterWeightAdamW, trainer.make_protocol,
              trainer.adapters.collect, trainer.score_training_trace, trainer.evidence_gradient_diagnostics)
    with opd_entry.evidence_runtime(trainer):
        assert trainer.MasterWeightAdamW(model) == 'original optimizer'
        model.set_evidence_request(request('first'))
        trace = trainer.adapters.collect(None, model, 'first')
        assert trace == {'original': 'first'} and model.evidence_readout.request is None
        assert list(trainer.score_training_trace(None, model, trace)) == ['original scoring graph']
        assert model.evidence_readout.request is None
        with pytest.raises(RuntimeError, match='scoring failed'):
            list(trainer.score_training_trace(None, model, trace, fail=True))
        assert model.evidence_readout.request is None
        assert trainer.evidence_gradient_diagnostics(model) == {'original': True}
        assert training_resume.save_atomic_checkpoint('unused', model) == 'saved'
    after = (trainer.validate_training_head, trainer.MasterWeightAdamW, trainer.make_protocol,
             trainer.adapters.collect, trainer.score_training_trace, trainer.evidence_gradient_diagnostics)
    assert before == after
    assert training_resume.save_atomic_checkpoint is original_checkpoint


def test_frozen_time_stays_exact_zero_through_real_master_adam_step():
    from shared.master_optim import MasterWeightAdamW
    model = head('text')
    ctx, noise, anchors = data()
    optimizer = MasterWeightAdamW(model, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, where='cpu')
    before = model.evidence_readout.output.weight.detach().clone()
    packed(model, ctx, noise, anchors).square().mean().backward()
    optimizer.step()
    local.assert_time_off(model)
    assert optimizer.master['evidence_readout.time.weight'].grad is None
    assert not torch.equal(before, model.evidence_readout.output.weight)
