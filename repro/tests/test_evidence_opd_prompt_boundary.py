"""CPU real-head/cache and real OPD scoring regressions for prompt_len-1."""
import copy
from pathlib import Path
import sys

import pytest
import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import configure
configure()
from local_h100.evidence_opd_prompt_boundary import known_visual_prefix, prompt_boundary_runtime
from local_h100.train_video_evidence_opd import evidence_runtime
from video_caption import train_video_9k as trainer
from opd_official_kl.plan import response_plan
from test_local_evidence_sft import head, Target
from test_evidence_readout import request


def actual_call(model, source, noise, cache=None):
    cache = DynamicCache() if cache is None else cache
    prefix = cache.get_seq_length()
    output = model(target_hidden=source, noise_embedding=noise,
                   position_ids=torch.arange(prefix, prefix + source.shape[1] + noise.shape[1])[None],
                   past_key_values=cache, use_cache=True, is_causal=False)
    return output, cache


def test_response_plan_first_anchor_is_last_prompt_token():
    plan = response_plan(5, 8, [2, 5, 7], 4)
    assert plan[0][0] == 4
    assert sum(count for _, count in plan) == 8


def test_shared_guard_remains_active_outside_local_boundary():
    model = head('visual'); req = request(); model.set_evidence_request(req)
    with pytest.raises(ValueError, match='complete known prompt'):
        actual_call(model, torch.randn(1, 4, 64), torch.randn(1, 4, 32))


def test_nonzero_boundary_forward_and_all_gradients_match_full_known_visual_oracle():
    model = head('visual'); oracle = copy.deepcopy(model)
    req, ref_req = request(), request()
    model.set_evidence_request(req); oracle.set_evidence_request(ref_req)
    torch.manual_seed(11)
    source, noise = torch.randn(1, 5, 64), torch.randn(1, 4, 32)
    mask_object, mask_before = req['prompt_visual_mask'], req['prompt_visual_mask'].clone()
    with known_visual_prefix(model, 5, req):
        actual, cache = actual_call(model, source[:, :4], noise)
    assert cache._glance_evidence_context['end'] == 4
    assert req['prompt_visual_mask'] is mask_object
    assert torch.equal(req['prompt_visual_mask'], mask_before)
    assert req['identity'] == 'video-a'
    assert '_visual' not in model.evidence_readout.__dict__
    # Oracle retains the identical truncated backbone, but pools visual memory
    # from the complete known prompt. Its final text row is not selected.
    original = oracle.evidence_readout._visual
    oracle.evidence_readout._visual = lambda partial, prefix: original(source[..., -32:], 0)
    try:
        expected, _ = actual_call(oracle, source[:, :4], noise)
    finally:
        del oracle.evidence_readout._visual
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(req['prepared_visual']['features'], ref_req['prepared_visual']['features'], atol=0, rtol=0)
    actual.square().mean().backward(); expected.square().mean().backward()
    for (name, a), (_, b) in zip(model.named_parameters(), oracle.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0, msg=name)
    for name in ('query', 'key', 'value', 'output'):
        assert getattr(model.evidence_readout, name).weight.grad.abs().sum() > 0


def test_incremental_cache_after_boundary_uses_original_guard_and_memory():
    model = head('visual'); req = request(); model.set_evidence_request(req)
    source, noise = torch.randn(1, 6, 64), torch.randn(1, 4, 32)
    with torch.no_grad(), known_visual_prefix(model, 5, req):
        _, cache = actual_call(model, source[:, :4], noise)
    cache.crop(4)
    with torch.no_grad():
        incremental, cache = actual_call(model, source[:, 4:6], noise, cache)
        reference, _ = actual_call(model, source[:, :6], noise)
    torch.testing.assert_close(incremental, reference, atol=2e-6, rtol=2e-5)
    assert cache._glance_evidence_context['end'] == 6


def test_unseen_visual_position_or_other_prefix_is_rejected():
    model = head('visual'); req = request()
    req['prompt_visual_mask'] = torch.tensor([False, True, False, False, True])
    model.set_evidence_request(req)
    with pytest.raises(ValueError, match='every visual position'):
        with known_visual_prefix(model, 5, req): pass
    req = request(); model.set_evidence_request(req)
    with pytest.raises(ValueError, match='exactly prompt_len-1'):
        with known_visual_prefix(model, 5, req):
            actual_call(model, torch.randn(1, 3, 64), torch.randn(1, 4, 32))
    assert '_visual' not in model.evidence_readout.__dict__


def trace():
    return dict(prompt_len=5, ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                fused=torch.randn(1, 8, 64), hidden=torch.randn(1, 8, 32),
                plan=response_plan(5, 3, [2], 4), rounds=[], evidence_request=request())


@pytest.mark.parametrize('mode', ['text', 'visual'])
def test_real_shared_opd_score_backward_and_all_wrappers_restore(mode):
    model, target = head(mode), Target()
    model.config.dflash_config['target_layer_ids'] = [1, 33]
    record = trace()
    originals = (trainer.score_training_trace, trainer.adapters.draft, trainer.make_protocol)
    with evidence_runtime(trainer), prompt_boundary_runtime(trainer):
        losses = list(trainer.score_training_trace(target, model, record, e2e_tv_weight=0.))
        assert len(losses) == 1
        losses[0].numerator.backward()
        for name in ('query', 'key', 'value', 'output'):
            assert getattr(model.evidence_readout, name).weight.grad.abs().sum() > 0
        assert model.evidence_readout.time.weight.grad is None
        assert model.evidence_readout.request is None
        assert all(p.grad is None for p in target.parameters())
    assert (trainer.score_training_trace, trainer.adapters.draft, trainer.make_protocol) == originals
    assert '_visual' not in model.evidence_readout.__dict__


def test_text_scoring_unchanged_exactly():
    a = head('text'); b = copy.deepcopy(a); target = Target(); record = trace()
    for model in (a, b): model.config.dflash_config['target_layer_ids'] = [1, 33]
    with evidence_runtime(trainer):
        before = list(trainer.score_training_trace(target, a, copy.deepcopy(record), e2e_tv_weight=0.))[0]
    with evidence_runtime(trainer), prompt_boundary_runtime(trainer):
        after = list(trainer.score_training_trace(target, b, copy.deepcopy(record), e2e_tv_weight=0.))[0]
    torch.testing.assert_close(before.numerator, after.numerator, atol=0, rtol=0)
    before.numerator.backward(); after.numerator.backward()
    for x, y in zip(a.parameters(), b.parameters()):
        if x.grad is not None: torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
