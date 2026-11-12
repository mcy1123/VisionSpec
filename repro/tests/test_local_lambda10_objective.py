"""Check the historical lambda10 proxy on actual A/B/C draft forwards."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_local_opd_long import head, evidence_scope, entry, Target, trace
from opd_official_kl import runtime


def score(model, target, record, weight, captures):
    original = entry.trainer.adapters.draft
    def captured(*args, **kwargs):
        logits = original(*args, **kwargs)
        captures.append(logits)
        return logits
    with evidence_scope(model.evidence_mode), patch.object(entry.trainer.adapters, 'draft', captured):
        return list(entry.trainer.score_training_trace(target, model, record,
            e2e_tv_weight=weight))


@pytest.mark.parametrize('mode', ['none', 'text', 'visual'])
@pytest.mark.parametrize('uncovered', [False, True])
def test_lambda10_is_exact_full_vocab_top8_prefix_addition(mode, uncovered):
    model = head(mode)
    reference = copy.deepcopy(model)
    target = Target().eval().requires_grad_(False)
    record = trace(19, uncovered=uncovered)
    actual_logits, reference_logits = [], []
    actual = score(model, target, copy.deepcopy(record), 10., actual_logits)
    base = score(reference, target, copy.deepcopy(record), 0., reference_logits)
    assert len(actual) == len(base) == len(record['plan'])
    expected_numerators = []
    for (pos,count), got, original, logits, other in zip(
            record['plan'],actual,base,actual_logits,reference_logits):
        torch.testing.assert_close(logits, other, atol=0, rtol=0)
        if count:
            # Independent formula: full-vocab normalization before top8;
            # truncate the support without renormalizing its probabilities.
            q = logits.float().softmax(-1)[:count]
            values, indices = q.topk(min(8, q.shape[-1]), dim=-1)
            p = target.lm_head(record['hidden'][0,pos:pos+count]).float().softmax(-1)
            overlap = torch.minimum(values,p.gather(-1,indices)).sum(-1)
            proxy = 1. - overlap.cumprod(0).mean()
            expected = original.numerator + 10. * count * proxy
        else:
            expected = original.numerator
        torch.testing.assert_close(got.numerator,expected,atol=3e-6,rtol=2e-6)
        torch.testing.assert_close(got.denominator,original.denominator + 10. * count,
                                   atol=0,rtol=0)
        expected_numerators.append(expected)
    # The production global_pooled contract uses the original base divisor,
    # keeping this proxy additive; Pool.denominator is not the divisor.
    denominator = runtime.denominator(record)
    loss = sum(pool.numerator for pool in actual) / denominator
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.ne(0).any()) for p in model.parameters())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert all(p.grad is None for p in target.parameters())
    if mode != 'none':
        assert model.evidence_readout.request is None
        assert model.evidence_readout.time.weight.grad is None
