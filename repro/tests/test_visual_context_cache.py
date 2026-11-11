"""CPU regression: full-prefix scoring and incremental xattn see the same context."""
import sys
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, Qwen3Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dflash_pkg/dflash-0.1.0"))
from dflash.model import DFlashDraftModel


def make_head():
    torch.manual_seed(42)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                     vocab_size=100, num_target_layers=6, block_size=4,
                     dflash_config={"use_xattn": True, "target_layer_ids": [1, 3],
                                    "mask_token_id": 99})
    cfg._attn_implementation = "sdpa"
    head = DFlashDraftModel(cfg).eval()
    # A zero residual would conceal the original bug.
    torch.nn.init.normal_(head.xattn_out.weight, std=0.1)
    return head


def call(head, context, noise, cache):
    start = cache.get_seq_length()
    return head(target_hidden=context, noise_embedding=noise,
                position_ids=torch.arange(start, start + context.shape[1] + noise.shape[1])[None],
                past_key_values=cache, use_cache=True, is_causal=False)


def test_incremental_and_cropped_context_matches_full_prefix():
    head = make_head()
    context = torch.randn(1, 9, 64)
    noise = torch.randn(1, 4, 32)
    with torch.no_grad():
        cache = DynamicCache()
        call(head, context[:, :5], noise, cache)
        cache.crop(5)
        actual = call(head, context[:, 5:8], noise, cache)
        expected = call(head, context[:, :8], noise, DynamicCache())
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        # A speculative branch can roll back; stale context must be discarded.
        cache.crop(6)
        replacement = torch.randn(1, 2, 64)
        actual = call(head, replacement, noise, cache)
        expected = call(head, torch.cat((context[:, :6], replacement), 1), noise, DynamicCache())
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_extension_with_no_new_hidden_reuses_committed_context():
    head = make_head()
    context = torch.randn(1, 5, 64)
    noise = torch.randn(1, 4, 32)
    with torch.no_grad():
        cache = DynamicCache()
        call(head, context, noise, cache)
        cache.crop(5)
        actual = call(head, context[:, :0], noise, cache)
        expected = call(head, context, noise, DynamicCache())
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_new_request_does_not_inherit_prior_visual_context():
    head = make_head()
    noise = torch.randn(1, 4, 32)
    with torch.no_grad():
        call(head, torch.randn(1, 8, 64), noise, DynamicCache())
        context = torch.randn(1, 3, 64)
        actual = call(head, context, noise, DynamicCache())
        expected = call(head, context, noise, DynamicCache())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_missing_context_cache_and_wrong_positions_fail():
    head = make_head()
    context = torch.randn(1, 3, 64)
    cache = DynamicCache()
    with torch.no_grad():
        call(head, context, torch.randn(1, 4, 32), cache)
    cache.crop(3)
    del cache._glance_xattn_context
    with pytest.raises(ValueError, match="populated context cache"):
        head._xattn_context(context[:, :1], torch.arange(3, 8)[None], cache, True)
    with pytest.raises(ValueError, match="positions disagree"):
        head._xattn_context(context, torch.arange(1, 8)[None], DynamicCache(), True)


def test_full_prefix_keeps_autograd_for_xattn_parameters():
    head = make_head()
    out = call(head, torch.randn(1, 5, 64), torch.randn(1, 4, 32), DynamicCache())
    out.square().mean().backward()
    assert head.xattn.in_proj_weight.grad is not None
    assert torch.isfinite(head.xattn.in_proj_weight.grad).all()
    assert head.xattn.in_proj_weight.grad.abs().sum() > 0
