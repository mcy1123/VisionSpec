"""CPU tests of evidence identity, causal context, gradients, and persistence."""
import copy
import sys
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, Qwen3Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dflash_pkg/dflash-0.1.0"))
from dflash.model import DFlashDraftModel


def make_head(mode="visual", nonzero=False):
    torch.manual_seed(42)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                     vocab_size=100, num_target_layers=6,
                     dflash_config={"target_layer_ids": [1, 3], "block_size": 4,
                                    "mask_token_id": 99, "evidence_mode": mode,
                                    "evidence_dim": 8, "evidence_max_tokens": 8,
                                    "evidence_spatial_bins": 1})
    cfg._attn_implementation = "sdpa"
    head = DFlashDraftModel(cfg).eval()
    if mode != "none" and nonzero:
        torch.nn.init.normal_(head.evidence_readout.output.weight, std=0.1)
    return head


def request(identity="video-a", intervention="normal"):
    return {"identity": identity, "prompt_visual_mask": torch.tensor([False, True, True, False, False]),
            "grid_thw": [2, 2, 2], "spatial_merge_size": 2,
            "time_values": torch.tensor([0.1, 0.8]), "intervention": intervention, "seed": 7}


def call(head, context, noise, cache=None):
    cache = DynamicCache() if cache is None else cache
    start = cache.get_seq_length()
    return head(target_hidden=context, noise_embedding=noise,
                position_ids=torch.arange(start, start + context.shape[1] + noise.shape[1])[None],
                past_key_values=cache, use_cache=True, is_causal=False)


@pytest.mark.parametrize("mode", ["text", "visual"])
def test_zero_output_preserves_original_head_exactly(mode):
    baseline, head = make_head("none"), make_head(mode)
    missing, extra = head.load_state_dict(baseline.state_dict(), strict=False)
    assert missing and all(key.startswith("evidence_readout.") for key in missing)
    assert not extra
    head.set_evidence_request(request())
    context, noise = torch.randn(1, 5, 64), torch.randn(1, 4, 32)
    with torch.no_grad():
        torch.testing.assert_close(call(head, context, noise), call(baseline, context, noise), atol=0, rtol=0)
    assert head.evidence_diagnostics()["gate"] == 0.5


@pytest.mark.parametrize("mode", ["text", "visual"])
def test_nonzero_incremental_rollback_and_empty_extension_equal_full_prefix(mode):
    head = make_head(mode, nonzero=True)
    head.set_evidence_request(request())
    context, noise = torch.randn(1, 9, 64), torch.randn(1, 4, 32)
    with torch.no_grad():
        cache = DynamicCache()
        call(head, context[:, :5], noise, cache)
        cache.crop(5)
        actual = call(head, context[:, 5:8], noise, cache)
        expected = call(head, context[:, :8], noise)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        cache.crop(6)
        replacement = torch.randn(1, 2, 64)
        actual = call(head, replacement, noise, cache)
        full = torch.cat((context[:, :6], replacement), dim=1)
        torch.testing.assert_close(actual, call(head, full, noise), atol=2e-6, rtol=2e-5)
        cache.crop(8)
        torch.testing.assert_close(call(head, context[:, :0], noise, cache),
                                   call(head, full, noise), atol=2e-6, rtol=2e-5)


def test_text_memory_contains_only_committed_non_visual_last_layer():
    head = make_head("text", nonzero=True)
    head.set_evidence_request(request())
    context, noise, cache = torch.randn(1, 7, 64), torch.randn(1, 4, 32), DynamicCache()
    with torch.no_grad():
        call(head, context[:, :5], noise, cache)
    state = cache._glance_evidence_context
    assert state["positions"].tolist() == [0, 3, 4]
    torch.testing.assert_close(state["features"], context[0, [0, 3, 4], -32:])
    assert not state["features"].requires_grad
    # Keeping speculative/noise KV instead of committed KV must fail, not leak.
    with pytest.raises(ValueError, match="draft noise"):
        call(head, context[:, 5:6], noise, cache)


def test_new_request_rejects_old_cache_and_prepared_identity():
    head = make_head("visual", nonzero=True)
    first, second = request(), request("video-b")
    head.set_evidence_request(first)
    context, noise, cache = torch.randn(1, 5, 64), torch.randn(1, 4, 32), DynamicCache()
    with torch.no_grad():
        call(head, context, noise, cache)
    cache.crop(5)
    head.set_evidence_request(second)
    with pytest.raises(ValueError, match="own populated request cache"):
        call(head, context[:, :1], noise, cache)
    second["prepared_visual"] = first["prepared_visual"]
    with pytest.raises(ValueError, match="identity/config mismatch"):
        call(head, context, noise)
    del second["prepared_visual"]
    with torch.no_grad():
        new = call(head, context * 2, noise)
        second.pop("prepared_visual")
        head.set_evidence_request(second)
        torch.testing.assert_close(new, call(head, context * 2, noise), atol=0, rtol=0)


def test_gradient_opens_output_first_then_query_key_value_and_keeps_source_frozen():
    head = make_head("visual")
    head.set_evidence_request(request())
    context = torch.randn(1, 5, 64, requires_grad=True)
    noise = torch.randn(1, 4, 32)
    # Exercise just the readout to isolate its frozen-memory contract from the
    # normal head path, which is allowed to consume target features as before.
    module = head.evidence_readout
    prepared = module.prepare(context[..., -32:], torch.arange(9)[None], DynamicCache(), False)
    module(noise, prepared).square().mean().backward()
    assert module.output.weight.grad.abs().sum() > 0
    for layer in (module.query, module.key, module.value):
        assert layer.weight.grad.abs().sum() == 0
    assert context.grad is None
    with torch.no_grad():
        module.output.weight.add_(-0.1 * module.output.weight.grad)
    module.zero_grad(set_to_none=True)
    module(noise, prepared).square().mean().backward()
    for layer in (module.query, module.key, module.value):
        assert layer.weight.grad.abs().sum() > 0


def test_no_grad_visual_kv_cache_invalidated_by_parameter_update_and_training():
    head = make_head("visual", nonzero=True)
    head.set_evidence_request(request())
    context, noise, cache = torch.randn(1, 5, 64), torch.randn(1, 4, 32), DynamicCache()
    with torch.no_grad():
        call(head, context, noise, cache)
        cache.crop(5)
        call(head, context[:, :0], noise, cache)
        assert head.evidence_diagnostics()["kv_cache_hits"] == 1
        head.evidence_readout.key.weight.add_(0.1)
        cache.crop(5)
        changed = call(head, context[:, :0], noise, cache)
        assert head.evidence_diagnostics()["kv_recomputes"] == 2
        torch.testing.assert_close(changed, call(head, context, noise), atol=2e-6, rtol=2e-5)
    cache.crop(5)
    before = head.evidence_diagnostics()["kv_recomputes"]
    call(head, context[:, :0], noise, cache).square().mean().backward()
    assert head.evidence_diagnostics()["kv_recomputes"] == before + 1
    assert head.evidence_readout.key.weight.grad.abs().sum() > 0


def test_time_shuffle_reassigns_times_without_reordering_content_and_bypass_exact():
    head = make_head("visual", nonzero=True)
    context, noise = torch.randn(1, 5, 64), torch.randn(1, 4, 32)
    normal = request()
    head.set_evidence_request(normal)
    a = head.evidence_readout.prepare(context[..., -32:], torch.arange(9)[None], DynamicCache(), False)
    shuffled = copy.deepcopy(normal)
    shuffled["intervention"] = "time_shuffle"
    head.set_evidence_request(shuffled)
    b = head.evidence_readout.prepare(context[..., -32:], torch.arange(9)[None], DynamicCache(), False)
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    assert not torch.equal(a[1], b[1])
    assert sorted(a[1].tolist()) == sorted(b[1].tolist())
    with torch.no_grad():
        out_shuffled = head.evidence_readout(noise, b)
        head.set_evidence_request(normal)
        assert not torch.equal(head.evidence_readout(noise, a), out_shuffled)
    bypass = copy.deepcopy(normal)
    bypass["intervention"] = "bypass"
    head.set_evidence_request(bypass)
    torch.testing.assert_close(head.evidence_readout(noise, a), noise, atol=0, rtol=0)


def test_fp32_evidence_save_reload_is_exact_even_with_bf16_head(tmp_path):
    head = make_head("visual", nonzero=True)
    # A value deliberately not representable in bf16 catches load-then-float.
    with torch.no_grad():
        head.evidence_readout.query.weight.add_(0.0001234567)
    expected = {k: v.detach().clone() for k, v in head.evidence_readout.state_dict().items()}
    head.save_pretrained(tmp_path)
    loaded = DFlashDraftModel.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
    assert loaded.fc.weight.dtype == torch.bfloat16
    for key, value in loaded.evidence_readout.state_dict().items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    assert loaded.evidence_readout.request is None
    loaded.set_evidence_request(request())
    with torch.inference_mode():
        result = call(loaded, torch.randn(1, 5, 64).bfloat16(),
                      torch.randn(1, 4, 32).bfloat16())
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()


def test_missing_request_and_bad_metadata_fail():
    head = make_head()
    with pytest.raises(ValueError, match="set_evidence_request"):
        call(head, torch.randn(1, 5, 64), torch.randn(1, 4, 32))
    bad = request()
    bad["time_values"] = torch.tensor([0.8, 0.1])
    with pytest.raises(ValueError, match="increase strictly"):
        head.set_evidence_request(bad)
    bad = request()
    bad["grid_thw"] = [3, 2, 2]
    with pytest.raises(ValueError, match="mask count"):
        head.set_evidence_request(bad)


def test_inference_mode_rollout_memory_can_be_reused_for_gradient_scoring():
    head = make_head("visual", nonzero=True)
    req = request()
    head.set_evidence_request(req)
    context, noise, cache = torch.randn(1, 5, 64), torch.randn(1, 4, 32), DynamicCache()
    with torch.inference_mode():
        call(head, context, noise, cache)
        cache.crop(5)
        call(head, context[:, :0], noise, cache)
        assert head.evidence_diagnostics()["kv_cache_hits"] == 1
    assert not req["prepared_visual"]["features"].is_inference()
    head.set_evidence_request(req)
    call(head, context, noise).square().mean().backward()
    assert head.evidence_readout.key.weight.grad.abs().sum() > 0


def test_spatial_temporal_pooling_preserves_time_and_budget():
    from dflash.evidence import EvidenceReadout
    module = EvidenceReadout(4, dim=3, max_tokens=8, spatial_bins=2)
    # Four temporal groups of 4x4 merged patches -> 2 temporal x 2x2 memory.
    source = torch.arange(66 * 4, dtype=torch.float32).reshape(1, 66, 4)
    req = {"identity": "pooling", "prompt_visual_mask": torch.tensor([False] + [True] * 64 + [False]),
           "grid_thw": [4, 8, 8], "spatial_merge_size": 2,
           "time_values": torch.tensor([0.1, 0.3, 0.5, 0.7]), "intervention": "normal"}
    module.set_request(req)
    features, times, _ = module.prepare(source, torch.arange(70)[None], DynamicCache(), False)
    assert features.shape == (8, 4)
    torch.testing.assert_close(times, torch.tensor([0.2] * 4 + [0.6] * 4))
    raw = source[0, 1:65].reshape(4, 4, 4, 4)
    torch.testing.assert_close(features[0], raw[:2, :2, :2].mean((0, 1, 2)))
    # A prepared cache cannot bypass the known-prefix boundary.
    with pytest.raises(ValueError, match="complete known prompt"):
        module.prepare(source[:, :4], torch.arange(8)[None], DynamicCache(), False)


def make_donor(head):
    donor_request = request("video-donor")
    head.set_evidence_request(donor_request)
    head.evidence_readout.prepare(torch.randn(1, 5, 32), torch.arange(9)[None], DynamicCache(), False)
    return donor_request["prepared_visual"]


def test_wrong_video_substitutes_only_evidence_and_invalidates_donor_kv():
    head = make_head("visual", nonzero=True)
    donor = make_donor(head)
    current = request("video-current", "wrong_video")
    current["donor_visual"] = donor
    head.set_evidence_request(current)
    context, noise, cache = torch.randn(1, 5, 64), torch.randn(1, 4, 32), DynamicCache()
    original_context = context.clone()
    with torch.inference_mode():
        call(head, context, noise, cache)
        cache.crop(5)
        call(head, context[:, :0], noise, cache)
        assert head.evidence_diagnostics()["kv_cache_hits"] == 1
    assert current["identity"] == "video-current"
    assert current["prepared_visual"]["identity"] == "video-current"
    assert current["donor_visual"]["identity"] == "video-donor"
    torch.testing.assert_close(context, original_context, atol=0, rtol=0)
    prepared = head.evidence_readout.prepare(context[..., -32:], torch.arange(9)[None], DynamicCache(), False)
    torch.testing.assert_close(prepared[0], donor["features"], atol=0, rtol=0)
    torch.testing.assert_close(prepared[1], donor["times"], atol=0, rtol=0)
    replacement = copy.deepcopy(donor)
    replacement["identity"] = "video-other-donor"
    replacement["features"].add_(torch.randn_like(replacement["features"]))
    current["donor_visual"] = replacement
    cache.crop(5)
    with torch.no_grad():
        actual = call(head, context[:, :0], noise, cache)
        assert head.evidence_diagnostics()["kv_recomputes"] == 2
        torch.testing.assert_close(actual, call(head, context, noise), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("corruption", ["same_identity", "config", "nonfinite", "requires_grad", "shape", "missing"])
def test_wrong_video_rejects_bad_donor(corruption):
    head = make_head("visual")
    donor = make_donor(head)
    current = request("video-current", "wrong_video")
    current["donor_visual"] = donor
    if corruption == "same_identity":
        donor["identity"] = current["identity"]
    elif corruption == "config":
        donor["config"] = ((2, 2, 2), 2, 16, 1, 32)
    elif corruption == "nonfinite":
        donor["features"][0, 0] = float("nan")
    elif corruption == "requires_grad":
        donor["features"].requires_grad_(True)
    elif corruption == "shape":
        donor["features"] = donor["features"][:, :2]
    else:
        del current["donor_visual"]
    with pytest.raises(ValueError, match="wrong_video"):
        head.set_evidence_request(current)


def test_donor_cannot_be_silently_attached_to_normal_or_text_request():
    head = make_head("visual")
    current = request()
    current["donor_visual"] = make_donor(head)
    with pytest.raises(ValueError, match="only for wrong_video"):
        head.set_evidence_request(current)
    current["intervention"] = "wrong_video"
    with pytest.raises(ValueError, match="visual evidence intervention"):
        make_head("text").set_evidence_request(current)


def test_wrong_video_allows_different_valid_raw_grid_with_same_pooled_shape():
    head = make_head("visual", nonzero=True)
    donor_request = {"identity": "wide-donor", "prompt_visual_mask": torch.tensor([False] + [True] * 8 + [False]),
                     "grid_thw": [2, 4, 4], "spatial_merge_size": 2,
                     "time_values": torch.tensor([0.2, 0.9]), "intervention": "normal"}
    head.set_evidence_request(donor_request)
    donor_source = torch.randn(1, 10, 32)
    head.evidence_readout.prepare(donor_source, torch.arange(14)[None], DynamicCache(), False)
    current = request("video-current", "wrong_video")  # [2,2,2] raw grid.
    current["donor_visual"] = donor_request["prepared_visual"]
    head.set_evidence_request(current)
    actual = head.evidence_readout.prepare(torch.randn(1, 5, 32), torch.arange(9)[None], DynamicCache(), False)
    assert actual[0].shape == (2, 32)
    torch.testing.assert_close(actual[0], current["donor_visual"]["features"], atol=0, rtol=0)


@pytest.mark.parametrize("bad_grid", [(3, 2, 2), (2, 3, 2), (2, 0, 2)])
def test_wrong_video_rejects_incompatible_groups_or_invalid_raw_grid(bad_grid):
    head = make_head("visual")
    donor = make_donor(head)
    donor["config"] = (bad_grid, *donor["config"][1:])
    current = request("video-current", "wrong_video")
    current["donor_visual"] = donor
    with pytest.raises(ValueError, match="wrong_video donor"):
        head.set_evidence_request(current)
