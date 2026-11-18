"""ViSpec joint-prefix attention, including DFlash's speculative cache crop."""
from pathlib import Path
import sys

import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dflash_pkg/dflash-0.1.0"))
sys.path.insert(0, str(ROOT / "GLANCE_repo"))
from dflash.model import DFlashDraftModel
from dflash.paper_transfer import FEATURE_SOURCE
from glance.decode import generate, greedy_generate


def make_head():
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      vocab_size=64, num_target_layers=4, block_size=4,
                      dflash_config=dict(target_layer_ids=[0, 2], mask_token_id=62, paper_method="vispec",
                                         evidence_mode="visual", evidence_dim=16,
                                         evidence_max_tokens=8, evidence_spatial_bins=2))
    cfg._attn_implementation = "sdpa"
    head = DFlashDraftModel(cfg).eval()
    mask = torch.tensor([0,1,1,1,1,0,0,1,1,1,1,0], dtype=torch.bool)
    request = dict(identity="vispec-video", prompt_visual_mask=mask, grid_thw=[2,4,4],
                   spatial_merge_size=2, time_values=torch.tensor([.25,.75]), intervention="normal")
    head.set_evidence_request(request)
    request["prepared_visual"] = dict(identity=request["identity"], config=head.evidence_readout._config_key(),
                                     features=torch.randn(8,32), times=request["time_values"].repeat_interleave(4),
                                     groups=2, spatial_bins=4, feature_source=FEATURE_SOURCE)
    return head


@pytest.mark.parametrize("mask_type", ["bool", "additive"])
def test_same_softmax_as_real_inserted_rows(mask_type):
    torch.manual_seed(65)
    head = make_head()
    attn = head.layers[0].self_attn
    hidden, context = torch.randn(1,4,32), torch.randn(1,12,32)
    prefix = torch.randn(2,32,requires_grad=True)
    ids = head.evidence_readout.prefix_positions(2, hidden.device)
    assert ids.tolist() == [4,10]
    positions = torch.arange(16)
    packed_rope = head.rotary_emb(hidden, torch.cat((ids, positions))[None])
    prefix_rope = tuple(x[:,:2] for x in packed_rope)
    rope = tuple(x[:,2:] for x in packed_rope)
    mask = torch.ones(1,1,4,16,dtype=torch.bool)
    mask[...,1,10] = False
    if mask_type == "additive":
        mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,float("-inf"))
    actual = attn(hidden, context, rope, mask, vispec_prefix=(prefix, ids, *prefix_rope))[0]
    expected = attn(hidden, torch.cat((prefix[None],context), dim=1), packed_rope,
                    torch.cat((mask.index_select(-1,ids),mask), dim=-1))[0]
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert prefix.grad.abs().sum() > 0


def test_crop_and_tree_decode(monkeypatch):
    torch.manual_seed(74)
    head = make_head()
    context, noise, cache = torch.randn(1,17,64), torch.randn(1,4,32), DynamicCache()
    def run(rows, kv):
        start = kv.get_seq_length()
        return head(target_hidden=rows, noise_embedding=noise, past_key_values=kv, use_cache=True,
                    position_ids=torch.arange(start, start + rows.shape[1] + 4)[None], is_causal=False)
    with torch.inference_mode():
        run(context[:,:12],cache)
        assert cache.get_seq_length() == 16
        cache.crop(12)
        incremental = run(context[:,12:15],cache)
        assert cache.get_seq_length() == 19
        torch.testing.assert_close(incremental, run(context[:,:15],DynamicCache()), atol=2e-6,rtol=2e-5)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **kw: None)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64)
    cfg._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(cfg).eval()
    ids = torch.arange(1,13)[None]
    out, stats = generate(head,target,ids,12,[],budget=15,branch_k=4,return_stats=True,
                          timing="decode_only",eos_stop="immediate")
    assert torch.equal(out,greedy_generate(target,ids,12,[]))
    assert stats.rounds > 1 and head.evidence_diagnostics()["kv_cache_hits"] > 0
