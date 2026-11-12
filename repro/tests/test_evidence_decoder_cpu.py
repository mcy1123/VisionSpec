"""CPU toy-LM decoder integration, not a real-VLM or GPU performance gate."""
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "GLANCE_repo"))
sys.path.insert(0, str(ROOT / "scripts/dflash_pkg/dflash-0.1.0"))
from dflash.model import DFlashDraftModel
from glance.decode import generate, greedy_generate


@pytest.mark.parametrize("mode", ["text", "visual"])
def test_nonzero_readout_uses_real_decoder_and_keeps_greedy_tokens(mode, monkeypatch):
    # Existing timing helpers synchronize CUDA unconditionally. This CPU test
    # replaces synchronization only, never the target/head/tree computations.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(81)
        cfg = Qwen3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                          vocab_size=64, eos_token_id=63)
        cfg._attn_implementation = "sdpa"
        target = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
        hcfg = Qwen3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                           vocab_size=64, num_target_layers=4, block_size=4,
                           dflash_config={"target_layer_ids": [0, 2], "mask_token_id": 62,
                                          "evidence_mode": mode, "evidence_dim": 8,
                                          "evidence_max_tokens": 8, "evidence_spatial_bins": 2})
        hcfg._attn_implementation = "sdpa"
        head = DFlashDraftModel(hcfg).eval()
        torch.nn.init.normal_(head.evidence_readout.output.weight, std=0.05)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
        mask = torch.zeros(ids.shape[1], dtype=torch.bool)
        mask[2:10] = True
        head.set_evidence_request({"identity": "toy_cpu_request", "prompt_visual_mask": mask,
                                   "grid_thw": [2, 4, 4], "spatial_merge_size": 2,
                                   "time_values": torch.tensor([0.25, 0.75]),
                                   "intervention": "normal", "seed": 0})
        actual, stats = generate(head, target, ids, 24, [], budget=15, branch_k=4,
                                  return_stats=True, timing="decode_only", eos_stop="immediate")
        expected = greedy_generate(target, ids, 24, [])
    assert torch.equal(actual, expected)
    assert stats.rounds > 1
    assert head.evidence_diagnostics()["calls"] == stats.rounds
    if mode == "visual":
        assert head.evidence_diagnostics()["kv_cache_hits"] > 0
