"""Exercise the actual Qwen3-VL merger hook and first OPD anchor on CPU."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dflash_pkg/dflash-0.1.0"))
sys.path.insert(0, str(ROOT / "repro"))
from dflash.paper_transfer import FEATURE_SOURCE, capture_projector, paper_readout
from local_h100.evidence_opd_prompt_boundary import known_visual_prefix


def test_actual_qwen_video_prefill_and_first_opd_anchor():
    torch.manual_seed(113)
    cfg = Qwen3VLConfig(
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=48, num_heads=4,
                           patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                           out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0]),
        text_config=dict(hidden_size=32, intermediate_size=64, head_dim=8, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                         rope_scaling=dict(rope_type="default", mrope_interleaved=True, mrope_section=[2,1,1])),
        video_token_id=58, image_token_id=57, vision_start_token_id=59, vision_end_token_id=60,
        attn_implementation="eager",
    )
    target = Qwen3VLForConditionalGeneration(cfg).eval()
    inputs = torch.tensor([[10,59,58,58,58,58,60,11,59,58,58,58,58,60,12]])
    grid = torch.tensor([[2,4,4]])
    pixels = torch.randn(32, 24)
    readout = paper_readout("vispec", hidden_size=32, dim=16, max_tokens=8, spatial_bins=2, mode="visual")
    head = SimpleNamespace(evidence_readout=readout)
    request = dict(identity="real-qwen", grid_thw=grid[0].tolist(), spatial_merge_size=2,
                   time_values=torch.tensor([0.1,0.9]), prompt_visual_mask=inputs[0] == 58, intervention="normal")
    readout.set_request(request)

    def prefill():
        return target(input_ids=inputs, attention_mask=torch.ones_like(inputs), pixel_values_videos=pixels,
                      video_grid_thw=grid, use_cache=False).logits

    with torch.inference_mode():
        baseline = prefill()
        expected = target.get_video_features(pixel_values_videos=pixels, video_grid_thw=grid)[0][0]
        with capture_projector(target, head, request):
            actual = prefill()

    packed = request["prepared_visual"]
    assert packed["feature_source"] == FEATURE_SOURCE and not packed["features"].is_inference()
    assert torch.equal(expected, packed["features"])
    assert torch.equal(baseline, actual)
    n = inputs.shape[-1]
    with known_visual_prefix(head, n, request):
        # Poison backbone features. Any fallback to target LM hidden states is visible.
        visual, _, _ = readout.prepare(torch.full((1, n - 1, 32), float("nan")), torch.arange(n + 3)[None], None, False)
        assert torch.equal(expected, visual)
    del request["prepared_visual"]
    with pytest.raises(ValueError, match="missing captured"):
        with known_visual_prefix(head, n, request):
            pass
