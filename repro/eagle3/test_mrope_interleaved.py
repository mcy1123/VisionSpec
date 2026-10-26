#!/usr/bin/env python3
"""Numerically prove the patched interleaved mRoPE equals transformers'.

This is the one patch in the EAGLE-3-on-Qwen3-VL path that fails *silently* if
it is wrong: feeding the chunked (Qwen2.5-VL) frequency layout to an interleaved
model (Qwen3-VL) raises nothing, trains fine, and simply produces worse
acceptance. So it is checked against the reference implementation rather than
eyeballed.

The test runs on CPU and needs no checkpoint:

    Qwen3VLTextRotaryEmbedding          <- transformers (the reference)
        cos,sin = rot(x, position_ids)                  -> (B, S, head_dim)
        q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)

    LlamaMutiRotaryEmbedding            <- SpecForge (ours)
        cos3,sin3 = rot_sf(x, position_ids)             -> (3, B, S, head_dim)
        q_sf, k_sf = apply_multimodal_rotary_pos_emb(
            q, k, cos3, sin3, mrope_section, mrope_interleaved=True)

    q_hf == q_sf  and  k_hf == k_sf

A passing comparison is only meaningful if the test can fail, so there is a
negative control: the same comparison with `mrope_interleaved=False` must NOT
match. If the control ever passes, the instrument is blind and this test is
worthless -- that is a failure, not a success.

    python test_mrope_interleaved.py --target-config /ckpt/qwen3vl8b_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def build_position_ids(batch, seq, mrope_section, seed=0):
    """A 3-axis position block with a vision-like middle span.

    Text positions carry the identical index on all three axes; a contiguous
    span emulates vision tokens, where temporal advances slowly and height/width
    advance within each frame. That is exactly the regime where chunked and
    interleaved layouts diverge, so a mismatch here is representative rather
    than accidental.
    """
    generator = torch.Generator().manual_seed(seed)
    positions = torch.zeros(3, batch, seq, dtype=torch.long)
    for b in range(batch):
        text_before = 5
        frame_h, frame_w = 3, 4
        frames = 2
        vision = frame_h * frame_w * frames
        cursor = 0
        for _ in range(text_before):
            positions[:, b, cursor] = cursor
            cursor += 1
        base = cursor
        for frame in range(frames):
            for h in range(frame_h):
                for w in range(frame_w):
                    positions[0, b, cursor] = base + frame
                    positions[1, b, cursor] = base + h
                    positions[2, b, cursor] = base + w
                    cursor += 1
        while cursor < seq:
            positions[:, b, cursor] = cursor
            cursor += 1
    # A little jitter on the vision span so H/W are not trivially equal.
    positions[1] += torch.randint(0, 2, positions[1].shape, generator=generator)
    positions[2] += torch.randint(0, 3, positions[2].shape, generator=generator)
    return positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-config", required=True,
                    help="Qwen3-VL config.json (the real target's)")
    ap.add_argument("--seq", type=int, default=37)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--batch-heads", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=2e-6)
    a = ap.parse_args()

    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    from specforge.modeling.draft.llama3_eagle import (
        LlamaMutiRotaryEmbedding,
        apply_multimodal_rotary_pos_emb,
    )

    raw = json.load(open(a.target_config))
    text_cfg = dict(raw["text_config"])
    rope_scaling = text_cfg.get("rope_scaling") or {}
    mrope_section = list(rope_scaling.get("mrope_section", [24, 20, 20]))
    interleaved = bool(rope_scaling.get("mrope_interleaved", False))
    head_dim = int(text_cfg.get("head_dim", 128))
    base = float(text_cfg["rope_theta"])
    max_pos = int(text_cfg["max_position_embeddings"])

    print(f"target config : {a.target_config}")
    print(f"head_dim={head_dim} rope_theta={base} max_pos={max_pos}")
    print(f"mrope_section={mrope_section} mrope_interleaved={interleaved}")
    if sum(mrope_section) * 1 != head_dim // 2:
        print(f"  note: sum(mrope_section)={sum(mrope_section)} vs "
              f"head_dim/2={head_dim // 2}")
    if not interleaved:
        print("  WARNING: this target is NOT interleaved; the negative control "
              "below is expected to match instead of differ")

    hf_rot = Qwen3VLTextRotaryEmbedding(Qwen3VLTextConfig(**text_cfg)).float()
    sf_rot = LlamaMutiRotaryEmbedding(
        head_dim, max_position_embeddings=max_pos, base=base).float()

    position_ids = build_position_ids(a.batch, a.seq, mrope_section)
    x = torch.randn(a.batch, a.seq, head_dim,
                    generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        cos, sin = hf_rot(x, position_ids)
        cos3, sin3 = sf_rot(x, position_ids)

        print(f"\ncos shape: transformers {tuple(cos.shape)}, "
              f"specforge {tuple(cos3.shape)}")

        q = torch.randn(a.batch, a.batch_heads, a.seq, head_dim,
                        generator=torch.Generator().manual_seed(2))
        k = torch.randn_like(q)
        q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        def specforge(mrope_interleaved):
            return apply_multimodal_rotary_pos_emb(
                q, k, cos3, sin3, mrope_section,
                unsqueeze_dim=1, mrope_interleaved=mrope_interleaved)

        q_sf, k_sf = specforge(True)
        q_chunk, k_chunk = specforge(False)

    def maxdiff(p, r):
        return float((p - r).abs().max())

    d_interleaved = max(maxdiff(q_sf, q_hf), maxdiff(k_sf, k_hf))
    d_chunked = max(maxdiff(q_chunk, q_hf), maxdiff(k_chunk, k_hf))
    scale = max(float(q_hf.abs().max()), float(k_hf.abs().max()))

    print(f"\n{'case':<34} {'max|diff|':>12} {'relative':>12}  verdict")
    for label, diff in (("patched interleaved vs transformers", d_interleaved),
                        ("unpatched chunked vs transformers", d_chunked)):
        rel = diff / scale if scale else float("inf")
        print(f"{label:<34} {diff:>12.3e} {rel:>12.3e}")

    failures = []
    if d_interleaved > a.tolerance:
        failures.append(
            f"interleaved path differs from transformers by {d_interleaved:.3e} "
            f"(tolerance {a.tolerance:.1e}) -- the patch is wrong")
    if interleaved and d_chunked <= a.tolerance:
        failures.append(
            "NEGATIVE CONTROL PASSED: the unpatched chunked layout also matches "
            "transformers, so this test cannot detect the bug it exists to "
            "detect. The instrument is blind; not a success.")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    ratio = (f"{d_chunked / d_interleaved:.1f}x worse"
             if d_interleaved else f"off by {d_chunked:.3e}")
    print(f"\nPASS: interleaved matches to {d_interleaved:.3e}; the unpatched "
          f"chunked layout is {ratio}, so the test discriminates.")
    print("The patch reproduces transformers' Qwen3-VL interleaved mRoPE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
