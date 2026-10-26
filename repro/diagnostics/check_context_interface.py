#!/usr/bin/env python3
"""Gate the shared target-prefill interface before Phase A4/B0 rely on it.

The plan puts continue-SFT and OPD on ONE decode-aligned feature path, and asks
for a gate: on the same row the SFT context and the OPD context must agree with
`max|delta| == 0`. Two functions exist today and are used by different callers:

  * `common.target_context_cached`  -- `use_cache=True` + DynamicCache (the path
    `train_fresh_video.py --feature-path cached` uses)
  * `aligned_forward.target_context_aligned` -- `use_cache=False` + all-visible
    2-D attention_mask (the path `opd_train_video.py --aligned-forward` uses)

They are supposed to be the same object. This script decides that empirically on
real video rows instead of assuming it, and reports cosine plus the exact-equality
count so a tolerance cannot hide a real disagreement.

It also re-measures the legacy `use_cache=False` (no mask) path's text-position
video-independence on the same rows, so the A4 motivation is checkable rather
than inherited from a previous report.

No try/except: a failure surfaces with its traceback.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from shared.aligned_forward import target_context_aligned, verify_aligned_to_decode
from transformers import AutoProcessor


def compare(a, b):
    d = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0).item()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "cosine": cos, "exact": bool(torch.equal(a, b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="/mnt/local/mcy/GLANCE/data/video_fresh30.parquet")
    ap.add_argument("--n", type=int, default=2, help="rows to check")
    ap.add_argument("--video-dependence", action="store_true",
                    help="also measure text-position dependence on the video input")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    target, _ = common.load_target(device=dev)          # bf16, the training dtype
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    df = pd.read_parquet(a.rows).head(a.n)

    report = {"rows": [], "n_rows": len(df)}
    for i, row in df.iterrows():
        text = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        ids = ins["input_ids"].to(dev)
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)

        ctx_cached = common.target_context_cached(
            target, ids, pixel_values_videos=pv, video_grid_thw=vg)
        ctx_aligned = target_context_aligned(
            target, ids, pixel_values_videos=pv, video_grid_thw=vg)
        ctx_legacy = common.target_context(
            target, ids, pixel_values_videos=pv, video_grid_thw=vg)

        rec = {"video": str(row["video_name"]), "L": int(ids.shape[1]),
               "cached_vs_aligned": compare(ctx_cached, ctx_aligned),
               "cached_vs_legacy": compare(ctx_cached, ctx_legacy),
               "aligned_vs_legacy": compare(ctx_aligned, ctx_legacy)}

        if a.video_dependence:
            # Blank the video and re-read the SAME positions, split into TEXT and
            # VISUAL rows. The 2026-10-05 report's claim is specifically about the
            # TEXT rows (legacy `use_cache=False` makes them video-independent),
            # so a whole-tensor max would conflate that with the visual rows, which
            # move under every path. Qwen3-VL marks visual rows with the video
            # token id, so the split is read off `input_ids` rather than guessed.
            vid_id = int(target.config.video_token_id)
            text_mask = (ids[0] != vid_id)
            zero = torch.zeros_like(pv)
            ref = {"cached": ctx_cached, "aligned": ctx_aligned, "legacy": ctx_legacy}
            got = {
                "cached": common.target_context_cached(
                    target, ids, pixel_values_videos=zero, video_grid_thw=vg),
                "legacy": common.target_context(
                    target, ids, pixel_values_videos=zero, video_grid_thw=vg),
            }
            dep = {}
            for k, g in got.items():
                d = (ref[k].float() - g.float()).abs()[0]        # (L, L*H)
                rel = d / ref[k].float()[0].abs().clamp_min(1e-6)
                dep[k] = {
                    "text_max_abs": d[text_mask].max().item(),
                    "text_mean_abs": d[text_mask].mean().item(),
                    "text_max_rel": rel[text_mask].max().item(),
                    "visual_max_abs": d[~text_mask].max().item(),
                    "n_text": int(text_mask.sum()), "n_visual": int((~text_mask).sum()),
                }
            rec["blank_video_dependence"] = dep

        report["rows"].append(rec)
        print(f"[{i+1}/{len(df)}] {rec['video'][:16]:16s} L={rec['L']:4d}  "
              f"cached_vs_aligned max|d|={rec['cached_vs_aligned']['max_abs']:.3e} "
              f"exact={rec['cached_vs_aligned']['exact']} cos={rec['cached_vs_aligned']['cosine']:.6f}")
        print(f"    cached_vs_legacy  max|d|={rec['cached_vs_legacy']['max_abs']:.3e} "
              f"cos={rec['cached_vs_legacy']['cosine']:.6f}")
        if a.video_dependence:
            dep = rec["blank_video_dependence"]
            print(f"    blank-video TEXT max|d|: cached {dep['cached']['text_max_abs']:.3e}  "
                  f"legacy {dep['legacy']['text_max_abs']:.3e}  "
                  f"(n_text={dep['cached']['n_text']})")

    all_exact = all(r["cached_vs_aligned"]["exact"] for r in report["rows"])
    worst = max(r["cached_vs_aligned"]["max_abs"] for r in report["rows"])
    report["cached_equals_aligned_all_rows"] = all_exact
    report["cached_vs_aligned_worst_max_abs"] = worst
    print("\n=== GATE ===")
    print(f"cached == aligned bit-for-bit on every checked row: {all_exact} "
          f"(worst max|d| = {worst:.3e})")
    print("VERDICT: " + ("one shared interface -- either function is the same object"
                         if all_exact else
                         "NOT the same object -- must unify to ONE function before A4/B0"))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
