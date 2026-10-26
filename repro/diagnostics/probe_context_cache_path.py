#!/usr/bin/env python3
"""Does the training context path see the video at text positions?

Two things are checked on real training rows, because both decide whether the
diagnosis and the V-arm comparison are meaningful:

1. **The trap.** `common.target_context` (the path `train_head_video.py` uses to
   build teacher features) runs `use_cache=False`. Zero the video and the text
   positions come out bit-identical, so the text rows the head trains on carry no
   visual information. Bypass that and the video demonstrably matters.

2. **The mismatch.** The decode path (`glance.decode.generate`) prefills with
   `use_cache=True`. If the two paths disagree about the text rows, then a head
   trained on one is evaluated on the other. This measures the disagreement
   directly, as relative L2 and cosine between the two teacher tensors for the
   SAME input.

No try/except swallowing: a failure surfaces with its traceback.
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
import statistics as st
import sys

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from transformers import AutoProcessor


def rel_l2(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-9))


def cos_mean(a, b):
    return float((F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(common.DATA, "pool_499.parquet"))
    ap.add_argument("--n", type=int, default=6, help="training rows to probe")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(common.ROOT, "results", "diag",
                                                  "context_cache_path.json"))
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.pool)
    df = df[df.split == "train"].head(a.n).reset_index(drop=True)

    target, _ = common.load_target(device=dev, attn_implementation="sdpa")
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    cfg = json.load(open(os.path.join(common.TARGET_PATH, "config.json")))
    vid, iid = int(cfg["video_token_id"]), int(cfg["image_token_id"])
    prompt = df["question"].astype(str).iloc[0]
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=False)

    res = []
    for i, row in df.iterrows():
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        ids = ins["input_ids"].to(dev)
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        vis = ((ids == vid) | (ids == iid))[0]
        txt = ~vis
        z = torch.zeros_like(pv)

        u_r = common.target_context(target, ids, pixel_values_videos=pv, video_grid_thw=vg)[0]
        u_z = common.target_context(target, ids, pixel_values_videos=z, video_grid_thw=vg)[0]
        c_r = common.target_context_cached(target, ids, pixel_values_videos=pv, video_grid_thw=vg)[0]
        c_z = common.target_context_cached(target, ids, pixel_values_videos=z, video_grid_thw=vg)[0]

        rec = {
            "video": str(row["video_name"]), "L": int(ids.shape[1]),
            "n_vis": int(vis.sum()), "n_txt": int(txt.sum()),
            # 1. the trap, per path
            "uncached_relL2_vis": rel_l2(u_r[vis], u_z[vis]),
            "uncached_relL2_txt": rel_l2(u_r[txt], u_z[txt]),
            "uncached_cos_txt": cos_mean(u_r[txt], u_z[txt]),
            "cached_relL2_vis": rel_l2(c_r[vis], c_z[vis]),
            "cached_relL2_txt": rel_l2(c_r[txt], c_z[txt]),
            "cached_cos_txt": cos_mean(c_r[txt], c_z[txt]),
            # 2. the mismatch between the two paths on the same input
            "paths_relL2_vis": rel_l2(u_r[vis], c_r[vis]),
            "paths_relL2_txt": rel_l2(u_r[txt], c_r[txt]),
            "paths_cos_txt": cos_mean(u_r[txt], c_r[txt]),
        }
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:16]:16s} L={rec['L']:4d} "
              f"vis={rec['n_vis']:4d} txt={rec['n_txt']:3d} | "
              f"video effect on TEXT rows: uncached {rec['uncached_relL2_txt']:.4f} "
              f"cached {rec['cached_relL2_txt']:.4f} | "
              f"path mismatch TEXT {rec['paths_relL2_txt']:.4f} cos {rec['paths_cos_txt']:+.3f}",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": res}, open(a.out, "w"), indent=1)
    print("\nmedian over rows:")
    for k in ["uncached_relL2_vis", "uncached_relL2_txt", "uncached_cos_txt",
              "cached_relL2_vis", "cached_relL2_txt", "cached_cos_txt",
              "paths_relL2_vis", "paths_relL2_txt", "paths_cos_txt"]:
        print(f"  {k:20s} {st.median(r[k] for r in res):+.4f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
