#!/usr/bin/env python3
"""Does the drafter's KV carry visual information anywhere but the visual rows?

The main table's most surprising result is that zeroing the visual KV rows (arm C)
costs almost nothing, while replacing them with the target's raw visual embeddings
(arm B) costs a lot. The mechanism this script tests is that the drafter's context
is not the only place vision lives: after a *cached* prefill the target's TEXT
hidden states already depend on the video, and those text rows are 92% of nothing
but ARE the rows arm C keeps. If that is true, arm C removes the explicit copy of
vision while the implicit copy survives, and arm B's damage is not "losing vision"
but "adding out-of-distribution distractor keys".

For each row, two cached prefills (real vs zeroed `pixel_values_videos`), then the
same five-layer fused tensor the head consumes is compared at visual and at text
positions, and the actually-projected KV (`hidden_norm(fc(fused))`) is compared
too. Reported as relative L2 change and cosine.

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
import statistics as st
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from glance.head import context_feature
from glance.vl import VisionPositions
from transformers import AutoProcessor, DynamicCache


def rel_l2(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-9))


def cos_mean(a, b):
    return float((F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(-1).mean())


@torch.no_grad()
def prefill_hidden(target, ids, vg, pixels, pos, dev):
    cache = DynamicCache()
    out = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                 output_hidden_states=True, **pos.prefill_inputs(ids, None))
    return out.hidden_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.rows)
    if a.n:
        df = df.head(a.n)

    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev).eval()
    cfg = json.load(open(os.path.join(common.TARGET_PATH, "config.json")))
    vid, iid = int(cfg["video_token_id"]), int(cfg["image_token_id"])
    print(f"KV visual content: {len(df)} rows, head {a.head}", flush=True)

    res = []
    for i, row in df.reset_index(drop=True).iterrows():
        t0 = time.perf_counter()
        text = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        ids = ins["input_ids"].to(dev)
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        vis = ((ids == vid) | (ids == iid))[0]
        txt = ~vis

        h_real = prefill_hidden(target, ids, vg, pv,
                                VisionPositions(pixel_values_videos=pv, video_grid_thw=vg), dev)
        h_blank = prefill_hidden(target, ids, vg, torch.zeros_like(pv),
                                 VisionPositions(pixel_values_videos=torch.zeros_like(pv),
                                                 video_grid_thw=vg), dev)
        lid = head.target_layer_ids
        fused_r = context_feature(h_real, lid)[0]
        fused_b = context_feature(h_blank, lid)[0]
        proj_r = head.hidden_norm(head.fc(fused_r))
        proj_b = head.hidden_norm(head.fc(fused_b))

        rec = {
            "video": str(row["video_name"]), "L": int(ids.shape[1]),
            "n_vis": int(vis.sum()), "n_txt": int(txt.sum()),
            "vis_frac": float(vis.float().mean()),
            "fused_relL2_vis": rel_l2(fused_r[vis], fused_b[vis]),
            "fused_relL2_txt": rel_l2(fused_r[txt], fused_b[txt]),
            "fused_cos_vis": cos_mean(fused_r[vis], fused_b[vis]),
            "fused_cos_txt": cos_mean(fused_r[txt], fused_b[txt]),
            "proj_relL2_vis": rel_l2(proj_r[vis], proj_b[vis]),
            "proj_relL2_txt": rel_l2(proj_r[txt], proj_b[txt]),
            "proj_cos_vis": cos_mean(proj_r[vis], proj_b[vis]),
            "proj_cos_txt": cos_mean(proj_r[txt], proj_b[txt]),
            "wall_s": time.perf_counter() - t0,
        }
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:16]:16s} vis_frac={rec['vis_frac']:.2f} "
              f"fused relL2 vis={rec['fused_relL2_vis']:.3f} txt={rec['fused_relL2_txt']:.3f} "
              f"| proj relL2 vis={rec['proj_relL2_vis']:.3f} txt={rec['proj_relL2_txt']:.3f} "
              f"| proj cos txt={rec['proj_cos_txt']:+.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": res}, open(a.out, "w"), indent=1)
    print("\nmedian over rows:")
    for k in ["vis_frac", "fused_relL2_vis", "fused_relL2_txt", "fused_cos_vis",
              "fused_cos_txt", "proj_relL2_vis", "proj_relL2_txt", "proj_cos_vis",
              "proj_cos_txt"]:
        print(f"  {k:18s} {st.median(r[k] for r in res):+.4f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
