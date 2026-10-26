#!/usr/bin/env python3
"""Phase 1.5: how much of the OPD gradient comes from each stream?

For each sampled batch the loss is decomposed exactly into
`contrib_acc + contrib_rej (+ contrib_k3)`, whose sum is the value returned by
`opd_train_video.py`. Their gradient norms are therefore an exact split of the
update the optimizer actually applies -- not an approximation by position count.

Run once on the shipped pool and once on the aligned pool: the teacher for the
rejected positions is the quantity the phase-0 fix changed, so this is where the
fix can show up as a changed gradient balance before any held-out number moves.

    python diagnose_opd_gradients.py --records <pool.pt> --pool <parquet> --gpu 6
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

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from historical import opd_loss_video
from shared.aligned_forward import target_context_aligned
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--pool", default="/mnt/local/mcy/GLANCE/data/pool_499.parquet")
    ap.add_argument("--holdout-parquet", default="/mnt/local/mcy/GLANCE/data/video_heldout_30.parquet")
    ap.add_argument("--head", default="/mnt/local/mcy/GLANCE/checkpoints/head_video439/step1000")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--max-anchors", type=int, default=96)
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--alpha-rev", type=float, default=0.0)
    ap.add_argument("--reject-target", default="topk", choices=["topk", "draft"])
    ap.add_argument("--drop-inexact", action="store_true", default=False)
    ap.add_argument("--aligned-teacher", action="store_true",
                    help="context also built on the aligned branch")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    embed = target.get_input_embeddings().weight

    blob = torch.load(a.records, weights_only=False)
    pool = pd.read_parquet(a.pool)
    split_of = dict(zip(pool["video_name"].astype(str), pool["split"].astype(str)))
    pool = pool.set_index("video_name")
    excluded = set(pd.read_parquet(a.holdout_parquet)["video_name"].astype(str))
    recs = [r for r in blob["records"]
            if split_of.get(r["task"]) == "train" and r["task"] not in excluded]
    prompt = pool["question"].astype(str).unique()[0]
    vtext = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=False)
    rng = np.random.default_rng(0)

    rows = []
    for step in range(a.steps):
        r = recs[int(rng.integers(len(recs)))]
        p = pool.loc[r["task"]]
        ins = proc(text=[vtext], videos=[p["cache_path"]], return_tensors="pt",
                   fps=float(p["fps"]))
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        full = r["input_ids"][:2048].to(dev).long()
        ctx = target_context_aligned(target, full[None], pixel_values_videos=pv,
                                     video_grid_thw=vg) if a.aligned_teacher else \
            common.target_context(target, full[None], pixel_values_videos=pv,
                                  video_grid_thw=vg)
        all_a = r["anchors"]
        keep = sorted(rng.choice(len(all_a), min(a.max_anchors, len(all_a)),
                                 replace=False).tolist())
        picked = [all_a[i] for i in keep]
        head.zero_grad(set_to_none=True)
        loss, npos, l_fwd, l_rev, nix, parts = opd_loss_video.opd_replay_loss_video(
            head, target, ctx, full,
            [x["pos"] for x in picked], [x["prop"] for x in picked],
            [x["n_acc"] for x in picked],
            r["topk_ids"].to(dev), r["topk_val"].to(dev),
            embed, head.block_size, common.MASK_ID, dev,
            beta_fwd=1.0, decay=a.decay, reject_target=a.reject_target,
            drop_inexact_teacher=a.drop_inexact, alpha_rev=a.alpha_rev,
            return_streams=True)
        if loss is None:
            continue
        g = {}
        for key in ("contrib_acc", "contrib_rej", "contrib_k3"):
            term = parts[key]
            if not torch.is_tensor(term) or not term.requires_grad:
                # An empty stream contributes exactly zero (e.g. the k3 stream when
                # the recipe routes every rejection into the forward loss).
                g[key] = 0.0
                continue
            gr = torch.autograd.grad(term, list(head.parameters()),
                                     retain_graph=True, allow_unused=True)
            g[key] = float(torch.sqrt(sum((x.float() ** 2).sum() for x in gr
                                          if x is not None)))
        tot = sum(v ** 2 for v in g.values()) ** 0.5
        rows.append({"step": step, "loss": float(loss.detach()), "positions": npos,
                     "n_fwd": parts["n_fwd"], "n_topk_rej": parts["n_topk_rej"],
                     "n_k3_rej": parts["n_k3_rej"],
                     "loss_acc": float(parts["contrib_acc"].detach()),
                     "loss_rej": float(parts["contrib_rej"].detach()),
                     "gnorm_acc": g["contrib_acc"], "gnorm_rej": g["contrib_rej"],
                     "gnorm_k3": g["contrib_k3"], "gnorm_total": tot,
                     "share_acc": g["contrib_acc"] / max(tot, 1e-12),
                     "share_rej": g["contrib_rej"] / max(tot, 1e-12),
                     "share_k3": g["contrib_k3"] / max(tot, 1e-12)})
        print(f"  step {step:2d} loss {rows[-1]['loss']:.4f} "
              f"pos {npos} (fwd {parts['n_fwd']} / rej {parts['n_topk_rej']}) "
              f"| gnorm acc {g['contrib_acc']:.4f} rej {g['contrib_rej']:.4f} "
              f"k3 {g['contrib_k3']:.4f} total {tot:.4f} "
              f"| share acc {rows[-1]['share_acc']:.3f} rej {rows[-1]['share_rej']:.3f} "
              f"k3 {rows[-1]['share_k3']:.3f}", flush=True)

    import statistics as st
    summary = {k: st.fmean(r[k] for r in rows) for k in
               ("loss", "positions", "n_fwd", "n_topk_rej", "n_k3_rej",
                "gnorm_acc", "gnorm_rej", "gnorm_k3", "gnorm_total",
                "share_acc", "share_rej", "share_k3")}
    print("\nmeans:", json.dumps({k: round(v, 4) for k, v in summary.items()}))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "steps": rows, "summary": summary},
              open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
