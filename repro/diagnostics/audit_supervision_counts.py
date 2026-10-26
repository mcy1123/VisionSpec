#!/usr/bin/env python3
"""Audit: how many positions does the continue-SFT arm ACTUALLY supervise?

The Phase C report claimed "C1 supervises 1200 positions (50 updates x 24
anchors), C2 supervises 18000, i.e. 15x". That is wrong: `train_head.build_blocks`
expands EVERY anchor into a full `block_size`-wide block and sets

    super_mask = (~visible & in_bounds)      visible = offset < reveal

so one anchor supervises `block_size - reveal` positions, not one. `1200` is an
ANCHOR count. The real numbers the audit needs are

    sum(super_mask)        # supervised positions
    sum(super_mask * w)    # weight sum == the loss's denominator
                           #   loss = (nll * w).sum() / w.sum()

`sequence_loss` normalises by the WEIGHT sum, not by the position count
(train_head.py: loss = (nll * flat_w).sum() / flat_w.sum().clamp_min(1.0)), and
`w_k = DECAY ** (k - reveal)`, so the two differ.

This reproduces `train_head_video.py`'s per-step anchor/reveal construction
EXACTLY -- same `np.random.default_rng(seed)` permutation and subsampling order,
same per-step `torch.Generator().manual_seed(seed + step)` -- but uses only the
pure block bookkeeping (`build_blocks` needs no model). No GPU, no forward pass.

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

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from historical import train_head as th

R = "/mnt/local/mcy/GLANCE"
BLOCK = common.BLOCK_SIZE


def run(records, pool, *, steps, max_anchors, seed, max_seq_len=2048):
    recs = torch.load(records, weights_only=False)["records"]
    pool_df = pd.read_parquet(pool)
    split_of = dict(zip(pool_df["video_name"].astype(str), pool_df["split"].astype(str)))
    n_all = len(recs)
    recs = [r for r in recs if split_of.get(r["task"]) == "train"]
    assert len(recs) == n_all - 60, (len(recs), n_all)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(recs))
    pos_counts, w_sums, n_anchors, n_rows = [], [], [], 0
    step = 0
    while step < steps:
        r = recs[int(order[step % len(order)])]
        full = r["input_ids"][:max_seq_len].long()
        prompt_len = int(min(r["prompt_len"], full.shape[0] - 2))
        total = int(full.shape[0])
        anchors = list(range(prompt_len, total - 1))
        reveals = th.sample_reveals(len(anchors), BLOCK,
                                    torch.Generator().manual_seed(seed + step), "cpu")
        anchors, reveals = th.live_anchors(anchors, reveals, total, BLOCK)
        if not anchors:
            continue
        if max_anchors and len(anchors) > max_anchors:
            keep = rng.choice(len(anchors), max_anchors, replace=False)
            keep.sort()
            anchors = [anchors[i] for i in keep]
            reveals = reveals[torch.as_tensor(keep)]
        _blocks, _tgt, super_mask, weights = th.build_blocks(
            full, anchors, reveals, BLOCK, common.MASK_ID, torch.device("cpu"))
        pos_counts.append(int(super_mask.sum().item()))
        w_sums.append(float(weights.sum().item()))
        n_anchors.append(len(anchors))
        n_rows += 1
        step += 1
    return {"steps": n_rows, "max_anchors": max_anchors, "block": BLOCK,
            "positions_total": sum(pos_counts), "weight_sum_total": sum(w_sums),
            "positions_mean_per_step": st.fmean(pos_counts),
            "positions_min": min(pos_counts), "positions_max": max(pos_counts),
            "weight_sum_mean_per_step": st.fmean(w_sums),
            "anchors_total": sum(n_anchors),
            "positions_per_anchor": sum(pos_counts) / max(sum(n_anchors), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=f"{R}/data/video_greedy_499.pt")
    ap.add_argument("--pool", default=f"{R}/data/pool_499.parquet")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{R}/results/phaseC/audit_supervision_counts.json")
    a = ap.parse_args()

    out = {}
    for label, ma in (("C1_phaseC_maxanchors24", 24), ("A4_maxanchors96", 96)):
        r = run(a.records, a.pool, steps=a.steps, max_anchors=ma, seed=a.seed)
        out[label] = r
        print(f"--- {label} (max-anchors {ma}, block {r['block']}, {r['steps']} steps) ---")
        print(f"    anchors total          : {r['anchors_total']}")
        print(f"    SUPERVISED POSITIONS   : {r['positions_total']}")
        print(f"    WEIGHT SUM (loss denom): {r['weight_sum_total']:.2f}")
        print(f"    positions / anchor     : {r['positions_per_anchor']:.3f}")
        print(f"    positions per step     : mean {r['positions_mean_per_step']:.2f} "
              f"[{r['positions_min']},{r['positions_max']}]")
        print()

    print("=== comparison with C2 (from results/phaseC/C2_opd_fullkl.jsonl) ===")
    print("    C2 supervised offsets  : 18000   (accepted 2043 + rejected 15957)")
    c1 = out["C1_phaseC_maxanchors24"]["positions_total"]
    print(f"    C1 supervised positions: {c1}")
    print(f"    ratio C2/C1            : {18000 / c1:.2f}x")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
