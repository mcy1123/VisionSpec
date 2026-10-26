#!/usr/bin/env python3
"""Freeze the two video subsets the plan needs, BEFORE any result is looked at.

Plan section 10.2 requires 8 videos drawn from the TRAINING set for the B0.5
teacher-precision calibration. Plan section 10.6 requires 8 of the 30
`video_fresh30` videos as a process-monitoring subset. Both selections must be
reproducible and must NOT depend on any model score, or the monitoring subset
stops being a monitoring subset and becomes a second selection set.

Selection rule (identical for both, and score-free):

    rank each candidate by sha256("<seed>:<video_name>") and take the lowest N

This is a fixed hash order: deterministic, independent of model output, and not
re-derivable by looking at results. `--seed` is recorded in the output so the
rule can be replayed.

The training-set draw additionally excludes anything that appears in either
held-out list, so B0.5 never touches validation rows.

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
import hashlib
import json
import os

import pandas as pd

ROOT = "/mnt/local/mcy/GLANCE"
DATA = f"{ROOT}/data"


def hash_rank(names, seed):
    def key(n):
        return hashlib.sha256(f"{seed}:{n}".encode()).hexdigest()
    return sorted(names, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=f"{DATA}/pool_499.parquet")
    ap.add_argument("--fresh30", default=f"{DATA}/video_fresh30.parquet")
    ap.add_argument("--heldout30", default=f"{DATA}/video_heldout_30.parquet")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-monitor", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/results/phaseB05/subsets.json")
    a = ap.parse_args()

    pool = pd.read_parquet(a.pool)
    fresh = pd.read_parquet(a.fresh30)
    held = pd.read_parquet(a.heldout30)
    pool["video_name"] = pool["video_name"].astype(str)
    fresh_names = sorted(fresh["video_name"].astype(str).tolist())
    held_names = sorted(held["video_name"].astype(str).tolist())

    train = pool[pool["split"].astype(str) == "train"]
    train_names = sorted(train["video_name"].astype(str).unique().tolist())
    heldout_split = sorted(
        pool[pool["split"].astype(str) != "train"]["video_name"].astype(str).unique().tolist())

    print(f"pool_499: {len(pool)} rows, {pool['video_name'].nunique()} videos")
    print(f"  train split      : {len(train_names)} videos")
    print(f"  non-train split  : {len(heldout_split)} videos")
    print(f"video_fresh30      : {len(fresh_names)} videos")
    print(f"video_heldout_30   : {len(held_names)} videos")

    # Hard leak checks before selecting anything.
    leak_fresh = sorted(set(train_names) & set(fresh_names))
    leak_held = sorted(set(train_names) & set(held_names))
    print(f"\nleak check  train ∩ fresh30    : {len(leak_fresh)} (expected 0) {leak_fresh[:5]}")
    print(f"leak check  train ∩ heldout30  : {len(leak_held)} (expected 30) {leak_held[:3]}...")
    if leak_fresh:
        raise SystemExit("video_fresh30 overlaps the train split -- selection assumptions broken")

    # B0.5 draw: training videos that are in neither held-out list.
    forbidden = set(fresh_names) | set(held_names)
    cand = [n for n in train_names if n not in forbidden]
    print(f"\nB0.5 candidates (train, not in either held-out list): {len(cand)}")
    b05 = hash_rank(cand, f"b05-seed{a.seed}")[: a.n_train]

    # Monitoring subset: 8 of the 30 fresh30 videos, same score-free rule.
    mon = hash_rank(fresh_names, f"monitor-seed{a.seed}")[: a.n_monitor]

    doc = {
        "selection_rule": "rank by sha256('<tag>:<video_name>'), take lowest N",
        "score_free": True,
        "seed": a.seed,
        "counts": {"pool_videos": int(pool["video_name"].nunique()),
                   "train_videos": len(train_names),
                   "nontrain_videos": len(heldout_split),
                   "fresh30": len(fresh_names), "heldout30": len(held_names)},
        "leak_checks": {"train_intersect_fresh30": leak_fresh,
                        "train_intersect_heldout30_count": len(leak_held)},
        "b05_train_videos": b05,
        "monitor_fresh30_videos": mon,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=1)
    print(f"\nB0.5 8 training videos ({len(b05)}):")
    for n in b05:
        print(f"  {n}")
    print(f"\nmonitoring 8 of fresh30 ({len(mon)}):")
    for n in mon:
        print(f"  {n}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
