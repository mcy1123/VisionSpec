#!/usr/bin/env python3
"""Stage 0 for the video OPD long task: freeze the splits, shard the rollout.

Three artifacts, all asserts and no silent fallback:

1. `data/video_heldout_30.parquet` -- 30 rows drawn with a FIXED seed from the
   439 train rows of `pool_499.parquet`. They were never used for any selection
   (the SFT-stage selections used the 30 `pool_150` held-out rows and were
   confirmed once on the 30 `pool_499_remaining` held-out rows), and they are
   asserted disjoint from BOTH of those sets. Caveat that is stated in the report
   rather than hidden: they WERE in `head_video439`'s training set, so the SFT arm
   has memorised them too; both arms share that, and the two historical 30-row
   sets are carried along as secondary (cross-check) sets.
2. `data/video_opd_train_409.parquet` -- the other 409 rows: the OPD training set.
3. `data/opd_roll_shard{0,1,2}.parquet` -- the whole 499-row pool cut three ways so
   `opd_rollout.py` can re-collect the OPD anchors on three GPUs at once.

Why re-collect the anchors at all: the shipped `video_greedy_499.pt` was rolled
out by the ZERO-SHOT image head (`checkpoints/head`), while this task warm-starts
OPD from `head_video439/step1000`. The committed tokens are head-independent
(GLANGE is lossless: the target commits its own greedy tokens either way), but
the ANCHORS -- which states OPD trains on -- are exactly what the head defines.
The image-side verified recipe collects its pool with the same head it then
warm-starts from, so the video pool is re-collected the same way.
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
import os

import numpy as np
import pandas as pd

ROOT = "/mnt/local/mcy/GLANCE"
DATA = os.path.join(ROOT, "data")
SEED = 0
N_HELDOUT = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(DATA, "pool_499.parquet"))
    ap.add_argument("--pool150", default=os.path.join(DATA, "pool_150.parquet"))
    ap.add_argument("--remaining", default=os.path.join(DATA, "pool_499_remaining.parquet"))
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--out-dir", default=DATA)
    a = ap.parse_args()

    pool = pd.read_parquet(a.pool).reset_index(drop=True)
    p150 = pd.read_parquet(a.pool150)
    rem = pd.read_parquet(a.remaining)

    n_tr = int((pool.split == "train").sum())
    n_ho = int((pool.split == "heldout").sum())
    print(f"pool_499: {len(pool)} rows = {n_tr} train + {n_ho} heldout")
    assert (n_tr, n_ho) == (439, 60), "pool_499 split drifted from 439/60"

    train = pool[pool.split == "train"].reset_index(drop=True)
    assert len(set(train.video_name) & set(pool[pool.split == "heldout"].video_name)) == 0

    # The two historical 30-row sets. h150 = pool_150's held-out rows;
    # fresh30 = pool_499_remaining's held-out rows. These were the selection and
    # confirmation sets of the SFT stage.
    h150 = set(p150[p150.split == "heldout"].video_name.astype(str))
    fresh30 = set(rem[rem.split == "heldout"].video_name.astype(str))
    assert len(h150) == 30 and len(fresh30) == 30, (len(h150), len(fresh30))
    hist = h150 | fresh30
    assert len(hist) == 60
    assert hist == set(pool[pool.split == "heldout"].video_name.astype(str))

    # New independent held-out, drawn from the 439 TRAIN rows with a fixed seed.
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(train), N_HELDOUT, replace=False))
    held = train.iloc[idx].reset_index(drop=True)
    held_names = set(held.video_name.astype(str))
    # Requirement from the task: must differ from the SFT stage's 30-row held-out.
    assert not (held_names & h150), "new held-out overlaps the SFT anchor 30"
    assert not (held_names & fresh30), "new held-out overlaps the SFT fresh 30"
    assert len(held_names) == N_HELDOUT

    train409 = train.drop(index=idx).reset_index(drop=True)
    assert len(train409) == 439 - N_HELDOUT == 409
    assert len(set(train409.video_name) & held_names) == 0
    print(f"heldout_30 drawn from the 439 train rows with seed {SEED}; "
          f"train set = {len(train409)} rows")
    print(f"  disjoint from pool_150 held-out: {len(held_names & h150)} overlap")
    print(f"  disjoint from fresh 30:          {len(held_names & fresh30)} overlap")

    os.makedirs(a.out_dir, exist_ok=True)
    for name, df in (("video_heldout_30.parquet", held),
                     ("video_opd_train_409.parquet", train409)):
        path = os.path.join(a.out_dir, name)
        df.to_parquet(path, index=False)
        print(f"wrote {path}: {len(df)} rows")

    # Rollout shards over the WHOLE 499-row pool: the trainer filters by split and
    # by the held-out list, so the pool must cover every row. Contiguous cuts.
    edges = np.linspace(0, len(pool), a.shards + 1).astype(int)
    for i in range(a.shards):
        part = pool.iloc[edges[i]:edges[i + 1]].reset_index(drop=True)
        path = os.path.join(a.out_dir, f"opd_roll_shard{i}.parquet")
        part.to_parquet(path, index=False)
        print(f"wrote {path}: {len(part)} rows "
              f"(train {int((part.split == 'train').sum())} / "
              f"heldout {int((part.split == 'heldout').sum())})")


if __name__ == "__main__":
    main()
