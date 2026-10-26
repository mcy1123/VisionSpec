#!/usr/bin/env python3
"""Build the multi-prompt pool index for the revival task.

499 cached videos x 4 prompts = 1996 rows (439 train videos x 4 = **1756 training
rows**, the number the task asks for; the 60 held-out videos also get the 4
prompts so the held-out set can be probed per prompt, but the reported held-out
numbers use prompt 0 so they stay comparable to the 3.49 baseline).

The primary key is `task = "<video_name>__p<pid>"`: the trainers index records by
`task` (`train_head_video.py`, `opd_train_video.py`) and the original single-prompt
pool would collide four ways otherwise. `opd_rollout.py` already prefers
`row["task"]` over `row["video_name"]` when it labels a record, and reads each
row's own `question` -- which is what makes a multi-prompt collection possible
without touching it.

Writes the index plus N rollout shards; no silent fallback anywhere.
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

PROMPT0 = ("Please provide a detailed description of the video, focusing on the "
           "main subjects, their actions, and the background scenes")
PROMPTS = [
    PROMPT0,                                        # p0 = the existing pool's prompt
    "What happens in this video?",                  # p1 = question form
    "Describe the sequence of events",              # p2 = temporal form
    "Summarize what happens in this video briefly.",  # p3 = short summary
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/mnt/local/mcy/GLANCE/data/pool_499.parquet")
    ap.add_argument("--out-index", default="/mnt/local/mcy/GLANCE/data/pool_multiprompt_1996.parquet")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--out-dir", default="/mnt/local/mcy/GLANCE/data")
    a = ap.parse_args()

    base = pd.read_parquet(a.pool).reset_index(drop=True)
    assert len(base) == 499, len(base)
    assert str(base["question"].astype(str).unique()[0]) == PROMPT0, (
        "pool_499's prompt is not prompt p0; refusing to guess the mapping")
    rows = []
    for _, r in base.iterrows():
        for pid, q in enumerate(PROMPTS):
            d = dict(r)
            d["prompt_id"] = pid
            d["question"] = q
            d["task"] = f"{r['video_name']}__p{pid}"
            d["base_video"] = r["video_name"]
            rows.append(d)
    df = pd.DataFrame(rows)
    assert df["task"].is_unique, "task keys are not unique"
    tr = df[df.split == "train"]
    ho = df[df.split == "heldout"]
    print(f"index: {len(df)} rows = {len(tr)} train + {len(ho)} heldout")
    assert len(tr) == 439 * 4 == 1756, len(tr)
    assert len(ho) == 60 * 4 == 240, len(ho)
    for pid in range(4):
        n = int((tr.prompt_id == pid).sum())
        assert n == 439, (pid, n)
    df.to_parquet(a.out_index, index=False)
    print(f"wrote {a.out_index}")

    # rollout shards: only the 3 NEW prompts need collecting (p0 already exists in
    # video_opd_pool_warm_499.pt), but the shards carry whole videos so a shard's
    # rows can be merged back per video if needed.
    new = df[df.prompt_id > 0].reset_index(drop=True)
    edges = np.linspace(0, len(new), a.shards + 1).astype(int)
    for i in range(a.shards):
        part = new.iloc[edges[i]:edges[i + 1]].reset_index(drop=True)
        p = os.path.join(a.out_dir, f"pool_multiprompt_roll_shard{i}.parquet")
        part.to_parquet(p, index=False)
        print(f"wrote {p}: {len(part)} rows "
              f"(train {int((part.split=='train').sum())} / heldout {int((part.split=='heldout').sum())}, "
              f"prompts {sorted(part.prompt_id.unique())})")


if __name__ == "__main__":
    main()
