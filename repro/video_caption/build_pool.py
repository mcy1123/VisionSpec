#!/usr/bin/env python3
"""Build the explicit video-caption pure-block pool + four-slot schedule.

Why this file exists: the active OPD data layer is task-keyed (`task` index plus
a `prompt` column and a schedule JSONL). The frozen video captioning pool
`data/pool_499.parquet` predates that contract and carries only
`video_name/question/answer/cache_path/...`, so it cannot be fed to
`opd_official_kl` unchanged. This adapter is additive: it does not modify the
frozen pool, it writes a new parquet under the run directory.

Inputs are VideoDetailCaption captions (500 clips, 499 pooled rows, 439 train).
The generated length on this set is measured at ~270 tokens per clip (see
docs/reports for the A4 evaluation), which is the long-generation regime the
image-side OPD work was validated on. Nothing here rebuilds frames or prompts:
`prompt` is exactly the pool `question` string the warm head was trained with,
and `cache_path/fps/n_frames` are copied verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_POOL = "/mnt/local/mcy/GLANCE/data/pool_499.parquet"


def prompt_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build(pool_path, out_dir, updates, smoke_updates, seed, slots=4):
    source = pd.read_parquet(pool_path)
    required = {"video_name", "question", "answer", "cache_path", "fps", "n_frames", "split"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"pool missing columns: {sorted(missing)}")
    train = source[source["split"] == "train"].copy()
    if train["video_name"].duplicated().any():
        raise ValueError("video_name must be unique in the single-prompt pool")
    if not train["cache_path"].map(lambda p: Path(p).exists()).all():
        raise ValueError("every cache_path must exist")

    pool = train.rename(columns={"question": "prompt"})
    pool["task"] = pool["video_name"].astype(str)
    order = ["task", "prompt", "cache_path", "fps", "n_frames", "answer",
             "video_name", "input_ids_len", "width", "height", "split"]
    pool = pool[order].reset_index(drop=True)
    pool.to_parquet(out_dir / "pool_video_caption.parquet", index=False)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pool)).tolist()
    needs = updates * slots
    if needs > len(perm):
        repeats = int(np.ceil(needs / len(perm)))
        perm = (perm * repeats)[:needs]
    tasks = pool["task"].tolist()
    shas = pool["prompt"].map(prompt_sha).tolist()

    schedules = {}
    for name, n_updates in (("schedule.jsonl", updates),
                            ("schedule_smoke.jsonl", smoke_updates)):
        lines = []
        for update in range(n_updates):
            for slot in range(slots):
                index = perm[update * slots + slot]
                lines.append(json.dumps(dict(update=update, slot=slot,
                                             task=tasks[index],
                                             prompt_sha256=shas[index])))
        (out_dir / name).write_text("\n".join(lines) + "\n")
        schedules[name] = n_updates

    summary = dict(pool=str(out_dir / "pool_video_caption.parquet"),
                   source_pool=pool_path, rows=len(pool), updates=updates,
                   smoke_updates=smoke_updates, slots=slots, seed=seed,
                   duplicated_rows_in_epoch=int(needs - len(set(perm[:needs]))),
                   schedules=schedules,
                   answer_tokens_source="gold answers are ~150 tokens; measured AR generation is ~270 tokens")
    (out_dir / "POOL.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--updates", type=int, default=110)
    ap.add_argument("--smoke-updates", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    build(args.pool, out, args.updates, args.smoke_updates, args.seed)


if __name__ == "__main__":
    main()
