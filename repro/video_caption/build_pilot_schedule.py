#!/usr/bin/env python3
"""Freeze a video-level training order; never silently drop/pad a last batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def build_schedule(rows, *, epochs=3, qa_per_update=4, seed=20261111):
    if epochs < 1 or qa_per_update < 1:
        raise ValueError("epochs and qa_per_update must be positive")
    if not rows or len(rows) % qa_per_update:
        raise ValueError("training row count must be nonzero and divisible by qa_per_update")
    tasks = [str(row["task"]) for row in rows]
    if len(set(tasks)) != len(tasks):
        raise ValueError("duplicate task IDs")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("schedule accepts only explicit split=train rows")
    videos = [str(row["video_id"]) for row in rows]
    if len(set(videos)) != len(videos):
        raise ValueError("pilot requires one task per source video, not repeated QA variants")
    rng = random.Random(seed)
    result = []
    for epoch in range(epochs):
        order = list(tasks)
        rng.shuffle(order)
        for offset, task in enumerate(order):
            result.append({"update": (epoch * len(tasks) + offset) // qa_per_update,
                           "slot": offset % qa_per_update, "task": task,
                           "epoch": epoch + 1})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True, help="one frame-tier training JSONL or parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--qa-per-update", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()
    path = Path(args.rows)
    if path.suffix == ".parquet":
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    schedule = build_schedule(rows, epochs=args.epochs,
                              qa_per_update=args.qa_per_update, seed=args.seed)
    out = Path(args.out)
    protocol_path = out.with_suffix(out.suffix + ".protocol.json")
    if out.exists() or protocol_path.exists():
        raise SystemExit("refusing to overwrite an existing schedule or protocol")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in schedule)
    protocol = {"version": 1, "rows": str(path.resolve()),
                "rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "schedule_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "epochs": args.epochs, "qa_per_update": args.qa_per_update,
                "seed": args.seed, "updates": len(schedule) // args.qa_per_update,
                "qa_exposures": len(schedule), "unique_videos": len(rows),
                "dropped_or_padded": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        handle.write(payload)
    with protocol_path.open("x") as handle:
        json.dump(protocol, handle, indent=2)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
