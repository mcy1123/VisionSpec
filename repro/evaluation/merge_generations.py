#!/usr/bin/env python3
"""Concatenate a generation run's shard files into the merged output.

Factored out of `scripts/10_run_gen.sh` so a catch-up worker can re-merge
without editing a script that a running shell may still be reading: bash reads
scripts lazily by byte offset, so rewriting one mid-run is its own hazard.

    python merge_generations.py --base $GLANCE_ROOT/data/target_greedy_textvqa
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

from shared import common


def merge(base, shard_count=2, extra=(), keep_duplicates=False):
    out = base + ".jsonl"
    rows = []
    for shard in list(range(shard_count)) + list(extra):
        path = f"{base}.shard{shard}.jsonl"
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        part = []
        bad = 0
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    part.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
        print(f"{path}: {len(part)} rows" + (f" ({bad} unreadable line(s) skipped)" if bad else ""))
        rows += part

    # A resume run can re-append a row that was already written before a kill,
    # so dedupe on the identity the resume logic itself uses. That default is
    # right for a crash-recovery merge and wrong for a final one: the corpus
    # itself contains repeated (image, question) pairs, and dropping them loses
    # real rows. TextVQA has at least one -- the same photograph asked "what
    # movie is this?" twice, at question indices 2672 and 3519 -- which is how a
    # merge of two 4000-row shards came out at 7999.
    if keep_duplicates:
        unique = rows
    else:
        seen = set()
        unique = []
        for row in rows:
            key = (row["image"], row["prompt"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)

    with open(out, "w") as handle:
        for row in unique:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(unique)} rows -> {out} ({len(rows) - len(unique)} duplicates dropped)")

    lengths = sorted(len(r["gen_ids"]) for r in unique)
    if lengths:
        n = len(lengths)
        print(f"generation length: min {lengths[0]} p10 {lengths[n//10]} "
              f"median {lengths[n//2]} p90 {lengths[9*n//10]} max {lengths[-1]}")
    return len(unique)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="output path without the .jsonl suffix")
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--extra", default="",
                        help="comma-separated extra shard indices (catch-up workers)")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="final merge: repeated (image, question) pairs are real rows")
    args = parser.parse_args()
    extra = [int(x) for x in args.extra.split(",") if x.strip()]
    count = merge(args.base, args.shards, extra, args.keep_duplicates)
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
