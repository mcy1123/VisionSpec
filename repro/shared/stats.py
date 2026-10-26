#!/usr/bin/env python3
"""Summarise the training corpus the head will actually see.

Two jobs. First, it is the data section of the write-up: rows per task,
generation lengths, prompt lengths. Second, it closes a loop that is otherwise
easy to leave open -- how much supervision the variable-prefix objective
actually produces. A row contributes one block per anchor, and a block only
supervises the offsets at or past its sampled reveal length, so "16K rows" is
not a number one can reason about without measuring.

    python stats.py --data $GLANCE_ROOT/data/target_greedy.jsonl
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
import collections
import json
import os
import statistics

from shared import common


def quantiles(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return {}
    pick = lambda q: values[min(n - 1, int(q * n))]
    return {"min": values[0], "p10": pick(0.10), "median": pick(0.50),
            "p90": pick(0.90), "max": values[-1], "mean": sum(values) / n}


def supervision(row, block_size=common.BLOCK_SIZE):
    """Anchors and supervised offsets for one row, in expectation over reveals.

    Mirrors `train_head.build_blocks`: anchors run from the first generated
    token to `total - 2`, block position `k` is supervised when `k >= reveal`
    and `k` stays inside the row, and `reveal` is drawn from the truncated
    geometric prior.
    """
    n_prompt = row["n_prompt"]
    total = n_prompt + len(row["gen_ids"])
    n_anchors = max(0, (total - 2) - n_prompt + 1)

    _, probs = common.prefix_weights(block_size)
    lengths = list(range(common.PREFIX_MIN, block_size))

    supervised = 0.0
    live = 0.0
    for a in range(n_prompt, total - 1):
        in_bounds = min(block_size - 1, total - 1 - a)
        # a block survives only if some offset >= reveal is also in bounds
        p_live = sum(p for l, p in zip(lengths, probs)
                     if in_bounds >= l)
        live += p_live
        for l, p in zip(lengths, probs):
            if in_bounds >= l:
                supervised += p * (in_bounds - l + 1)
    return n_anchors, live, supervised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(common.DATA, "target_greedy.jsonl"))
    parser.add_argument("--sample", type=int, default=400,
                        help="rows sampled for the supervision estimate")
    parser.add_argument("--out", default=os.path.join(common.RESULTS, "data_stats.json"))
    args = parser.parse_args()

    rows = common.read_jsonl(args.data)
    by_task = collections.Counter(r["task"] for r in rows)
    print(f"{len(rows)} rows from {args.data}")
    print(f"  task mix: {dict(by_task)}")

    gen = quantiles([len(r["gen_ids"]) for r in rows])
    prompt = quantiles([r["n_prompt"] for r in rows])
    print(f"  generation length: {gen}")
    print(f"  prompt length    : {prompt}")

    finished = sum(1 for r in rows if r.get("finish") == "eos")
    print(f"  finished by EOS  : {finished} ({100*finished/len(rows):.1f}%)")
    short = sum(1 for r in rows if len(r["gen_ids"]) < common.BLOCK_SIZE)
    print(f"  shorter than one block: {short} ({100*short/len(rows):.2f}%)")

    step = max(1, len(rows) // args.sample)
    sampled = rows[::step][: args.sample]
    per_row = [supervision(r) for r in sampled]
    anchors = statistics.fmean(a for a, _, _ in per_row)
    live = statistics.fmean(l for _, l, _ in per_row)
    sup = statistics.fmean(s for _, _, s in per_row)
    print(f"  per row (n={len(sampled)} sampled): anchors {anchors:.1f}, "
          f"non-empty blocks {live:.1f}, supervised offsets {sup:.1f}")
    print(f"  full epoch supervision ~= {sup * len(rows) / 1e6:.1f}M offsets")

    payload = {
        "rows": len(rows), "task_mix": dict(by_task),
        "generation_length": gen, "prompt_length": prompt,
        "eos_fraction": finished / len(rows),
        "shorter_than_block": short,
        "per_row": {"anchors": anchors, "non_empty_blocks": live,
                    "supervised_offsets": sup, "sampled": len(sampled)},
        "epoch_supervised_offsets": sup * len(rows),
        "prefix_base": common.PREFIX_BASE,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
