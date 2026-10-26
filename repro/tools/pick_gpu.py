#!/usr/bin/env python3
"""Print the index of the single GPU with the most free memory (or exit 3).

WHY this exists: `nvidia-smi` cannot talk to the driver on this host
(`NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`),
so every launcher that selected a card with `nvidia-smi --query-gpu=...` now
fails. This uses the torch runtime instead, which does see all devices.

It deliberately returns ONE card -- the emptiest -- rather than a set, because the
project rule is few cards with a real share of memory each, never a thin slice of
every card. `--min-free-gib` makes the caller wait instead of starting an
under-provisioned run.
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
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gib", type=float, default=0.0)
    ap.add_argument("--exclude", default="",
                    help="comma-separated device indices to keep out of the "
                         "choice (used to reserve the largest card for the fp32 "
                         "job while a bf16 job runs on another one)")
    ap.add_argument("--only", default="",
                    help="restrict the choice to this single device index; exits "
                         "3 if it does not meet --min-free-gib, which lets a "
                         "caller ask 'is my pinned card still good?'")
    ap.add_argument("--require-count", type=int, default=8,
                    help="how many devices the host is expected to expose; a "
                         "smaller number means CUDA_VISIBLE_DEVICES is narrowing "
                         "the view and the printed index is then RELATIVE")
    ap.add_argument("--all", action="store_true", help="print every device, "
                                                      "most-free first")
    a = ap.parse_args()

    n = torch.cuda.device_count()
    if n == 0:
        print("no CUDA device visible", file=sys.stderr)
        return 2
    skip = {int(t) for t in a.exclude.split(",") if t.strip()}
    if a.only.strip():
        idx_only = int(a.only)
        free, total = torch.cuda.mem_get_info(idx_only)
        if free / 2**30 < a.min_free_gib:
            print(f"device {idx_only} has {free/2**30:.1f} GiB free, "
                  f"need {a.min_free_gib:.1f}", file=sys.stderr)
            return 3
        print(idx_only)
        return 0
    rows = []
    for i in range(n):
        if i in skip:
            continue
        free, total = torch.cuda.mem_get_info(i)
        rows.append((free / 2**30, i, total / 2**30))
    if not rows:
        print(f"every device is excluded ({sorted(skip)})", file=sys.stderr)
        return 3
    rows.sort(reverse=True)
    if a.all:
        for free, i, total in rows:
            print(f"{i}\t{free:.1f}\t{total:.1f}")
        return 0
    free, i, _ = rows[0]
    print(i)
    if free < a.min_free_gib:
        print(f"best device {i} has only {free:.1f} GiB free, "
              f"need {a.min_free_gib:.1f}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
