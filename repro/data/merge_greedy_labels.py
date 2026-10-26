#!/usr/bin/env python3
"""Concatenate per-shard label files into one, asserting the union is complete.

The pool was relabelled in two passes (the 150 rows that already existed, then
the 349 new ones) purely to keep the GPU busy against the CPU predecode. The
trainer indexes labels by `task` (= video_name), so the only failure modes are a
duplicate or a missing row -- both asserted here.
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
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--expect", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs, configs = [], []
    for p in a.parts:
        d = torch.load(p, weights_only=False)
        print(f"{os.path.basename(p)}: {len(d['records'])} records")
        recs.extend(d["records"])
        configs.append({k: v for k, v in d["config"].items() if k != "out"})

    tasks = [r["task"] for r in recs]
    dup = sorted({t for t in tasks if tasks.count(t) > 1})
    assert not dup, f"duplicate tasks across label parts: {dup[:5]}"
    if a.expect:
        assert len(recs) == a.expect, f"expected {a.expect} records, got {len(recs)}"

    # Every part must have been rolled out under the same prompt, or the labels
    # are two different objects glued together.
    keys = ("video_prompt", "max_new", "topk", "budget", "branch_k", "head")
    for k in keys:
        vals = {str(c.get(k)) for c in configs}
        assert len(vals) == 1, f"label parts disagree on {k}: {vals}"
        print(f"  {k} = {vals.pop()}")

    torch.save({"records": recs, "config": {"merged_from": a.parts,
                                            "n": len(recs)}}, a.out)
    print(f"wrote {a.out}: {len(recs)} records "
          f"({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
