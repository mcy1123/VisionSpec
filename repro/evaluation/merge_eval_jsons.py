#!/usr/bin/env python3
"""Concatenate sharded `eval_video_opd.py` outputs, asserting they are one run.

Only the shard/out fields may differ between parts: same rows file, same head,
same budget/branch/max_new/warmup/ar. Duplicate or overlapping row sets are
rejected, so a lost shard cannot silently shrink the denominator.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--expect", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, cfgs = [], []
    for p in a.parts:
        d = json.load(open(p))
        print(f"{os.path.basename(p)}: {len(d['rows'])} rows")
        rows.extend(d["rows"])
        cfgs.append(d["config"])

    keys = ("rows", "head", "budget", "branch", "max_new", "ar", "warmup", "dtype")
    for k in keys:
        vals = {str(c.get(k)) for c in cfgs}
        assert len(vals) == 1, f"shards disagree on {k}: {vals}"
        print(f"  {k} = {vals.pop()}")

    vids = [r["video"] for r in rows]
    dup = sorted({v for v in vids if vids.count(v) > 1})
    assert not dup, f"rows scored twice across shards: {dup[:5]}"
    if a.expect:
        assert len(rows) == a.expect, f"expected {a.expect} rows, got {len(rows)}"

    cfg = dict(cfgs[0])
    cfg["merged_from"] = a.parts
    cfg["shards"] = len(a.parts)
    json.dump({"config": cfg, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
