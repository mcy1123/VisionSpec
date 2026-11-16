#!/usr/bin/env python3
"""Stratified budget-head registry: select a head by the request's frame tier.

This is an inference-side architecture component, not a new trained module. It
keeps 64- and 128-frame heads separate and combines their per-video metrics only
when the request tier is known. It never re-runs the decoder and never mixes
rows from different tiers into a single-model claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path):
    data = json.loads(Path(path).read_text())
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing rows")
    return data, {str(row["video"]): row for row in rows}


def route(tier64, tier128, out):
    data64, rows64 = read_rows(tier64)
    data128, rows128 = read_rows(tier128)
    if set(rows64) != set(rows128):
        raise ValueError("tier eval files must cover the same videos")
    combined = []
    for video in rows64:
        r64, r128 = rows64[video], rows128[video]
        if not (bool(r64.get("lossless")) and bool(r128.get("lossless"))):
            raise ValueError(f"{video}: tier eval is not lossless")
        for tier, row in ((64, r64), (128, r128)):
            combined.append({"video": video, "tier": tier, "selected": row["tau"],
                             "head": "64" if tier == 64 else "128",
                             "lossless": True,
                             "source_tau": row["tau"]})
    by_tier = {
        str(t): {
            "n": sum(1 for row in combined if row["tier"] == t),
            "mean_tau": sum(row["selected"] for row in combined if row["tier"] == t)
            / max(1, sum(1 for row in combined if row["tier"] == t)),
        }
        for t in (64, 128)
    }
    all_mean = sum(row["selected"] for row in combined) / len(combined)
    result = {
        "schema": "budget_head_registry_v1",
        "description": "known-tier head registry; stratified mean, not single-model mixed inference",
        "inputs": {"tier64": str(Path(tier64).resolve()), "tier128": str(Path(tier128).resolve())},
        "n_videos": len(rows64),
        "tiers": by_tier,
        "two_tier_mean_tau": all_mean,
        "lossless": True,
        "rows": combined,
    }
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("n_videos", "tiers", "two_tier_mean_tau", "lossless")}, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier64", required=True)
    ap.add_argument("--tier128", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing to overwrite")
    route(args.tier64, args.tier128, args.out)


if __name__ == "__main__":
    main()
