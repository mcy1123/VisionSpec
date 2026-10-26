#!/usr/bin/env python3
"""Concatenate the predecode shards into one pool, with the invariants checked.

The shards all compute the SAME split plan (planning happens before the shard
slice), so the only risk is bookkeeping: a duplicated row, a dropped row, or a
shard that quietly re-cut the split. Each of those is asserted here rather than
hoped for.
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
import glob
import os
import sys

sys.path.insert(0, str(_REPRO_ROOT))
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", required=True)
    ap.add_argument("--prior", required=True, help="pool whose splits are frozen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-total", type=int, default=0)
    a = ap.parse_args()

    paths = sorted(glob.glob(a.shard_glob))
    if not paths:
        raise SystemExit(f"no shard parquets matched {a.shard_glob}")
    parts = [pd.read_parquet(p) for p in paths]
    print(f"shards: {len(parts)} -> " + ", ".join(
        f"{os.path.basename(p)}={len(d)}" for p, d in zip(paths, parts)))
    df = pd.concat(parts, ignore_index=True)

    dup = df["video_name"][df["video_name"].duplicated()].tolist()
    assert not dup, f"duplicate video_name across shards: {dup[:5]}"
    if a.expect_total:
        assert len(df) == a.expect_total, \
            f"expected {a.expect_total} rows, got {len(df)}"

    prior = pd.read_parquet(a.prior)
    pmap = dict(zip(prior["video_name"].astype(str), prior["split"].astype(str)))
    moved = [(n, pmap[n], s) for n, s in
             zip(df["video_name"].astype(str), df["split"].astype(str))
             if n in pmap and pmap[n] != s]
    assert not moved, f"prior splits changed: {moved[:5]}"
    missing = [n for n in pmap if n not in set(df["video_name"].astype(str))]
    assert not missing, f"prior rows dropped: {missing[:5]}"
    print(f"prior {len(pmap)} rows: all present, all splits unchanged")

    df.to_parquet(a.out, index=False)
    vc = df["split"].value_counts().to_dict()
    print(f"wrote {a.out}: {len(df)} rows {vc}")
    L = df["input_ids_len"]
    print(f"input_ids_len: min {L.min()} p25 {int(L.quantile(.25))} "
          f"median {int(L.median())} p75 {int(L.quantile(.75))} max {L.max()}")
    ho = df[df.split == "heldout"]
    print(f"heldout: {len(ho)} rows; "
          f"anchor(prior) {(ho['video_name'].astype(str).isin(pmap)).sum()} / "
          f"fresh {(~ho['video_name'].astype(str).isin(pmap)).sum()}")


if __name__ == "__main__":
    main()
