#!/usr/bin/env python3
"""Paired tau comparison of video heads against the zero-shot arm.

Why paired: tau varies ~0.25 sd across videos while the effects being chased are
~0.03, so the unpaired standard error of a 30-row mean (~0.046) is larger than
the signal. Comparing the SAME videos under two heads removes the video effect.

Anchor vs fresh held-out. The 30 "anchor" rows are the held-out split of the
original 150-row pool, i.e. the rows every earlier number in this project is
tied to; the 30 "fresh" rows were added when the pool grew to 499 and were never
trained on by any arm. Reporting them separately is what keeps checkpoint
selection from being scored on its own selection set.

No p-value theatre: the t statistic and its df are printed next to the raw
per-row deltas so the reader can see it is n=30.
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
import json
import math
import os
import statistics as st


def load(path):
    d = json.load(open(path))
    return {r["video"]: r for r in d["rows"]}, d["config"]


def paired(base, arm, keys):
    d = [arm[k]["tau"] - base[k]["tau"] for k in keys]
    n = len(d)
    m = st.fmean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    return m, sd, t, n, sum(1 for x in d if x > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--zero", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="tag=path")
    ap.add_argument("--prior", default=None,
                    help="pool whose held-out rows are the anchor set")
    a = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(a.pool)
    ho = df[df.split == "heldout"]
    prior = set()
    if a.prior:
        prior = set(pd.read_parquet(a.prior)["video_name"].astype(str))
    anchors = set(ho["video_name"].astype(str)) & prior
    fresh = set(ho["video_name"].astype(str)) - prior

    zero, zcfg = load(a.zero)
    print(f"zero-shot head: {zcfg['head']}")
    print(f"held-out: {len(ho)} rows = anchor {len(anchors)} + fresh {len(fresh)}\n")

    hdr = ("arm", "n", "tau_med", "tau_mean", "tau_sd", "Ea_med",
           "d_anchor", "t", "+/n", "d_fresh", "t", "+/n", "d_all", "t", "noEOS", "s_row")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for spec in [f"zero={a.zero}"] + a.arms:
        tag, path = spec.split("=", 1)
        arm, _ = load(path)
        keys = [k for k in zero if k in arm]
        miss = set(zero) - set(arm)
        if miss:
            print(f"  !! {tag}: {len(miss)} rows missing from arm")
        taus = [arm[k]["tau"] for k in keys]
        ka = [k for k in keys if k in anchors]
        kf = [k for k in keys if k in fresh]
        row = [tag, str(len(keys)), f"{st.median(taus):.4f}",
               f"{st.fmean(taus):.4f}", f"{st.stdev(taus):.4f}",
               f"{st.median(arm[k]['tau'] - 1 for k in keys):.4f}"]
        if tag == "zero":
            row += ["-", "-", "-", "-", "-", "-", "-", "-"]
        else:
            for ks in (ka, kf, keys):
                if not ks:
                    # arms swept on the anchor rows only: no fresh rows to pair
                    row += ["n/a", "-", "0/0"]
                    continue
                m, sd, t, n, pos = paired(zero, arm, ks)
                row += [f"{m:+.4f}", f"{t:+.2f}", f"{pos}/{n}"]
        row += [f"{sum(1 for k in keys if not arm[k]['hit_eos'])}/{len(keys)}",
                f"{st.median(arm[k]['wall_s'] for k in keys):.1f}"]
        print("| " + " | ".join(row) + " |")

    if not a.arms:
        return
    print("\nper-checkpoint tau (median / mean) -- all held-out rows, then anchor only:")
    for spec in a.arms:
        tag, path = spec.split("=", 1)
        arm, _ = load(path)
        ka = [k for k in arm if k in anchors]
        taus = [arm[k]["tau"] for k in arm]
        ta = [arm[k]["tau"] for k in ka]
        print(f"  {tag:28s} all n={len(taus):3d} med {st.median(taus):.4f} "
              f"mean {st.fmean(taus):.4f} | anchor n={len(ta):3d} "
              f"med {st.median(ta):.4f} mean {st.fmean(ta):.4f}")


if __name__ == "__main__":
    main()
