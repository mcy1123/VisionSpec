#!/usr/bin/env python3
"""Phase 3: score every OPD checkpoint on the frozen held-out 30 rows, pick one.

Selection is on held-out tau ONLY (never on training loss -- that is explicitly
forbidden by the task). The criterion reported is the median tau, the paired
per-row delta against the SFT baseline and its t statistic, plus E[a] and the
row wall time. The chosen checkpoint is the one with the highest median tau among
those whose paired t > 3 (the task's threshold); ties broken by t, then by fewer
no-EOS rows.
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
    # All 30 rows are kept for every tau/delta statistic: the protocol's warm-up
    # drop is a TIMING convention (first-sample GPU state), not a reason to throw
    # a row out of a paired tau comparison. `warmup` is read where it matters.
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
    ap.add_argument("--sft", required=True, help="SFT baseline eval json")
    ap.add_argument("--ckpts", nargs="+", required=True, help="tag=path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sft, scfg = load(a.sft)
    print(f"SFT baseline: {scfg['head']}  n={len(sft)}")
    hdr = ("ckpt", "n", "tau_med", "tau_mean", "Ea_med", "gen_med",
           "d_tau", "sd", "t", "+/n", "noEOS", "spec_s_row")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    table, best = [], None
    for spec in a.ckpts:
        tag, path = spec.split("=", 1)
        arm, _ = load(path)
        keys = sorted(set(sft) & set(arm))
        miss = set(sft) - set(arm)
        if miss:
            print(f"  !! {tag}: {len(miss)} held-out rows missing")
        taus = [arm[k]["tau"] for k in keys]
        m, sd, t, n, pos = paired(sft, arm, keys)
        row = {"tag": tag, "path": path, "n": n,
               "tau_med": st.median(taus), "tau_mean": st.fmean(taus),
               "Ea_med": st.median(arm[k]["Ea"] for k in keys),
               "gen_med": st.median(arm[k]["gen_tokens"] for k in keys),
               "d_tau": m, "sd": sd, "t": t, "pos": pos,
               "no_eos": sum(1 for k in keys if not arm[k]["hit_eos"]),
               # timing excludes the warm-up row (first-sample GPU state)
               "spec_s_row": st.median(arm[k]["spec_wall_s"] for k in keys
                                       if not arm[k].get("warmup"))}
        table.append(row)
        print("| " + " | ".join([
            tag, str(n), f"{row['tau_med']:.4f}", f"{row['tau_mean']:.4f}",
            f"{row['Ea_med']:.4f}", f"{row['gen_med']:.0f}",
            f"{m:+.4f}", f"{sd:.4f}", f"{t:+.2f}", f"{pos}/{n}",
            f"{row['no_eos']}/{n}", f"{row['spec_s_row']:.1f}"]) + " |")

    eligible = [r for r in table if r["n"] > 0 and r["t"] > 3.0]
    pool = eligible or table
    best = max(pool, key=lambda r: (r["tau_med"], r["t"], -r["no_eos"]))
    print(f"\nbest checkpoint (median tau, t>3 gate): {best['tag']}  "
          f"tau_med {best['tau_med']:.4f}  d_tau {best['d_tau']:+.4f}  "
          f"t {best['t']:+.2f}")
    if not eligible:
        print("  NOTE: no checkpoint reached paired t > 3; the best median-tau "
              "checkpoint is reported and the gate failure is recorded.")
    json.dump({"sft": a.sft, "table": table, "best": best}, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
