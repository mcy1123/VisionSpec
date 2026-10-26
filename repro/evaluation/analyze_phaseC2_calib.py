#!/usr/bin/env python3
"""Apply the PRE-REGISTERED candidate-selection rule for the continue-SFT calibration.

The rule was written to `docs/reports/2026-10-10-phase-c2-prereg.md` BEFORE any
run, and is applied here verbatim:

  candidates = {lr 3e-5, 1e-5} x {update 5, 10, 25}
  1. score each on the 8-video monitoring subset, paired per video, Delta-tau vs C0
     (C0 = `head_video439_aligned/step1000` read from the phase-A4 eval)
  2. KEEP if mean Delta-tau >= -0.02   ("holds the start point", ~0.5% of a 3.74 tau)
  3. among the kept, take the highest mean Delta-tau; ties -> smaller lr, then
     earlier update (more conservative)
  4. if none is kept -> conclude no stable continue-SFT setting in this range and
     STOP; do not restart the C1/C2 comparison

Step 4 of the task then re-checks the single chosen candidate on the FULL
validation set, with the same >= -0.02 threshold.

Bootstrap resamples VIDEOS, never tokens.
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
import statistics as st

import numpy as np

RES = "/mnt/local/mcy/GLANCE/results"
A4 = f"{RES}/phaseA4/sft439_step1000_fresh30_tree_float32.json"
C2 = f"{RES}/phaseC2"
THRESHOLD = -0.02
LRS = ["3e-5", "1e-5"]
NODES = [5, 10, 25]


def kept(path):
    doc = json.load(open(path))
    return {r["video"]: r for r in doc["rows"] if not r["warmup"]}


def boot(deltas, n=10000, seed=0):
    d = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="monitor8", help="monitor8 or fresh30")
    ap.add_argument("--dir", default=C2, help="directory holding the candidate JSONs")
    ap.add_argument("--tag", default="cal_lr{lr}_u{node}_monitor8",
                    help="filename pattern; {lr} and {node} are substituted")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c0 = kept(A4)
    rows, results = [], {}
    print(f"{'candidate':<22} {'n':>3} {'mean dtau':>10} {'median':>8} {'95% CI':>22} {'pos':>5}  keep")
    for lr in LRS:
        for node in NODES:
            tag = a.tag.format(lr=lr, node=node)
            path = f"{a.dir}/{tag}.json"
            if not os.path.exists(path):
                print(f"{tag:<22} MISSING")
                continue
            d = kept(path)
            keys = sorted(set(d) & set(c0))
            deltas = [d[k]["tau"] - c0[k]["tau"] for k in keys]
            ci = boot(deltas)
            mean = st.fmean(deltas)
            rec = {"lr": lr, "update": node, "n": len(keys), "mean": mean,
                   "median": st.median(deltas), "ci95": ci,
                   "positive": sum(1 for x in deltas if x > 0),
                   "per_video": {k: d[k]["tau"] - c0[k]["tau"] for k in keys},
                   "mean_tau": st.fmean(d[k]["tau"] for k in keys),
                   "lossless": sum(1 for k in keys if d[k].get("lossless")),
                   "keep": mean >= THRESHOLD}
            results[f"lr{lr}_u{node}"] = rec
            rows.append(rec)
            print(f"{tag:<22} {len(keys):>3} {mean:>+10.4f} {st.median(deltas):>+8.4f} "
                  f"[{ci[0]:>+7.4f},{ci[1]:>+7.4f}] {rec['positive']:>3}/{len(keys):<2} "
                  f"{'YES' if rec['keep'] else 'no'}")

    print(f"\nC0 on the subset: n={len(c0)} videos, "
          f"mean tau={st.fmean(v['tau'] for v in c0.values()):.4f}")

    keepers = [r for r in rows if r["keep"]]
    print(f"\n--- pre-registered rule (threshold mean dtau >= {THRESHOLD}) ---")
    if not keepers:
        print("RESULT: no candidate holds the start point in lr {3e-5,1e-5} x updates {5,10,25}")
        print("DECISION: STOP -- do not restart the C1/C2 comparison; pivot to data/objective diagnosis")
        winner = None
    else:
        # ties -> smaller lr, then earlier update
        def key(r):
            return (-r["mean"], float(r["lr"]), r["update"])
        winner = min(keepers, key=key)
        names = ", ".join(f"lr{r['lr']}_u{r['update']}" for r in keepers)
        print(f"RESULT: {len(keepers)} candidate(s) hold the start point: {names}")
        print(f"WINNER by the frozen tie-break: lr={winner['lr']} update={winner['update']} "
              f"(mean dtau {winner['mean']:+.4f}, 95% CI "
              f"[{winner['ci95'][0]:+.4f},{winner['ci95'][1]:+.4f}])")
        print("NEXT: step-4 full-validation re-check of this single candidate")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"threshold": THRESHOLD, "suffix": a.suffix,
                   "candidates": results,
                   "winner": winner,
                   "decision": ("stop" if winner is None else "advance-to-full-validation")},
                  open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
