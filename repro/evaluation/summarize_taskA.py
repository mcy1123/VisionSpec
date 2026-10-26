#!/usr/bin/env python3
"""Paired per-video tau summary for the Task A arms (and any other video eval).

Everything here is per VIDEO, never per token or per round: the videos are the
statistical unit (AGENTS.md "指标纪律" / "以视频为主统计单位"). The first
`--warmup` rows of each eval are dropped, matching phase-A4/phase-C4 (`warmup 1`
-> n=29).

Usage:
  python summarize_taskA.py --arm control=.../control_u10_fresh30.json \
      --arm p1_k3=.../p1_k3_u10_fresh30.json \
      --ref c0=/mnt/local/mcy/GLANCE/results/phaseA4/sft439_step1000_fresh30_tree_float32.json \
      --out summary.json
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


def load(path):
    with open(path) as handle:
        return json.load(handle)


def arm_rows(path, warmup=None):
    doc = load(path)
    rows = [r for r in doc["rows"] if not r.get("warmup", False)] \
        if warmup is None else \
        [r for i, r in enumerate(doc["rows"]) if i >= int(warmup)]
    return {str(r["video"]): r for r in rows}, doc


def lossless_gate(doc):
    """The SPEC==AR gate is checked on EVERY row, warmup included."""
    rows = doc["rows"]
    if not rows or "lossless" not in rows[0]:
        return {"checked": False, "n": len(rows), "n_lossless": None}
    return {"checked": True, "n": len(rows),
            "n_lossless": sum(1 for r in rows if r.get("lossless")),
            "mismatched": [r["video"] for r in rows if not r.get("lossless")]}


def describe(rows, doc):
    tau = [r["tau"] for r in rows.values()]
    def mean(key):
        vals = [r[key] for r in rows.values() if key in r]
        return st.fmean(vals) if vals else None
    return {
        "n": len(rows),
        "mean_tau": st.fmean(tau),
        "median_tau": st.median(tau),
        "mean_rounds": mean("rounds"),
        "mean_gen_tokens": mean("gen_tokens"),
        "mean_spec_decode_ms_per_tok": mean("spec_decode_ms_per_tok"),
        "mean_spec_wall_s": mean("spec_wall_s"),
        "lossless": lossless_gate(doc),
    }


def bootstrap_ci(deltas, n_boot=10000, seed=20261015, lo=2.5, hi=97.5):
    d = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def compare(name, a, b):
    keys = sorted(set(a) & set(b))
    d = [a[k]["tau"] - b[k]["tau"] for k in keys]
    sd = st.stdev(d) if len(d) > 1 else 0.0
    return {
        "name": name,
        "n": len(keys),
        "mean_delta_tau": st.fmean(d),
        "median_delta_tau": st.median(d),
        "positive_videos": sum(1 for x in d if x > 0),
        "negative_videos": sum(1 for x in d if x < 0),
        "changed_videos": sum(1 for x in d if x != 0),
        "ci95": bootstrap_ci(d),
        "t": (st.fmean(d) / (sd / len(d) ** 0.5)) if sd > 0 else float("nan"),
        "mean_tau_a": st.fmean(a[k]["tau"] for k in keys),
        "mean_tau_b": st.fmean(b[k]["tau"] for k in keys),
        "per_video_delta": {k: a[k]["tau"] - b[k]["tau"] for k in keys},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="tag=path.json (repeatable; order defines the table)")
    ap.add_argument("--ref", action="append", default=[],
                    help="tag=path.json reference arm (repeatable)")
    ap.add_argument("--warmup", type=int, default=None,
                    help="drop the first N rows per file; default uses the "
                         "per-row 'warmup' flag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=None,
                    help="also write a per-video table (markdown) with every arm's "
                         "tau, rounds, generated tokens and the lossless flag")
    a = ap.parse_args()

    arms, order = {}, []
    for spec in a.arm + a.ref:
        tag, _, path = spec.partition("=")
        rows, doc = arm_rows(path, a.warmup)
        arms[tag] = (rows, doc)
        order.append(tag)
        print(f"[load] {tag}: {len(rows)} videos from {path}")

    summary = {
        "files": {spec.split("=", 1)[0]: spec.split("=", 1)[1]
                  for spec in a.arm + a.ref},
        "arms": {tag: describe(rows, doc) for tag, (rows, doc) in arms.items()},
        "paired": [],
    }
    for i, tag in enumerate(order):
        for other in order[i + 1:]:
            summary["paired"].append(compare(f"{tag} - {other}",
                                             arms[tag][0], arms[other][0]))
    with open(a.out, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\narm                 n  mean_tau  median  rounds  ms/tok  lossless gate")
    for tag in order:
        s = summary["arms"][tag]
        g = s["lossless"]
        gate = (f"{g['n_lossless']}/{g['n']}" if g["checked"] else "n/a")
        print(f"{tag:18s} {s['n']:3d}  {s['mean_tau']:.4f}  "
              f"{s['median_tau']:.4f}  "
              f"{(s['mean_rounds'] or float('nan')):6.2f}  "
              f"{(s['mean_spec_decode_ms_per_tok'] or float('nan')):6.2f}  {gate}")
    print("\npaired (a - b), video-level bootstrap 95% CI")
    for p in summary["paired"]:
        print(f"  {p['name']:26s} mean {p['mean_delta_tau']:+.4f} "
              f"CI[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] t {p['t']:+.2f} "
              f"pos {p['positive_videos']}/{p['n']} neg {p['negative_videos']} "
              f"changed {p['changed_videos']}")

    if a.table:
        keys = sorted(set().union(*[set(arms[t][0]) for t in order]))
        cols = []
        for tag in order:
            rows_t = arms[tag][0]
            cols += [f"{tag}_tau", f"{tag}_rounds", f"{tag}_gen",
                     f"{tag}_lossless"]
        lines = ["| video | " + " | ".join(cols) + " |",
                 "|---" * (len(cols) + 1) + "|"]
        for k in keys:
            cells = []
            for tag in order:
                r = arms[tag][0].get(k)
                if r is None:
                    cells += ["", "", "", ""]
                else:
                    cells += [f"{r['tau']:.4f}", str(r.get("rounds", "")),
                              str(r.get("gen_tokens", "")),
                              "yes" if r.get("lossless") else "NO"]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        with open(a.table, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"wrote {a.table} ({len(keys)} videos)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
