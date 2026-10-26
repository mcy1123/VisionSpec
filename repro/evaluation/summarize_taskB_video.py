#!/usr/bin/env python3
"""Task B video-QA summary: per-video tau, paired video-level bootstrap.

Reads the evaluator's own `per_question.csv` (clean, named columns) instead of
`all_per_video*.csv`, whose header and rows are misaligned in
`qa_opd_fair/aggregate_eval.py` (the header declares a `stage` column the row
writer never emits, so every field after `run` is shifted by one). The numbers
here are computed from the per-question records directly.

Statistics follow the project protocol: one tau per video per arm first, then a
paired bootstrap that resamples VIDEOS (never questions, never rounds). Timing
columns are reported but flagged `timing_contaminated` when the evaluator says so.

Usage: summarize_taskB_video.py <per_question.csv> --format B --out summary.json
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
import csv
import json
import statistics as st
import sys

import numpy as np


def boot(deltas, n=2000, seed=20261013):
    d = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    return float(np.percentile(d[idx].mean(axis=1), 2.5)), float(np.percentile(d[idx].mean(axis=1), 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--format", default="B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=None)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    rows = [r for r in rows if r["mode"] == a.format]
    if not rows:
        raise SystemExit(f"no rows with mode == {a.format}")

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    per_video = collections.defaultdict(dict)   # arm -> video -> [rows]
    for r in rows:
        per_video[r["arm"]].setdefault(r["video_id"], []).append(r)

    arms = list(per_video)
    summary = {"format": a.format, "n_questions": len(rows),
               "n_videos": len(next(iter(per_video.values()))),
               "arms": {}, "paired": [],
               "timing_contaminated_all": all(
                   str(r.get("timing_contaminated", "")).lower() == "true"
                   for r in rows)}
    for arm in arms:
        vt = {v: st.fmean([f(x["tau"]) for x in rs])
              for v, rs in per_video[arm].items()}
        vr = {v: st.fmean([f(x["rounds"]) for x in rs]) for v, rs in per_video[arm].items()}
        ms = [f(x["decode_ms_token"]) for rs in per_video[arm].values() for x in rs]
        ms = [x for x in ms if x == x]
        summary["arms"][arm] = {
            "n_videos": len(vt),
            "mean_per_video_tau": st.fmean(vt.values()),
            "median_per_video_tau": st.median(vt.values()),
            "mean_per_question_tau": st.fmean([f(x["tau"]) for x in per_video[arm][next(iter(vt))]] or [float("nan")])
            if False else st.fmean([f(x["tau"]) for rs in per_video[arm].values() for x in rs]),
            "mean_rounds": st.fmean(vr.values()),
            "mean_decode_ms_token": st.fmean(ms) if ms else None,
            "n_lossless": sum(1 for rs in per_video[arm].values() for x in rs
                              if str(x["lossless_spec_eq_ar"]).lower() == "true"),
            "n_records": sum(len(rs) for rs in per_video[arm].values()),
            "per_video_tau": vt,
        }

    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            common = sorted(set(per_video[x]) & set(per_video[y]))
            d = [summary["arms"][x]["per_video_tau"][v]
                 - summary["arms"][y]["per_video_tau"][v] for v in common]
            summary["paired"].append({
                "name": f"{x} - {y}", "n_videos": len(common),
                "mean_delta_tau": st.fmean(d), "median_delta_tau": st.median(d),
                "positive": sum(1 for v in d if v > 0),
                "negative": sum(1 for v in d if v < 0),
                "unchanged": sum(1 for v in d if v == 0),
                "ci95": boot(d),
                "per_video_delta": dict(zip(common, d)),
            })
    with open(a.out, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"format {a.format}: {summary['n_questions']} questions, "
          f"{summary['n_videos']} videos, timing_contaminated={summary['timing_contaminated_all']}")
    print("arm          n_vid  mean_video_tau  median  rounds  ms/tok  lossless")
    for arm in arms:
        s = summary["arms"][arm]
        print(f"{arm:12s} {s['n_videos']:5d}  {s['mean_per_video_tau']:.4f}         "
              f"{s['median_per_video_tau']:.4f}  {s['mean_rounds']:6.2f}  "
              f"{(s['mean_decode_ms_token'] or float('nan')):6.2f}  "
              f"{s['n_lossless']}/{s['n_records']}")
    print("\npaired (video-level bootstrap 95% CI)")
    for p in summary["paired"]:
        print(f"  {p['name']:22s} mean {p['mean_delta_tau']:+.4f} "
              f"CI[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] "
              f"pos {p['positive']} neg {p['negative']} unch {p['unchanged']}")
    if a.table:
        lines = ["| video | " + " | ".join(arms) + " |",
                 "|---" * (len(arms) + 1) + "|"]
        for v in sorted(set().union(*[set(per_video[x]) for x in arms])):
            cells = [f"{summary['arms'][x]['per_video_tau'].get(v, float('nan')):.4f}"
                     for x in arms]
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        open(a.table, "w").write("\n".join(lines) + "\n")
        print(f"wrote {a.table}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
