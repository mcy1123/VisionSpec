#!/usr/bin/env python3
"""Phase A3 judgement for 后续计划_视频OPD_2026-10-08.md section 3.

Reads the per-checkpoint `eval_video_opd.py` JSONs from `results/phaseA/` and
applies the plan's pre-registered plateau rule, in the plan's own words:

  * plateau confirmed  : max pairwise difference across step500/1000/1500 < 0.03 tau
                        AND not monotonically increasing
                        -> take the checkpoint with the highest clean mean tau
  * still rising       : monotonically increasing -> keep SFT-ing to first plateau
  * rise-then-fall     : take the clean generalisation peak

Also reports, per checkpoint: mean/median tau, per-sample tau, rounds, generated
length, EOS hit rate, and the fp32 speculative==AR lossless gate that A2 makes a
validity condition. A checkpoint whose lossless rate is not 100% is reported but
excluded from the plateau decision, because the plan voids those results.

No try/except: a malformed or missing result surfaces.
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
import sys

PLATEAU_EPS = 0.03
DEFAULT_STEPS = [500, 1000, 1500]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def keep_rows(doc):
    """Drop warmup rows exactly as eval_video_opd.py does (it marks them)."""
    return [r for r in doc["rows"] if not r["warmup"]]


def agg(doc):
    rows = keep_rows(doc)
    taus = [r["tau"] for r in rows]
    lossless = [bool(r.get("lossless")) for r in doc["rows"]]  # gate uses ALL rows
    n_bad = sum(1 for x in lossless if not x)
    return {
        "n": len(rows),
        "mean": st.fmean(taus),
        "median": st.median(taus),
        "sd": st.stdev(taus) if len(taus) > 1 else float("nan"),
        "min": min(taus),
        "max": max(taus),
        "eos": sum(1 for r in rows if r["hit_eos"]),
        "rounds_median": st.median(r["rounds"] for r in rows),
        "gen_median": st.median(r["gen_tokens"] for r in rows),
        "cov60_median": st.median(r["cov60"] for r in rows),
        "lossless_n": len(lossless) - n_bad,
        "lossless_total": len(lossless),
        "lossless_ok": n_bad == 0,
        "per_sample": {r["video"]: r["tau"] for r in rows},
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/local/mcy/GLANCE/results/phaseA")
    ap.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = {}
    for s in a.steps:
        p = os.path.join(a.dir, f"sft439_step{s}_fresh30_tree_float32.json")
        data[s] = agg(load(p))

    print("=" * 78)
    print("Phase A clean SFT plateau curve -- video_fresh30, fp32, GLANCE tree b63/k8")
    print("=" * 78)
    hdr = f"{'step':>6} {'n':>3} {'mean tau':>9} {'median':>8} {'sd':>7} {'min':>7} {'max':>7} {'EOS':>7} {'rnds':>6} {'gen':>6} {'lossless':>10}"
    print(hdr)
    for s in a.steps:
        d = data[s]
        print(f"{s:>6} {d['n']:>3} {d['mean']:>9.4f} {d['median']:>8.4f} "
              f"{d['sd']:>7.4f} {d['min']:>7.4f} {d['max']:>7.4f} "
              f"{d['eos']:>3}/{d['n']:<3} {d['rounds_median']:>6.0f} "
              f"{d['gen_median']:>6.0f} {d['lossless_n']:>4}/{d['lossless_total']:<5}")

    usable = [s for s in a.steps if data[s]["lossless_ok"]]
    voided = [s for s in a.steps if not data[s]["lossless_ok"]]
    for s in voided:
        d = data[s]
        print(f"\n!! step{s}: fp32 lossless {d['lossless_n']}/{d['lossless_total']} "
              f"-> result VOID per A2; excluded from the plateau decision")
        bad = [r["video"] for r in d["rows"] if not r.get("lossless")]
        if bad:
            print(f"   mismatched rows: {bad}")

    print("\n--- pairwise differences on the common scored rows (paired) ---")
    common = None
    for s in usable:
        keys = set(data[s]["per_sample"])
        common = keys if common is None else (common & keys)
    common = sorted(common or [])
    print(f"common rows across usable checkpoints: n={len(common)}")
    for i, s1 in enumerate(usable):
        for s2 in usable[i + 1:]:
            d = [data[s2]["per_sample"][k] - data[s1]["per_sample"][k] for k in common]
            t = st.fmean(d) / (st.stdev(d) / len(d) ** 0.5) if len(d) > 1 and st.stdev(d) > 0 else float("nan")
            pos = sum(1 for x in d if x > 0)
            print(f"  step{s2} - step{s1}: mean {st.fmean(d):+.4f}  "
                  f"median {st.median(d):+.4f}  t {t:+.2f}  positive {pos}/{len(d)}")

    print("\n--- A3 decision ---")
    means = [data[s]["mean"] for s in usable]
    spread = max(means) - min(means) if len(means) > 1 else 0.0
    rising = all(means[i] < means[i + 1] for i in range(len(means) - 1)) if len(means) > 1 else False
    print(f"usable checkpoints: {usable}")
    print(f"mean tau spread (max-min) = {spread:.4f}   (plateau threshold <{PLATEAU_EPS})")
    print(f"monotonically increasing  = {rising}")

    verdict = None
    if len(usable) < 2:
        verdict = "INSUFFICIENT: fewer than 2 valid checkpoints"
    elif spread < PLATEAU_EPS and not rising:
        best = max(usable, key=lambda s: data[s]["mean"])
        verdict = (f"PLATEAU CONFIRMED -- take step{best} "
                   f"(clean mean tau {data[best]['mean']:.4f})")
    elif rising:
        verdict = ("STILL RISING -- continue SFT past step1500 until the curve "
                   "first flattens, then enter Phase B (plan section 3 A3)")
    else:
        best = max(usable, key=lambda s: data[s]["mean"])
        verdict = (f"RISE-THEN-FALL / NOT FLAT (spread {spread:.4f}) -- take the clean "
                   f"generalisation peak step{best} (mean tau {data[best]['mean']:.4f})")
    print(f"\nVERDICT: {verdict}")

    if a.out:
        slim = {str(s): {k: v for k, v in data[s].items() if k != "rows"}
                for s in a.steps}
        json.dump({"steps": a.steps, "usable": usable, "voided": voided,
                   "common_n": len(common), "spread": spread, "rising": rising,
                   "verdict": verdict, "agg": slim},
                  open(a.out, "w"), indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
