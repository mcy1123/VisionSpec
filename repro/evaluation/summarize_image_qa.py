#!/usr/bin/env python3
"""Per-arm ChartQA summary with prompt-level paired bootstrap (Task B image QA).

`eval_chartqa.py` stores, per budget arm, BOTH a pooled per-round tau (`tau`) and
the per-prompt tau list (`per_prompt.tau`). The historical headline numbers are
the pooled ones, so both are printed; the paired comparison resamples PROMPTS
(never rounds), because rounds inside a prompt are not independent.

Usage: summarize_image_qa.py --arm warm=<json> --arm sft=<json> ... --out s.json
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
import statistics as st
import sys

import numpy as np

BUDGETS = ["glance_b15", "glance_b31", "glance_b47", "glance_b63"]


def load(path):
    with open(path) as handle:
        return json.load(handle)


def bootstrap_ci(deltas, n_boot=2000, seed=20261013):
    d = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    return (float(np.percentile(d[idx].mean(axis=1), 2.5)),
            float(np.percentile(d[idx].mean(axis=1), 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="tag=path.json")
    ap.add_argument("--budget", default="glance_b63",
                    help="which tree budget arm to summarise (default the reported b63)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    order, docs = [], {}
    for spec in a.arm:
        tag, _, path = spec.partition("=")
        docs[tag] = load(path)
        order.append(tag)

    per_prompt = {t: docs[t]["arms"][a.budget]["per_prompt"]["tau"] for t in order}
    summary = {"budget": a.budget, "arms": {}, "paired": []}
    for t in order:
        arm = docs[t]["arms"][a.budget]
        pp = per_prompt[t]
        summary["arms"][t] = {
            "file": None, "pooled_tau": arm["tau"],
            "pooled_tau_ci95": arm.get("tau_ci95"),
            "mean_per_prompt_tau": st.fmean(pp),
            "median_per_prompt_tau": st.median(pp),
            "n_prompts": len(pp),
            "ms_per_token": arm.get("ms_per_token"),
            "speedup": arm.get("speedup"),
            "tree_size": arm.get("tree_size"),
            "config_head": docs[t]["config"]["head"],
            "config_dtype": docs[t]["config"]["dtype"],
            "config_n": docs[t]["config"]["n"],
        }
    for i, t in enumerate(order):
        for u in order[i + 1:]:
            d = [x - y for x, y in zip(per_prompt[t], per_prompt[u])]
            summary["paired"].append({
                "name": f"{t} - {u}", "n_prompts": len(d),
                "mean_delta_tau": st.fmean(d), "median_delta_tau": st.median(d),
                "positive": sum(1 for x in d if x > 0),
                "negative": sum(1 for x in d if x < 0),
                "ci95": bootstrap_ci(d),
                "t": (st.fmean(d) / (st.stdev(d) / len(d) ** 0.5)) if len(d) > 1 and st.stdev(d) > 0 else None,
                "pooled_delta": (summary["arms"][t]["pooled_tau"]
                                 - summary["arms"][u]["pooled_tau"]),
            })
    with open(a.out, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"budget arm {a.budget}")
    print("arm                 n   pooled_tau  mean_pp  median_pp  ms/tok  speedup")
    for t in order:
        s = summary["arms"][t]
        print(f"{t:18s} {s['n_prompts']:3d}  {s['pooled_tau']:.4f}    "
              f"{s['mean_per_prompt_tau']:.4f}  {s['median_per_prompt_tau']:.4f}   "
              f"{(s['ms_per_token'] or float('nan')):6.2f}  "
              f"{(s['speedup'] or float('nan')):5.2f}")
    print("\npaired (prompt-level bootstrap 95% CI)")
    for p in summary["paired"]:
        t = f"{p['t']:+.2f}" if p["t"] is not None else "n/a"
        print(f"  {p['name']:26s} mean {p['mean_delta_tau']:+.4f} "
              f"CI[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] t {t} "
              f"pos {p['positive']} neg {p['negative']} pooled_delta {p['pooled_delta']:+.4f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
