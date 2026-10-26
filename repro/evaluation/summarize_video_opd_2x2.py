#!/usr/bin/env python3
"""Summarize the video OPD freshness x topology diagnostic."""

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
import math
import os
import statistics as st


ARMS = ("frozen_chain", "frozen_tree", "online_chain", "online_tree")


def paired_summary(values):
    n = len(values)
    mean = st.fmean(values)
    sd = st.stdev(values)
    se = sd / math.sqrt(n)
    # df=28 for the four-arm intersection; sufficiently close for the n=28
    # warm comparison too (exact t critical there is 2.052).
    crit = 2.048
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95": [mean - crit * se, mean + crit * se],
        "t": mean / se if se else None,
        "wins": sum(x > 0 for x in values),
        "ties": sum(x == 0 for x in values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/local/mcy/GLANCE")
    ap.add_argument("--stamp", default="s200_seed0_r32_a24_v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result_dir = os.path.join(args.root, "results", "opd_video_2x2")
    tau = {}
    arm_summary = {}
    train_summary = {}
    for arm in ARMS:
        path = os.path.join(
            result_dir, f"{arm}_{args.stamp}_fresh30_tree_float32.json")
        blob = json.load(open(path))
        rows = [r for r in blob["rows"] if not r["warmup"]]
        tau[arm] = {r["video"]: float(r["tau"]) for r in rows}
        vals = list(tau[arm].values())
        arm_summary[arm] = {
            "n": len(vals), "mean_tau": st.fmean(vals),
            "median_tau": st.median(vals), "sd_tau": st.stdev(vals),
        }

        log_path = os.path.join(result_dir, f"{arm}_{args.stamp}.jsonl")
        logs = [json.loads(line) for line in open(log_path)]
        n_fwd = sum(x["n_fwd"] for x in logs)
        n_rej = sum(x["n_topk_rej"] for x in logs)
        n_pos = sum(x["positions"] for x in logs)
        n_inexact = sum(x["inexact_frac"] * x["positions"] for x in logs)
        n_anchor = 200 * 24
        train_summary[arm] = {
            "accepted_positions_per_anchor": n_fwd / n_anchor,
            "rejected_positions_per_anchor": n_rej / n_anchor,
            "inexact_teacher_fraction": n_inexact / n_pos,
        }

    keys = set.intersection(*(set(v) for v in tau.values()))
    fc, ft, oc, ot = (tau[a] for a in ARMS)
    contrasts = {
        "online_with_chain": [oc[k] - fc[k] for k in keys],
        "online_with_tree": [ot[k] - ft[k] for k in keys],
        "online_main_effect": [
            ((oc[k] - fc[k]) + (ot[k] - ft[k])) / 2 for k in keys],
        "tree_main_effect": [
            ((ft[k] - fc[k]) + (ot[k] - oc[k])) / 2 for k in keys],
        "interaction": [
            (ot[k] - ft[k]) - (oc[k] - fc[k]) for k in keys],
    }

    warm_path = os.path.join(
        args.root, "results", "opd_fix", "lossless_fp32_fresh30.json")
    warm_blob = json.load(open(warm_path))
    warm = {r["video"]: float(r["tau"]) for r in warm_blob["rows"]
            if not r["warmup"]}
    warm_vals = list(warm.values())
    warm_summary = {
        "n": len(warm_vals), "mean_tau": st.fmean(warm_vals),
        "median_tau": st.median(warm_vals),
        "arm_minus_warm": {},
    }
    for arm in ARMS:
        common = set(tau[arm]) & set(warm)
        warm_summary["arm_minus_warm"][arm] = paired_summary(
            [tau[arm][k] - warm[k] for k in common])

    out = {
        "config": {"stamp": args.stamp, "eval": "fresh30 fp32 tree b63/k8"},
        "arms": arm_summary,
        "training_distribution": train_summary,
        "paired_contrasts": {k: paired_summary(v) for k, v in contrasts.items()},
        "warm_start": warm_summary,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
