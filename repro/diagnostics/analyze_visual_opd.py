#!/usr/bin/env python3
"""Compare a visual-weighted OPD fine-tune against its matched baseline.

Joins per-question tokens-per-round with the visual-dependence map and reports
overall / high-g / low-g effects. This is a reporting script only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def boot_ci(values, n=5000, seed=20261102):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, values.size, values.size)].mean())
             for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def read_tau(path, name):
    df = pd.read_csv(path)
    return df[["pid", "tokens_per_round", "answer_correct"]].rename(
        columns={"tokens_per_round": f"tau_{name}", "answer_correct": f"correct_{name}"})


def block(df, mask):
    sub = df[mask]
    return {
        "n": int(len(sub)),
        "g_mean": float(sub["g_mean"].mean()) if len(sub) else float("nan"),
        "tau_base": float(sub["tau_base"].mean()) if len(sub) else float("nan"),
        "tau_weighted": float(sub["tau_weighted"].mean()) if len(sub) else float("nan"),
        "delta_weighted_minus_base": float(sub["delta"].mean()) if len(sub) else float("nan"),
        "delta_ci95": boot_ci(sub["delta"].tolist()) if len(sub) else [float("nan")] * 2,
        "correct_base": float(sub["correct_base"].mean()) if len(sub) else float("nan"),
        "correct_weighted": float(sub["correct_weighted"].mean()) if len(sub) else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual", required=True, help="test200 visual-dependence CSV")
    ap.add_argument("--base", required=True, help="baseline per-question CSV")
    ap.add_argument("--weighted", required=True, help="weighted per-question CSV")
    ap.add_argument("--orig-sft", default=None)
    ap.add_argument("--orig-opd", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.visual)[["pid", "g_mean", "n_tokens", "subject"]]
    df = df.merge(read_tau(a.base, "base"), on="pid", how="inner")
    df = df.merge(read_tau(a.weighted, "weighted"), on="pid", how="inner")
    if a.orig_sft and a.orig_opd:
        df = df.merge(read_tau(a.orig_sft, "orig_sft"), on="pid", how="left")
        df = df.merge(read_tau(a.orig_opd, "orig_opd"), on="pid", how="left")
        df["delta_orig"] = df["tau_orig_opd"] - df["tau_orig_sft"]
    df["delta"] = df["tau_weighted"] - df["tau_base"]

    med = float(df["g_mean"].median())
    q75 = float(df["g_mean"].quantile(0.75))
    q25 = float(df["g_mean"].quantile(0.25))
    summary = {
        "n": int(len(df)),
        "overall": block(df, np.ones(len(df), dtype=bool)),
        "high_median": block(df, df["g_mean"] >= med),
        "low_median": block(df, df["g_mean"] < med),
        "top_quartile": block(df, df["g_mean"] >= q75),
        "bottom_quartile": block(df, df["g_mean"] <= q25),
        "spearman_g_delta": float(df["g_mean"].corr(df["delta"], method="spearman")),
    }
    if "delta_orig" in df:
        summary["orig_opd_minus_sft"] = float(df["delta_orig"].mean())
        summary["orig_ci95"] = boot_ci(df["delta_orig"].dropna().tolist())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
