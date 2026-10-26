#!/usr/bin/env python3
"""Aggregate evaluation runs into the paired tables the report needs.

Reads every `eval/*/per_question.csv` under the run and produces:
  * `eval/all_per_video.csv`   -- video-level aggregation per (dataset, format,
    arm, time_mode, stage), the unit every interval is computed on;
  * `eval/paired_main_effects.csv` -- the frozen primary comparison
    (`QA-OPD - QA-SFT`) plus `- C0` and the legacy C1/C2 contrasts, for tau,
    decode ms/token and model wall ms, with a video-level paired bootstrap;
  * `eval/selected_vs_last_checkpoint.csv` -- the update-120 vs selected
    checkpoint contrast, when both are present;
  * `eval/ar_lossless_gates.json` -- the lossless audit across every run.

Statistics: per-video mean first, then a paired bootstrap over the VIDEOS of the
same dataset/format/time mode with a fixed seed; two questions of the same video
are never independent samples.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

METRICS = ["tau", "decode_ms_token", "model_wall_ms", "n_tokens", "rounds",
           "effective_tokens_per_round"]
BOOT = 2000
SEED = 20261013


def load_csvs(R, runs):
    """Rows from EXACTLY the requested run directories.

    Aggregating every `eval/*` directory would silently mix runs that share an
    arm name -- a bf16 timing run and an fp32 dev run both contain `C0`, and a
    dict keyed on the arm would keep whichever it saw last. The caller therefore
    names the runs that belong to one comparison.
    """
    rows = []
    for name in runs:
        path = R / "eval" / name / "per_question.csv"
        if not path.exists():
            raise SystemExit(f"run {name!r} has no per_question.csv at {path}")
        with path.open() as fh:
            for r in csv.DictReader(fh):
                r["_run"] = name
                rows.append(r)
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def video_means(rows, metric):
    """{(dataset, mode, time_mode, arm, stage): {video_id: mean}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        stage = r["_run"].split("_")[0]
        key = (r["dataset"], r["mode"], r["time_mode"], r["arm"], stage)
        out[key][r["video_id"]].append(fnum(r[metric]))
    return {k: {v: float(np.nanmean(x)) for v, x in d.items()}
            for k, d in out.items()}


def paired(a, b, n=BOOT, seed=SEED):
    if set(a) != set(b) or not a:
        return None
    vids = sorted(a)
    d = np.array([a[v] - b[v] for v in vids], dtype=float)
    if not np.isfinite(d).all():
        d = d[np.isfinite(d)]
        if d.size == 0:
            return None
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(len(d), size=(n, len(d)))].mean(1)
    return {"mean": float(d.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "positive": int((d > 0).sum()), "negative": int((d < 0).sum()),
            "unchanged": int((d == 0).sum()), "n_videos": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs", required=True,
                    help="comma list of eval/<name> directories that form ONE "
                         "comparison (e.g. dev_seed1_legacy_b)")
    ap.add_argument("--label", default=None,
                    help="suffix for the output tables, so several comparisons "
                         "can coexist without overwriting each other")
    a = ap.parse_args()
    R = pathlib.Path(a.run)
    E = R / "eval"
    runs = [x.strip() for x in a.runs.split(",") if x.strip()]
    rows = load_csvs(R, runs)
    sfx = f"_{a.label}" if a.label else ""
    if not rows:
        raise SystemExit("no eval/*/per_question.csv found")
    # The stage is the first token of the run directory name (dev / main /
    # confirm / smoke); it is only a label for the report.
    stage_of = {r["_run"]: r["_run"].split("_")[0] for r in rows}

    # ---- per-video table
    with (E / f"all_per_video{sfx}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "stage", "dataset", "mode", "time_mode", "arm",
                    "video_id", "n_qa"] + METRICS)
        groups = collections.defaultdict(list)
        for r in rows:
            groups[(r["_run"], r["dataset"], r["mode"], r["time_mode"], r["arm"],
                    r["video_id"])].append(r)
        for key, rs in sorted(groups.items()):
            w.writerow(list(key) + [len(rs)] +
                       [float(np.nanmean([fnum(x[m]) for x in rs])) for m in METRICS])

    # ---- paired effects
    out_rows = []
    combos = sorted({(r["dataset"], r["mode"], r["time_mode"], stage_of[r["_run"]])
                     for r in rows})
    for ds, mode, tm, stage in combos:
        means = {m: video_means(rows, m) for m in METRICS}
        arms = sorted({r["arm"] for r in rows
                       if r["dataset"] == ds and r["mode"] == mode
                       and r["time_mode"] == tm
                       and stage_of[r["_run"]] == stage})
        for opd in [x for x in arms if x.startswith("QA-OPD")]:
            seed_tag = opd.split("-")[-1]
            sft = f"QA-SFT-{seed_tag}"
            for other, label in ((sft, "QA-OPD-QA-SFT"), ("C0", "QA-OPD-C0")):
                if other not in arms:
                    continue
                for metric in ("tau", "decode_ms_token", "model_wall_ms"):
                    k_opd = (ds, mode, tm, opd, stage)
                    k_oth = (ds, mode, tm, other, stage)
                    if k_opd not in means[metric] or k_oth not in means[metric]:
                        continue
                    p = paired(means[metric][k_opd], means[metric][k_oth])
                    if p is None:
                        continue
                    out_rows.append({"dataset": ds, "format": mode,
                                     "time_mode": tm, "stage": stage,
                                     "metric": metric, "comparison": label,
                                     "arm_a": opd, "arm_b": other, **p})
            if sft in arms and "C0" in arms:
                for metric in ("tau", "decode_ms_token", "model_wall_ms"):
                    p = paired(means[metric][(ds, mode, tm, sft, stage)],
                               means[metric][(ds, mode, tm, "C0", stage)])
                    if p:
                        out_rows.append({"dataset": ds, "format": mode,
                                         "time_mode": tm, "stage": stage,
                                         "metric": metric,
                                         "comparison": "QA-SFT-C0",
                                         "arm_a": sft, "arm_b": "C0", **p})
    with (E / f"paired_main_effects{sfx}.csv").open("w", newline="") as fh:
        cols = ["dataset", "format", "time_mode", "stage", "metric", "comparison",
                "arm_a", "arm_b", "mean", "ci_low", "ci_high", "positive",
                "negative", "unchanged", "n_videos"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # ---- selected vs last checkpoint
    svl = [r for r in out_rows if r["stage"] == "dev"]
    with (E / f"selected_vs_last_checkpoint{sfx}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "format", "time_mode",
                                           "stage", "metric", "comparison",
                                           "arm_a", "arm_b", "mean", "ci_low",
                                           "ci_high", "positive", "negative",
                                           "unchanged", "n_videos"],
                           extrasaction="ignore")
        w.writeheader()
        for r in svl:
            w.writerow(r)

    # ---- lossless audit
    gate = {
        "runs": sorted({r["_run"] for r in rows}),
        "n_question_records": len(rows),
        "n_lossless": sum(1 for r in rows if r["lossless_spec_eq_ar"] == "True"),
        "all_lossless": all(r["lossless_spec_eq_ar"] == "True" for r in rows),
        "timing_contaminated_runs": sorted({r["_run"] for r in rows
                                            if r["timing_contaminated"] == "True"}),
        "by_dataset": {},
    }
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        gate["by_dataset"][ds] = {
            "n": len(sub),
            "lossless": sum(1 for r in sub if r["lossless_spec_eq_ar"] == "True"),
            "arms": sorted({r["arm"] for r in sub}),
        }
    (E / f"ar_lossless_gates{sfx}.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False))
    print(json.dumps({"per_video_rows": len(groups),
                      "paired_rows": len(out_rows),
                      "lossless": gate["all_lossless"],
                      "n_question_records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
