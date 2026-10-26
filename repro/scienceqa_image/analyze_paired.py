#!/usr/bin/env python3
"""Paired analysis of the ScienceQA arms: image-group bootstrap + every table.

Statistics unit is the **image group**, not the question and not the token: the
official data reuses one picture across several questions, and the frozen
selection deliberately kept those questions together, so treating them as
independent observations would understate the interval. For a paired comparison
the per-group difference of the two arms' means is bootstrapped over groups
(`--boot` draws, `--boot-seed`), which is the paired form the task asks for.

Outputs (all under ``--out-dir``):
  per_question.csv             every evaluated record with all reported columns
  per_image_group.csv          the statistics unit's aggregates
  per_seed_summary.csv         one row per arm/seed/split
  paired_deltas.csv            every comparison with its bootstrap CI
  length_buckets.csv           tau by natural output-length bucket
  subject_breakdown.csv        tau by ScienceQA subject
  spec_equals_ar_summary.csv   the lossless gate, per arm/seed/split
  candidate_and_tree_failures.csv  fixed-anchor coverage probe aggregates
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import pathlib
import re
import sys

import numpy as np

BUCKETS = [(0, 31), (32, 63), (64, 127), (128, 255), (256, 10 ** 9)]


def read_evals(eval_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        name = os.path.basename(p)
        if any(k in name for k in ("spec_equals_ar", "_per_question",
                                   "latency_load", "coverage_", "ar_")):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "rows" not in d or "tag" not in d:
            continue
        d["_path"] = p
        out.append(d)
    return out


def parse_tag(tag):
    """`<arm>[s<seed>[u<updates>]]` -> (arm, seed, updates).

    The update count is part of the identity on purpose: a split is evaluated at
    several checkpoints, so a tag that named only arm and seed would make two
    different checkpoints collide in every lookup.
    """
    m = re.match(r"^(?P<arm>[a-z0-9]+?)(?:s(?P<seed>\d{8}))?(?:u(?P<upd>\d+))?$", tag)
    if not m:
        return tag, None, None
    return (m.group("arm"),
            int(m.group("seed")) if m.group("seed") else None,
            int(m.group("upd")) if m.group("upd") else None)


def group_means(rows, key="tau"):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["image_group_id"]].append(float(r[key]))
    return {k: float(np.mean(v)) for k, v in g.items()}


def boot_ci(values, n, seed):
    a = np.asarray(values, float)
    if a.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n, a.size))
    draws = a[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=20261015)
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    outd = pathlib.Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    evals = read_evals(run / "eval")
    if not evals:
        raise SystemExit("no eval JSONs found")
    by = {}
    for d in evals:
        arm, seed, upd = parse_tag(d["tag"])
        d["_arm"], d["_seed"], d["_updates"] = arm, seed, upd
        by[(d["tag"], d["split"])] = d
    print(f"[stats] {len(evals)} eval files: "
          f"{sorted({(d['tag'], d['split']) for d in evals})}", flush=True)

    # ---- per question
    def bucket(n):
        for lo, hi in BUCKETS:
            if lo <= n <= hi:
                return f"{lo}-{hi if hi < 10**8 else 'cap'}"
        return "unknown"

    allrows = []
    for d in evals:
        for r in d["rows"]:
            # mutate the ORIGINAL record too: the bucket and subject tables below
            # iterate d["rows"], not the copies written to the per-question CSV
            r["length_bucket"] = bucket(r["n_tokens"])
            row = dict(r)
            row["arm"] = d["_arm"]
            row["seed"] = d["_seed"] if d["_seed"] is not None else ""
            allrows.append(row)
    cols = ["tag", "arm", "seed", "split", "pid", "image_group_id", "subject",
            "grade", "n_choices", "gold_letter", "pred_letter", "answer_correct",
            "prompt_len", "n_tokens", "ar_n_tokens", "spec_rounds", "ar_rounds",
            "tau", "tokens_per_round", "tree_size", "ms_per_token_decode_only",
            "ms_per_token_wall", "prefill_ms", "ar_ms_per_token", "spec_total_ms",
            "ar_total_ms", "decode_speedup", "decode_speedup_wall",
            "spec_end_to_end_ms", "ar_end_to_end_ms", "ends_with_eos", "hit_cap",
            "ar_hit_cap", "token_exact", "first_mismatch", "length_bucket"]
    with open(outd / "per_question.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(allrows)

    # ---- per image group
    with open(outd / "per_image_group.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "split", "image_group_id", "subject", "n_questions",
                    "mean_tau", "mean_n_tokens", "mean_decode_ms_per_token",
                    "all_token_exact", "answer_accuracy"])
        for d in evals:
            g = collections.defaultdict(list)
            for r in d["rows"]:
                g[r["image_group_id"]].append(r)
            for gid, v in sorted(g.items()):
                w.writerow([d["tag"], d["split"], gid, v[0]["subject"], len(v),
                            float(np.mean([x["tau"] for x in v])),
                            float(np.mean([x["n_tokens"] for x in v])),
                            float(np.mean([x["ms_per_token_decode_only"] for x in v])),
                            all(x["token_exact"] for x in v),
                            float(np.mean([x["answer_correct"] for x in v]))])

    # ---- per arm/seed/split summary
    with open(outd / "per_seed_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "arm", "seed", "split", "n_records", "n_image_groups",
                    "mean_tau_per_image_group", "mean_tau_per_record",
                    "median_tau", "total_rounds", "tokens_per_round",
                    "mean_tree_size", "ms_per_token_decode_only",
                    "ms_per_token_wall", "prefill_ms", "decode_speedup",
                    "decode_speedup_wall", "spec_end_to_end_ms",
                    "ar_end_to_end_ms",
                    "n_token_exact", "lossless", "n_natural_eos", "n_cap_hits",
                    "answer_accuracy", "timing_contaminated"])
        for d in evals:
            gm = group_means(d["rows"])
            w.writerow([d["tag"], d["_arm"], d["_seed"], d["split"],
                        len(d["rows"]), len(gm), float(np.mean(list(gm.values()))),
                        float(np.mean([r["tau"] for r in d["rows"]])),
                        float(np.median([r["tau"] for r in d["rows"]])),
                        sum(r["spec_rounds"] for r in d["rows"]),
                        float(d["total_tokens_per_round"]),
                        float(d["mean_tree_size"]), float(d["ms_per_token_decode_only"]),
                        float(d["ms_per_token_wall"]), float(d["prefill_ms"]),
                        float(d["mean_decode_speedup"]),
                        float(d.get("mean_decode_speedup_wall", float("nan"))),
                        float(d.get("mean_spec_end_to_end_ms", float("nan"))),
                        float(d.get("mean_ar_end_to_end_ms", float("nan"))),
                        d["n_token_exact"],
                        d["lossless"], d["n_natural_eos"], d["n_cap_hits"],
                        d["answer_accuracy"], d["timing_contaminated"]])

    # ---- paired deltas
    METRICS = ("tau", "n_tokens", "ms_per_token_decode_only", "tokens_per_round",
               "decode_speedup", "decode_speedup_wall", "ms_per_token_wall",
               "spec_end_to_end_ms", "ar_end_to_end_ms")

    def find(split, arm, sd, upd):
        for d in evals:
            if (d["split"] == split and d["_arm"] == arm
                    and d["_seed"] == sd and d["_updates"] == upd):
                return d
        return None

    def pair_row(split, seed, upd, name, x, y, metric):
        gm_x, gm_y = group_means(x["rows"], metric), group_means(y["rows"], metric)
        common_g = sorted(set(gm_x) & set(gm_y))
        if not common_g:
            return None
        diffs = [gm_x[g] - gm_y[g] for g in common_g]
        mean = float(np.mean(diffs))
        ci = boot_ci(diffs, a.boot, a.boot_seed)
        pos = sum(1 for v in diffs if v > 0)
        neg = sum(1 for v in diffs if v < 0)
        return {"split": split, "seed": seed if seed is not None else "shared",
                "updates": upd, "comparison": name, "metric": metric,
                "n_image_groups": len(common_g), "delta": mean,
                "ci95_low": ci[0], "ci95_high": ci[1],
                "crosses_zero": bool(ci[0] <= 0.0 <= ci[1]),
                "groups_positive": pos, "groups_negative": neg,
                "groups_unchanged": len(diffs) - pos - neg,
                "arm_a": x["tag"], "arm_b": y["tag"]}

    pairs = []
    for split in sorted({d["split"] for d in evals}):
        ups = sorted({d["_updates"] for d in evals
                      if d["split"] == split and d["_updates"] is not None})
        seeds = sorted({d["_seed"] for d in evals if d["split"] == split},
                       key=lambda x: (x is None, x))
        ups = ups or [None]
        for upd in ups:
            # I0 is a single zero-update head, so its tag carries no update suffix
            # and its parsed `_updates` is None. Try the loop's update first and
            # then the unsuffixed form, otherwise every -I0 comparison silently
            # disappears from the table.
            i0 = (find(split, "i0", None, upd) or find(split, "i0", None, 0)
                  or find(split, "i0", None, None))
            for seed in seeds:
                sft = find(split, "sft", seed, upd)
                opd = find(split, "opd", seed, upd)
                if sft is None and opd is None:
                    continue
                for metric in METRICS:
                    for name, x, y in (("OPD-SFT", opd, sft), ("OPD-I0", opd, i0),
                                       ("SFT-I0", sft, i0)):
                        if x is None or y is None:
                            continue
                        r = pair_row(split, seed, upd, name, x, y, metric)
                        if r:
                            pairs.append(r)

    if not pairs:
        raise SystemExit("[stats] no pairable evaluations found; refusing to write "
                         "an empty paired-deltas table")
    with open(outd / "paired_deltas.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)

    # ---- length buckets / subject
    for fname, key in (("length_buckets.csv", "length_bucket"),
                       ("subject_breakdown.csv", "subject")):
        with open(outd / fname, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tag", "split", "bucket", "n_questions", "mean_tau",
                        "mean_n_tokens", "ms_per_token_decode_only",
                        "answer_accuracy"])
            for d in evals:
                g = collections.defaultdict(list)
                for r in d["rows"]:
                    g[r[key]].append(r)
                for b, v in sorted(g.items()):
                    w.writerow([d["tag"], d["split"], b, len(v),
                                float(np.mean([x["tau"] for x in v])),
                                float(np.mean([x["n_tokens"] for x in v])),
                                float(np.mean([x["ms_per_token_decode_only"] for x in v])),
                                float(np.mean([x["answer_correct"] for x in v]))])

    # ---- lossless gate summary
    with open(outd / "spec_equals_ar_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "split", "n_records", "n_token_exact", "lossless",
                    "n_mismatch", "budget", "branch_k", "max_new", "eos_stop"])
        for d in evals:
            w.writerow([d["tag"], d["split"], len(d["rows"]), d["n_token_exact"],
                        d["lossless"], len(d["rows"]) - d["n_token_exact"],
                        d["budget"], d["branch_k"], d["max_new"], "immediate"])

    # ---- coverage probe
    cps = sorted(glob.glob(str(run / "eval" / "coverage_*.json")))
    if cps:
        with open(outd / "candidate_and_tree_failures.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tag", "split", "n_anchors", "top1_first", "top8_first",
                        "L_raw", "L_tree", "L_actual", "L_correct_in_actual_path",
                        "tree_gap", "phi", "blockers", "tree_rebuild_checks_passed"])
            for p in cps:
                d = json.load(open(p))
                o = d["overall"]
                w.writerow([d["tag"], d["split"], d["n_anchors"],
                            o.get("top1_first"), o.get("top8_first"),
                            o.get("L_raw"), o.get("L_tree"), o.get("L_actual"),
                            o.get("L_correct_in_actual_path"), o.get("tree_gap"),
                            o.get("phi"), json.dumps(o.get("first_blocker", {})),
                            d["tree_rebuild_checks_passed"]])
        # coverage vs online tau, for the mechanism question
        with open(outd / "coverage_vs_online.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tag", "split_coverage", "n_anchors", "L_raw", "L_tree",
                        "L_actual", "top8_first", "online_split", "mean_tau_online",
                        "mean_tau_online_ci_low", "mean_tau_online_ci_high"])
            for p in cps:
                d = json.load(open(p))
                for split in ("dev", "test"):
                    for dd in evals:
                        if dd["tag"] == d["tag"] and dd["split"] == split:
                            gm = group_means(dd["rows"])
                            ci = boot_ci(list(gm.values()), a.boot, a.boot_seed)
                            o = d["overall"]
                            w.writerow([d["tag"], d["split"], d["n_anchors"],
                                        o.get("L_raw"), o.get("L_tree"),
                                        o.get("L_actual"), o.get("top8_first"),
                                        split, float(np.mean(list(gm.values()))),
                                        ci[0], ci[1]])

    print(f"[stats] wrote {len(os.listdir(outd))} files to {outd}", flush=True)
    for p in pairs:
        if p["metric"] == "tau":
            print(f"   {p['split']:5s} seed {p['seed']}: {p['comparison']} "
                  f"d_tau {p['delta']:+.4f} CI[{p['ci95_low']:+.4f},"
                  f"{p['ci95_high']:+.4f}] groups +{p['groups_positive']}/"
                  f"-{p['groups_negative']}/={p['groups_unchanged']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
