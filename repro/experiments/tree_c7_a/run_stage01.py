"""Task A (parallel_20261012) -- Stage 0 replay + Stage 1 CPU counterfactual.

Reads ONLY the frozen C6 GPU candidate capture (top8_ids / top8_logp per
anchor) and builds, entirely on the CPU:

  * A0 = unmodified production ``glance.tree.build_budget_tree`` (replay gate)
  * A1 = fixed-depth-quota tree (``fixed_quota_tree.py``)

No model, no GPU, no new head call, no target logits. The preregistration is
read and hashed first; the run refuses to start if its rules changed.

Usage:
  PY=/home/mcy/.conda/envs/fin_research/bin/python
  "$PY" repro/experiments/tree_c7_a/run_stage01.py \
      --out /mnt/local/mcy/GLANCE/results/parallel_A_tree/<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROJECT = HERE.parents[2]
for extra in (str(REPO), str(PROJECT / "GLANCE_repo")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from glance.tree import build_budget_tree  # noqa: E402

from experiments.tree_c7_a.fixed_quota_tree import (  # noqa: E402
    DEFAULT_QUOTA,
    MAX_DRAFT_DEPTH,
    ROOT,
    build_fixed_quota_tree,
    first_uncovered_offset,
    quota_for_depths,
    tree_coverage,
    tree_phi,
)

TRACE = Path(
    "/mnt/local/mcy/GLANCE/results/phaseC6/"
    "current_ar_closed_loop_20261012T221104_gpu_scores/trace_current_ar.jsonl"
)
TRACE_SHA256 = "0ffabf0352f502305f88dff3ba1d913e54ee8463f50ba6fd082f9617bc20340c"
EOS_ID = 151645
BUDGET = 63
BRANCH_K = 8
SEED = 20261012
N_BOOT = 2000
EARLY_MAX_DEPTH = 4
EXPECTED_SUM_GAP = 40
EXPECTED_MEAN_L_TREE = 2.8958333333333335
EXPECTED_ANCHORS = 48


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def as_tuple(nodes):
    return tuple((int(n.token), int(n.depth), int(n.parent), int(n.rank)) for n in nodes)


class Ref:
    """Minimal node view so the shared walk helpers can be reused."""

    __slots__ = ("token", "depth", "parent", "rank")

    def __init__(self, token, depth, parent, rank):
        self.token, self.depth, self.parent, self.rank = token, depth, parent, rank


def dicts_to_nodes(rows):
    return [Ref(r["token"], r["depth"], r["parent"], r["rank"]) for r in rows]


def commit_walk(nodes, argmax_tokens):
    """Production commit walk: follow the target's greedy token through the tree."""
    children = {ROOT: []}
    for i, node in enumerate(nodes):
        children.setdefault(i, [])
        children.setdefault(node.parent, []).append(i)
    cursor = ROOT
    path = []
    while True:
        token = int(argmax_tokens[cursor + 1])
        nxt = next((c for c in children.get(cursor, []) if nodes[c].token == token), None)
        if nxt is None:
            return path, token
        path.append(nxt)
        cursor = nxt


def reference_cap(reference_tokens, root_token, eos_id=EOS_ID):
    """Preregistered L_raw cap rule (mirrors C6 local_oracles)."""
    ref = list(reference_tokens)[:MAX_DRAFT_DEPTH]
    cap = len(ref)
    if root_token == eos_id:
        cap = 0
    if eos_id in ref[:cap]:
        cap = ref.index(eos_id) + 1
    return ref[:cap], cap


def raw_coverage(candidate_ids, ref_capped):
    raw = 0
    for offset, token in enumerate(ref_capped):
        if token not in candidate_ids[offset]:
            break
        raw += 1
    return raw


def early_coverage(nodes, ref_capped, max_depth=EARLY_MAX_DEPTH):
    """Reference tokens at depths 1..max_depth that appear as a root-to-node path."""
    cursor = ROOT
    covered = 0
    for token in ref_capped[:max_depth]:
        nxt = next((i for i, n in enumerate(nodes)
                    if n.parent == cursor and n.token == token), None)
        if nxt is None:
            break
        cursor = nxt
        covered += 1
    return covered


def covered_offsets(nodes, ref_capped):
    cursor = ROOT
    out = []
    for offset, token in enumerate(ref_capped, start=1):
        nxt = next((i for i, n in enumerate(nodes)
                    if n.parent == cursor and n.token == token), None)
        if nxt is None:
            break
        out.append(offset)
        cursor = nxt
    return out


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return [lo, hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", default=str(TRACE))
    ap.add_argument("--expect-trace-sha256", default=TRACE_SHA256)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace = Path(args.trace)
    if (out / "stage01_summary.json").exists():
        raise SystemExit(f"{out} already holds a stage01 run; use a fresh directory")

    # ---- freeze the rules before any A1 quantity is computed -------------
    prereg_path = out / "prereg.json"
    if not prereg_path.exists():
        raise SystemExit(f"prereg.json missing in {out}; refusing to run")
    prereg_sha = sha256_file(prereg_path)
    prereg = json.loads(prereg_path.read_text())
    assert prereg["arms"]["A1"]["quota_per_depth"]["1"] == 5, prereg["arms"]["A1"]
    assert prereg["arms"]["A1"]["quota_per_depth"]["4"] == 4
    assert prereg["stage_1_gate_rules"]["gate_1_net_gain"]["pass_if"] == "> 0"
    assert prereg["statistical_units"]["bootstrap"].startswith("video-level")

    trace_sha = sha256_file(trace)
    if args.expect_trace_sha256 and trace_sha != args.expect_trace_sha256:
        raise SystemExit(
            f"trace hash mismatch: {trace_sha} != {args.expect_trace_sha256}")

    records = [json.loads(line) for line in trace.read_text().splitlines()]
    c0 = [r for r in records if r["arm"] == "C0"]
    if len(c0) != EXPECTED_ANCHORS:
        raise SystemExit(f"expected {EXPECTED_ANCHORS} C0 anchors, found {len(c0)}")

    source_hashes = {}
    for rel in ("GLANCE_repo/glance/tree.py", "GLANCE_repo/glance/decode.py",
                "repro/diagnostics/diagnostic_acl_metrics.py",
                "repro/experiments/tree_c7_a/fixed_quota_tree.py"):
        p = PROJECT / rel
        source_hashes[rel] = sha256_file(p) if p.exists() else None

    # ---- Stage 0: replay A0 -------------------------------------------------
    stage0_checks = OrderedDict()
    failures = []
    rows = []
    a0_mismatch, ltree_mismatch, lraw_mismatch, lactual_mismatch = [], [], [], []
    tree_shape_errors = []

    sample_ids = []
    for r in c0:
        sample_ids.append((r["video"], int(r["round_index"])))

        a0 = build_budget_tree(r["top8_ids"], r["top8_logp"], BUDGET, BRANCH_K)
        if as_tuple(a0) != as_tuple(dicts_to_nodes(r["nodes"])):
            a0_mismatch.append(f"{r['video']}#{r['round_index']}")

        for i, n in enumerate(a0):
            if n.parent >= i or n.parent < ROOT:
                tree_shape_errors.append(f"{r['video']}#{r['round_index']} node{i} parent")
            if n.parent != ROOT and n.depth != a0[n.parent].depth + 1:
                tree_shape_errors.append(f"{r['video']}#{r['round_index']} node{i} depth")
            if n.token != r["top8_ids"][n.depth - 1][n.rank]:
                tree_shape_errors.append(f"{r['video']}#{r['round_index']} node{i} token")
        if len(a0) != BUDGET:
            tree_shape_errors.append(f"{r['video']}#{r['round_index']} len={len(a0)}")

        ref_capped, cap = reference_cap(r["reference_tokens"], r["root"])

        l_tree_a0 = tree_coverage(a0, ref_capped)
        if l_tree_a0 != r["oracle"]["L_tree_oracle"]:
            ltree_mismatch.append(f"{r['video']}#{r['round_index']}")

        l_raw = raw_coverage(r["top8_ids"], ref_capped)
        if l_raw != r["oracle"]["L_raw"]:
            lraw_mismatch.append(f"{r['video']}#{r['round_index']}")

        path, bonus = commit_walk(a0, r["verification_argmax"])
        l_actual = len(path)
        if l_actual != r["oracle"]["L_actual"]:
            lactual_mismatch.append(f"{r['video']}#{r['round_index']}")

        # ---- Stage 1: A1 ---------------------------------------------------
        a1, a1_info = build_fixed_quota_tree(
            r["top8_ids"], r["top8_logp"], BUDGET, BRANCH_K, DEFAULT_QUOTA)
        l_tree_a1 = tree_coverage(a1, ref_capped)

        cov_a0 = covered_offsets(a0, ref_capped)
        cov_a1 = covered_offsets(a1, ref_capped)
        recovered = [o for o in cov_a1 if o not in cov_a0]
        lost = [o for o in cov_a0 if o not in cov_a1]

        phi_a0 = tree_phi(a0, r["top8_logp"])
        phi_a1 = tree_phi(a1, r["top8_logp"])
        early_a0 = early_coverage(a0, ref_capped)
        early_a1 = early_coverage(a1, ref_capped)

        rows.append({
            "video": r["video"],
            "round_index": int(r["round_index"]),
            "pos": int(r["pos"]),
            "root": int(r["root"]),
            "H_valid": int(r["H_valid"]),
            "ref_cap": cap,
            "L_raw": l_raw,
            "L_candidate_oracle": r["oracle"]["L_candidate_oracle"],
            "L_tree_A0": l_tree_a0,
            "L_tree_A1": l_tree_a1,
            "Delta_L_tree": l_tree_a1 - l_tree_a0,
            "L_actual_A0_trace": l_actual,
            "commit_gap_trace": r["commit_gap"],
            "Phi_A0": phi_a0,
            "Phi_A1": phi_a1,
            "Delta_Phi": phi_a1 - phi_a0,
            "first_missing_depth_A1": first_uncovered_offset(a1, ref_capped),
            "recovered_paths": ";".join(str(o) for o in recovered),
            "lost_paths": ";".join(str(o) for o in lost),
            "n_recovered": len(recovered),
            "n_lost": len(lost),
            "early_covered_A0": early_a0,
            "early_covered_A1": early_a1,
            "early_delta": early_a1 - early_a0,
            "nodes_A0": len(a0),
            "nodes_A1": len(a1),
            "leftover_budget_A1": a1_info["leftover_budget"],
            "depth_counts_A1": json.dumps(a1_info["depths"], sort_keys=True),
            "A0_matches_trace": as_tuple(a0) == as_tuple(dicts_to_nodes(r["nodes"])),
            "termination": r["termination"],
            "_top8_ids": r["top8_ids"],
            "_top8_logp": r["top8_logp"],
        })

    sum_gap = sum(row["L_raw"] - row["L_tree_A0"] for row in rows)
    mean_l_tree = sum(row["L_tree_A0"] for row in rows) / len(rows)
    sum_commit_gap = sum(row["commit_gap_trace"] for row in rows)

    stage0_checks["A0_rebuild_equals_trace_nodes"] = not a0_mismatch
    stage0_checks["L_tree_matches_recorded_oracle"] = not ltree_mismatch
    stage0_checks["L_raw_matches_recorded_oracle"] = not lraw_mismatch
    stage0_checks["L_actual_matches_recorded_oracle"] = not lactual_mismatch
    stage0_checks["tree_shape_prefix_closed"] = not tree_shape_errors
    stage0_checks["sum_tree_gap_equals_40"] = (sum_gap == EXPECTED_SUM_GAP)
    stage0_checks["mean_L_tree_equals_2.8958333333333335"] = (
        abs(mean_l_tree - EXPECTED_MEAN_L_TREE) < 1e-12)
    stage0_checks["sum_commit_gap_equals_0"] = (sum_commit_gap == 0)
    stage0_checks["anchor_count_48"] = (len(rows) == EXPECTED_ANCHORS)
    stage0_checks["all_48_A0_trees_have_63_nodes"] = all(
        row["nodes_A0"] == BUDGET for row in rows)
    stage0_pass = all(stage0_checks.values())
    for key, ok in stage0_checks.items():
        if not ok:
            failures.append(f"stage0:{key}")

    # ---- Stage 1 aggregates -------------------------------------------------
    videos = []
    for video in dict.fromkeys(row["video"] for row in rows):
        videos.append(video)
    per_video = []
    for video in videos:
        sub = [row for row in rows if row["video"] == video]
        n_pred = sum(1 for row in sub if row["ref_cap"] > 0)
        mean_d = sum(row["Delta_L_tree"] for row in sub) / len(sub)
        early_net = sum(row["early_covered_A0"] - row["early_covered_A1"] for row in sub)
        threshold = 2.0 * n_pred / 8.0
        per_video.append({
            "video": video,
            "n_anchor": len(sub),
            "n_predictive_anchor": n_pred,
            "mean_L_tree_A0": sum(row["L_tree_A0"] for row in sub) / len(sub),
            "mean_L_tree_A1": sum(row["L_tree_A1"] for row in sub) / len(sub),
            "mean_Delta_L_tree": mean_d,
            "positive_anchors": sum(1 for row in sub if row["Delta_L_tree"] > 0),
            "negative_anchors": sum(1 for row in sub if row["Delta_L_tree"] < 0),
            "changed_anchors": sum(1 for row in sub if row["Delta_L_tree"] != 0),
            "sum_Phi_A0": sum(row["Phi_A0"] for row in sub),
            "sum_Phi_A1": sum(row["Phi_A1"] for row in sub),
            "mean_Phi_A0": sum(row["Phi_A0"] for row in sub) / len(sub),
            "mean_Phi_A1": sum(row["Phi_A1"] for row in sub) / len(sub),
            "mean_delta_Phi": sum(row["Delta_Phi"] for row in sub) / len(sub),
            "early_covered_A0": sum(row["early_covered_A0"] for row in sub),
            "early_covered_A1": sum(row["early_covered_A1"] for row in sub),
            "early_net_loss": early_net,
            "early_threshold": threshold,
            "severe_early_loss": bool(early_net > threshold),
            "n_recovered": sum(row["n_recovered"] for row in sub),
            "n_lost": sum(row["n_lost"] for row in sub),
        })

    total_delta = sum(row["Delta_L_tree"] for row in rows)
    mean_delta = total_delta / len(rows)
    video_deltas = [pv["mean_Delta_L_tree"] for pv in per_video]
    boot = bootstrap_ci(video_deltas)
    boot_abs = bootstrap_ci([sum(r["Delta_L_tree"] for r in rows if r["video"] == v)
                             for v in videos])

    gate_1 = total_delta > 0
    gate_2 = sum(1 for d in video_deltas if d > 0) >= 2
    gate_3 = not any(pv["severe_early_loss"] for pv in per_video)
    gates = {
        "gate_1_net_gain": {"statistic": total_delta, "pass": gate_1},
        "gate_2_at_least_two_positive_videos": {
            "positive_videos": sum(1 for d in video_deltas if d > 0),
            "videos": videos, "per_video_mean_delta": video_deltas, "pass": gate_2},
        "gate_3_no_severe_early_loss": {
            "per_video": [{k: pv[k] for k in
                           ("video", "n_predictive_anchor", "early_covered_A0",
                            "early_covered_A1", "early_net_loss", "early_threshold",
                            "severe_early_loss")} for pv in per_video],
            "pass": gate_3},
        "all_pass": bool(gate_1 and gate_2 and gate_3),
    }

    summary = {
        "stage": "stage0_and_stage1_cpu",
        "prereg_sha256": prereg_sha,
        "trace": str(trace),
        "trace_sha256": trace_sha,
        "source_sha256": source_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "anchors": len(rows),
        "videos": len(videos),
        "budget": BUDGET,
        "branch_k": BRANCH_K,
        "quota": {str(d): q for d, q in sorted(quota_for_depths().items())},
        "stage0_checks": stage0_checks,
        "stage0_pass": stage0_pass,
        "stage0_failures": failures,
        "mean_L_tree_A0": mean_l_tree,
        "sum_tree_gap_A0": sum_gap,
        "sum_commit_gap_trace": sum_commit_gap,
        "A0_total_delta_L_tree": total_delta,
        "A0_mean_delta_L_tree": mean_delta,
        "A0_bootstrap_95ci_video_mean": boot,
        "A0_bootstrap_95ci_video_sum": boot_abs,
        "phi": {
            "sum_Phi_A0": sum(row["Phi_A0"] for row in rows),
            "sum_Phi_A1": sum(row["Phi_A1"] for row in rows),
            "Phi_A0_ge_Phi_A1": sum(row["Phi_A0"] for row in rows)
            >= sum(row["Phi_A1"] for row in rows),
        },
        "nodes": {
            "A1_total_nodes": sum(row["nodes_A1"] for row in rows),
            "A1_leftover_budget": sum(row["leftover_budget_A1"] for row in rows),
            "A1_anchors_short_of_budget": sum(
                1 for row in rows if row["nodes_A1"] < BUDGET),
        },
        "gates": gates,
        "outcome_label": (
            "correctness_or_resource_blocked" if not stage0_pass else
            ("gate_passed_proceed_stage2" if gates["all_pass"] else
             "no_net_local_gain")),
        "per_video": per_video,
    }

    # ---- write artifacts ----------------------------------------------------
    (out / "stage01_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    csv_fields = [k for k in rows[0] if not k.startswith("_")]
    with open(out / "stage1_per_anchor.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in csv_fields})
    with open(out / "stage1_per_video.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_video[0].keys()))
        w.writeheader()
        w.writerows(per_video)
    (out / "stage0_failures.json").write_text(json.dumps({
        "A0_mismatch": a0_mismatch,
        "L_tree_mismatch": ltree_mismatch,
        "L_raw_mismatch": lraw_mismatch,
        "L_actual_mismatch": lactual_mismatch,
        "tree_shape_errors": tree_shape_errors,
    }, indent=2) + "\n")
    tree_lines = []
    for row in rows:
        a1_nodes, a1_info = build_fixed_quota_tree(
            row["_top8_ids"], row["_top8_logp"], BUDGET, BRANCH_K, DEFAULT_QUOTA)
        tree_lines.append(json.dumps({
            "video": row["video"], "round_index": row["round_index"],
            "quota_nodes": [
                {"token": int(n.token), "depth": int(n.depth),
                 "parent": int(n.parent), "rank": int(n.rank)} for n in a1_nodes],
            "depth_counts": a1_info["depths"],
            "leftover_budget": a1_info["leftover_budget"],
        }, sort_keys=True))
    (out / "a1_trees.jsonl").write_text("\n".join(tree_lines) + "\n")

    print(json.dumps({
        "stage0_pass": stage0_pass,
        "stage0_checks": stage0_checks,
        "total_delta_L_tree": total_delta,
        "mean_delta_L_tree": mean_delta,
        "bootstrap_95ci": boot,
        "gates": {"gate_1": gate_1, "gate_2": gate_2, "gate_3": gate_3,
                  "all_pass": gates["all_pass"]},
        "phi": summary["phi"],
        "outcome": summary["outcome_label"],
        "out": str(out),
    }, indent=2))
    if not stage0_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
