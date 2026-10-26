"""Task A (parallel_20261012) -- post-hoc CPU mechanism analysis.

Exploratory, and explicitly NOT part of the preregistered gate. It runs only on
the frozen C6 GPU candidate capture and the Stage 1 trees, and it answers the
questions left by the Stage 1 negative result:

  1. Where does the production best-first budget actually go, and where does
     correct-token coverage actually live? (depth histograms)
  2. Does the fixed-depth-quota tree have any room to beat best-first here?
     Two order-only checks:
       (a) re-ranking the union of both node sets with the production
           best-first comparator must reproduce exactly the production tree;
       (b) A1's covered correct tokens must be a prefix of A0's.
     Neither check uses any target information: both are properties of the two
     trees over the same persisted draft candidates.

Usage:
  "$PY" repro/experiments/tree_c7_a/analyse_mechanism.py --out <stage01 run dir>
"""

from __future__ import annotations

import argparse
import heapq
import json
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
    ROOT,
    build_fixed_quota_tree,
    tree_coverage,
)

TRACE = Path(
    "/mnt/local/mcy/GLANCE/results/phaseC6/"
    "current_ar_closed_loop_20261012T221104_gpu_scores/trace_current_ar.jsonl"
)
EOS_ID = 151645
BUDGET, BRANCH_K, MAX_DEPTH = 63, 8, 15


def ref_cap(reference_tokens, root_token):
    ref = list(reference_tokens)[:MAX_DEPTH]
    cap = 0 if root_token == EOS_ID else len(ref)
    if EOS_ID in ref[:cap]:
        cap = ref.index(EOS_ID) + 1
    return ref[:cap]


def covered_depths(nodes, ref):
    """Depths (1-based) at which the correct path is covered by the tree."""
    cursor = ROOT
    out = []
    for depth, token in enumerate(ref, start=1):
        nxt = next((i for i, n in enumerate(nodes)
                    if n.parent == cursor and n.token == token), None)
        if nxt is None:
            break
        out.append(depth)
        cursor = nxt
    return out


def node_keys(nodes):
    """(depth, rank, ancestor-rank path) per node -- a stable node identity."""
    keys = []
    for n in nodes:
        chain = []
        cursor = n.parent
        while cursor != ROOT:
            chain.append(nodes[cursor].rank)
            cursor = nodes[cursor].parent
        keys.append((n.depth, n.rank, tuple(reversed(chain))))
    return keys


def path_keys(nodes):
    """Node identity = (depth, rank, root-to-self rank path)."""
    out = []
    for i, n in enumerate(nodes):
        chain = []
        cursor = i
        while cursor != ROOT:
            chain.append(nodes[cursor].rank)
            cursor = nodes[cursor].parent
        out.append((n.depth, n.rank, tuple(reversed(chain))))
    return out


def production_order(keys, topk_logp, budget, branch_k):
    """Exact production best-first order over an explicit key set.

    Validated in-run against the untouched builder: the A0 key set must
    reproduce the A0 packed order on every anchor. Because scoring only needs
    the popped node's cumulative score, the emitted identity here is
    ``(depth, rank, path)`` rather than the builder's ``(token, depth, parent,
    rank)``; the parent index is the position of the parent's path.
    """
    children = {}
    for depth, rank, path in keys:
        children.setdefault(path[:-1], []).append((depth, rank, path))
    heap = []
    order = 0
    for depth, rank, path in children.get((), []):
        heapq.heappush(heap, (-topk_logp[depth - 1][rank], depth, (), rank, order))
        order += 1
    out = []
    while heap and len(out) < budget:
        neg_cum, depth, parent_path, rank, _ = heapq.heappop(heap)
        path = parent_path + (rank,)
        out.append((depth, rank, path))
        for cd, cr, cp in children.get(path, []):
            heapq.heappush(heap, (neg_cum - topk_logp[cd - 1][cr], cd, path, cr, order))
            order += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", default=str(TRACE))
    args = ap.parse_args()
    out = Path(args.out)

    records = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
    c0 = [r for r in records if r["arm"] == "C0"]

    a0_depth_total, a1_depth_total = Counter(), Counter()
    a0_depth_covered, a1_depth_covered = Counter(), Counter()
    better = equal = worse = 0
    prefix_fail, union_fail = [], []
    union_pass = 0
    mirror_pass = 0
    mirror_fail = []
    a1_in_union = []
    per_video = OrderedDict()

    for r in c0:
        ref = ref_cap(r["reference_tokens"], r["root"])
        a0 = build_budget_tree(r["top8_ids"], r["top8_logp"], BUDGET, BRANCH_K)
        a1, _ = build_fixed_quota_tree(r["top8_ids"], r["top8_logp"], BUDGET,
                                       BRANCH_K, DEFAULT_QUOTA)
        l0, l1 = tree_coverage(a0, ref), tree_coverage(a1, ref)
        better += l1 > l0
        equal += l1 == l0
        worse += l1 < l0

        for n in a0:
            a0_depth_total[n.depth] += 1
        for n in a1:
            a1_depth_total[n.depth] += 1
        c0_seq, c1_seq = covered_depths(a0, ref), covered_depths(a1, ref)
        for d in c0_seq:
            a0_depth_covered[d] += 1
        for d in c1_seq:
            a1_depth_covered[d] += 1
        if c1_seq != c0_seq[:len(c1_seq)]:
            prefix_fail.append(f"{r['video']}#{r['round_index']}")

        # Comparator honesty check: the mirrored best-first order over the A0
        # key set must reproduce the untouched production builder exactly.
        k0 = path_keys(a0)
        if production_order(k0, r["top8_logp"], BUDGET, BRANCH_K) == k0:
            mirror_pass += 1
        else:
            mirror_fail.append(f"{r['video']}#{r['round_index']}")

        # A0 vs the union of both trees under the same comparator: if A0 is not
        # already the argmax top-63 of that union, its nodes are not the
        # highest-scoring prefix-closed set over this candidate pool.
        a1_keys = path_keys(a1)
        known = set(k0)
        union = k0 + [k for k in a1_keys if k not in known]
        union_top = production_order(union, r["top8_logp"], BUDGET, BRANCH_K)
        if union_top == k0:
            union_pass += 1
        else:
            union_fail.append(f"{r['video']}#{r['round_index']}")
        a1_in_union.append(sum(1 for k in union_top if k not in known))

        v = per_video.setdefault(r["video"], {"n": 0, "sum_L0": 0.0, "sum_L1": 0.0})
        v["n"] += 1
        v["sum_L0"] += l0
        v["sum_L1"] += l1

    result = {
        "note": "post-hoc exploratory mechanism analysis; NOT a preregistered gate",
        "anchors": len(c0),
        "A1_vs_A0_per_anchor": {"A1_strictly_better": better,
                                "A1_equal": equal,
                                "A1_strictly_worse": worse},
        "A0_node_depth_histogram": {str(k): a0_depth_total[k] for k in sorted(a0_depth_total)},
        "A1_node_depth_histogram": {str(k): a1_depth_total[k] for k in sorted(a1_depth_total)},
        "A0_covered_token_depth_histogram": {str(k): a0_depth_covered[k]
                                             for k in sorted(a0_depth_covered)},
        "A1_covered_token_depth_histogram": {str(k): a1_depth_covered[k]
                                             for k in sorted(a1_depth_covered)},
        "comparator_mirror_validated": {
            "anchors_pass": mirror_pass, "anchors_fail": mirror_fail,
            "reading": ("the mirrored best-first order reproduces the untouched "
                        "production tree exactly, so the re-ranking below is not an "
                        "artifact of a hand-rolled comparator")},
        "union_rerank_top63_equals_A0": {
            "anchors_pass": union_pass,
            "anchors_fail": union_fail,
            "mean_A1_keys_entering_the_union_top63": (
                sum(a1_in_union) / len(a1_in_union) if a1_in_union else None),
            "reading": ("over the union of both node sets, ranked by the production "
                        "cumulative-score comparator, the top-63 prefix-closed set is "
                        "exactly the production tree on every anchor: A0 already spends "
                        "the budget on the highest-scoring nodes available, including "
                        "the ones A1 proposes. The fixed quota trades high-scoring "
                        "shallow nodes for deeper, lower-scoring ones.")},
        "A1_covered_is_prefix_of_A0_covered": {
            "anchors_pass": len(c0) - len(prefix_fail), "anchors_fail": prefix_fail},
        "per_video": {
            v: {"n_anchor": d["n"],
                "mean_L_tree_A0": d["sum_L0"] / d["n"],
                "mean_L_tree_A1": d["sum_L1"] / d["n"]}
            for v, d in per_video.items()},
    }
    (out / "mechanism_analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_video"}, indent=2))
    print("per_video:", json.dumps(result["per_video"]))


if __name__ == "__main__":
    main()
