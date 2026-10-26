"""Task A (parallel_20261012) -- tests for the A1 fixed-depth-quota builder.

Run:  PY=/home/mcy/.conda/envs/fin_research/bin/python
      "$PY" -m unittest -v repro.experiments.tree_c7_a.test_fixed_quota_tree
or:   "$PY" repro/experiments/tree_c7_a/test_fixed_quota_tree.py

No model, no GPU. The real-candidate fixture reads the frozen C6 trace (a
read-only artifact) and asserts on it; the remaining tests are synthetic.
"""

from __future__ import annotations

import json
import os
import random
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../repro/experiments/tree_c7_a
REPO = HERE.parents[1]                          # .../repro
PROJECT = HERE.parents[2]                       # project root
for extra in (str(REPO), str(PROJECT), str(PROJECT / "GLANCE_repo")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from glance.tree import build_budget_tree  # noqa: E402

if __package__:
    from .fixed_quota_tree import (  # noqa: E402
        DEFAULT_QUOTA,
        MAX_DRAFT_DEPTH,
        ROOT as ROOT_INDEX,
        build_fixed_quota_tree,
        cumulative_path_logp,
        first_uncovered_offset,
        quota_for_depths,
        tree_coverage,
        tree_phi,
    )
else:  # executed as a plain script
    from fixed_quota_tree import (  # type: ignore # noqa: E402
        DEFAULT_QUOTA,
        MAX_DRAFT_DEPTH,
        ROOT as ROOT_INDEX,
        build_fixed_quota_tree,
        cumulative_path_logp,
        first_uncovered_offset,
        quota_for_depths,
        tree_coverage,
        tree_phi,
    )


TRACE = Path(
    "/mnt/local/mcy/GLANCE/results/phaseC6/"
    "current_ar_closed_loop_20261012T221104_gpu_scores/trace_current_ar.jsonl"
)

TOTAL_QUOTA = 63


def toy(seed=0, depth_max=15, k=8, spread=3.0):
    rng = random.Random(seed)
    ids, logp = [], []
    for _ in range(depth_max):
        row = rng.sample(range(1000, 2000), k)
        vals = sorted((-abs(rng.gauss(0, spread)) for _ in range(k)))
        # strictly increasing negatives, first is the best
        ids.append(row)
        logp.append([float(v) for v in vals])
    return ids, logp


class TestQuotaTable(unittest.TestCase):
    def test_default_quota_shape(self):
        table = quota_for_depths()
        self.assertEqual([table[d] for d in range(1, 4)], [5, 5, 5])
        self.assertEqual([table[d] for d in range(4, 16)], [4] * 12)
        self.assertEqual(sum(table.values()), TOTAL_QUOTA)
        self.assertEqual(len(table), MAX_DRAFT_DEPTH)

    def test_uncovered_depth_is_zero_not_default(self):
        table = quota_for_depths(quota=((1, 2, 3),), max_depth=5)
        self.assertEqual([table[d] for d in range(1, 6)], [3, 3, 0, 0, 0])

    def test_quota_total_is_budget(self):
        # Each spec entry is (lo_depth, hi_depth, nodes_per_depth).
        total = sum((hi - lo + 1) * q for lo, hi, q in DEFAULT_QUOTA)
        self.assertEqual(total, TOTAL_QUOTA)


class TestStructure(unittest.TestCase):
    def test_quota_respected_and_total(self):
        ids, logp = toy(1)
        nodes, info = build_fixed_quota_tree(ids, logp, 63, 8)
        self.assertEqual(len(nodes), 63)
        self.assertEqual(info["leftover_budget"], 0)
        counts = info["depths"]
        for d in range(1, 4):
            self.assertLessEqual(counts[d], 5)
        for d in range(4, 16):
            self.assertLessEqual(counts[d], 4)
        # every depth below has candidates, so quota is exactly saturated
        self.assertEqual([counts[d] for d in range(1, 4)], [5, 5, 5])
        self.assertEqual([counts[d] for d in range(4, 16)], [4] * 12)

    def test_prefix_closed_and_topologically_ordered(self):
        for seed in range(6):
            ids, logp = toy(seed)
            nodes, _ = build_fixed_quota_tree(ids, logp, 63, 8)
            for i, node in enumerate(nodes):
                if node.parent == ROOT_INDEX:
                    self.assertEqual(node.depth, 1)
                else:
                    self.assertLess(node.parent, i, "parent must precede child")
                    self.assertEqual(node.depth, nodes[node.parent].depth + 1)

    def test_candidates_come_from_own_offset_row_and_rank(self):
        ids, logp = toy(2)
        nodes, _ = build_fixed_quota_tree(ids, logp, 63, 8)
        for node in nodes:
            self.assertEqual(node.token, ids[node.depth - 1][node.rank])
            self.assertEqual(node.rank, ids[node.depth - 1].index(node.token))
            self.assertLess(node.rank, 8)

    def test_no_duplicate_node_identity(self):
        ids, logp = toy(3)
        nodes, _ = build_fixed_quota_tree(ids, logp, 63, 8)
        seen = {(n.parent, n.token) for n in nodes}
        self.assertEqual(len(seen), len(nodes), "duplicate (parent, token) node")

    def test_branch_k_limits_used_candidates(self):
        ids, logp = toy(4)
        nodes, info = build_fixed_quota_tree(ids, logp, 63, 3)
        self.assertTrue(all(n.rank < 3 for n in nodes))
        # Depth 1 can only reach branch_k=3, and the missing slots are NOT
        # back-filled from deeper depths, so the tree ends short of budget.
        self.assertEqual(info["depths"][1], 3)
        self.assertLess(len(nodes), 63)
        self.assertEqual(info["leftover_budget"], 63 - len(nodes))

    def test_budget_smaller_than_quota_is_hard_cap(self):
        ids, logp = toy(5)
        nodes, info = build_fixed_quota_tree(ids, logp, 7, 8)
        self.assertEqual(len(nodes), 7)
        self.assertGreaterEqual(info["leftover_budget"], 0)
        # shallow to deep: the first 7 all live at depths 1 and 2
        self.assertEqual(max(n.depth for n in nodes), 2)

    def test_budget_zero_gives_empty_tree(self):
        ids, logp = toy(6)
        nodes, info = build_fixed_quota_tree(ids, logp, 0, 8)
        self.assertEqual(nodes, [])
        self.assertEqual(info["leftover_budget"], 0)
        self.assertEqual(tree_coverage(nodes, [ids[0][0]]), 0)

    def test_zero_quota_depth_blocks_deeper_depths(self):
        ids, logp = toy(7)
        nodes, info = build_fixed_quota_tree(ids, logp, 63, 8, quota=((1, 1, 2),))
        self.assertEqual(len(nodes), 2)
        self.assertEqual(info["depths"][1], 2)
        self.assertEqual(sum(v for k, v in info["depths"].items() if k > 1), 0)
        self.assertTrue(all(n.depth == 1 for n in nodes))

    def test_parent_starved_depth_is_not_backfilled(self):
        # Every depth-1 candidate is unique; keep only one, so depth 2 still
        # has k children and the sweep continues normally.
        ids, logp = toy(8)
        nodes, info = build_fixed_quota_tree(ids, logp, 63, 8, quota=((1, 1, 1), (2, 15, 4)))
        self.assertEqual(info["depths"][1], 1)
        self.assertEqual(info["depths"][2], 4)
        self.assertEqual([info["depths"][d] for d in range(2, 16)], [4] * 14)
        self.assertEqual(len(nodes), 1 + 4 * 14)
        self.assertEqual(info["leftover_budget"], 63 - len(nodes))


class TestRanking(unittest.TestCase):
    def test_within_depth_sorted_by_cumulative_score(self):
        ids, logp = toy(9)
        nodes, info = build_fixed_quota_tree(ids, logp, 63, 8)
        cum = cumulative_path_logp(nodes, logp)
        by_depth = {}
        for i, n in enumerate(nodes):
            by_depth.setdefault(n.depth, []).append(cum[i])
        for depth, scores in by_depth.items():
            self.assertEqual(scores, sorted(scores, reverse=True),
                             f"depth {depth} not chosen in cumulative-score order")

    def test_deterministic_repeat(self):
        ids, logp = toy(10)
        a, ia = build_fixed_quota_tree(ids, logp, 63, 8)
        b, ib = build_fixed_quota_tree(ids, logp, 63, 8)
        self.assertEqual(a, b)
        self.assertEqual(ia, ib)

    def test_tie_break_is_parent_slot_then_rank(self):
        # Two depth-1 parents with identical scores, identical children, and a
        # depth-2 quota of one: the earlier parent slot must win.
        ids = [[10, 11], [20, 21], [30, 31]]
        logp = [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]
        nodes, _ = build_fixed_quota_tree(
            ids, logp, 3, 2, quota=((1, 1, 2), (2, 3, 1)))
        self.assertEqual([(n.token, n.depth, n.parent) for n in nodes],
                         [(10, 1, -1), (11, 1, -1), (20, 2, 0)])

    def test_exhaustive_tiny_case_matches_brute_force(self):
        """Exhaustively check the greedy rule against a brute-force reference.

        Two depths, breadth 2, quota (depth1 <= 2, depth2 <= 2). The frozen
        rule is: keep the best ``q1`` depth-1 nodes by cumulative score, then
        among the children of those kept parents keep the best ``q2`` by
        (cumulative score, parent slot, rank).
        """
        ids = [[10, 11], [20, 21]]
        levels = ([0.0, -1.0], [-1.0, 0.0], [-0.5, -0.5])
        for l0 in levels:
            for l1 in levels:
                logp = [list(l0), list(l1)]
                nodes, _ = build_fixed_quota_tree(
                    ids, logp, 4, 2, quota=((1, 1, 2), (2, 2, 2)))
                order0 = sorted(range(2), key=lambda r: (-logp[0][r], r))
                kept0 = order0[:2]
                slot_of = {p: slot for slot, p in enumerate(kept0)}
                cands = [(-(logp[0][p] + logp[1][r]), slot_of[p], r, p)
                         for p in kept0 for r in (0, 1)]
                cands.sort(key=lambda c: (c[0], c[1], c[2]))
                expect = [(ids[0][r], 1, -1) for r in kept0]
                # parent is an index into the kept list (slot), not the offset row
                expect += [(ids[1][c[2]], 2, c[1]) for c in cands[:2]]
                self.assertEqual(
                    [(n.token, n.depth, n.parent) for n in nodes], expect,
                    msg=f"l0={l0} l1={l1}")


class TestPhiAndCoverage(unittest.TestCase):
    def test_phi_matches_manual_exp_sum(self):
        import math
        ids, logp = toy(11)
        nodes, _ = build_fixed_quota_tree(ids, logp, 63, 8)
        cum = cumulative_path_logp(nodes, logp)
        self.assertAlmostEqual(tree_phi(nodes, logp), sum(math.exp(c) for c in cum))

    def test_phi_of_single_chain_uses_path_probabilities(self):
        # A single chain a -> b -> c: Phi is the sum over the three nodes of
        # exp(cumulative path log-probability), i.e. exp(-1)+exp(-3)+exp(-6).
        import math
        ids = [[5], [6], [7]]
        logp = [[-1.0], [-2.0], [-3.0]]
        nodes, _ = build_fixed_quota_tree(ids, logp, 3, 1, quota=((1, 3, 1),))
        # exp(-1) + exp(-1-2) + exp(-1-2-3)
        expect = math.exp(-1) + math.exp(-3) + math.exp(-6)
        self.assertAlmostEqual(tree_phi(nodes, logp), expect)

    def test_coverage_and_first_uncovered(self):
        ids = [[10, 11], [20, 21], [30, 31]]
        logp = [[-0.1, -5.0], [-0.1, -5.0], [-0.1, -5.0]]
        nodes, _ = build_fixed_quota_tree(ids, logp, 3, 2, quota=((1, 2, 1), (2, 3, 1), (3, 3, 1)))
        self.assertEqual(tree_coverage(nodes, [10, 20, 30]), 3)
        self.assertEqual(tree_coverage(nodes, [10, 21]), 1)
        self.assertEqual(first_uncovered_offset(nodes, [10, 20, 30]), None)
        self.assertEqual(first_uncovered_offset(nodes, [10, 21, 30]), 2)
        self.assertEqual(first_uncovered_offset(nodes, [11]), 1)

    def test_coverage_on_empty_reference(self):
        ids, logp = toy(12)
        nodes, _ = build_fixed_quota_tree(ids, logp, 63, 8)
        self.assertEqual(tree_coverage(nodes, []), 0)
        self.assertIsNone(first_uncovered_offset(nodes, []))


@unittest.skipUnless(TRACE.exists(), f"frozen C6 trace not found: {TRACE}")
class TestFrozenCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in TRACE.read_text().splitlines()]
        cls.c0 = [r for r in cls.rows if r["arm"] == "C0"]
        assert len(cls.c0) == 48, len(cls.c0)

    def test_a0_rebuild_from_persisted_candidates_is_exact(self):
        for r in self.c0:
            nodes = build_budget_tree(r["top8_ids"], r["top8_logp"], 63, 8)
            got = [{"token": int(n.token), "depth": int(n.depth),
                    "parent": int(n.parent), "rank": int(n.rank)} for n in nodes]
            self.assertEqual(got, r["nodes"],
                             msg=f"{r['video']}#{r['round_index']}")

    def test_a1_shape_and_only_original_candidates(self):
        for r in self.c0:
            nodes, info = build_fixed_quota_tree(r["top8_ids"], r["top8_logp"], 63, 8)
            self.assertEqual(len(nodes), 63)
            for n in nodes:
                self.assertIn(n.token, r["top8_ids"][n.depth - 1])
                self.assertEqual(n.token, r["top8_ids"][n.depth - 1][n.rank])

    def test_coverage_matches_c6_L_tree_on_A0(self):
        for r in self.c0:
            nodes = build_budget_tree(r["top8_ids"], r["top8_logp"], 63, 8)
            qnodes = [(n.token, n.depth, n.parent, n.rank) for n in nodes]
            # reuse the same walk through the frozen tree from the trace
            covered = tree_coverage(
                [type("N", (), {"token": t, "depth": d, "parent": p, "rank": k})()
                 for t, d, p, k in qnodes], r["reference_tokens"])
            self.assertEqual(covered, r["oracle"]["L_tree_oracle"],
                             msg=f"{r['video']}#{r['round_index']}")

    def test_a1_uses_same_budget_and_never_exceeds_A0_coverage_inputs(self):
        for r in self.c0:
            nodes, info = build_fixed_quota_tree(r["top8_ids"], r["top8_logp"], 63, 8)
            self.assertEqual(len(nodes), 63)
            self.assertEqual(info["leftover_budget"], 0)
            self.assertLessEqual(max(n.depth for n in nodes), len(r["top8_ids"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
