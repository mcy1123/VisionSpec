"""Task A (parallel_20261012) -- A1 fixed-depth-quota tree builder.

Independent experiment module. It never touches the production builder in
``glance.tree`` / ``glance.decode``; callers import this module and use its
function explicitly. Nothing here needs a GPU, a model or torch.

Frozen strategy (see ``prereg.json`` in the run directory, written before any
A1 result was computed):

    A0 baseline = production ``glance.tree.build_budget_tree``: best-first on
    cumulative path log-probability, budget 63 nodes excluding the root,
    branch factor 8, at most 15 draft offsets (block size 16).

    A1 = fixed depth quota: depth 1..3 keep at most 5 nodes each, depth 4..15
    keep at most 4 nodes each (3*5 + 12*4 = 63). Depths are filled from
    shallow to deep; a depth may only extend parents already kept at the
    previous depth; within a depth candidates are ranked by the SAME
    cumulative path log-probability A0 uses, ties broken deterministically by
    (parent slot, rank). Unused quota is never back-filled from deeper depths.

Only the original draft candidate ids, their per-offset log-probabilities, the
depth and the prefix are used. No target logits, no correct token, no oracle
path, no extra head call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

__all__ = [
    "QuotaNode",
    "DEFAULT_QUOTA",
    "MAX_DRAFT_DEPTH",
    "quota_for_depths",
    "build_fixed_quota_tree",
    "cumulative_path_logp",
    "tree_coverage",
    "first_uncovered_offset",
    "tree_phi",
]

# Depth 1..3 -> 5 nodes, depth 4..15 -> 4 nodes; 3*5 + 12*4 = 63.
DEFAULT_QUOTA = ((1, 3, 5), (4, 15, 4))
MAX_DRAFT_DEPTH = 15
ROOT = -1


@dataclass(frozen=True)
class QuotaNode:
    """One kept candidate token, mirroring ``glance.tree.TreeNode`` fields."""

    token: int
    depth: int
    parent: int  # index into the kept node list, -1 = child of the root
    rank: int  # rank among the top-k ids at this node's own depth


def quota_for_depths(quota=DEFAULT_QUOTA, max_depth: int = MAX_DRAFT_DEPTH):
    """Return ``{depth: quota}`` for every depth in 1..max_depth.

    A depth not covered by any spec entry gets quota 0, which is recorded
    rather than silently defaulted to something else.
    """
    table = {d: 0 for d in range(1, max_depth + 1)}
    for lo, hi, q in quota:
        for d in range(max(1, lo), min(max_depth, hi) + 1):
            table[d] = q
    return table


def build_fixed_quota_tree(topk_ids: Sequence[Sequence[int]],
                           topk_logp: Sequence[Sequence[float]],
                           budget: int = 63,
                           branch_k: int | None = None,
                           quota=DEFAULT_QUOTA):
    """Build the A1 fixed-depth-quota tree.

    Args:
        topk_ids: ``(D, K)`` candidate ids per draft offset, offset 1 first.
        topk_logp: ``(D, K)`` matching per-offset log-probabilities.
        budget: hard node cap, root excluded (production value 63).
        branch_k: how many of the K per-offset candidates may be used.
        quota: ``((lo, hi, q), ...)`` depth quota spec.

    Returns:
        ``(nodes, info)``. ``nodes`` is a list of :class:`QuotaNode` in packed
        order (every parent before its children, so it is directly usable as a
        prefix-closed packed tree). ``info`` records the per-depth node counts,
        the per-depth quotas actually applied, and the leftover budget.

    The fill is a single shallow-to-deep sweep. Every candidate is one that
    extends an already-kept parent, so the result is always prefix closed. If
    a depth has fewer legal candidates than its quota, it keeps what exists
    and the missing slots are lost -- never filled from a deeper depth.
    """
    ids = [list(row) for row in topk_ids]
    logp = [[float(x) for x in row] for row in topk_logp]
    if len(ids) != len(logp):
        raise ValueError("topk_ids and topk_logp must have the same length")
    depth_max = len(ids)
    if depth_max == 0:
        return [], {"depths": {}, "quota_table": {}, "leftover_budget": budget}
    width = len(ids[0])
    if any(len(row) != width for row in ids):
        raise ValueError("ragged per-offset candidate rows")
    if any(len(row) != width for row in logp):
        raise ValueError("ragged per-offset log-probability rows")
    k = width if branch_k is None else min(branch_k, width)
    if k < 1:
        raise ValueError("branch_k must select at least one candidate")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    quota_table = quota_for_depths(quota, max_depth=depth_max)
    nodes: List[QuotaNode] = []
    cum: List[float] = []
    counts: dict[int, int] = {}
    kept_at_prev_depth = [ROOT]

    for depth in range(1, depth_max + 1):
        room = min(quota_table.get(depth, 0), budget - len(nodes))
        if room <= 0:
            counts[depth] = 0
            kept_at_prev_depth = []
            continue
        candidates = []
        for slot, parent in enumerate(kept_at_prev_depth):
            base = 0.0 if parent == ROOT else cum[parent]
            for rank in range(k):
                candidates.append((-(base + logp[depth - 1][rank]),
                                   slot, rank, parent))
        if not candidates:
            counts[depth] = 0
            kept_at_prev_depth = []
            continue
        # Primary key: the same cumulative path log-probability A0 ranks by
        # (so ascending negated score). Ties: (parent slot, rank).
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        kept_here = []
        for neg_score, _, rank, parent in candidates[:room]:
            idx = len(nodes)
            nodes.append(QuotaNode(int(ids[depth - 1][rank]), depth, parent, rank))
            cum.append(-neg_score)
            kept_here.append(idx)
        counts[depth] = len(kept_here)
        kept_at_prev_depth = kept_here

    info = {
        "depths": counts,
        "quota_table": {d: quota_table.get(d, 0) for d in range(1, depth_max + 1)},
        "leftover_budget": budget - len(nodes),
        "branch_k": k,
        "depth_max": depth_max,
    }
    return nodes, info


def cumulative_path_logp(nodes, topk_logp) -> List[float]:
    """Cumulative path log-probability per node, in packed order.

    ``cum[i] = sum of the per-offset marginal log-probabilities along the root
    to node path``. Parents always precede children in the packed order, so a
    single pass is enough.
    """
    logp = [[float(x) for x in row] for row in topk_logp]
    cum: List[float] = []
    for node in nodes:
        base = 0.0 if node.parent == ROOT else cum[node.parent]
        cum.append(base + logp[node.depth - 1][node.rank])
    return cum


def tree_phi(nodes, topk_logp) -> float:
    """``Phi(T) = sum_{node in T} exp(cumulative path log-probability)``."""
    return float(sum(_safe_exp(c) for c in cumulative_path_logp(nodes, topk_logp)))


def _safe_exp(x: float) -> float:
    import math
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def tree_coverage(nodes, reference_tokens: Sequence[int]) -> int:
    """Length of the longest reference prefix present as a root-to-node path.

    This is the tree-side oracle ``L_tree``: it equals the number of draft
    offsets the tree covers along the correct AR continuation, root excluded.
    """
    cursor = ROOT
    covered = 0
    for token in reference_tokens:
        nxt = next((i for i, n in enumerate(nodes)
                    if n.parent == cursor and n.token == token), None)
        if nxt is None:
            break
        cursor = nxt
        covered += 1
    return covered


def first_uncovered_offset(nodes, reference_tokens: Sequence[int]) -> int | None:
    """1-based offset of the first reference token the tree does not cover.

    ``None`` when the whole provided reference is covered.
    """
    cursor = ROOT
    for offset, token in enumerate(reference_tokens, start=1):
        nxt = next((i for i, n in enumerate(nodes)
                    if n.parent == cursor and n.token == token), None)
        if nxt is None:
            return offset
        cursor = nxt
    return None
