#!/usr/bin/env python3
"""The frozen anchor-selection rule shared by both QA training arms.

The budget is 4 QA per update and at most 6 anchors per QA (24 per update). The
rule below is a PURE function of the candidate positions: it never reads a loss,
a gradient, a model score or the arm, so it cannot be tuned per arm and it cannot
drift between the two seeds.

Both QA trainers use the same fixed target-AR candidate positions and this
selection rule. These anchors are not necessarily visited by online decoding.
The legacy opd_candidate_positions helper is only a utility for rollout callers.
"""
from __future__ import annotations

import math

MAX_ANCHORS_PER_QA = 6


def select_anchors(positions, k=MAX_ANCHORS_PER_QA):
    """Pick at most `k` positions from `positions`, evenly spaced, order kept.

    `positions` must be sorted ascending and unique. When it has `k` or fewer
    entries, all of them are kept (a short answer legitimately supervises fewer
    than 6 positions; nothing is duplicated to reach the cap). Otherwise the
    indices are the rounded `linspace(0, n-1, k)`, which always includes the
    first and the last candidate, i.e. the last answer-token prediction is
    always supervised.

    A pure integer function so it can be unit tested without a model, and so the
    training log can be re-derived from the schedule alone.
    """
    pos = sorted(set(int(p) for p in positions))
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n = len(pos)
    if n == 0:
        return []
    if n <= k:
        return pos
    picks = sorted({int(round(i)) for i in
                    (j * (n - 1) / (k - 1) for j in range(k))})
    if len(picks) != k:
        # `round` can collide only for tiny n, which the n <= k branch already
        # handled; a collision here would silently shrink the budget, so fail.
        raise RuntimeError(f"stride selection produced {len(picks)} picks from "
                           f"n={n}, k={k}")
    return [pos[i] for i in picks]


def sft_candidate_positions(prompt_len, eos_index_in_answer, n_answer_tokens):
    """Offline anchors available to QA-SFT for one QA.

    Anchor `p` predicts the token at stream position `p + 1`, so the answer
    tokens (the ones after the prompt) are predicted by anchors
    `prompt_len - 1 .. prompt_len - 1 + answer_len - 1`. Nothing past the first
    EOS is a real answer token, so the candidate list stops at the anchor that
    predicts the EOS itself: `prompt_len - 1 + eos_index_in_answer`.

    Returns the full candidate list; `select_anchors` applies the budget.
    """
    last_answer_index = (int(eos_index_in_answer) if eos_index_in_answer is not None
                         else int(n_answer_tokens) - 1)
    if last_answer_index < 0:
        return []
    first = int(prompt_len) - 1
    last = first + last_answer_index
    if last < first:
        return []
    return list(range(first, last + 1))


def opd_candidate_positions(round_positions):
    """Offline view of an online rollout's positions, for logging and tests."""
    return sorted({int(p) for p in round_positions})
