#!/usr/bin/env python3
"""CPU tests for `opd_train_video_fullkl.py` (Phase C2 arithmetics and planning).

WHY these tests exist, and what they are NOT. The GPU-side behaviour of the C2
trainer (one cached prefill per video, fork purity, the KL values themselves)
is already pinned by `test_opd_fullkl.py` for the module it imports. What is
new here, and can therefore break, is the trainer's OWN arithmetic and its
ordering/eligibility logic:

  1. UPDATE-LEVEL NORMALISATION -- the paper objective is per block, but the
     trainer must not average per-block means. It scales each microbatch by the
     UPDATE's global `N_acc_total` / `Z_total` so that the per-microbatch losses
     sum to exactly the paper objective. Tested against an independent
     oracle (a float64 literal summation) for anchor sets with mixed accepted
     counts, mixed `gamma`, and all three empty-stream cases.
  2. DETERMINISM -- the anchor budget/order (`select_anchors`, `video_order`) is
     what makes a run reproducible, and it is pure Python, so it is tested
     exactly, including that the order is the ROLLOUT's order and not a set's.
  3. EOS/TRAJECTORY EXCLUSION -- a supervised offset must have a real target
     token at or before the first EOS. `valid` is what carries that into both
     streams and both denominators, so it is checked against the exclusion rule
     and against `opd_fullkl.stream_split`.
  4. DATA GUARD -- the `split` column really does leave 439 train rows on the
     produced `pool_499.parquet`, so the trainer's primary leak guard is
     grounded in the file the run will read (skipped when the pool is absent).

Everything here is CPU-only and model-free. The one thing that genuinely needs a
GPU is the model path (`build_video_trace`, `attach_cropped_prefix`,
`score_microbatch`); it is behind `--gpu-smoke` and is NOT run by this suite.

No bare `try`/`except`: the runner catches `AssertionError` only, so an
unexpected error surfaces with its traceback instead of being scored FAIL.
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
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, str(_REPRO_ROOT))

from historical import opd_fullkl
from historical import opd_train_video_fullkl as T


class _quiet:
    """Silence expected argparse usage text on stderr for one block.

    Used only around calls whose failure IS the assertion; every other failure
    path in this file stays loud, and an unexpected error inside one of these
    blocks is still re-raised by the runner with its traceback.
    """

    def __enter__(self):
        import contextlib
        self._devnull = open(os.devnull, "w")
        self._ctx = contextlib.redirect_stderr(self._devnull)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc_info):
        self._ctx.__exit__(*exc_info)
        self._devnull.close()
        return False


# --------------------------------------------------------------------------
# Small builders: fake anchors with KNOWN counts (no model)
# --------------------------------------------------------------------------

def _anchor(n_acc, K, device=None):
    """A rollout-shaped anchor dict whose proposal is `K` distinct tokens."""
    return {"pos": 0, "prop": list(range(100, 100 + K)), "n_acc": int(n_acc)}


def _build_plan(label, n_acc, *, valid=None, pos=10, total=10_000,
                valid_end=10_000, gamma=0.8, block=T.common.BLOCK_SIZE,
                device="cpu"):
    """One `AnchorPlan` with an EXPLICIT validity mask.

    The mask is what decides which offsets are supervised; `build_anchor_plan`
    turns it into the two offset sets through `opd_fullkl.stream_split`, and the
    tests never re-derive that split from `n_acc` themselves.
    """
    k = block - 1
    valid = [True] * k if valid is None else list(valid)
    return T.build_anchor_plan(
        label, {"pos": pos, "prop": list(range(100, 100 + k)), "n_acc": int(n_acc)},
        total=total, valid_end=valid_end, prompt_len=0, block=block,
        gamma=gamma, device=device, valid=valid)


def fake_plans(specs, *, pos0=10, total=10_000, valid_end=10_000, gamma=0.8,
               block=T.common.BLOCK_SIZE, device="cpu"):
    """`AnchorPlan`s for `(n_acc, valid)` specs, positions spaced out.

    `valid` is an explicit `(block-1)` bool list (`None` means all-True) and is
    handed to `build_anchor_plan` unchanged. Taking the mask rather than an
    invalid-offset count is deliberate: a spec like `(8, 3-invalid)` would ask
    for eight accepted offsets and then mask three of them away, which is a
    contradictory request and the reason the first version of these tests failed.
    Positions are spaced by `block` so failures stay readable.
    """
    plans = []
    for i, (n_acc, valid) in enumerate(specs):
        plans.append(_build_plan(f"v{i}", n_acc, valid=valid,
                                 pos=pos0 + i * block, total=total,
                                 valid_end=valid_end, gamma=gamma, block=block,
                                 device=device))
    return plans


def all_invalid_plans(n_accs, *, gamma=0.8, block=T.common.BLOCK_SIZE):
    """Plans whose valid mask is all-False: empty in BOTH streams.

    Built with an explicit all-False mask rather than a short `valid_end`, so the
    "both streams empty" case is exercised through the same `stream_split` path
    an EOS-truncated block would take.
    """
    return fake_plans([(n, [False] * (block - 1)) for n in n_accs], gamma=gamma,
                      block=block)


def _oracle_streams(plans, gamma):
    """The paper objective's `streams` argument, built from the plans alone.

    Uses `opd_fullkl`'s verified KL helpers on synthetic logits keyed by the
    plan's own offset sets, so the oracle and the trainer consume the same index
    sets and only the AGGREGATION differs -- which is exactly what is under test.
    """
    gen = torch.Generator().manual_seed(1234)
    streams = []
    for i, plan in enumerate(plans):
        K = len(plan.prop)
        p = torch.randn(K, 7, generator=gen) * 1.5
        q = torch.randn(K, 7, generator=gen) * 1.5
        acc = plan.offsets_acc
        rej = plan.offsets_rej
        acc_kl = ([float(x) for x in
                   opd_fullkl.full_kl_forward(p[acc - 1], q[acc - 1])]
                  if acc.numel() else [])
        rej_kl = ([float(x) for x in
                   opd_fullkl.full_kl_reverse(q[rej - 1], p[rej - 1])]
                  if rej.numel() else [])
        rej_w = ([float(w) for w in T.build_offset_weights(K, gamma, rej)]
                 if rej.numel() else [])
        streams.append((acc_kl, int(acc.numel()), rej_kl, rej_w))
    return streams


def _trainer_sum(plans, gamma, *, max_denom_shift=0):
    """Sum the trainer's per-microbatch scaled losses, as a float, and the stats.

    This is what `backward()` accumulates: the trainer adds each `loss_mb` into
    `.grad`, so the gradient of the update objective equals the gradient of the
    SUM of these scalars. `max_denom_shift` perturbs the global denominators
    deliberately, so a test can show they actually enter the arithmetic.

    `per_anchor` also records each anchor's raw accepted/rejected numerators, so a
    test can check the scaling of one microbatch without re-deriving the random
    logits (which a fresh generator seeded inside the test would not reproduce).
    """
    n_acc_total, z_total, acc_present, rej_present = \
        T.plan_update_denominators(plans)
    n_acc_total = int(n_acc_total) + int(max_denom_shift)
    z_total = float(z_total) + float(max_denom_shift)
    gen = torch.Generator().manual_seed(1234)
    total = 0.0
    per_anchor = []
    for plan in plans:
        K = len(plan.prop)
        p = torch.randn(K, 7, generator=gen) * 1.5
        q = torch.randn(K, 7, generator=gen) * 1.5
        acc = plan.offsets_acc
        rej = plan.offsets_rej
        acc_kl = (opd_fullkl.full_kl_forward(p[acc - 1], q[acc - 1])
                  if acc.numel() else q.new_zeros(0))
        rej_kl = (opd_fullkl.full_kl_reverse(q[rej - 1], p[rej - 1])
                  if rej.numel() else q.new_zeros(0))
        rej_w = T.build_offset_weights(K, gamma, rej)
        loss = T.microbatch_scaled_loss(
            acc_kl, rej_kl, rej_w, n_acc_total=n_acc_total, z_total=z_total,
            acc_present=acc_present, rej_present=rej_present)
        anchor_loss = float(loss) if isinstance(loss, torch.Tensor) else float(loss)
        per_anchor.append({
            "loss": anchor_loss,
            "acc_num": float(acc_kl.sum()) if acc_kl.numel() else 0.0,
            "rej_num": float((rej_w * rej_kl).sum()) if rej_kl.numel() else 0.0,
            "n_acc": int(acc.numel()), "n_rej": int(rej.numel()),
        })
        total += anchor_loss
    return total, {"N_acc_total": n_acc_total, "Z_total": z_total,
                   "acc_present": acc_present, "rej_present": rej_present,
                   "per_anchor": per_anchor}


# --------------------------------------------------------------------------
# 1. The normalisation: microbatch scales must sum to the paper objective
# --------------------------------------------------------------------------

def t01_sum_equals_paper_objective():
    """The headline contract, over anchor sets with mixed accepted counts."""
    cases = [
        ("mixed acc counts, equal gammas",
         [(0, None), (3, None), (15, None), (7, None)], 0.8),
        ("mixed counts + rejected suffixes",
         [(0, None), (2, None), (5, None), (9, None)], 0.8),
        ("gamma = 0.5 changes the actual value",
         [(1, None), (4, None), (6, None)], 0.5),
        ("gamma = 1.0 (no decay)", [(0, None), (3, None), (11, None)], 1.0),
        ("one anchor only", [(4, None)], 0.8),
        ("all accepted", [(15, None), (15, None), (15, None)], 0.8),
        ("all rejected", [(0, None), (0, None)], 0.8),
    ]
    for name, specs, gamma in cases:
        plans = fake_plans(specs, gamma=gamma)
        got, info = _trainer_sum(plans, gamma)
        ref = T.paper_objective(_oracle_streams(plans, gamma), gamma)
        # 1e-6 is the float32 quantization floor, not slack: the trainer's KLs
        # and their sums are fp32 (`opd_fullkl` casts internally), the oracle is
        # float64, and V=7 rows make each KL ~1.3, so a sum of 25 of them can
        # differ from float64 by a few 1e-8.
        assert abs(got - ref["loss"]) < 1e-6, (name, got, ref["loss"])
        assert abs(info["N_acc_total"] - ref["n_acc_total"]) == 0, (name, info, ref)
        print(f"      {name:<34} sum {got:.8f} == paper {ref['loss']:.8f} "
              f"(N_acc {info['N_acc_total']}, Z {float(info['Z_total']):.5f})")


def t02_global_denominators_not_per_block():
    """Anti-regression: per-block means MUST disagree with the update objective.

    If the trainer ever goes back to averaging per-anchor means, the sum below
    stops matching the oracle. This test asserts the disagreement is real (so
    t01 cannot pass vacuously) and shows the two numbers.
    """
    specs = [(1, None), (15, None), (2, None)]
    gamma = 0.8
    plans = fake_plans(specs, gamma=gamma)
    got, info = _trainer_sum(plans, gamma)
    # The wrong aggregation: plain mean of per-anchor means, with each anchor
    # normalised by its own offsets.
    wrong = float(np.mean([row["loss"] for row in info["per_anchor"]]))
    ref = T.paper_objective(_oracle_streams(plans, gamma), gamma)
    assert abs(got - ref["loss"]) < 1e-6
    assert abs(wrong - ref["loss"]) > 1e-4, (wrong, ref["loss"])
    # ... and the denominators are NOT the per-anchor ones: perturbing the global
    # counts must move the sum, which is what makes them global.
    shifted, _ = _trainer_sum(plans, gamma, max_denom_shift=1)
    assert abs(shifted - got) > 1e-6, (shifted, got)
    print(f"      update objective {got:.8f} vs mean-of-block-means {wrong:.8f} "
          f"(delta {abs(got - wrong):.2e}); +1 to both denominators moves it to "
          f"{shifted:.8f}")


def t03_empty_stream_renormalisation():
    """One empty stream is DROPPED, not halved; both empty contributes 0."""
    gamma = 0.8
    # -- all accepted: loss == L_acc exactly, NOT L_acc/2
    accept_plans = fake_plans([(15, None), (15, None)], gamma=gamma)
    got, info = _trainer_sum(accept_plans, gamma)
    ref = T.paper_objective(_oracle_streams(accept_plans, gamma), gamma)
    assert info["acc_present"] and not info["rej_present"], info
    assert info["Z_total"] == 0.0 and info["N_acc_total"] == 30, info
    assert abs(got - ref["l_acc"]) < 1e-6, (got, ref)
    assert abs(got - ref["l_acc"] / 2.0) > 1e-6
    # -- all rejected: loss == L_rej exactly
    reject_plans = fake_plans([(0, None), (0, None)], gamma=gamma)
    got_r, info_r = _trainer_sum(reject_plans, gamma)
    ref_r = T.paper_objective(_oracle_streams(reject_plans, gamma), gamma)
    assert info_r["rej_present"] and not info_r["acc_present"], info_r
    assert info_r["N_acc_total"] == 0, info_r
    assert abs(got_r - ref_r["l_rej"]) < 1e-6, (got_r, ref_r)
    assert abs(got_r - ref_r["l_rej"] / 2.0) > 1e-6
    # -- both empty: every plan contributes exactly the float 0.0. A plan is
    # empty in BOTH streams only when the valid mask removed every offset.
    empty_plans = all_invalid_plans([0, 5], gamma=gamma)
    n_acc, z, acc_p, rej_p = T.plan_update_denominators(empty_plans)
    assert (n_acc, z, acc_p, rej_p) == (0, 0.0, False, False), \
        (n_acc, z, acc_p, rej_p)
    losses = []
    for plan in empty_plans:
        assert plan.n_invalid == 15 and plan.n_acc_total == 0
        assert plan.offsets_acc.numel() == 0 and plan.offsets_rej.numel() == 0
        losses.append(T.microbatch_scaled_loss(
            torch.zeros(0), torch.zeros(0), torch.zeros(0), n_acc_total=0,
            z_total=0.0, acc_present=False, rej_present=False))
    assert all(x == 0.0 and not isinstance(x, torch.Tensor) for x in losses), losses
    print(f"      all-accepted {got:.8f} == L_acc (half would be "
          f"{ref['l_acc']/2:.8f}); all-rejected {got_r:.8f} == L_rej; "
          f"both-empty -> float 0.0 (no graph, no step)")


def t04_single_stream_anchor_in_mixed_update():
    """An anchor with only one stream gets the UPDATE's divisor, not its own.

    This is the case the first version of the trainer got wrong. With both
    streams present over the update, every microbatch is divided by 2 -- even an
    all-accepted or all-rejected anchor, whose other term is simply absent. The
    check is exact: an anchor with no rejected offsets must contribute
    `acc_num / N_acc_total / 2`, and one with no accepted offsets
    `rej_num / Z_total / 2`, using the numerators the trainer actually used.
    """
    gamma = 0.8
    specs = [(15, None), (0, None), (3, None)]
    plans = fake_plans(specs, gamma=gamma)
    got, info = _trainer_sum(plans, gamma)
    ref = T.paper_objective(_oracle_streams(plans, gamma), gamma)
    assert info["acc_present"] and info["rej_present"], info
    assert abs(got - ref["loss"]) < 1e-6, (got, ref["loss"])
    per = info["per_anchor"]
    assert per[0]["n_rej"] == 0 and per[1]["n_acc"] == 0, per
    want_acc = per[0]["acc_num"] / info["N_acc_total"] / 2.0
    want_rej = per[1]["rej_num"] / info["Z_total"] / 2.0
    # 1e-7: the loss is a torch fp32 scalar while `want_*` is a python float
    # division, so the two orders of rounding can differ in the last ulp.
    assert abs(per[0]["loss"] - want_acc) < 1e-7, (per[0], want_acc)
    assert abs(per[1]["loss"] - want_rej) < 1e-7, (per[1], want_rej)
    # The wrong (per-microbatch) divisor would be exactly twice these.
    assert abs(per[0]["loss"] * 2.0 - per[0]["acc_num"]
               / info["N_acc_total"]) < 1e-7
    print(f"      N_acc {info['N_acc_total']} Z {float(info['Z_total']):.5f}; "
          f"acc-only anchor {per[0]['loss']:.8f} == "
          f"acc_num/N_acc/2 ({want_acc:.8f}); rej-only {per[1]['loss']:.8f} == "
          f"rej_num/Z/2 ({want_rej:.8f}); sum {got:.8f} == paper "
          f"{ref['loss']:.8f}")


def t05_invalid_offsets_excluded_everywhere():
    """Invalid (EOS/trajectory) offsets leave BOTH streams and BOTH denominators."""
    gamma = 0.8
    block = T.common.BLOCK_SIZE
    # Two anchors: one with the last 3 offsets invalid, one with the last 5.
    masks = [[True] * 12 + [False] * 3, [True] * 10 + [False] * 5]
    specs = [(4, masks[0]), (2, masks[1])]
    plans = fake_plans(specs, gamma=gamma)
    n_acc, z, acc_p, rej_p = T.plan_update_denominators(plans)
    ref = T.paper_objective(_oracle_streams(plans, gamma), gamma)
    assert n_acc == ref["n_acc_total"] == 6, (n_acc, ref)
    # Z is accumulated in float64 on both sides; the residual is the single
    # narrowing of each weight to fp32.
    assert abs(float(z) - ref["z_total"]) < 1e-6, (float(z), ref["z_total"])
    # The rejected denominator is the sum of the SURVIVING offsets' weights, not
    # the geometric series' full sum.
    full = sum(gamma ** (k - 1) for k in range(1, block))
    assert float(z) < full, (float(z), full)
    # Every plan's own Z equals the mask-filtered sum, checked from the rule.
    for plan, (n_acc_i, mask) in zip(plans, specs):
        want = sum(gamma ** (int(k) - 1) for k in plan.offsets_rej)
        # `sum_w_rej` accumulates the fp32-narrowed weights in float64, so its
        # residual against an exact float64 sum is one rounding per weight.
        assert abs(plan.sum_w_rej - want) < 1e-6, (plan.as_log(), want)
        assert plan.valid == mask, plan.as_log()
        assert plan.n_invalid == mask.count(False), plan.as_log()
        assert int(plan.offsets_acc.numel()) == n_acc_i, plan.as_log()
        # Nothing the mask rejected appears in either set.
        assert all(mask[int(k) - 1] for k in plan.offsets_acc), plan.as_log()
        assert all(mask[int(k) - 1] for k in plan.offsets_rej), plan.as_log()
    print(f"      N_acc {n_acc} Z {float(z):.5f} < full-series {full:.5f}; the "
          f"6 masked offsets are absent from both denominators")


# --------------------------------------------------------------------------
# 2. Order and budget determinism
# --------------------------------------------------------------------------

def _rollout(n, block=T.common.BLOCK_SIZE):
    """A chain rollout's anchors: consecutive positions, mixed accept counts."""
    out = []
    pos = 100
    for i in range(n):
        out.append({"pos": pos, "prop": list(range(i, i + block - 1)),
                    "n_acc": i % block})
        pos += 1 + (i % block)
    return out


def t06_anchor_order_is_rollout_order():
    """`select_anchors` keeps the rollout's order and honours the budget."""
    anchors = _rollout(12)
    kept, info = T.select_anchors(anchors, budget=5, max_seq_len=100_000,
                                  prompt_len=10, valid_end=100_000)
    assert len(kept) == 5, kept
    assert [a["pos"] for a in kept] == [a["pos"] for a in anchors[:5]], kept
    assert info["stopped_at_budget"] is True
    assert info["skipped_tail"] == 0 and info["skipped_short"] == 0
    # A per-video cap is a separate bound and does not perturb the order.
    kept2, _ = T.select_anchors(anchors, budget=99, max_seq_len=100_000,
                                prompt_len=10, valid_end=100_000, per_video=3)
    assert [a["pos"] for a in kept2] == [a["pos"] for a in anchors[:3]], kept2
    # The same input twice gives the same output (no set/dict iteration).
    again, _ = T.select_anchors(anchors, budget=5, max_seq_len=100_000,
                                prompt_len=10, valid_end=100_000)
    assert [a["pos"] for a in again] == [a["pos"] for a in kept]
    print(f"      budget kept {[a['pos'] for a in kept]}; per-video cap kept "
          f"{[a['pos'] for a in kept2]}")


def t07_eligibility_tail_and_short_blocks():
    """An anchor without room for a FULL block is skipped, never padded."""
    block = T.common.BLOCK_SIZE
    good = {"pos": 100, "prop": list(range(block - 1)), "n_acc": 2}
    short = {"pos": 200, "prop": list(range(3)), "n_acc": 1}
    late = {"pos": 300, "prop": list(range(block - 1)), "n_acc": 0}
    kept, info = T.select_anchors([good, short, late, good], budget=9,
                                  max_seq_len=100_000, prompt_len=10,
                                  valid_end=300 + 3)   # only `late` runs past EOS
    assert kept == [good, good], kept
    assert info["skipped_short"] == 1, info
    assert info["skipped_tail"] == 1, info
    # `--max-seq-len` is an eligibility bound too: the same anchors under a tight
    # cap lose the ones whose block would fall outside the prefill.
    kept2, info2 = T.select_anchors(
        [{"pos": 100, "prop": list(range(block - 1)), "n_acc": 1}], budget=9,
        max_seq_len=100 + (block - 1), prompt_len=10, valid_end=100_000)
    assert kept2 == [] and info2["skipped_tail"] == 1, (kept2, info2)
    print(f"      kept {len(kept)}, skipped short {info['skipped_short']}, "
          f"skipped tail {info['skipped_tail']}; tight max-seq-len skips "
          f"{info2['skipped_tail']}")


def t08_video_order_determinism():
    """`video_order` is a pure function of `(n, seed, epoch)` and is a permutation."""
    for n in (1, 5, 439):
        a = T.video_order(n, 0, 0)
        b = T.video_order(n, 0, 0)
        assert a == b, (n, a[:5], b[:5])
        assert sorted(a) == list(range(n)), n
        assert T.video_order(n, 1, 0) != a or n < 2, n
        assert T.video_order(n, 0, 1) != a or n < 2, n
    # The fixed order must be stable across calls in one process, which is what
    # makes a resumed run's video sequence reproducible from --seed.
    first = T.video_order(50, 7, 0)[:5]
    T.video_order(50, 7, 1)
    assert T.video_order(50, 7, 0)[:5] == first
    print(f"      permutations stable for n in (1, 5, 439); seed/epoch change "
          f"them; first 5 of n=50, seed 7: {first}")


def t09_valid_mask_matches_exclusion_rule():
    """`anchor_valid_mask` is exactly "inside the trajectory and not past EOS"."""
    block = T.common.BLOCK_SIZE
    K = block - 1
    cases = [
        (100, 1000, 999, None),          # nothing excluded
        (100, 1000, 106, None),          # offsets 7..15 past EOS
        (100, 1000, 100, None),          # only offset 1 is supervised
        (100, 110, 999, None),           # trajectory ends inside the block
        (995, 1000, 999, None),          # anchor near the end
    ]
    for pos, total, valid_end, _ in cases:
        mask = T.anchor_valid_mask(pos, total, valid_end, block)
        want = [pos + k < total and pos + k <= valid_end
                for k in range(1, K + 1)]
        assert mask == want, (pos, total, valid_end, mask, want)
        assert len(mask) == K
        # `opd_fullkl.stream_split` consumes this mask; its two sets must
        # partition the surviving offsets, which is how the exclusion reaches
        # both streams and both denominators at once.
        for n_acc in (0, 1, K // 2, K):
            acc, rej = opd_fullkl.stream_split(K, n_acc, mask)
            assert int(acc.numel()) + int(rej.numel()) == sum(mask), (mask, n_acc)
            assert all(mask[int(k) - 1] for k in acc), (mask, n_acc, acc)
            assert all(mask[int(k) - 1] for k in rej), (mask, n_acc, rej)
    # The counts the trainer logs.
    assert T.eos_excluded_positions(100, 1000, 106, block) == K - 6
    assert T.eos_excluded_positions(100, 1000, 999, block) == 0
    print(f"      5 (pos, total, eos) cases match the rule; stream_split "
          f"partitions the survivors for n_acc in (0, 1, {K//2}, {K})")


def t10_trajectory_valid_end():
    """The first EOS is found, and the EOS token itself is still supervised."""
    ids = torch.tensor([5, 6, 7, 151645, 8, 9], dtype=torch.long)
    end = T.trajectory_valid_end(ids, 151645)
    assert end == 3, end
    # `end` is an INDEX: offsets whose target token sits AT it are supervised,
    # offsets past it are not.
    mask = T.anchor_valid_mask(0, len(ids), end, block=6)
    assert mask == [True, True, True, False, False], mask
    none = T.trajectory_valid_end(torch.tensor([5, 6, 7]), 151645)
    assert none == 2, none
    first = T.trajectory_valid_end(torch.tensor([151645, 1, 2]), 151645)
    assert first == 0, first
    print(f"      EOS at 3 -> valid_end 3, mask {mask}; no EOS -> len-1; "
          f"EOS first -> 0")


def t11_build_anchor_plan_contract():
    """`build_anchor_plan` refuses impossible anchors and counts its offsets."""
    block = T.common.BLOCK_SIZE
    good = T.build_anchor_plan("v", {"pos": 50, "prop": list(range(block - 1)),
                                     "n_acc": 4}, total=1000, valid_end=999,
                               prompt_len=10, block=block, gamma=0.8)
    assert good.n_acc_total == 4 and int(good.offsets_rej.numel()) == block - 5
    assert abs(good.sum_w_rej
               - sum(0.8 ** (k - 1) for k in range(5, block))) < 1e-6
    assert good.n_invalid == 0
    for bad in ({"pos": 50, "prop": list(range(3)), "n_acc": 1},
                {"pos": 50, "prop": list(range(block - 1)), "n_acc": block + 3}):
        raised = False
        try:
            T.build_anchor_plan("v", bad, total=1000, valid_end=999,
                                prompt_len=10, block=block, gamma=0.8)
        except ValueError:
            raised = True
        assert raised, bad
    # `n_acc == K` is legal (an all-accepted block): it must produce an empty
    # rejected set and a Z of exactly 0.
    all_acc = T.build_anchor_plan(
        "v", {"pos": 50, "prop": list(range(block - 1)), "n_acc": block - 1},
        total=1000, valid_end=999, prompt_len=10, block=block, gamma=0.8)
    assert all_acc.offsets_rej.numel() == 0 and all_acc.sum_w_rej == 0.0
    assert all_acc.n_acc_total == block - 1
    # A non-None device would move the index tensors off CPU; the trainer passes
    # the CUDA device and the tests stay on CPU.
    assert good.offsets_acc.device.type == "cpu"
    print(f"      n_acc=4 -> acc {good.n_acc_total}, Z {good.sum_w_rej:.5f}; "
          f"n_acc=K -> Z 0.0; short/over-K anchors raise")


def t12_offset_weights_float64_schedule():
    """`w_k = gamma**(k-1)`, with the schedule evaluated in float64 then narrowed.

    The comparison tolerance is the float32 quantization floor (rtol 1e-6), not
    slack: the returned tensor is the fp32 value the loss multiplies, so it can
    only be as exact as fp32. The float64 claim is checked separately, on the
    module's pre-narrowing value, and against a naive fp32 `**` chain.
    """
    idx = torch.arange(1, 16)
    for gamma in (0.8, 0.5, 1.0, 0.99):
        w = T.build_offset_weights(15, gamma, idx)
        exact = torch.tensor([gamma ** (k - 1) for k in range(1, 16)],
                             dtype=torch.float64)
        assert torch.allclose(w.double(), exact, atol=1e-8, rtol=1e-6), gamma
        assert w[0].item() == 1.0
        if gamma != 1.0:
            assert abs(w[1].item() - gamma) < 1e-7, (gamma, w[1].item())
        # The fp32 narrowing is the ONLY loss of precision: recomputing the
        # schedule in float64 and narrowing once reproduces the tensor exactly.
        narrow_once = (torch.as_tensor(gamma, dtype=torch.float64) **
                       (idx.double() - 1.0)).float()
        assert torch.equal(w, narrow_once), gamma
    # An empty offset set gives an empty weight tensor, which is what makes the
    # all-accepted anchor's `sum_w_rej` exactly 0.0 rather than a NaN.
    empty = T.build_offset_weights(15, 0.8, torch.zeros(0, dtype=torch.long))
    assert empty.numel() == 0 and float(empty.sum()) == 0.0
    # The k-shift is observable: a `gamma**k` schedule would start at gamma.
    assert T.build_offset_weights(15, 0.8, torch.tensor([1])).item() == 1.0
    assert abs(T.build_offset_weights(15, 0.8, torch.tensor([2])).item()
               - 0.8) < 1e-7
    # A DENOMINATOR summed from these weights is accumulated in float64 by the
    # caller (`build_anchor_plan`); against the exact float64 series its residual
    # is one fp32 rounding per weight (each at most ~6e-8 relative), so the
    # comparison is bounded at 1e-5 rather than at the 1e-12 an fp64 sum would
    # allow. What is being pinned is that the residual is tiny and bounded, not
    # that it is zero: the weights reaching the loss are fp32 by design.
    w_all = T.build_offset_weights(15, 0.8, idx)
    exact_sum = float(sum(0.8 ** (k - 1) for k in range(1, 16)))
    exact64_sum = float((torch.as_tensor(0.8, dtype=torch.float64)
                         ** (idx.double() - 1.0)).sum())
    assert abs(exact64_sum - exact_sum) < 1e-12, (exact64_sum, exact_sum)
    assert abs(float(w_all.double().sum()) - exact_sum) < 1e-5
    print(f"      gamma in (0.8, 0.5, 1.0, 0.99): w_1 = 1.0, w_2 = gamma; "
          f"float64-then-narrow reproduces the tensor exactly; "
          f"sum = {float(w_all.double().sum()):.12f}")


# --------------------------------------------------------------------------
# 3. Objective identity against the module's own block loss
# --------------------------------------------------------------------------

def t13_one_anchor_update_equals_block_loss():
    """With ONE anchor, the update objective must equal its per-block loss.

    `opd_fullkl.opd_fullkl_loss` is the verified per-block implementation. A
    single-anchor update has `N_acc_total = |I_acc|` and `Z_total = Z`, so the
    two must agree -- for both streams present and for each single-stream case.
    This is the bridge between this trainer's arithmetic and the tested module.
    """
    gen = torch.Generator().manual_seed(99)
    K, gamma = T.common.BLOCK_SIZE - 1, 0.8
    for n_acc in (0, 1, 4, K - 1, K):
        p = torch.randn(K, 7, generator=gen) * 1.5
        q = torch.randn(K, 7, generator=gen) * 1.5
        plan = T.build_anchor_plan("v", {"pos": 50, "prop": list(range(100, 100 + K)),
                                         "n_acc": n_acc},
                                   total=10_000, valid_end=9999, prompt_len=10,
                                   gamma=gamma, device="cpu")
        n_acc_total, z_total, acc_p, rej_p = T.plan_update_denominators([plan])
        acc_kl = (opd_fullkl.full_kl_forward(p[plan.offsets_acc - 1],
                                             q[plan.offsets_acc - 1])
                  if plan.offsets_acc.numel() else q.new_zeros(0))
        rej_kl = (opd_fullkl.full_kl_reverse(q[plan.offsets_rej - 1],
                                             p[plan.offsets_rej - 1])
                  if plan.offsets_rej.numel() else q.new_zeros(0))
        rej_w = T.build_offset_weights(K, gamma, plan.offsets_rej)
        got = T.microbatch_scaled_loss(
            acc_kl, rej_kl, rej_w, n_acc_total=n_acc_total, z_total=z_total,
            acc_present=acc_p, rej_present=rej_p)
        ref, _m = opd_fullkl.opd_fullkl_loss(p, q, n_acc)
        assert abs(float(got) - float(ref)) < 1e-6, (n_acc, float(got), float(ref))
    print(f"      n_acc in (0, 1, 4, {K-1}, {K}): trainer loss == "
          f"opd_fullkl.opd_fullkl_loss (per-block verified objective)")


# --------------------------------------------------------------------------
# 4. CLI surface (the plan's own contract) and the data guard
# --------------------------------------------------------------------------

def t14_cli_surface():
    """Every documented flag parses, `--steps` is an alias for `--updates`."""
    required = ["--records", "r.pt", "--pool", "p.parquet", "--init-from", "ck",
                "--out", "o"]
    a = T.parse_args(required + ["--updates", "7"])
    assert a.updates == 7
    b = T.parse_args(required + ["--steps", "3"])
    assert b.updates == 3
    c = T.parse_args(required + [
        "--holdout-parquet", "h.parquet", "--anchors-per-update", "24",
        "--anchors-per-video", "2", "--max-rounds", "64", "--lr", "3e-4",
        "--gamma", "0.8", "--seed", "0", "--save-every", "5", "--no-final-save",
        "--log-every", "1", "--log-file", "l.jsonl", "--gpu", "0",
        "--teacher-dtype", "float32", "--head-dtype", "bfloat16",
        "--max-seq-len", "2048", "--clip", "1.0", "--deterministic",
        "--skip-empty-updates"])
    assert c.anchors_per_update == 24 and c.no_final_save and c.deterministic
    assert c.teacher_dtype == "float32" and c.head_dtype == "bfloat16"
    assert c.clip == 1.0 and c.skip_empty_updates
    # The rejected-stream objective switch: default is the paper version, the
    # official k3 is opt-in, and any other name is refused by `choices=`.
    assert c.rejected_kl == T.opd_fullkl.REJECTED_KL_FULL_REVERSE
    e = T.parse_args(required + ["--rejected-kl", "k3"])
    assert e.rejected_kl == "k3"
    # The defaults the task pins.
    d = T.parse_args(required)
    assert d.anchors_per_update == 24, d.anchors_per_update
    assert d.teacher_dtype == "float32" and d.head_dtype == "bfloat16"
    assert d.lr == 3e-4 and d.gamma == 0.8 and d.clip == 1.0
    assert T.K_OFFSETS == 15 and T.common.BLOCK_SIZE == 16
    # `--steps` and `--updates` together is an argparse error, not a silent
    # precedence rule. argparse prints its own usage line on the way out, so the
    # expected stderr is swallowed here to keep the run's output readable.
    with _quiet():
        raised = False
        try:
            T.parse_args(required + ["--updates", "1", "--steps", "2"])
        except SystemExit:
            raised = True
        assert raised, "both --updates and --steps must be rejected"
        for bad in (["--updates", "0"], ["--anchors-per-update", "0"],
                    ["--gamma", "0"], ["--max-seq-len", "4"]):
            raised = False
            try:
                T.parse_args(required + bad)
            except SystemExit:
                raised = True
            assert raised, bad
    print("      all documented flags parse; --steps == --updates; invalid "
          "combinations exit")


def t14b_dry_run_is_cpu_only():
    """`--dry-run` parses, refuses missing inputs, and needs no CUDA.

    It is `main`'s CPU preflight, so it must never reach a model load: the test
    asserts that this box (no CUDA in the test process) can drive the argument
    path and that a missing input raises before anything else happens.
    """
    required = ["--records", "/nonexistent/records.pt", "--pool",
                "/nonexistent/pool.parquet", "--init-from", "/nonexistent/ckpt",
                "--out", "/tmp/c2_never_written", "--dry-run"]
    a = T.parse_args(required)
    assert a.dry_run is True and a.updates == 50
    raised = False
    try:
        T.dry_run(a)
    except FileNotFoundError as exc:
        raised = "records.pt" in str(exc)
    assert raised, "a missing --records must raise FileNotFoundError"
    assert not os.path.exists("/tmp/c2_never_written"), \
        "--dry-run must not create the output directory"
    print("      --dry-run parses; missing inputs raise before any model load")


def t14c_dry_run_on_the_real_pool():
    """`--dry-run` against the real pool/records validates the trained-on rows.

    This is the CPU preflight the parent can run before spending a GPU: it reads
    the same pool, records and split filter `main` reads, and asserts the
    trajectory/prompt contract for all 439 train rows. Skipped when the produced
    data is not on this box, so the suite still runs on a bare checkout.
    """
    records = os.path.join(T.common.DATA, "opd_pool_warm499_aligned.pt")
    pool = os.path.join(T.common.DATA, "pool_499.parquet")
    ckpt = os.path.join(T.common.CKPT, "head_video439_aligned", "step1000")
    for path in (records, pool, os.path.join(ckpt, "config.json")):
        if not os.path.exists(path):
            print(f"      SKIPPED: {path} is not present on this box")
            return
    args = T.parse_args([
        "--records", records, "--pool", pool, "--init-from", ckpt,
        "--out", os.path.join("/tmp", "c2_dry_run_test_only"), "--dry-run",
        "--max-seq-len", str(T.common.MAX_SEQ_LEN)])
    assert args.dry_run
    # The output directory must not be created by a dry run.
    out_dir = args.out
    T.dry_run(args)
    assert not os.path.exists(out_dir), "a dry run created its output directory"
    # The contract `dry_run` asserts, re-stated here so the test would fail if the
    # loader silently stopped checking it.
    blob = torch.load(records, weights_only=False, map_location="cpu")
    import pandas as pd
    df = pd.read_parquet(pool)
    train = set(df.loc[df["split"] == "train", "video_name"].astype(str))
    recs = [r for r in blob["records"] if str(r["task"]) in train]
    assert len(recs) == 439, len(recs)
    lens = [int(r["input_ids"].shape[0]) for r in recs]
    assert max(lens) <= T.common.MAX_SEQ_LEN, max(lens)
    for r in recs:
        assert int(r["input_ids"].shape[0]) >= int(r["prompt_len"]) + T.K_OFFSETS + 1, r["task"]
    # Every train row can host a full block, so no update can be born empty for
    # lack of room (emptiness can still come from EOS/every-offset-invalid, which
    # the log counts).
    print(f"      real pool: {len(recs)} train rows, L {min(lens)}..{max(lens)}, "
          f"all host a {T.K_OFFSETS}-offset block; dry run created no directory")


def t15_pool_split_guard():
    """The guard the trainer relies on: `split == 'train'` is 439 of 499 rows.

    Skipped when the produced pool is not on this box (the tests must still run
    on a CPU-only checkout without the data mount).
    """
    path = os.path.join(T.common.DATA, "pool_499.parquet")
    if not os.path.exists(path):
        print(f"      SKIPPED: {path} is not present on this box")
        return
    import pandas as pd
    pool = pd.read_parquet(path)
    counts = pool["split"].astype(str).value_counts().to_dict()
    assert counts.get("train") == 439, counts
    assert counts.get("heldout") == 60, counts
    assert len(pool) == 499, len(pool)
    # The trainer's filter is `split_of[task] == "train"`; re-deriving the same
    # set here is the assertion that the guard leaves the validation rows out.
    train_names = set(pool.loc[pool["split"] == "train", "video_name"].astype(str))
    held = set(pool.loc[pool["split"] != "train", "video_name"].astype(str))
    assert not (train_names & held), "a video is in both splits"
    print(f"      pool_499.parquet: {counts}; train/heldout disjoint "
          f"({len(train_names)}/{len(held)})")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

CPU_TESTS = [
    ("01 microbatch scales sum to the paper objective", t01_sum_equals_paper_objective),
    ("02 denominators are the UPDATE's, not per block", t02_global_denominators_not_per_block),
    ("03 empty streams: drop-and-renormalise / zero", t03_empty_stream_renormalisation),
    ("04 single-stream anchors inside a mixed update", t04_single_stream_anchor_in_mixed_update),
    ("05 invalid offsets leave both streams and denominators", t05_invalid_offsets_excluded_everywhere),
    ("06 anchor order is rollout order, budget honoured", t06_anchor_order_is_rollout_order),
    ("07 eligibility: short blocks and tails skipped", t07_eligibility_tail_and_short_blocks),
    ("08 video_order determinism", t08_video_order_determinism),
    ("09 valid mask == exclusion rule; stream_split partitions", t09_valid_mask_matches_exclusion_rule),
    ("10 trajectory_valid_end", t10_trajectory_valid_end),
    ("11 build_anchor_plan contract", t11_build_anchor_plan_contract),
    ("12 w_k = gamma**(k-1) schedule", t12_offset_weights_float64_schedule),
    ("13 one-anchor update == opd_fullkl per-block loss", t13_one_anchor_update_equals_block_loss),
    ("14 CLI surface", t14_cli_surface),
    ("14b --dry-run is CPU-only and validates inputs", t14b_dry_run_is_cpu_only),
    ("14c --dry-run on the real pool (data-dependent)", t14c_dry_run_on_the_real_pool),
    ("15 pool split guard (data-dependent)", t15_pool_split_guard),
]


def run_cpu() -> bool:
    print("=" * 78)
    print("C2 trainer CPU tests (V=7, K=15, no model, no CUDA)")
    print("=" * 78)
    passed, failed = 0, []
    for name, fn in CPU_TESTS:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL {name}\n     {exc}")
            failed.append(name)
        else:
            print(f"PASS {name}")
            passed += 1
    print(f"\nCPU: {passed} passed, {len(failed)} failed ({len(CPU_TESTS)} tests)")
    if failed:
        print("failed: " + ", ".join(failed))
    return not failed


def gpu_smoke(args) -> bool:
    """The model-facing path, one video and a few anchors. NOT run by default.

    It is the only thing here that needs CUDA, so it stays behind the flag: it
    builds a real `VideoTrace` (one cached target prefill + a chain rollout),
    checks that `attach_cropped_prefix` does not move the shared cache, that the
    teacher's first row matches the prefill's own anchor row, and that one
    gradient step reaches the head. Gated and unexercised here on purpose --
    another agent owns the GPUs.
    """
    import pandas as pd

    dev = f"cuda:{args.gpu}"
    dtype_of = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    target, _ = T.common.load_target(device=dev, dtype=dtype_of[args.teacher_dtype])
    proc = __import__("transformers").AutoProcessor.from_pretrained(
        T.common.TARGET_PATH)
    head = T.common.load_head(args.head, device=dev,
                              dtype=dtype_of[args.head_dtype])
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    pool = pd.read_parquet(args.rows)
    row = pool.iloc[args.row_index]
    blob = torch.load(args.records, weights_only=False, map_location="cpu")
    rec = [r for r in blob["records"]
           if str(r["task"]) == str(row["video_name"])]
    assert len(rec) == 1, len(rec)
    trace, anchors = T.build_video_trace(
        target, head, proc, rec[0], row, device=dev,
        max_seq_len=args.max_seq_len, max_rounds=4, rollout_mode="legacy-reference")
    print(f"[smoke] {trace.task}: total {int(trace.full_ids.shape[0])} "
          f"valid_end {trace.valid_end} anchors {len(anchors)} "
          f"held {trace.bytes_held()/2**30:.3f} GiB")
    kept, info = T.select_anchors(anchors, budget=args.anchors,
                                  max_seq_len=args.max_seq_len,
                                  prompt_len=trace.prompt_len,
                                  valid_end=trace.valid_end)
    assert kept, info
    plans = [T.build_anchor_plan(trace.task, a, total=int(trace.full_ids.shape[0]),
                                 valid_end=trace.valid_end,
                                 prompt_len=trace.prompt_len, gamma=args.gamma,
                                 device=dev) for a in kept]
    denoms = T.plan_update_denominators(plans)
    print(f"[smoke] denoms {denoms}")
    total = 0.0
    for plan in plans:
        loss, stats = T.score_microbatch(
            head, target, trace, plan, gamma=args.gamma, denoms=denoms,
            head_dtype=head.parameters().__next__().dtype,
            teacher_dtype=next(target.parameters()).dtype)
        assert isinstance(loss, torch.Tensor), stats
        loss.backward()
        total += float(loss.detach())
        del loss
    norm, scale = T.clip_and_step(
        torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                          lr=0.0),
        [p for p in head.parameters() if p.requires_grad], args.clip)
    print(f"[smoke] summed loss {total:.6f} |g| {norm:.4e} clip scale {scale} "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    assert math.isfinite(total) and math.isfinite(norm) and norm > 0.0
    print("GPU SMOKE PASS (C2 trainer model path)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--gpu-smoke", action="store_true",
                    help="run the model-facing path; needs a GPU and is NOT run "
                         "by the CPU suite")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--head",
                    default=f"{T.common.CKPT}/head_video439_aligned/step1000")
    ap.add_argument("--rows", default=f"{T.common.DATA}/pool_499.parquet")
    ap.add_argument("--records",
                    default=f"{T.common.DATA}/opd_pool_warm499_aligned.pt")
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument("--anchors", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-seq-len", type=int, default=T.common.MAX_SEQ_LEN)
    ap.add_argument("--teacher-dtype", default="float32",
                    choices=["float32", "bfloat16"])
    ap.add_argument("--head-dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    args = ap.parse_args()

    ok = run_cpu() if not args.skip_cpu else True
    if args.gpu_smoke:
        ok = gpu_smoke(args) and ok
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
