#!/usr/bin/env python3
"""CPU tests for the frozen anchor rule and the QA-SFT EOS supervision guard.

These pin the two decision rules that the two arms must share and that no model
call can silently change:

1. `qa_anchors.select_anchors` -- the budget is a pure function of the candidate
   list: at most k positions, evenly spaced, first and last always included.
2. The QA-SFT label guard -- a supervised offset must never label a token at or
   past the first EOS, and the offset that labels the EOS token itself MUST stay
   supervised (predicting EOS is the last thing the drafter has to do).

Run: /home/mcy/.conda/envs/fin_research/bin/python repro/qa_opd_fair/test_qa_anchors.py
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
ROOT = os.path.dirname(REPRO)
for p in (HERE, REPRO, os.path.join(ROOT, "GLANCE_repo"),
          os.path.join(ROOT, "scripts", "dflash_pkg", "dflash-0.1.0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import qa_anchors  # noqa: E402
from historical import train_head as th  # noqa: E402


def check(name, got, want):
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")


def test_select_anchors():
    check("fewer than k keeps all", qa_anchors.select_anchors([3, 4, 5], 6),
          [3, 4, 5])
    check("exactly k keeps all", qa_anchors.select_anchors(list(range(10, 16)), 6),
          list(range(10, 16)))
    got = qa_anchors.select_anchors(list(range(20, 120)), 6)
    check("stride count", len(got), 6)
    check("stride keeps first and last", (got[0], got[-1]), (20, 119))
    check("stride is sorted and unique", (got, sorted(set(got))), (got, got))
    check("empty input", qa_anchors.select_anchors([], 6), [])
    check("duplicates collapse", qa_anchors.select_anchors([1, 1, 2, 2, 3], 6),
          [1, 2, 3])


def test_candidate_positions():
    # prompt_len 100, EOS is answer token 9 (0-based) => last answer token index 9,
    # anchors prompt_len-1 .. prompt_len-1+9.
    pos = qa_anchors.sft_candidate_positions(100, 9, 30)
    check("candidate range", (pos[0], pos[-1], len(pos)), (99, 108, 10))
    # no EOS: last answer token is n_answer-1.
    pos2 = qa_anchors.sft_candidate_positions(100, None, 30)
    check("no-EOS candidate range", (pos2[0], pos2[-1], len(pos2)),
          (99, 128, 30))
    check("zero-length answer", qa_anchors.sft_candidate_positions(100, 0, 0), [99])


def test_eos_guard():
    """The masked loss must drop labels past EOS and keep the EOS label."""
    torch.manual_seed(0)
    seq = torch.arange(1, 41, dtype=torch.long)     # 1..40, no EOS inside
    prompt_len = 10
    eos_pos = 4                                     # answer token index 4 is EOS
    seq = seq.clone()
    seq[prompt_len + eos_pos] = 151645              # the EOS token
    anchors = qa_anchors.sft_candidate_positions(prompt_len, eos_pos, 20)
    reveals = torch.ones(len(anchors), dtype=torch.long)
    from train_qa_sft import qa_blocks
    blocks, targets, super_mask, weights = qa_blocks(
        seq, anchors, reveals, 16, 151669, prompt_len + eos_pos, torch.device("cpu"))
    offsets = torch.arange(16).unsqueeze(0)
    label_pos = torch.tensor(anchors).unsqueeze(1) + offsets
    sup = super_mask.bool()
    # Every supervised label position is at or before the first EOS...
    eos_like = (label_pos <= prompt_len + eos_pos)
    if not bool((sup <= eos_like).all()):
        raise AssertionError("a supervised offset labels a position past EOS")
    # ...and the offset that labels EOS itself IS supervised.
    # Several blocks can see the EOS position (one per anchor whose block spans
    # it), so the count is >= 1 rather than exactly 1; what matters is that the
    # EOS token is supervised at all and that its label is the EOS id.
    at_eos = sup & (label_pos == prompt_len + eos_pos)
    if int(at_eos.sum()) < 1:
        raise AssertionError("the offset labelling the EOS token is not supervised")
    check("EOS labels", sorted(set(targets[at_eos].tolist())), [151645])
    # No supervised label may sit past the first EOS.
    past = (sup & (label_pos > prompt_len + eos_pos)).sum()
    check("no supervision past EOS", int(past), 0)
    # And the EOS id must appear as a supervised label exactly at the EOS
    # positions, never anywhere else.
    where_eos = sup & (targets == 151645)
    check("EOS label only at the EOS position",
          bool((label_pos[where_eos] == prompt_len + eos_pos).all()), True)
    # Weights are zero exactly where the mask is zero (no weight without a label).
    if bool(((weights > 0) & ~sup.any(dim=0).unsqueeze(0).expand_as(weights)).any()):
        raise AssertionError("weight present at an unsupervised slot")


def test_build_blocks_agreement():
    """`qa_blocks` must agree with `train_head.build_blocks` when nothing is cut."""
    seq = torch.arange(1, 41, dtype=torch.long)
    anchors = [10, 15, 20]
    reveals = torch.tensor([1, 5, 9])
    from train_qa_sft import qa_blocks
    b1, t1, s1, w1 = qa_blocks(seq, anchors, reveals, 16, 151669, 10_000,
                               torch.device("cpu"))
    b0, t0, s0, w0 = th.build_blocks(seq, anchors, reveals, 16, 151669,
                                     torch.device("cpu"))
    check("blocks agree", bool(torch.equal(b1, b0)), True)
    check("targets agree", bool(torch.equal(t1, t0)), True)
    check("masks agree", bool(torch.equal(s1, s0)), True)
    check("weights agree", bool(torch.allclose(w1, w0)), True)


def main():
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as err:                    # noqa: BLE001
                failures.append((name, err))
                print(f"FAIL {name}: {err!r}")
    print()
    if failures:
        print(f"CPU: {len(failures)} failed")
        raise SystemExit(1)
    print("CPU: all anchor/guard tests passed")


if __name__ == "__main__":
    main()
