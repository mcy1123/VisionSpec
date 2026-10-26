#!/usr/bin/env python3
"""The convention tests for OPD replay -- the ones that must never be skipped.

Why this file is the most important one in the OPD port. This project already
shipped one convention bug here that *reversed the conclusion*: the training
path fed slot 0 `tok[pos-1]` instead of `tok[pos]` and hard-coded a causal mask
inside the block, while the evaluation path used the correct layout. The SFT arm
was trained under one convention and evaluated under the other and looked 2.4x
worse than it was, which manufactured an "OPD beats SFT" result that did not
exist. The documented lesson (`方法与本轮实验总结_2026-09-23.md` §7) is:

    any implementation where a training path and an evaluation path coexist must
    compare, bitwise, what each feeds the model, as a unit test -- and the
    reference must be the training source of truth, not the other eval path.

So these tests pin, without needing the 8B target or a GPU:

1. `slot 0 = seq[pos]` and the proposals follow it, MASK beyond -- the decode
   layout, which is what makes a replay a replay.
2. The teacher row is `pos + off - 1`, verified with sentinel distributions so
   an off-by-one cannot pass.
3. `found` is reported for tokens outside the stored top-k, and the fallback is
   the remaining-mass upper bound rather than a silent wrong value.
4. Forward KL is zero when student and teacher agree, and positive otherwise.

    python test_opd_convention.py
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import os
import sys

import torch

sys.path.insert(0, str(_REPRO_ROOT))

from historical import opd_loss

MASK = 99


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    if not ok:
        print(f"      got  {got}\n      want {want}")
    return ok


def main():
    ok = True

    # ---------------------------------------------------------------- layout
    # seq: [0,1,2,...,9]; anchor 3; proposals [7,8,9] (deliberately NOT the true
    # tokens 4,5,6 -- a replay must use what the drafter proposed, and if this
    # ever equals the true sequence the test stops discriminating).
    seq = torch.arange(10)
    blocks, super_mask, offsets = opd_loss.opd_blocks(
        seq, [3], [[7, 8, 9]], block_size=6, mask_id=MASK, device="cpu")
    ok &= check("slot 0 is the anchor's own token", blocks[0, 0].item(), 3)
    ok &= check("proposals land in slots 1..len(prop)",
                blocks[0, 1:5].tolist(), [7, 8, 9, MASK])
    ok &= check("nothing beyond the proposal is supervised",
                super_mask[0].tolist(), [0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    ok &= check("offsets are 0..B-1", offsets[0].tolist(), [0, 1, 2, 3, 4, 5])

    # A block that is not filled from the true sequence must differ from it --
    # guards against a rewrite that silently falls back to teacher forcing.
    true_next = seq[4:7].tolist()
    ok &= check("proposals differ from the true continuation (test is meaningful)",
                blocks[0, 1:4].tolist() != true_next, True)

    # ------------------------------------------------------- teacher row rule
    # Sentinel top-k: row r contains token 1000+r with logprob -r, so if the
    # lookup returns the logprob belonging to the wrong row the row rule is off
    # by one and the test fails loudly.
    S, K = 12, 4
    topk_ids = torch.full((S - 1, K), -1, dtype=torch.long)
    topk_val = torch.full((S - 1, K), -30.0)
    for r in range(S - 1):
        topk_ids[r, 0] = 1000 + r
        topk_val[r, 0] = float(-r)

    # offset off at anchor pos must consult row pos+off-1: propose exactly the
    # sentinel token of that row and require the matching logprob back.
    pos, offs = 3, [1, 2, 3]
    rows = torch.tensor([opd_loss.teacher_rows(pos, o) for o in offs])
    toks = torch.tensor([1000 + int(r) for r in rows])
    logp, found = opd_loss.lookup_teacher(topk_ids, topk_val, rows, toks)
    ok &= check("teacher row = pos + off - 1", rows.tolist(), [3, 4, 5])
    ok &= check("sentinel tokens all found", found.tolist(), [True] * 3)
    ok &= check("logp comes from the matching row", logp.tolist(), [-3.0, -4.0, -5.0])

    # A token absent from the stored top-k must set found=False and fall back to
    # the remaining-mass bound, never to a plausible-looking wrong logprob.
    miss_rows = torch.tensor([2])
    miss_logp, miss_found = opd_loss.lookup_teacher(
        topk_ids, topk_val, miss_rows, torch.tensor([424242]))
    ok &= check("absent token reports found=False", miss_found.tolist(), [False])
    expect_bound = (1.0 - torch.tensor([-2.0]).exp()).clamp_min(1e-30).log()
    ok &= check("absent token falls back to the remaining-mass bound",
                bool(torch.allclose(miss_logp, expect_bound, atol=1e-6)), True)

    # ------------------------------------------------------------- KL sanity
    same = torch.log(torch.tensor([0.25, 0.5, 0.1]))
    ok &= check("forward KL is 0 when student == teacher",
                float(opd_loss.bernoulli_forward_kl(same, same).abs().max()) < 1e-6,
                True)
    worse = torch.log(torch.tensor([0.05, 0.05, 0.01]))
    ok &= check("forward KL is positive when the student is worse",
                bool((opd_loss.bernoulli_forward_kl(worse, same) > 0).all()), True)
    ok &= check("k3 reverse KL is 0 when student == teacher",
                float(opd_loss.k3_reverse_kl(same, same).abs().max()) < 1e-5, True)

    # ------------------------------------------------- anchors_from_rounds
    # The accepted offsets must carry the COMMITTED PATH token, which can differ
    # from the per-offset argmax (that is the whole point of a tree); only the
    # rejected tail uses the argmax. Getting this backwards would train the
    # forward KL on tokens the head never proposed.
    from historical import opd_rollout
    rounds = [{
        "pos": 10,
        "path_tokens": [501, 502],          # what the tree committed
        "argmax_tokens": [501, 999, 503, 504, 505],   # offset 2 differs!
        "accepted": 2,
    }]
    anchors = opd_rollout.anchors_from_rounds(rounds, block_size=6)
    ok &= check("anchor pos carried through", anchors[0]["pos"], 10)
    ok &= check("n_acc carried through", anchors[0]["n_acc"], 2)
    ok &= check("accepted offsets use the committed path, rejected use argmax",
                anchors[0]["prop"], [501, 502, 503, 504, 505])
    # If the accepted段 had used the argmax, offset 2 would read 999.
    ok &= check("accepted segment did NOT use the argmax", anchors[0]["prop"][:2] != [501, 999], True)
    # prop is truncated to B-1, since the block has B slots and slot 0 is the root.
    long_round = [{"pos": 0, "path_tokens": [], "argmax_tokens": list(range(100, 120)),
                   "accepted": 0}]
    ok &= check("prop is truncated to block_size - 1",
                len(opd_rollout.anchors_from_rounds(long_round, block_size=6)[0]["prop"]), 5)
    ok &= check("an empty proposal is dropped",
                opd_rollout.anchors_from_rounds(
                    [{"pos": 0, "path_tokens": [], "argmax_tokens": [], "accepted": 0}], 6),
                [])

    print("\nall OPD convention tests passed" if ok else "\nTHERE WERE FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
