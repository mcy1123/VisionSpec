#!/usr/bin/env python3
"""Does the `steer` hook reach the candidate tree, and is it inert when unused?

Three things have to be true for Arm M to be measurable at all, and none is
obvious from reading the diff in `glance/decode.py`:

1. **Inert when absent.** `steer=None` must reproduce the published
   block-diffusion decode bit for bit, or every baseline number on record stops
   being comparable. Checked three ways: `None`, an identity callable, and a
   Markov module whose output projection is zeroed (a zero bias must change
   nothing). All three must agree on committed ids *and* on the full
   accepted-length schedule.

2. **Effective when present.** The bias has to change the *tree*, not merely be
   loaded. A Markov head with a scaled output projection must change the
   accepted-length schedule. The default init is deliberately tiny (std 0.01
   against logits of order 1), so scaling is what makes this a test of the
   wiring rather than of an initialisation accident -- the same reasoning as
   `test_markov.test_steering_is_path_dependent`.

3. **Still lossless.** The output must remain the target's greedy sequence no
   matter what the drafter proposes, including under a deliberately bad steer.
   This is a property of the decode, not of the steer, and it is worth pinning
   because it is what "lossless" has to mean here.

A note on what to compare. The obvious assertion -- "committed ids differ" -- is
*wrong*, and the first version of this test asserted it and failed while the
mechanism was in fact working (tau moved 4.4000 -> 4.3000 with identical ids).
The target picks the committed token along its own argmax, so a different draft
tree changes only *how many* tokens each round accepts, never *which* tokens are
committed. The schedule is therefore the sensitive instrument, and identity of
the ids is the losslessness check rather than a sign of failure.

    python test_steer_hook.py --gpu 2
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
import json
import os
import sys

import torch

from shared import common
from shared import markov as markov_mod

sys.path.insert(0, os.path.join(common.WORKSPACE, "GLANCE_repo"))

import glance
from glance.vl import VisionPositions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=os.path.join(common.CKPT, "head"))
    ap.add_argument("--data", default=os.path.join(common.DATA, "vl_eval_plain"))
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--prompts", type=int, default=2)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    target, processor = common.load_target(device=device)
    head = common.load_head(args.head, device=device).eval()

    rows = []
    with open(os.path.join(args.data, "chartqa.jsonl")) as fh:
        for line in fh:
            candidate = json.loads(line)
            if os.path.exists(os.path.join(args.data, candidate["image"])):
                rows.append(candidate)
            if len(rows) >= args.prompts:
                break
    if not rows:
        raise SystemExit("no usable ChartQA row")

    stop = [processor.tokenizer.eos_token_id]
    vocab = int(target.get_input_embeddings().weight.shape[0])

    def run_one(row, steer):
        image = common.load_image(os.path.join(args.data, row["image"]), 896)
        ids, pix, grid = common.encode(processor, row["prompt"], image)
        ids = ids.to(device)
        pos = VisionPositions(pix.to(device) if pix is not None else None,
                              grid.to(device) if grid is not None else None)
        rounds = []
        out, stats = glance.generate(
            head, target, ids, args.max_new, stop, budget=args.budget, branch_k=8,
            temperature=0.0, positions=pos, return_stats=True,
            timing="decode_only", steer=steer, on_round=rounds.append)
        return out[0, ids.shape[1]:].tolist(), stats, [r["accepted"] for r in rounds]

    arms = {}
    for name, steer in [
            ("None", None),
            ("identity", lambda logits, root: logits)]:
        arms[name] = [run_one(r, steer) for r in rows]

    zero = markov_mod.MarkovHead(vocab, vocab, rank=256, init_std=0.01).to(device)
    with torch.no_grad():
        zero.w2.weight.zero_()
    arms["zero-bias"] = [run_one(r, lambda logits, root, m=zero:
                                 markov_mod.steer_block_greedy(logits, root, m)[0])
                         for r in rows]

    scaled = markov_mod.MarkovHead(vocab, vocab, rank=256, init_std=0.01).to(device)
    with torch.no_grad():
        scaled.w2.weight.mul_(50.0)
    arms["scaled"] = [run_one(r, lambda logits, root, m=scaled:
                              markov_mod.steer_block_greedy(logits, root, m)[0])
                      for r in rows]

    for name, runs in arms.items():
        taus = [s.tau for _, s, _ in runs]
        print(f"{name:>10}: " + " | ".join(
            f"{len(ids)} tok, tau {s.tau:.4f}, accepted {acc}" for ids, s, acc in runs))

    ok = True
    print("\n=== verdict ===")

    base = arms["None"]
    for name in ("identity", "zero-bias"):
        same_ids = all(a[0] == b[0] for a, b in zip(arms[name], base))
        same_tau = all(abs(a[1].tau - b[1].tau) < 1e-12 for a, b in zip(arms[name], base))
        same_sched = all(a[2] == b[2] for a, b in zip(arms[name], base))
        if same_ids and same_tau and same_sched:
            print(f"PASS: steer={name} is bit-identical to steer=None "
                  f"(hook inert when unused)")
        else:
            print(f"FAIL: steer={name} changed the decode "
                  f"(ids {same_ids}, tau {same_tau}, schedule {same_sched})")
            ok = False

    sc = arms["scaled"]
    sched_moved = any(a[2] != b[2] for a, b in zip(sc, base))
    tau_moved = any(abs(a[1].tau - b[1].tau) > 1e-12 for a, b in zip(sc, base))
    if sched_moved or tau_moved:
        print(f"PASS: scaled Markov changes the accepted-length schedule "
              f"(schedule moved {sched_moved}, tau moved {tau_moved}) "
              f"-- the hook reaches the tree score")
    else:
        print("FAIL: scaled Markov left the schedule and tau untouched "
              "-- the hook never reached the tree")
        ok = False

    lossless = all(a[0] == b[0] for a, b in zip(sc, base))
    if lossless:
        print("PASS: output ids identical under a deliberately bad steer "
              "(losslessness holds independently of the drafter)")
    else:
        print("FAIL: a bad steer changed the committed ids -- decode is not lossless")
        ok = False

    print("ALL PASS" if ok else "THERE WERE FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
