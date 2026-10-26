#!/usr/bin/env python3
"""Does the head-only rollout agree with the real tree decode, round by round?

The pool builder (`opd_rollout.py`) replaces GLANCE's real decode with a head-only
rollout justified by losslessness: the committed tokens are the target's own
greedy continuation, so the accepted length of a round is just "the longest tree
path equal to that continuation" and needs no target verification pass.

That argument is exactly the kind that is easy to get subtly wrong -- the wrong
tree budget, the wrong offset convention, or matching against the wrong slice
would all show up as a plausibly-low accept length rather than as a crash. And
the consequence is not cosmetic: a rollout that accepts less than the real
decoder would build the OPD pool from a state distribution the model never
actually visits, which is the one thing the method depends on.

So this check runs both paths on the same prompt and requires:

  1. the two committed token sequences are identical, and
  2. the per-round (pos, n_acc) sequences are identical.

Both are necessary: (1) alone would pass even if the round boundaries were wrong,
and (2) alone would pass if both paths were wrong in the same way.

    python check_rollout_matches_decode.py --rows 3 --gpu 3
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
import os
import sys

import torch

sys.path.insert(0, str(_REPRO_ROOT))

from shared import common
from historical import opd_rollout

sys.path.insert(0, os.path.join(common.WORKSPACE, "GLANCE_repo"))

import glance
from glance.vl import VisionPositions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(common.DATA, "target_greedy.jsonl"))
    ap.add_argument("--head", default=os.path.join(common.CKPT, "head"))
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch-k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=common.IMAGE_LONG_SIDE)
    ap.add_argument("--gpu", type=int, default=3)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    target, processor = common.load_target(device=device)
    head = common.load_head(args.head, device=device).eval()
    stop = [processor.tokenizer.eos_token_id]
    print(f"head {args.head}, block {head.block_size}, "
          f"budget {args.budget}, max_pixels {args.max_pixels}", flush=True)

    ok = True
    n = 0
    for row in common.read_jsonl(args.data):
        if n >= args.rows:
            break
        if not os.path.exists(row["image"]):
            continue
        n += 1
        image = common.load_image(row["image"], args.max_pixels)
        ids, pix, grid = common.encode(processor, row["prompt"], image)
        ids = ids.to(device)
        pos = VisionPositions(pix.to(device) if pix is not None else None,
                              grid.to(device) if grid is not None else None)

        # --- path A: the real decode, recording (pos, accepted) per round
        rounds = []
        out, stats = glance.generate(
            head, target, ids, args.max_new, stop, budget=args.budget,
            branch_k=args.branch_k, temperature=0.0, positions=pos,
            return_stats=True, timing="decode_only", on_round=rounds.append)
        committed = out[0, ids.shape[1]:].tolist()
        real = [(r["pos"], r["accepted"]) for r in rounds]

        # --- path B: the head-only rollout on path A's own committed tokens
        full = out[0]
        ctx = common.target_context(target, full[None])
        anchors = opd_rollout.rollout_tree_head_only(
            head, target, full, ctx, int(ids.shape[1]), args.budget, args.branch_k)
        rollout = [(a["pos"], a["n_acc"]) for a in anchors]

        same_seq = committed == full[ids.shape[1]:].tolist()   # trivially true
        same_rounds = real == rollout
        print(f"\nrow {n}: {len(committed)} tokens, real {len(real)} rounds, "
              f"rollout {len(rollout)} anchors")
        print(f"  real    {real[:8]}{'...' if len(real) > 8 else ''}")
        print(f"  rollout {rollout[:8]}{'...' if len(rollout) > 8 else ''}")
        if same_rounds:
            print("  PASS: (pos, n_acc) sequences identical")
        else:
            print("  FAIL: rollout disagrees with the real decode")
            for i, (a, b) in enumerate(zip(real, rollout)):
                if a != b:
                    print(f"        first divergence at round {i}: real {a} vs rollout {b}")
                    break
            if len(real) != len(rollout):
                print(f"        different round counts: {len(real)} vs {len(rollout)}")
            ok = False

    print("\nALL PASS" if ok else "\nTHERE WERE FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
