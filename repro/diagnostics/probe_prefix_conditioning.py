#!/usr/bin/env python3
"""Is the in-block decay a conditioning problem or a capacity problem?

The question this settles. GLANCE's head predicts offsets 1..15 from a block
whose only real token is the root -- at decode `block_ids[:, 1:] = mask_id`. Its
per-offset accuracy falls 0.52 -> 0.08. DSpark attributes that shape to
"multi-modal collision": each position marginalises over all possible prior
tokens instead of conditioning on one sampled prefix. If DSpark is right, then
handing the head the *correct* prefix should restore most of the lost accuracy,
and a cheap semi-autoregressive mechanism can recover it. If the accuracy stays
low even with the true prefix in place, the information is not in the head's
representation at all and no conditioning trick will help.

Measured here, per offset k and reveal length l:

  P1  l = 1            the decode-time block: [root, MASK, ...]
  P2  l = k            [root, Y_1..Y_{k-1}, MASK, ...] with Y the TARGET's tokens
  P3  l = k            same shape but Y replaced by the head's OWN argmax

P2 - P1 is what conditioning could buy. P2 - P3 is the exposure-bias cost of
actually doing it at inference, which is the gap DSpark never ablates.

CPU-light, no training.

    python probe_prefix_conditioning.py --n 20 --gpu 0
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
import statistics
import sys

import torch

from shared import common

sys.path.insert(0, os.path.join(common.WORKSPACE, "GLANCE_repo"))

import glance
from glance.vl import VisionPositions


def block_logits(head, target, seq_ids, fused, anchor, reveal, prev_tokens):
    block = head.block_size
    ids = torch.full((1, block), head.mask_token_id, dtype=torch.long,
                     device=fused.device)
    ids[0, 0] = seq_ids[anchor]
    if reveal > 1 and prev_tokens is not None:
        ids[0, 1:reveal] = prev_tokens[: reveal - 1]
    ctx = fused[:, :anchor]
    position_ids = torch.arange(anchor + block, device=fused.device).unsqueeze(0)
    hidden = head(target_hidden=ctx,
                  noise_embedding=target.get_input_embeddings()(ids),
                  position_ids=position_ids)
    return target.lm_head(hidden[:, 1 - block:, :])[0]      # (block-1, vocab): offsets 1..15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", default=os.path.join(common.CKPT, "head"))
    parser.add_argument("--data", default=os.path.join(common.DATA, "vl_eval_plain"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--gen", type=int, default=96, help="target greedy tokens to analyse")
    parser.add_argument("--max-pixels", type=int, default=896)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--reveals", default="1,2,4,8,15")
    parser.add_argument("--stride", type=int, default=4, help="anchor stride")
    parser.add_argument("--out", default=os.path.join(common.RESULTS, "prefix_conditioning.json"))
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    reveals = [int(x) for x in args.reveals.split(",")]
    target, processor = common.load_target(device=device)
    head = common.load_head(args.head, device=device).eval()
    stop = [processor.tokenizer.eos_token_id]
    block = head.block_size

    rows = []
    for row in common.read_jsonl(os.path.join(args.data, "chartqa.jsonl")):
        img = os.path.join(args.data, row["image"])
        if os.path.exists(img):
            rows.append((img, row["prompt"]))
        if len(rows) >= args.n:
            break
    print(f"{len(rows)} prompts, reveals {reveals}, block {block}", flush=True)

    # acc[arm][l][k] -> [correct, total]
    acc = {"true": {}, "self": {}}
    for arm in acc:
        for l in reveals:
            acc[arm][l] = {k: [0, 0] for k in range(1, block)}

    for i, (img_path, prompt) in enumerate(rows):
        image = common.load_image(img_path, args.max_pixels)
        ids, pix, grid = common.encode(processor, prompt, image)
        ids = ids.to(device)
        pos = VisionPositions(pix.to(device) if pix is not None else None,
                              grid.to(device) if grid is not None else None)
        final = glance.greedy_generate(target, ids, args.gen, stop, positions=pos)
        cont = final[0, ids.shape[1]:]
        full = torch.cat([ids[0], cont])
        prompt_len = ids.shape[1]
        if full.shape[0] < prompt_len + block + 2:
            continue
        fused = common.target_context(target, full[None])

        last_anchor = full.shape[0] - block - 1
        for anchor in range(prompt_len, last_anchor + 1, args.stride):
            # P1: the decode-time block, and the head's own proposals for P3
            base = block_logits(head, target, full, fused, anchor, 1, None)
            self_prop = base.argmax(dim=-1)                 # offsets 1..15
            for l in reveals:
                if l >= block:
                    continue
                true_prev = full[anchor + 1: anchor + l]
                for arm, prev in (("true", true_prev), ("self", self_prop[: l - 1])):
                    logits = base if l == 1 else block_logits(
                        head, target, full, fused, anchor, l, prev)
                    pred = logits.argmax(dim=-1)
                    for k in range(max(l, 1), block):
                        gold = int(full[anchor + k])
                        correct = int(pred[k - 1]) == gold
                        slot = acc[arm][l][k]
                        slot[0] += correct
                        slot[1] += 1
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(rows)} prompts", flush=True)

    def rate(arm, l, k):
        c, t = acc[arm][l][k]
        return (c / t) if t else float("nan")

    print("\n=== per-offset top-1, TRUE prefix (P2) vs decode-time (P1) ===")
    print(f"{'k':>3} " + "".join(f"{'l='+str(l):>9}" for l in reveals) + f"{'P2-P1':>9}")
    for k in range(1, block):
        cells = []
        for l in reveals:
            cells.append(rate("true", l, k) if l <= k else float("nan"))
        base = cells[0]
        best = max((c for c in cells if c == c), default=float("nan"))
        row = "".join(f"{c:>9.3f}" if c == c else f"{'-':>9}" for c in cells)
        print(f"{k:>3} {row}{best - base:>9.3f}")

    print("\n=== same table, SELF prefix (P3): the exposure-bias view ===")
    print(f"{'k':>3} " + "".join(f"{'l='+str(l):>9}" for l in reveals))
    for k in range(1, block):
        cells = [rate("self", l, k) if l <= k else float("nan") for l in reveals]
        row = "".join(f"{c:>9.3f}" if c == c else f"{'-':>9}" for c in cells)
        print(f"{k:>3} {row}")

    print("\n=== headline numbers ===")
    for k in (4, 8, 15):
        p1 = rate("true", 1, k)
        p2 = max((rate("true", l, k) for l in reveals if l <= k and rate("true", l, k) == rate("true", l, k)),
                 default=float("nan"))
        p3 = max((rate("self", l, k) for l in reveals if l <= k and rate("self", l, k) == rate("self", l, k)),
                 default=float("nan"))
        print(f"  offset {k:2d}: decode-time {p1:.3f} | best TRUE prefix {p2:.3f} "
              f"(+{p2-p1:.3f}) | best SELF prefix {p3:.3f}")

    payload = {"reveals": reveals, "block": block, "n_prompts": len(rows),
               "acc": {arm: {str(l): {str(k): acc[arm][l][k] for k in acc[arm][l]}
                             for l in acc[arm]} for arm in acc}}
    with open(args.out, "w") as h:
        json.dump(payload, h, indent=2)
    print(f"\nwrote {args.out}")

    p1_8 = rate("true", 1, 8)
    p2_8 = max((rate("true", l, 8) for l in reveals if l <= 8), default=float("nan"))
    verdict = ("CONDITIONING: the prefix carries the missing information -- "
               "a semi-autoregressive / Markov mechanism is worth training"
               if p2_8 - p1_8 > 0.15 else
               "CAPACITY: the true prefix does not restore accuracy -- "
               "no conditioning trick will help; pivot to representation-side fixes")
    print(f"VERDICT at offset 8: P1 {p1_8:.3f} -> P2 {p2_8:.3f} => {verdict}")


if __name__ == "__main__":
    main()
