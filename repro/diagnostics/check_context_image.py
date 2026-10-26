#!/usr/bin/env python3
"""How much does the target context change when the image is actually passed?

`common.target_context` only forwards `pixel_values`/`image_grid_thw` (and the
VLM `position_ids`) when it is given them; called without, it runs the target as
a plain text model over image-placeholder tokens. Several call sites in this
repro pass no pixels -- including the OPD rollout and `measure_offset_curve.py`
-- so this measures the size of that mistake rather than assuming it.

Decisive reading: if the two contexts differ materially, every per-offset curve
measured through the no-pixel path is describing a model state the head never
sees in real decoding, and those numbers cannot be compared against tau.

    python check_context_image.py --gpu 3
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
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=common.IMAGE_LONG_SIDE)
    ap.add_argument("--gpu", type=int, default=3)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    target, processor = common.load_target(device=device)
    head = common.load_head(args.head, device=device).eval()
    stop = [processor.tokenizer.eos_token_id]

    row = None
    for r in common.read_jsonl(args.data):
        if os.path.exists(r["image"]):
            row = r
            break
    image = common.load_image(row["image"], args.max_pixels)
    ids, pix, grid = common.encode(processor, row["prompt"], image)
    ids = ids.to(device)
    pix = pix.to(device) if pix is not None else None
    grid = grid.to(device) if grid is not None else None
    print(f"row: task={row['task']}  prompt_len={ids.shape[1]}  "
          f"pixels={'yes' if pix is not None else 'NO'}  grid={None if grid is None else tuple(grid.shape)}")

    pos = VisionPositions(pix, grid)
    final = glance.greedy_generate(target, ids, args.gen, stop, positions=pos)
    full = torch.cat([ids[0], final[0, ids.shape[1]:]])
    prompt_len = int(ids.shape[1])

    ctx_no = common.target_context(target, full[None])
    ctx_yes = common.target_context(target, full[None], pixel_values=pix, image_grid_thw=grid)
    diff = (ctx_no.float() - ctx_yes.float()).abs()
    print(f"\ncontext shapes {tuple(ctx_no.shape)} vs {tuple(ctx_yes.shape)}")
    print(f"  max|delta| {diff.max().item():.4f}   mean|delta| {diff.mean().item():.4f}")
    print(f"  relative: ||delta|| / ||with|| = "
          f"{(diff.norm() / ctx_yes.float().norm()).item():.4f}")
    # cosine per position, averaged over the generated region
    a = ctx_no.float()[0, prompt_len:]
    b = ctx_yes.float()[0, prompt_len:]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print(f"  cosine per position over generated tokens: mean {cos.mean().item():.4f} "
          f"min {cos.min().item():.4f}")

    # Per-offset agreement of the head's own proposals under the two contexts.
    for name, ctx in (("no-image", ctx_no), ("with-image", ctx_yes)):
        hits = [0] * (head.block_size - 1)
        tot = [0] * (head.block_size - 1)
        for anchor in range(prompt_len, len(full) - head.block_size - 1, args.stride):
            lg = opd_rollout.block_logits(head, target, full, ctx, anchor)
            pred = lg.argmax(dim=-1)
            for k in range(1, head.block_size):
                tot[k - 1] += 1
                hits[k - 1] += int(pred[k - 1]) == int(full[anchor + k])
        acc = [h / max(t, 1) for h, t in zip(hits, tot)]
        print(f"\n  {name:11s} per-offset top-1: "
              + " ".join(f"{x:.2f}" for x in acc[:8]) + " ... "
              + f"off15 {acc[-1]:.2f}  (n={tot[0]})")


if __name__ == "__main__":
    main()
