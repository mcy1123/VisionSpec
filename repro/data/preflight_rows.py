#!/usr/bin/env python3
"""Pre-flight: push the riskiest rows through a full training step.

The crash that killed the first training attempt was an out-of-range CUDA
`index_select`, which raises a device-side assert rather than a Python error and
surfaces hundreds of steps later in unrelated code. Reasoning about it
statically is what missed it the first time, so this runs the actual step --
target forward, head forward, backward -- over the rows most likely to break:
every row that needs the reveal = 1 retry, plus a random spread.

Run with `CUDA_LAUNCH_BLOCKING=1` so a bad kernel fails at its own call site
instead of wherever the next synchronisation happens to be.

    CUDA_LAUNCH_BLOCKING=1 python preflight_rows.py --gpu 0
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
import random

import torch

from shared import common
from historical import train_head as T


def risky_positions(rows):
    """Indices of rows whose blocks all come out empty at the sampled reveal."""
    out = []
    dev = torch.device("cpu")
    for pos, row in enumerate(rows):
        total = row["n_prompt"] + len(row["gen_ids"])
        first, last = row["n_prompt"], total - 2
        anchors = list(range(first, last + 1))
        rev = T.sample_reveals(len(anchors), common.BLOCK_SIZE,
                               torch.Generator().manual_seed(common.SEED + pos), dev)
        a, _ = T.live_anchors(anchors, rev, total, common.BLOCK_SIZE)
        if not a:
            out.append(pos)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(common.DATA, "target_greedy.jsonl"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--random", type=int, default=100)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    rows = [r for r in common.read_jsonl(args.data) if os.path.exists(r["image"])]
    rows = [r for r in rows if len(r["gen_ids"]) >= 2]
    print(f"{len(rows)} rows")

    risky = risky_positions(rows)
    print(f"{len(risky)} rows need the reveal=1 retry: {risky[:20]}"
          + (" ..." if len(risky) > 20 else ""))

    rng = random.Random(common.SEED)
    sample = sorted(set(risky) | set(rng.sample(range(len(rows)), min(args.random, len(rows)))))
    print(f"pre-flight over {len(sample)} rows (all risky ones plus {args.random} random)")

    target, processor = common.load_target(device=device)
    head = common.load_head(common.HEAD_WARM_PATH, device=device)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    embed_weight = target.get_input_embeddings().weight

    failures = 0
    for n, pos in enumerate(sample):
        row = rows[pos]
        tag = (f"{row['task']} prompt={row['n_prompt']} gen={len(row['gen_ids'])} "
               f"{os.path.basename(row['image'])}")
        try:
            image = common.load_image(row["image"], common.IMAGE_LONG_SIDE)
            ids, pix, grid = common.encode(processor, row["prompt"], image)
            full = torch.cat([ids[0], torch.tensor(row["gen_ids"], dtype=torch.long)])
            full = full[: common.MAX_SEQ_LEN]
            prompt_len = min(ids.shape[1], full.shape[0])
            total = full.shape[0]
            full = full.to(device)

            ctx = common.target_context(
                target, full[None],
                pix.to(device) if pix is not None else None,
                grid.to(device) if grid is not None else None)

            first, last = prompt_len, total - 2
            anchors = list(range(first, last + 1))
            rev = T.sample_reveals(len(anchors), common.BLOCK_SIZE,
                                   torch.Generator().manual_seed(common.SEED + pos), device)
            anchors, rev = T.live_anchors(anchors, rev, total, common.BLOCK_SIZE)
            if not anchors:
                anchors = list(range(first, last + 1))
                rev = torch.ones(len(anchors), dtype=torch.long, device=device)
                anchors, rev = T.live_anchors(anchors, rev, total, common.BLOCK_SIZE)

            loss, _, _, _ = T.sequence_loss(
                head, target, ctx, full, anchors, rev, embed_weight,
                common.MASK_ID, common.BLOCK_SIZE, device)
            loss.backward()
            head.zero_grad(set_to_none=True)
            if n % 20 == 0:
                print(f"  [{n+1}/{len(sample)}] ok  {tag}  loss {loss.item():.3f}")
        except Exception as exc:
            failures += 1
            print(f"  [{n+1}/{len(sample)}] FAIL {tag}: {type(exc).__name__}: {str(exc)[:200]}")
            if "device-side assert" in str(exc):
                print("  device-side assert: the CUDA context is gone, stopping here")
                break

    print(f"pre-flight: {len(sample) - failures}/{len(sample)} rows trained cleanly")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
