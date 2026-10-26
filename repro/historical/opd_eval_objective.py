#!/usr/bin/env python3
"""Score each head on OPD's OWN objective, on a fixed set of positions.

Why this is the diagnostic that matters. OPD ended at tau 5.02, below both the
warm start (5.199) and the continue-SFT control (5.36). A negative result like
that has two very different readings:

  (a) OPD optimised its objective but the objective does not transfer to the
      tree-verified acceptance length; or
  (b) the run simply damaged a well-trained head without learning anything.

The training log cannot tell them apart: it reports the loss on ~27 positions
per micro-batch, resampled every step, and the step-to-step swing (2.5..4.6) is
far larger than any real trend. So this evaluates a FIXED set of positions (same
records, same anchors, same teacher) for every head, which removes the sampling
noise and makes the comparison exact.

    python opd_eval_objective.py --heads warm=.../head,sft=.../head_sft_control,opd=.../head_opd
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

sys.path.insert(0, str(_REPRO_ROOT))

from shared import common
from historical import opd_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(common.DATA, "opd_pool_textvqa_2000.pt"))
    ap.add_argument("--data", default=os.path.join(common.DATA, "target_greedy_textvqa2000.jsonl"))
    ap.add_argument("--heads", required=True,
                    help="comma list name=path")
    ap.add_argument("--records", type=int, default=40)
    ap.add_argument("--anchors-per-record", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=common.IMAGE_LONG_SIDE)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    target, processor = common.load_target(device=device)
    embed_weight = target.get_input_embeddings().weight

    blob = torch.load(args.pool, weights_only=False)
    recs = [r for r in blob["records"] if r["anchors"]]
    tasks = [t for t in (blob["config"].get("tasks") or "").split(",") if t]
    src = [json.loads(l) for l in open(args.data)]
    order = [i for i, r in enumerate(src) if not tasks or r.get("task") in tasks]

    # Deterministic subset: same records and same anchors for every head.
    step = max(1, len(recs) // args.records)
    picked = recs[::step][:args.records]

    fixed = []
    for rec in picked:
        row = src[order[rec["row"]]]
        image = common.load_image(row["image"], args.max_pixels)
        _, pix, grid = common.encode(processor, row["prompt"], image)
        pix = pix.to(device) if pix is not None else None
        grid = grid.to(device) if grid is not None else None
        seq = rec["input_ids"].to(device).long()
        ctx = common.target_context(target, seq[None], pix, grid)
        a = rec["anchors"]
        chose = a[:: max(1, len(a) // args.anchors_per_record)][:args.anchors_per_record]
        fixed.append((seq, ctx, chose, rec["topk_ids"].to(device),
                      rec["topk_val"].to(device)))
    print(f"{len(fixed)} fixed records, "
          f"{sum(len(f[2]) for f in fixed)} fixed anchors\n", flush=True)

    heads = []
    for item in args.heads.split(","):
        name, path = item.split("=", 1)
        heads.append((name, common.load_head(path, device=device).eval()))

    # Report BOTH objectives for every head. The two disagree, and that
    # disagreement is the whole finding: the Bernoulli form only constrains the
    # mass on the sampled token, while the top-k form constrains the ordering the
    # tree actually spends. A head can improve one while worsening the other, so
    # a single column would be misleading.
    print(f"{'head':>10} {'bern fwd':>10} {'topk fwd':>10} {'rev KL':>9} {'pos':>7}")
    for name, head in heads:
        acc = {}
        for obj in ("bernoulli", "topk"):
            tw, tf, tr, n = 0.0, 0.0, 0.0, 0
            for seq, ctx, chose, tk_ids, tk_val in fixed:
                with torch.no_grad():
                    loss, npos, l_fwd, l_rev, _ = opd_loss.opd_replay_loss(
                        head, target, ctx, seq,
                        [a["pos"] for a in chose], [a["prop"] for a in chose],
                        [a["n_acc"] for a in chose], tk_ids, tk_val,
                        embed_weight, head.block_size, common.MASK_ID, device,
                        alpha_rev=0.0, objective=obj)
                if loss is None:
                    continue
                tf += l_fwd * npos
                tr += l_rev * npos
                tw += npos
                n += npos
            acc[obj] = (tf / max(tw, 1), tr / max(tw, 1), n)
        print(f"{name:>10} {acc['bernoulli'][0]:>10.4f} {acc['topk'][0]:>10.4f} "
              f"{acc['bernoulli'][1]:>9.4f} {acc['bernoulli'][2]:>7}")


if __name__ == "__main__":
    main()
