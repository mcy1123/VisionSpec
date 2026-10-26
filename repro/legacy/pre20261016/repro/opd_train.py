#!/usr/bin/env python3
"""Train the GLANCE block head with Draft-OPD's on-policy replay loss.

The one thing this changes relative to `train_head.py`. SFT supervises the head
at anchors taken from the TARGET's greedy trajectory, with the block filled from
the target's own continuation. This trains it at the anchors the DRAFTER reaches
during tree-verified speculative decoding, with the block filled from what the
drafter proposed, scored by KL against the target's distribution at those same
states. The newest OPD report (`C组报告_2026-09-23.md`) shows rollout mode is the
decisive variable and that the reverse-KL term's marginal contribution is ~0, so
`--alpha-rev 0.1` is the default with 0.0 as the clean mechanism arm.

It deliberately reuses `opd_loss`, which reuses `train_head.forward_anchors` --
the forward already checked against the released implementation. There is no
second draft pass to drift out of convention.

Single-GPU on purpose: an 8B target plus a 1.05B head fits in ~25GB, and a
DDP path here would add a step-count/broadcast failure mode for no gain at this
scale.

    python opd_train.py --pool $GLANCE_ROOT/data/opd_pool_textvqa_2000.pt \
        --data $GLANCE_ROOT/data/target_greedy_textvqa2000.jsonl \
        --init-from $GLANCE_ROOT/checkpoints/head \
        --out $GLANCE_ROOT/checkpoints/head_opd --steps 1000 --gpu 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import opd_loss
import train_head as th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--data", default=os.path.join(common.DATA, "target_greedy.jsonl"),
                    help="source rows, for the images the pool's ids refer to")
    ap.add_argument("--init-from", default=os.path.join(common.CKPT, "head"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-anchors", type=int, default=8)
    ap.add_argument("--alpha-rev", type=float, default=0.1)
    ap.add_argument("--beta-fwd", type=float, default=1.0)
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--max-pixels", type=int, default=common.IMAGE_LONG_SIDE)
    ap.add_argument("--grad-clip", type=float, default=common.GRAD_CLIP)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix-sft-frac", type=float, default=0.0,
                    help="per micro-batch probability of using the ORIGINAL SFT "
                         "block loss instead of the OPD loss. SFT and OPD supply "
                         "different signals -- dense off-policy block coverage vs "
                         "the sparse states the drafter actually visits -- and on "
                         "this head SFT alone gave +0.156 tau while OPD alone gave "
                         "+0.096, so a mixture tests whether they are complementary.")
    ap.add_argument("--log-file", default=os.path.join(common.ROOT, "logs", "opd_train.jsonl"))
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--objective",
                    choices=["bernoulli", "topk", "topk_tail", "ce"],
                    default="bernoulli")
    ap.add_argument("--reject-mode", choices=["first", "suffix"], default="first",
                    help="'suffix' = paper Eq 9 (whole rejected tail, weight "
                         "gamma^(k-1) normalised by Z); 'first' = first rejected "
                         "position only")
    ap.add_argument("--drop-inexact", action="store_true", default=True)
    ap.add_argument("--keep-inexact", dest="drop_inexact", action="store_false")
    ap.add_argument("--cache-ctx", action="store_true", default=True,
                    help="precompute every row's target context once, into RAM")
    ap.add_argument("--no-cache-ctx", dest="cache_ctx", action="store_false")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    target, processor = common.load_target(device=device)
    head = common.load_head(args.init_from, device=device)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    embed_weight = target.get_input_embeddings().weight

    blob = torch.load(args.pool, weights_only=False)
    records = [r for r in blob["records"] if r["anchors"]]
    if not records:
        raise SystemExit("pool has no usable records")
    print(f"pool {args.pool}: {len(records)} rows, "
          f"{sum(len(r['anchors']) for r in records)} anchors, "
          f"topk {blob['config']['topk']}", flush=True)

    src = [json.loads(l) for l in open(args.data)]

    # `record["row"]` is the index among the rows the pool actually visited, not
    # the line number in --data: opd_rollout applies its --tasks filter first and
    # only then counts. Indexing the source file directly therefore paired each
    # sequence with the WRONG image, which surfaces as "Image features and image
    # tokens do not match" (a different image resizes to a different token
    # count) -- a loud failure, but only by luck. Rebuild the same filtered order
    # here and assert the task agrees, so a future filter change cannot silently
    # misalign the two files.
    tasks = [t for t in (blob["config"].get("tasks") or "").split(",") if t]
    order = [i for i, r in enumerate(src)
             if not tasks or r.get("task") in tasks]
    for r in records:
        if r["row"] >= len(order):
            raise SystemExit(f"pool row {r['row']} beyond {len(order)} filtered rows")
        abs_idx = order[r["row"]]
        if r.get("task") is not None and src[abs_idx].get("task") != r["task"]:
            raise SystemExit(
                f"pool/source misalignment at row {r['row']}: pool says "
                f"{r['task']!r}, {args.data}[{abs_idx}] says {src[abs_idx].get('task')!r}")
        r["src_idx"] = abs_idx
    if tasks:
        print(f"pool tasks {tasks}: remapped {len(order)} filtered rows to source "
              f"line numbers", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                            betas=common.BETAS, weight_decay=common.WEIGHT_DECAY)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
    log = open(args.log_file, "a")

    # Precompute each row's target context once, into RAM.
    #
    # Without this every optimizer step re-reads an image from disk and re-runs an
    # 8B prefill, and on this box the disk is the binding constraint (24 MB/s
    # under load, steps parked in uninterruptible page-cache waits). The context
    # is ~12 MB per row, so 2000 rows is ~25 GB against 276 GB available -- far
    # cheaper than paying that read once per step. It also removes the target
    # forward from the step, leaving only the head to run and backprop.
    cache = None
    if args.cache_ctx:
        print(f"precomputing target context for {len(records)} rows...", flush=True)
        cache, t0 = [], time.time()
        for i, rec in enumerate(records):
            row = src[rec["src_idx"]]
            image = common.load_image(row["image"], args.max_pixels)
            _, pix, grid = common.encode(processor, row["prompt"], image)
            pix = pix.to(device) if pix is not None else None
            grid = grid.to(device) if grid is not None else None
            seq = rec["input_ids"].to(device).long()
            with torch.no_grad():
                cache.append(common.target_context(target, seq[None], pix, grid).cpu())
            if (i + 1) % 200 == 0:
                r = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(records)} cached, {r:.2f} rows/s, "
                      f"eta {(len(records)-i-1)/max(r,1e-9)/60:.1f} min", flush=True)
        print(f"context cache ready in {(time.time()-t0)/60:.1f} min", flush=True)

    step, micro, running = 0, 0, 0.0
    n_pos_tot, n_fwd, n_rev, n_inexact = 0, 0, 0, 0
    started = time.time()
    opt.zero_grad(set_to_none=True)

    while step < args.steps:
        ri = rng.randrange(len(records))
        rec = records[ri]
        # .long() is required, not cosmetic: the pool stores ids as int32 to
        # halve its size, and feeding int32 into Qwen3-VL's get_rope_index
        # raises "Index put requires the source and destination dtypes match".
        seq_ids = rec["input_ids"].to(device).long()
        if cache is not None:
            ctx = cache[ri].to(device)
        else:
            row = src[rec["src_idx"]]
            image = common.load_image(row["image"], args.max_pixels)
            _, pix, grid = common.encode(processor, row["prompt"], image)
            pix = pix.to(device) if pix is not None else None
            grid = grid.to(device) if grid is not None else None
            # The image MUST be passed -- without it the target runs as a text
            # model over image placeholders (measured 0.59 relative difference)
            # and the head is trained on states it never sees.
            ctx = common.target_context(target, seq_ids[None], pix, grid)

        # Mixed mode: with probability `mix_sft_frac` run the original SFT block
        # loss (reveal-prior blocks built from the target's own continuation)
        # instead of the OPD replay loss, on this same row. Reuses
        # train_head.sequence_loss and sample_reveals, so the SFT half is exactly
        # the control arm's objective rather than a re-implementation.
        if args.mix_sft_frac > 0 and rng.random() < args.mix_sft_frac:
            total = int(rec["seq_len"])
            first, last = int(rec["prompt_len"]), total - 2
            if last < first:
                continue
            anchors_s = list(range(first, last + 1))
            gen = torch.Generator(device="cpu").manual_seed(rng.randrange(1 << 30))
            reveals = th.sample_reveals(len(anchors_s), head.block_size, gen, "cpu")
            anchors_s, reveals = th.live_anchors(anchors_s, reveals, total, head.block_size)
            if not anchors_s:
                anchors_s = list(range(first, last + 1))
                reveals = torch.ones(len(anchors_s), dtype=torch.long)
                anchors_s, reveals = th.live_anchors(anchors_s, reveals, total, head.block_size)
            if not anchors_s:
                continue
            reveals = reveals.to(device)
            loss, _corr, _w, _bo = th.sequence_loss(
                head, target, ctx, seq_ids, anchors_s, reveals, embed_weight,
                common.MASK_ID, head.block_size, device)
            (loss / args.accum).backward()
            running += float(loss.detach())
            n_pos_tot += len(anchors_s) * (head.block_size - 1)
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == 1:
                    rec_row = {"step": step, "loss": running / args.accum,
                               "fwd": float("nan"), "rev": float("nan"),
                               "positions": float(n_pos_tot),
                               "inexact_frac": 0.0,
                               "elapsed_min": (time.time() - started) / 60}
                    print(f"step {step} obj mixed(sft-batch) loss "
                          f"{rec_row['loss']:.4f} ({rec_row['elapsed_min']:.1f} min)",
                          flush=True)
                    log.write(json.dumps(rec_row) + "\n")
                    log.flush()
                running = 0.0
                n_pos_tot = 0
                n_inexact = 0
                if args.save_every and step % args.save_every == 0:
                    head.save_pretrained(os.path.join(args.out, f"step{step}"))
                    processor.tokenizer.save_pretrained(os.path.join(args.out, f"step{step}"))
                    print(f"  saved step{step}", flush=True)
            continue

        all_a = rec["anchors"]
        k = min(args.max_anchors, len(all_a))
        picked = [all_a[i] for i in rng.sample(range(len(all_a)), k)]
        loss, npos, l_fwd, l_rev, nix = opd_loss.opd_replay_loss(
            head, target, ctx, seq_ids,
            [a["pos"] for a in picked], [a["prop"] for a in picked],
            [a["n_acc"] for a in picked],
            rec["topk_ids"].to(device), rec["topk_val"].to(device),
            embed_weight, head.block_size, common.MASK_ID, device,
            alpha_rev=args.alpha_rev, beta_fwd=args.beta_fwd, decay=args.decay,
            objective=args.objective, drop_inexact_teacher=args.drop_inexact,
            reject_mode=args.reject_mode)
        if loss is None:
            continue
        (loss / args.accum).backward()
        running += float(loss.detach())
        n_pos_tot += npos
        n_inexact += nix
        micro += 1

        if micro % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                rec_row = {"step": step, "loss": running / args.accum,
                           "fwd": l_fwd, "rev": l_rev,
                           # Per optimizer step, NOT divided by the global step
                           # counter: n_pos_tot is reset every step, so dividing
                           # by `step` made this decay like 1/step and look like
                           # the supervision was collapsing when it was flat.
                           "positions": float(n_pos_tot),
                           "inexact_frac": n_inexact / max(n_pos_tot, 1),
                           "elapsed_min": (time.time() - started) / 60}
                print(f"step {step} obj {args.objective}/{args.reject_mode} loss {rec_row['loss']:.4f} "
                      f"fwd {l_fwd:.4f} rev {l_rev:.4f} "
                      f"pos/step {rec_row['positions']:.1f} "
                      f"inexact {rec_row['inexact_frac']:.2f} "
                      f"({rec_row['elapsed_min']:.1f} min)", flush=True)
                log.write(json.dumps(rec_row) + "\n")
                log.flush()
            running = 0.0
            n_pos_tot = 0
            n_inexact = 0
            if args.save_every and step % args.save_every == 0:
                head.save_pretrained(os.path.join(args.out, f"step{step}"))
                processor.tokenizer.save_pretrained(os.path.join(args.out, f"step{step}"))
                print(f"  saved step{step}", flush=True)

    head.save_pretrained(args.out)
    processor.tokenizer.save_pretrained(args.out)
    print(f"done at step {step}, saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
