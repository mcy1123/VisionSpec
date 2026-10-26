#!/usr/bin/env python3
"""Train the GLANCE block head: DFlash variable-prefix coverage objective.

Recipe (all traced in CONFIG.md):

* 5-layer DFlash block head, B=16, horizon L=15, warm-started from
  `z-lab/Qwen3-8B-DFlash-b16`.                       [Table 5]
* data: the target's own greedy generations, 16K teacher-forced rows,
  images at long side 768 px.                        [§4.3, Table 5, App.D]
* objective: offset-wise top-1 coverage CE with a 0.8^offset weight and a
  variable-mask block prefix (the DFlash variable-prefix objective).
                                                      [Table 5, §4.3]
* AdamW lr 4e-5, betas 0.9/0.95, no weight decay, clip 1.0, 1 epoch,
  accumulation 8, bf16, seed 0.                      [Table 5, App.D]

Why this file rolls its own forward
-----------------------------------
The released head's forward puts the context and the block in one sequence:
keys are `cat([k_proj(context), k_proj(block)])` and RoPE numbers that
concatenation from 0, while the queries take the *last* `q_len` positions. A
single block therefore trains correctly with the stock forward, but every
anchor would recompute `fc` (20480->4096) and the context K/V over the whole
prefix, which dominates the step.

Anchors of one sequence share a prefix, so their context K/V are prefixes of
one another. This file projects the context once per (sequence, layer) and
builds every anchor of the sequence in a single forward, with an explicit
mask. `--verify` checks that forward against the stock one on a single
anchor; that check must pass before any training run is trusted.

    python train_head.py --data $GLANCE_ROOT/data/target_greedy.jsonl \
        --gpus 3,6 --out $GLANCE_ROOT/checkpoints/head
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
import time

import torch
import torch.nn.functional as F
from PIL import Image

from shared import common
from shared import markov as markov_mod


# --------------------------------------------------------------------------
# RoPE helpers (the head's own tables, applied with explicit positions)
# --------------------------------------------------------------------------







# --------------------------------------------------------------------------
# The batched multi-anchor forward
# --------------------------------------------------------------------------





# --------------------------------------------------------------------------
# One sequence -> one loss
# --------------------------------------------------------------------------












# --------------------------------------------------------------------------
# Verification against the released forward
# --------------------------------------------------------------------------



from shared.sft_blocks import rotate_half, rope_tables, apply_rope, build_anchor_mask, forward_anchors, sample_reveals, live_anchors, build_blocks, self_conditioned_source, sequence_loss, verify

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(common.DATA, "target_greedy.jsonl"))
    parser.add_argument("--out", default=os.path.join(common.CKPT, "head"))
    parser.add_argument("--gpus", default="3")
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=common.ACCUM)
    parser.add_argument("--lr", type=float, default=common.LR)
    parser.add_argument("--epochs", type=int, default=common.EPOCHS)
    parser.add_argument("--max-seq-len", type=int, default=common.MAX_SEQ_LEN)
    parser.add_argument("--long-side", type=int, default=common.IMAGE_LONG_SIDE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--markov-rank", type=int, default=0,
                        help="DSpark sequential logit bias; 0 disables it (default)")
    parser.add_argument("--markov-input", choices=["gt", "block", "self"], default="gt",
                        help="what conditions the bias during training: 'gt' the "
                             "target's own tokens (DSpark's recipe, does not match "
                             "decode), 'block' the block's contents, or 'self' the "
                             "model's own proposal -- which is what decode feeds it")
    parser.add_argument("--markov-lr", type=float, default=None,
                        help="separate lr for the Markov bias; defaults to --lr, "
                             "which is what keeps Arm M a controlled comparison")
    parser.add_argument("--init-from", default=None,
                        help="load the block head from this path instead of the "
                             "warm start. Used to attach a new module to an "
                             "already-trained head so both arms share it exactly.")
    parser.add_argument("--freeze-head", action="store_true",
                        help="train only the new module. The head stays exactly "
                             "as loaded, so a tau delta is attributable to the "
                             "new module alone rather than to more head training. "
                             "Also removes the backward pass through the head.")
    parser.add_argument("--prefix-base", type=float, default=None,
                        help="override the reveal prior's base; CONFIG.md §8 item 10")
    parser.add_argument("--steps", type=int, default=0, help="debug: stop after N optimizer steps")
    # The paper's recipe fixes seed 0, and `common.SEED` encoded that. Seed has to
    # be an argument to run the multi-seed comparison that distinguishes a real
    # OPD gain from run-to-run noise: the eval set is paired, but the training run
    # is the unit of replication, so without this every "seed" is the same run.
    parser.add_argument("--skip-image-verify", action="store_true",
                        help="skip the Image.open().verify() pass over every row. "
                             "It is a data-hygiene check, and it reads every image "
                             "in full: on this box that is ~13 min per 2000 rows "
                             "under load, so 16K rows costs hours of disk thrash "
                             "before the first optimizer step. Only safe when these "
                             "exact rows have already passed the check in an earlier "
                             "run, which they have.")
    parser.add_argument("--seed", type=int, default=common.SEED,
                        help="overrides common.SEED for all RNG in this run")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-every", type=int, default=200)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--ddp", action="store_true",
                        help="data-parallel across torchrun ranks")
    # Deliberately not `--log`: torchrun's own parser has `--log-dir` and
    # `--logs-specs`, and argparse's prefix matching turns `--log` into
    # "ambiguous option" -- which kills a `torchrun train_head.py ... --log X`
    # invocation before the script ever starts.
    parser.add_argument("--log-file",
                        default=os.path.join(common.ROOT, "logs", "train_head.jsonl"))
    args = parser.parse_args()

    if args.prefix_base is not None:
        common.PREFIX_BASE = float(args.prefix_base)
    print(f"[config] reveal prior base = {common.PREFIX_BASE} "
          f"(mean reveal {float((common.prefix_weights()[0] * common.prefix_weights()[1]).sum()):.2f})",
          flush=True)

    ddp = args.ddp
    if ddp:
        from datetime import timedelta

        import torch.distributed as dist

        # A default 30-minute collective timeout turns any residual
        # accumulation-boundary mismatch into half an hour of silence; twenty
        # minutes is far longer than any single step here needs, so a hang
        # surfaces as an exception instead.
        dist.init_process_group("nccl", timeout=timedelta(minutes=20))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
        gpus = [local]
    else:
        rank, world = 0, 1
        gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
        device = f"cuda:{gpus[0]}"
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0

    target, processor = common.load_target(device=device)
    head = common.load_head(args.init_from or common.HEAD_WARM_PATH, device=device)
    head.train()
    if args.grad_checkpoint:
        head.gradient_checkpointing_enable()
        head.enable_input_require_grads = lambda: None
    embed_weight = target.get_input_embeddings().weight

    markov = None
    if args.markov_rank > 0:
        vocab = embed_weight.shape[0]
        markov = markov_mod.MarkovHead(
            verifier_vocab_size=vocab, draft_vocab_size=vocab,
            rank=args.markov_rank).to(device)
        n = sum(p.numel() for p in markov.parameters())
        print(f"[config] Markov head: rank {args.markov_rank}, "
              f"{n/1e6:.1f}M params, input={args.markov_input}", flush=True)

    def say(*parts):
        if is_main:
            print(*parts, flush=True)

    # ---- correctness gate -------------------------------------------------
    # Runs before the DDP wrap: the released forward is a real forward pass, so
    # driving it on one rank only would deadlock the other ranks' collectives.
    # fp32 shows the two forwards agree to ~3e-6 relative, which is the
    # equality that matters; in bf16 five layers of rounding leave a small
    # residual, so the gate is on direction, not on absolute error.
    rows = [r for r in common.read_jsonl(args.data) if os.path.exists(r["image"])]
    if args.limit:
        rows = rows[: args.limit]
    if rows and is_main:
        row = rows[0]
        image = common.load_image(row["image"], args.long_side)
        input_ids, pixel_values, grid = common.encode(processor, row["prompt"], image)
        full = torch.cat([input_ids[0], torch.tensor(row["gen_ids"], dtype=torch.long)])
        full = full[: args.max_seq_len].to(device)
        ctx = common.target_context(
            target, full[None],
            pixel_values.to(device) if pixel_values is not None else None,
            grid.to(device) if grid is not None else None)
        diff, scale, cosine = verify(head, target, ctx, full, common.MASK_ID,
                                     common.BLOCK_SIZE, device)
        print(f"[verify] bf16 max|stock - mine| = {diff:.3e} "
              f"(scale {scale:.3e}), hidden cosine = {cosine:.6f}", flush=True)
        if cosine < 0.999:
            raise SystemExit("custom forward does not match the released forward "
                             f"(cosine {cosine:.6f}); refusing to train")
        if args.verify_only:
            return
    if ddp:
        import torch.distributed as dist

        dist.barrier()

    # No DistributedDataParallel wrapper, on purpose. DDP installs its
    # gradient hooks inside *its own* forward(), and this file never calls it:
    # the head is driven through `forward_anchors`, which reaches into the
    # head's submodules directly. A DDP-wrapped head would therefore never
    # synchronise a single gradient -- the two ranks would train two different
    # heads on two halves of the data and finish with no error at all.
    # Gradients are averaged explicitly at the optimizer step instead, which is
    # what DDP does minus the overlap.

    # Drop rows that cannot form a block and images that will not open, before
    # the shard split, so every rank trains on the same-length row list. This
    # happens after the verify gate, which still wants the untouched first row.
    before = len(rows)
    rows = [r for r in rows if len(r["gen_ids"]) >= 2]
    if args.skip_image_verify:
        kept = rows
        say(f"row filter: {before} rows, image verify SKIPPED "
            f"(--skip-image-verify)")
    else:
        kept = []
        for r in rows:
            try:
                with Image.open(r["image"]) as probe:
                    probe.verify()
                kept.append(r)
            except Exception:
                pass
    rows = kept
    say(f"row filter: {before} -> {len(rows)} usable rows")

    rows = rows[rank::world]
    say(f"{len(rows)} training rows on rank {rank}/{world}")

    for param in head.parameters():
        param.requires_grad_(not args.freeze_head)

    trainable = [p for p in head.parameters() if p.requires_grad]
    if markov is not None:
        for param in markov.parameters():
            param.requires_grad_(True)
        trainable += list(markov.parameters())

    # The new module gets its own learning rate. A warm-started backbone and a
    # freshly-initialised head want different step sizes: Adam moves a weight by
    # roughly lr per step, so at the shared 4e-5 a 1000-step epoch shifts the
    # Markov bias by at most ~0.04 from an init std of 0.0058 -- too small to
    # change any argmax (`test_steer_hook.py` needed a 50x lift to move the
    # tree, and the trained-head probe had the spine equal to the unconditioned
    # argmax at 98.3% of positions). Defaulting --markov-lr to --lr keeps Arm M
    # exactly controlled against the baseline; passing a larger value asks the
    # separate question of whether the mechanism works once the bias is
    # actually driven, which is the one thing Arm M cannot distinguish.
    groups = []
    if trainable_head := [p for p in head.parameters() if p.requires_grad]:
        groups.append({"params": trainable_head, "lr": args.lr})
    if markov is not None:
        groups.append({"params": list(markov.parameters()),
                       "lr": args.markov_lr if args.markov_lr else args.lr})
    if not groups:
        raise SystemExit("nothing to train: --freeze-head without a new module")
    optimizer = torch.optim.AdamW(
        groups, lr=args.lr, betas=common.BETAS, weight_decay=common.WEIGHT_DECAY)

    if is_main:
        os.makedirs(args.out, exist_ok=True)
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    log = open(args.log_file, "a") if is_main else None

    step = 0
    micro = 0
    running = 0.0
    offset_acc = torch.zeros(common.BLOCK_SIZE, 2, device=device)
    started = time.time()
    stop = False
    last_good = None
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch * 1000 + rank)
        order = torch.randperm(len(rows), generator=generator).tolist()
        for position in order:
            # The micro counter advances for every row *before* anything can go
            # wrong. Under DDP the accumulation boundary has to be identical on
            # every rank, so no data-dependent `continue` may sit above it:
            # ranks that skipped different numbers of rows would call
            # `optimizer.step()` at different times and the gradient allreduce
            # would wait forever for a peer that is not coming.
            micro += 1
            last_micro = micro % args.accum == 0

            row = rows[position]
            try:
                image = common.load_image(row["image"], args.long_side)
                last_good = (row, image)
            except Exception:
                # Rows are pre-validated before sharding, so this is close to
                # unreachable; if it fires, repeat the previous row rather than
                # skipping, to keep every rank's backward count equal. On the
                # very first row there is nothing to repeat, and a skip there
                # would desynchronise the ranks -- so fail loudly instead of
                # hanging in a collective nobody else will reach.
                if last_good is None:
                    raise RuntimeError(
                        f"first row failed to load ({row['image']}) and there is "
                        "no fallback; pre-validation should have caught this")
                row, image = last_good
            input_ids, pixel_values, grid = common.encode(processor, row["prompt"], image)
            gen = torch.tensor(row["gen_ids"], dtype=torch.long)
            full = torch.cat([input_ids[0], gen])[: args.max_seq_len]
            prompt_len = min(input_ids.shape[1], full.shape[0])
            total = full.shape[0]
            full = full.to(device)

            # M-RoPE positions must be computed explicitly: Qwen3-VL falls back
            # to plain 1-D positions otherwise, which is wrong for image
            # sequences and would train the head on mispositioned context.
            ctx = common.target_context(
                target, full[None],
                pixel_values.to(device) if pixel_values is not None else None,
                grid.to(device) if grid is not None else None)

            # Anchors are the positions decode actually uses as a root: the
            # first is the first *generated* token (`glance/decode.py` sets
            # `out_ids[:, n_in]` from the prefill and starts there), and every
            # later one is a committed generated token. So the first root sits
            # at `prompt_len`, not at the last prompt token. Block position k
            # predicts anchor + k. The tail of a short generation still
            # contributes, through a partial block whose out-of-row offsets
            # are masked.
            first = prompt_len
            last = total - 2
            if last < first:
                # Unreachable: the pre-filter keeps only rows with at least two
                # generated tokens, which makes `last - first >= 0`. Failing
                # beats skipping, because a skip desynchronises the ranks.
                raise RuntimeError(
                    f"generation too short ({len(row['gen_ids'])} tokens) to form "
                    "a block; the pre-filter should have removed this row")
            anchors = list(range(first, last + 1))
            seed = args.seed + position
            reveals = sample_reveals(len(anchors), common.BLOCK_SIZE,
                                     torch.Generator(device="cpu").manual_seed(seed), device,
                                     base=args.prefix_base)
            anchors, reveals = live_anchors(anchors, reveals, total, common.BLOCK_SIZE)
            if not anchors:
                # Every sampled block came out empty: a three-token generation
                # leaves one or two in-bounds offsets, so a reveal of 8 or 14
                # supervises nothing. Force reveal = 1 -- the decode-time reveal
                # -- rather than skipping the row, because a skip would
                # desynchronise the DDP accumulation boundary above.
                #
                # The reveal tensor must be built from the FULL anchor list.
                # Building it from `len(anchors)` here is how this crashed: at
                # this point `anchors` is already empty, so the expression
                # produced ONE reveal for TWO anchors, `live_anchors` indexed a
                # size-1 CUDA tensor with [0, 1], and an out-of-range CUDA
                # index_select raises a device-side assert rather than a Python
                # error. The assert is asynchronous, so it surfaced several
                # hundred steps later inside an unrelated tensor construction
                # and wedged both ranks in NCCL for 10 minutes.
                anchors = list(range(first, last + 1))
                reveals = torch.ones(len(anchors), dtype=torch.long, device=device)
                anchors, reveals = live_anchors(anchors, reveals, total, common.BLOCK_SIZE)
                if not anchors:
                    raise RuntimeError(
                        f"no supervised position for a {len(row['gen_ids'])}-token "
                        "generation even at reveal=1; the pre-filter allows "
                        "gen_ids>=2, so this should be impossible")

            loss, correct, weights, by_offset = sequence_loss(
                head, target, ctx, full, anchors, reveals, embed_weight,
                common.MASK_ID, common.BLOCK_SIZE, device,
                markov=markov, markov_input=args.markov_input)
            (loss / args.accum).backward()
            running += float(loss.detach())
            offset_acc += by_offset

            if last_micro:
                torch.nn.utils.clip_grad_norm_(trainable, common.GRAD_CLIP)
                if ddp:
                    # AVG, so the step matches what one card would do on the
                    # global batch. Every rank reaches this line the same number
                    # of times, which is what keeps the collective in step.
                    import torch.distributed as dist

                    for param in trainable:
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if is_main:
                    seen = offset_acc[:, 1].clamp_min(1.0)
                    per_offset = (offset_acc[:, 0] / seen).tolist()
                    record = {"step": step, "loss": running / args.accum,
                              "top1": float((correct * weights).sum() / weights.sum()),
                              "per_offset_top1": per_offset,
                              "seq_len": total, "anchors": len(anchors),
                              "elapsed_min": (time.time() - started) / 60}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(f"step {step} loss {record['loss']:.4f} "
                          f"top1 {record['top1']:.4f} len {total} anchors {len(anchors)} "
                          f"off1-5 " + " ".join(f"{v:.2f}" for v in per_offset[1:6]) +
                          f" ({record['elapsed_min']:.1f} min)", flush=True)
                    if args.save_every and step % args.save_every == 0:
                        head.save_pretrained(os.path.join(args.out, f"step{step}"))
                        if markov is not None:
                            torch.save(markov.state_dict(),
                                       os.path.join(args.out, f"step{step}", "markov_head.pt"))
                            # Ship the config beside the weights. Without it a
                            # stepN checkpoint is not self-describing, and a
                            # loader that keys off the config would silently
                            # evaluate the head with the bias switched off.
                            with open(os.path.join(args.out, f"step{step}",
                                                   "markov_config.json"), "w") as fh:
                                json.dump({"rank": args.markov_rank,
                                           "input": args.markov_input}, fh, indent=2)
                        processor.tokenizer.save_pretrained(
                            os.path.join(args.out, f"step{step}"))
                running = 0.0
                offset_acc = torch.zeros(common.BLOCK_SIZE, 2, device=device)
                if args.steps and step >= args.steps:
                    stop = True
                    break
        if stop:
            break

    if is_main:
        head.save_pretrained(args.out)
        processor.tokenizer.save_pretrained(args.out)
        if markov is not None:
            torch.save(markov.state_dict(), os.path.join(args.out, "markov_head.pt"))
            with open(os.path.join(args.out, "markov_config.json"), "w") as fh:
                json.dump({"rank": args.markov_rank, "input": args.markov_input}, fh, indent=2)
        print(f"done at step {step}, saved to {args.out}", flush=True)
    if ddp:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
