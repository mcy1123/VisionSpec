#!/usr/bin/env python3
"""Collect the OPD pool: tree-rollout anchors + the target's top-k.

What the pool is for. SFT trains the head at anchors taken from the TARGET's
greedy trajectory, with the block filled from the target's own continuation.
OPD trains it at anchors the DRAFTER actually reaches during tree-verified
speculative decoding, with the block filled from what the drafter proposed. That
shift in the state distribution is the method.

**Why this uses the real `glance.generate` instead of a head-only rollout.** The
reference implementation (`opd_multimodal/opd_train_vlm.py::rollout_tree`) can
roll out with the drafter alone, because it verifies losslessness by tree-walking
against a pre-collected trajectory. This port first did the same and it was
WRONG: the reimplemented block pass disagreed with the real decoder (round 0:
accepted 1 vs the decoder's 4; 89 anchors vs 78) and so built the pool from
states the model never visits -- the one thing the method depends on. Two causes,
found by direct measurement rather than inspection:

  1. `common.target_context(target, ids)` without `pixel_values` runs the target
     as a plain text model over image-placeholder tokens. On one row:
     ||delta|| / ||with|| = 0.59, max|delta| 1327 -- while the cosine over the
     generated region stayed 0.99, so it looked benign but broke exact argmax
     matches.
  2. The head call is convention-heavy (`position_ids` sliced off a KV cache,
     `is_causal=False`, cache crop). Re-deriving it is exactly how this project
     previously shipped a convention bug that reversed a conclusion.

So the anchors come from the decoder that actually runs, via the `on_round`
payload (`pos`, `accepted`, `path_tokens`, `argmax_tokens`).

`prop` semantics (subtle, and easy to get wrong): the ACCEPTED offsets must carry
the token the tree committed, which can be a non-argmax candidate -- not the
per-offset argmax. Only the rejected tail uses the argmax, because there the
committed path ran out. Scoring the forward KL on the argmax would train the head
on tokens it never proposed.

    python opd_rollout.py --limit 200 --out $GLANCE_ROOT/data/opd_pool.pt \
        --data $GLANCE_ROOT/data/target_greedy.jsonl --gpu 3
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
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))

from shared import common

sys.path.insert(0, os.path.join(common.WORKSPACE, "GLANCE_repo"))

import glance
from glance.vl import VisionPositions


def anchors_from_rounds(rounds, block_size):
    """Turn the decoder's per-round payload into OPD anchors.

    Accepted offsets take the committed tree path; the rejected tail takes the
    per-offset argmax. Model-free, so it is unit-tested in
    `test_opd_convention.py`.
    """
    n_prop = block_size - 1
    out = []
    for r in rounds:
        path = list(r["path_tokens"])
        argmax = list(r["argmax_tokens"])
        n_acc = int(r["accepted"])
        prop = (path + argmax[n_acc:])[:n_prop]
        if not prop:
            continue
        out.append({"pos": int(r["pos"]), "prop": prop, "n_acc": n_acc})
    return out


def target_topk(target, full_ids, k, pixel_values=None, image_grid_thw=None,
                pixel_values_videos=None, video_grid_thw=None, chunk=64):
    """Top-k next-token distribution at every position, in one target forward.

    Returns `(topk_ids, topk_val)` of shape `(S-1, k)`; row `r` predicts
    `full_ids[r+1]`, the official convention `opd_loss.teacher_rows` encodes.
    Chunked so the full `(S, V)` tensor is never materialised, and the image is
    passed -- omitting it silently changes the distribution.
    """
    kwargs = dict(input_ids=full_ids, use_cache=False, logits_to_keep=0)
    if pixel_values is not None:
        kwargs.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        kwargs["position_ids"] = common.target_positions(
            target, full_ids, image_grid_thw)
    if pixel_values_videos is not None:
        kwargs.update(pixel_values_videos=pixel_values_videos,
                      video_grid_thw=video_grid_thw)
        pos, _ = target.model.get_rope_index(full_ids, None, video_grid_thw, None)
        kwargs["position_ids"] = pos
    with torch.no_grad():
        out = target(**kwargs)
    logits = out.logits[0]                      # (S, V)
    S = logits.shape[0]
    ids_out, val_out = [], []
    for lo in range(0, S - 1, chunk):
        hi = min(lo + chunk, S - 1)
        lp = F.log_softmax(logits[lo:hi].float(), dim=-1)
        v, i = torch.topk(lp, k, dim=-1)
        ids_out.append(i.to(torch.int32).cpu())
        val_out.append(v.to(torch.float16).cpu())
        del lp
    return torch.cat(ids_out), torch.cat(val_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(common.DATA, "target_greedy.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default=os.path.join(common.CKPT, "head"),
                    help="the drafter whose proposals define the pool")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--topk", type=int, default=64,
                    help="the reference measured 18%% of positions outside top-32")
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch-k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    # Default matches the SFT recipe's train-side size, so the OPD arm differs
    # from the continue-SFT control in the training signal alone (the eval side
    # is 896 for both).
    ap.add_argument("--max-pixels", type=int, default=common.IMAGE_LONG_SIDE)
    ap.add_argument("--tasks", default="",
                    help="comma list to filter tasks, e.g. textvqa; empty = all")
    ap.add_argument("--cap-tokens", type=int, default=0,
                    help="truncate each trajectory to this many generated tokens")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--video-parquet", default=None)
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--video-frames", type=int, default=16)
    # Cache mode: read the PRE-NORMALIZED clip path and its fps straight from the
    # pool parquet instead of deriving them from video_name + a source directory.
    # That is the whole point of the cache: frame count is fixed at build time, so
    # the collector cannot accidentally re-derive a different count from the
    # source's duration.
    ap.add_argument("--video-cache", default=None,
                    help="pool parquet produced by predecode_cache.py; when set, "
                         "cache_path and fps are taken from it verbatim")
    # The video pool carries ONE prompt for every row, and the trainer/evaluator
    # both use that template. Rolling the teacher out under the dataset's own
    # wording instead ("background scenes" vs "overall scene") put a token-level
    # mismatch between the collected labels and everything downstream of them.
    ap.add_argument("--video-prompt", default=None,
                    help="override the per-row question for the video branches; "
                         "must match the prompt used at train/eval time")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    target, processor = common.load_target(device=device)
    head = common.load_head(args.head, device=device).eval()
    stop = [processor.tokenizer.eos_token_id]
    print(f"head {args.head}: block {head.block_size}, budget {args.budget}, "
          f"max_pixels {args.max_pixels}", flush=True)

    records = []
    started = time.time()
    n = 0
    video_mode = bool(args.video_parquet) or bool(args.video_cache)
    cache_mode = bool(args.video_cache)
    if cache_mode:
        args.video_parquet = args.video_cache
    if video_mode:
        import pandas as _pd
        _df = _pd.read_parquet(args.video_parquet)
        rows_iter = _df.to_dict("records")
        if args.limit:
            rows_iter = rows_iter[: args.limit]
        print(f"VIDEO mode: {len(rows_iter)} rows, {args.video_frames} frames", flush=True)
    else:
        rows_iter = common.read_jsonl(args.data)
    for row in rows_iter:
        if n >= args.limit:
            break
        if not video_mode:
            if not os.path.exists(row["image"]):
                continue
            if args.tasks and row.get("task") not in args.tasks.split(","):
                continue
        n += 1
        if video_mode:
            _vq = args.video_prompt or row["question"]
        if video_mode and cache_mode:
            # Cache mode: path and fps come from the index verbatim, so the frame
            # count cannot be re-derived from the source duration.
            vpath = row["cache_path"]
            vfps = float(row["fps"])
            if not os.path.exists(vpath):
                raise FileNotFoundError(f"cache_path missing: {vpath}")
            vtext = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"},
                                              {"type": "text", "text": _vq}]}],
                add_generation_prompt=True, tokenize=False)
            _ins = processor(text=[vtext], videos=[vpath], return_tensors="pt", fps=vfps)
            ids = _ins["input_ids"].to(device)
            pv = _ins["pixel_values_videos"].to(device)
            vg = _ins["video_grid_thw"].to(device)
            vpos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
        elif video_mode:
            import av as _av
            # parquet stores video_name without an extension; the files on disk
            # are .mp4 or .mkv. Resolve explicitly and fail loudly otherwise.
            vpath = os.path.join(args.video_root, row["video_name"])
            if not os.path.exists(vpath):
                for _e in (".mp4", ".mkv", ".webm"):
                    if os.path.exists(vpath + _e):
                        vpath = vpath + _e
                        break
                else:
                    raise FileNotFoundError(
                        f"no video for {row['video_name']!r} under {args.video_root}")
            with _av.open(vpath) as _c:
                vdur = float(_c.duration) / _av.time_base
            vtext = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"},
                                              {"type": "text", "text": _vq}]}],
                add_generation_prompt=True, tokenize=False)
            _ins = processor(text=[vtext], videos=[vpath], return_tensors="pt",
                             fps=args.video_frames / vdur)
            ids = _ins["input_ids"].to(device)
            pv = _ins["pixel_values_videos"].to(device)
            vg = _ins["video_grid_thw"].to(device)
            vpos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
        else:
            image = common.load_image(row["image"], args.max_pixels)
            ids, pix, grid = common.encode(processor, row["prompt"], image)
            ids = ids.to(device)
            pix = pix.to(device) if pix is not None else None
            grid = grid.to(device) if grid is not None else None
            vpos = VisionPositions(pix, grid)

        max_new = args.cap_tokens or args.max_new
        rounds = []
        out, stats = glance.generate(
            head, target, ids, max_new, stop, budget=args.budget,
            branch_k=args.branch_k, temperature=0.0, positions=vpos,
            return_stats=True, timing="decode_only", on_round=rounds.append)
        full = out[0]
        anchors = anchors_from_rounds(rounds, head.block_size)
        if not anchors:
            continue
        if video_mode:
            tk_ids, tk_val = target_topk(target, full[None], args.topk, None, None, pv, vg)
        else:
            tk_ids, tk_val = target_topk(target, full[None], args.topk, pix, grid)
        records.append({
            "row": n - 1, "task": row.get("task", row.get("video_name")),
            "prompt_len": int(ids.shape[1]), "seq_len": int(full.shape[0]),
            "tau": float(stats.tau),
            "input_ids": full.cpu().to(torch.int32),
            "anchors": anchors, "topk_ids": tk_ids, "topk_val": tk_val,
        })
        if n % 10 == 0:
            rate = n / (time.time() - started)
            print(f"  {n}/{args.limit} rows, {rate:.3f} rows/s, "
                  f"eta {(args.limit-n)/max(rate,1e-9)/60:.1f} min", flush=True)

    tot_a = sum(len(r["anchors"]) for r in records)
    rej = sum(1 for r in records for a in r["anchors"] if a["n_acc"] < len(a["prop"]))
    acc = sum(a["n_acc"] for r in records for a in r["anchors"])
    print(f"\n{len(records)} rows, {tot_a} anchors "
          f"({tot_a/max(len(records),1):.1f} per row), accept/round "
          f"{acc/max(tot_a,1):.3f}, tau {acc/max(tot_a,1)+1:.3f}")
    print(f"mean decoder tau over rows: "
          f"{sum(r['tau'] for r in records)/max(len(records),1):.3f}")
    print(f"anchors with a rejection: {rej}/{tot_a} "
          f"({100.0*rej/max(tot_a,1):.1f}%)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"records": records, "config": vars(args)}, args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
