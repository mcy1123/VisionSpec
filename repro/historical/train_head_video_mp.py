#!/usr/bin/env python3
"""Video SFT with PER-ROW prompts (the multi-prompt variant).

Why a separate file: `train_head_video.py` is the verified single-prompt recipe and
asserts exactly one prompt for the whole pool (`assert len(qs) == 1`). This file
imports the same loss pieces (`train_head.sequence_loss`, `sample_reveals`,
`live_anchors`) and changes only where the prompt comes from: the pool's
`question` column, looked up per row through the unique `task` key
(`<video_name>__p<pid>`). The objective, the reveal prior, the block layout, the
optimizer and the anchor cap are the same functions with the same defaults.

    python train_head_video_mp.py --records .../video_pool_multiprompt_1756.pt \
        --pool .../pool_multiprompt_1996.parquet --init-from .../step1000 \
        --out .../head_video_mp --steps 3000 --save-every 500 --gpu 3
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
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from historical import train_head as th
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--pool", required=True, help="multi-prompt index parquet")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--max-anchors", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.init_from or os.path.join(common.CKPT, "head"), device=dev)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    embed = target.get_input_embeddings().weight

    recs = torch.load(a.records, weights_only=False)["records"]
    pool = pd.read_parquet(a.pool)
    split_of = dict(zip(pool["task"].astype(str), pool["split"].astype(str)))
    pool = pool.set_index("task")
    n_all = len(recs)
    recs = [r for r in recs if split_of.get(r["task"]) == a.split]
    assert recs, f"no records in split {a.split!r}"
    missing = [r["task"] for r in recs if r["task"] not in pool.index]
    assert not missing, f"records missing from the index: {missing[:5]}"
    print(f"train rows: {len(recs)} (of {n_all}) | prompts: "
          f"{pool.loc[[r['task'] for r in recs[:5000]], 'question'].nunique()} unique "
          f"in the first 5000 records", flush=True)
    for r in recs:
        row = pool.loc[r["task"]]
        assert str(row["question"]) and "cache_path" in row, r["task"]
    n_prompt = pool.loc[[r["task"] for r in recs], "question"].nunique()
    # A single-prompt pool is a legitimate use of this trainer (the per-row prompt
    # lookup then returns the same string for every row); the multi-prompt case is
    # what the file was written for, but forbidding the 1-prompt case made the
    # task-pool arms fail on an assertion that is about the pool design, not about
    # correctness. Both are allowed; the count is printed either way.
    assert n_prompt >= 1, n_prompt
    print(f"prompts across the training rows: {n_prompt}", flush=True)

    vtext_cache = {}

    def vtext_for(task):
        if task not in vtext_cache:
            q = str(pool.loc[task]["question"])
            vtext_cache[task] = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"},
                                              {"type": "text", "text": q}]}],
                add_generation_prompt=True, tokenize=False)
        return vtext_cache[task]

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=common.BETAS,
                            weight_decay=common.WEIGHT_DECAY)
    os.makedirs(a.out, exist_ok=True)
    log = open(a.log_file, "a") if a.log_file else None
    step, running, t0 = 0, 0.0, time.time()
    order = rng.permutation(len(recs))
    # Draw cursor. It advances on EVERY iteration, including the iterations that
    # `continue` below because a row's anchors were all filtered out by
    # live_anchors. Indexing by `step` instead made such a row an infinite loop:
    # `continue` skips the `step += 1`, so the same row (and the same empty anchor
    # set) was redrawn forever -- a hang with the process burning a core and no
    # output. No previously trained arm is affected: for a pool where no row is
    # ever skipped, cursor == step and the draw sequence is bit-identical.
    cursor = 0
    skipped = 0
    opt.zero_grad(set_to_none=True)
    while step < a.steps:
        r = recs[int(order[cursor % len(order)])]
        cursor += 1
        p = pool.loc[r["task"]]
        ins = proc(text=[vtext_for(r["task"])], videos=[p["cache_path"]],
                   return_tensors="pt", fps=float(p["fps"]))
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        full = r["input_ids"][: a.max_seq_len].to(dev).long()
        prompt_len = int(min(r["prompt_len"], full.shape[0] - 2))
        total = int(full.shape[0])
        assert int(ins["input_ids"].shape[1]) == int(r["prompt_len"]), (
            f"{r['task']}: encoded prompt {ins['input_ids'].shape[1]} != record "
            f"{r['prompt_len']} -- the prompt string changed after collection")
        ctx = common.target_context(target, full[None],
                                    pixel_values_videos=pv, video_grid_thw=vg)
        anchors = list(range(prompt_len, total - 1))
        reveals = th.sample_reveals(len(anchors), head.block_size,
                                    torch.Generator().manual_seed(a.seed + step), "cpu")
        anchors, reveals = th.live_anchors(anchors, reveals, total, head.block_size)
        if not anchors:
            # loud, not silent: if a whole sweep of the pool trains nothing, stop.
            skipped += 1
            if skipped > 20 * len(order):
                raise RuntimeError(
                    f"no trainable record in the pool: {skipped} consecutive rows "
                    f"yielded an empty anchor set (last: {r['task']}, "
                    f"seq_len {total}, prompt_len {prompt_len})")
            continue
        skipped = 0
        if a.max_anchors and len(anchors) > a.max_anchors:
            keep = rng.choice(len(anchors), a.max_anchors, replace=False)
            keep.sort()
            anchors = [anchors[i] for i in keep]
            reveals = reveals[torch.as_tensor(keep)]
        loss, _c, _w, _bo = th.sequence_loss(
            head, target, ctx, full, anchors, reveals.to(dev), embed,
            common.MASK_ID, head.block_size, dev)
        (loss / a.accum).backward()
        running += float(loss.detach())
        if (step + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), common.GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0 or step == 1:
            el = time.time() - t0
            msg = (f"step {step}/{a.steps} loss {running/max(a.log_every,1):.4f} "
                   f"L {total} anchors {len(anchors)} {el/step:.2f}s/step "
                   f"({el/60:.1f} min)")
            print(msg, flush=True)
            if log:
                log.write(json.dumps({"step": step, "loss": running/max(a.log_every, 1),
                                      "L": total, "anchors": len(anchors),
                                      "s_per_step": el/step, "elapsed_min": el/60}) + "\n")
                log.flush()
            running = 0.0
        if a.save_every and step % a.save_every == 0:
            head.save_pretrained(os.path.join(a.out, f"step{step}"))
            print(f"  saved step{step}", flush=True)
    head.save_pretrained(a.out)
    print(f"done at step {step} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
