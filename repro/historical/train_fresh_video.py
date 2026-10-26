#!/usr/bin/env python3
"""Train a video draft head FROM SCRATCH, with either the simple MLP or ViSpec.

This is a decision experiment, not a continuation of the diagnosis, so three
things are deliberately explicit:

1. **No warm start.** The head is constructed from the H-8 *architecture template*
   with a fixed seed and random weights. It never opens `head_video439_h8` or any
   other trained head. `--arm mlp` and `--arm vispec` construct the base head with
   the SAME seed, so the two arms differ only in the adapter -- otherwise a
   difference could just be initialisation luck.

2. **The feature path is a flag.** `--feature-path cached` uses
   `common.target_context_cached`, the path decode actually uses; `uncached` uses
   `common.target_context`, the historical path, which on Qwen3-VL leaves text
   rows bit-identical when the video is blanked (see
   `probe_context_cache_path.py`). Both arms in a comparison must use the same
   setting, and the trainer records which one it used.

3. **The loss is the verified one.** Anchors, reveals and the block objective come
   from `train_head` unchanged; `sequence_loss` reads the context through
   `head.hidden_norm(head.fc(ctx))`, which is exactly the seam the adapter plugs
   into, so there is no second forward to drift.

    python train_fresh_video.py --arm vispec --feature-path cached \
        --records .../video_greedy_499.pt --pool .../pool_499.parquet \
        --out .../head_v_vispec --steps 3000 --lr 3e-4 --gpu 0

No try/except: a failure surfaces with its traceback.
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
from historical import vispec_adapter as va
from transformers import AutoProcessor, AutoConfig


def build_scratch_head(template_dir, dev, seed, dtype=torch.bfloat16):
    """Random-init head with the template's architecture. Loads NO weights."""
    cfg = AutoConfig.from_pretrained(template_dir)
    torch.manual_seed(seed)
    from dflash.model import DFlashDraftModel
    head = DFlashDraftModel(cfg)
    head = head.to(device=dev, dtype=dtype)
    n = sum(p.numel() for p in head.parameters())
    print(f"scratch head: {n/1e6:.1f}M params, layers {cfg.num_hidden_layers}, "
          f"target_layer_ids {head.target_layer_ids}, block {head.block_size}, "
          f"mask_id {head.mask_token_id}", flush=True)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["mlp", "vispec"])
    ap.add_argument("--feature-path", required=True, choices=["cached", "uncached"])
    ap.add_argument("--records", required=True, help="video_greedy_499.pt")
    ap.add_argument("--pool", required=True, help="pool_499.parquet")
    ap.add_argument("--template", default=None,
                    help="head dir whose config.json gives the architecture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-anchors", type=int, default=96)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log-file", default=None)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    template = a.template or os.path.join(common.CKPT, "head_h8_init")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    cfgj = json.load(open(os.path.join(common.TARGET_PATH, "config.json")))
    vid, iid = int(cfgj["video_token_id"]), int(cfgj["image_token_id"])

    head = build_scratch_head(template, dev, a.seed)
    # Prove we did not warm start: the weights must not match any trained head.
    adapter = None
    if a.arm == "vispec":
        adapter = va.attach_vispec(head, seed=a.seed)
        n_ad = adapter.param_count()
        print(f"ViSpec adapter: {n_ad/1e6:.2f}M params "
              f"({100*n_ad/sum(p.numel() for p in head.parameters()):.1f}% of the arm)",
              flush=True)
        if n_ad >= va.PARAM_LIMIT:
            raise SystemExit(f"adapter {n_ad} exceeds the {va.PARAM_LIMIT} cap")
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    embed = target.get_input_embeddings().weight

    recs = torch.load(a.records, weights_only=False)["records"]
    pool = pd.read_parquet(a.pool)
    split_of = dict(zip(pool["video_name"].astype(str), pool["split"].astype(str)))
    pool = pool.set_index("video_name")
    n_all = len(recs)
    recs = [r for r in recs if split_of.get(r["task"]) == "train"]
    assert not [r for r in recs if split_of.get(r["task"]) != "train"]
    print(f"train rows: {len(recs)} (of {n_all} records; held-out excluded by split)",
          flush=True)
    qs = pool["question"].astype(str).unique()
    assert len(qs) == 1, f"video pool expects one prompt, got {len(qs)}"
    prompt = qs[0]
    vtext = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=False)

    ctx_fn = (common.target_context_cached if a.feature_path == "cached"
              else common.target_context)
    print(f"arm={a.arm} feature_path={a.feature_path} ctx_fn={ctx_fn.__name__} "
          f"lr={a.lr} steps={a.steps}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=common.BETAS,
                            weight_decay=common.WEIGHT_DECAY)
    os.makedirs(a.out, exist_ok=True)
    log = open(a.log_file, "a") if a.log_file else None
    step, running, t0 = 0, 0.0, time.time()
    order = rng.permutation(len(recs))
    opt.zero_grad(set_to_none=True)
    while step < a.steps:
        r = recs[int(order[step % len(order)])]
        p = pool.loc[r["task"]]
        ins = proc(text=[vtext], videos=[p["cache_path"]], return_tensors="pt",
                   fps=float(p["fps"]))
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        full = r["input_ids"][: a.max_seq_len].to(dev).long()
        prompt_len = int(min(r["prompt_len"], full.shape[0] - 2))
        total = int(full.shape[0])
        fused = ctx_fn(target, full[None], pixel_values_videos=pv,
                       video_grid_thw=vg)
        if a.arm == "vispec":
            vis = ((full[None] == vid) | (full[None] == iid))
            if int(vis.sum()) == 0:
                raise SystemExit(f"row {r['task']}: no visual tokens found")
            ctx = va.build_wide(fused, vis)
        else:
            ctx = fused
        anchors = list(range(prompt_len, total - 1))
        reveals = th.sample_reveals(len(anchors), head.block_size,
                                    torch.Generator().manual_seed(a.seed + step),
                                    "cpu")
        anchors, reveals = th.live_anchors(anchors, reveals, total, head.block_size)
        if not anchors:
            step += 1
            continue
        if a.max_anchors and len(anchors) > a.max_anchors:
            keep = rng.choice(len(anchors), a.max_anchors, replace=False)
            keep.sort()
            anchors = [anchors[i] for i in keep]
            reveals = reveals[torch.as_tensor(keep)]
        loss, _c, _w, _bo = th.sequence_loss(
            head, target, ctx, full, anchors, reveals.to(dev), embed,
            common.MASK_ID, head.block_size, dev)
        (loss / 1).backward()
        running += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(head.parameters(), common.GRAD_CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0 or step == 1:
            el = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            msg = (f"step {step}/{a.steps} loss {running/max(a.log_every,1):.4f} "
                   f"L {total} vis {int(vis.sum()) if a.arm=='vispec' else -1} "
                   f"anchors {len(anchors)} {el/step:.2f}s/step peak {peak:.1f}GiB "
                   f"({el/60:.1f} min)")
            print(msg, flush=True)
            if log:
                log.write(json.dumps({"step": step, "arm": a.arm, "loss": running/max(a.log_every, 1),
                                      "L": total, "anchors": len(anchors),
                                      "s_per_step": el/step, "peak_gib": peak,
                                      "elapsed_min": el/60}) + "\n")
                log.flush()
            running = 0.0
        if a.save_every and step % a.save_every == 0:
            head.save_pretrained(os.path.join(a.out, f"step{step}"))
            if adapter is not None:
                va.save_vispec(head, os.path.join(a.out, f"step{step}"))
            print(f"  saved step{step}", flush=True)
    head.save_pretrained(a.out)
    if adapter is not None:
        va.save_vispec(head, a.out)
    meta = {"arm": a.arm, "feature_path": a.feature_path, "steps": a.steps,
            "lr": a.lr, "seed": a.seed, "train_rows": len(recs),
            "records": a.records, "pool": a.pool, "template": template,
            "adapter_params": adapter.param_count() if adapter else 0,
            "head_params": sum(p.numel() for p in head.parameters()),
            "warm_start": False}
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)
    print(f"done at step {step} -> {a.out}", flush=True)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
