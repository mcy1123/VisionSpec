#!/usr/bin/env python3
"""Video SFT for the block head, reusing the image trainer's loss machinery.

Why a separate file instead of a branch inside `train_head.py`. `train_head.py`
is the verified image-side recipe (its `verify()` gate, its row filter, its DDP
sharding, its reveal schedule). Adding a second data source inside it would put
video code on the path of the artifact every image-side number in this project
depends on. This file therefore imports that trainer's loss pieces
(`sequence_loss`, `sample_reveals`, `live_anchors`) and changes only where the
data comes from -- the objective, the reveal prior and the block layout are
literally the same functions.

Labels are the TARGET's own greedy continuation, taken from the pool collector's
rollout (`video_greedy_150.pt`), NOT the dataset's reference `answer`. GLANCE's
recipe trains on the target's generations; the dataset answer is a different
object and using it would silently turn this into plain caption distillation.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")
from shared import common
from historical import train_head as th
from transformers import AutoProcessor


class _LmHeadShim:
    """`sequence_loss` only ever reads `target.lm_head`, and calls it as
    `lm_head(hidden)` (via `head.compute_logits`). When the target is fp32 that
    projection rejects the head's bf16 hidden rows, so expose a callable that
    casts the input first. The logits still go through `cross_entropy(logits
    .float(), ...)` inside `sequence_loss`, so the objective's precision is set by
    that cast, not by this one."""

    def __init__(self, target, head_dtype):
        self._w = target.lm_head
        self._w_dtype = target.lm_head.weight.dtype
        self._head_dtype = head_dtype

    def lm_head(self, hidden):
        if hidden.dtype != self._w_dtype:
            hidden = hidden.to(self._w_dtype)
        return self._w(hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="video_greedy_150.pt")
    ap.add_argument("--pool", required=True, help="pool_150.parquet")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--out", required=True)
    # Which target feature path the head is trained on (2026-10-09, Phase A4).
    # `uncached` is `common.target_context` -- `use_cache=False` with no explicit
    # mask, under which the TEXT-position hidden states are bit-identical no
    # matter what the video is (`check_context_interface.py`: max|delta| exactly
    # 0.000e+00 on 3 rows). Every video head before A4 was trained that way while
    # `glance.decode` prefills through the KV cache, so training and deployment
    # disagreed about where the visual information lives. `cached` is
    # `common.target_context_cached`, which is bit-for-bit identical to
    # `aligned_forward.target_context_aligned` (max|delta| 0.000e+00) and
    # therefore to the decode prefill. Default stays `uncached` so every existing
    # historical number remains reproducible.
    ap.add_argument("--target-dtype", default="bfloat16",
                    choices=["bfloat16", "float32"],
                    help="dtype of the FROZEN TARGET, i.e. the numerical path that "
                         "produces the head's context. It must match the OPD arm's "
                         "`--teacher-dtype`, or the two arms differ in their context "
                         "features as well as in their loss, and the comparison stops "
                         "being 'only the loss differs'")
    ap.add_argument("--master-weights", default="none",
                    choices=["none", "cpu", "gpu"],
                    help="none = historical bf16 AdamW (default, bit-identical); "
                         "cpu/gpu = fp32 master params + fp32 Adam state, stepped "
                         "in fp32 and written back into the bf16 params. Required "
                         "for small learning rates: at lr 3e-5 the bf16 path "
                         "realises only 0.223 of the intended update norm")
    ap.add_argument("--feature-path", default="uncached",
                    choices=["uncached", "cached"],
                    help="uncached = historical path (default); cached = "
                         "decode-aligned path, must match OPD/continue-SFT arms")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=1)
    # Memory, not recipe: a video row has L~1136 and ~370 usable anchors (an image
    # row has L~532 and ~250), and the CE logits alone are
    # 370*15*151936*4B = 3.4 GB in fp32. Capping anchors per step bounds the
    # block tensor without touching the objective, the reveal prior or the block
    # layout -- those stay the identical train_head functions.
    ap.add_argument("--max-anchors", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log-file", default=None)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    target, _ = common.load_target(
        device=dev,
        dtype=torch.float32 if a.target_dtype == "float32" else torch.bfloat16)
    print(f"target dtype (head context path): {next(target.parameters()).dtype}",
          flush=True)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    ctx_fn = (common.target_context_cached if a.feature_path == "cached"
              else common.target_context)
    print(f"feature path: {a.feature_path} ({ctx_fn.__name__})", flush=True)
    head = common.load_head(a.init_from or os.path.join(common.CKPT, "head"),
                            device=dev)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    head_dtype = next(head.parameters()).dtype
    embed = target.get_input_embeddings().weight
    if embed.dtype != head_dtype:
        # One cast outside the loop, not per step: the table is 151936x4096.
        embed = embed.to(head_dtype)

    recs = torch.load(a.records, weights_only=False)["records"]
    import pandas as pd
    pool = pd.read_parquet(a.pool)
    split_of = dict(zip(pool["video_name"].astype(str), pool["split"].astype(str)))
    # the per-row lookup below is by video_name; `split_of` above is a plain dict
    # so that the split filter does not depend on the index
    pool = pool.set_index("video_name")
    # keep only rows present in both (the pool index carries cache_path/fps) AND
    # in the train split. Without the split filter the held-out rows are trained
    # on and every held-out number becomes an upper bound on memorisation, not a
    # generalisation estimate -- the first video run (log 72) had exactly that
    # bug: it printed "train rows: 150" for a 120/30 pool.
    n_all = len(recs)
    recs = [r for r in recs
            if split_of.get(r["task"]) == "train"]
    leaked = [r["task"] for r in recs if split_of.get(r["task"]) != "train"]
    assert not leaked, f"held-out rows leaked into training: {leaked[:5]}"
    print(f"train rows: {len(recs)} (of {n_all} records; held-out excluded "
          f"by split)", flush=True)
    qs = pool["question"].astype(str).unique()
    assert len(qs) == 1, f"video pool expects one prompt, got {len(qs)}"
    prompt = qs[0]
    print(f"prompt (from pool): {prompt!r}", flush=True)
    vtext = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=False)

    # `--master-weights` (2026-10-10, additive; default `none` = the exact
    # historical optimizer). The head is loaded in bf16, so a plain
    # `AdamW(head.parameters())` keeps Adam's moments in bf16 and applies the
    # update with a bf16 `addcdiv_`; a step smaller than half an ulp then rounds
    # to a bit-identical parameter and, with no fp32 master, never accumulates.
    # Measured at lr 3e-5: the realized update was 0.223 of the fp32 ideal and
    # 96.9% of elements received a sub-half-ulp update
    # (`repro/diagnostics/diag_update_precision.py`). See `repro/shared/master_optim.py`.
    if a.master_weights == "none":
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=common.BETAS,
                                weight_decay=common.WEIGHT_DECAY)
        master_opt = None
    else:
        from shared.master_optim import MasterWeightAdamW
        master_opt = MasterWeightAdamW(
            head, lr=a.lr, betas=common.BETAS,
            weight_decay=common.WEIGHT_DECAY, where=a.master_weights,
            clip=common.GRAD_CLIP)
        opt = None
        print(f"optimizer: fp32 master weights on {a.master_weights} "
              f"({master_opt.master_state_bytes()/2**30:.2f} GiB of fp32 params "
              f"+ Adam state)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    log = open(a.log_file, "a") if a.log_file else None
    step, running, t0 = 0, 0.0, time.time()
    order = rng.permutation(len(recs))
    (master_opt or opt).zero_grad(set_to_none=True)
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
        ctx = ctx_fn(target, full[None], pixel_values_videos=pv,
                     video_grid_thw=vg)
        # `--target-dtype float32` makes the context and the embedding table fp32
        # while the head is bf16; both are head INPUTS, so they are cast into the
        # head's dtype here. The KL/CE is unaffected: `sequence_loss` already
        # computes `cross_entropy(logits.float(), ...)`.
        if ctx.dtype != head_dtype:
            ctx = ctx.to(head_dtype)
        anchors = list(range(prompt_len, total - 1))
        reveals = th.sample_reveals(len(anchors), head.block_size,
                                    torch.Generator().manual_seed(a.seed + step),
                                    "cpu")
        anchors, reveals = th.live_anchors(anchors, reveals, total, head.block_size)
        if not anchors:
            continue
        if a.max_anchors and len(anchors) > a.max_anchors:
            keep = rng.choice(len(anchors), a.max_anchors, replace=False)
            keep.sort()
            anchors = [anchors[i] for i in keep]
            reveals = reveals[torch.as_tensor(keep)]
        # `sequence_loss` projects the head's hidden rows with `target.lm_head`,
        # which is fp32 when the target is fp32. It accepts any object exposing
        # `lm_head`, so a casting shim keeps the projection legal without editing
        # the shared, verified `train_head.sequence_loss`.
        loss, _c, _w, _bo = th.sequence_loss(
            head, _LmHeadShim(target, head_dtype), ctx, full, anchors,
            reveals.to(dev), embed, common.MASK_ID, head.block_size, dev)
        (loss / a.accum).backward()
        running += float(loss.detach())
        if (step + 1) % a.accum == 0:
            if master_opt is None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), common.GRAD_CLIP)
                opt.step(); opt.zero_grad(set_to_none=True)
            else:
                # Clipping happens inside the master step, in fp32, on the fp32
                # gradients -- clipping bf16 grads vs fp32 grads is a different
                # number, so the master path does it itself rather than here.
                master_opt.step(); master_opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0 or step == 1:
            el = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            msg = (f"step {step}/{a.steps} loss {running/max(a.log_every,1):.4f} "
                   f"L {total} anchors {len(anchors)} "
                   f"{el/step:.2f}s/step peak {peak:.1f}GiB "
                   f"({el/60:.1f} min)")
            print(msg, flush=True)
            if log:
                log.write(json.dumps({"step": step, "loss": running/max(a.log_every,1),
                                      "L": total, "anchors": len(anchors),
                                      "s_per_step": el/step, "peak_gib": peak,
                                      "elapsed_min": el/60}) + "\n"); log.flush()
            running = 0.0
        if a.save_every and step % a.save_every == 0:
            head.save_pretrained(os.path.join(a.out, f"step{step}"))
            print(f"  saved step{step}", flush=True)
    head.save_pretrained(a.out)
    print(f"done at step {step} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
