#!/usr/bin/env python3
"""Video OPD: the image-side on-policy replay recipe on the video block head.

Warm start `head_video439/step1000`; train on the 409 rows left after removing
the 30-row independent held-out (`data/video_heldout_30.parquet`); anchors come
from the drafter's own tree-verified rollout over the same videos, collected with
that same warm-start head (`opd_rollout.py`, see scripts/80).

What differs from `train_head_video.py` (the SFT arm): the blocks are filled from
the DRAFTER's proposals at the states the decoder actually visits, instead of
from the target's continuation at anchors on the target's own trajectory. The
student forward, the offset convention and the pool layout are not re-derived --
they are `train_head.forward_anchors` / `opd_loss.opd_blocks` /
`opd_loss.teacher_rows` (see `opd_loss_video.py`).

    python opd_train_video.py --records .../video_opd_pool_warm_499.pt \
        --pool .../pool_499.parquet --holdout-parquet .../video_heldout_30.parquet \
        --init-from .../head_video439/step1000 --out .../head_opd_video409 \
        --steps 2000 --save-every 250 --max-anchors 96 --lr 3e-4 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

import common
import opd_loss_video
import opd_online_rollout
from aligned_forward import target_context_aligned, verify_aligned_to_decode
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="OPD pool .pt (anchors+topk)")
    ap.add_argument("--pool", required=True, help="pool parquet (cache_path/fps/split)")
    ap.add_argument("--holdout-parquet", default=None,
                    help="rows to exclude from training even when split=='train'")
    ap.add_argument("--init-from", required=True,
                    help="warm start, e.g. checkpoints/head_video439/step1000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--max-anchors", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta-fwd", type=float, default=1.0)
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--alpha-rev", type=float, default=0.0,
                    help="multiplies the reverse-KL stream. With the recipe's "
                         "--reject-target topk that stream is EMPTY by "
                         "construction, so this is a no-op; it is kept as an "
                         "explicit knob and the run prints n_k3_rej=0.")
    ap.add_argument("--reject-target", default="topk", choices=["topk", "draft"])
    ap.add_argument("--drop-inexact", action="store_true", default=False,
                    help="image-side recipe ships with this OFF (no drop)")
    ap.add_argument("--drop-inexact-teacher", dest="drop_inexact",
                    action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--no-final-save", action="store_true",
                    help="run diagnostics without writing a final checkpoint")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--cache-ctx", action="store_true",
                    help="precompute every row's target context into RAM")
    ap.add_argument("--deterministic", action="store_true",
                    help="CUBLAS_WORKSPACE_CONFIG + use_deterministic_algorithms("
                         "warn_only=True), the same form the image-side trainer "
                         "uses")
    ap.add_argument("--aligned-forward", action="store_true",
                    help="build the training context on the SAME attention branch "
                         "every decode path uses (2-D all-visible mask -> SDPA "
                         "is_causal), which is bit-identical to the decode prefill. "
                         "See aligned_forward.py for the measurement.")
    ap.add_argument("--rollout-mode", default="stored",
                    choices=["stored", "online", "frozen"],
                    help="stored=replay anchors in --records (legacy); "
                         "online=re-roll proposals with the current head every "
                         "step; frozen=re-roll with an immutable copy of "
                         "--init-from (matched static-policy control)")
    ap.add_argument("--verify", default="tree", choices=["chain", "tree"],
                    help="verification topology used by online/frozen rollout")
    ap.add_argument("--tree-budget", type=int, default=63)
    ap.add_argument("--branch-k", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=0,
                    help="cap sequential rollout rounds; 0 means full response")
    a = ap.parse_args()

    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[deterministic] use_deterministic_algorithms(True, warn_only=True)",
              flush=True)

    dev = f"cuda:{a.gpu}"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.init_from, device=dev)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    embed = target.get_input_embeddings().weight

    rollout_head = None
    if a.rollout_mode == "frozen":
        rollout_head = common.load_head(a.init_from, device=dev).eval()
        rollout_head.requires_grad_(False)
        print("rollout policy: immutable init checkpoint", flush=True)
    elif a.rollout_mode == "online":
        print("rollout policy: current train head (refreshed every step)", flush=True)
    else:
        if a.verify != "tree":
            raise ValueError("--rollout-mode stored contains tree anchors; use "
                             "--rollout-mode frozen for a static chain control")
        print("rollout policy: stored anchors", flush=True)

    blob = torch.load(a.records, weights_only=False)
    recs_all = blob["records"]
    pool = pd.read_parquet(a.pool)
    # Multi-prompt pools key rows by `task` = "<video>__p<pid>" and repeat
    # video_name; single-prompt pools key by video_name. Records always carry a
    # unique `task`, so index by that when the column exists.
    key = "task" if "task" in pool.columns else "video_name"
    split_of = dict(zip(pool[key].astype(str), pool["split"].astype(str)))
    pool = pool.set_index(key)
    print(f"pool index key: {key!r}", flush=True)

    excluded = set()
    if a.holdout_parquet:
        _h = pd.read_parquet(a.holdout_parquet)
        _hcol = "task" if "task" in _h.columns else "video_name"
        excluded = set(_h[_hcol].astype(str))
    recs = [r for r in recs_all
            if split_of.get(r["task"]) == "train" and r["task"] not in excluded]
    leaked = [r["task"] for r in recs if r["task"] in excluded]
    assert not leaked, f"held-out rows leaked into OPD training: {leaked[:5]}"
    n_expect = int((pool["split"] == "train").sum()) - len(excluded)
    assert len(recs) == n_expect, (
        f"OPD train rows {len(recs)} != expected {n_expect} "
        f"({int((pool['split'] == 'train').sum())} train - {len(excluded)} held-out)")
    print(f"OPD train rows: {len(recs)} of {len(recs_all)} pool records "
          f"(split=train, {len(excluded)} held-out rows excluded)", flush=True)

    qs = pool["question"].astype(str).unique()
    if len(qs) == 1:
        print(f"prompt (from pool, single): {qs[0]!r}", flush=True)
    else:
        n_p = pool.loc[[r["task"] for r in recs], "question"].nunique() \
            if key == "task" else len(qs)
        print(f"multi-prompt pool: {len(qs)} distinct prompts ({n_p} among the "
              f"training rows) -> each row uses its own", flush=True)
        assert n_p > 1

    _vtext_cache = {}

    def vtext_for(task):
        """Per-row prompt template; the prompt must be exactly what the pool
        collector used, or the encoded prompt length check below fires."""
        if task not in _vtext_cache:
            q = str(pool.loc[task]["question"])
            _vtext_cache[task] = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"},
                                              {"type": "text", "text": q}]}],
                add_generation_prompt=True, tokenize=False)
        return _vtext_cache[task]
    n_anchors = [len(r["anchors"]) for r in recs]
    print(f"stored anchors/row: median {int(np.median(n_anchors))} "
          f"min {min(n_anchors)} max {max(n_anchors)}; "
          f"rollout={a.rollout_mode}/{a.verify}; --max-anchors {a.max_anchors}",
          flush=True)

    if a.aligned_forward:
        # Gate: the training context must be bit-identical to the decode prefill,
        # otherwise the run is not the repaired pipeline and must not proceed.
        _p = pool.loc[recs[0]["task"]]
        _ins = proc(text=[vtext_for(recs[0]["task"])], videos=[_p["cache_path"]],
                    return_tensors="pt", fps=float(_p["fps"]))
        _ids = _ins["input_ids"].to(dev)
        _d, _cos = verify_aligned_to_decode(
            target, _ids, None, None, _ins["pixel_values_videos"].to(dev),
            _ins["video_grid_thw"].to(dev))
        print(f"[aligned] gate {recs[0]['task']}: aligned-vs-decode-prefill ctx "
              f"max|Δ| {_d:.6f} cos {_cos:.8f}", flush=True)
        assert _d == 0.0, f"aligned forward is not bit-identical to decode: {_d}"

    def context_for(rec):
        p = pool.loc[rec["task"]]
        ins = proc(text=[vtext_for(rec["task"])], videos=[p["cache_path"]],
                   return_tensors="pt", fps=float(p["fps"]))
        assert int(ins["input_ids"].shape[1]) == int(rec["prompt_len"]), (
            f"{rec['task']}: encoded prompt {ins['input_ids'].shape[1]} != record "
            f"{rec['prompt_len']} -- prompt string drifted from the pool")
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        full = rec["input_ids"][: a.max_seq_len].to(dev).long()
        if a.aligned_forward:
            # Same attention branch as every decode path; bit-identical to the
            # decode prefill (asserted below on the first row).
            ctx = target_context_aligned(target, full[None],
                                         pixel_values_videos=pv, video_grid_thw=vg)
        else:
            ctx = common.target_context(target, full[None],
                                        pixel_values_videos=pv, video_grid_thw=vg)
        return full, ctx

    cache = None
    if a.cache_ctx:
        tot = sum(min(int(r["seq_len"]), a.max_seq_len) for r in recs) * 20480 * 2 / 2**30
        print(f"precomputing target context for {len(recs)} rows "
              f"(~{tot:.1f} GiB)...", flush=True)
        cache, t0 = [], time.time()
        for i, r in enumerate(recs):
            _, ctx = context_for(r)
            cache.append(ctx.cpu())
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(recs)} cached, {rate:.2f} rows/s, "
                      f"eta {(len(recs)-i-1)/max(rate,1e-9)/60:.1f} min", flush=True)
        print(f"context cache ready in {(time.time()-t0)/60:.1f} min", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=common.BETAS,
                            weight_decay=common.WEIGHT_DECAY)
    os.makedirs(a.out, exist_ok=True)
    log = open(a.log_file, "a") if a.log_file else None
    step, running, t0 = 0, 0.0, time.time()
    n_pos_tot, n_inexact_tot, n_fwd_tot, n_rej_tot, n_k3_tot = 0, 0, 0, 0, 0
    order = rng.permutation(len(recs))
    opt.zero_grad(set_to_none=True)
    consec_none = 0
    while step < a.steps:
        r = recs[int(order[step % len(order)])]
        if cache is not None:
            full = r["input_ids"][: a.max_seq_len].to(dev).long()
            ctx = cache[int(order[step % len(order)])].to(dev)
        else:
            full, ctx = context_for(r)
        total = int(full.shape[0])
        if a.rollout_mode == "stored":
            all_a = r["anchors"]
        else:
            policy = head if a.rollout_mode == "online" else rollout_head
            all_a = opd_online_rollout.rollout(
                policy, target, ctx, full,
                prompt_len=int(r["prompt_len"]), verify=a.verify,
                tree_budget=a.tree_budget, branch_k=a.branch_k,
                max_rounds=a.max_rounds)
            if not all_a:
                raise RuntimeError(
                    f"{r['task']}: {a.rollout_mode}/{a.verify} rollout returned "
                    "zero anchors")
        keep = list(range(len(all_a)))
        if a.max_anchors and len(keep) > a.max_anchors:
            keep = sorted(rng.choice(len(all_a), a.max_anchors, replace=False).tolist())
        picked = [all_a[i] for i in keep]
        # A record whose seq_len exceeds --max-seq-len gets truncated below, and any
        # anchor past the truncation point would index outside `full` (measured:
        # "CUDA error: device-side assert ... index out of bounds" on the MLVU mix,
        # whose rows reach L=2037+). Drop those anchors explicitly instead of
        # relying on a device-side assert to find them.
        n_before = len(picked)
        if a.rollout_mode == "stored":
            # Legacy pools contain a fixed full-width proposal even at the
            # tail, so retain their historical conservative guard.
            picked = [x for x in picked if x["pos"] + head.block_size < total]
        else:
            # Live rollout already truncates each proposal to the available
            # suffix.  A partial final block is valid; only an anchor with no
            # next token is unsupervisable.
            picked = [x for x in picked if x["pos"] < total - 1]
        if n_before != len(picked):
            n_dropped_anchor = n_before - len(picked)
            print(f"  [warn] {r['task']}: {n_dropped_anchor}/{n_before} anchors past "
                  f"the truncated length ({total} of {int(r['seq_len'])}); dropped",
                  flush=True)
        loss, npos, l_fwd, l_rev, nix, parts = opd_loss_video.opd_replay_loss_video(
            head, target, ctx, full,
            [x["pos"] for x in picked], [x["prop"] for x in picked],
            [x["n_acc"] for x in picked],
            r["topk_ids"].to(dev), r["topk_val"].to(dev),
            embed, head.block_size, common.MASK_ID, dev,
            beta_fwd=a.beta_fwd, decay=a.decay, reject_target=a.reject_target,
            drop_inexact_teacher=a.drop_inexact, alpha_rev=a.alpha_rev,
            return_parts=True)
        if loss is None:
            # Never a silent no-op: if every sampled row yields no supervised
            # position, the run is broken, not "converged".
            consec_none += 1
            assert consec_none < 50, (
                f"{consec_none} consecutive rows produced no supervised position "
                f"(last row {r['task']}, {len(picked)} anchors, L={total}); "
                f"aborting instead of training on nothing")
            continue
        consec_none = 0
        (loss / a.accum).backward()
        running += float(loss.detach())
        n_pos_tot += npos
        n_inexact_tot += nix
        n_fwd_tot += parts["n_fwd"]
        n_rej_tot += parts["n_topk_rej"]
        n_k3_tot += parts["n_k3_rej"]
        if (step + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), common.GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0 or step == 1:
            el = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            rec_row = {"step": step, "loss": running / max(a.log_every, 1),
                       "fwd": l_fwd, "rev": l_rev, "positions": n_pos_tot,
                       "inexact_frac": n_inexact_tot / max(n_pos_tot, 1),
                       "n_fwd": n_fwd_tot, "n_topk_rej": n_rej_tot, "n_k3_rej": n_k3_tot,
                       "L": total, "anchors": len(picked),
                       "s_per_step": el / step, "peak_gib": peak,
                       "elapsed_min": el / 60}
            print(f"step {step}/{a.steps} loss {rec_row['loss']:.4f} "
                  f"fwd {l_fwd:.4f} rev {l_rev:.4f} "
                  f"pos/logwin {n_pos_tot} fwd_pos {n_fwd_tot} rejtopk {n_rej_tot} "
                  f"k3 {n_k3_tot} inexact {rec_row['inexact_frac']:.3f} "
                  f"L {total} anc {len(picked)} {el/step:.2f}s/step "
                  f"peak {peak:.1f}GiB ({el/60:.1f} min)", flush=True)
            if log:
                log.write(json.dumps(rec_row) + "\n")
                log.flush()
            running = 0.0
            n_pos_tot = n_inexact_tot = n_fwd_tot = n_rej_tot = n_k3_tot = 0
        if a.save_every and step % a.save_every == 0:
            head.save_pretrained(os.path.join(a.out, f"step{step}"))
            proc.tokenizer.save_pretrained(os.path.join(a.out, f"step{step}"))
            print(f"  saved step{step}", flush=True)
    if not a.no_final_save:
        head.save_pretrained(a.out)
        proc.tokenizer.save_pretrained(a.out)
        print(f"done at step {step} -> {a.out}", flush=True)
    else:
        print(f"done at step {step} (final save disabled)", flush=True)


if __name__ == "__main__":
    main()
