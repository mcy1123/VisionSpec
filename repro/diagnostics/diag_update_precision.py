#!/usr/bin/env python3
"""Diagnostic: are the small updates actually landing in the parameters?

Motivation (user, 2026-10-10). The continue-SFT calibration found lr 3e-5 barely
moved the weights: `mean|d|/mean|w| = 4.86e-05`, 130-180x smaller than the lr 3e-4
arm, and the tau was IDENTICAL to the start point on 26 of 29 videos. The
optimistic reading is "a very gentle update". The pessimistic reading is
"bf16 parameters plus a plain AdamW (no fp32 master weights) round the update
away entirely": `p.addcdiv_(...)` on a bf16 tensor cannot represent a step
smaller than half an ulp of `p`, so a non-zero gradient and a healthy-looking
optimizer can still produce a bit-identical parameter.

`train_head_video.py:110` and `opd_train_video_fullkl.py:1163` both build
`torch.optim.AdamW(head.parameters())` with the head loaded in bf16, and neither
creates a master-weight copy, so `exp_avg`/`exp_avg_sq` are bf16 too.

This script answers it on the REAL head, one fixed training batch, from the real
start checkpoint:

  arm `bf16`        -- the current implementation (AdamW on bf16 params)
  arm `fp32master`  -- fp32 master params + fp32 Adam state, bf16 forward,
                       master written back into the bf16 params after each step

Per tensor it records: nonzero-gradient fraction, the fraction of parameter
elements that actually changed, the update norm, and the update relative to the
bf16 spacing (ulp) at that magnitude. One step and ten steps.

No full video evaluation: this is an optimizer-precision probe, not a recipe test.

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

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from historical import train_head as th
from transformers import AutoProcessor

R = "/mnt/local/mcy/GLANCE"
BLOCK = common.BLOCK_SIZE


def build_fixed_batch(target, proc, dev, *, task, max_anchors, seed):
    """One fixed (video, anchors, reveals, ctx) batch, target forward done once."""
    recs = torch.load(f"{R}/data/video_greedy_499.pt", weights_only=False)["records"]
    pool = pd.read_parquet(f"{R}/data/pool_499.parquet").set_index("video_name")
    rec = next(r for r in recs if str(r["task"]) == task)
    p = pool.loc[task]
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": p["question"]}]}],
        add_generation_prompt=True, tokenize=False)
    ins = proc(text=[text], videos=[p["cache_path"]], return_tensors="pt",
               fps=float(p["fps"]))
    pv = ins["pixel_values_videos"].to(dev)
    vg = ins["video_grid_thw"].to(dev)
    full = rec["input_ids"][:2048].to(dev).long()
    ctx = common.target_context_cached(target, full[None],
                                       pixel_values_videos=pv, video_grid_thw=vg)
    prompt_len = int(min(rec["prompt_len"], full.shape[0] - 2))
    anchors = list(range(prompt_len, int(full.shape[0]) - 1))
    reveals = th.sample_reveals(len(anchors), BLOCK,
                                torch.Generator().manual_seed(seed), "cpu")
    anchors, reveals = th.live_anchors(anchors, reveals, int(full.shape[0]), BLOCK)
    rng = np.random.default_rng(seed)
    if len(anchors) > max_anchors:
        keep = rng.choice(len(anchors), max_anchors, replace=False)
        keep.sort()
        anchors = [anchors[i] for i in keep]
        reveals = reveals[torch.as_tensor(keep)]
    _sm = th.build_blocks(full, anchors, reveals.to(dev), BLOCK,
                          common.MASK_ID, dev)[2]
    print(f"[batch] {task}  L={int(full.shape[0])}  anchors={len(anchors)}  "
          f"supervised={int(_sm.sum())}", flush=True)
    return full, ctx, anchors, reveals.to(dev)


def bf16_ulp(x):
    """Spacing to the next representable bf16 value, per element (approximate)."""
    xf = x.float().abs()
    tiny = torch.finfo(torch.bfloat16).tiny
    return torch.clamp(xf, min=tiny) * (2.0 ** -8)


def snapshot(head):
    """CPU copies on purpose: two live GPU snapshots would be ~17 GiB and OOM.

    The diagnostic compares before/after per step, so it must hold two of these
    at once; on the GPU that is 4 fp32 copies of a 1.05B head. Host memory is
    plentiful and the comparison is not on the critical path.
    """
    return {n: p.detach().float().cpu().clone() for n, p in head.named_parameters()}


def _quant(x, q):
    """Quantile over a bounded subsample.

    torch.quantile refuses tensors past a size limit (a 12288x4096 fc weight is
    50M elements), and the update/ulp distribution is not meaningfully changed by
    a 1M-element deterministic stride sample.
    """
    flat = x.reshape(-1)
    if flat.numel() > 1_000_000:
        step = flat.numel() // 1_000_000 + 1
        flat = flat[::step]
    return float(torch.quantile(flat, q))


def stats(before, after, grads, ideal_before=None, ideal_after=None):
    """Per-tensor update accounting."""
    out = {}
    for n in before:
        b, a = before[n], after[n]
        d = (a - b).abs()
        nz = d > 0
        g = grads.get(n)
        ulp = bf16_ulp(b)
        rec = {
            "numel": int(b.numel()),
            "grad_nonzero_frac": (float((g != 0).float().mean()) if g is not None else None),
            "grad_norm": (float(g.norm()) if g is not None else None),
            "changed_frac": float(nz.float().mean()),
            "update_norm": float((a - b).norm()),
            "param_norm": float(b.norm()),
            "rel_update": float((a - b).norm() / b.norm().clamp_min(1e-12)),
            "median_update_over_ulp": _quant(d / ulp, 0.5),
            "p90_update_over_ulp": _quant(d / ulp, 0.9),
            "frac_update_lt_half_ulp": float((d < 0.5 * ulp).float().mean()),
        }
        if ideal_before is not None and ideal_after is not None:
            di = ideal_after[n] - ideal_before[n]
            d = a - b
            rec["ideal_update_norm"] = float(di.norm())
            rec["actual_over_ideal_norm"] = float(
                d.norm() / di.norm().clamp_min(1e-12))
            # Direction, not just magnitude: a norm ratio near 1 can still hide a
            # badly rotated update. cos is NaN when either side is exactly zero
            # (a tensor the bf16 write-back rounded to no change at all).
            if d.norm() > 0 and di.norm() > 0:
                rec["update_cosine"] = float(
                    torch.nn.functional.cosine_similarity(
                        d.reshape(1, -1), di.reshape(1, -1)).item())
            else:
                rec["update_cosine"] = None
        out[n] = rec
    return out


def run_arm(mode, head, target, batch, *, steps, lr, betas, wd, dev):
    full, ctx, anchors, reveals = batch
    embed = target.get_input_embeddings().weight
    params = [p for p in head.parameters() if p.requires_grad]

    if mode == "fp32master":
        # Master weights AND Adam state on the CPU (optimizer offload). On the GPU
        # this arm needs +4.2 GiB of fp32 params and +8.4 GiB of Adam moments on
        # top of the bf16 target, which does not fit beside another user's job;
        # the measurement is about WHERE the update accumulates, not about how
        # fast the step runs, so host-side accumulation is equivalent here.
        master = {n: p.detach().float().cpu().clone()
                  for n, p in head.named_parameters()}
        names = [n for n, _ in head.named_parameters()]
        opt = torch.optim.AdamW([master[n] for n in names], lr=lr, betas=betas,
                                weight_decay=wd)
        ideal_before = {n: master[n].detach().clone() for n in names}
        opt_state_dtype = str(master[names[0]].dtype) + " (CPU offload)"
    else:
        opt = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
        master, names = None, None
        ideal_before = None
        opt_state_dtype = str(next(head.parameters()).dtype)

    per_step = []
    for step in range(1, steps + 1):
        before = snapshot(head)
        opt.zero_grad(set_to_none=True)
        loss, _c, _w, _bo = th.sequence_loss(
            head, target, ctx, full, anchors, reveals, embed,
            common.MASK_ID, BLOCK, dev)
        loss.backward()
        grads = {n: (p.grad.detach().float().clone() if p.grad is not None else None)
                 for n, p in head.named_parameters()}

        if mode == "fp32master":
            for n in names:
                master[n].grad = grads[n].cpu()
            opt.step()
            with torch.no_grad():
                for n, p in head.named_parameters():
                    p.copy_(master[n])
            ideal_after = {n: master[n].detach().clone() for n in names}
        else:
            opt.step()
            ideal_after = None

        after = snapshot(head)
        per_step.append({"step": step, "loss": float(loss.detach()),
                         "tensors": stats(before, after, grads,
                                          ideal_before, ideal_after)})
        ideal_before = ideal_after
        print(f"  [{mode}] step {step}/{steps} loss {float(loss.detach()):.6f} "
              f"changed_frac(mean) "
              f"{np.mean([t['changed_frac'] for t in per_step[-1]['tensors'].values()]):.4f}",
              flush=True)
    return {"mode": mode, "optimizer_state_dtype": opt_state_dtype,
            "lr": lr, "steps": steps, "per_step": per_step}


def agg(arm):
    """Aggregate the first step and the last step across tensors."""
    def one(step_idx):
        t = arm["per_step"][step_idx]["tensors"]
        tot = sum(v["numel"] for v in t.values())
        return {
            "changed_elem_frac": sum(v["changed_frac"] * v["numel"] for v in t.values()) / tot,
            "frac_tensors_unchanged": sum(1 for v in t.values() if v["changed_frac"] == 0) / len(t),
            "total_update_norm": float(np.sqrt(sum(v["update_norm"] ** 2 for v in t.values()))),
            "total_param_norm": float(np.sqrt(sum(v["param_norm"] ** 2 for v in t.values()))),
            "grad_nonzero_frac_mean": float(np.mean(
                [v["grad_nonzero_frac"] for v in t.values() if v["grad_nonzero_frac"] is not None])),
            "median_update_over_ulp_mean": float(np.median(
                [v["median_update_over_ulp"] for v in t.values()])),
            "frac_update_lt_half_ulp_mean": float(np.mean(
                [v["frac_update_lt_half_ulp"] for v in t.values()])),
            "update_cosine_median": (
                float(np.nanmedian([v["update_cosine"] for v in t.values()
                                    if v.get("update_cosine") is not None]))
                if any(v.get("update_cosine") is not None for v in t.values()) else None),
            "tensors_with_zero_update": sum(
                1 for v in t.values() if v.get("update_cosine") is None) / len(t),
            "actual_over_ideal_norm_mean": (
                float(np.median([v["actual_over_ideal_norm"] for v in t.values()
                                 if "actual_over_ideal_norm" in v]))
                if "actual_over_ideal_norm" in next(iter(t.values())) else None),
        }
    return {"step1": one(0), "step_last": one(-1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="v_-6dz6tBH77I", help="a train-split video")
    ap.add_argument("--init-from", default=f"{R}/checkpoints/head_video439_aligned/step1000")
    ap.add_argument("--max-anchors", type=int, default=96)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=f"{R}/results/phaseC2/diag_update_precision.json")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"

    target, _ = common.load_target(device=dev)          # bf16; the diagnostic is
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)   # about the OPTIMIZER
    batch = build_fixed_batch(target, proc, dev, task=a.task,
                              max_anchors=a.max_anchors, seed=a.seed)

    results = {}
    for mode in ("bf16", "fp32master"):
        head = common.load_head(a.init_from, device=dev, dtype=torch.bfloat16)
        for p in head.parameters():
            p.requires_grad_(True)
        print(f"=== arm {mode} (Adam state dtype={mode}) lr={a.lr} "
              f"steps={a.steps} ===", flush=True)
        arm = run_arm(mode, head, target, batch, steps=a.steps, lr=a.lr,
                      betas=common.BETAS, wd=common.WEIGHT_DECAY, dev=dev)
        arm["agg"] = agg(arm)
        results[mode] = arm
        del head
        torch.cuda.empty_cache()

    print("\n=== summary ===")
    hdr = (f"{'arm':<12} {'step':<6} {'changed elem frac':>18} "
           f"{'unchanged tensors':>18} {'median upd/ulp':>15} "
           f"{'frac upd<0.5ulp':>16} {'actual/ideal':>13} {'cos(actual,ideal)':>18} "
           f"{'tensors w/ zero upd':>20}")
    print(hdr)
    for mode in ("bf16", "fp32master"):
        for k in ("step1", "step_last"):
            g = results[mode]["agg"][k]
            ai = g["actual_over_ideal_norm_mean"]
            cos = g["update_cosine_median"]
            print(f"{mode:<12} {k:<6} {g['changed_elem_frac']:>18.6f} "
                  f"{g['frac_tensors_unchanged']:>18.4f} "
                  f"{g['median_update_over_ulp_mean']:>15.4f} "
                  f"{g['frac_update_lt_half_ulp_mean']:>16.6f} "
                  f"{(f'{ai:.4f}' if ai is not None else 'n/a'):>13} "
                  f"{(f'{cos:.4f}' if cos is not None else 'n/a'):>18} "
                  f"{g['tensors_with_zero_update']:>20.4f}")

    b1 = results["bf16"]["agg"]["step1"]["changed_elem_frac"]
    f1 = results["fp32master"]["agg"]["step1"]["changed_elem_frac"]
    bl = results["bf16"]["agg"]["step_last"]["changed_elem_frac"]
    fl = results["fp32master"]["agg"]["step_last"]["changed_elem_frac"]
    print(f"\nVERDICT: after 1 step the bf16 arm changes {b1*100:.4f}% of elements vs "
          f"{f1*100:.4f}% for the fp32-master arm; after {a.steps} steps "
          f"{bl*100:.4f}% vs {fl*100:.4f}%.")
    if bl < 0.5 * fl:
        print("=> bf16 parameters are DISCARDING a large share of the update; "
              "an fp32 master weight is required before any control-arm calibration.")
    else:
        print("=> no large rounding loss detected; the gentle movement is real optimisation.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "results": results}, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
