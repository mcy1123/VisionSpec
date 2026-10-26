#!/usr/bin/env python3
"""Sweep the Markov bias magnitude at inference, without retraining.

Why this is the decisive experiment. Arm M trains the DSpark bias with the
reference recipe (w1 std 0.01, w2 at PyTorch's default Linear init), which puts
the bias at std ~0.0058 against block logits of order 1. `test_steer_hook.py`
needed a 50x lift before the bias changed the tree's accepted-length schedule,
and at lr 4e-5 a 1000-step Adam epoch moves each weight by at most ~0.04 -- so
the trained bias may still be far too small to change any argmax.

That leaves a null result ambiguous between two opposite conclusions:

  (a) the mechanism does not transfer to a VLM block head under tree
      verification -- which is itself the answer to the contradiction between
      DSpark's own LLM gains and the MMSpec survey's multimodal regression; or
  (b) the bias was never driven hard enough, which says nothing about the
      mechanism.

Scaling the bias at inference separates them for the price of one eval pass:
a trained bias has a *direction* learned from data, so if that direction is
useful, tau(scale) must rise above the baseline somewhere. A random bias has no
useful direction, which is why the scaled *untrained* head in `test_steer_hook`
made tau worse (4.4000 -> 4.3000) -- that is the control this sweep needs in
order to be interpretable.

The 8B target is loaded once and reused across scales; AR is measured once.

    python sweep_markov_scale.py --head $GLANCE_ROOT/checkpoints/head_markov \
        --scales 1,5,20,50,100,200 --n 30 --budget 63 --gpu 2
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
import math
import os
import statistics
import sys
import time

import torch

from shared import common
from shared import markov as markov_mod

sys.path.insert(0, os.path.join(common.WORKSPACE, "GLANCE_repo"))

import glance
from glance.vl import VisionPositions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--data", default=os.path.join(common.DATA, "vl_eval"))
    ap.add_argument("--task", default="chartqa")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch-k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=896)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--scales", default="1,5,20,50,100,200")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    # scale 0 is the reference arm, not the AR arm: scaling the bias to zero
    # leaves `steer` an identity, so scale 0 reproduces `steer=None` exactly.
    # Comparing against AR would be wrong -- autoregressive decoding commits one
    # token per round, so its tau is 1.0 by construction and is not the number
    # the criterion is written against. It also makes the sweep self-checking:
    # scale 0 must land on the recorded baseline tau at the same n and budget.
    scales = sorted({0.0} | {float(s) for s in args.scales.split(",") if s.strip()})
    target, processor = common.load_target(device=device)
    head = common.load_head(args.head, device=device).eval()
    mk = markov_mod.load_markov(args.head, device)
    if mk is None:
        raise SystemExit(f"{args.head} has no markov_head.pt -- nothing to sweep")

    path = os.path.join(args.data, f"{args.task}.jsonl")
    prompts = []
    for row in common.read_jsonl(path):
        img = os.path.join(args.data, row["image"])
        if os.path.exists(img):
            prompts.append((img, row["prompt"]))
        if len(prompts) >= args.n + 1:      # first is warm-up, matching eval_chartqa
            break
    print(f"{args.task}: {len(prompts)} prompts (first is warm-up), "
          f"budget {args.budget}, scales {scales}", flush=True)
    stop = [processor.tokenizer.eos_token_id]

    def encode(prompt, image_path):
        image = common.load_image(image_path, args.max_pixels)
        ids, pix, grid = common.encode(processor, prompt, image)
        ids = ids.to(device)
        pos = VisionPositions(pix.to(device) if pix is not None else None,
                              grid.to(device) if grid is not None else None)
        return ids, pos

    ar_ms, ar_taus = [], []
    arm = {s: {"ms": [], "taus": [], "rounds": []} for s in scales}
    started = time.time()

    for index, (image_path, prompt) in enumerate(prompts):
        ids, positions = encode(prompt, image_path)
        _, ar = glance.greedy_generate(target, ids, args.max_new, stop,
                                       positions=positions, return_stats=True)
        if index > 0:
            ar_ms.append(ar.ms_per_token)
            ar_taus.append(ar)
        for s in scales:
            def steer(logits, root, s=s):
                return markov_mod.steer_block_greedy(logits, root, mk, scale=s)[0]
            _, st = glance.generate(head, target, ids, args.max_new, stop,
                                    budget=args.budget, branch_k=args.branch_k,
                                    temperature=0.0, positions=positions,
                                    return_stats=True, timing="decode_only",
                                    steer=steer)
            if index > 0:
                arm[s]["ms"].append(st.ms_per_token)
                arm[s]["taus"].append(st)
                arm[s]["rounds"].append(st.rounds)
        if index % 5 == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index+1}/{len(prompts)} prompts, {rate:.2f} p/s, "
                  f"eta {(len(prompts)-index-1)/max(rate,1e-9)/60:.1f} min", flush=True)

    base_ms = statistics.fmean(ar_ms)
    ar_tau = glance.acceptance_length(
        [s.tau for s in ar_taus for _ in range(s.rounds)])

    def tau_of(s):
        d = arm[s]
        return glance.acceptance_length(
            [x.tau for x in d["taus"] for _ in range(x.rounds)])

    base_tau = tau_of(0.0)          # the unsteered GLANCE head -- the real reference

    print(f"\nAR baseline: {base_ms:.2f} ms/token, tau {ar_tau:.3f} "
          f"(tau is 1.0 by construction for AR)")
    print(f"GLANCE reference (scale 0): tau {base_tau:.3f}  <- must match the "
          f"recorded baseline at this n/budget")
    print(f"\n{'scale':>7} {'tau':>8} {'vs scale 0':>11} {'ms/tok':>9} {'vs AR':>7} "
          f"{'rounds':>7}")
    rows = []
    for s in scales:
        d = arm[s]
        ms = statistics.fmean(d["ms"])
        tau = tau_of(s)
        rows.append({"scale": s, "tau": tau, "ms_per_token": ms,
                     "speedup": base_ms / ms,
                     "delta_vs_scale0": tau - base_tau})
        print(f"{s:>7.1f} {tau:>8.3f} {tau-base_tau:>+11.3f} {ms:>9.2f} "
              f"{base_ms/ms:>6.2f}x {sum(d['rounds']):>7}")

    payload = {"head": args.head, "task": args.task, "budget": args.budget,
               "n_prompts": len(prompts) - 1, "ar_ms_per_token": base_ms,
               "reference_tau_scale0": base_tau, "rows": rows}
    out = args.out or os.path.join(common.RESULTS, "markov_scale_sweep.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}")

    best = max(rows, key=lambda r: r["tau"])
    print(f"\nbest tau {best['tau']:.3f} at scale {best['scale']:.1f} "
          f"(reference {base_tau:.3f})")
    moved = best["tau"] - base_tau
    if moved > 0.15:
        print("VERDICT: the trained bias direction DOES help once it is large enough "
              "-- the mechanism transfers and Arm M's magnitude was the problem")
    elif best["tau"] <= base_tau + 1e-9:
        print("VERDICT: no scale helps; the trained direction carries no usable "
              "signal -- the negative result is robust to the magnitude confound")
    else:
        print(f"VERDICT: at most a marginal {moved:+.3f} tau gain; below the "
              f"pre-registered +0.15, so the negative result stands but a weak "
              f"positive trend should be reported honestly")


if __name__ == "__main__":
    main()
