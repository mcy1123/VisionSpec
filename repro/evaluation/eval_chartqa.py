#!/usr/bin/env python3
"""ChartQA acceptance length and decode speed, in the paper's budget-sweep shape.

Table 10 runs all four budgets of a task in one process against one shared
autoregressive baseline, so the speedup column is a function of the tree budget
alone. This driver does the same, and it uses the repository's own decode loop
and metrics so no convention is re-invented here:

* `timing="decode_only"` on both arms           [repo README, §5.1]
* the first prompt of a run is warm-up and dropped, so `--n 100` reads 101
* tau pools rounds rather than averaging prompts [glance/metrics.py]
* speedup is a within-card ratio of mean decode ms per token

Target numbers for ChartQA (CONFIG.md §7): tau 5.16 / 3.42x at budget 63, and
4.43/4.81/5.00/5.16 at budgets 15/31/47/63.

    python eval_chartqa.py --head $GLANCE_ROOT/checkpoints/head \
        --data $GLANCE_ROOT/data/vl_eval --budgets 15,31,47,63 \
        --n 100 --out $GLANCE_ROOT/results/chartqa.json
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
import statistics
import sys
import time

import torch
from PIL import Image

from shared import common
from shared import markov as markov_mod

import glance
from glance.vl import VisionPositions


def load_task(path, n, max_pixels):
    rows = []
    for row in common.read_jsonl(path):
        image_path = os.path.join(os.path.dirname(path), row["image"])
        if not os.path.exists(image_path):
            continue
        rows.append((image_path, row["prompt"]))
        if len(rows) >= n:
            break

    loaded = []
    for image_path, prompt in rows:
        image = Image.open(image_path).convert("RGB")
        if max_pixels and max(image.size) > max_pixels:
            w, h = image.size
            scale = max_pixels / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)))
        loaded.append((image, prompt))
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", required=True)
    parser.add_argument("--data", default=os.path.join(common.DATA, "vl_eval"))
    parser.add_argument("--task", default="chartqa")
    parser.add_argument("--n", type=int, default=100, help="prompts after warm-up")
    parser.add_argument("--budgets", default="63")
    parser.add_argument("--branch-k", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=896,
                        help="repo default; App.D's 768 px is the training side")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--chain", action="store_true",
                        help="also run the same head as a width-1 chain")
    parser.add_argument("--out", default=os.path.join(common.RESULTS, "chartqa.json"))
    parser.add_argument("--round-log", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-markov", action="store_true",
                        help="ignore a markov_head.pt shipped with the head")
    parser.add_argument("--markov-scale", type=float, default=1.0,
                        help="multiplier on the Markov bias (1.0 = as trained)")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    device = f"cuda:{args.gpu}"
    dtype = getattr(torch, args.dtype)

    target, processor = common.load_target(device=device, dtype=dtype)
    head = common.load_head(args.head, dtype=dtype, device=device).eval()
    stop_ids = [processor.tokenizer.eos_token_id]
    print(f"head {args.head}: block {head.block_size}, "
          f"layers read {list(head.target_layer_ids)}", flush=True)

    # A head trained with the semi-autoregressive Markov bias ships
    # markov_head.pt + markov_config.json beside itself. It has to be wired into
    # the draft pass, not merely loaded: the bias changes the per-offset
    # marginals the tree is scored from, so a Markov head that is loaded but
    # never applied would measure exactly like the baseline.
    mk = None if args.no_markov else markov_mod.load_markov(args.head, device)
    steer = None
    if mk is not None:
        def steer(draft_logits, root):
            return markov_mod.steer_block_greedy(
                draft_logits, root, mk, scale=args.markov_scale)[0]
    print(f"markov bias: {'ON' if steer is not None else 'off'}"
          + (f" at scale {args.markov_scale}" if steer is not None else ""), flush=True)

    path = os.path.join(args.data, f"{args.task}.jsonl")
    prompts = load_task(path, args.n + 1, args.max_pixels)
    if len(prompts) < 2:
        raise SystemExit(f"{args.task}: need at least two prompts, found {len(prompts)}")
    print(f"{args.task}: {len(prompts)} prompts, first is warm-up", flush=True)

    def encode(prompt, image):
        ids, pixels, grid = common.encode(processor, prompt, image)
        ids = ids.to(device)
        positions = VisionPositions(
            pixels.to(device) if pixels is not None else None,
            grid.to(device) if grid is not None else None)
        return ids, positions

    arms = {"ar": []}
    for budget in budgets:
        arms[f"glance_b{budget}"] = []
    if args.chain:
        arms["chain"] = []

    round_log = open(args.round_log, "w") if args.round_log else None
    started = time.time()

    for index, (image, prompt) in enumerate(prompts):
        input_ids, positions = encode(prompt, image)
        _, ar = glance.greedy_generate(target, input_ids, args.max_new, stop_ids,
                                       positions=positions, return_stats=True)
        arms["ar"].append(ar)

        for budget in budgets:
            sink = None
            if round_log is not None and index > 0:
                key = f"glance_b{budget}"

                def sink(row, key=key, index=index):
                    row.update(arm=key, prompt=index - 1)
                    round_log.write(json.dumps(row) + "\n")
            _, ours = glance.generate(
                head, target, input_ids, args.max_new, stop_ids,
                budget=budget, branch_k=args.branch_k, temperature=0.0,
                positions=positions, return_stats=True, timing="decode_only",
                on_round=sink, steer=steer)
            arms[f"glance_b{budget}"].append(ours)

        if args.chain:
            _, chain = glance.chain_generate(
                head, target, input_ids, args.max_new, stop_ids,
                positions=positions, return_stats=True)
            arms["chain"].append(chain)

        if index % 10 == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index+1}/{len(prompts)} prompts, {rate:.2f} prompts/s, "
                  f"eta {(len(prompts)-index-1)/max(rate,1e-9)/60:.1f} min", flush=True)

    measured = {name: glance.drop_warmup(runs) for name, runs in arms.items()}
    baseline = statistics.fmean(s.ms_per_token for s in measured["ar"])

    result = {"config": {k: v for k, v in vars(args).items()},
              "task": args.task, "n_prompts": len(measured["ar"]),
              "markov": mk is not None, "ar_ms_per_token": baseline, "arms": {}}

    ar_ms = [s.ms_per_token for s in measured["ar"]]

    for name, runs in measured.items():
        if name == "ar":
            continue
        ms = statistics.fmean(s.ms_per_token for s in runs)
        tau = glance.acceptance_length([s.tau for s in runs for _ in range(s.rounds)])
        arm_ms = [s.ms_per_token for s in runs]

        # [App.D "Statistics"] every reported ratio carries a bootstrap 95%
        # interval, and the paired form is the right one here because the arms
        # ran back to back on the same prompt. `paired_bootstrap(a, b)` returns
        # the interval on the mean of a - b, so feeding it the log-ratios gives
        # the interval on the log speedup, which is exponentiated back.
        speed_ci = None
        if len(arm_ms) == len(ar_ms) and len(arm_ms) > 1:
            import math as _math
            mean, lo, hi = glance.paired_bootstrap(
                [_math.log(a / b) for a, b in zip(ar_ms, arm_ms)], [0.0] * len(arm_ms))
            speed_ci = [_math.exp(lo), _math.exp(hi)]

        tau_ci = None
        per_prompt_tau = [s.tau for s in runs]
        if len(per_prompt_tau) > 1:
            import numpy as _np
            arr = _np.asarray(per_prompt_tau, float)
            rng = _np.random.default_rng(0)
            draws = arr[rng.integers(0, len(arr), size=(4000, len(arr)))].mean(axis=1)
            tau_ci = [float(_np.percentile(draws, 2.5)),
                      float(_np.percentile(draws, 97.5))]

        result["arms"][name] = {
            "tau": tau,
            "tau_ci95": tau_ci,
            "ms_per_token": ms,
            "speedup": glance.speedup(baseline, ms),
            "speedup_ci95": speed_ci,
            "tree_size": statistics.fmean(s.tree_size for s in runs),
            "rounds": sum(s.rounds for s in runs),
            "per_prompt": {"tau": per_prompt_tau,
                           "ms_per_token": arm_ms,
                           "rounds": [s.rounds for s in runs]},
        }
        ci = result["arms"][name]
        print(f"{args.task} {name}: tau {tau:.2f}"
              + (f" [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]" if tau_ci else "")
              + f", {ms:.2f} ms/tok, {ci['speedup']:.2f}x"
              + (f" [{speed_ci[0]:.2f}, {speed_ci[1]:.2f}]" if speed_ci else ""),
              flush=True)

    result["ar_ms_per_token"] = baseline
    if round_log is not None:
        round_log.close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
