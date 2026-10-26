#!/usr/bin/env python3
"""A3.2 scale measurement: how big is the official k3 rejected stream, really?

WHY this file exists. `docs/reports/2026-10-15-*` / AGENTS.md "新任务 A" step A3.2
makes one measurement a precondition of the k3 arm (P1): the official rejected
objective is a SINGLE-SAMPLE, CLAMPED estimator, while GLANCE's current one is a
full-vocabulary reverse KL with no upper bound. They are different objectives with
different scales, so a training comparison that only reports `tau` cannot tell
"the objective changed" apart from "the effective learning rate changed". This
script measures the gap on the frozen C0 policy before any GPU training is spent.

WHAT it measures, on the SAME anchors and the SAME logits (one forward each):
  * per-offset and per-anchor `l_rej` under both objectives, with mean/var/rms;
  * the ratio k3 / full-vocabulary (its own scale factor);
  * how often each official clamp bites
    (`|log p - log q| >= 20`, `kld >= 10`) and the raw `log p - log q` extremes;
  * `l_acc` and the two streams' relative contribution to the combined loss.

HOW it stays honest. The forward path is NOT re-implemented: the script calls
`opd_train_video_fullkl.score_microbatch` (the trainer's own scoring function)
with `keep_kl_variants=True`, so the anchor selection, the teacher fork, the
student block replay and the `n_acc` split are all the trainer's. It changes no
file, trains nothing, and does not touch the head weights (no backward at all).

Data discipline: only summary statistics and token ids reach the disk; no logits,
no caches. The run is read-only apart from its own `--out` JSON and `--log` text.
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
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(_REPRO_ROOT))

from shared import common  # noqa: E402
from historical import opd_fullkl  # noqa: E402
from historical import opd_train_video_fullkl as T  # noqa: E402
from transformers import AutoProcessor  # noqa: E402


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stats_of(values):
    values = [float(v) for v in values]
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    var = statistics.fmean([(v - mean) ** 2 for v in values]) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "var": var,
        "std": var ** 0.5,
        "rms": (statistics.fmean([v * v for v in values])) ** 0.5,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def build_parser():
    ap = argparse.ArgumentParser(
        description="A3.2 scale measurement: the official k3 rejected stream vs "
                    "the full-vocabulary reverse KL, on identical anchors")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--head", default=f"{common.CKPT}/head_video439_aligned/step1000",
                    help="frozen C0 start point whose policy defines the anchors")
    ap.add_argument("--records", default=f"{common.DATA}/video_greedy_499.pt")
    ap.add_argument("--pool", default=f"{common.DATA}/pool_499.parquet")
    ap.add_argument("--videos", type=int, default=4,
                    help="training videos to roll out (each contributes its anchors)")
    ap.add_argument("--anchors-per-video", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=24)
    ap.add_argument("--rollout-mode", choices=["online", "legacy-reference"],
                    default="online")
    ap.add_argument("--teacher-dtype", default="float32",
                    choices=["float32", "bfloat16"])
    ap.add_argument("--head-dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=common.MAX_SEQ_LEN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    return ap


def main():
    a = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this measurement needs the real target; no CUDA")
    dev = f"cuda:{a.gpu}"
    torch.manual_seed(a.seed)

    import pandas as pd
    dtype_of = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    teacher_dtype, head_dtype = dtype_of[a.teacher_dtype], dtype_of[a.head_dtype]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    lines: list[str] = []

    def say(msg):
        print(msg, flush=True)
        lines.append(msg)

    say(f"=== k3 scale measurement {time.strftime('%Y-%m-%dT%H:%M:%S%z')} ===")
    say(f"[cfg] head={a.head} target={a.teacher_dtype} head={a.head_dtype} "
        f"rollout={a.rollout_mode} videos={a.videos} "
        f"anchors/video<={a.anchors_per_video} gamma={a.gamma}")

    target, _ = common.load_target(device=dev, dtype=teacher_dtype)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev, dtype=head_dtype)
    head.eval() if a.rollout_mode == "online" else head.train()
    for p in head.parameters():
        p.requires_grad_(False)
    actual_teacher = next(target.parameters()).dtype
    actual_head = next(head.parameters()).dtype
    say(f"[load] target {actual_teacher} head {actual_head} "
        f"peak {torch.cuda.max_memory_allocated(a.gpu)/2**30:.2f} GiB")

    blob = torch.load(a.records, weights_only=False, map_location="cpu")
    pool = pd.read_parquet(a.pool)
    key = "task" if "task" in pool.columns else "video_name"
    split_of = dict(zip(pool[key].astype(str), pool["split"].astype(str)))
    pool = pool.set_index(key)
    recs = [r for r in blob["records"]
            if split_of.get(str(r["task"])) == "train"]
    if len(recs) < a.videos:
        raise RuntimeError(f"only {len(recs)} train records for {a.videos} videos")
    say(f"[data] {len(recs)} train records; taking the first {a.videos} in "
        f"seed order")

    order = T.video_order(len(recs), a.seed, 0)
    rows: list[dict] = []
    used_videos: list[str] = []
    t0 = time.time()
    for idx in order:
        if len(used_videos) >= a.videos:
            break
        rec = recs[idx]
        task = str(rec["task"])
        try:
            trace, anchors = T.build_video_trace(
                target, head, proc, rec, pool.loc[task], device=dev,
                max_seq_len=a.max_seq_len, max_rounds=a.max_rounds,
                verify="chain", head_dtype=head_dtype,
                rollout_mode=a.rollout_mode,
                max_new_tokens=common.GEN_NEW_TOKENS)
        except Exception as exc:  # a data failure is a finding, keep the run alive
            say(f"[skip] {task}: rollout failed: {type(exc).__name__}: {exc}")
            continue
        if not anchors:
            say(f"[skip] {task}: no anchor")
            continue
        kept, info = T.select_anchors(
            anchors, budget=a.anchors_per_video, max_seq_len=a.max_seq_len,
            prompt_len=trace.prompt_len, valid_end=trace.valid_end,
            per_video=a.anchors_per_video)
        if not kept:
            say(f"[skip] {task}: no valid anchor ({info})")
            continue
        plans = [T.build_anchor_plan(
            task, anchor, total=int(trace.full_ids.shape[0]),
            valid_end=trace.valid_end, prompt_len=trace.prompt_len,
            gamma=a.gamma, device=dev) for anchor in kept]
        denoms = T.plan_update_denominators(plans)
        n_acc_t, z_t, acc_p, rej_p = denoms
        say(f"[video] {task}: prompt_len {trace.prompt_len} "
            f"L {int(trace.full_ids.shape[0])} anchors {len(plans)} "
            f"N_acc {n_acc_t} Z {z_t:.4f}")
        for plan in plans:
            with torch.no_grad():
                _, s = T.score_microbatch(
                    head, target, trace, plan, gamma=a.gamma, denoms=denoms,
                    head_dtype=head_dtype, teacher_dtype=teacher_dtype,
                    rejected_kl=opd_fullkl.REJECTED_KL_FULL_REVERSE,
                    keep_kl_variants=True)
            row = {"video": task, "pos": int(plan.pos),
                   "K": len(plan.prop), "n_acc": int(plan.n_acc),
                   "n_acc_used": int(s["n_acc"]), "n_rej_used": int(s["n_rej"]),
                   "sum_w_rej": float(s["sum_w_rej"]),
                   "l_acc_anchor": float(s["acc_kl_mean"]),
                   "l_rej_full_anchor": float(s["rej_kl_mean"]),
                   "l_rej_k3_anchor": float(s["rej_k3_mean"]),
                   "prop": [int(t) for t in plan.prop]}
            for key_ in ("rej_log_ratio_absmax", "rej_log_ratio_clamp_frac",
                         "rej_kld_clamp_frac", "rej_full_abs_ge_kld_clamp_frac"):
                row[key_] = float(s.get(key_, 0.0))
            rows.append(row)
        used_videos.append(task)
        del trace
        torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("no anchor was scored; nothing to measure")

    # ---- update-level per-anchor l_rej (the objective's own aggregation)
    def anchor_weighted(key):
        return [r[key] for r in rows if r["n_rej_used"] > 0]

    sacc = [r["l_acc_anchor"] for r in rows if r["n_acc_used"] > 0]
    sfull = anchor_weighted("l_rej_full_anchor")
    sk3 = anchor_weighted("l_rej_k3_anchor")
    clamp_ratio = [r["rej_log_ratio_clamp_frac"] for r in rows if r["n_rej_used"] > 0]
    clamp_kld = [r["rej_kld_clamp_frac"] for r in rows if r["n_rej_used"] > 0]
    clamp_full = [r["rej_full_abs_ge_kld_clamp_frac"] for r in rows
                  if r["n_rej_used"] > 0]
    absmax = [r["rej_log_ratio_absmax"] for r in rows if r["n_rej_used"] > 0]

    summary = {
        "n_anchors": len(rows),
        "n_videos": len(used_videos),
        "n_rej_offsets": int(sum(r["n_rej_used"] for r in rows)),
        "n_acc_offsets": int(sum(r["n_acc_used"] for r in rows)),
        "l_acc": stats_of(sacc),
        "l_rej_full_vocab": stats_of(sfull),
        "l_rej_k3": stats_of(sk3),
        "l_rej_k3_over_full": (stats_of(sk3)["mean"] / stats_of(sfull)["mean"]
                               if sfull and stats_of(sfull)["mean"] else None),
        "rej_log_ratio_clamp_fraction_mean": (statistics.fmean(clamp_ratio)
                                              if clamp_ratio else 0.0),
        "rej_kld_clamp_fraction_mean": (statistics.fmean(clamp_kld)
                                        if clamp_kld else 0.0),
        "rej_full_abs_ge_kld_clamp_fraction_mean": (statistics.fmean(clamp_full)
                                                    if clamp_full else 0.0),
        "rej_log_ratio_absmax": max(absmax) if absmax else 0.0,
        "anchors_with_any_kld_clamp": sum(1 for v in clamp_kld if v > 0),
        "anchors_with_any_log_ratio_clamp": sum(1 for v in clamp_ratio if v > 0),
        "anchors_with_any_full_over_kld_clamp": sum(1 for v in clamp_full if v > 0),
    }
    # Relative contribution to the paper objective on these anchors, per anchor:
    # loss = (l_acc + l_rej)/2 when both streams exist, so the SHARE of the total
    # that the rejected term carries is l_rej / (l_acc + l_rej).
    shares = [r["l_rej_full_anchor"] / (r["l_acc_anchor"] + r["l_rej_full_anchor"])
              for r in rows if r["n_acc_used"] > 0 and r["n_rej_used"] > 0
              and (r["l_acc_anchor"] + r["l_rej_full_anchor"]) > 0]
    summary["l_rej_share_of_total_full_vocab"] = stats_of(shares)
    shares3 = [r["l_rej_k3_anchor"] / (r["l_acc_anchor"] + r["l_rej_k3_anchor"])
               for r in rows if r["n_acc_used"] > 0 and r["n_rej_used"] > 0
               and (r["l_acc_anchor"] + r["l_rej_k3_anchor"]) > 0]
    summary["l_rej_share_of_total_k3"] = stats_of(shares3)

    head_file = os.path.join(a.head, "model.safetensors")
    payload = {
        "what": "A3.2 k3 vs full-vocabulary reverse-KL scale measurement",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "AGENTS.md 新任务 A / A3.2",
        "head": a.head,
        "head_sha256": sha256_file(head_file) if os.path.exists(head_file) else None,
        "target_dtype": str(actual_teacher), "head_dtype": str(actual_head),
        "rollout_mode": a.rollout_mode,
        "gamma": a.gamma, "seed": a.seed,
        "objective_active": opd_fullkl.REJECTED_KL_FULL_REVERSE,
        "k3_clamps": {"log_ratio": opd_fullkl.K3_LOG_RATIO_CLAMP,
                      "kld": opd_fullkl.K3_KLD_CLAMP},
        "videos": used_videos,
        "anchors": rows,
        "summary": summary,
        "wall_s": time.time() - t0,
        "peak_gib": torch.cuda.max_memory_allocated(a.gpu) / 2**30,
    }
    with open(a.out, "w") as handle:
        json.dump(payload, handle, indent=2)

    say("")
    say("---- summary (per anchor, weighted l_rej = the objective's own L_rej) ----")
    for name in ("l_acc", "l_rej_full_vocab", "l_rej_k3"):
        st = summary[name]
        say(f"  {name:16s} n={st['n']:3d} mean={st['mean']:.6f} "
            f"var={st['var']:.6f} rms={st['rms']:.6f} median={st['median']:.6f} "
            f"max={st['max']:.4f}")
    say(f"  k3/full mean ratio      {summary['l_rej_k3_over_full']}")
    say(f"  clamp |log p-log q|>=20  fraction {summary['rej_log_ratio_clamp_fraction_mean']:.6f} "
        f"(anchors {summary['anchors_with_any_log_ratio_clamp']}/{len(rows)}), "
        f"absmax {summary['rej_log_ratio_absmax']:.3f}")
    say(f"  clamp k3 kld>=10         fraction {summary['rej_kld_clamp_fraction_mean']:.6f} "
        f"(anchors {summary['anchors_with_any_kld_clamp']}/{len(rows)})")
    say(f"  full-vocab >=10          fraction {summary['rej_full_abs_ge_kld_clamp_fraction_mean']:.6f} "
        f"(anchors {summary['anchors_with_any_full_over_kld_clamp']}/{len(rows)})")
    for name in ("l_rej_share_of_total_full_vocab", "l_rej_share_of_total_k3"):
        st = summary[name]
        if st.get("n"):
            say(f"  {name}: mean {st['mean']:.4f} median {st['median']:.4f}")
    say(f"  peak {payload['peak_gib']:.2f} GiB, wall {payload['wall_s']:.1f}s -> {a.out}")
    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        with open(a.log, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
