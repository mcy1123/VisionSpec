#!/usr/bin/env python3
"""B0.5 calibration: how far does the teacher's floating-point dtype move Draft-OPD?

WHY this file exists. Plan section 10.2 fixes the B0.5 sample (8 TRAINING videos,
at most 4 valid chain anchors each, aligned step1000 start point) and asks one
question before any real OPD training is scheduled: if the frozen target runs in
`bfloat16` instead of `float32`, how much of the full-vocabulary objective moves?
`proto_fullkl_b0` showed the incremental teacher path is bit-identical in fp32
and off by ~0.375 max-|dlogit| in bf16 on ONE old-head row; that is a forward-path
observation, not a statement about the KL, the total loss, the head gradient, or
the memory/time cost of scoring a whole frozen subset. This script measures those.

WHAT it measures, and why it is structured this way. The comparison is only
meaningful if BOTH dtypes see the *same* rollout: the same anchors (`pos`,
`prop`, `n_acc`), the same committed tokens, the same teacher prefix and the same
M-RoPE positions (hence the same mask). So the run is a single process with two
phases; the anchors are frozen in phase A (fp32) and replayed verbatim in phase B
(bf16), with the frozen anchor records round-tripped through JSON so "identical"
is a checked property, not object identity:

  * Phase A -- target and head in fp32. One shared target prefill over the
    recorded target-greedy trajectory (`target_prefill_shared`) gives the head
    context; `opd_online_rollout.rollout(verify="chain", max_rounds=4)` gives the
    candidate anchors; a deterministic validity filter keeps the scoreable ones.
    For each anchor the teacher rows come from `teacher_block_logits` (forked
    prefix KV, draft-induced prefix, one prefill over `ids[:p+1]`) and the student
    rows from the chain replay through `draft_block_logits` (carried cache plus
    `cache.crop`, exactly `rollout`'s own step, so `q` is the distribution the
    rollout proposed from). `opd_fullkl_loss` gives the block loss; ONE backward
    per video over the summed block losses accumulates the head gradient into a
    CPU copy (`{name: tensor.cpu()}`). The fp32 teacher logits `p_A` and the fp32
    student logits `q_A` stay in RAM so phase B can isolate teacher from student.
  * Phase B -- free the fp32 target, reload in bf16, reuse the frozen anchors.
    Same prefill/teacher/student/loss/gradient path, so the teacher dtype (and,
    unavoidably, the dtype of the hidden states the head consumes) is what
    changes. Two backward passes per video: the main one on `(p_B, q_B)`, and a
    second one on `(p_A, q_B)` -- the same bf16 student graph scored against the
    cached fp32 teacher rows -- which isolates the teacher's own gradient effect.

WHY the gradient is a per-video backward over the summed block losses. A trainer
does one backward per rollout, not one per anchor, and the head's context keys for
earlier rounds must be part of the graph the later rounds read (the chain cache is
carried, as in `rollout`). Backward per anchor would either reuse a freed graph or
silently detach the context path; summing the block losses first leaves the graph
exactly as the rollout built it. `grad_cosine_and_norm_ratio` then reduces the two
gradient sets tensor by tensor, so the whole model gradient is never concatenated.

DATA DISCIPLINE. Only summary statistics, configuration and token IDs reach the
disk. No full logits, no distributions, no KV caches: `p_A`/`q_A` live in RAM for
the duration of the run, and the closing sweep asserts the watched directories
gained nothing except the one JSON and the one log.

No `try`/`except`: a failure is a finding and must surface with its traceback.
Nothing in the repository is modified; this file is additive.
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
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

_HERE = str(_REPRO_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from shared import common                      # noqa: E402
from historical import opd_fullkl as M             # noqa: E402
from historical import opd_online_rollout          # noqa: E402
from transformers import AutoConfig, AutoProcessor, DynamicCache  # noqa: E402

# The four (teacher, student) combinations the run reports. The keys are used in
# the JSON and in every printed table, so no reader has to infer which logits a
# number came from.
COMBO_NAMES = {
    "pA_qA": "fp32 teacher / fp32 student (phase A's own)",
    "pB_qB": "bf16 teacher / bf16 student (phase B's own)",
    "pB_qA": "bf16 teacher / fp32 student (fixed q_A)",
    "pA_qB": "fp32 teacher / bf16 student (fixed q_B)",
}

# The deterministic anchor rule, recorded verbatim in the config block.
SELECTION_RULE = (
    "chain rollout (verify='chain', max_rounds=4) of the aligned step1000 head "
    "over the frozen target-greedy trajectory; keep the rounds in position order "
    "and drop an anchor iff it has no offset to score, its scored window leaves "
    "the trajectory, or a target token inside the scored window is EOS or pad"
)


# --------------------------------------------------------------------------
# Logging: every line goes to stdout and to the one log file
# --------------------------------------------------------------------------

class _Tee:
    """Write through to two streams so a traceback lands in the log too."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def log(message=""):
    print(message, flush=True)


# --------------------------------------------------------------------------
# Small statistics helpers
# --------------------------------------------------------------------------

def dist_summary(values):
    """n/mean/min/P50/P90/P99/max of a pooled sample. Percentiles: numpy linear."""
    arr = np.asarray(list(values), dtype=np.float64)
    out = {"n": int(arr.size)}
    if arr.size:
        out.update({
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "p50": float(np.percentile(arr, 50.0)),
            "p90": float(np.percentile(arr, 90.0)),
            "p99": float(np.percentile(arr, 99.0)),
            "max": float(arr.max()),
        })
    return out


def diff_stats(reference, other):
    """Absolute and relative difference of two scalars (`reference` is `a`)."""
    absolute = abs(float(reference) - float(other))
    return {"reference": float(reference), "other": float(other), "abs": absolute,
            "rel": (absolute / abs(float(reference))) if reference != 0 else None}


def sha256_tensor(t):
    """Stable hash of an integer tensor, for the cross-phase identity asserts."""
    arr = t.detach().to(torch.int64).cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def anchors_digest(anchors_by_video):
    """Structural digest of the frozen anchors (pos/prop/n_acc), used to prove
    that phase B consumed exactly what phase A selected."""
    payload = {video: [{"pos": int(a["pos"]), "n_acc": int(a["n_acc"]),
                        "prop": [int(t) for t in a["prop"]]} for a in anchors]
               for video, anchors in anchors_by_video.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sanitize(obj):
    """JSON cannot carry inf/nan: replace them with None for downstream tooling."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return sanitize(obj.item())
    return obj


class StreamStats:
    """Pooled per-offset KLs and block losses for ONE (teacher, student) combo.

    The accepted pool holds forward KL(p||q) of the accepted offsets and the
    rejected pool holds reverse KL(q||p) of the rejected offsets -- exactly the
    two directions `opd_fullkl_loss` combines. Kept in memory; the JSON gets
    `summary()`.
    """

    def __init__(self, name):
        self.name = name
        self.acc = []
        self.rej = []
        self.losses = []
        self.l_acc = []
        self.l_rej = []

    def add(self, p_logits, q_logits, n_acc, loss_value, metrics):
        with torch.no_grad():
            kl_fwd = M.full_kl_forward(p_logits, q_logits)   # KL(p||q), per offset
            kl_rev = M.full_kl_reverse(q_logits, p_logits)   # KL(q||p), per offset
        K = int(p_logits.shape[0])
        self.acc.extend(float(v) for v in kl_fwd[:n_acc])
        self.rej.extend(float(v) for v in kl_rev[n_acc:K])
        self.losses.append(float(loss_value))
        self.l_acc.append(float(metrics["l_acc"]))
        self.l_rej.append(float(metrics["l_rej"]))

    def summary(self):
        return {
            "accepted": dist_summary(self.acc),
            "rejected": dist_summary(self.rej),
            "loss": {
                "n_anchors": len(self.losses),
                "total": float(sum(self.losses)),
                "mean": float(np.mean(self.losses)) if self.losses else None,
                "min": float(np.min(self.losses)) if self.losses else None,
                "max": float(np.max(self.losses)) if self.losses else None,
            },
            "l_acc_per_anchor_mean": float(np.mean(self.l_acc)) if self.l_acc else None,
            "l_rej_per_anchor_mean": float(np.mean(self.l_rej)) if self.l_rej else None,
            "n_acc_positions": len(self.acc),
            "n_rej_positions": len(self.rej),
        }


# --------------------------------------------------------------------------
# Frozen inputs
# --------------------------------------------------------------------------

@dataclass
class VideoInputs:
    """One frozen video row plus the processor output and the stored trajectory."""

    name: str
    question: str
    cache_path: str
    fps: float
    split: str
    prompt_ids: torch.Tensor          # (1, prompt_len) cpu, long
    pv: torch.Tensor                  # processor's pixel_values_videos, cpu
    vg: torch.Tensor                  # processor's video_grid_thw, cpu
    seq_ids: torch.Tensor             # (total,) cpu, long: target-greedy trajectory
    prompt_len: int
    total: int
    trajectory_sha256: str


def build_videos(args, processor):
    """Load the frozen 8 training videos, their processor inputs and trajectory.

    Selection is NOT done here: `results/phaseB05/subsets.json` already froze the
    8 names with a score-free sha256 rule, and re-selecting would silently change
    the sample. This function only verifies the frozen properties -- every name is
    a pool `train` row and none appears in either held-out list -- and checks that
    the processor still encodes the stored `prompt_len` exactly.
    """
    subsets = json.load(open(args.subsets))
    names = [str(n) for n in subsets["b05_train_videos"]]
    if len(names) != 8:
        raise ValueError(f"the frozen B0.5 sample is 8 videos, got {len(names)}")
    if args.videos:
        names = names[:args.videos]
        log(f"[warn] --videos {args.videos}: SMOKE subset, not the frozen 8-video sample")

    pool = pd.read_parquet(args.pool)
    pool["video_name"] = pool["video_name"].astype(str)
    heldout = set(pool.loc[pool["split"].astype(str) == "heldout", "video_name"])
    monitor = {str(n) for n in subsets["monitor_fresh30_videos"]}
    for name in names:
        row = pool[pool["video_name"] == name]
        if len(row) != 1:
            raise ValueError(f"{name} appears {len(row)} times in {args.pool}")
        if str(row.iloc[0]["split"]) != "train":
            raise ValueError(f"{name} is not a train row")
        if name in heldout:
            raise ValueError(f"{name} is in the held-out split")
        if name in monitor:
            raise ValueError(f"{name} is in the fresh30 monitor subset")

    blob = torch.load(args.records, weights_only=False, map_location="cpu")
    by_task = {}
    for rec in blob["records"]:
        by_task.setdefault(str(rec["task"]), []).append(rec)

    videos = {}
    for name in names:
        row = pool[pool["video_name"] == name].iloc[0]
        recs = by_task.get(name, [])
        if len(recs) != 1:
            raise ValueError(f"{name} appears {len(recs)} times in {args.records}; "
                             "one stored trajectory is needed")
        rec = recs[0]
        prompt_len = int(rec["prompt_len"])
        seq_ids = rec["input_ids"].to(torch.long).reshape(-1)
        total = int(seq_ids.shape[0])
        if total < prompt_len + 2:
            raise ValueError(f"{name}: prompt {prompt_len} vs trajectory {total}")

        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"},
                                          {"type": "text", "text": str(row["question"])}]}],
            add_generation_prompt=True, tokenize=False)
        ins = processor(text=[text], videos=[str(row["cache_path"])],
                        return_tensors="pt", fps=float(row["fps"]))
        prompt_ids = ins["input_ids"].to(torch.long)
        if int(prompt_ids.shape[1]) != prompt_len:
            raise ValueError(f"{name}: encoded prompt {int(prompt_ids.shape[1])} != "
                             f"stored prompt_len {prompt_len}; the prompt string "
                             "drifted from the pool the trajectory came from")
        videos[name] = VideoInputs(
            name=name, question=str(row["question"]), cache_path=str(row["cache_path"]),
            fps=float(row["fps"]), split=str(row["split"]),
            prompt_ids=prompt_ids,
            pv=ins["pixel_values_videos"].clone(),
            vg=ins["video_grid_thw"].clone(),
            seq_ids=seq_ids, prompt_len=prompt_len, total=total,
            trajectory_sha256=sha256_tensor(seq_ids))
        log(f"[row] {name:18s} split {row['split']:6s} prompt {prompt_len:5d}  "
            f"trajectory {total:5d}  grid {ins['video_grid_thw'].tolist()}  "
            f"pv {tuple(ins['pixel_values_videos'].shape)} "
            f"{ins['pixel_values_videos'].dtype}  sha {videos[name].trajectory_sha256[:12]}")
    return videos


def validate_anchors(seq_ids, raw_anchors, *, eos_id, pad_id):
    """The deterministic validity filter named in SELECTION_RULE.

    An anchor is kept iff it has at least one offset, its whole scored window
    `pos+1 .. pos+K` lies inside the committed trajectory, and no token in that
    window is the trajectory's EOS or the pad id. Everything else is dropped and
    reported with the exact reason -- a dropped anchor is a datum, not a warning
    to swallow.
    """
    total = int(seq_ids.shape[0])
    kept, dropped = [], []
    for index, raw in enumerate(raw_anchors):
        pos = int(raw["pos"])
        prop = [int(t) for t in raw["prop"]]
        n_acc = int(raw["n_acc"])
        K = len(prop)
        if n_acc > K:
            raise ValueError(f"anchor at {pos} accepted {n_acc} of {K} offsets")
        reason = None
        if K < 1:
            reason = "no offset to score (empty proposal)"
        elif pos + K > total - 1:
            reason = (f"scored window ends at {pos + K} but the trajectory has "
                      f"{total} tokens")
        else:
            window = [int(t) for t in seq_ids[pos + 1:pos + K + 1]]
            for offset, token in enumerate(window, start=1):
                if token == eos_id:
                    reason = f"offset {offset} target token is EOS ({eos_id})"
                    break
                if token == pad_id:
                    reason = f"offset {offset} target token is pad ({pad_id})"
                    break
        if reason is not None:
            dropped.append({"rollout_index": index, "pos": pos, "n_acc": n_acc,
                            "prop": prop, "reason": reason})
            continue
        kept.append({"rollout_index": index, "pos": pos, "n_acc": n_acc, "K": K,
                     "prop": prop,
                     "committed": [int(t) for t in seq_ids[pos + 1:pos + K + 1]]})
    return kept, dropped


# --------------------------------------------------------------------------
# The phase loop
# --------------------------------------------------------------------------

def _position_hash(target, prefix_ids, pv, vg):
    """sha256 of the M-RoPE positions this prefix binds, for the two-phase assert.

    The mask follows from the positions, so hashing them is the cheapest honest
    statement that phase A and phase B scored the same prefix under the same mask.
    """
    vispos = M.prefix_vision_positions(target, prefix_ids,
                                       pixel_values_videos=pv, video_grid_thw=vg)
    return sha256_tensor(vispos.prefill_inputs(prefix_ids, None)["position_ids"])


def _accumulate_cpu(acc, head):
    """Add the head's current `.grad` into a CPU copy, then leave `.grad` alone.

    The plan's instruction is `{name: tensor.cpu()}`: keeping the accumulator on
    the host costs one PCIe copy per video and removes a full model-gradient
    buffer from the GPU peak, which is the binding constraint in the fp32 phase.
    """
    for name, param in head.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().to(torch.float32).cpu()
        acc[name] = grad if name not in acc else acc[name] + grad


def run_phase(*, tag, dtype, args, videos, head_path, dev, main_combo,
              combos_this_phase, anchor_digest_wanted=None,
              anchors_by_video=None, raw_anchors_A=None, anchor_store=None,
              logits_store=None, teacher_A=None, q_A=None, select_anchors=False,
              want_teacher_comparison=False):
    """Load a fresh target+head in `dtype` and score every frozen anchor.

    Returns `(result, grads_main, grads_extra, anchor_store)`. `anchor_store` is
    non-empty only for the phase that selects the anchors (phase A).
    """
    phase_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.gpu)
    target, _ = common.load_target(device=dev, dtype=dtype)
    head = common.load_head(head_path, device=dev, dtype=dtype).eval()
    head.requires_grad_(True)
    load_seconds = time.perf_counter() - phase_start
    after_load = torch.cuda.memory_allocated(args.gpu) / 2**30

    trainable_target = [n for n, p in target.named_parameters() if p.requires_grad]
    if trainable_target:
        raise AssertionError(f"the frozen target has trainable parameters: "
                             f"{trainable_target[:3]}")
    n_target_params = sum(p.numel() for p in target.parameters())
    n_head_params = sum(p.numel() for p in head.parameters())
    log(f"[{tag}] target {n_target_params / 1e9:.3f}B  head {n_head_params / 1e6:.1f}M  "
        f"dtype {dtype}  load {load_seconds:.1f}s  alloc after load {after_load:.2f} GiB")

    head.zero_grad(set_to_none=True)
    combos = {name: StreamStats(name) for name in combos_this_phase}
    per_anchor = []
    per_video_time = []
    acc_main = {}
    acc_extra = {}
    if anchor_store is None:
        anchor_store = {"kept": {}, "raw": {}, "dropped": [],
                        "position_hash": {}, "trajectory_hash": {}}
    # Defaults are the affirmative reading: every check below can only flip one
    # to False, so "True" means the check ran and held.
    identity = {"anchor_tokens_identical": True, "position_hash_match": True,
                "trajectory_hash_match": True}
    rollout_diag = []
    argmax_ok = argmax_tot = 0
    student_vs_A_tv = []
    teacher_stat = {"tv": [], "dlogit": [], "scale_pA": [], "scale_pB": [],
                    "argmax_match": 0, "argmax_total": 0, "n_rows": 0}

    if not select_anchors:
        if anchor_digest_wanted is not None:
            identity["anchor_tokens_identical"] = (
                anchors_digest(anchors_by_video) == anchor_digest_wanted)
            if not identity["anchor_tokens_identical"]:
                raise AssertionError("phase B did not consume phase A's anchors")

    for name, vi in videos.items():
        video_start = time.perf_counter()
        seq = vi.seq_ids.to(dev)
        pv = vi.pv.to(dev)
        vg = vi.vg.to(dev)

        # ---- one shared prefill over the recorded greedy trajectory: the head
        # context that the rollout and the chain replay both read.
        t0 = time.perf_counter()
        fused_full, kv_full = M.target_prefill_shared(
            target, seq[None], pixel_values_videos=pv, video_grid_thw=vg,
            max_seq_len=common.MAX_SEQ_LEN)
        del kv_full                      # only the head context is wanted here
        torch.cuda.synchronize()
        ctx_seconds = time.perf_counter() - t0

        # ---- anchors: selected once, in phase A, by the validated chain rollout
        if select_anchors:
            raw = opd_online_rollout.rollout(
                head, target, fused_full, seq, prompt_len=vi.prompt_len,
                verify="chain", max_rounds=args.max_anchors)
            kept, dropped = validate_anchors(seq, raw, eos_id=args.eos_id,
                                             pad_id=args.pad_id)
            for rec in dropped:
                rec["video"] = name
            anchor_store["raw"][name] = [
                {"pos": int(a["pos"]), "n_acc": int(a["n_acc"]),
                 "prop": [int(t) for t in a["prop"]]} for a in raw]
            anchor_store["dropped"].extend(dropped)
            if dropped:
                log(f"[{tag}] {name}: dropped {len(dropped)} invalid anchor(s): "
                    + "; ".join(f"pos {d['pos']} ({d['reason']})" for d in dropped))
            if not kept:
                raise RuntimeError(f"no valid anchor for {name}")
            anchor_store["kept"][name] = kept
            anchor_store["trajectory_hash"][name] = vi.trajectory_sha256
        anchors = anchors_by_video[name] if not select_anchors else kept

        # ---- teacher rows first: the 8B prefill then never overlaps the head's
        # backward graph, which is what keeps the fp32 phase inside the A6000.
        t0 = time.perf_counter()
        p_rows = []
        anchor_index = 0
        for anchor in anchors:
            pos = int(anchor["pos"])
            prefix_ids = seq[:pos + 1][None]
            pos_hash = _position_hash(target, prefix_ids, pv, vg)
            if select_anchors:
                anchor_store["position_hash"][f"{name}:{pos}"] = pos_hash
            elif anchor_store["position_hash"].get(f"{name}:{pos}") != pos_hash:
                identity["position_hash_match"] = False
            vispos = M.prefix_vision_positions(target, prefix_ids,
                                               pixel_values_videos=pv,
                                               video_grid_thw=vg)
            _, prefix_kv = M.target_prefill_shared(
                target, prefix_ids, pixel_values_videos=pv, video_grid_thw=vg,
                max_seq_len=common.MAX_SEQ_LEN, vispos=vispos)
            if int(prefix_kv.get_seq_length()) != pos + 1:
                raise RuntimeError(f"prefix KV has {int(prefix_kv.get_seq_length())} "
                                   f"rows for anchor {pos}")
            p_logits = M.teacher_block_logits(
                target, vispos, prefix_kv, anchor_pos=pos,
                prop=list(anchor["prop"]), chunk=args.teacher_chunk)
            if int(p_logits.shape[0]) != int(anchor["K"]):
                raise RuntimeError(f"teacher scored {tuple(p_logits.shape)} for a "
                                   f"{anchor['K']}-offset anchor")
            if want_teacher_comparison:
                p_a = teacher_A[name][anchor_index].to(dev)
                with torch.no_grad():
                    pa = torch.softmax(p_a.float(), -1)
                    pb = torch.softmax(p_logits.float(), -1)
                    row_tv = 0.5 * (pa - pb).abs().sum(-1)
                    row_d = (p_a.float() - p_logits.float()).abs().max(-1).values
                    teacher_stat["tv"].extend(float(v) for v in row_tv)
                    teacher_stat["dlogit"].extend(float(v) for v in row_d)
                    teacher_stat["scale_pA"].extend(
                        float(v) for v in p_a.float().abs().max(-1).values)
                    teacher_stat["scale_pB"].extend(
                        float(v) for v in p_logits.float().abs().max(-1).values)
                    teacher_stat["argmax_match"] += int(
                        (p_a.argmax(-1) == p_logits.argmax(-1)).sum())
                    teacher_stat["argmax_total"] += int(p_logits.shape[0])
                    teacher_stat["n_rows"] += int(p_logits.shape[0])
            p_rows.append(p_logits)
            del prefix_kv, vispos
            anchor_index += 1
        torch.cuda.synchronize()
        teacher_seconds = time.perf_counter() - t0

        # ---- diagnostic: would the SAME rollout under this dtype have picked the
        # same anchors? Reported, never used to change the frozen sample.
        if not select_anchors:
            if anchor_store["trajectory_hash"].get(name) != vi.trajectory_sha256:
                identity["trajectory_hash_match"] = False
        if not select_anchors and args.diagnostic_rollout:
            raw_here = opd_online_rollout.rollout(
                head, target, fused_full, seq, prompt_len=vi.prompt_len,
                verify="chain", max_rounds=args.max_anchors)
            here = [{"pos": int(a["pos"]), "n_acc": int(a["n_acc"]),
                     "prop": [int(t) for t in a["prop"]]} for a in raw_here]
            rollout_diag.append({"video": name,
                                 "matches_frozen_rollout": here == raw_anchors_A[name],
                                 "frozen": raw_anchors_A[name], "this_phase": here})

        # ---- student chain replay with the graph: the same cache-carry and crop
        # the rollout used, so `q` is the distribution the rollout proposed from.
        t0 = time.perf_counter()
        cache = DynamicCache()
        total_main = None
        total_extra = None
        q_cpu = []
        expected_pos = int(vi.prompt_len)
        for index, anchor in enumerate(anchors):
            pos = int(anchor["pos"])
            n_acc = int(anchor["n_acc"])
            K = int(anchor["K"])
            if pos != expected_pos:
                raise RuntimeError(f"{name}: anchor {index} starts at {pos} but the "
                                   f"chain replay is at {expected_pos}")
            base = int(cache.get_seq_length())
            q_full = M.draft_block_logits(
                head, target, fused_full[:, base:pos], anchor_pos=pos,
                root_token=int(seq[pos]), prompt_len=int(vi.prompt_len), cache=cache)
            cache.crop(pos)                      # exactly `rollout`'s step
            q_logits = q_full[:K]
            p_logits = p_rows[index]

            loss, metrics = M.opd_fullkl_loss(p_logits, q_logits, n_acc)
            total_main = loss if total_main is None else total_main + loss
            losses = {main_combo: float(loss.detach())}
            metrics_out = {main_combo: metrics}

            # The extra teacher rows must stay ON the graph: this loss is the
            # second backward's objective, so it cannot be built under no_grad.
            p_extra = None
            loss_extra = metrics_extra = None
            if want_teacher_comparison:
                p_extra = teacher_A[name][index].to(dev)
                loss_extra, metrics_extra = M.opd_fullkl_loss(p_extra, q_logits, n_acc)
                total_extra = (loss_extra if total_extra is None
                               else total_extra + loss_extra)
                losses["pA_qB"] = float(loss_extra.detach())
                metrics_out["pA_qB"] = metrics_extra

            with torch.no_grad():
                combos[main_combo].add(p_logits, q_logits, n_acc,
                                       float(loss.detach()), metrics)
                if want_teacher_comparison:
                    # Same bf16 student rows against the cached fp32 teacher rows:
                    # the fixed-q loss isolates the teacher's dtype.
                    q_a = q_A[name][index].to(dev)
                    loss_fixed, metrics_fixed = M.opd_fullkl_loss(p_logits, q_a, n_acc)
                    losses["pB_qA"] = float(loss_fixed)
                    metrics_out["pB_qA"] = metrics_fixed
                    combos["pA_qB"].add(p_extra, q_logits, n_acc, float(loss_extra.detach()),
                                        metrics_extra)
                    combos["pB_qA"].add(p_logits, q_a, n_acc, float(loss_fixed),
                                        metrics_fixed)
                    tv_q = 0.5 * (torch.softmax(q_a.float(), -1)
                                  - torch.softmax(q_logits.float(), -1)).abs().sum(-1)
                    student_vs_A_tv.extend(float(v) for v in tv_q)

            with torch.no_grad():
                matches = q_logits.argmax(-1) == torch.as_tensor(
                    anchor["prop"], dtype=torch.long, device=q_logits.device)
                argmax_ok += int(matches.sum())
                argmax_tot += K
                per_anchor.append({
                    "video": name, "index": index,
                    "rollout_index": int(anchor["rollout_index"]),
                    "pos": pos, "n_acc": n_acc, "K": K, "losses": losses,
                    "student_argmax_matches_prop": int(matches.sum()),
                    "n_offsets": K,
                })
            if logits_store is not None:
                q_cpu.append(q_logits.detach().to(torch.float32).cpu())
            expected_pos = pos + n_acc + 1

        # ---- one backward per video over the summed block losses
        if want_teacher_comparison:
            if total_extra is None:
                raise RuntimeError("teacher comparison requested but no extra loss built")
            total_main.backward(retain_graph=True)
            _accumulate_cpu(acc_main, head)
            head.zero_grad(set_to_none=True)
            total_extra.backward()
            _accumulate_cpu(acc_extra, head)
        else:
            total_main.backward()
            _accumulate_cpu(acc_main, head)
        head.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        student_seconds = time.perf_counter() - t0

        # ---- keep p_A/q_A in RAM for the second phase (never on disk)
        if logits_store is not None:
            logits_store["p"][name] = [p.detach().to(torch.float32).cpu()
                                       for p in p_rows]
            logits_store["q"][name] = q_cpu

        del cache, fused_full, p_rows, q_cpu
        per_video_time.append({"video": name,
                               "ctx_seconds": ctx_seconds,
                               "teacher_seconds": teacher_seconds,
                               "student_backward_seconds": student_seconds,
                               "total_seconds": time.perf_counter() - video_start})
        log(f"[{tag}] {name:18s} anchors {len(anchors)}  ctx {ctx_seconds:5.1f}s  "
            f"teacher {teacher_seconds:5.1f}s  student+backward {student_seconds:5.1f}s  "
            f"peak {torch.cuda.max_memory_allocated(args.gpu) / 2**30:5.2f} GiB")

    peak_alloc = torch.cuda.max_memory_allocated(args.gpu) / 2**30
    peak_reserved = torch.cuda.max_memory_reserved(args.gpu) / 2**30
    wall = time.perf_counter() - phase_start

    n_with_grad = sum(1 for v in acc_main.values() if v is not None)
    grad_norm = math.sqrt(sum(float(v.double().pow(2).sum()) for v in acc_main.values()))
    for value in acc_main.values():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"[{tag}] a head gradient is not finite")
    target_grads = [n for n, p in target.named_parameters() if p.grad is not None]
    if target_grads:
        raise AssertionError(f"[{tag}] the frozen target acquired gradients: "
                             f"{target_grads[:3]}")

    result = {
        "tag": tag,
        "target_dtype": str(dtype),
        "head_dtype": str(dtype),
        "n_target_params": int(n_target_params),
        "n_head_params": int(n_head_params),
        "load_seconds": load_seconds,
        "wall_seconds": wall,
        "allocated_after_load_gib": after_load,
        "peak_allocated_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
        "per_video": per_video_time,
        "combos": {name: stats.summary() for name, stats in combos.items()},
        "per_anchor": per_anchor,
        "gradient": {"n_tensors_with_grad": n_with_grad,
                     "n_head_parameters": sum(1 for _ in head.named_parameters()),
                     "global_norm": grad_norm,
                     "target_acquired_gradients": bool(target_grads)},
        "student_argmax_vs_prop": {
            "offsets_matching": argmax_ok, "offsets_total": argmax_tot,
            "rate": (argmax_ok / argmax_tot) if argmax_tot else None},
        "student_vs_phaseA_logits": dist_summary(student_vs_A_tv),
        "identity": identity,
        "rollout_diagnostic": rollout_diag,
        "teacher_comparison": {
            "n_rows": teacher_stat["n_rows"],
            "tv": dist_summary(teacher_stat["tv"]),
            "max_abs_logit_diff": dist_summary(teacher_stat["dlogit"]),
            "scale_pA": dist_summary(teacher_stat["scale_pA"]),
            "scale_pB": dist_summary(teacher_stat["scale_pB"]),
            "argmax_agreement": {
                "matching": teacher_stat["argmax_match"],
                "total": teacher_stat["argmax_total"],
                "rate": (teacher_stat["argmax_match"] / teacher_stat["argmax_total"])
                        if teacher_stat["argmax_total"] else None,
            },
        },
    }

    del target, head
    gc.collect()
    torch.cuda.empty_cache()
    return result, acc_main, acc_extra, anchor_store


# --------------------------------------------------------------------------
# Aggregate report
# --------------------------------------------------------------------------

def aggregate(phase_a, phase_b, cfg, env, logits_meta, grads_a, grads_b,
              grads_extra, sweep, anchor_records, dropped, anchor_digest):
    combos = {"pA_qA": phase_a["combos"]["pA_qA"],
              "pB_qB": phase_b["combos"]["pB_qB"],
              "pB_qA": phase_b["combos"]["pB_qA"],
              "pA_qB": phase_b["combos"]["pA_qB"]}
    streams = {name: {
        "description": COMBO_NAMES[name],
        "accepted": combos[name]["accepted"],
        "rejected": combos[name]["rejected"],
        "n_acc_positions": combos[name]["n_acc_positions"],
        "n_rej_positions": combos[name]["n_rej_positions"],
        "loss": combos[name]["loss"],
        "l_acc_per_anchor_mean": combos[name]["l_acc_per_anchor_mean"],
        "l_rej_per_anchor_mean": combos[name]["l_rej_per_anchor_mean"],
    } for name in COMBO_NAMES}

    losses = {name: combos[name]["loss"]["total"] for name in COMBO_NAMES}
    loss_comparison = {
        "totals": losses,
        "own_student__bf16_minus_fp32": diff_stats(losses["pA_qA"], losses["pB_qB"]),
        "fixed_qA__bf16_teacher_minus_fp32_teacher":
            diff_stats(losses["pA_qA"], losses["pB_qA"]),
        "fixed_qB__bf16_teacher_minus_fp32_teacher":
            diff_stats(losses["pA_qB"], losses["pB_qB"]),
        "note": ("totals are the sum of the per-anchor block losses over all "
                 "anchors; the fixed-q rows use the same student logits on both "
                 "sides, so only the teacher's dtype differs"),
    }

    gradients = {
        "between_phases": M.grad_cosine_and_norm_ratio(grads_a, grads_b),
        "between_phases_note": ("phase A is fp32 target + fp32 head, phase B is bf16 "
                                "target + bf16 head; the cosine therefore mixes the "
                                "teacher's dtype with the student's"),
        "isolated_teacher__bf16_student":
            M.grad_cosine_and_norm_ratio(grads_b, grads_extra) if grads_extra else None,
        "isolated_teacher_note": ("within phase B, the same bf16 student graph "
                                  "backwarded against cached fp32 teacher rows vs "
                                  "bf16 teacher rows; the head dtype is fixed, so "
                                  "this is the teacher's own gradient effect"),
        "phase_A": phase_a["gradient"],
        "phase_B": phase_b["gradient"],
    }

    accepted_lengths = [a["n_acc"] for a in phase_a["per_anchor"]]
    hist = {}
    for value in accepted_lengths:
        hist[str(int(value))] = hist.get(str(int(value)), 0) + 1
    accepted_length = {
        "n_anchors": len(accepted_lengths),
        "histogram": hist,
        "mean": float(np.mean(accepted_lengths)),
        "median": float(np.median(accepted_lengths)),
        "min": int(np.min(accepted_lengths)),
        "max": int(np.max(accepted_lengths)),
        "total_offsets": int(sum(a["K"] for a in phase_a["per_anchor"])),
        "total_accepted": int(sum(accepted_lengths)),
        "total_rejected": int(sum(a["K"] - a["n_acc"] for a in phase_a["per_anchor"])),
    }

    merged = {}
    for rec in phase_a["per_anchor"]:
        merged[(rec["video"], rec["index"])] = {
            "video": rec["video"], "index": rec["index"], "pos": rec["pos"],
            "n_acc": rec["n_acc"], "K": rec["K"], "losses": dict(rec["losses"]),
            "student_argmax_matches_prop_A": rec["student_argmax_matches_prop"],
            "n_offsets": rec["n_offsets"]}
    for rec in phase_b["per_anchor"]:
        entry = merged[(rec["video"], rec["index"])]
        entry["losses"].update(rec["losses"])
        entry["student_argmax_matches_prop_B"] = rec["student_argmax_matches_prop"]
    per_anchor_table = [merged[k] for k in sorted(merged)]

    teacher = phase_b["teacher_comparison"]
    return {
        "config": cfg,
        "environment": env,
        "identity_checks": {
            "anchors_identical_across_phases": bool(
                phase_b["identity"]["anchor_tokens_identical"]),
            "anchor_digest": anchor_digest,
            "teacher_prefix_position_hash_identical": bool(
                phase_b["identity"]["position_hash_match"]),
            "trajectory_hash_identical": bool(phase_b["identity"]["trajectory_hash_match"]),
            "frozen_rollout_used_in_both_phases": True,
            "bf16_rollout_would_match_frozen_anchors": (
                all(d["matches_frozen_rollout"] for d in phase_b["rollout_diagnostic"])
                if phase_b["rollout_diagnostic"] else None),
        },
        "anchors": anchor_records,
        "anchors_dropped": dropped,
        "accepted_length": accepted_length,
        "streams": streams,
        "teacher_distribution": {
            "n_rows": teacher["n_rows"],
            "total_variation": teacher["tv"],
            "max_abs_logit_diff": teacher["max_abs_logit_diff"],
            "argmax_agreement": teacher["argmax_agreement"],
            "fp32_logit_absmax": teacher["scale_pA"],
            "bf16_logit_absmax": teacher["scale_pB"],
        },
        "loss_comparison": loss_comparison,
        "per_anchor_losses": per_anchor_table,
        "gradients": gradients,
        "memory_time": {
            "phase_A_fp32": {k: phase_a[k] for k in
                             ("load_seconds", "wall_seconds", "allocated_after_load_gib",
                              "peak_allocated_gib", "peak_reserved_gib")},
            "phase_B_bf16": {k: phase_b[k] for k in
                             ("load_seconds", "wall_seconds", "allocated_after_load_gib",
                              "peak_allocated_gib", "peak_reserved_gib")},
            "per_video": {"phase_A": phase_a["per_video"], "phase_B": phase_b["per_video"]},
            "gpu": env["gpu_name"],
        },
        "student_diagnostics": {
            "phase_A_argmax_vs_prop": phase_a["student_argmax_vs_prop"],
            "phase_B_argmax_vs_prop": phase_b["student_argmax_vs_prop"],
            "phase_B_minus_phase_A_student_logits_tv": phase_b["student_vs_phaseA_logits"],
        },
        "logits_in_ram_only": logits_meta,
        "rollout_diagnostic": phase_b["rollout_diagnostic"],
        "data_discipline": sweep,
        "notes": [
            "Aggregate statistics only; no logits, distributions or KV caches were "
            "written to disk.",
            "The teacher's dtype cannot change without also changing the dtype of the "
            "hidden states the head consumes, so the head follows the target's dtype "
            "(fp32 in A, bf16 in B). The fixed-q rows and the isolated-teacher gradient "
            "exist to separate those two effects.",
            "KL percentiles are numpy linear percentiles over pooled per-offset values; "
            "the accepted pool is forward KL(p||q), the rejected pool is reverse KL(q||p).",
            "The anchors are the FIRST four chain rounds, i.e. the prompt-adjacent part "
            "of the trajectory, where the aligned head accepts few offsets; the accepted "
            "stream is therefore estimated from very few positions and its P99/max are "
            "not robust.",
            "Anchors (pos/prop/n_acc) are frozen from the fp32 rollout and replayed in "
            "bf16. A bf16 rollout of the same trajectory can propose different later "
            "tokens, so phase B's student argmax need not reproduce the frozen prop; "
            "see identity_checks.bf16_rollout_would_match_frozen_anchors and "
            "student_diagnostics.",
        ],
    }


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def _fmt(summary):
    if not summary or summary.get("n", 0) == 0:
        return "n=0"
    return (f"n={summary['n']:d} mean={summary['mean']:.6g} p50={summary['p50']:.6g} "
            f"p90={summary['p90']:.6g} p99={summary['p99']:.6g} max={summary['max']:.6g}")


def print_report(rep):
    log("")
    log("=" * 100)
    log("B0.5 calibration -- aggregate table")
    log("=" * 100)
    al = rep["accepted_length"]
    log(f"anchors {al['n_anchors']}  offsets {al['total_offsets']}  "
        f"accepted {al['total_accepted']}  rejected {al['total_rejected']}  "
        f"n_acc mean {al['mean']:.3f} median {al['median']:.1f} max {al['max']}  "
        f"histogram {al['histogram']}")
    log("")
    log("per-stream KL (accepted = KL(p||q), rejected = KL(q||p), full vocabulary)")
    for name, entry in rep["streams"].items():
        log(f"  {name}  ({COMBO_NAMES[name]})")
        log(f"    accepted ({entry['n_acc_positions']:4d} positions): {_fmt(entry['accepted'])}")
        log(f"    rejected ({entry['n_rej_positions']:4d} positions): {_fmt(entry['rejected'])}")
    log("")
    t = rep["teacher_distribution"]
    log(f"teacher fp32 vs bf16 on {t['n_rows']} identical rows")
    log(f"  total variation    : {_fmt(t['total_variation'])}")
    log(f"  max abs logit diff : {_fmt(t['max_abs_logit_diff'])}")
    log(f"  logit absmax fp32/bf16: {_fmt(t['fp32_logit_absmax'])} | "
        f"{_fmt(t['bf16_logit_absmax'])}")
    log(f"  argmax agreement   : {t['argmax_agreement']['matching']}/"
        f"{t['argmax_agreement']['total']} = {t['argmax_agreement']['rate']:.6f}")
    log("")
    lc = rep["loss_comparison"]
    for name, total in lc["totals"].items():
        log(f"  total loss {name:6s} = {total:.6f}   ({COMBO_NAMES[name]})")
    for key in ("own_student__bf16_minus_fp32",
                "fixed_qA__bf16_teacher_minus_fp32_teacher",
                "fixed_qB__bf16_teacher_minus_fp32_teacher"):
        d = lc[key]
        log(f"  {key:44s} abs={d['abs']:.6e}  rel={d['rel']:.6e}")
    log("")
    g = rep["gradients"]
    log("head gradients (per-tensor accumulate; whole-model concatenation never built)")
    for key in ("between_phases", "isolated_teacher__bf16_student"):
        entry = g[key]
        if entry is None:
            log(f"  {key}: not computed")
            continue
        log(f"  {key:32s} cosine={entry['cosine']:.6f}  norm_ratio(A/B)="
            f"{entry['norm_ratio']:.6f}  tensors={entry['n_tensors']}  "
            f"||A||={entry['norm_a']:.4e}  ||B||={entry['norm_b']:.4e}")
    log("")
    mt = rep["memory_time"]
    for tag in ("phase_A_fp32", "phase_B_bf16"):
        m = mt[tag]
        log(f"  {tag:12s} wall {m['wall_seconds']:7.1f}s (load {m['load_seconds']:.1f}s)  "
            f"peak allocated {m['peak_allocated_gib']:.2f} GiB  "
            f"peak reserved {m['peak_reserved_gib']:.2f} GiB")
    log("")
    sd = rep["student_diagnostics"]
    log(f"student argmax vs the rollout's prop: "
        f"A {sd['phase_A_argmax_vs_prop']['rate']}  "
        f"B {sd['phase_B_argmax_vs_prop']['rate']}")
    log(f"bf16 rollout would re-select the frozen anchors: "
        f"{rep['identity_checks']['bf16_rollout_would_match_frozen_anchors']}")
    log(f"identity: {rep['identity_checks']}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="B0.5: fp32-vs-bf16 teacher calibration on the frozen 8-video sample")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--head", default=f"{common.CKPT}/head_video439_aligned/step1000")
    ap.add_argument("--subsets", default=f"{common.RESULTS}/phaseB05/subsets.json")
    ap.add_argument("--pool", default=f"{common.DATA}/pool_499.parquet")
    ap.add_argument("--records", default=f"{common.DATA}/video_greedy_499.pt")
    ap.add_argument("--out", default=f"{common.RESULTS}/phaseB05/b05_calibration.json")
    ap.add_argument("--log", default=f"{common.ROOT}/logs/phaseB05_calibration.log")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-anchors", type=int, default=4)
    ap.add_argument("--teacher-chunk", type=int, default=8)
    ap.add_argument("--videos", type=int, default=0,
                    help="smoke only: score the first N of the frozen 8 (0 = all)")
    ap.add_argument("--no-diagnostic-rollout", action="store_true")
    args = ap.parse_args()
    args.diagnostic_rollout = not args.no_diagnostic_rollout

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    watched = [os.path.dirname(os.path.abspath(args.out)),
               os.path.dirname(os.path.abspath(args.log))]
    before = {d: set(os.listdir(d)) for d in watched}

    handle = open(args.log, "w")
    sys.stdout = _Tee(sys.__stdout__, handle)
    sys.stderr = _Tee(sys.__stderr__, handle)

    # The pipeline is deterministic (greedy rollout, frozen sample); the seed is
    # set and recorded so "fixed seed" is a fact about the run, not a claim.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # A "float32 teacher" means float32 GEMMs: TF32 is a different, lower-precision
    # mode and must not be silently substituted. The effective flags are recorded.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dev = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    log("=" * 100)
    log("B0.5 calibration: bf16-vs-fp32 teacher on the Draft-OPD full-KL objective")
    log("=" * 100)
    log(f"[env] gpu {args.gpu} = {torch.cuda.get_device_name(args.gpu)}  "
        f"torch {torch.__version__}  cuda {torch.version.cuda}  "
        f"transformers {__import__('transformers').__version__}")
    log(f"[env] allow_tf32 matmul={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn={torch.backends.cudnn.allow_tf32}  seed={args.seed}")
    log(f"[cfg] head {args.head}")
    log(f"[cfg] out  {args.out}")
    log(f"[cfg] log  {args.log}")

    processor = AutoProcessor.from_pretrained(common.TARGET_PATH)
    videos = build_videos(args, processor)

    # Token ids the validity filter uses. The text config gives EOS; the pad id is
    # the head's bos/pad (151643), which is what Qwen pads a short row with and
    # what the target's own config leaves as None.
    target_cfg = AutoConfig.from_pretrained(common.TARGET_PATH)
    args.eos_id = int(target_cfg.text_config.eos_token_id)
    args.pad_id = int(json.load(open(os.path.join(args.head, "config.json")))
                      .get("bos_token_id", 151643))
    log(f"[cfg] eos_id {args.eos_id}  pad_id {args.pad_id}")

    cfg = {
        "plan_section": "10.2 B0.5 calibration",
        "head": args.head,
        "head_note": "aligned SFT start point (validation mean tau 3.7377)",
        "target": common.TARGET_PATH,
        "videos": list(videos.keys()),
        "n_videos": len(videos),
        "video_source": args.subsets,
        "video_selection": "frozen score-free sha256 draw (subsets.json b05_train_videos)",
        "trajectories": args.records,
        "pool": args.pool,
        "prompt_check": "encoded processor prompt length == stored prompt_len",
        "seed": args.seed,
        "anchor_rule": SELECTION_RULE,
        "max_anchors": args.max_anchors,
        "rollout_verify": "chain",
        "teacher_chunk": args.teacher_chunk,
        "block_size": 16,
        "gamma": 0.8,
        "lambda_acc": 1.0,
        "lambda_rej": 1.0,
        "loss": "opd_fullkl.opd_fullkl_loss (full vocabulary, no top-k)",
        "objective": "accepted = forward KL(p||q) plain mean; rejected = reverse "
                     "KL(q||p) weighted 0.8**(k-1); each stream normalised over the "
                     "streams present",
        "student_logits": "opd_fullkl.draft_block_logits, carried chain cache with "
                          "cache.crop (the rollout's own proposal path)",
        "teacher_logits": "opd_fullkl.teacher_block_logits (forked prefix KV, "
                          "draft-induced prefix)",
        "gradient_accumulation": "one backward per video over the summed block losses, "
                                 "then {name: tensor.cpu()}",
        "gradient_reduction": "opd_fullkl.grad_cosine_and_norm_ratio (per tensor)",
        "eos_id": args.eos_id,
        "pad_id": args.pad_id,
        "dtype_policy": "phase A target+head fp32; phase B target+head bf16",
        "identical_across_phases": "anchor pos/prop/n_acc, committed tokens, teacher "
                                   "prefix ids, M-RoPE positions (hashed) and mask",
        "smoke_subset": bool(args.videos),
    }
    env = {
        "gpu_index": args.gpu,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "gpu_total_gib": torch.cuda.get_device_properties(args.gpu).total_memory / 2**30,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "torch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }

    # ---------------- phase A: fp32 target, selects and freezes the anchors -----
    log("")
    log("-" * 100)
    log("PHASE A -- fp32 target, anchor selection, fp32 teacher/student gradients")
    log("-" * 100)
    logits_store = {"p": {}, "q": {}}
    phase_a, grads_a, _, anchor_store = run_phase(
        tag="A_fp32", dtype=torch.float32, args=args, videos=videos,
        head_path=args.head, dev=dev, main_combo="pA_qA",
        combos_this_phase=("pA_qA",), logits_store=logits_store,
        select_anchors=True)

    # Round-trip the frozen anchors through JSON before phase B uses them, so
    # "identical anchors" is a structural property of the data, not object identity.
    kept_roundtrip = json.loads(json.dumps(anchor_store["kept"]))
    want_digest = anchors_digest(kept_roundtrip)
    anchor_records = [{"video": name, **anchor}
                      for name, anchors in kept_roundtrip.items() for anchor in anchors]
    log(f"[anchors] frozen {len(anchor_records)} anchors over {len(kept_roundtrip)} "
        f"videos; digest {want_digest[:16]}")

    # ---------------- phase B: bf16 target, frozen anchors replayed -------------
    log("")
    log("-" * 100)
    log("PHASE B -- bf16 target, frozen anchors replayed, bf16 teacher/student gradients")
    log("-" * 100)
    phase_b, grads_b, grads_extra, _ = run_phase(
        tag="B_bf16", dtype=torch.bfloat16, args=args, videos=videos,
        head_path=args.head, dev=dev, main_combo="pB_qB",
        combos_this_phase=("pB_qB", "pB_qA", "pA_qB"),
        anchor_digest_wanted=want_digest, anchors_by_video=kept_roundtrip,
        raw_anchors_A=anchor_store["raw"], anchor_store=anchor_store,
        teacher_A=logits_store["p"], q_A=logits_store["q"],
        select_anchors=False, want_teacher_comparison=True)

    sweep = {"unexpected_new_files": [], "checked_dirs": watched,
             "logits_or_kv_written": False}
    expected = {os.path.abspath(args.out), os.path.abspath(args.log)}
    for directory in watched:
        new = set(os.listdir(directory)) - before[directory]
        for name in sorted(new):
            path = os.path.abspath(os.path.join(directory, name))
            if path not in expected:
                sweep["unexpected_new_files"].append(path)
    if sweep["unexpected_new_files"]:
        raise AssertionError(f"unexpected files appeared: {sweep['unexpected_new_files']}")
    for directory in watched:
        for name in os.listdir(directory):
            if name.endswith((".pt", ".safetensors", ".npz", ".npy", ".pth", ".bin")):
                sweep["logits_or_kv_written"] = True
    if sweep["logits_or_kv_written"]:
        raise AssertionError("a tensor file appeared in a watched directory")

    report = aggregate(
        phase_a, phase_b, cfg, env,
        {"p_cached": int(sum(t.numel() for rows in logits_store["p"].values() for t in rows)),
         "q_cached": int(sum(t.numel() for rows in logits_store["q"].values() for t in rows)),
         "note": "phase A teacher/student logits kept in RAM only, freed at process exit"},
        grads_a, grads_b, grads_extra, sweep, anchor_records,
        anchor_store["dropped"], want_digest)

    print_report(report)
    with open(args.out, "w") as fh:
        json.dump(sanitize(report), fh, indent=1)
    log("")
    log(f"[done] wrote {args.out}")
    log(f"[done] wrote {args.log}")
    log(f"[done] sweep: unexpected files {sweep['unexpected_new_files']}, "
        f"logits_or_kv_written={sweep['logits_or_kv_written']}")
    handle.flush()


if __name__ == "__main__":
    main()
