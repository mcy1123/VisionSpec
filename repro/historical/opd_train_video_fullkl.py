#!/usr/bin/env python3
"""Compatibility entry to opd_official_kl.cli (Bernoulli/k3 explicit override).

Historical helper definitions remain importable by diagnostic scripts only.
The former training main is disabled; original source is under legacy/pre20261016.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


# Historical helper API retained for diagnostics. New training dispatches only to the canonical runtime.
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _repro = _REPRO_ROOT
    _sys.path.insert(0, str(_repro))
    from opd_official_kl.cli import main as _official_main
    _official_main(["--modality", "video", *_sys.argv[1:]])
    raise SystemExit(0)


import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, str(_REPRO_ROOT))

from shared import common  # noqa: E402
from historical import opd_fullkl  # noqa: E402
from historical import opd_online_rollout  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

#: The one block layout this trainer scores. `draft_block_logits` returns
#: `block_size - 1` rows (the head's last `block-1` hidden rows are the block
#: marginals), so the objective's `K` is fixed by the head, not by `len(prop)`.
K_OFFSETS = common.BLOCK_SIZE - 1


# ==========================================================================
# Pure arithmetic: the update-level normalisation (CPU-testable, no model)
# ==========================================================================

def build_offset_weights(K: int, gamma: float, rej_offsets: torch.Tensor,
                         *, device=None, dtype=torch.float32) -> torch.Tensor:
    """`w_k = gamma**(k-1)` for the 1-based `rej_offsets`, evaluated in float64.

    The schedule is a constant, so it is evaluated in float64 and rounded once:
    `0.8 ** 2` through fp32 `pow()` is a few ulp off `gamma**(k-1)`, and that
    would show up as a spurious 1e-7 in every denominator check (the same reason
    `opd_fullkl.opd_fullkl_loss` computes it this way).
    """
    device = torch.device("cpu") if device is None else device
    idx = rej_offsets.detach().to(device=device)
    # The exponentiation AND the cast happen in float64 (`.double()` on the
    # float64 base would be a no-op, and `idx.double()` keeps the exponent exact
    # for the offsets we use); the result is narrowed once, at the end. Narrowing
    # first and comparing later is what produced the last-ulp disagreements this
    # comment is here to prevent.
    w = torch.pow(torch.as_tensor(float(gamma), dtype=torch.float64, device=device),
                  idx.double() - 1.0)
    return w.to(dtype)


def plan_update_denominators(plans):
    """Freeze one update's global denominators from its anchors.

    `plans` is a sequence of objects carrying the two per-anchor numbers as
    attributes -- `AnchorPlan` below, or any stand-in in a test:
    `n_acc_total` (int, the anchor's supervised accepted offsets) and
    `sum_w_rej` (float, the anchor's own `Z`).

    Returns `(N_acc_total, Z_total, acc_present, rej_present)`, where "present"
    means non-empty over the WHOLE update, which is the condition the paper's
    drop-and-renormalise rule is written for. `Z_total` accumulates in float64
    so that summing thousands of weights cannot lose the denominator's last
    digits; the caller casts the value next to the tensor multiply.
    """
    n_acc = 0
    z = 0.0
    for plan in plans:
        n_acc += int(plan.n_acc_total)
        z += float(plan.sum_w_rej)
    return n_acc, z, n_acc > 0, z > 0.0


def microbatch_scaled_loss(acc_kl, rej_kl, rej_w, *, n_acc_total, z_total,
                           acc_present, rej_present):
    """One microbatch's contribution to the update objective.

    With `S_acc = sum_{I_acc} KL(p||q)` and
    `S_rej = sum_{I_rej} w_k * KL(q||p)`:

        both streams -> (S_acc / N_acc_total + S_rej / Z_total) / 2
        accepted only ->  S_acc / N_acc_total
        rejected only ->  S_rej / Z_total
        neither       ->  0.0 (a float: no graph exists to join)

    `acc_kl`/`rej_kl` are this anchor's per-offset KLs (`opd_fullkl.full_kl_forward`
    / `full_kl_reverse` on the `stream_split` index sets) and `rej_w` the matching
    `w_k`. A stream can be present over the update and absent in this microbatch;
    that anchor then contributes nothing and, importantly, no term to the other
    stream's normalisation.

    The caller guarantees a stream is present over the update iff its global
    count is positive (`plan_update_denominators`), so both denominators used
    here are non-zero whenever a term is built.
    """
    if n_acc_total < 0 or z_total < 0.0:
        raise ValueError(f"denominators must be non-negative, got "
                         f"{n_acc_total}, {z_total}")
    if acc_present and n_acc_total == 0:
        raise ValueError("accepted stream marked present with N_acc_total == 0")
    if rej_present and z_total <= 0.0:
        raise ValueError("rejected stream marked present with Z_total == 0")
    if not acc_present and not rej_present:
        return 0.0

    terms = []
    if acc_present and acc_kl.numel() > 0:
        terms.append(acc_kl.sum() / float(n_acc_total))
    if rej_present and rej_kl.numel() > 0:
        terms.append((rej_w * rej_kl).sum() / float(z_total))
    if not terms:
        # The update has streams, this anchor has none. Contribute nothing
        # rather than a zero that could perturb a denominator.
        return 0.0
    # Equal stream weights (lambda_acc = lambda_rej = 1) over the streams the
    # UPDATE has, NOT over the streams this microbatch happens to have. The
    # divisor is fixed by the update (`2` when both streams exist there, `1`
    # when one was dropped and renormalised), because summing the microbatches
    # must reproduce
    #     (S_acc_total / N_acc_total + S_rej_total / Z_total) / 2.
    # A per-microbatch divisor would weight an all-accepted anchor twice as
    # heavily as a mixed one and the sum would overshoot the objective.
    return sum(terms) / (2.0 if (acc_present and rej_present) else 1.0)


def paper_objective(streams, gamma: float):
    """The paper's Eq. 8-13 objective for one UPDATE, from per-offset KLs.

    `streams` is a sequence of `(acc_kl, n_acc_used, rej_kl, rej_w)` per anchor,
    with `acc_kl`/`rej_kl` sequences of python floats (or 1-element tensors) and
    `rej_w` the matching weights. `gamma` is accepted for signature clarity and
    checked against nothing here: the weights are inputs, so a test can build
    them with `build_offset_weights` and see the same numbers the trainer uses.

    This is the oracle `microbatch_scaled_loss` is checked against; the
    per-anchor scaled losses must SUM to exactly `loss`, for every combination
    of empty streams. The two halves are returned separately because a wrong
    denominator can cancel in the total.
    """
    acc_all: list[float] = []
    rej_pairs: list[tuple[float, float]] = []
    for acc_kl, n_acc_used, rej_kl, rej_w in streams:
        if int(n_acc_used) != len(acc_kl):
            raise ValueError(f"n_acc_used {n_acc_used} != {len(acc_kl)} accepted KLs")
        if len(rej_kl) != len(rej_w):
            raise ValueError(f"{len(rej_kl)} rejected KLs but {len(rej_w)} weights")
        acc_all.extend(float(x) for x in acc_kl)
        rej_pairs.extend((float(k), float(w)) for k, w in zip(rej_kl, rej_w))
    if float(gamma) <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    l_acc = sum(acc_all) / len(acc_all) if acc_all else 0.0
    z_all = sum(w for _, w in rej_pairs)
    l_rej = sum(k * w for k, w in rej_pairs) / z_all if z_all > 0.0 else 0.0
    terms = []
    if acc_all:
        terms.append(l_acc)
    if z_all > 0.0:
        terms.append(l_rej)
    total = sum(terms) / len(terms) if terms else 0.0
    return {"loss": total, "l_acc": l_acc, "l_rej": l_rej,
            "n_acc_total": len(acc_all), "z_total": z_all}


# ==========================================================================
# Pure planning: anchor selection, validity and order (CPU-testable)
# ==========================================================================

def trajectory_valid_end(full_ids: torch.Tensor, eos_token_id: int,
                         start: int = 0) -> int:
    """Index of the first EOS in `full_ids[start:]`, or `len-1` when there is none.

    Nothing whose TARGET TOKEN sits past this index is supervised; the EOS token
    at this index itself is (see the module docstring's EOS paragraph).

    `start` MUST be the prompt length. The chat template ends the user turn with
    `<|im_end|>`, which IS the eos id (151645), so it appears inside the prompt of
    every row -- e.g. one train row has prompt_len 958 and an `<|im_end|>` at 953.
    Searching from index 0 therefore returns a valid_end *before* the prompt,
    every offset is judged past-EOS, `select_anchors` keeps nothing, and pass 1
    walks the whole dataset building and discarding traces (measured: 439 videos
    x ~2.3 s of 100%-GPU work with flat memory, i.e. an apparent hang). Anchors
    live at `pos >= prompt_len` by construction, so only EOS at or after the
    prompt can end the response.
    """
    if full_ids.ndim != 1:
        raise ValueError(f"full_ids must be 1-D, got {tuple(full_ids.shape)}")
    n = int(full_ids.shape[0])
    lo = max(0, int(start))
    hits = (full_ids[lo:] == int(eos_token_id)).nonzero(as_tuple=False)
    if hits.numel() == 0:
        return n - 1
    return lo + int(hits[0, 0].item())


def anchor_valid_mask(pos: int, total: int, valid_end: int,
                      block: int = common.BLOCK_SIZE) -> list[bool]:
    """The `valid` mask `opd_fullkl.stream_split` takes, for one anchor.

    Offset `k` (1-based) supervises the target token at stream position `pos+k`;
    it is supervised iff that position exists in the stored trajectory
    (`< total`) and is not past the first EOS (`<= valid_end`). `valid_end` is an
    absolute stream index so this stays a pure index computation.
    """
    k = block - 1
    out = []
    for off in range(1, k + 1):
        at = int(pos) + off
        out.append(at < int(total) and at <= int(valid_end))
    return out


def eos_excluded_positions(pos: int, total: int, valid_end: int,
                           block: int = common.BLOCK_SIZE) -> int:
    """How many of this anchor's offsets the EOS/trajectory guard removed.

    Logged per update (`n_eos_excluded`) so an EOS-heavy pool cannot silently
    delete supervision without the log showing it.
    """
    return sum(1 for keep in anchor_valid_mask(pos, total, valid_end, block)
               if not keep)


def select_anchors(anchors, *, budget: int, max_seq_len: int, prompt_len: int,
                   valid_end: int, block: int = common.BLOCK_SIZE,
                   per_video: int = 0):
    """Choose this video's anchors for one update, in rollout order.

    `anchors` are the dicts `opd_online_rollout.rollout` returns, in the order
    the rollout visited them; they are EXAMINED and KEPT in that order, so the
    update's anchor order is a pure function of a deterministic rollout plus the
    budget -- never of a dict or set iteration order.

    An anchor is eligible iff a FULL `block-1` block fits inside the supervised
    trajectory:

        prompt_len <= pos  and  pos + (block-1) < total
        and  pos + (block-1) <= valid_end  and  pos + (block-1) < max_seq_len

    i.e. its last offset has a real target token at or before the first EOS and
    inside the prefilled trajectory. Every offset of an eligible anchor is then
    filtered individually by `anchor_valid_mask`, which is what carries the
    EOS/trajectory exclusion into both streams and both denominators. Ineligible
    anchors are skipped (their count is reported), which keeps a chain rollout
    usable through its tail instead of padding a short block.

    Returns `(kept, info)`.
    """
    if block < 2:
        raise ValueError(f"block must be >= 2, got {block}")
    k = block - 1
    kept = []
    info = {"available": len(anchors), "skipped_tail": 0, "skipped_short": 0,
            "stopped_at_budget": False}
    for anchor in anchors:
        if len(kept) >= int(budget):
            info["stopped_at_budget"] = True
            break
        if per_video > 0 and len(kept) >= int(per_video):
            break
        pos = int(anchor["pos"])
        if pos < int(prompt_len):
            raise ValueError(f"rollout returned anchor pos {pos} before "
                             f"prompt_len {prompt_len}")
        if len(anchor["prop"]) < k:
            # A truncated final block: the rollout can end with fewer than the
            # full horizon. Scoring it would mean padding `prop` (teacher) and
            # the student's logits with tokens the model never proposed.
            info["skipped_short"] += 1
            continue
        if anchor.get("source") == "current_target":
            if pos + k >= int(max_seq_len) or not any(anchor["valid"]):
                info["skipped_tail"] += 1
                continue
            kept.append(anchor)
            continue
        if pos + k > int(max_seq_len) - 1 or pos + k > int(valid_end):
            info["skipped_tail"] += 1
            continue
        kept.append(anchor)
    return kept, info


@dataclass
class AnchorPlan:
    """One supervised anchor: everything pass 2 needs, and nothing else.

    `offsets_acc`/`offsets_rej` are the 1-based offset tensors
    `opd_fullkl.stream_split` produced, already excluding every invalid offset;
    their `numel()`s are the anchor's contribution to the update's denominators.
    `n_acc_total` and `sum_w_rej` are the two numbers
    `plan_update_denominators` sums.
    """

    video: str
    pos: int
    prop: list[int]
    n_acc: int
    valid: list[bool]
    offsets_acc: torch.Tensor
    offsets_rej: torch.Tensor
    n_acc_total: int
    sum_w_rej: float
    n_invalid: int = 0
    teacher_ids: list[int] | None = None

    def as_log(self) -> dict:
        return {"video": self.video, "pos": self.pos, "K": len(self.prop),
                "n_acc": self.n_acc, "n_acc_supervised": self.n_acc_total,
                "sum_w_rej": self.sum_w_rej, "n_invalid": self.n_invalid,
                **({"source": "current_target", "prop": self.prop,
                    "teacher_ids": self.teacher_ids, "valid": self.valid}
                   if self.teacher_ids is not None else {})}


def build_anchor_plan(video: str, anchor: dict, *, total: int, valid_end: int,
                      prompt_len: int, block: int = common.BLOCK_SIZE,
                      gamma: float = 0.8, device=None,
                      valid: list[bool] | None = None) -> AnchorPlan:
    """Turn one eligible rollout anchor into an `AnchorPlan`.

    The split comes from `opd_fullkl.stream_split(K, n_acc, valid)`, the verified
    single source of truth for which offsets are accepted/rejected under the
    `valid` mask -- `k <= n_acc` is deliberately not re-derived here, because a
    second copy of that rule is exactly the kind of thing that drifts.

    `valid` defaults to `None`, which means "derive it from `(pos, total,
    valid_end)` with `anchor_valid_mask`" -- the trainer's only mode. Tests pass
    an explicit mask so they can exercise an EOS-truncated block without having to
    construct a trajectory that produces one; the mask is validated like any
    other so an injected mask cannot become a second, unchecked code path.
    """
    pos = int(anchor["pos"])
    prop = [int(t) for t in anchor["prop"]]
    n_acc = int(anchor["n_acc"])
    k = block - 1
    if len(prop) < k:
        raise ValueError(f"{video!r} anchor at {pos} carries a {len(prop)}-token "
                         f"proposal; a full {k}-offset block is required")
    if not 0 <= n_acc <= k:
        raise ValueError(f"{video!r} anchor at {pos}: n_acc {n_acc} outside "
                         f"[0, {k}]")
    if valid is None:
        valid = anchor.get("valid")
    if valid is None:
        valid = anchor_valid_mask(pos, total, valid_end, block)
    else:
        valid = [bool(v) for v in valid]
        if len(valid) != k:
            raise ValueError(f"valid must have {k} entries, got {len(valid)}")
    acc_idx, rej_idx = opd_fullkl.stream_split(k, n_acc, valid, device=device)
    if int(acc_idx.numel()) + int(rej_idx.numel()) != int(sum(valid)):
        raise RuntimeError("stream_split's two sets do not partition the valid "
                           "offsets; the module and this trainer disagree")
    w = build_offset_weights(k, gamma, rej_idx, device=device)
    return AnchorPlan(
        video=video, pos=pos, prop=prop, n_acc=n_acc, valid=valid,
        offsets_acc=acc_idx, offsets_rej=rej_idx,
        n_acc_total=int(acc_idx.numel()),
        # Accumulated in float64 and narrowed once: this number is a DENOMINATOR,
        # and summing 15 fp32 weights is exactly where a denominator check would
        # otherwise lose its last digits.
        sum_w_rej=(float(w.double().sum()) if w.numel() else 0.0),
        n_invalid=sum(1 for v in valid if not v),
        teacher_ids=anchor.get("teacher_ids"),
    )


def video_order(n: int, seed: int, epoch: int) -> list[int]:
    """A fixed, seeded permutation of the training rows for one epoch.

    WHY a permutation rather than a step-indexed draw: an update consumes a
    VARIABLE number of videos (it stops when the anchor budget is full), so
    `step % n` would make the video sequence depend on how many anchors each
    video yielded, which depends on the policy being trained. A per-epoch
    permutation is reproducible from `(seed, epoch)` alone and still shuffles.
    """
    order = list(range(int(n)))
    random.Random(int(seed) * 1_000_003 + int(epoch)).shuffle(order)
    return order


# ==========================================================================
# Model-facing helpers: the only functions that touch CUDA
# ==========================================================================

@dataclass
class VideoTrace:
    """Pass-1 state pass 2 reuses for one video's anchors.

    `fused` is the head's context for the WHOLE trajectory and `prefix_kv` the
    teacher's full-trajectory prefix cache: both are the output of the single
    `opd_fullkl.target_prefill_shared` call the design requires per video.
    `prefix_hidden` is that same prefill's final hidden state for every position,
    `(1, total, H)`, i.e. the array `lm_head` consumes; the teacher's offset-1
    row for anchor `pos` is `prefix_hidden[:, pos]`. Carrying all positions
    (instead of just the last) is what lets ONE prefill serve every anchor.
    """

    task: str
    full_ids: torch.Tensor          # (total,) long, on device
    prompt_len: int
    valid_end: int
    fused: torch.Tensor             # (1, total, L*H)
    prefix_kv: object               # DynamicCache over the whole trajectory
    prefix_hidden: torch.Tensor      # (1, total, H), detached
    vispos: object
    pixel_values_videos: torch.Tensor   # kept so a per-anchor `VisionPositions`
    video_grid_thw: torch.Tensor        # can be re-bound to the cropped prefix
    n_anchors_seen: int = 0
    rollout_mode: str = "legacy-reference"
    stop_reason: str = "stored_trajectory"

    def bytes_held(self) -> int:
        """GPU bytes this trace keeps resident after pass 1 (KV + fused + hidden).

        Used for the memory report in `README_b1.md`; a trace's tensors are all
        that survives pass 1 (the rollout's own cache and logits are freed when
        `rollout` returns).
        """
        total = 0
        cache = self.prefix_kv
        for layer in getattr(cache, "layers", []):
            for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if t is not None:
                    total += t.numel() * t.element_size()
        for t in (self.fused, self.prefix_hidden):
            total += t.numel() * t.element_size()
        return total


def build_video_trace(target, head, proc, record, row, *, device, max_seq_len,
                      max_rounds: int, verify: str = "chain",
                      head_dtype=None, rollout_mode="online",
                      max_new_tokens=common.GEN_NEW_TOKENS):
    """Build an online trace, or explicitly replay the historical reference.

    In online mode only stored prompt IDs are checked; the continuation is
    ignored. opd_verified_rollout owns target verification and state commits.
    The following prefill notes apply to legacy-reference mode only.

    One video's single cached target prefill + chain rollout (pass 1).

    Returns `(trace, anchors)`. The prefill is
    `opd_fullkl.target_prefill_shared` -- the decode-aligned cached path.
    `common.target_context` is NEVER called: it uses `use_cache=False`, under
    which the text-position hidden states are independent of the video, which is
    the training/deployment disagreement Phase A4 removed. The rollout runs on
    the SAME fused context, so the head proposes from the context the trainer
    will later score it against.
    """
    p = row
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": str(p["question"])}]}],
        add_generation_prompt=True, tokenize=False)
    ins = proc(text=[text], videos=[p["cache_path"]], return_tensors="pt",
               fps=float(p["fps"]))
    prompt_ids = ins["input_ids"]
    stored = int(record["prompt_len"])
    if int(prompt_ids.shape[1]) != stored:
        raise ValueError(
            f"{record['task']}: encoded prompt is {int(prompt_ids.shape[1])} "
            f"tokens but the trajectory record says {stored}; the prompt string "
            "or the processor drifted from the pool the trajectory came from")
    if not torch.equal(prompt_ids[0].cpu().long(),
                       record["input_ids"][:stored].cpu().long()):
        raise ValueError(f"{record['task']}: prompt token IDs differ from record")
    if rollout_mode == "online":
        if verify != "chain":
            raise ValueError("strict online OPD currently supports chain verification only")
        from historical import opd_verified_rollout
        state, anchors = opd_verified_rollout.rollout(
            target, head, prompt_ids.to(device),
            pixel_values_videos=ins["pixel_values_videos"].to(device),
            video_grid_thw=ins["video_grid_thw"].to(device),
            max_seq_len=max_seq_len, max_new_tokens=max_new_tokens,
            max_rounds=max_rounds, head_dtype=head_dtype,
            draft_fn=draft_block_logits)
        return VideoTrace(task=str(record["task"]), **state,
                          rollout_mode="online", n_anchors_seen=len(anchors)), anchors
    if rollout_mode != "legacy-reference":
        raise ValueError(f"unknown rollout mode {rollout_mode!r}")
    full_ids = record["input_ids"].to(device).long()
    total = int(full_ids.shape[0])
    if total > int(max_seq_len):
        # A guard, never a truncation: the trainer's anchors, the head's context
        # and the teacher's cache all key off ABSOLUTE stream positions, so a
        # shortened trajectory would score anchors against a context that does
        # not exist. `opd_fullkl.target_prefill_shared` raises for the same
        # reason; raising a different cap is a data decision, not something a
        # trainer may make silently.
        raise ValueError(
            f"{record['task']}: trajectory is {total} tokens > --max-seq-len "
            f"{max_seq_len}; raise the limit explicitly if this row is intended")
    if total < stored + K_OFFSETS + 1:
        raise ValueError(f"{record['task']}: trajectory {total} has no room for a "
                         f"{K_OFFSETS}-offset block after prompt {stored}")
    pv = ins["pixel_values_videos"].to(device)
    vg = ins["video_grid_thw"].to(device)
    full = full_ids[None]
    vispos = opd_fullkl.prefix_vision_positions(target, full,
                                               pixel_values_videos=pv,
                                               video_grid_thw=vg)
    fused, prefix_kv = opd_fullkl.target_prefill_shared(
        target, full, pixel_values_videos=pv, video_grid_thw=vg,
        max_seq_len=max_seq_len, vispos=vispos)
    # The per-position final hidden states the teacher's offset-1 rows need. One
    # extra forward per video (the prefill itself is the expensive part; this is
    # the same call with the hidden-state hook exposed), under no_grad, and it is
    # what keeps the shared prefix cache uncropped while every anchor still gets
    # an exact anchor-position teacher row.
    prefix_hidden = _prefill_last_hidden(target, full, vispos)
    assert_prefill_agrees(target, prefix_hidden, prefix_kv)
    # Qwen3-VL keeps `eos_token_id` on the nested `text_config`; the outer config
    # returns None for it. Walk the chain the same way the rest of the project
    # does (eval_video_opd.py reads the tokenizer), so the EOS guard is evaluated
    # rather than skipped. A single scalar is enforced because the guard compares
    # against one id; a list would silently pick the wrong member.
    eos = getattr(target.config, "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(target.config, "text_config", None), "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    if eos is None:
        raise ValueError("no eos_token_id on target.config or target.config."
                         "text_config; the EOS supervision guard cannot be evaluated")
    valid_end = trajectory_valid_end(full_ids, int(eos), start=stored)
    trace = VideoTrace(task=str(record["task"]), full_ids=full_ids,
                       prompt_len=stored, valid_end=valid_end, fused=fused,
                       prefix_kv=prefix_kv, prefix_hidden=prefix_hidden,
                       vispos=vispos, pixel_values_videos=pv, video_grid_thw=vg)
    with torch.no_grad():
        anchors = opd_online_rollout.rollout(
            head, target, fused, full_ids, prompt_len=stored, verify=verify,
            max_rounds=max_rounds, head_dtype=head_dtype)
    trace.n_anchors_seen = len(anchors)
    return trace, anchors


def _prefill_last_hidden(target, full, vispos):
    """`(1, total, H)` last hidden states of one cached prefill, detached.

    The same forward `opd_fullkl.target_prefill_shared` performs (cached path,
    M-RoPE positions from `vispos`), kept separate because that function returns
    only the anchor row on the cache. `use_cache=True` is required: it is the
    decode-aligned branch, and the whole point of this trainer is that training
    and deployment read the same features.

    CHEAP SELF-CHECK, because this is the one deliberate duplicate forward. The
    prefill already ran once inside `target_prefill_shared`, which left its own
    last hidden state on the cache under `PREFIX_LAST_HIDDEN`; this function
    compares its own final hidden state against that row and raises on any
    disagreement. That catches a drifting position/attention path here, at the
    price of one `(1, H)` comparison, instead of surfacing much later as a
    teacher whose offset-1 rows are silently on a different distribution.
    """
    from transformers import DynamicCache
    with torch.no_grad():
        out = target.model(full, past_key_values=DynamicCache(), use_cache=True,
                           **vispos.prefill_inputs(full, None))
    hidden = out.last_hidden_state.detach()
    return hidden


def assert_prefill_agrees(target, hidden, prefix_kv):
    """The duplicate prefill's last row must equal `target_prefill_shared`'s.

    Both forward the same tokens through the same cached path; a mismatch means
    the two calls are not the same function any more (a mutated rope delta, a
    different attention implementation, a changed position tensor), which would
    make every offset-1 teacher row this trainer attaches wrong.
    """
    reference = getattr(prefix_kv, opd_fullkl.PREFIX_LAST_HIDDEN, None)
    if reference is None:
        raise RuntimeError("the shared prefix cache carries no "
                           f"{opd_fullkl.PREFIX_LAST_HIDDEN!r} row to check")
    got = hidden[:, -1:, :].float()
    want = reference.float()
    diff = float((got - want).abs().max())
    if diff != 0.0:
        raise RuntimeError(
            f"the duplicate prefill disagrees with target_prefill_shared on the "
            f"last hidden row (max|d| = {diff:.3e}); the two paths are not "
            "identical, so the per-anchor offset-1 teacher rows cannot be "
            "trusted. Re-run with the prefill that produced the cache.")


def attach_cropped_prefix(trace: VideoTrace, anchor_pos: int):
    """A per-anchor prefix cache ending AT the anchor.

    `opd_fullkl.teacher_block_logits` requires a cache whose last row is the
    anchor's own teacher row, because the draft-induced prefix for offset 1 is
    the anchor prefix itself. The shared cache spans the whole trajectory, so
    this builds a second `DynamicCache` holding its first `anchor_pos + 1`
    key/value rows and attaches the prefill's hidden state at `anchor_pos`. No
    re-prefill happens; the source cache is not mutated (the caller asserts its
    length is unchanged).

    COST, because it is part of the memory budget: `torch.cat` on a non-contiguous
    slice COPIES, so each call materialises ~`2 * 36 * (anchor_pos+1) * 8 * 128 *
    2 B` of keys+values -- about 12 MiB at a 800-token anchor, ~25 MiB at the
    1982-token maximum. It is freed with the anchor's microbatch, so it does not
    accumulate across anchors, but it is not a free view either.

    A `DynamicCache()` constructed with no arguments has ZERO layers -- it is the
    `layer_class_to_replicate` form, which appends a layer on the first `update`,
    and its `get_seq_length()` therefore returns 0. transformers 4.57's
    `DynamicCache` exposes one public way to pre-populate it: the `ddp_cache_data`
    iterable of `(key_states, value_states)` pairs, one per layer. That is used
    here, so the cache is built through the public constructor rather than by
    poking `layers` in afterwards. `cache_position` is copied per layer when the
    source carries one, because that attribute is not part of `ddp_cache_data` and
    the forward path may read it.
    """
    from transformers import DynamicCache
    ap = int(anchor_pos)
    src = trace.prefix_kv
    src_layers = list(getattr(src, "layers", []))
    if not src_layers:
        raise RuntimeError("the shared prefix cache has no layers; the prefill "
                           "did not populate it")
    data = []
    for idx, layer in enumerate(src_layers):
        if layer.keys is None or layer.values is None:
            raise RuntimeError(f"shared prefix cache layer {idx} has no keys/values")
        if int(layer.keys.shape[-2]) <= ap:
            raise RuntimeError(f"shared prefix cache layer {idx} holds "
                               f"{int(layer.keys.shape[-2])} rows, shorter than "
                               f"the anchor + 1 = {ap + 1}")
        data.append((layer.keys[..., :ap + 1, :], layer.values[..., :ap + 1, :]))
    cropped = DynamicCache(ddp_cache_data=data)
    if len(cropped.layers) != len(src_layers):
        raise RuntimeError(f"cropped cache has {len(cropped.layers)} layers but "
                           f"the prefill's had {len(src_layers)}")
    for new, layer in zip(cropped.layers, src_layers):
        pos = getattr(layer, "cache_position", None)
        if pos is not None:
            new.cache_position = pos[:ap + 1]
    if int(cropped.get_seq_length()) != ap + 1:
        raise RuntimeError(f"cropped prefix cache has {cropped.get_seq_length()} "
                           f"rows, expected anchor_pos+1 = {ap + 1}")
    setattr(cropped, opd_fullkl.PREFIX_LEN, ap + 1)
    setattr(cropped, opd_fullkl.PREFIX_LAST_HIDDEN,
            trace.prefix_hidden[:, ap:ap + 1, :])
    return cropped


def draft_block_logits(head, target, fused, *, anchor_pos, root_token,
                       prompt_len, head_dtype, cache=None):
    """`opd_fullkl.draft_block_logits` with the dtype contract made explicit.

    WHY this wrapper and not a call to the module directly. That function builds
    its noise embeddings with `target.get_input_embeddings()(ids)` -- the TARGET's
    embedding table -- and feeds them, together with `fused`, into the HEAD's
    own Linears. It therefore only works when all three share a dtype.
    `opd_train_video.py` gets that for free because both models are bf16. This
    trainer exposes `--teacher-dtype float32` (the target, which IS the teacher)
    alongside `--head-dtype bfloat16`, and in that combination the target's
    embedding is fp32 while the head is bf16 -- a dtype ERROR, not a silent
    precision change. The head is a separate module from the target, so this
    wrapper reads the target's embedding, casts it and `fused` into the head's
    dtype, and calls the module with a target stub whose only used methods are
    `get_input_embeddings` and `lm_head`.

    What this wrapper ALSO does (2026-10-10 fix): cast the head's hidden rows
    into `lm_head`'s weight dtype before projecting. The project's `lm_head`
    belongs to the TARGET, so with `--teacher-dtype float32` it is fp32 while the
    head's hidden is bf16 -- `lm_head(bf16_hidden)` is a dtype ERROR. The KL is
    still computed in fp32 (`opd_fullkl.full_kl_forward` casts internally), so
    this only makes the student-side projection legal, it does not change the
    fp32 KL contract.
    """
    fused = fused if fused.dtype == head_dtype else fused.to(head_dtype)
    embed = target.get_input_embeddings()

    class _HeadDtypeTarget:
        """`lm_head` and `get_input_embeddings` are all the module calls."""

        def __init__(self, real, embedding):
            self._lm = real.lm_head
            self._lm_dtype = real.lm_head.weight.dtype
            self._embedding = embedding

        def lm_head(self, hidden):
            if hidden.dtype != self._lm_dtype:
                hidden = hidden.to(self._lm_dtype)
            return self._lm(hidden)

        def get_input_embeddings(self):
            return self._embedding

    def _embed(ids):
        out = embed(ids)
        return out if out.dtype == head_dtype else out.to(head_dtype)

    return opd_fullkl.draft_block_logits(
        head, _HeadDtypeTarget(target, _embed), fused, anchor_pos=anchor_pos,
        root_token=int(root_token), prompt_len=prompt_len, cache=cache)


def score_microbatch(head, target, trace: VideoTrace, plan: AnchorPlan, *,
                     gamma: float, denoms, head_dtype, teacher_dtype, chunk=8,
                     rejected_kl: str = opd_fullkl.REJECTED_KL_FULL_REVERSE,
                     keep_kl_variants: bool = False):
    """One anchor's scaled loss and its logging statistics.

    `denoms` is `(n_acc_total, z_total, acc_present, rej_present)` frozen by
    `plan_update_denominators`. The teacher runs under `no_grad` (its own
    decorator plus this scope) and never reaches the graph; the student runs with
    grad. The returned loss is a tensor unless neither stream exists over the
    update, in which case it is the float `0.0` and the caller must not call
    `backward()` on it (nothing in that update could receive a gradient).

    `rejected_kl` selects the rejected stream's objective: the paper's
    full-vocabulary reverse KL (default, historical) or the official k3
    single-sample estimator, which needs the proposal tokens the rollout carried
    (`plan.prop`). Nothing else about the objective changes with it.
    """
    n_acc_total, z_total, acc_present, rej_present = denoms
    pos = int(plan.pos)
    k = K_OFFSETS

    # ---- teacher: a fork of the shared prefix, cropped at the anchor
    # `teacher_block_logits` binds `vispos` to exactly the prefix it scores, so a
    # `VisionPositions` bound to the WHOLE trajectory is rejected (its rope index
    # table is longer than the prefix). Re-binding to `full_ids[:pos+1]` costs one
    # cheap `get_rope_index` call and no model forward; that the cropped prefix's
    # rope indices equal the full sequence's first `pos+1` columns is exactly what
    # `opd_fullkl.teacher_block_logits` asserts internally.
    anchor_vispos = opd_fullkl.prefix_vision_positions(
        target, trace.full_ids[:pos + 1][None],
        pixel_values_videos=trace.pixel_values_videos,
        video_grid_thw=trace.video_grid_thw)
    before = int(trace.prefix_kv.get_seq_length())
    anchor_kv = attach_cropped_prefix(trace, pos)
    with torch.no_grad():
        p_logits = opd_fullkl.teacher_block_logits(
            target, anchor_vispos, anchor_kv, anchor_pos=pos, prop=plan.prop,
            chunk=1 if trace.rollout_mode == "online" else chunk)
    after = int(trace.prefix_kv.get_seq_length())
    if after != before:
        raise RuntimeError(f"the shared prefix cache moved from {before} to "
                           f"{after} rows while scoring anchor {pos}; "
                           "teacher_block_logits must only fork it")

    # ---- student: a fresh head cache over `fused[:, :pos]`
    q_logits = draft_block_logits(
        head, target, trace.fused[:, :pos], anchor_pos=pos,
        root_token=int(trace.full_ids[pos]), prompt_len=trace.prompt_len,
        head_dtype=head_dtype)

    if int(p_logits.shape[0]) != k or int(q_logits.shape[0]) != k:
        raise RuntimeError(f"anchor {pos}: expected {k} rows per side, got "
                           f"teacher {tuple(p_logits.shape)} student "
                           f"{tuple(q_logits.shape)}")

    if trace.rollout_mode == "online":
        from historical import opd_verified_rollout
        opd_verified_rollout.assert_replay(plan, p_logits, q_logits)

    idx_acc = plan.offsets_acc
    idx_rej = plan.offsets_rej
    acc_kl = (opd_fullkl.full_kl_forward(p_logits[idx_acc - 1], q_logits[idx_acc - 1])
              if idx_acc.numel() else q_logits.new_zeros(0))
    k3_diag: dict[str, float] = {}
    if idx_rej.numel():
        p_rej = p_logits[idx_rej - 1]
        q_rej = q_logits[idx_rej - 1]
        # The official sampled estimator scores the trajectory's OWN token at each
        # rejected offset -- `plan.prop[k-1]` is offset k's draft token.
        tok = torch.tensor([int(plan.prop[int(k) - 1]) for k in idx_rej],
                           dtype=torch.long, device=q_logits.device)
        k3 = None
        if keep_kl_variants or rejected_kl == opd_fullkl.REJECTED_KL_K3:
            k3 = opd_fullkl.k3_reverse_kl(q_rej, p_rej, tok)
            logq_t, logp_t = opd_fullkl.sampled_logprobs(q_rej, p_rej, tok)
            raw = (logp_t - logq_t).detach()
            k3_diag = {
                "rej_log_ratio_absmax": float(raw.abs().max()),
                "rej_log_ratio_clamp_frac":
                    float((raw.abs() >= opd_fullkl.K3_LOG_RATIO_CLAMP).float().mean()),
                "rej_kld_clamp_frac":
                    float((k3.detach().abs() >= opd_fullkl.K3_KLD_CLAMP).float().mean()),
                "rej_k3_mean": float(k3.detach().mean()),
                "rej_k3_max": float(k3.detach().max()),
                "rej_k3_sq_mean": float((k3.detach() ** 2).mean()),
            }
        if rejected_kl == opd_fullkl.REJECTED_KL_K3:
            rej_kl = k3
        else:
            rej_kl = opd_fullkl.full_kl_reverse(q_rej, p_rej)
        if keep_kl_variants:
            # Diagnostic-only second value on the SAME teacher/student rows: the
            # scale comparison A3.2 asks for, without a second forward pass.
            full = opd_fullkl.full_kl_reverse(q_rej, p_rej).detach()
            k3_diag.update({
                "rej_full_mean": float(full.mean()),
                "rej_full_max": float(full.max()),
                "rej_full_sq_mean": float((full ** 2).mean()),
                "rej_full_abs_ge_kld_clamp_frac":
                    float((full.abs() >= opd_fullkl.K3_KLD_CLAMP).float().mean()),
            })
    else:
        rej_kl = q_logits.new_zeros(0)
    rej_w = build_offset_weights(k, gamma, idx_rej, device=q_logits.device)

    loss = microbatch_scaled_loss(
        acc_kl, rej_kl, rej_w, n_acc_total=int(n_acc_total),
        z_total=float(z_total), acc_present=bool(acc_present),
        rej_present=bool(rej_present))
    stats = {
        "n_acc": int(idx_acc.numel()), "n_rej": int(idx_rej.numel()),
        "sum_w_rej": float(plan.sum_w_rej),
        "acc_kl_mean": float(acc_kl.mean()) if acc_kl.numel() else 0.0,
        "acc_kl_max": float(acc_kl.max()) if acc_kl.numel() else 0.0,
        "rej_kl_mean": float(rej_kl.mean()) if rej_kl.numel() else 0.0,
        "rej_kl_max": float(rej_kl.max()) if rej_kl.numel() else 0.0,
        "rej_kl_sq_mean": (float((rej_kl ** 2).mean()) if rej_kl.numel() else 0.0),
        **k3_diag,
    }
    return loss, stats


def clip_and_step(optimizer, params, clip: float, master=None):
    """Clip, step and zero; returns `(pre_clip_norm, applied_scale)`.

    `get_total_norm` is the pre-clip total L2 norm of the gradients -- the number
    the plan's stop conditions are written against. The applied scale is
    `min(1, clip / norm)`, so `1 - scale` is the fraction of the gradient that
    was removed. `clip <= 0` disables clipping while still reporting the norm,
    which keeps a diagnostic run possible without a second code path. A
    non-finite norm raises instead of silently poisoning the weights.
    """
    if master is not None:
        # fp32 master path: sync the bf16 grads into fp32, read the PRE-CLIP norm
        # there, and let `MasterWeightAdamW.step()` clip in fp32 and write the
        # result back. Clipping bf16 grads and clipping their fp32 copies are
        # different numbers, so it has to happen on one side consistently -- the
        # master side, which is also where the optimizer state lives.
        master.sync_grads()
        norm = master.grad_norm()
        if not math.isfinite(norm):
            raise RuntimeError(f"gradient norm is not finite ({norm}); refusing "
                               "to step, the update is numerically broken")
        scale = min(1.0, float(clip) / (norm + 1e-12)) if clip and clip > 0 else 1.0
        master.step()
        master.zero_grad(set_to_none=True)
        return norm, scale

    params = [p for p in params if p.grad is not None]
    if not params:
        raise RuntimeError("no parameter received a gradient this update; the "
                           "objective is not connected to the head")
    norm = float(torch.nn.utils.get_total_norm([p.grad for p in params],
                                               foreach=False))
    if not math.isfinite(norm):
        raise RuntimeError(f"gradient norm is not finite ({norm}); refusing to "
                           "step, the update is numerically broken")
    if clip and clip > 0:
        scale = min(1.0, float(clip) / (norm + 1e-12))
        if scale < 1.0:
            for p in params:
                p.grad.mul_(scale)
    else:
        scale = 1.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm, scale


# ==========================================================================
# CLI
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    """The entry point's arguments, in the style of `opd_train_video.py`.

    `--updates` is the primary name and `--steps` the aliases' familiar spelling;
    both write the same destination, so passing both is an error rather than a
    silent precedence rule.
    """
    ap = argparse.ArgumentParser(
        description="C2: paper-faithful full-KL Draft-OPD trainer for the video "
                    "block head")
    ap.add_argument("--records", required=True,
                    help="pool .pt holding each task's committed target "
                         "trajectory (input_ids) and prompt_len; online mode uses only task/prompt identity")
    ap.add_argument("--pool", required=True,
                    help="pool parquet carrying cache_path/fps/question/split")
    ap.add_argument("--holdout-parquet", default=None,
                    help="extra rows to exclude; the split=='train' filter is "
                         "the primary guard")
    ap.add_argument("--init-from", required=True,
                    help="warm start, e.g. "
                         "$R/checkpoints/head_video439_aligned/step1000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", "--steps", dest="updates", type=int, default=50,
                    help="optimizer updates to run (alias: --steps)")
    ap.add_argument("--anchors-per-update", type=int, default=24,
                    help="supervised anchors per optimizer update; the update's "
                         "global denominators are computed over exactly these")
    ap.add_argument("--anchors-per-video", type=int, default=0,
                    help="cap anchors taken from one video per update (0 = no "
                         "cap; the per-update budget still applies)")
    ap.add_argument("--max-rounds", type=int, default=0,
                    help="cap the chain rollout's rounds per video (0 = run to "
                         "the end of the trajectory)")
    ap.add_argument("--rollout-mode", choices=["online", "legacy-reference"],
                    default="online", help="current target verifies and commits each round; "
                    "legacy-reference explicitly reproduces old cached-continuation behaviour")
    ap.add_argument("--rollout-max-new-tokens", type=int, default=common.GEN_NEW_TOKENS,
                    help="online generation budget including initial root and EOS")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.8,
                    help="w_k = gamma**(k-1) for the rejected stream")
    ap.add_argument("--rejected-kl",
                    choices=[opd_fullkl.REJECTED_KL_FULL_REVERSE,
                             opd_fullkl.REJECTED_KL_K3],
                    default=opd_fullkl.REJECTED_KL_FULL_REVERSE,
                    help="rejected-stream objective: 'full_reverse' = paper "
                         "Eq. 11 full-vocabulary reverse KL (default, "
                         "bit-identical to every earlier run); 'k3' = the official "
                         "sampled estimator of run_qwen_gsm8k.sh (d1476f5), "
                         "computed on the trajectory's own draft tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--no-final-save", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--teacher-dtype", default="float32",
                    choices=["float32", "bfloat16"],
                    help="dtype the TARGET is loaded in (the target IS the "
                         "teacher; this is not autocast)")
    ap.add_argument("--head-dtype", default="bfloat16",
                    choices=["bfloat16", "float32"],
                    help="dtype the DRAFT HEAD is loaded and trained in")
    ap.add_argument("--max-seq-len", type=int, default=common.MAX_SEQ_LEN,
                    help="guard on the trajectory length; an over-long row "
                         "raises (see the module docstring)")
    ap.add_argument("--master-weights", default="none",
                    choices=["none", "cpu", "gpu"],
                    help="none = historical bf16 AdamW (default, bit-identical to "
                         "every earlier OPD run); cpu/gpu = fp32 master params + "
                         "fp32 Adam state, stepped in fp32 and written back into "
                         "the bf16 forward params. Required: diag_update_precision"
                         ".py measured that without it the realized update at "
                         "lr 3e-5 keeps only 0.223 of the intended norm and 31% of "
                         "tensors get an exactly-zero update")
    ap.add_argument("--clip", type=float, default=common.GRAD_CLIP,
                    help="gradient-norm clip applied to the update's accumulated "
                         "gradients")
    ap.add_argument("--deterministic", action="store_true",
                    help="CUBLAS_WORKSPACE_CONFIG + "
                         "use_deterministic_algorithms(warn_only=True), the form "
                         "the existing trainers use")
    ap.add_argument("--skip-empty-updates", action="store_true",
                    help="do not count an update with zero supervised offsets "
                         "against --updates; it still logs and never steps")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU-only preflight: validate the split/trajectory "
                         "contract and print the byte budget, then exit without "
                         "loading a model or touching CUDA")
    return ap


def _reject_duplicate_update_flags(argv) -> None:
    """`--updates` and `--steps` are aliases; passing BOTH is a user error.

    argparse writes both to the same destination and silently keeps the last one,
    which is exactly the kind of quiet precedence rule a schedule-length argument
    must not have.
    """
    if argv is None:
        argv = sys.argv[1:]
    seen = [flag for flag in ("--updates", "--steps")
            if any(tok == flag or tok.startswith(flag + "=") for tok in argv)]
    if len(seen) > 1:
        raise SystemExit(f"error: {seen[0]} and {seen[1]} are aliases for the "
                         "same setting; pass exactly one of them")


def parse_args(argv=None) -> argparse.Namespace:
    _reject_duplicate_update_flags(argv)
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.rollout_max_new_tokens < 1 or a.max_rounds < 0:
        ap.error("rollout token budget must be positive and max-rounds nonnegative")
    if a.updates < 1:
        ap.error(f"--updates must be >= 1, got {a.updates}")
    if a.anchors_per_update < 1:
        ap.error(f"--anchors-per-update must be >= 1, got {a.anchors_per_update}")
    if not 0.0 < a.gamma <= 1.0:
        ap.error(f"--gamma must be in (0, 1], got {a.gamma}")
    if a.max_seq_len < common.BLOCK_SIZE + 2:
        ap.error(f"--max-seq-len {a.max_seq_len} cannot hold a "
                 f"{K_OFFSETS}-offset block")
    return a


# ==========================================================================
# Main
# ==========================================================================

def dry_run(a) -> int:
    """CPU-only preflight: validate the data contract and report the budget.

    WHY this exists. The expensive failure modes of this entry point are data
    failures -- a prompt that no longer matches the stored `prompt_len`, a
    trajectory longer than `--max-seq-len`, a non-train row in the pool, a
    checkpoint that is not where `--init-from` says. Every one of those can be
    detected from the pool and the records alone, on a CPU, before a GPU-hour is
    spent. This mode loads NO model and touches no CUDA: it reads the same pool,
    the same records and the same split filter `main` uses, prints the byte
    budget from the observed lengths, and exits.
    """
    import pandas as pd
    records = [f"{a.records}", f"{a.pool}", f"{a.init_from}"]
    if a.holdout_parquet:
        records.append(a.holdout_parquet)
    for path in records:
        if not os.path.exists(path):
            raise FileNotFoundError(f"--dry-run input does not exist: {path}")
    blob = torch.load(a.records, weights_only=False, map_location="cpu")
    recs_all = blob["records"]
    pool = pd.read_parquet(a.pool)
    key = "task" if "task" in pool.columns else "video_name"
    split_of = dict(zip(pool[key].astype(str), pool["split"].astype(str)))
    pool = pool.set_index(key)
    excluded = set()
    if a.holdout_parquet:
        hold = pd.read_parquet(a.holdout_parquet)
        hcol = "task" if "task" in hold.columns else "video_name"
        excluded = set(hold[hcol].astype(str))
    recs = [r for r in recs_all
            if split_of.get(str(r["task"])) == "train"
            and str(r["task"]) not in excluded]
    n_pool_train = int((pool["split"] == "train").sum())
    n_excluded_in_train = len(excluded & {str(t) for t in pool.index
                                          if split_of.get(str(t)) == "train"})
    if len(recs) != n_pool_train - n_excluded_in_train:
        raise AssertionError(f"train records {len(recs)} != pool train rows "
                             f"{n_pool_train} - {n_excluded_in_train} excluded")
    if not recs:
        raise AssertionError("no train records survived the split filter")
    missing = [str(r["task"]) for r in recs if str(r["task"]) not in pool.index]
    if missing:
        raise KeyError(f"{len(missing)} records are absent from the pool, e.g. "
                       f"{missing[:3]}; their cache_path/fps are unknown")
    # The per-row contract main() enforces at load time, checked here so it fails
    # before the model is loaded.
    for r in recs:
        stored = int(r["prompt_len"])
        total = int(r["input_ids"].shape[0])
        if a.rollout_mode == "online":
            if stored >= a.max_seq_len:
                raise ValueError(f"{r['task']}: prompt exceeds online sequence budget")
            continue
        if total > a.max_seq_len:
            raise ValueError(f"{r['task']}: trajectory {total} > --max-seq-len "
                             f"{a.max_seq_len}")
        if total < stored + K_OFFSETS + 1:
            raise ValueError(f"{r['task']}: trajectory {total} has no room for a "
                             f"{K_OFFSETS}-offset block after prompt {stored}")
    L = np.array([min(a.max_seq_len, int(r["prompt_len"]) + a.rollout_max_new_tokens)
                  if a.rollout_mode == "online" else int(r["input_ids"].shape[0])
                  for r in recs])
    # Same byte model as README_b1.md: 2 * 36 layers * (L+1) * 8 kv heads * 128
    # head_dim * dtype_bytes for the prefix KV, plus the fused context and the
    # per-position last hidden state.
    per_video_kv = 2 * 36 * (L + 1) * 8 * 128 * {"float32": 4, "bfloat16": 2}[
        a.teacher_dtype]
    fused = (L + 1) * 5 * 4096 * 2
    hidden = (L + 1) * 4096 * 2
    per_video = per_video_kv + fused + hidden
    worst = np.sort(per_video)[-a.anchors_per_update:]
    print(f"[dry-run] records {len(recs)} of {len(recs_all)} (pool train "
          f"{n_pool_train}, excluded {n_excluded_in_train})")
    print(f"[dry-run] trajectory L: min {int(L.min())} median "
          f"{int(np.median(L))} max {int(L.max())}  (--max-seq-len "
          f"{a.max_seq_len})")
    print(f"[dry-run] rollout={a.rollout_mode}; online anchor availability depends on current target EOS")
    print(f"[dry-run] hold up to {a.anchors_per_update} videos' pass-1 state "
          f"(worst case): {worst.sum()/2**30:.2f} GiB "
          f"(--anchors-per-video {a.anchors_per_video})")
    print(f"[dry-run] target weights ~{8.19*{'float32': 4, 'bfloat16': 2}[a.teacher_dtype]:.1f} GiB "
          f"({a.teacher_dtype}) + head 1.95 + grads 3.91 + Adam 7.81 GiB")
    print(f"[dry-run] OK; nothing was loaded and no CUDA call was made")
    return 0


def main(argv=None):
    """New callers cannot invoke historical training through an imported main."""
    from opd_official_kl.cli import main as gate
    return gate(["--modality", "video", *(sys.argv[1:] if argv is None else argv)])


def _historical_main_disabled(*args, **kwargs):
    """The original training loop is archived under legacy/pre20261016."""
    raise RuntimeError("Use legacy/replay.py for explicitly acknowledged historical replay")


if __name__ == "__main__":
    sys.exit(main())
