#!/usr/bin/env python3
"""Draft-OPD, the faithful objective: full-vocabulary KL on the draft-induced prefix.

WHY this module exists. The video OPD trainer (`opd_loss_video.py` /
`opd_train_video.py`) supervises the draft head with a *stored* top-k teacher row
along the TARGET's own greedy trajectory. Measured in `proto_fullkl_b0.py`, that
is not Draft-OPD (arXiv 2605.29343, Eq. 8-13) in three ways:

  1. the teacher row is truncated to 64 stored ids and renormalised, so the loss
     never sees the ~151k tokens outside the support;
  2. the rejected stream is one offset long (`opd_loss_video.is_first_rej`), and
     with the image-side `--reject-target topk` it is empty by construction;
  3. the stored row is the target's logits along the TARGET's own greedy
     trajectory, so a rejected offset is scored on a context the drafter never
     produced.

This module is the additive replacement: a self-contained objective plus the two
scoring paths it needs, with no change to any existing file.

THE OBJECTIVE (verified ground truth, implemented literally, not re-derived).
For one draft block at anchor position `p` with `K` valid offsets (1-based):
`I_acc = {k <= n_acc}`, `I_rej = {n_acc < k <= K}`, `w_k = gamma**(k-1)`:

    L_acc = (1/|I_acc|) * sum_{k in I_acc} KL( p_k || q_k )      # plain mean, no weights
    L_rej = (1/Z) * sum_{k in I_rej} w_k * KL( q_k || p_k )      # Z = sum_{k in I_rej} w_k
    L     = (lambda_acc * L_acc + lambda_rej * L_rej) / (lambda_acc + lambda_rej)

with `p_k` the target (teacher) and `q_k` the draft (student) full-vocabulary
softmaxes, ``gamma = 0.8`` and both lambdas 1.0. Three points of this are easy to
get wrong and are called out because they differ from the prototype:

  * the accepted stream carries **no position weight** at all -- it is a plain
    mean over the accepted offsets, unlike `proto_fullkl_b0.weighted_stream_mean`
    which applied `w_k` to both streams (that was a pre-verification guess);
  * the rejected stream is the **whole** rejected suffix, not just its first
    offset, so a single rejected token can no longer swallow the term;
  * the two streams are renormalised **only over the streams that exist**. An
    empty stream is dropped and the remaining lambda is renormalised to 1, so an
    all-accepted anchor has `L == L_acc` rather than `L_acc / 2`. Only when BOTH
    streams are empty does the block contribute exactly zero (and the caller can
    see that in the returned counts).

Always in fp32: `log_softmax` and the KL are computed on `float()` copies of the
logits regardless of the input dtype, because the objective is a difference of
two nearly-equal quantities on a 151936-way support.

HOW the teacher is scored (and why it is cheap). One cached target prefill over
`ids` produces BOTH the head's fused context and the teacher's prefix KV
(`target_prefill_shared`, the single interface SFT and OPD share). A block is
then scored by pushing the drafter's own proposal through a *fork* of that cache
(`teacher_block_logits`), which reproduces `ids[:p+1] + prop[:k-1]` at every
offset without re-prefilling the video and without mutating the shared cache.
`proto_fullkl_b0` measured the incremental path against one independent full
forward at `max|dlogit| = 0.375` in bf16 (3.4e-05 in fp32, its own chunking
control at 0.375) -- that residue is bf16 GEMM sensitivity to the number of rows
in a batch, not a wrong prefix or a wrong position.

Contracts that are asserted rather than assumed:
  * `prefix_kv.get_seq_length() == anchor_pos + 1` -- the cache must end at the
    anchor, since its last row is the teacher row for offset 1;
  * the `fused` context handed to the head spans exactly the stream positions the
    head's own KV would have held (`anchor_pos - cache_len` rows), and the block's
    positions follow `opd_online_rollout.rollout`'s `stream` convention.

Nothing here touches the disk, and nothing runs on import: the module is
importable on a CPU-only box, and every GPU call happens inside a function.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# `common` also puts GLANCE_repo and the dflash package on sys.path, which is
# what makes the two imports below resolve from anywhere.
import common  # noqa: E402
from glance.head import context_feature  # noqa: E402
from glance.vl import VisionPositions  # noqa: E402
# The prefix-fork mechanism is imported, not re-derived: it is the part B0 has
# already validated against an independent full forward, and a second copy of it
# drifting from the first is exactly the bug this project keeps re-learning.
from proto_fullkl_b0 import append_through_fork, fork_cache  # noqa: E402

__all__ = [
    "FullKLLossConfig",
    "K3_KLD_CLAMP",
    "K3_LOG_RATIO_CLAMP",
    "REJECTED_KL_FULL_REVERSE",
    "REJECTED_KL_K3",
    "draft_block_logits",
    "full_kl_forward",
    "full_kl_reverse",
    "grad_cosine_and_norm_ratio",
    "k3_from_logprobs",
    "k3_reverse_kl",
    "opd_fullkl_loss",
    "prefix_last_hidden",
    "prefix_vision_positions",
    "reference_teacher_logits",
    "sampled_logprobs",
    "stream_split",
    "target_prefill_shared",
    "teacher_block_logits",
]

#: Attribute the prefill hangs on the cache it returns. The teacher needs the
#: anchor's own output row to score offset 1 (the draft-induced prefix for that
#: offset is `ids[:p+1]` itself); carrying it on the DynamicCache is what lets
#: `teacher_block_logits` stay a function of `(cache, anchor, proposal)` with no
#: second forward over the video prompt. See `target_prefill_shared`.
PREFIX_LAST_HIDDEN = "opd_prefix_last_hidden"
PREFIX_LEN = "opd_prefix_len"


# --------------------------------------------------------------------------
# The one shared target prefill
# --------------------------------------------------------------------------

def prefix_vision_positions(target, ids, *, pixel_values_videos=None,
                            video_grid_thw=None):
    """Bind a `VisionPositions` adapter to `ids` (rows 0..len(ids)-1).

    M-RoPE positions for a Qwen-family VLM come from the video grid, not from a
    counter, so every post-video position is the 1-D stream index shifted by the
    prefill delta. `pos(index)` reproduces the full sequence's positions for the
    appended tokens, and `pos.prefill_inputs(...)` carries the same claim into
    the prefill. Both are asserted against `get_rope_index` in
    `test_opd_fullkl.py`'s GPU smoke rather than assumed here.
    """
    if ids.ndim == 1:
        ids = ids[None]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"ids must be (1, n), got {tuple(ids.shape)}")
    pos = VisionPositions(pixel_values_videos=pixel_values_videos,
                          video_grid_thw=video_grid_thw)
    pos.bind(target, ids)
    return pos


def _check_bound(pos, n):
    """Reject a `VisionPositions` that was not bound to this exact prefix."""
    bound = getattr(pos, "_prefill_positions", None)
    if bound is None:
        raise ValueError("vispos has no prefill positions: build it with "
                         "prefix_vision_positions(target, ids, ...) or call "
                         "VisionPositions.bind(target, ids) on it first")
    if int(bound.shape[-1]) != int(n):
        raise ValueError(f"vispos is bound to {int(bound.shape[-1])} positions "
                         f"but the prefix has {int(n)} tokens")


@torch.no_grad()
def target_prefill_shared(target, ids, *, pixel_values_videos=None,
                          video_grid_thw=None, max_seq_len=None, layer_ids=None,
                          vispos=None):
    """ONE cached target prefill over `ids`; returns `(fused_hidden, prefix_kv)`.

    `fused_hidden` is `common.context_feature` over the model's hidden states at
    the kept layers, `(1, n, L*H)`, detached: the head's context. `prefix_kv` is
    the `DynamicCache` of that same forward: the teacher's prefix KV. Returning
    both from one forward is the project requirement that SFT and OPD share one
    target prefill interface -- a head trained through this function sees exactly
    the cache the teacher is scored against.

    Two attributes ride on the returned cache (`PREFIX_LAST_HIDDEN`, `PREFIX_LEN`)
    so the teacher can score offset 1 against the anchor's own prefill row
    without re-running the video. They are this module's convention, not
    transformers', and `teacher_block_logits` refuses a cache that does not carry
    the row.

    `max_seq_len` is a guard, never a truncation: cutting a video row inside its
    video-token block makes `get_rope_index` reject the grid, so an over-long row
    is a caller error and is raised. Defaults to `common.MAX_SEQ_LEN`.
    """
    if ids.ndim == 1:
        ids = ids[None]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"ids must be (1, n), got {tuple(ids.shape)}")
    n = int(ids.shape[1])
    cap = common.MAX_SEQ_LEN if max_seq_len is None else int(max_seq_len)
    if n > cap:
        raise ValueError(f"prefix of {n} tokens exceeds max_seq_len {cap}; "
                         "truncate in the caller, never inside a video block")

    pos = vispos if vispos is not None else prefix_vision_positions(
        target, ids, pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw)
    _check_bound(pos, n)

    # The inner model is called directly, exactly as the wrapper does, because
    # the wrapper projects its output through `lm_head` and would materialise an
    # (n, V) result this function has no use for; `common.target_context_cached`
    # shows the same cached prefill but throws the cache away.
    out = target.model(ids, past_key_values=DynamicCache(), use_cache=True,
                       output_hidden_states=True, **pos.prefill_inputs(ids, None))
    if out.hidden_states is None:
        raise RuntimeError("target.model returned no hidden states; the prefill "
                           "must be called with output_hidden_states=True")
    fused = context_feature(out.hidden_states,
                            common.KEPT_LAYERS if layer_ids is None else layer_ids)
    prefix_kv = out.past_key_values
    if prefix_kv is None:
        raise RuntimeError("target.model returned no past_key_values; use_cache "
                           "must be True for the prefix KV to exist")
    # The anchor's own output row: offset 1's teacher row is conditioned on
    # `ids` alone, so this is the prefill's last hidden state (post final norm,
    # i.e. what lm_head consumes), detached like every other teacher quantity.
    setattr(prefix_kv, PREFIX_LAST_HIDDEN, out.last_hidden_state[:, -1:, :].detach())
    setattr(prefix_kv, PREFIX_LEN, n)
    return fused.detach(), prefix_kv


def prefix_last_hidden(prefix_kv):
    """The anchor's own output row carried by `target_prefill_shared`."""
    hidden = getattr(prefix_kv, PREFIX_LAST_HIDDEN, None)
    if hidden is None:
        raise ValueError(
            f"cache has no {PREFIX_LAST_HIDDEN!r} attribute: it was not built by "
            "opd_fullkl.target_prefill_shared, so the teacher row for offset 1 "
            "(the distribution conditioned on the anchor prefix alone) is unknown")
    return hidden


# --------------------------------------------------------------------------
# Teacher logits on the draft-induced prefix
# --------------------------------------------------------------------------

@torch.no_grad()
def teacher_block_logits(target, vispos, prefix_kv, *, anchor_pos, prop, chunk=8):
    """Full-vocabulary teacher logits `(K, V)` fp32 for one chain block.

    Row `k-1` is the teacher distribution for offset `k` (the token at stream
    position `anchor_pos + k`), conditioned on `ids[:anchor_pos+1] + prop[:k-1]`
    -- the DRAFT-INDUCED prefix, i.e. the context the drafter actually created,
    not the target's own greedy path.

    Row 0 is the prefix's last hidden state: offset 1's draft-induced prefix is
    `ids` itself, so its teacher distribution is the prefill's own last output
    row. Rows 1..K-1 come from pushing `prop[:K-1]` through a FORK of
    `prefix_kv` in `chunk`-sized pieces, which is what keeps the peak bounded:
    `prop` may be a 15-token block, but the fork is never re-prefilled and no
    `(n, V)` tensor and no `(chunk, V)` tensor outlives its iteration. The
    returned tensor is the only logits that survive, and it is fp32 (the caller
    converts nothing).

    The shared cache is never mutated: `fork_cache` deep-copies every layer's
    keys/values, so appending to the fork cannot move the shared prefix or a
    sibling fork. `test_opd_fullkl.py` asserts the shared length is unchanged.

    `vispos` must be bound to the prefix (row k's positions are
    `vispos(anchor_pos + k)`), and the cache must end at the anchor, since its
    last row is the offset-1 teacher row.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    prop = [int(t) for t in prop]
    K = len(prop)
    if K < 1:
        raise ValueError("prop is empty; a block with no offsets has nothing to "
                         "score (skip the anchor in the caller)")

    prefix_len = int(prefix_kv.get_seq_length())
    if prefix_len != int(anchor_pos) + 1:
        raise ValueError(
            f"prefix KV has {prefix_len} rows but the anchor is at {anchor_pos}; "
            "the cache must hold exactly ids[:anchor_pos+1], because its last row "
            "is the teacher row for offset 1")
    _check_bound(vispos, prefix_len)
    recorded = getattr(prefix_kv, PREFIX_LEN, None)
    if recorded is not None and int(recorded) != prefix_len:
        raise ValueError(f"cache claims a {int(recorded)}-token prefill but "
                         f"holds {prefix_len} rows")
    hidden_pre = prefix_last_hidden(prefix_kv)
    device = hidden_pre.device

    # Offset 1: the prefix's own output distribution, through the same lm_head
    # the wrapper would use. One row, so this is bit-identical to the prefill's
    # `lm_head(h_pre[-1:])` in proto_fullkl_b0.score_anchor.
    rows = [target.lm_head(hidden_pre[0])[0].float()]     # (V,), not (1, V)

    fork = fork_cache(prefix_kv)
    for start in range(0, K - 1, chunk):
        # Only prop[:K-1] is ever appended: the K-th row is the distribution at
        # anchor_pos+K-1, which the (K-1)-th proposal's forward already produces.
        piece = prop[start:min(start + chunk, K - 1)]
        if not piece:
            break
        tokens = torch.tensor([piece], dtype=torch.long, device=device)
        # The first hidden row of this piece sits at stream position
        # anchor_pos+1+start and is therefore the distribution for the token at
        # anchor_pos+2+start -- offset start+2, the (start+1)-th row after the
        # prefix's own. Chunks are visited in order, so appending keeps row k-1
        # aligned with offset k.
        _, logits = append_through_fork(target, vispos, fork, tokens,
                                        int(anchor_pos) + 1 + start)
        # Rows arrive in offset order (start increases, and within a piece j
        # does too), so they are appended: the row for offset k is the k-th
        # appended row after the prefix's own.
        for j in range(len(piece)):
            rows.append(logits[j].float())
    if len(rows) != K:
        raise RuntimeError(f"scored {len(rows)} rows for a {K}-offset block; the "
                           "chunked append and the row accounting disagree")
    return torch.stack(rows, dim=0)


@torch.no_grad()
def reference_teacher_logits(target, prefix_ids, prop, *, pixel_values_videos=None,
                             video_grid_thw=None, image_grid_thw=None):
    """Verification path: ONE independent full forward over `prefix_ids + prop`.

    This is B0's G1 reference, deliberately re-derived the expensive way (fresh
    cache, the whole draft-induced text in one batch, its own single `lm_head`
    call), so `teacher_block_logits` can be checked against something that shares
    none of its code. It is not used by the objective.

    Returns `(logits, positions, hidden)`: row `k-1` of `logits` is the
    distribution for offset `k` (hidden rows `p..p+K-1`), `positions` are the
    M-RoPE indices `get_rope_index` assigns to the whole sequence, and `hidden`
    is the last hidden state for the sequence (so the caller can re-check the
    positions and the hidden-state scale).
    """
    prop = [int(t) for t in prop]
    if prefix_ids.ndim == 1:
        prefix_ids = prefix_ids[None]
    p = int(prefix_ids.shape[1]) - 1
    K = len(prop)
    if K < 1:
        raise ValueError("prop is empty")
    full = torch.cat([prefix_ids,
                      torch.tensor([prop], dtype=torch.long,
                                   device=prefix_ids.device)], dim=1)
    positions, _ = target.model.get_rope_index(full, image_grid_thw,
                                               video_grid_thw, None)
    kwargs = {}
    if image_grid_thw is not None:
        kwargs.update(pixel_values=None, image_grid_thw=image_grid_thw)
    if video_grid_thw is not None:
        kwargs.update(pixel_values_videos=pixel_values_videos,
                      video_grid_thw=video_grid_thw)
    out = target.model(full, position_ids=positions, past_key_values=DynamicCache(),
                       use_cache=True, **kwargs)
    hidden = out.last_hidden_state[0]
    return target.lm_head(hidden[p:p + K]).float(), positions, hidden


# --------------------------------------------------------------------------
# Draft logits for one anchor's block
# --------------------------------------------------------------------------

def draft_block_logits(head, target, fused, *, anchor_pos, root_token,
                       block_size=None, prompt_len=None, cache=None):
    """The draft's full-vocabulary block logits `(K, V)`, `K = block_size - 1`.

    Obtained exactly as `opd_online_rollout.rollout` obtains them -- a block of
    `[root_token, MASK, ...]` embeddings is run through the head and projected
    with `target.lm_head(hidden[:, 1-block:, :])`, i.e. the LAST `block-1` rows --
    because `q` is only meaningful as the distribution the rollout actually
    proposed from. `head.compute_logits` would be a different call than the one
    the rollout made and could not be asserted equal to `prop`.

    Position convention (`rollout`'s `stream`): the block's positions are the
    absolute stream indices `[cache_len, anchor_pos + block)`, the same indices
    the head's context rows carry, so `prompt_len` shifts nothing (it is accepted
    only for a sanity check that the anchor is past the prompt).

    Two ways to supply the head's context, both mirroring the rollout:
      * `cache=None`: a fresh cache, so `fused` must hold the context rows for
        stream positions `0..anchor_pos-1`, i.e. `fused.shape[1] == anchor_pos`.
        This is exactly the rollout's FIRST round (and for that round it is
        bit-identical, asserted in the GPU smoke).
      * `cache=<DynamicCache>`: a carried cache, as `rollout` leaves it after
        `cache.crop(pos)`, so `fused` must hold only the newly consumed rows,
        `fused.shape[1] == anchor_pos - cache.get_seq_length()`. The caller owns
        the crop, exactly as the rollout does; this function appends the block's
        keys to the cache it is given.

    The rollout's cache holds only context keys (its noise keys are cropped away
    every round), so a chain replay with this function reproduces the rollout;
    a fresh-cache call at the SAME anchor is mathematically identical but feeds
    `fc`/`k_proj` `anchor_pos` rows instead of one round's `n_acc+1`, and bf16
    row-count noise moves the logits (measured at 0.25-0.375 in B0).

    No `torch.no_grad` here on purpose: the whole point of `q` is that the
    objective differentiates it back into the head. Wrap the call in
    `torch.no_grad()` when scoring only. The output keeps `lm_head`'s dtype
    (bf16 in this project); `opd_fullkl_loss` casts to fp32 internally.
    """
    block = int(head.block_size) if block_size is None else int(block_size)
    if block < 2:
        raise ValueError(f"block_size must be >= 2, got {block}")
    if prompt_len is not None and int(anchor_pos) < int(prompt_len):
        raise ValueError(f"anchor_pos {anchor_pos} is before prompt_len {prompt_len}")

    cache = DynamicCache() if cache is None else cache
    cache_len = int(cache.get_seq_length())
    want = int(anchor_pos) - cache_len
    if fused.ndim != 3 or fused.shape[0] != 1:
        raise ValueError(f"fused must be (1, n, L*H), got {tuple(fused.shape)}")
    if int(fused.shape[1]) != want:
        raise ValueError(
            f"fused has {int(fused.shape[1])} context rows but the head needs "
            f"{want} (anchor_pos {anchor_pos} - cache {cache_len}); pass the "
            "rows for stream positions cache_len..anchor_pos-1")

    ids = torch.full((1, block), int(head.mask_token_id), dtype=torch.long,
                     device=fused.device)
    ids[0, 0] = int(root_token)
    position_ids = torch.arange(cache_len, int(anchor_pos) + block,
                                device=fused.device).unsqueeze(0)
    hidden = head(target_hidden=fused,
                  noise_embedding=target.get_input_embeddings()(ids),
                  position_ids=position_ids, past_key_values=cache,
                  use_cache=True, is_causal=False)
    return target.lm_head(hidden[:, 1 - block:, :])[0]


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------

def full_kl_forward(p_logits, q_logits):
    """Per-row `KL(p || q)` over the FULL vocabulary, computed in fp32.

    `(..., V)` in, `(...)` out. No top-k, no renormalisation, no support
    argument: the vocabulary is the support. Differentiable in both arguments,
    which is why the teacher side must be built under `no_grad`.
    """
    logp = F.log_softmax(p_logits.float(), dim=-1)
    logq = F.log_softmax(q_logits.float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def full_kl_reverse(q_logits, p_logits):
    """Per-row `KL(q || p)` over the FULL vocabulary, computed in fp32."""
    logq = F.log_softmax(q_logits.float(), dim=-1)
    logp = F.log_softmax(p_logits.float(), dim=-1)
    return (logq.exp() * (logq - logp)).sum(dim=-1)


# --------------------------------------------------------------------------
# The OFFICIAL Draft-OPD rejected-stream KL (the k3 sampled estimator)
# --------------------------------------------------------------------------
#
# `run_qwen_gsm8k.sh` (`d1476f5`, 2026-08-08) is the newest KL-version entry of
# Draft-OPD and sets `DISTILLATION_LOSS_MODE=k3` with
# `REJECTED_DRAFT_USE_REVERSE_KL=True`: the rejected suffix is scored with the
# k3 single-sample estimator, not with a full-vocabulary reverse KL. The two are
# different objectives (`docs/reports/2026-10-15-*`): the paper's Eq. 10/11 sum
# over `y in V`, while the official code only needs the log-probability of the
# token the trajectory actually carries.
#
# The transcription below is literal; the constants are the official clamps.

#: `core_algos.kl_penalty`, k3 branch: clamp on `kl = ref_logprob - logprob`.
K3_LOG_RATIO_CLAMP = 20.0
#: `core_algos.kl_penalty`, k3 branch: clamp on `kld = ratio - kl - 1`.
K3_KLD_CLAMP = 10.0

#: Names of the two rejected-stream objectives (`FullKLLossConfig.rejected_kl`).
REJECTED_KL_FULL_REVERSE = "full_reverse"   # paper Eq. 11, current GLANCE default
REJECTED_KL_K3 = "k3"                       # official `run_qwen_gsm8k.sh`


def k3_from_logprobs(student_logprob, teacher_logprob, *,
                     log_ratio_clamp: float = K3_LOG_RATIO_CLAMP,
                     kld_clamp: float = K3_KLD_CLAMP):
    """The official k3 estimator on log-probabilities, literal transcription.

    Source: `Draft-OPD/refs/Draft-OPD/verl/verl/trainer/ppo/core_algos.py:2183-2191`
    (`kl_penalty(..., "low_var_kl"/"k3")`), called by the rejected-draft stream
    with `logprob=student`, `ref_logprob=teacher`:

        kl    = ref_logprob - logprob          # log(p/q)
        kl    = clamp(kl, min=-20, max=20)
        ratio = exp(kl)
        kld   = ratio - kl - 1
        return clamp(kld, min=-10, max=10)

    `E_q[ratio - kl - 1] = KL(q || p)` when the token is a sample of `q`, which is
    why the official entry pairs it with `REJECTED_DRAFT_USE_REVERSE_KL=True`. The
    token here is the trajectory's own draft token (the proposal the rollout
    made), exactly as `opd_multimodal.opd_train_vlm._k3_reverse_kl` cross-refers.

    Differentiable in `student_logprob`, which is what the objective needs; the
    clamps are the official ones, so the estimator is bounded by `kld_clamp`.
    """
    kl = (teacher_logprob - student_logprob).clamp(min=-float(log_ratio_clamp),
                                                   max=float(log_ratio_clamp))
    ratio = torch.exp(kl)
    return (ratio - kl - 1.0).clamp(min=-float(kld_clamp), max=float(kld_clamp))


def sampled_logprobs(q_logits, p_logits, sample_tokens):
    """`(student_logprob, teacher_logprob)` of `sample_tokens`, fp32, full vocab.

    `q_logits`/`p_logits` are `(..., V)`; `sample_tokens` is `(...)` int. The
    log-probabilities are the ordinary full-softmax ones (no top-k, no
    renormalisation), so the sampled estimator is computed against exactly the
    distributions the full-vocabulary objective would use.
    """
    if q_logits.shape != p_logits.shape:
        raise ValueError(f"teacher {tuple(p_logits.shape)} and student "
                         f"{tuple(q_logits.shape)} must have the same shape")
    idx = torch.as_tensor(sample_tokens, dtype=torch.long, device=q_logits.device)
    if idx.shape != q_logits.shape[:-1]:
        raise ValueError(f"sample_tokens must have shape {tuple(q_logits.shape[:-1])}, "
                         f"got {tuple(idx.shape)}")
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= q_logits.shape[-1]):
        raise ValueError(f"sample tokens outside [0, {q_logits.shape[-1]})")
    col = idx.unsqueeze(-1)
    logq = F.log_softmax(q_logits.float(), dim=-1).gather(-1, col).squeeze(-1)
    logp = F.log_softmax(p_logits.float(), dim=-1).gather(-1, col).squeeze(-1)
    return logq, logp


def k3_reverse_kl(q_logits, p_logits, sample_tokens, **kwargs):
    """Per-row k3 reverse-KL estimate of the official objective, on `sample_tokens`."""
    logq, logp = sampled_logprobs(q_logits, p_logits, sample_tokens)
    return k3_from_logprobs(logq, logp, **kwargs)


@dataclass
class FullKLLossConfig:
    """The verified constants of Eq. 8-13: `w_k = gamma**(k-1)`, equal lambdas.

    `rejected_kl` selects the rejected stream's objective:
      * `"full_reverse"` (default) -- paper Eq. 11, `sum_{y in V} q(log q - log p)`;
      * `"k3"` -- the official sampled estimator, needs `sample_tokens`.
    The default is the paper version, so every historical run stays bit-identical
    unless the switch is explicitly flipped.
    """

    gamma: float = 0.8
    lambda_acc: float = 1.0
    lambda_rej: float = 1.0
    rejected_kl: str = REJECTED_KL_FULL_REVERSE


def _stream_masks(K, n_acc, valid, device):
    """`(acc_mask, rej_mask)` over offsets 1..K, honouring `valid`.

    The split is by OFFSET, not by surviving position: an invalid offset inside
    the accepted range is simply not supervised (the caller's `n_acc` came from
    the verifier, which counts offers, not usable rows). Both masks exclude
    invalid offsets, which is what keeps them out of the denominators too.
    """
    if not 0 <= n_acc <= K:
        raise ValueError(f"n_acc {n_acc} outside [0, {K}]")
    idx = torch.arange(1, K + 1, device=device)
    if valid is None:
        valid_mask = torch.ones(K, dtype=torch.bool, device=device)
    else:
        valid_mask = torch.as_tensor(valid, dtype=torch.bool, device=device)
        if valid_mask.shape != (K,):
            raise ValueError(f"valid must have shape ({K},), got "
                             f"{tuple(valid_mask.shape)}")
    return valid_mask & (idx <= n_acc), valid_mask & (idx > n_acc)


def stream_split(K, n_acc, valid=None, *, device=None):
    """The objective's offset split as two int64 tensors of 1-based offsets.

    Public because update-level aggregation is deliberately NOT this module's
    job. The verified objective is per block, and a trainer pooling several
    anchors with different accepted/rejected counts must pool the per-offset KLs
    (not average per-block means, which would weight a 1-offset block like a
    15-offset one). This returns the index sets so the caller can compute the
    KLs itself -- `full_kl_forward(p[i-1], q[i-1])` -- and combine them by
    whatever pooling the update rule needs, with autograd intact.

    The offsets are 1-based and match `w_k = gamma**(k-1)`; invalid offsets are
    absent from both sets.
    """
    device = torch.device("cpu") if device is None else device
    acc_mask, rej_mask = _stream_masks(int(K), int(n_acc), valid, device)
    idx = torch.arange(1, int(K) + 1, device=device)
    return idx[acc_mask], idx[rej_mask]


def opd_fullkl_loss(p_logits, q_logits, n_acc, *, config=None, valid=None,
                    sample_tokens=None):
    """Draft-OPD's KL block loss. Returns `(loss, metrics)`.

    `p_logits`/`q_logits` are `(K, V)` (fp32 or bf16) teacher/student logits for
    ONE block's offsets 1..K; `n_acc` is how many leading offsets the verifier
    accepted. `valid` is an optional `(K,)` bool mask of usable offsets (EOS,
    padding and truncation are the caller's job); invalid offsets leave BOTH
    streams and BOTH denominators. `sample_tokens` is the `(K,)` int tensor of the
    tokens the trajectory carries, i.e. the draft's own proposals -- required iff
    `config.rejected_kl == "k3"` (the official sampled estimator), ignored by the
    default full-vocabulary paper objective.

    Empty streams are dropped and the surviving lambdas renormalised, so an
    only-accepted block is `L = L_acc` and not `L_acc/2`; only when both streams
    are empty is the loss exactly zero (`q_logits.sum() * 0`, so it still
    backpropagates a zero gradient). `metrics["streams_present"]` and the counts
    are how the caller sees that case.

    Metrics are Python floats (logging values, detached): `l_acc`, `l_rej`,
    `n_acc_used`, `n_rej_used`, `sum_w_rej`, `acc_present`, `rej_present`,
    `streams_present`, plus `lambda_acc_used`/`lambda_rej_used`, the weights the
    two streams actually entered the combination with. Under `rejected_kl="k3"`
    the k3 diagnostics are added: `k3_log_ratio_clamp_fraction`,
    `k3_kld_clamp_fraction` and `k3_log_ratio_max` (the raw, unclamped
    `log p - log q`), which is what makes the estimator's scale auditable.

    Differentiable w.r.t. `q_logits`; `p_logits` is never differentiated and is
    expected to be a detached teacher tensor (`requires_grad=False`).
    """
    cfg = FullKLLossConfig() if config is None else config
    if cfg.rejected_kl not in (REJECTED_KL_FULL_REVERSE, REJECTED_KL_K3):
        raise ValueError(f"unknown rejected_kl {cfg.rejected_kl!r}; expected "
                         f"{REJECTED_KL_FULL_REVERSE!r} or {REJECTED_KL_K3!r}")
    if p_logits.shape != q_logits.shape:
        raise ValueError(f"teacher {tuple(p_logits.shape)} and student "
                         f"{tuple(q_logits.shape)} must have the same shape")
    if p_logits.ndim != 2:
        raise ValueError(f"logits must be (K, V), got {tuple(p_logits.shape)}")
    K = int(p_logits.shape[0])
    n_acc = int(n_acc)

    acc_mask, rej_mask = _stream_masks(K, n_acc, valid, q_logits.device)
    n_acc_used = int(acc_mask.sum())
    n_rej_used = int(rej_mask.sum())
    acc_present = n_acc_used > 0
    rej_present = n_rej_used > 0

    l_acc = 0.0
    l_rej = 0.0
    sum_w_rej = 0.0
    terms: list[torch.Tensor] = []
    lambdas: list[float] = []
    k3_metrics: dict[str, float] = {}

    if acc_present:
        # Plain mean over the accepted offsets: no position weight anywhere in
        # this stream (verified ground truth; the prototype's `weighted_stream_mean`
        # applied w_k here and that was a pre-verification guess).
        acc_term = full_kl_forward(p_logits[acc_mask], q_logits[acc_mask]).mean()
        l_acc = float(acc_term.detach())
        terms.append(cfg.lambda_acc * acc_term)
        lambdas.append(float(cfg.lambda_acc))

    if rej_present:
        # w_k belongs to the OFFSET k, not to the surviving-position index. The
        # schedule is a constant, so it is evaluated in float64 and rounded once:
        # `0.8 ** 2` in fp32 pow() is a few ulp off the paper's `0.8**(k-1)`,
        # which would show up as a spurious 1e-7 in every denominator check.
        idx = torch.arange(1, K + 1, device=q_logits.device)
        w = (torch.as_tensor(float(cfg.gamma), dtype=torch.float64,
                             device=q_logits.device)
             ** (idx[rej_mask] - 1).double()).float()
        if cfg.rejected_kl == REJECTED_KL_K3:
            if sample_tokens is None:
                raise ValueError("rejected_kl='k3' needs sample_tokens (the "
                                 "trajectory's own draft tokens), got None")
            tok = torch.as_tensor(sample_tokens, dtype=torch.long,
                                  device=q_logits.device)
            if tok.shape != (K,):
                raise ValueError(f"sample_tokens must have shape ({K},), got "
                                 f"{tuple(tok.shape)}")
            kl = k3_reverse_kl(q_logits[rej_mask], p_logits[rej_mask],
                               tok[rej_mask])
            # Diagnostics of the estimator's own clamps: the raw, UNCLAMPED
            # `log p - log q` and how often each official clamp bites.
            logq_t, logp_t = sampled_logprobs(q_logits[rej_mask], p_logits[rej_mask],
                                              tok[rej_mask])
            raw = (logp_t - logq_t).detach()
            k3_metrics = {
                "k3_log_ratio_clamp_fraction":
                    float((raw.abs() >= K3_LOG_RATIO_CLAMP).float().mean()),
                "k3_kld_clamp_fraction":
                    float((kl.detach().abs() >= K3_KLD_CLAMP).float().mean()),
                "k3_log_ratio_max": float(raw.abs().max()),
            }
        else:
            kl = full_kl_reverse(q_logits[rej_mask], p_logits[rej_mask])
        w_sum = w.sum()
        rej_term = (w * kl).sum() / w_sum
        l_rej = float(rej_term.detach())
        sum_w_rej = float(w_sum)
        terms.append(cfg.lambda_rej * rej_term)
        lambdas.append(float(cfg.lambda_rej))

    if terms:
        loss = sum(terms) / sum(lambdas)
    else:
        # Both streams empty: contribute nothing, but stay on the graph so a
        # training step over a fully-invalid block is a no-op rather than an
        # error. The added 0.0 is what makes the result +0.0 rather than -0.0
        # when `q_logits.sum()` happens to be negative.
        loss = q_logits.sum() * 0.0 + 0.0

    metrics = {
        "l_acc": l_acc,
        "l_rej": l_rej,
        "n_acc_used": n_acc_used,
        "n_rej_used": n_rej_used,
        "sum_w_rej": sum_w_rej,
        "acc_present": bool(acc_present),
        "rej_present": bool(rej_present),
        "streams_present": bool(acc_present or rej_present),
        "lambda_acc_used": float(cfg.lambda_acc) if acc_present else 0.0,
        "lambda_rej_used": float(cfg.lambda_rej) if rej_present else 0.0,
        "rejected_kl": str(cfg.rejected_kl),
        **k3_metrics,
    }
    return loss, metrics


# --------------------------------------------------------------------------
# Gradient diagnostics
# --------------------------------------------------------------------------

def grad_cosine_and_norm_ratio(grads_a, grads_b):
    """Cosine and norm ratio between two gradient sets, tensor by tensor.

    WHY per tensor: concatenating a whole model's gradients to take one dot
    product costs a second copy of the model in memory, which on an 8B target is
    the difference between a diagnostic that runs next to training and one that
    does not. Accumulating `sum(a.b)`, `sum(a.a)`, `sum(b.b)` per tensor gives the
    identical numbers in float64 without ever materialising the concatenation.

    A key with a grad in only one dict contributes to that side's norm (and
    nothing to the dot), which is the honest reading of a cosine over the union
    of the two gradient sets. `n_tensors` counts the keys compared.

    Degenerate cases are defined rather than left to produce NaNs silently:
    zero norm on either side gives `cosine = 0.0`, and `norm_ratio` is `inf`
    when only `b` is zero and `nan` when both are.
    """
    keys = {k for k, v in list(grads_a.items()) + list(grads_b.items())
            if v is not None}
    dot = 0.0
    sq_a = 0.0
    sq_b = 0.0
    for key in sorted(keys):
        a = grads_a.get(key)
        b = grads_b.get(key)
        a = None if a is None else a.detach().reshape(-1)
        b = None if b is None else b.detach().reshape(-1)
        if a is not None and b is not None and a.numel() != b.numel():
            raise ValueError(f"gradient {key!r} has {a.numel()} elements in A "
                             f"and {b.numel()} in B")
        if a is not None:
            a = a.double()
            sq_a += float(torch.dot(a, a))
        if b is not None:
            b = b.double()
            sq_b += float(torch.dot(b, b))
        if a is not None and b is not None:
            dot += float(torch.dot(a, b))

    norm_a = math.sqrt(sq_a)
    norm_b = math.sqrt(sq_b)
    if norm_a > 0.0 and norm_b > 0.0:
        # Clamped because a cosine cannot leave [-1, 1]: without it, identical
        # gradients come back as 0.9999999999999998 from sqrt round-off.
        cosine = min(1.0, max(-1.0, dot / (norm_a * norm_b)))
        ratio = norm_a / norm_b
    else:
        cosine = 0.0
        if norm_b > 0.0:
            ratio = norm_a / norm_b
        else:
            ratio = math.inf if norm_a > 0.0 else math.nan
    return {"cosine": cosine, "norm_ratio": ratio, "n_tensors": len(keys),
            "norm_a": norm_a, "norm_b": norm_b}
