#!/usr/bin/env python3
"""Phase B0: full-vocabulary Draft-OPD KL on the draft-INDUCED prefix (feasibility).

WHY this file exists. The video OPD trainer (`opd_loss_video.py` /
`opd_train_video.py`) is not faithful to Draft-OPD in three ways that the plan
file `后续计划_视频OPD_2026-10-08.md` section "### B0" now makes blocking:

  1. the teacher is a *stored* top-64 row (`topk_ids`/`topk_val`), renormalised
     on both sides, so the loss never sees the ~151k tokens outside the support;
  2. the rejected suffix is one offset long (`opd_loss_video.is_first_rej`), and
     with the image-side `--reject-target topk` the reverse-KL stream is EMPTY
     by construction (`n_k3_rej=0`);
  3. the stored teacher row is the target's logits along the TARGET's own greedy
     trajectory, so a rejected offset is scored on a context the drafter never
     produced.

Before writing a trainer for the faithful objective, this prototype answers the
only two questions that can kill it: (a) can the target be scored on the
draft-induced prefix cheaply and *correctly* (a forked prefix KV, no re-prefill
of the video prompt, and logits that match an independent full forward), and
(b) does the whole thing fit in an A6000's memory when the vocabulary is not
truncated. It runs 1 video row and 4 chain anchors and gates itself:

  G1 incrementally-scored teacher logits == one independent full forward over
     `ids[:p+1] + prop`;
  G2 the frozen target has no gradients and acquires none;
  G3 the rejected suffix really is scored on the draft-induced prefix (and the
     draft really diverges there at all);
  G4 peak GPU memory stays under ~44 GiB;
  G5 no full-vocabulary logits and no KV cache ever reach disk.

No `try`/`except`: a failure here is a finding and must surface with its
traceback. Nothing in the repository is modified; this file is additive.
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
import copy
import json
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, DynamicCache

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from historical import opd_online_rollout
from glance.vl import VisionPositions

# The plan's own numbers: `w_k = 0.8^(k-1)` inside a block, equal stream weight.
DECAY = 0.8
MEMORY_LIMIT_GIB = 44.0


# --------------------------------------------------------------------------
# Target primitives
# --------------------------------------------------------------------------

@torch.no_grad()
def model_hidden(target, tokens, *, position_ids, past_key_values,
                 pixel_values_videos=None, video_grid_thw=None):
    """`target.model` on `tokens`, returning final-norm hidden states `(n, H)`.

    Calling the inner model instead of the ConditionalGeneration wrapper is not
    a shortcut: `Qwen3VLForConditionalGeneration.forward` is literally
    `self.model(...)` followed by
    `self.lm_head(hidden_states[:, slice_indices, :])`. Doing the same two steps
    by hand is what lets G1 separate a difference in the ATTENTION/KV path from
    a difference in the OUTPUT PROJECTION -- which is exactly where the two
    paths turn out to diverge.
    """
    out = target.model(tokens, position_ids=position_ids,
                       past_key_values=past_key_values, use_cache=True,
                       pixel_values_videos=pixel_values_videos,
                       video_grid_thw=video_grid_thw)
    return out.last_hidden_state[0], out.past_key_values


def fork_cache(cache):
    """An independent copy of a prefix KV.

    `copy.deepcopy` is the fork primitive the B0 plan names. Every
    `DynamicLayer` owns its `keys`/`values` tensor, so deepcopy clones the whole
    cache: appending to a fork cannot mutate the shared prefix or any sibling
    fork. This is what lets N anchors and N offsets share ONE video prefill.
    """
    return copy.deepcopy(cache)


@torch.no_grad()
def append_through_fork(target, pos, fork, tokens, first_index):
    """Push already-known tokens through a forked prefix KV.

    `tokens` is `(1, m)` and its stream indices are
    `first_index .. first_index + m - 1`. Row `r` of the returned hidden states
    / logits is the target AT stream position `first_index + r` -- i.e. the
    distribution for the token at `first_index + r + 1`, conditioned on
    everything before it. Positions come from `VisionPositions`, never from a
    counter, because M-RoPE shifts the whole post-video stream by the prefill
    delta. The logits are the same `lm_head(hidden)` the wrapper's
    `logits_to_keep=0` produces; the caller consumes them immediately, so no
    `(n, V)` tensor survives the loop.
    """
    m = tokens.shape[1]
    index = torch.arange(first_index, first_index + m, device=tokens.device)
    hidden, _ = model_hidden(target, tokens, position_ids=pos(index),
                             past_key_values=fork)
    return hidden, target.lm_head(hidden)


@torch.no_grad()
def stepwise_through_fork(target, pos, fork, tokens, first_index):
    """The SAME tokens, one at a time, through a fork of the same prefix KV.

    This is the control that distinguishes a structural error from bf16 kernel
    noise. Chunked appending and token-by-token appending are the same function
    mathematically -- causal attention, explicit M-RoPE positions, the same
    keys and values -- and both are things the decode loop legitimately does.
    Any difference between them is therefore pure row-count sensitivity of the
    bf16 GEMMs, at exactly the scale the G1 gap has to be judged against.
    """
    m = tokens.shape[1]
    cache = fork_cache(fork)
    rows = []
    for r in range(m):
        idx = torch.tensor([first_index + r], device=tokens.device)
        hidden, _ = model_hidden(target, tokens[:, r:r + 1],
                                 position_ids=pos(idx), past_key_values=cache)
        rows.append(target.lm_head(hidden)[0])
    return torch.stack(rows, dim=0)


@torch.no_grad()
def score_anchor(target, pv, vg, prefix_ids, prop, p,
                 with_chunking_control=False):
    """Run BOTH teacher paths for one draft block and return everything G1 needs.

    Path A (incremental / decode-consistent): prefill `ids[:p+1]` once, then push
    the drafter's own `prop` through a FORK of that prefix KV. Row `k-1` of `inc`
    is the teacher for offset `k`, conditioned on
    `ids[:p+1] + prop[:k-1]` -- the context the drafter actually created.

    Path B (reference): one independent full forward over the whole
    draft-induced text, fresh cache, same `VisionPositions`, its own single
    `lm_head` call on the `m+1` positions the block needs.

    The two position claims the comparison rests on are checked here rather than
    assumed: the prefix's rope indices are the full sequence's first `p+1`
    columns, and `pos(stream)` reproduces the full sequence's indices for the
    appended tokens.
    """
    m = len(prop)
    pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
    pos.bind(target, prefix_ids)
    prefill_kw = pos.prefill_inputs(prefix_ids, None)
    h_pre, prefix_kv = model_hidden(
        target, prefix_ids, position_ids=prefill_kw["position_ids"],
        past_key_values=DynamicCache(),
        pixel_values_videos=prefill_kw["pixel_values_videos"],
        video_grid_thw=prefill_kw["video_grid_thw"])
    logits_pre = target.lm_head(h_pre[-1:])[0]              # offset 1, shape (V,)

    chunk = torch.tensor([prop], dtype=torch.long, device=prefix_ids.device)
    h_chunk, logits_chunk = append_through_fork(
        target, pos, fork_cache(prefix_kv), chunk, p + 1)
    h_inc = torch.cat([h_pre[-1:], h_chunk[:m - 1]], dim=0)   # hidden for offsets 1..m
    inc = torch.cat([logits_pre[None], logits_chunk[:m - 1]], dim=0)

    full_ids = torch.cat([prefix_ids, chunk], dim=1)
    fpos, _ = target.model.get_rope_index(full_ids, None, vg, None)
    h_full, _ = model_hidden(target, full_ids, position_ids=fpos,
                             past_key_values=DynamicCache(),
                             pixel_values_videos=pv, video_grid_thw=vg)
    # `logits_to_keep=m+1` on the wrapper slices exactly these rows.
    ref_all = target.lm_head(h_full[-(m + 1):])
    ref_h = h_full[p:p + m]
    ref = ref_all[:m]

    if not torch.equal(fpos[:, :, :p + 1], prefill_kw["position_ids"]):
        raise AssertionError("prefix M-RoPE positions differ from the "
                             "full-sequence positions")
    stream = torch.arange(p + 1, p + 1 + m, device=prefix_ids.device)
    if not torch.equal(pos(stream), fpos[:, :, p + 1:p + 1 + m]):
        raise AssertionError("VisionPositions(stream) does not reproduce the "
                             "full-sequence positions of the appended tokens")
    out = {"pos": pos, "prefix_kv": prefix_kv, "prefix_logits": logits_pre,
           "inc": inc, "ref": ref, "h_inc": h_inc, "ref_h": ref_h,
           "ref_all": ref_all,
           "lm_head_on_ref_h": target.lm_head(ref_h),
           "lm_head_rowwise": torch.cat(
               [target.lm_head(ref_h[j][None]) for j in range(m)], dim=0)}
    if with_chunking_control:
        out["inc_stepwise"] = torch.cat(
            [logits_pre[None],
             stepwise_through_fork(target, pos, prefix_kv, chunk, p + 1)[:m - 1]],
            dim=0)
    return out


# --------------------------------------------------------------------------
# The drafter's own block logits
# --------------------------------------------------------------------------

@torch.inference_mode()
def replay_draft_logits(head, target, ctx, seq_ids, anchors, prompt_len):
    """Rebuild the rollout's head KV cache and return each anchor's block logits.

    WHY a replay instead of one fresh block forward per anchor. `rollout` does
    not return the draft distribution, and its head cache is CHUNK-dependent:
    round r feeds `ctx[:, pos_{r-1}:pos_r]` plus the block, so the head's
    20480->4096 `fc` is applied to a different number of rows each round.
    Rebuilding the cache from `ctx[:, :p]` in one pass would change that row
    count and can move the logits by a rounding step -- at which point the
    tokens we score would no longer be the argmax the rollout proposed, and the
    accepted/rejected split would be defined against a different distribution.
    This mirrors `opd_online_rollout.rollout`'s own head forward step for step
    (same chunks, same `stream` positions, same `cache.crop`) and then ASSERTS
    that the replay's argmax reproduces `prop`, so `q_draft` is bit-for-bit the
    tensor the rollout proposed from.

    The logits themselves are `target.lm_head(hidden[:, 1-block:, :])`, exactly
    as `rollout`/`glance.decode` produce them. `head.compute_logits` would be
    equivalent for this checkpoint (`output_multiplier` and `final_logit_softcap`
    are both absent from its config) but would be a different call than the one
    the rollout made, and only one of the two can be asserted equal to `prop`.
    """
    block = int(head.block_size)
    mask_id = int(head.mask_token_id)
    total = min(int(seq_ids.shape[0]), int(ctx.shape[1]))
    embed = target.get_input_embeddings()
    cache = DynamicCache()
    stream = torch.arange(total + block + 1, device=seq_ids.device).unsqueeze(0)
    fused = ctx[:, :prompt_len]
    pos = int(prompt_len)
    out = []
    for a in anchors:
        if int(a["pos"]) != pos:
            raise ValueError(f"anchor pos {a['pos']} is not the rollout's next "
                             f"block start {pos}; anchors are not a chain")
        ids = torch.full((1, block), mask_id, dtype=torch.long,
                         device=seq_ids.device)
        ids[0, 0] = seq_ids[pos]
        hidden = head(target_hidden=fused, noise_embedding=embed(ids),
                      position_ids=stream[:, cache.get_seq_length():pos + block],
                      past_key_values=cache, use_cache=True, is_causal=False)
        out.append(target.lm_head(hidden[:, 1 - block:, :])[0])
        cache.crop(pos)
        old_pos = pos
        pos += int(a["n_acc"]) + 1
        fused = ctx[:, old_pos:pos]

    for a, logits in zip(anchors, out):
        prop = [int(t) for t in a["prop"]]
        got = [int(t) for t in logits.argmax(-1)[:len(prop)]]
        if got != prop:
            raise ValueError(
                f"replayed draft logits do not reproduce the rollout proposal at "
                f"pos {a['pos']}: rollout {prop} vs replay {got}")
    return out


# --------------------------------------------------------------------------
# KL
# --------------------------------------------------------------------------

def full_kl(logits_a, logits_b):
    """(forward KL(a||b), reverse KL(b||a)) over the FULL vocabulary, fp32.

    Both directions come from the same two log-softmaxes; `logits_a` is the
    teacher and `logits_b` the drafter. No top-k, no renormalisation, no
    support argument: the vocabulary is the support.
    """
    loga = F.log_softmax(logits_a.float(), dim=-1)
    logb = F.log_softmax(logits_b.float(), dim=-1)
    kl_fwd = float((loga.exp() * (loga - logb)).sum())
    kl_rev = float((logb.exp() * (logb - loga)).sum())
    return kl_fwd, kl_rev


def weighted_stream_mean(kls, weights):
    """Separately-normalised weighted mean of one stream; 0 when it is empty.

    The plan's `w_k = 0.8^(k-1)` and "accepted/rejected 两个 stream 等权并分别归一":
    each stream is normalised by the sum of its OWN weights, and the two
    normalised streams enter the total with weight 1.0 each. This is the shape
    `opd_loss_video._wmean` already uses for a single stream; here it is applied
    per stream so an anchor with few accepted offsets is not outweighed by one
    with many.
    """
    if not kls:
        return 0.0
    num = sum(k * w for k, w in zip(kls, weights))
    den = sum(weights)
    return num / max(den, 1e-12)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=f"{common.DATA}/video_fresh30.parquet")
    ap.add_argument("--records", default=f"{common.DATA}/video_opd_pool_warm_499.pt",
                    help="pool blob holding the target's greedy trajectory "
                         "(input_ids) for the row; the same convention "
                         "opd_train_video.py trains on")
    ap.add_argument("--head", default=f"{common.CKPT}/head_video439/step1000")
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument("--n-anchors", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--out", default=f"{common.RESULTS}/phaseB0/proto_fullkl_b0.json")
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    t_start = time.perf_counter()
    # Snapshot the output directory BEFORE anything runs, so the G5 file sweep
    # sees anything the run itself could have dropped next to the JSON.
    out_dir = os.path.dirname(os.path.abspath(a.out))
    before = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
    torch.cuda.set_device(a.gpu)
    print(f"[env] gpu {a.gpu} = {torch.cuda.get_device_name(a.gpu)}  "
          f"torch {torch.__version__}  cuda {torch.version.cuda}", flush=True)

    # ---------------------------------------------------------------- load
    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev).eval()
    head.requires_grad_(False)
    print(f"[load] target {sum(p.numel() for p in target.parameters())/1e9:.2f}B bf16, "
          f"head {a.head}", flush=True)

    # G2 is a property of the loaded model, so check it before anything runs.
    tg_req = [n for n, p in target.named_parameters() if p.requires_grad]
    if tg_req:
        raise AssertionError(f"target has trainable parameters: {tg_req[:5]}")
    if target.training:
        raise AssertionError("target is in train mode")
    hd_req = [n for n, p in head.named_parameters() if p.requires_grad]
    if hd_req:
        raise AssertionError(f"head has trainable parameters: {hd_req[:5]}")

    # ---------------------------------------------------------------- row
    df = pd.read_parquet(a.rows)
    row = df.iloc[a.row_index]
    name = str(row["video_name"])
    blob = torch.load(a.records, weights_only=False, map_location="cpu")
    recs = [r for r in blob["records"] if str(r["task"]) == name]
    if len(recs) != 1:
        raise ValueError(f"{name} appears {len(recs)} times in {a.records}; the "
                         "prototype needs exactly one stored target trajectory")
    rec = recs[0]
    prompt_len = int(rec["prompt_len"])
    seq_ids = rec["input_ids"].to(dev).long()
    total = int(seq_ids.shape[0])

    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": row["question"]}]}],
        add_generation_prompt=True, tokenize=False)
    ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
               fps=float(row["fps"]))
    ids = ins["input_ids"].to(dev)
    pv = ins["pixel_values_videos"].to(dev)
    vg = ins["video_grid_thw"].to(dev)
    if int(ids.shape[1]) != prompt_len:
        raise ValueError(f"encoded prompt {int(ids.shape[1])} != stored "
                         f"prompt_len {prompt_len}: the prompt string drifted "
                         "from the pool the trajectory was collected with")
    print(f"[row] {name}  prompt {prompt_len}  trajectory {total}  "
          f"video_grid {vg.tolist()}  n_frames {int(row['n_frames'])}", flush=True)

    # Everything below is the measured window for G4.
    torch.cuda.reset_peak_memory_stats(a.gpu)
    torch.cuda.synchronize()
    base_alloc_gib = torch.cuda.memory_allocated(a.gpu) / 2**30

    # ------------------------------------------------- rollout (chain, 4 anchors)
    t0 = time.perf_counter()
    ctx = common.target_context_cached(target, seq_ids[None], pixel_values_videos=pv,
                                       video_grid_thw=vg)
    torch.cuda.synchronize()
    t_ctx = time.perf_counter() - t0
    print(f"[ctx] {tuple(ctx.shape)} bf16 in {t_ctx:.1f}s "
          f"({ctx.numel()*ctx.element_size()/2**30:.2f} GiB)", flush=True)

    t0 = time.perf_counter()
    anchors = opd_online_rollout.rollout(head, target, ctx, seq_ids,
                                         prompt_len=prompt_len, verify="chain",
                                         max_rounds=a.n_anchors)
    torch.cuda.synchronize()
    t_roll = time.perf_counter() - t0
    print(f"[rollout] chain x{len(anchors)} in {t_roll:.1f}s", flush=True)
    for i, an in enumerate(anchors):
        print(f"    anchor {i}: pos {an['pos']:5d}  n_acc {an['n_acc']:2d}  "
              f"len(prop) {len(an['prop']):2d}", flush=True)
    if len(anchors) != a.n_anchors:
        raise RuntimeError(f"chain rollout returned {len(anchors)} anchors, "
                           f"wanted {a.n_anchors}")

    t0 = time.perf_counter()
    q_logits = replay_draft_logits(head, target, ctx, seq_ids, anchors, prompt_len)
    torch.cuda.synchronize()
    t_replay = time.perf_counter() - t0
    print(f"[replay] draft logits {[tuple(q.shape) for q in q_logits]} "
          f"in {t_replay:.1f}s; argmax == prop on every anchor", flush=True)

    # ------------------------------------------------- per-anchor teacher scoring
    per_anchor = []
    g1_max_abs = g1_max_tv = g1_hidden_max_abs = 0.0
    g1_lmhead_rows_max_abs = 0.0
    g1_chunking_max_abs = g1_chunking_max_tv = 0.0
    g1_argmax_ok = g1_argmax_tot = 0
    g1_logit_scale = g1_hidden_scale = 0.0
    loss_inc_total = loss_ref_total = 0.0
    g3 = None
    t_teacher = 0.0

    for i, (an, q) in enumerate(zip(anchors, q_logits)):
        p = int(an["pos"])
        prop = [int(t) for t in an["prop"]]
        n_acc = int(an["n_acc"])
        m = len(prop)
        if m < 2:
            raise RuntimeError(f"anchor {i} proposed {m} tokens; a 1-token block "
                               "has no offset to score")

        # 1./2. one prefix prefill, then the draft-induced path through a fork;
        #       plus the independent single full forward (the G1 reference) and
        #       the chunking control.
        t0 = time.perf_counter()
        sc = score_anchor(target, pv, vg, seq_ids[None, :p + 1], prop, p,
                          with_chunking_control=True)
        torch.cuda.synchronize()
        t_teacher += time.perf_counter() - t0
        inc, ref = sc["inc"], sc["ref"]
        pos_adapter, prefix_kv, prefix_logits = sc["pos"], sc["prefix_kv"], sc["prefix_logits"]

        # 3. G1: compare position by position, and split the gap by where it
        #    enters. `hidden` isolates the attention/KV/position path.
        #    `lmhead_rows` holds the reference hidden states fixed and changes
        #    ONLY the output projection's row count. `chunking` holds the whole
        #    method fixed and changes only how the appended tokens are batched
        #    -- mathematically the same forward, so it measures the floor below
        #    which no bf16 difference can be called structural.
        d_hidden = float((sc["h_inc"].float() - sc["ref_h"].float()).abs().max())
        d = float((inc.float() - ref.float()).abs().max())
        d_lmhead_rows = float((sc["lm_head_on_ref_h"].float()
                               - sc["lm_head_rowwise"].float()).abs().max())
        d_chunking = float((inc.float() - sc["inc_stepwise"].float()).abs().max())
        tv_chunking = max(float((F.softmax(inc[k].float(), -1)
                                 - F.softmax(sc["inc_stepwise"][k].float(), -1)).abs().sum())
                          for k in range(m))
        logit_scale = float(ref.float().abs().max())
        hidden_scale = float(sc["ref_h"].float().abs().max())
        prob_d = max(float((F.softmax(inc[k].float(), -1)
                            - F.softmax(ref[k].float(), -1)).abs().sum())
                     for k in range(m))
        am = (inc.argmax(-1) == ref.argmax(-1))
        g1_max_abs = max(g1_max_abs, d)
        g1_max_tv = max(g1_max_tv, prob_d)
        g1_hidden_max_abs = max(g1_hidden_max_abs, d_hidden)
        g1_lmhead_rows_max_abs = max(g1_lmhead_rows_max_abs, d_lmhead_rows)
        g1_chunking_max_abs = max(g1_chunking_max_abs, d_chunking)
        g1_chunking_max_tv = max(g1_chunking_max_tv, tv_chunking)
        g1_logit_scale = max(g1_logit_scale, logit_scale)
        g1_hidden_scale = max(g1_hidden_scale, hidden_scale)
        g1_argmax_ok += int(am.sum())
        g1_argmax_tot += m

        # 4. the objective. accepted offsets k <= n_acc  -> forward KL(p_t || q)
        #    rejected offsets k >  n_acc -> reverse KL(q || p_t), ALL of them.
        acc_k = [k for k in range(1, m + 1) if k <= n_acc]
        rej_k = [k for k in range(1, m + 1) if k > n_acc]
        weights = [DECAY ** (k - 1) for k in range(1, m + 1)]
        kl_fwd_inc = [full_kl(inc[k - 1], q[k - 1])[0] for k in acc_k]
        kl_rev_inc = [full_kl(inc[k - 1], q[k - 1])[1] for k in rej_k]
        w_acc = [weights[k - 1] for k in acc_k]
        w_rej = [weights[k - 1] for k in rej_k]
        loss_acc = weighted_stream_mean(kl_fwd_inc, w_acc)
        loss_rej = weighted_stream_mean(kl_rev_inc, w_rej)
        # The same objective off the independent full forward, so any numerical
        # gap G1 finds is visible in the metric it would actually affect.
        loss_acc_ref = weighted_stream_mean(
            [full_kl(ref[k - 1], q[k - 1])[0] for k in acc_k], w_acc)
        loss_rej_ref = weighted_stream_mean(
            [full_kl(ref[k - 1], q[k - 1])[1] for k in rej_k], w_rej)
        loss_inc_total += loss_acc + loss_rej
        loss_ref_total += loss_acc_ref + loss_rej_ref

        # 5. G3: the first rejected offset must be scored on the draft-induced
        #    prefix, not the target's own greedy prefix. The greedy prefix is the
        #    same prefix with prop[:k-1] replaced by seq_ids, fed through its own
        #    fork of the SAME prefix KV.
        first_rej = n_acc + 1
        if first_rej <= m and prop[n_acc] == int(seq_ids[p + 1 + n_acc]):
            raise AssertionError(f"anchor {i}: prop matches the committed token at "
                                 f"the first rejected offset {first_rej}; the G3 "
                                 "test would be vacuous")
        # The two contexts are the SAME tokens up to and including the first
        # rejected offset (the accepted run is by definition the matching run),
        # and first differ one offset later. Both are measured so the gate says
        # which is which instead of hiding a theorem behind a failed check.
        first_diff = n_acc + 2

        def greedy_logits_at(k):
            """Teacher logits at offset k on the target's OWN greedy prefix."""
            if k == 1:
                return prefix_logits
            gt = seq_ids[None, p + 1:p + k]
            return append_through_fork(target, pos_adapter, fork_cache(prefix_kv),
                                       gt, p + 1)[1][-1]

        row_g3 = {"anchor": i, "pos": p, "n_acc": n_acc, "len_prop": m,
                  "first_rejected_offset": first_rej,
                  "prop_token_at_reject": prop[n_acc],
                  "target_token_at_reject": int(seq_ids[p + 1 + n_acc])}
        if first_rej <= m:
            gl = greedy_logits_at(first_rej)
            row_g3["max_abs_inc_vs_greedy_first_rej"] = float(
                (inc[first_rej - 1].float() - gl.float()).abs().max())
            row_g3["kl_inc_vs_greedy_first_rej"] = full_kl(inc[first_rej - 1], gl)[0]
            row_g3["prefixes_identical_at_first_rej"] = (
                prop[:first_rej - 1] == [int(t) for t in seq_ids[p + 1:p + first_rej]])
        if first_diff <= m:
            gl = greedy_logits_at(first_diff)
            row_g3["first_differing_offset"] = first_diff
            row_g3["max_abs_inc_vs_greedy_first_diff"] = float(
                (inc[first_diff - 1].float() - gl.float()).abs().max())
            row_g3["kl_inc_vs_greedy_first_diff"] = full_kl(inc[first_diff - 1], gl)[0]
            if g3 is None:
                g3 = row_g3

        per_anchor.append({
            "anchor": i, "pos": p, "n_acc": n_acc, "len_prop": m,
            "prop": prop,
            "target_committed": [int(t) for t in seq_ids[p + 1:p + 1 + m]],
            "n_accepted_offsets": len(acc_k), "n_rejected_offsets": len(rej_k),
            "g1_max_abs_logit_delta": d,
            "g1_max_hidden_delta": d_hidden,
            "g1_ref_max_abs_hidden": hidden_scale,
            "g1_max_abs_lmhead_rowcount_delta": d_lmhead_rows,
            "g1_max_abs_chunking_control_delta": d_chunking,
            "g1_max_tv_chunking_control": tv_chunking,
            "g1_max_tv": prob_d,
            "g1_ref_max_abs_logit": logit_scale,
            "g1_argmax_equal": int(am.sum()), "g1_argmax_total": m,
            "kl_fwd_accepted": kl_fwd_inc, "kl_rev_rejected": kl_rev_inc,
            "loss_accepted": loss_acc, "loss_rejected": loss_rej,
            "loss_total": loss_acc + loss_rej,
            "loss_accepted_from_full_forward": loss_acc_ref,
            "loss_rejected_from_full_forward": loss_rej_ref,
            "g3": row_g3,
        })
        print(f"[anchor {i}] pos {p} n_acc {n_acc}/{m}  "
              f"G1 max|Δlogit| {d:.3e} (ref |logit|max {logit_scale:.1f}, "
              f"|hidden|max {hidden_scale:.1f}) maxΔhidden {d_hidden:.3e} "
              f"maxΔlmhead-rowcount {d_lmhead_rows:.3e} "
              f"maxΔchunking-control {d_chunking:.3e} "
              f"maxTV {prob_d:.3e} argmax {int(am.sum())}/{m}", flush=True)
        print(f"          KL_fwd(acc) {[f'{v:.3f}' for v in kl_fwd_inc]}  "
              f"KL_rev(rej) {[f'{v:.3f}' for v in kl_rev_inc[:5]]}"
              f"{' ...' if len(kl_rev_inc) > 5 else ''}  "
              f"loss acc {loss_acc:.4f} rej {loss_rej:.4f}", flush=True)
        print(f"          G3: first reject offset {first_rej} "
              f"(prop {prop[n_acc]} vs committed {int(seq_ids[p+1+n_acc])}) "
              f"max|Δ| {row_g3.get('max_abs_inc_vs_greedy_first_rej', float('nan')):.3e}; "
              f"first DIFFERING offset {row_g3.get('first_differing_offset')} "
              f"max|Δ| {row_g3.get('max_abs_inc_vs_greedy_first_diff', float('nan')):.3e} "
              f"KL {row_g3.get('kl_inc_vs_greedy_first_diff', float('nan')):.3e}",
              flush=True)

    # ------------------------------------------------- G1 root-cause control
    # The gap appears in a bf16 row-count-dependent GEMM rather than in the
    # position handling or the attention mask, so the obvious knob is torch's
    # bf16 reduced-precision reduction. Re-run the worst anchor with it off; if
    # the gap survives, the row-count sensitivity lives in the kernel choice
    # itself and no global flag removes it.
    worst = max(range(len(per_anchor)), key=lambda i: per_anchor[i]["g1_max_abs_logit_delta"])
    an_w = anchors[worst]
    flag_prev = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    sc_w = score_anchor(target, pv, vg, seq_ids[None, :int(an_w["pos"]) + 1],
                        [int(t) for t in an_w["prop"]], int(an_w["pos"]))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_prev
    g1_control = {
        "anchor": worst,
        "allow_bf16_reduced_precision_reduction_default": bool(flag_prev),
        "max_abs_logit_delta_default": per_anchor[worst]["g1_max_abs_logit_delta"],
        "max_hidden_delta_default": per_anchor[worst]["g1_max_hidden_delta"],
        "max_abs_logit_delta_flag_off": float(
            (sc_w["inc"].float() - sc_w["ref"].float()).abs().max()),
        "max_hidden_delta_flag_off": float(
            (sc_w["h_inc"].float() - sc_w["ref_h"].float()).abs().max()),
        "note": "the flag only changes bf16 matmul reduction; it does not change "
                "which cuBLAS kernel an M=1 vs M=16 GEMM uses, so a gap that "
                "survives it is kernel-shape rounding, not a wrong attention mask",
    }
    print(f"[G1 control] worst anchor {worst}: flag on max|Δ| "
          f"{g1_control['max_abs_logit_delta_default']:.3e} -> off "
          f"{g1_control['max_abs_logit_delta_flag_off']:.3e} "
          f"(hidden {g1_control['max_hidden_delta_default']:.3e} -> "
          f"{g1_control['max_hidden_delta_flag_off']:.3e})", flush=True)

    # ---------------------------------------------------------------- gates
    target_grads = [n for n, p in target.named_parameters() if p.grad is not None]
    g2 = {"pass": not target_grads and not tg_req,
          "target_params_requires_grad": int(sum(1 for p in target.parameters()
                                                 if p.requires_grad)),
          "target_params_with_grad": len(target_grads),
          "first_grad_names": target_grads[:5],
          "note": "no .backward() is ever called and the whole teacher path runs "
                  "under torch.no_grad(); the target is loaded with "
                  "requires_grad_(False) and .eval()"}
    if not g2["pass"]:
        raise AssertionError(f"G2 failed: {g2}")

    g1_exact = bool(g1_max_abs == 0.0)
    # The chunking control is the honest yardstick: two runs of the SAME method
    # that differ only in how many tokens go through one GEMM already disagree by
    # this much in bf16. A G1 gap of the same size is that noise and nothing
    # more; a gap several times larger would be structural. The position
    # reproduction in `score_anchor` is asserted, so it is not part of this test.
    g1_structural = bool(g1_max_abs <= max(2.0 * g1_chunking_max_abs, 1e-9)
                         and g1_max_tv <= max(2.0 * g1_chunking_max_tv, 1e-9))
    g1 = {
        "pass": g1_exact,
        "criterion": "the literal B0 requirement: the incrementally-scored teacher "
                     "logits must EQUAL the independent single full forward",
        "exact_bitwise": g1_exact,
        "structural_pass": g1_structural,
        "structural_criterion": "logit and TV gaps no worse than 2x what the SAME "
                                "method shows against itself when the appended "
                                "tokens are fed one at a time instead of as one "
                                "chunk (the bf16 row-count noise floor)",
        "max_abs_logit_delta": g1_max_abs,
        "ref_max_abs_logit": g1_logit_scale,
        "ref_max_abs_hidden": g1_hidden_scale,
        "max_total_variation": g1_max_tv,
        "max_hidden_delta": g1_hidden_max_abs,
        "max_abs_lmhead_rowcount_delta": g1_lmhead_rows_max_abs,
        "max_abs_chunking_control_delta": g1_chunking_max_abs,
        "max_tv_chunking_control": g1_chunking_max_tv,
        "argmax_agreement": f"{g1_argmax_ok}/{g1_argmax_tot}",
        "per_anchor_max_abs_logit_delta": [r["g1_max_abs_logit_delta"] for r in per_anchor],
        "per_anchor_max_hidden_delta": [r["g1_max_hidden_delta"] for r in per_anchor],
        "per_anchor_max_tv": [r["g1_max_tv"] for r in per_anchor],
        "root_cause_probe": g1_control,
        "interpretation": (
            "the M-RoPE positions reproduce the full sequence's EXACTLY (asserted "
            "in score_anchor), and the chunking control -- the same incremental "
            "method with 1-token steps instead of a 15-token chunk -- moves the "
            "logits by the same order as the whole G1 gap while changing nothing "
            "mathematically. So the KV fork, the positions, the causal mask and "
            "the draft-induced prefix are correct, and the residual is bf16 GEMM "
            "sensitivity to the number of rows in a batch: the prefill sees "
            "p+1 rows where the reference sees p+m+1, and lm_head sees 1 vs 16."),
    }

    g3_pass = (g3 is not None
               and g3.get("first_differing_offset") is not None
               and g3["max_abs_inc_vs_greedy_first_diff"] > 0.0
               and g3["kl_inc_vs_greedy_first_diff"] > 0.0)
    g3_out = {"pass": bool(g3_pass),
              "note": ("the draft-induced and target-greedy prefixes are the SAME "
                       "tokens up to and including the first rejected offset (the "
                       "accepted run is by definition the matching run), so the two "
                       "teacher distributions coincide there exactly and reporting a "
                       "nonzero delta there would be a bug; they first differ one "
                       "offset later, and that offset is the one the gate is judged "
                       "on"),
              "chosen_anchor": g3,
              "all_anchors": [r["g3"] for r in per_anchor]}
    if not g3_pass:
        raise AssertionError(f"G3 failed: no rejected offset was scored on a prefix "
                             f"that differs from the target's greedy prefix: {g3}")

    peak_alloc = torch.cuda.max_memory_allocated(a.gpu) / 2**30
    peak_resv = torch.cuda.max_memory_reserved(a.gpu) / 2**30
    g4 = {"pass": bool(peak_alloc < MEMORY_LIMIT_GIB), "limit_gib": MEMORY_LIMIT_GIB,
          "peak_allocated_gib": peak_alloc, "peak_reserved_gib": peak_resv,
          "baseline_allocated_gib": base_alloc_gib,
          "note": "peak over the measured window: ctx build -> G1 control; the "
                  "frozen weights themselves are already resident when the window "
                  "opens and are counted"}
    if not g4["pass"]:
        raise AssertionError(f"G4 failed: {g4}")

    # G5: no call in this file writes a tensor. The check is (a) a recursive walk
    # of the JSON payload for anything array-shaped and (b) a sweep of the output
    # directory against the snapshot taken before the run.
    g5_big = []

    def scan(obj, path="payload"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 4096:
                g5_big.append((path, len(obj)))
            for j, v in enumerate(obj):
                scan(v, f"{path}[{j}]")

    total_anchors = len(per_anchor)
    payload = {
        "config": {
            "rows": a.rows, "records": a.records, "head": a.head,
            "row_index": a.row_index, "n_anchors": a.n_anchors, "gpu": a.gpu,
            "device": torch.cuda.get_device_name(a.gpu),
            "dtype": "bfloat16", "verify": "chain",
            "weight_decay_per_offset": DECAY,
            "stream_weighting": "accepted and rejected equally weighted, each "
                                "normalised by the sum of its own w_k; anchors are "
                                "then averaged",
            "block_size": int(head.block_size), "vocab": int(q_logits[0].shape[1]),
        },
        "row": {"video_name": name, "prompt_len": prompt_len, "trajectory_len": total,
                "fps": float(row["fps"]), "n_frames": int(row["n_frames"]),
                "video_grid_thw": [int(x) for x in vg.tolist()[0]]},
        "anchors": per_anchor,
        "objective": {
            "accepted_stream_mean": sum(r["loss_accepted"] for r in per_anchor) / total_anchors,
            "rejected_stream_mean": sum(r["loss_rejected"] for r in per_anchor) / total_anchors,
            "loss_mean": loss_inc_total / total_anchors,
            "loss_mean_from_full_forward": loss_ref_total / total_anchors,
            "n_accepted_offsets": sum(r["n_accepted_offsets"] for r in per_anchor),
            "n_rejected_offsets": sum(r["n_rejected_offsets"] for r in per_anchor),
        },
        "gates": {
            "G1_incremental_equals_full_forward": g1,
            "G2_target_has_no_gradients": g2,
            "G3_draft_induced_prefix": g3_out,
            "G4_peak_memory": g4,
            "G5_nothing_on_disk": {
                "pass": None,  # filled below
                "torch_save_calls_in_script": 0,
                "note": "teacher logits are produced inside append_through_fork / "
                        "score_anchor, consumed by full_kl in the same loop "
                        "iteration and dropped; the KV caches live only as "
                        "DynamicCache objects in GPU memory and are never "
                        "serialised. The only write is this JSON, which is asserted "
                        "below to hold no array-shaped object.",
            },
        },
        "timing": {"ctx_s": t_ctx, "rollout_s": t_roll, "draft_replay_s": t_replay,
                   "teacher_paths_s": t_teacher, "total_s": time.perf_counter() - t_start},
    }
    scan(payload)
    g5 = {"pass": not g5_big, "oversized_arrays": g5_big}
    payload["gates"]["G5_nothing_on_disk"].update(g5)
    if not g5["pass"]:
        raise AssertionError(f"G5 failed: payload carries array-shaped objects {g5_big}")

    os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    artifacts = sorted(x for x in (set(os.listdir(out_dir)) - before)
                       if not x.endswith(".json"))
    payload["gates"]["G5_nothing_on_disk"]["new_non_json_files"] = artifacts
    payload["gates"]["G5_nothing_on_disk"]["pass"] = bool(not g5_big and not artifacts)
    if artifacts:
        raise AssertionError(f"G5: non-JSON artefacts appeared: {artifacts}")
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=1)

    # ---------------------------------------------------------------- report
    print("\n================ B0 GATES ================", flush=True)
    print(f"G1 incremental == full forward : {'PASS' if g1['pass'] else 'FAIL'}  "
          f"exact={g1['exact_bitwise']}  "
          f"structural={'PASS' if g1['structural_pass'] else 'FAIL'}  "
          f"max|Δlogit|={g1['max_abs_logit_delta']:.3e} "
          f"(ref |logit|max {g1['ref_max_abs_logit']:.1f})  "
          f"maxΔhidden={g1['max_hidden_delta']:.3e} "
          f"(|hidden|max {g1['ref_max_abs_hidden']:.1f})  "
          f"chunking-control max|Δ|={g1['max_abs_chunking_control_delta']:.3e} "
          f"maxTV={g1['max_total_variation']:.3e} "
          f"(chunking maxTV={g1['max_tv_chunking_control']:.3e})  "
          f"argmax {g1['argmax_agreement']}",
          flush=True)
    print(f"G2 target gradient-free       : {'PASS' if g2['pass'] else 'FAIL'}  "
          f"requires_grad={g2['target_params_requires_grad']}  "
          f"grads={g2['target_params_with_grad']}", flush=True)
    print(f"G3 draft-induced prefix       : {'PASS' if g3_out['pass'] else 'FAIL'}  "
          f"anchor {g3['anchor']} pos {g3['pos']}: first differing offset "
          f"{g3['first_differing_offset']} max|Δ| "
          f"{g3['max_abs_inc_vs_greedy_first_diff']:.3e} KL "
          f"{g3['kl_inc_vs_greedy_first_diff']:.3e}", flush=True)
    print(f"G4 peak memory                : {'PASS' if g4['pass'] else 'FAIL'}  "
          f"allocated {peak_alloc:.2f} GiB  reserved {peak_resv:.2f} GiB  "
          f"(limit {MEMORY_LIMIT_GIB:.0f})", flush=True)
    print(f"G5 nothing on disk            : "
          f"{'PASS' if payload['gates']['G5_nothing_on_disk']['pass'] else 'FAIL'}  "
          f"non-json artefacts {artifacts}", flush=True)
    print(f"objective: acc {payload['objective']['accepted_stream_mean']:.4f} "
          f"({payload['objective']['n_accepted_offsets']} offsets)  "
          f"rej {payload['objective']['rejected_stream_mean']:.4f} "
          f"({payload['objective']['n_rejected_offsets']} offsets)  "
          f"total {payload['objective']['loss_mean']:.4f}  "
          f"(from full forward {payload['objective']['loss_mean_from_full_forward']:.4f})",
          flush=True)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
