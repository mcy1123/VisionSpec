#!/usr/bin/env python3
"""The image-side OPD recipe, applied to the GLANCE block head on video.

Why a separate file instead of a flag inside `opd_loss.py`. `opd_loss.py` is the
verified image-side port and every image number in this project depends on it, so
it is not edited. This file imports its *building blocks* -- `opd_blocks`,
`teacher_rows`, `lookup_teacher`, `bernoulli_forward_kl`, `k3_reverse_kl` -- and
`train_head.forward_anchors` / `build_anchor_mask` for the student pass. There is
no second implementation of the student forward or of the offset convention: the
one subtraction that historically reversed an OPD-vs-SFT conclusion
(`teacher_rows`) is literally the same function object, and
`test_opd_loss_video.py` pins the two losses to each other on synthetic data.

The recipe (image side, `opd_train_vlm.py`, as prescribed for this task):

  * `--accept-target bernoulli`   accepted offsets -> Bernoulli forward KL on the
    token the tree committed (weight `decay**(off-1)`).
  * `--reject-target topk`        the FIRST rejected offset -> full forward KL
    over the teacher's stored top-k support, renormalised on both sides. The
    rejected stream is therefore part of the FORWARD loss, which is what makes
    `--alpha-rev` a no-op here (the reverse-KL stream is empty by construction).
  * no drop (`drop_inexact_teacher=False`)  a proposed token outside the stored
    top-k keeps its position, with the teacher log-prob replaced by the upper
    bound `log(1 - sum top-k)` -- which is what the image trainer does with the
    flag off.
  * `first_token_only` (official default): offsets after the first rejection are
    not supervised.
  * `anchor-mode front`: the anchors themselves come from the drafter's
    sequential tree-verified rollout (the pool collector), so this file only
    samples a subset of them; it never re-positions anchors on the sequence.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))

from historical import opd_loss
from historical import train_head as th

__all__ = ["opd_replay_loss_video"]


def opd_replay_loss_video(head, target, ctx, seq_ids, anchors, props, n_accs,
                          topk_ids, topk_val, embed_weight, block_size, mask_id,
                          device, beta_fwd: float = 1.0, decay: float = 0.9,
                          reject_target: str = "topk",
                          drop_inexact_teacher: bool = False,
                          alpha_rev: float = 0.0, return_parts: bool = False,
                          return_streams: bool = False):
    """One OPD step's loss over the sampled anchors.

    Args mirror `opd_loss.opd_replay_loss`. `reject_target`:
      * ``"topk"``  -- image-side recipe: first rejected offset gets the teacher's
        top-k forward KL and joins the forward stream (alpha_rev provably no-op).
      * ``"draft"`` -- the GLANCE port's original form: k3 reverse KL on the
        proposed token, scaled by alpha_rev. Kept because it is what makes this
        function checkable against `opd_loss.opd_replay_loss` bit for bit.

    Returns ``(loss, n_positions, l_fwd, l_rev, n_inexact)``; `loss` is None when
    no position survives. `return_streams=True` additionally returns the two
    tensors whose sum is ``beta_fwd * l_fwd`` (accepted / first-rejection), for
    exact per-stream gradient attribution.
    """
    if reject_target not in ("topk", "draft"):
        raise ValueError(f"reject_target must be 'topk' or 'draft', got {reject_target!r}")

    def _none(n_inexact=0):
        """Every early exit returns the same arity, so a caller unpacking the
        parts dict can never see a short tuple (which would look like a crash
        inside the loss instead of an empty step)."""
        if return_streams:
            z = torch.zeros((), device=device)
            return (None, 0, z, z, n_inexact,
                    {"contrib_acc": z, "contrib_rej": z, "contrib_k3": z,
                     "n_fwd": 0, "n_topk_rej": 0, "n_k3_rej": 0})
        if return_parts:
            return (None, 0, 0.0, 0.0, n_inexact,
                    {"n_fwd": 0, "n_topk_rej": 0, "n_k3_rej": 0})
        return None, 0, 0.0, 0.0, n_inexact

    keep = [i for i, p in enumerate(props) if p]
    if not keep:
        return _none()
    anchors = [anchors[i] for i in keep]
    props = [props[i] for i in keep]
    n_accs = [n_accs[i] for i in keep]
    n = len(anchors)

    blocks, super_mask, offsets = opd_loss.opd_blocks(
        seq_ids, anchors, props, block_size, mask_id, device)
    flat_super = super_mask.reshape(-1).bool()
    if not flat_super.any():
        return _none()

    length = n * block_size
    embed = F.embedding(blocks, embed_weight).reshape(1, length, -1)
    ctx_len = ctx.shape[1]
    ctx_positions = torch.arange(ctx_len, device=device).unsqueeze(0)
    q_positions = (torch.tensor(anchors, device=device).unsqueeze(1)
                   + torch.arange(block_size, device=device).unsqueeze(0)).reshape(1, length)
    attn_mask = th.build_anchor_mask(anchors, block_size, ctx_len, device, embed.dtype)

    hidden = th.forward_anchors(head, ctx, embed, ctx_positions, q_positions, attn_mask)
    hidden = hidden.reshape(1, length, -1)[0]
    logits = head.compute_logits(hidden[flat_super][None], target.lm_head)[0]
    logq_all = F.log_softmax(logits.float(), dim=-1)

    anch_idx = torch.arange(length, device=device) // block_size
    off_all = offsets.reshape(-1)
    prop_tok_all = blocks.reshape(-1)
    pos_all = torch.tensor(anchors, device=device)[anch_idx]

    off_s = off_all[flat_super]
    pos_s = pos_all[flat_super]
    tok_s = prop_tok_all[flat_super]
    n_acc_s = torch.tensor(n_accs, device=device)[anch_idx][flat_super]

    is_fwd = off_s <= n_acc_s
    # Official first-reject-only: the first rejected offset is n_acc+1; later
    # rejected offsets are built on a prefix the verifier already falsified.
    is_first_rej = off_s == (n_acc_s + 1)
    if reject_target == "topk":
        sel = is_fwd | is_first_rej
        is_topk_rej = is_first_rej
        is_k3_rej = torch.zeros_like(is_fwd)
    else:
        sel = is_fwd | is_first_rej
        is_topk_rej = torch.zeros_like(is_fwd)
        is_k3_rej = is_first_rej
    if not sel.any():
        return _none()
    off_s, pos_s, tok_s = off_s[sel], pos_s[sel], tok_s[sel]
    is_fwd, is_topk_rej, is_k3_rej = is_fwd[sel], is_topk_rej[sel], is_k3_rej[sel]
    logq_sel = logq_all[sel]

    rows = opd_loss.teacher_rows(pos_s, off_s)
    in_range = (rows >= 0) & (rows < topk_ids.shape[0])
    rows = rows.clamp(0, topk_ids.shape[0] - 1)
    logp, found = opd_loss.lookup_teacher(topk_ids, topk_val, rows, tok_s)
    logq = logq_all.gather(1, tok_s.unsqueeze(1)).squeeze(1)
    w = decay ** (off_s - 1).float()

    # No drop (image-side flag off): a position whose proposed token is outside
    # the stored top-k stays, with the bound log(1 - sum top-k). Only rows that
    # fall outside the sequence are unusable.
    usable = in_range
    if drop_inexact_teacher:
        usable = usable & found
    n_inexact = int((in_range & ~found).sum())
    if not usable.any():
        return _none(n_inexact)
    rows_u = rows[usable]
    logq_u, logp_u, w_u = logq[usable], logp[usable], w[usable]
    is_fwd_u = is_fwd[usable]
    is_topk_rej_u = is_topk_rej[usable]
    is_k3_rej_u = is_k3_rej[usable]
    logq_sel_u = logq_sel[usable]

    def _wmean(terms, weights):
        if terms.numel() == 0:
            return torch.zeros((), device=device)
        return (terms * weights).sum() / weights.sum().clamp_min(1e-8)

    # ---- forward stream: bernoulli on accepted + top-k KL on first rejection --
    # The two groups are kept separate so the diagnostic harness can attribute the
    # gradient to each stream exactly: their additive contributions sum to l_fwd.
    t_acc = w_acc = t_rej = w_rej = None
    if is_fwd_u.any():
        t_acc = opd_loss.bernoulli_forward_kl(logq_u[is_fwd_u], logp_u[is_fwd_u])
        w_acc = w_u[is_fwd_u]
    if is_topk_rej_u.any():
        ids_r = topk_ids[rows_u[is_topk_rej_u]]           # (P, K) teacher support
        tval = topk_val[rows_u[is_topk_rej_u]].float()
        tprob = torch.softmax(tval, dim=-1)
        slog = logq_sel_u[is_topk_rej_u].gather(1, ids_r)
        sprob = torch.softmax(slog, dim=-1)
        t_rej = (tprob * (tprob.clamp_min(1e-12).log()
                          - sprob.clamp_min(1e-12).log())).sum(dim=-1)
        w_rej = w_u[is_topk_rej_u]
    parts_list = [x for x in (t_acc, t_rej) if x is not None]
    w_list = [x for x in (w_acc, w_rej) if x is not None]
    if parts_list:
        t_all = torch.cat(parts_list)
        w_all = torch.cat(w_list)
        denom = w_all.sum().clamp_min(1e-8)
        l_fwd = (t_all * w_all).sum() / denom
        contrib_acc = ((t_acc * w_acc).sum() / denom) if t_acc is not None \
            else torch.zeros((), device=device)
        contrib_rej = ((t_rej * w_rej).sum() / denom) if t_rej is not None \
            else torch.zeros((), device=device)
    else:
        l_fwd = torch.zeros((), device=device)
        contrib_acc = contrib_rej = torch.zeros((), device=device)

    l_rev = _wmean(opd_loss.k3_reverse_kl(logq_u[is_k3_rej_u],
                                          logp_u[is_k3_rej_u]), w_u[is_k3_rej_u])
    loss = beta_fwd * l_fwd + alpha_rev * l_rev
    n_pos = int(usable.sum())
    if return_streams:
        return (loss, n_pos, l_fwd, l_rev, n_inexact,
                {"contrib_acc": beta_fwd * contrib_acc,
                 "contrib_rej": beta_fwd * contrib_rej,
                 "contrib_k3": alpha_rev * l_rev,
                 "n_fwd": int(is_fwd_u.sum()), "n_topk_rej": int(is_topk_rej_u.sum()),
                 "n_k3_rej": int(is_k3_rej_u.sum())})
    if return_parts:
        return (loss, n_pos, float(l_fwd.detach()), float(l_rev.detach()), n_inexact,
                {"n_fwd": int(is_fwd_u.sum()), "n_topk_rej": int(is_topk_rej_u.sum()),
                 "n_k3_rej": int(is_k3_rej_u.sum())})
    return loss, n_pos, float(l_fwd.detach()), float(l_rev.detach()), n_inexact
