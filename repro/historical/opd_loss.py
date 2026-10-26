#!/usr/bin/env python3
"""Draft-OPD replay loss for the GLANCE block head.

Ported from the text/multimodal OPD work in
`/home/mcy/Projects/Draft-OPD/opd_multimodal/opd_train_vlm.py::replay_loss`,
whose two KL estimators are verbatim copies of the official implementation
(`refs/Draft-OPD/verl/verl/trainer/distillation/losses.py::_local_bernoulli_forward_kl`
and `verl/trainer/ppo/core_algos.py` k3). What changed here is only the plumbing:

* **The student forward is not re-written.** It reuses `train_head.forward_anchors`,
  which `train_head.verify()` already checks against the released forward
  (fp32 max|Δ| ~3.5e-6) and which the training pre-flight re-checks. Writing a
  second forward is precisely how this project once got a convention bug that
  *reversed* the OPD-vs-SFT conclusion.
* **Anchors are batched.** The reference replays one anchor per forward because
  its GQA path is cheap per call; here all of a step's anchors go through one
  pass, exactly as `train_head.sequence_loss` does. `build_anchor_mask` gives
  each block only its own `ctx[:anchor]` plus itself, which is the decode
  condition, so batching is not an approximation.

The recipe follows the newest report (`C组报告_2026-09-23.md`), which revised an
earlier claim: **tree rollout is the decisive variable; the reverse-KL term's
marginal contribution is ~0** (alpha 0.0 vs 0.1: +0.0046, t=+0.29), so the
default here is alpha_rev=0.1 with 0.0 as the clean mechanism arm.

Loss, per supervised position (offset `off`, 1-based):
  * `off <= n_acc`  -> Bernoulli forward KL, weight `decay**(off-1)`
  * `off == n_acc+1` -> k3 reverse KL, weight `alpha_rev * decay**(off-1)`
  * `off >  n_acc+1` -> not supervised (first-reject-only)
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))

from historical import train_head as th

__all__ = ["opd_blocks", "k3_reverse_kl", "bernoulli_forward_kl",
           "opd_replay_loss", "teacher_rows", "lookup_teacher"]


def opd_blocks(seq_ids, anchors, props, block_size, mask_id, device):
    """Blocks filled with the DRAFTER's proposals, not the target's tokens.

    This is the whole difference from SFT. SFT builds a block from the target's
    own continuation with a reveal boundary (`train_head.build_blocks`); OPD
    builds it from what the drafter actually proposed during the tree rollout,
    so the head is trained on the states it really visits.

    Slot 0 is the committed token `seq_ids[pos]` (the root, never supervised).
    Slots 1..len(prop) hold the proposals; the rest are MASK, matching the
    decode layout for a block shorter than B.

    Returns:
        blocks ``(N, B)``, super_mask ``(N, B)`` float, offsets ``(N, B)`` int.
    """
    n = len(anchors)
    offsets = torch.arange(block_size, device=device).unsqueeze(0)
    index = torch.tensor(anchors, device=device).unsqueeze(1) + offsets
    in_bounds = index <= (seq_ids.shape[0] - 1)

    blocks = torch.full((n, block_size), mask_id, dtype=torch.long, device=device)
    super_mask = torch.zeros((n, block_size), device=device)
    for i, prop in enumerate(props):
        if prop:
            k = min(len(prop), block_size - 1)
            blocks[i, 1:1 + k] = torch.tensor(prop[:k], dtype=torch.long, device=device)
            super_mask[i, 1:1 + k] = 1.0
    # Slot 0 is the anchor's own token. It is never supervised, but it is the
    # RoPE/attention anchor of the block, so it must be the real committed token.
    blocks[:, 0] = seq_ids[torch.tensor(anchors, device=device)]
    super_mask = super_mask * in_bounds.to(super_mask.dtype)
    super_mask[:, 0] = 0.0
    return blocks, super_mask, offsets.expand(n, block_size)


def k3_reverse_kl(student_log: torch.Tensor, teacher_log: torch.Tensor):
    """Reverse KL(q||p) via the k3 estimator -- verbatim from the official code.

        kl   = log(p/q)
        kld  = p/q - log(p/q) - 1  =  E_q[p/q - log(p/q) - 1]

    A single-sample unbiased estimate: it needs only the logprob of the sampled
    token, not the full-vocabulary distributions. This is the form Draft-OPD
    uses at the first rejected position.
    """
    kl = (teacher_log - student_log).clamp(-20.0, 20.0)
    ratio = torch.exp(kl)
    return (ratio - kl - 1.0).clamp(-10.0, 10.0)


def bernoulli_forward_kl(student_log: torch.Tensor, teacher_log: torch.Tensor):
    """Forward KL on the accepted positions -- verbatim from the official code.

    Each position is treated as a binary distribution over
    {the sampled token, everything else}:
        p*(log p - log q) + (1-p)*(log(1-p) - log(1-q))
    """
    student_log = student_log.float().clamp_min(-80.0)
    teacher_log = teacher_log.float().clamp_min(-80.0)
    eps = torch.finfo(student_log.dtype).eps
    max_log = math.log1p(-eps)
    student_log = student_log.clamp(max=max_log)
    teacher_log = teacher_log.clamp(max=max_log)
    sp, tp = student_log.exp(), teacher_log.exp()
    return tp * (teacher_log - student_log) + (1.0 - tp) * (
        torch.log1p(-tp) - torch.log1p(-sp))


def teacher_rows(pos, off):
    """The official teacher row index for a student offset.

    Student row `off` at block start `pos` predicts `seq[pos + off]`; the
    target's next-token distribution for that position lives at row
    `pos + off - 1` (because `topk_ids[r]` predicts `seq[r+1]`).

    Extracted and exported because *this single subtraction* is where this
    project's historical convention bug lived -- an off-by-one here reversed the
    OPD-vs-SFT conclusion (see `全局检查_块头anchor约定_2026-09-21.md`). It is
    unit-tested with sentinel data in `test_opd_convention.py`.
    """
    return pos + off - 1


def lookup_teacher(topk_ids, topk_val, rows, toks):
    """Gather `log p(token)` from the stored top-k, vectorised.

    Returns ``(logp, found)``. Where the token is outside the stored top-k the
    true log p is unknown but bounded above by the remaining probability mass,
    ``log(1 - sum top-k probs)``; callers decide whether to keep those.
    """
    ids_r = topk_ids[rows]
    val_r = topk_val[rows].float()
    hit = ids_r == toks.unsqueeze(1)
    found = hit.any(dim=1)
    hit_idx = hit.float().argmax(dim=1)
    logp_hit = val_r.gather(1, hit_idx.unsqueeze(1)).squeeze(1)
    mass = (1.0 - val_r.exp().sum(dim=1)).clamp_min(1e-30)
    return torch.where(found, logp_hit, mass.log()), found


def opd_replay_loss(head, target, ctx, seq_ids, anchors, props, n_accs,
                    topk_ids, topk_val, embed_weight, block_size, mask_id,
                    device, alpha_rev: float = 0.1, beta_fwd: float = 1.0,
                    decay: float = 0.9, drop_inexact_teacher: bool = True,
                    objective: str = "bernoulli", reject_mode: str = "first"):
    """The OPD replay loss over one step's anchors.

    Args:
        ctx: concatenated target hidden for the whole sequence, ``(1, S, L*H)``.
        seq_ids: the committed token ids ``(S,)`` (the target's greedy tokens --
            GLANCE is lossless, so the drafter's rollout commits exactly these).
        anchors: block starts ``(pos,)``.
        props: per anchor, the drafter's proposed tokens.
        n_accs: per anchor, how many proposals the target accepted.
        topk_ids/topk_val: ``(S-1, K)`` target top-k. Row ``r`` is the target's
            next-token distribution at position ``r``, i.e. it predicts
            ``seq_ids[r+1]`` -- the official convention, asserted in
            `test_opd_convention.py`.

    Returns:
        ``(loss, n_positions, l_fwd, l_rev, n_inexact)``; ``loss`` is None when
        no position survives.
    """
    keep = [i for i, p in enumerate(props) if p]
    if not keep:
        return None, 0, 0.0, 0.0, 0
    anchors = [anchors[i] for i in keep]
    props = [props[i] for i in keep]
    n_accs = [n_accs[i] for i in keep]
    n = len(anchors)

    blocks, super_mask, offsets = opd_blocks(
        seq_ids, anchors, props, block_size, mask_id, device)
    flat_super = super_mask.reshape(-1).bool()
    if not flat_super.any():
        return None, 0, 0.0, 0.0, 0

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

    # Which position is which: anchor index, offset, and whether it is in the
    # accepted segment or is the first rejected position.
    anch_idx = (torch.arange(length, device=device) // block_size)
    off_all = offsets.reshape(-1)
    prop_tok_all = blocks.reshape(-1)
    pos_all = torch.tensor(anchors, device=device)[anch_idx]

    off_s = off_all[flat_super]
    pos_s = pos_all[flat_super]
    tok_s = prop_tok_all[flat_super]
    n_acc_s = torch.tensor(n_accs, device=device)[anch_idx][flat_super]

    is_fwd = off_s <= n_acc_s
    if reject_mode == "suffix":
        # Paper Eq 9: I_rej = {(m,k) : r_m < k <= K}, i.e. the FIRST rejected
        # token AND the whole rejected suffix, weighted by gamma^(k-1) and
        # normalised by Z = sum of weights. The "first" mode keeps only the
        # first rejected position, which is what this port shipped originally --
        # and which is close to the paper's own WORSE ablation (all-forward KL,
        # Table 3: 6.35 vs 6.57 tau), because a small alpha_rev on a single
        # position deletes almost all of the rejection signal.
        is_rev = off_s > n_acc_s
    else:
        is_rev = off_s == (n_acc_s + 1)
    sel = is_fwd | is_rev
    if not sel.any():
        return None, 0, 0.0, 0.0, 0
    off_s, pos_s, tok_s = off_s[sel], pos_s[sel], tok_s[sel]
    is_fwd, is_rev = is_fwd[sel], is_rev[sel]
    logq_sel = logq_all[sel]          # (P, V) kept for the top-k objective

    # --- teacher lookup -----------------------------------------------------
    rows = teacher_rows(pos_s, off_s)
    in_range = (rows >= 0) & (rows < topk_ids.shape[0])
    rows = rows.clamp(0, topk_ids.shape[0] - 1)
    logp, found = lookup_teacher(topk_ids, topk_val, rows, tok_s)

    logq = logq_all.gather(1, tok_s.unsqueeze(1)).squeeze(1)
    w = decay ** (off_s - 1).float()

    n_inexact = 0
    # `found` is False when the sampled token sits outside the stored top-k. The
    # k3 reverse estimator needs that token's own log p (only the tail aggregate
    # is known), so those positions remain droppable; the forward-KL objectives
    # do not have that problem and would be fine keeping them.
    if objective == "ce":
        # CE needs only the teacher's ARGMAX, which the pool always carries as
        # topk_ids[:, 0]. `found` asks whether the SAMPLED token is inside the
        # stored top-k, which matters only for the KL estimators' tail handling,
        # so filtering on it here would throw away usable labels for nothing.
        usable = in_range
    else:
        usable = in_range & (found | (not drop_inexact_teacher))
    if not usable.any():
        return None, 0, 0.0, 0.0, n_inexact
    # Report only positions actually DISCARDED, over all selected ones, so the
    # fraction means "share of selected positions thrown away" and cannot exceed
    # 1. The earlier form divided the not-found count by the kept count, which
    # printed ratios above 1 whenever the drop flag was on.
    n_inexact = int(sel.sum()) - int(usable.sum())
    logq, logp, w = logq[usable], logp[usable], w[usable]
    is_fwd, is_rev = is_fwd[usable], is_rev[usable]

    def _wmean(terms, weights):
        if terms.numel() == 0:
            return torch.zeros((), device=device)
        return (terms * weights).sum() / weights.sum().clamp_min(1e-8)

    if objective == "bernoulli":
        # Official Draft-OPD: a binary {sampled token, everything else} forward KL.
        l_fwd = _wmean(bernoulli_forward_kl(logq[is_fwd], logp[is_fwd]), w[is_fwd])
    elif objective == "topk":
        # Forward KL over the teacher's top-k support instead of a binary one.
        #
        # Why this exists. The Bernoulli form only ever looks at the sampled
        # token, so it constrains the mass on one candidate and leaves the
        # RELATIVE order of the others free. But the tree is scored as a product
        # of per-offset marginals, so acceptance depends on how the candidates
        # are ranked, not just on whether the top-1 is right. This objective
        # matches the whole top-k ordering, which is the quantity the tree
        # actually spends.
        #
        # Both sides are renormalised over the same K candidates. The teacher's
        # true tail beyond top-k is unknown (only log(1 - sum) is bounded), so a
        # truncation is unavoidable here; it is the standard top-k distillation
        # convention and is reported as such.
        rows_u = rows[usable]
        ids_u = topk_ids[rows_u]                     # (P, K)
        tval = topk_val[rows_u].float()              # (P, K) log p
        tprob = torch.softmax(tval, dim=-1)
        slog = logq_sel[usable].gather(1, ids_u)     # (P, K) log q
        sprob = torch.softmax(slog, dim=-1)
        kl = (tprob * (tprob.clamp_min(1e-12).log()
                       - sprob.clamp_min(1e-12).log())).sum(dim=-1)
        l_fwd = _wmean(kl[is_fwd], w[is_fwd])
    elif objective == "topk_tail":
        # Forward KL between the coarse-grained distributions (teacher top-k plus
        # one tail bucket), not a renormalised top-k.
        #
        # Renormalising over the top-k (the plain "topk" objective) throws away
        # the teacher's tail mass and the student's, which is a real distortion:
        # 11-22% of supervised positions have the sampled token outside the stored
        # top-64. Adding a tail bucket keeps both tails as atoms, so the
        # comparison is over the whole distribution up to the coarsening. It is
        # still not the paper's exact full-vocabulary KL (Eq 10), which needs
        # logits this pool does not store, and that limitation is recorded here
        # rather than hidden.
        rows_u = rows[usable]
        ids_u = topk_ids[rows_u]
        tval = topk_val[rows_u].float()
        slog = logq_sel[usable].gather(1, ids_u)
        p_tail = (1.0 - tval.exp().sum(-1)).clamp_min(1e-12)
        q_tail = (1.0 - slog.exp().sum(-1)).clamp_min(1e-12)
        p = torch.cat([tval.exp(), p_tail.unsqueeze(1)], dim=-1)
        q = torch.cat([slog.exp(), q_tail.unsqueeze(1)], dim=-1)
        p = p / p.sum(-1, keepdim=True)
        q = q / q.sum(-1, keepdim=True)
        kl = (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(-1)
        l_fwd = _wmean(kl[is_fwd], w[is_fwd])
    elif objective == "ce":
        # Cross-entropy to the teacher's argmax, on the drafter's OWN states.
        #
        # Why this arm exists. Two independent measurements point the same way.
        # Here, the best tau came from the objective that is NOT a KL: the
        # continue-SFT control (CE on target-forced blocks) beat both KL arms.
        # And in the literature, arXiv 2510.05421 (Draft, Verify & Improve)
        # reports that a reward-masked CE plus policy-gradient schedule beats
        # KL-only online distillation. So this keeps OPD's contribution -- the
        # on-policy state distribution from the tree rollout -- and swaps only
        # the loss for CE against the token the target actually committed, which
        # is exactly the token the verifier required.
        lab = topk_ids[rows[usable]][:, 0]                 # teacher argmax
        ce = -logq_sel[usable].gather(1, lab.unsqueeze(1)).squeeze(1)
        l_fwd = _wmean(ce[is_fwd], w[is_fwd])
        l_rev = _wmean(ce[is_rev], w[is_rev])
        loss = beta_fwd * l_fwd + alpha_rev * l_rev
        n_inexact_ret = int(usable.sum())
        return loss, n_inexact_ret, float(l_fwd.detach()), float(l_rev.detach()), n_inexact
    else:
        raise ValueError(
            f"objective must be 'bernoulli', 'topk', 'topk_tail' or 'ce', "
            f"got {objective!r}")

    l_rev = _wmean(k3_reverse_kl(logq[is_rev], logp[is_rev]), w[is_rev])
    loss = beta_fwd * l_fwd + alpha_rev * l_rev
    return loss, int(usable.sum()), float(l_fwd.detach()), float(l_rev.detach()), n_inexact
