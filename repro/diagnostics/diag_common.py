#!/usr/bin/env python3
"""Shared machinery for the visual-adaptation diagnosis (see 视觉适配诊断_日期.md).

The question this file exists to answer is narrow: the draft head receives the
target's fused hidden states through one learned projection

    target_hidden = hidden_norm(fc(cat(hidden_states[1,9,17,25,33])))

and that projection is the ONLY place where vision enters the drafter. This
module builds four alternative KV sources on top of ONE frozen trained head, so
the four arms differ in exactly one thing (what the vision-language context is)
and in nothing else -- not the head weights, not the tree, not the data.

  A  current MLP projection   hidden_norm(fc(cat(layers)))                [baseline]
  B  raw visual tokens        A, but the KV rows at visual positions are the
                              target's own post-merger visual embeddings
                              (hidden_states[0]) instead of the deep-layer ones
  C  text-only KV             A, but the visual positions are zeroed before fc,
                              so the drafter's context carries no visual content
  D  final layer, no MLP      the target's last-layer hidden state goes straight
                              to k_proj/v_proj; fc and hidden_norm are skipped

Nothing here edits the model code. `fc`, `hidden_norm` and the layers are used
exactly as the checkpoint defines them; the arms only decide what tensor is fed
to them.

The mechanism is a single patched call site. `glance.decode.generate` produces
the context with

    fused = context_feature(out.hidden_states, head.target_layer_ids)

at the prefill and, once per round, gathers it down to the committed path with

    fused = context_feature(out.hidden_states, head.target_layer_ids)[0][keep][None]

Both the prefill tensor and the per-round gather therefore go through one
function. This module replaces that function with a *wider* return value: the
five concatenated layers, plus the embedding output, plus the final layer, plus a
one-channel visual-position flag. Because the gather indexes dim 1 (positions),
every extra channel is gathered consistently with the positions it belongs to --
so the flag and the raw features can never drift out of alignment with the rows
they annotate. The patched head forward slices the channels back out and applies
the arm. That is the whole trick, and it is why no attention mask, position
bookkeeping or cache logic has to be replicated.

IMPORTANT interpretation limit, stated here because it governs every number the
diagnosis reports: arms B/C/D are applied to a head that was TRAINED under arm A.
They are therefore out-of-distribution for the head, and they measure how much
the trained head *depends on* the arm-A representation, not the best accuracy
attainable with that KV source. `视觉适配诊断_日期.md` reports this as the
primary confound.
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
import types

import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common  # noqa: F401  (wires REPO + DFLASH_PKG onto sys.path)

ARMS = ("A", "B", "C", "D", "Dn")

ARM_DOC = {
    "A": "current MLP projection (hidden_norm(fc(fused)))",
    "B": "raw visual tokens at visual positions (bypass deep layers)",
    "C": "no visual information (visual KV rows zeroed)",
    "D": "target final-layer hidden, fc/hidden_norm skipped",
    "Dn": "target final-layer hidden, RMS-normalised, fc skipped",
}


def _target_cfg(path=None):
    import json
    path = path or common.TARGET_PATH
    return json.load(open(os.path.join(path, "config.json")))


class WideContext:
    """Patches `glance.decode.context_feature` to emit four channels.

    Channel layout on the last dim, with D = hidden_size and n = len(layer_ids):

        [0        : n*D       ]  cat(hidden_states[i+1] for i in layer_ids)  (arm A/B/C input)
        [n*D      : (n+1)*D   ]  hidden_states[0]   -- post-merger visual/embedding output
        [(n+1)*D  : (n+2)*D   ]  hidden_states[-1]  -- final layer
        [(n+2)*D  : (n+2)*D+1 ]  visual-position flag (1.0 at image/video tokens)

    `ids` is refreshed by a forward pre-hook on the target, because every call
    that produces hidden states in `generate` passes the tokens it is running on
    as input_ids -- the prefill passes the prompt, the verify pass passes the
    packed tree -- so the recorded ids always match the rows being returned.
    """

    def __init__(self, target, target_path=None, record_feats=False):
        import glance.decode as dec

        cfg = _target_cfg(target_path)
        self.video_id = int(cfg["video_token_id"])
        self.image_id = int(cfg["image_token_id"])
        self.hidden = int(cfg["text_config"]["hidden_size"])
        self.ids = None
        self.record_feats = record_feats
        self.feats = None

        self._dec = dec
        self._orig = dec.context_feature
        holder = self

        def wide_context_feature(hidden_states, layer_ids):
            D = holder.hidden
            n = len(layer_ids)
            base = torch.cat([hidden_states[i + 1] for i in layer_ids], dim=-1)
            raw = hidden_states[0]
            last = hidden_states[-1]
            t = base.shape[1]
            ids = holder.ids
            if ids is None:
                raise RuntimeError("target ids not recorded; the pre-hook did not fire")
            vis = ((ids[:, :t] == holder.video_id) | (ids[:, :t] == holder.image_id))
            vis = vis.to(base.dtype).unsqueeze(-1)
            if holder.record_feats:
                # Capture the prefill (the only call whose rows include visual
                # tokens); later calls are single committed tokens.
                holder.feats = {
                    "base": base.detach().float().cpu(),
                    "raw": raw.detach().float().cpu(),
                    "last": last.detach().float().cpu(),
                    "vis": vis.squeeze(-1).bool().cpu(),
                    "n_layers": n,
                }
            return torch.cat([base, raw, last, vis], dim=-1)

        dec.context_feature = wide_context_feature
        self._hook = target.register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def _pre_hook(self, module, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is not None:
            self.ids = ids

    def close(self):
        self._dec.context_feature = self._orig
        self._hook.remove()


def _split_wide(target_hidden, head):
    D = head.config.hidden_size
    n = len(head.target_layer_ids)
    base = target_hidden[..., : n * D]
    raw = target_hidden[..., n * D:(n + 1) * D]
    last = target_hidden[..., (n + 1) * D:(n + 2) * D]
    vis = target_hidden[..., (n + 2) * D:(n + 2) * D + 1] > 0.5
    return base, raw, last, vis


def bind_arm(head, arm, stats=None):
    """Replace `head.forward` with the original computation under a chosen arm.

    The no-MLP arm (D) changes the width of what reaches k_proj/v_proj, which is
    why the arm lives at the forward and not at `context_feature`: `fc` maps
    5*D -> D, so "skip the projection" cannot be expressed as an edit of its
    input without still running it.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    D = head.config.hidden_size

    def forward(self, position_ids=None, attention_mask=None, noise_embedding=None,
                target_hidden=None, past_key_values=None, use_cache=False, **kwargs):
        if target_hidden is None:
            raise ValueError("arm forward requires target_hidden")
        base, raw, last, vis = _split_wide(target_hidden, self)
        if arm == "A":
            th = self.hidden_norm(self.fc(base))
        elif arm == "B":
            th = self.hidden_norm(self.fc(base))
            th = torch.where(vis, raw.to(th.dtype), th)
        elif arm == "C":
            th = self.hidden_norm(self.fc(base.masked_fill(vis, 0.0)))
        elif arm == "D":
            th = last.to(self.norm.weight.dtype)
        else:  # Dn -- same information as D, but scale-controlled so the arm
            # measures the loss of `fc` alone rather than a broken activation
            # scale. Without this control, D's collapse cannot be attributed to
            # information loss: `hidden_states[-1]` has a far larger RMS than the
            # normalised fc output the head was trained on, and v_proj has no
            # normalisation after it.
            th = self.hidden_norm(last.to(base.dtype))
        if stats is not None:
            with torch.no_grad():
                v = vis[0, :, 0]
                stats.append({
                    "rows": int(th.shape[1]),
                    "n_visual": int(v.sum()),
                    "th_absmean_vis": float(th[0][v].abs().mean()) if v.any() else 0.0,
                    "th_absmean_txt": float(th[0][~v].abs().mean()) if (~v).any() else 0.0,
                    "vis_frac": float(v.float().mean()) if v.numel() else 0.0,
                })
        hidden_states = noise_embedding
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=th,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    head.forward = types.MethodType(forward, head)
    return head


# ---------------------------------------------------------------------------
# Stage 3: draft attention over the KV context
# ---------------------------------------------------------------------------

class AttnCapture:
    """Per-layer statistics of the draft block's attention over its KV context.

    A forward hook on every draft attention module recomputes q and k from the
    module's own inputs and reads the softmax weights off them. It is a pure
    observer: the module's real forward (sdpa) still produces the output the
    decode loop consumes, so capturing cannot perturb tau. The hook records
    aggregates only -- never the weights themselves -- because a full (heads x
    block x context) tensor per layer is far too large to carry around, and the
    diagnosis only needs mass and entropy.

    The context is `k_ctx` (the projected target hidden, length ctx_len) followed
    by `k_noise` (the block itself). Only the k_ctx part is scored; the block's
    own self-attention is not what "does the drafter look at the image" means.
    """

    def __init__(self, head, vis_ctx=None):
        from dflash.model import Qwen3DFlashAttention

        self.head = head
        self.rec = []
        self.enabled = False
        self.max_rounds = 1
        self._n_calls = 0
        self._mod = Qwen3DFlashAttention
        # visual-position mask over the round-0 context rows (the prompt). Set
        # per row before the decode call.
        self.vis_ctx = vis_ctx
        self._handles = [layer.self_attn.register_forward_hook(
            self._hook, with_kwargs=True) for layer in head.layers]

    def _hook(self, module, args, kwargs, output):
        if not self.enabled:
            return
        if self._n_calls >= self.max_rounds * len(self.head.layers):
            return
        hs = kwargs.get("hidden_states")
        th = kwargs.get("target_hidden")
        pe = kwargs.get("position_embeddings")
        if hs is None or th is None or pe is None:
            return
        self._n_calls += 1
        with torch.no_grad():
            bsz, q_len = hs.shape[:-1]
            ctx_len = th.shape[1]
            q = module.q_proj(hs).view(bsz, q_len, -1, module.head_dim)
            q = module.q_norm(q).transpose(1, 2)
            k_ctx = module.k_proj(th)
            k_noise = module.k_proj(hs)
            k = torch.cat([k_ctx, k_noise], dim=1).view(
                bsz, ctx_len + q_len, -1, module.head_dim)
            k = module.k_norm(k).transpose(1, 2)
            # Grouped-query attention: q has num_attention_heads heads, k/v have
            # num_key_value_heads. Broadcast the KV heads the way the real
            # attention does, or the score matmul cannot line up.
            n_rep = q.shape[1] // k.shape[1]
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=1)
            cos, sin = pe
            from dflash.model import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # (bsz, heads, q_len, ctx+q_len)
            logits = (q @ k.transpose(-1, -2)) * module.scaling
            w = torch.softmax(logits.float(), dim=-1)
            w_ctx = w[..., :ctx_len]

            if self.vis_ctx is None or self.vis_ctx.numel() != ctx_len:
                raise RuntimeError(
                    f"attention capture needs a visual mask of length {ctx_len}, "
                    f"got {None if self.vis_ctx is None else self.vis_ctx.numel()}")
            st = attn_stats_from_weights(
                w_ctx, self.vis_ctx, context_thirds(ctx_len))
            st["layer"] = int(module.layer_idx)
            st["q_len"] = int(q_len)
            self.rec.append(st)

    def take(self):
        out = self.rec
        self.rec = []
        return out

    def close(self):
        for h in self._handles:
            h.remove()


def attn_stats_from_weights(w_ctx, vis_ctx, thirds):
    """Reduce one layer's (bsz, heads, q_len, ctx_len) weights to scalars.

    The reduction is PER QUERY ROW, then averaged: each block row has its own
    attention distribution over the context, and pooling before normalising would
    mix 16 different distributions into one unnormalised vector (whose "entropy"
    is then meaningless and can exceed log2(ctx_len)). Heads are averaged first
    because the diagnosis is about where the block looks, not about individual
    heads.

    `vis_ctx` is the boolean visual-position mask over context rows; `thirds`
    splits the context into contiguous position bands. Entropy is in bits and
    normalised by log2(ctx_len), so 1.0 means uniform over the whole context and
    0.0 means the row looks at exactly one context position.
    """
    w = w_ctx.float().mean(dim=1)[0]                # (q_len, ctx_len)
    ctx_len = int(w.shape[-1])
    vis_ctx = vis_ctx.to(w.device).float()
    total = w.sum(-1, keepdim=True).clamp_min(1e-9)
    wn = w / total                                  # normalised per query row
    vis_mass = float((wn * vis_ctx).sum(-1).mean())
    ent = float((-(wn.clamp_min(1e-12) * wn.clamp_min(1e-12).log2()).sum(-1)).mean())
    max_ent = float(torch.log2(torch.tensor(float(ctx_len))))
    third_mass = [float(wn[:, a:b].sum(-1).mean()) for a, b in thirds]
    k = max(1, int(round(0.01 * ctx_len)))
    top = float(torch.topk(wn, k, dim=-1).values.sum(-1).mean())
    return {
        "vis_mass": vis_mass,
        "entropy_bits": ent,
        "entropy_norm": (ent / max_ent) if max_ent > 0 else 0.0,
        "third_mass": third_mass,
        "top1pct_mass": top,
        "n_ctx": ctx_len,
    }


def context_thirds(ctx_len, n_bands=3):
    edges = [round(i * ctx_len / n_bands) for i in range(n_bands + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n_bands)]


def visual_mask_for_row(ids, cfg_ids):
    video_id, image_id = cfg_ids
    return ((ids == video_id) | (ids == image_id))[0]
