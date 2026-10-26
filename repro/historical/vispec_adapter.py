#!/usr/bin/env python3
"""A ViSpec-style trainable visual adapter for the block draft head.

What it replaces
----------------
The head's only vision path is one shared linear map plus a norm, applied to
every position alike:

    target_hidden = hidden_norm(fc(cat(hidden_states[1, 9, 17, 25, 33])))

Vision and text therefore compete for the same projection, and vision gets no
compression and no global summary. ViSpec's premise is that vision wants its own
lightweight encoder. This module supplies one:

    visual rows -> dedicated Transformer encoder -> n_q latent queries
                                                     |         |
                                                     |         +-> global vector
                                                     |             added to every
                                                     |             TEXT row
                                                     +-> bottleneck that the visual
                                                         rows are reconstructed from
    text rows   -> unchanged hidden_norm(fc(...)) + global vector

Design constraint that shaped it
--------------------------------
`dflash.model.apply_rotary_pos_emb` broadcasts `cos/sin` over the key axis, so the
number of context rows the head returns MUST equal the number of positions the
decoder hands it. Genuine KV-count compression (dropping visual rows) would
therefore require editing `glance.decode.generate`'s position bookkeeping, which
would put the verified tau pipeline on the critical path of an experiment whose
whole point is to be comparable to it. So compression here is an INFORMATION
bottleneck, not a row-count reduction: every visual row is reconstructed from the
`n_q` latents, so the visual KV can carry only what those latents hold. The
row count, the positions, the tree and the decoder are untouched. This deviation
from ViSpec is recorded in ViSpec判定_日期.md section 2.

Parameter count is reported by `param_count()` and asserted under 100M by the
trainer, so a win cannot come from size.

`--use-vispec` is off by default everywhere; arms that do not ask for it are
bit-identical to before.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

PARAM_LIMIT = 100_000_000


class Block(nn.Module):
    """Pre-norm Transformer block (self-attention over the visual rows)."""

    def __init__(self, d, n_heads, ffn):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, mask=None):
        h = self.n1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=mask, need_weights=False)
        x = x + h
        return x + self.mlp(self.n2(x))


class ViSpecAdapter(nn.Module):
    """Compress visual rows into latents; rebuild visual KV from them; summarise for text.

    Args:
        hidden_size: D of the draft head (4096).
        n_concat: number of concatenated target layers the head reads (5).
        d_v: adapter width.
        n_heads: attention heads inside the adapter.
        n_queries: latent count, i.e. the compression ratio's denominator.
        enc_layers: Transformer depth over visual rows.
    """

    def __init__(self, hidden_size=4096, n_concat=5, d_v=512, n_heads=8,
                 n_queries=16, enc_layers=2, ffn=1024, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_concat = n_concat
        self.in_dim = hidden_size * n_concat
        self.d_v = d_v
        self.n_queries = n_queries

        self.vis_in = nn.Linear(self.in_dim, d_v)
        self.enc = nn.ModuleList([Block(d_v, n_heads, ffn) for _ in range(enc_layers)])
        self.latents = nn.Parameter(torch.randn(1, n_queries, d_v) * 0.02)
        # visual rows query the latents: this is the bottleneck
        self.q_from_vis = nn.Linear(d_v, d_v)
        self.to_vis = nn.Linear(d_v, hidden_size)
        # global summary added to every text row
        self.to_global = nn.Linear(d_v, hidden_size)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, base, vis_mask):
        """base (1, T, n_concat*D); vis_mask (1, T) bool. Returns (1, T, D)."""
        if base.shape[-1] != self.in_dim:
            raise ValueError(f"expected {self.in_dim} input channels, got {base.shape[-1]}")
        if base.shape[1] != vis_mask.shape[1]:
            raise ValueError("visual mask must align with the context rows")
        v = vis_mask[0]
        out = torch.zeros(base.shape[0], base.shape[1], self.hidden_size,
                          dtype=base.dtype, device=base.device)
        if not bool(v.any()):
            # No visual rows at all (text-only prompt): the adapter contributes
            # nothing rather than inventing a summary from an empty set.
            return out
        vis_rows = base[0][v]                                   # (Tv, n_concat*D)
        h = self.vis_in(vis_rows).unsqueeze(0)                  # (1, Tv, d_v)
        for blk in self.enc:
            h = blk(h)
        lat = self.latents.expand(1, -1, -1).to(h.dtype)        # (1, n_q, d_v)
        # bottleneck: visual rows may only read the n_q latents
        q = self.q_from_vis(h)                                  # (1, Tv, d_v)
        recon = F.scaled_dot_product_attention(q, lat, lat)     # (1, Tv, d_v)
        out[0, v] = self.to_vis(recon)[0]
        return out

    def global_vector(self, base, vis_mask):
        """The summary that enhances text rows (kept separate for inspection)."""
        v = vis_mask[0]
        if not bool(v.any()):
            return torch.zeros(base.shape[0], 1, self.hidden_size,
                               dtype=base.dtype, device=base.device)
        h = self.vis_in(base[0][v]).unsqueeze(0)
        for blk in self.enc:
            h = blk(h)
        lat = self.latents.expand(1, -1, -1).to(h.dtype)
        return self.to_global(lat.mean(dim=1, keepdim=True))


class ArmProjection(nn.Module):
    """Drop-in replacement for `head.fc` + `head.hidden_norm`.

    `train_head.forward_anchors` and `DFlashDraftModel.forward` both compute
    `head.hidden_norm(head.fc(ctx))`, so substituting this module for `fc` and
    swapping `hidden_norm` for `nn.Identity` routes BOTH the training and the
    decode path through one implementation. That is why there is no parallel
    forward to drift from the verified one.

    Input is the *slim wide* context: `cat([fused(n_concat*D), vis_mask(1)])`.
    """

    def __init__(self, fc, hidden_norm, adapter):
        super().__init__()
        self.fc = fc
        self.hidden_norm = hidden_norm
        self.adapter = adapter

    def forward(self, wide):
        n = self.adapter.n_concat * self.adapter.hidden_size
        base = wide[..., :n]
        vis = wide[..., n] > 0.5
        txt = self.hidden_norm(self.fc(base))
        v = vis[0]
        if not bool(v.any()):
            return txt
        vis_kv = self.adapter(base, vis)
        g = self.adapter.global_vector(base, vis)
        out = txt.clone()
        out[0, v] = vis_kv[0, v]
        if bool((~v).any()):
            out[0, ~v] = out[0, ~v] + g[0]
        return out


def attach_vispec(head, d_v=512, n_queries=16, enc_layers=2, n_heads=8, ffn=1024,
                  seed=0):
    """Give `head` a ViSpec adapter and route fc+hidden_norm through it.

    Returns the adapter. Raises if the head already has one, so a double attach
    cannot silently stack two adapters.
    """
    if getattr(head, "_vispec_attached", False):
        raise RuntimeError("head already has a ViSpec adapter attached")
    dtype = next(head.parameters()).dtype
    adapter = ViSpecAdapter(hidden_size=head.config.hidden_size,
                            n_concat=len(head.target_layer_ids),
                            d_v=d_v, n_heads=n_heads, n_queries=n_queries,
                            enc_layers=enc_layers, ffn=ffn)
    torch.manual_seed(seed)
    adapter = adapter.to(dtype=dtype, device=next(head.parameters()).device)
    head.fc = ArmProjection(head.fc, head.hidden_norm, adapter)
    head.hidden_norm = nn.Identity()
    head._vispec_attached = True
    head._vispec_cfg = {"d_v": d_v, "n_queries": n_queries,
                        "enc_layers": enc_layers, "n_heads": n_heads, "ffn": ffn}
    return adapter


def vispec_state(head):
    if not getattr(head, "_vispec_attached", False):
        raise RuntimeError("no ViSpec adapter attached")
    return head.fc.adapter.state_dict()


def save_vispec(head, out_dir, name="vispec.safetensors"):
    """Save the adapter next to the head, so the head file stays loadable as-is."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    save_file({k: v.contiguous() for k, v in vispec_state(head).items()}, p)
    with open(os.path.join(out_dir, "vispec_config.json"), "w") as fh:
        json.dump(head._vispec_cfg, fh, indent=1)
    return p


def load_vispec(head, ckpt_dir, name="vispec.safetensors"):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(ckpt_dir, "vispec_config.json")))
    adapter = attach_vispec(head, **cfg)
    sd = load_file(os.path.join(ckpt_dir, name))
    missing = adapter.load_state_dict(sd, strict=True)
    return adapter


class SlimWideContext:
    """Make `glance.decode.context_feature` return `cat([fused(5D), vis_mask(1)])`.

    The adapter needs to know which context rows are visual, and that mask has to
    survive the per-round `[0][keep][None]` gather that `generate` applies to the
    context. Appending it as one extra channel is the cheapest way to guarantee
    that: the gather indexes positions, so every channel is gathered consistently
    with the rows it annotates. Mirrors `diag_common.WideContext`.
    """

    def __init__(self, target, target_path=None):
        import glance.decode as dec
        path = target_path or "/mnt/local/mcy/GLANCE/models/Qwen3-VL-8B-Instruct"
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.video_id = int(cfg["video_token_id"])
        self.image_id = int(cfg["image_token_id"])
        self.ids = None
        self._dec = dec
        self._orig = dec.context_feature
        holder = self

        def slim_context_feature(hidden_states, layer_ids):
            base = torch.cat([hidden_states[i + 1] for i in layer_ids], dim=-1)
            t = base.shape[1]
            if holder.ids is None:
                raise RuntimeError("target ids not recorded; pre-hook did not fire")
            vis = ((holder.ids[:, :t] == holder.video_id)
                   | (holder.ids[:, :t] == holder.image_id))
            return torch.cat([base, vis.to(base.dtype).unsqueeze(-1)], dim=-1)

        dec.context_feature = slim_context_feature
        self._hook = target.register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def _pre_hook(self, module, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is not None:
            self.ids = ids

    def close(self):
        self._dec.context_feature = self._orig
        self._hook.remove()


def build_wide(fused, vis_mask):
    """Slim wide tensor from an already-computed `fused` and a bool mask."""
    return torch.cat([fused, vis_mask.to(fused.dtype).unsqueeze(-1)], dim=-1)
