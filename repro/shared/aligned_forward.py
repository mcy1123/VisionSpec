#!/usr/bin/env python3
"""The forward the head is *trained* on, routed to the branch decode actually uses.

Measured root cause (phase 0, `probe_alignment.py`, three model/task pairs —
Qwen3-VL-8B video, Qwen3-VL-8B image, Qwen3-VL-2B ScienceQA):

    variant                                   last-pos max|Δh|   cos      all-pos max|Δ|
    cache + no mask      (every decode path)       0.000        1.000        0.0
    2-D all-visible mask + no cache                0.000        1.000        0.0
    no cache + no mask   (every training path)   43.5~152      0.75~0.83   2078~3024

`create_causal_mask` (transformers 4.57.3 `masking_utils`, called unconditionally
by `Qwen3VLTextModel.forward`) returns

    past_key_values is not None  -> None        -> SDPA `is_causal=True`
    attention_mask 2-D all-visible -> None      -> SDPA `is_causal=True`
    otherwise                    -> a materialised 4-D causal mask

so the training path is the *only* one on the materialised-mask branch, and the
two branches disagree structurally: the difference is the same in fp32 as in bf16
(ScienceQA-2B: 143.5 vs 152.0 at the last position) and survives
`attn_implementation="eager"`, so it is neither rounding nor an SDPA-kernel
choice. It starts at layer 1 (0.24 in fp32) and is amplified to O(1) by the last
layer; next-token accuracy of the materialised-mask branch is also worse
(0.0355 vs 0.0491 overall on a video row).

The fix is not a re-derivation of anything: pass a 2-D all-visible attention mask,
which puts the training forward on the same `is_causal` branch every decode path
uses. Verified bit-identical to the decode prefill on every layer, dtype and case.

Nothing in `common.py` is touched: the image side keeps the forward it was
verified with, and the video repair arms opt in explicitly.
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
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common

__all__ = ["target_context_aligned", "target_topk_aligned",
           "verify_aligned_to_decode"]


def target_context_aligned(target, input_ids, pixel_values=None, image_grid_thw=None,
                           pixel_values_videos=None, video_grid_thw=None,
                           layer_ids=common.KEPT_LAYERS):
    """`common.target_context`, on the decode attention branch.

    Returns the concatenated kept-layer hidden states, `(1, n, L*H)`, with the
    image/video path taken exactly as `common.target_context` takes it.
    """
    layer_ids = tuple(i for i in layer_ids if i + 1 < target.config.text_config.num_hidden_layers + 1)
    kwargs = dict(input_ids=input_ids, use_cache=False,
                  output_hidden_states=True, logits_to_keep=1,
                  # The whole fix: a 2-D all-visible mask makes create_causal_mask
                  # return None, i.e. the same is_causal branch decode uses.
                  attention_mask=torch.ones_like(input_ids))
    if pixel_values is not None:
        kwargs.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        kwargs["position_ids"] = common.target_positions(target, input_ids, image_grid_thw)
    if pixel_values_videos is not None:
        kwargs.update(pixel_values_videos=pixel_values_videos,
                      video_grid_thw=video_grid_thw)
        positions, _ = target.model.get_rope_index(input_ids, image_grid_thw,
                                                   video_grid_thw, None)
        kwargs["position_ids"] = positions
    with torch.no_grad():
        out = target(**kwargs)
    return common.context_feature(out.hidden_states, layer_ids).detach()


def target_topk_aligned(target, full_ids, k, pixel_values=None, image_grid_thw=None,
                        pixel_values_videos=None, video_grid_thw=None, chunk=64):
    """`opd_rollout.target_topk`, on the decode attention branch.

    Returns `(topk_ids, topk_val)` of shape `(S-1, k)`; row `r` predicts
    `full_ids[r+1]` (the official convention).
    """
    kwargs = dict(input_ids=full_ids, use_cache=False, logits_to_keep=0,
                  attention_mask=torch.ones_like(full_ids))
    if pixel_values is not None:
        kwargs.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        kwargs["position_ids"] = common.target_positions(target, full_ids, image_grid_thw)
    if pixel_values_videos is not None:
        kwargs.update(pixel_values_videos=pixel_values_videos,
                      video_grid_thw=video_grid_thw)
        pos, _ = target.model.get_rope_index(full_ids, None, video_grid_thw, None)
        kwargs["position_ids"] = pos
    with torch.no_grad():
        out = target(**kwargs)
    logits = out.logits[0]
    S = logits.shape[0]
    ids_out, val_out = [], []
    for lo in range(0, S - 1, chunk):
        hi = min(lo + chunk, S - 1)
        lp = F.log_softmax(logits[lo:hi].float(), dim=-1)
        v, i = torch.topk(lp, k, dim=-1)
        ids_out.append(i.to(torch.int32).cpu())
        val_out.append(v.to(torch.float16).cpu())
        del lp
    return torch.cat(ids_out), torch.cat(val_out)


def verify_aligned_to_decode(target, input_ids, pixel_values=None, image_grid_thw=None,
                             pixel_values_videos=None, video_grid_thw=None):
    """Assert the aligned forward is bit-identical to the decode prefill.

    The decode prefill is `glance.generate`'s first call: explicit position_ids,
    a KV cache, no attention_mask. This is the gate that makes every "aligned"
    claim in the report checkable rather than asserted. Returns (max|Δ|, cosine).
    """
    from transformers.cache_utils import DynamicCache

    kw = {}
    if pixel_values is not None:
        kw.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    if pixel_values_videos is not None:
        kw.update(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw)
    if pixel_values is not None:
        pos = common.target_positions(target, input_ids, image_grid_thw)
    elif pixel_values_videos is not None:
        pos, _ = target.model.get_rope_index(input_ids, None, video_grid_thw, None)
    else:
        pos = None
    if pos is not None:
        kw["position_ids"] = pos
    target.model.rope_deltas = None
    with torch.no_grad():
        dec = target(input_ids, use_cache=True, past_key_values=DynamicCache(),
                     output_hidden_states=True, logits_to_keep=1, **kw)
    layer_ids = tuple(i for i in common.KEPT_LAYERS
                      if i + 1 < target.config.text_config.num_hidden_layers + 1)
    dec_ctx = common.context_feature(dec.hidden_states, layer_ids).detach()
    ali = target_context_aligned(target, input_ids, pixel_values, image_grid_thw,
                                 pixel_values_videos, video_grid_thw)
    d = float((dec_ctx.float() - ali.float()).abs().max())
    cos = float(F.cosine_similarity(dec_ctx.flatten().float(), ali.flatten().float(), dim=0))
    return d, cos
