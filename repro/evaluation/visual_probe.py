"""Absolute-position masking for a draft-only visual-hidden diagnostic.

The draft receives the whole prompt on its first call and only newly committed
hidden states thereafter. A prefix slice of the prompt mask is therefore wrong
on every incremental call. The head's one-dimensional ``position_ids`` locates
the context rows; its cache length independently checks that interpretation.
This helper never changes the target input, decoder, or verifier.
"""
from __future__ import annotations

import torch


def zero_visual_context(target_hidden, prompt_visual_mask, position_ids,
                        *, cache_start=None):
    """Return ``(hidden, number_of_zeroed_rows)`` using absolute stream positions.

    Only batch size one is supported by the evaluation decoder. Positions for
    context rows precede positions for the draft block. Generated positions past
    the prompt are always text, even if their integer offset resembles a visual
    position in the prompt. Fail closed if a future head changes this contract.
    """
    if target_hidden.ndim != 3 or target_hidden.shape[0] != 1:
        raise ValueError("zero-visual expects target_hidden with batch size one")
    if prompt_visual_mask.ndim != 1 or prompt_visual_mask.dtype != torch.bool:
        raise ValueError("prompt_visual_mask must be a one-dimensional bool tensor")
    n_context = target_hidden.shape[1]
    if position_ids is None or position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise ValueError("zero-visual requires explicit one-dimensional stream position_ids")
    if position_ids.shape[1] < n_context:
        raise ValueError("position_ids does not cover all target_hidden rows")
    positions = position_ids[0, :n_context].to(target_hidden.device)
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("stream position_ids must be integer-valued")
    if n_context == 0:
        return target_hidden, 0
    start = int(positions[0])
    if start < 0 or not torch.equal(
        positions, torch.arange(start, start + n_context, device=positions.device)
    ):
        raise ValueError("target_hidden positions must be nonnegative and contiguous")
    if cache_start is not None and int(cache_start) != start:
        raise ValueError(f"context starts at {start}, but draft cache starts at {cache_start}")

    mask = prompt_visual_mask.to(target_hidden.device)
    local_mask = torch.zeros(n_context, dtype=torch.bool, device=target_hidden.device)
    in_prompt = positions < mask.numel()
    local_mask[in_prompt] = mask[positions[in_prompt].long()]
    count = int(local_mask.sum())
    if not count:
        return target_hidden, 0
    result = target_hidden.clone()
    result[:, local_mask, :] = 0
    return result, count


class VisualZeroProbe:
    """Install only for diagnostics, reset once per SPEC generation."""

    def __init__(self, head):
        self.original_forward = head.forward
        self.prompt_mask = None
        self.records = []

    def reset(self, prompt_mask):
        self.prompt_mask = prompt_mask
        self.records = []

    def __call__(self, *args, **kwargs):
        if args:
            raise ValueError("zero-visual probe requires keyword head arguments")
        hidden = kwargs.get("target_hidden")
        if hidden is None or self.prompt_mask is None:
            raise ValueError("zero-visual probe has no target_hidden or prompt identity")
        cache = kwargs.get("past_key_values")
        start = int(cache.get_seq_length()) if cache is not None else 0
        kwargs["target_hidden"], count = zero_visual_context(
            hidden, self.prompt_mask, kwargs.get("position_ids"), cache_start=start)
        self.records.append({"context_start": start,
                             "context_rows": int(hidden.shape[1]),
                             "zeroed_visual_rows": count})
        return self.original_forward(**kwargs)
