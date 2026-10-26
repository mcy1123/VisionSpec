#!/usr/bin/env python3
"""Current-policy OPD rollout over a cached lossless target trajectory.

The target's greedy tokens and fused hidden states determine the committed
trajectory. The draft head is nevertheless run afresh at every decoding
round, with the same KV-cache update used by :func:`glance.decode.generate`.
This makes the proposals genuinely current-policy without paying for a second
video-target verification pass: at temperature zero GLANCE's verifier can only
commit that cached greedy trajectory.
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from glance.tree import build_budget_tree, children_of


@dataclass
class Anchor:
    pos: int
    prop: list[int]
    n_acc: int


def anchor_from_logits(logits, want, *, verify="tree", tree_budget=63,
                       branch_k=8) -> Anchor:
    """Turn one block's offset logits into an OPD anchor (position omitted)."""
    avail = min(int(logits.shape[0]), len(want))
    if avail <= 0:
        return Anchor(pos=-1, prop=[], n_acc=0)
    logits = logits[:avail].float()
    amax = logits.argmax(dim=-1)

    if verify == "chain":
        target = torch.as_tensor(want[:avail], device=amax.device)
        match = (amax == target).to(torch.int32)
        n_acc = int(match.cumprod(0).sum().item())
        return Anchor(pos=-1, prop=[int(x) for x in amax], n_acc=n_acc)
    if verify != "tree":
        raise ValueError(f"verify must be 'chain' or 'tree', got {verify!r}")

    k = max(1, min(int(branch_k), int(logits.shape[-1])))
    logp = F.log_softmax(logits, dim=-1)
    top = torch.topk(logp, k, dim=-1)
    top_ids, top_logp = top.indices.cpu(), top.values.cpu()
    nodes = build_budget_tree(top_ids, top_logp, int(tree_budget), k)
    children = children_of(nodes)
    cursor = -1
    path_tokens: list[int] = []
    for token in want[:avail]:
        hit = next((i for i in children.get(cursor, [])
                    if nodes[i].token == int(token)), None)
        if hit is None:
            break
        path_tokens.append(int(nodes[hit].token))
        cursor = hit
    n_acc = len(path_tokens)
    prop = path_tokens + [int(x) for x in amax[n_acc:avail]]
    return Anchor(pos=-1, prop=prop, n_acc=n_acc)


@torch.inference_mode()
def rollout(head, target, ctx, seq_ids, *, prompt_len, verify="tree",
            tree_budget=63, branch_k=8, max_rounds=0, head_dtype=None):
    """Sequentially re-roll the draft head on the current policy.

    ``ctx[:, :prompt_len]`` initializes the head context. After a round that
    accepts ``a`` draft tokens, ``ctx[:, pos:pos+a+1]`` supplies exactly the
    target hidden states retained by lossless verification (root + accepted
    path); the target's bonus token is the next round's root and is not
    included yet. This mirrors ``glance.decode.generate`` step for step.

    ``head_dtype`` (2026-10-10, additive; default None = exact previous
    behaviour) casts the context rows and the noise embeddings into the head's
    dtype before the forward and the head's output back into the target's dtype
    before ``lm_head``. It exists for the mixed-precision configuration
    "fp32 teacher + bf16 head": an fp32 target hands over fp32 fused hidden
    states and an fp32 embedding table while the head's ``fc``/``hidden_norm``
    are bf16, which otherwise raises
    ``RuntimeError: expected mat1 and mat2 to have the same dtype``.
    """
    if seq_ids.ndim != 1 or ctx.ndim != 3:
        raise ValueError(f"expected seq_ids [S], ctx [1,S,H], got "
                         f"{tuple(seq_ids.shape)}, {tuple(ctx.shape)}")
    block = int(head.block_size)
    mask_id = int(head.mask_token_id)
    total = min(int(seq_ids.shape[0]), int(ctx.shape[1]))
    pos = int(prompt_len)
    if not 0 <= pos < total:
        raise ValueError(f"prompt_len {pos} outside trajectory length {total}")

    if head_dtype is None:
        head_dtype = next(head.parameters()).dtype
    lm_dtype = target.lm_head.weight.dtype
    embed = target.get_input_embeddings()
    # One cast of the whole context rather than one per round: `fused` is a view.
    ctx = ctx.to(head_dtype)
    cache = DynamicCache()
    stream = torch.arange(total + block + 1, device=seq_ids.device).unsqueeze(0)
    fused = ctx[:, :pos]
    anchors: list[dict] = []

    while pos < total - 1 and (max_rounds <= 0 or len(anchors) < max_rounds):
        ids = torch.full((1, block), mask_id, dtype=torch.long,
                         device=seq_ids.device)
        ids[0, 0] = seq_ids[pos]
        hidden = head(
            target_hidden=fused,
            noise_embedding=embed(ids).to(head_dtype),
            position_ids=stream[:, cache.get_seq_length():pos + block],
            past_key_values=cache,
            use_cache=True,
            is_causal=False,
        )
        logits = target.lm_head(hidden[:, 1 - block:, :].to(lm_dtype))[0]
        cache.crop(pos)

        want = [int(x) for x in seq_ids[pos + 1:min(total, pos + block)]]
        anchor = anchor_from_logits(
            logits, want, verify=verify, tree_budget=tree_budget,
            branch_k=branch_k)
        anchor.pos = pos
        anchors.append(asdict(anchor))

        old_pos = pos
        pos += anchor.n_acc + 1
        fused = ctx[:, old_pos:pos]

    return anchors
