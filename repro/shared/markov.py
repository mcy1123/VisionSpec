"""Markov (sequential) logit bias for the block head, ported from DSpark.

Why this file exists. GLANCE fills all B=16 block positions in one forward and
scores candidate prefixes as a product of per-offset marginals, which is exact
only where the block's tokens are conditionally independent given the context
(the paper's Assumption 2, self-labelled as an assumption). DSpark names the
failure that follows -- "multi-modal collision": each position marginalises over
all possible prior tokens instead of conditioning on one sampled prefix -- and
fixes it with a low-rank logit bias conditioned on the previous block token.

Measured on this reproduction before writing any of it (probe_prefix_conditioning.py,
ChartQA, block 16):

    offset   decode-time   true prefix   gain     self prefix
      4        0.259         0.464      +0.205      0.295
      8        0.205         0.420      +0.214      0.214
     15        0.125         0.277      +0.152      0.125

Two things fall out. Handing the head the correct prefix roughly doubles deep
offset accuracy, so the information is in the representation and the deficit is
conditioning -- that is what justifies training a sequential module rather than
making the head bigger. And the head's *own* proposals recover almost nothing
(0.214 vs 0.205 at offset 8), which is why the untrained in-block refinement of
`refine_decode.py` scored -0.4%: a wrong prefix carries no new information. The
sequential module has to be trained, not bolted on.

Index convention. DSpark's `sample_from_anchor=False` is GLANCE's convention:
slot 0 is the root, slot k predicts `anchor + k`, so the previous token for slot k
is the block's slot k-1. DSpark's own code builds that with
`prev = cat([blocks[:, :1], blocks[:, :-1]], dim=1)`.
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
import os

import torch
import torch.nn as nn

__all__ = ["MarkovHead", "block_prev_tokens", "steer_block_greedy", "load_markov"]


class MarkovHead(nn.Module):
    """Low-rank logit bias ``B = W1 @ W2`` conditioned on the previous block token.

    ``w1`` indexes the *target's* vocabulary (the previous token id), ``w2``
    projects the rank-256 code to the draft vocabulary so the bias adds directly
    onto the head's logits. Rank 0 disables the head entirely, which the tests
    use to assert bit-identical equivalence with the unmodified block head.
    """

    def __init__(self, verifier_vocab_size: int, draft_vocab_size: int,
                 rank: int = 256, init_std: float = 0.01) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.rank = rank
        self.w1 = nn.Embedding(verifier_vocab_size, rank)
        nn.init.normal_(self.w1.weight, std=init_std)
        self.w2 = nn.Linear(rank, draft_vocab_size, bias=False)

    def bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """``[...,] * long -> [..., draft_vocab]`` logit bias, shape-preserving."""
        return self.w2(self.w1(prev_token_ids.long()))


def block_prev_tokens(block_tokens: torch.Tensor) -> torch.Tensor:
    """Slot k's Markov condition: the block's slot k-1, with slot 0 self-referential.

    Slot 0 is the root and is never supervised, so what sits there is irrelevant;
    DSpark keeps the anchor token itself. Shape ``[N, B] -> [N, B]``.
    """
    return torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)


@torch.no_grad()
def steer_block_greedy(base_logits: torch.Tensor, root_token: torch.Tensor,
                       markov: MarkovHead, temperature: float = 0.0,
                       scale: float = 1.0):
    """Turn one bias-free draft pass into a semi-autoregressive block.

    DSpark samples the block position by position, each conditioned on the token
    it just produced, and verifies the sampled block. GLANCE builds a tree from
    per-offset marginals instead, so the analogous object is the block's
    *conditioned* marginals along the greedy spine: offset k's distribution
    conditioned on the spine tokens at 1..k-1. That makes the plugin score a
    product along one realised prefix rather than a product of independent
    marginals, which is the quantity Assumption 2 wanted but could not express.

    Args:
        base_logits: ``[B-1, V]`` draft logits for offsets 1..B-1, no Markov bias.
        root_token: ``[1]`` the committed root the block is anchored on.
        markov: the trained head.
        scale: multiplier on the bias. The reference init leaves the bias at
            std ~0.01 against logits of order 1, and at lr 4e-5 a 1000-step
            epoch moves each weight by at most ~0.04, so a trained bias can
            still be too small to change any argmax. Sweeping this separates
            "the mechanism does not transfer" from "the mechanism was never
            driven hard enough", without retraining. 1.0 is the trained value.

    Returns:
        ``(steered_logits, spine)`` -- ``[B-1, V]`` conditioned marginals and the
        ``[B-1]`` greedy spine they were conditioned on.
    """
    offsets, vocab = base_logits.shape
    steered = torch.empty_like(base_logits)
    spine = torch.empty(offsets, dtype=torch.long, device=base_logits.device)
    prev = root_token.reshape(1)
    for k in range(offsets):
        logits_k = base_logits[k] + scale * markov.bias(prev)[0]
        steered[k] = logits_k
        if temperature > 0:
            probs = torch.softmax(logits_k.float() / temperature, dim=-1)
            prev = torch.multinomial(probs, 1)
        else:
            prev = logits_k.argmax().reshape(1)
        spine[k] = prev
    return steered, spine


def load_markov(head_path, device="cuda"):
    """The Markov module a trained head ships beside itself, or ``None``.

    ``train_head.py`` writes ``markov_head.pt`` (a state dict) next to the head
    whenever it trained one, so that file is what distinguishes a Markov head
    from the baseline. The companion ``markov_config.json`` is written for the
    final checkpoint; intermediate ``stepN`` checkpoints written by
    ``--save-every`` historically shipped only the state dict. The rank is
    recoverable from the state dict itself (``w1`` is ``(vocab, rank)``), so a
    missing config is inferred rather than treated as "no Markov head" -- that
    distinction matters, because returning ``None`` here silently evaluates the
    head with the bias switched off and looks exactly like a baseline result.
    """
    cfg_path = os.path.join(head_path, "markov_config.json")
    w_path = os.path.join(head_path, "markov_head.pt")
    if not os.path.exists(w_path):
        return None
    state = torch.load(w_path, map_location="cpu")
    if "w2.weight" not in state:
        raise ValueError(f"{w_path} has no 'w2.weight'; not a MarkovHead state dict")
    rank = int(state["w1.weight"].shape[1])
    input_mode = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        rank = int(cfg.get("rank", rank))
        input_mode = cfg.get("input")
    else:
        print(f"[markov] no markov_config.json in {head_path}; inferring rank "
              f"{rank} from the state dict (stepN checkpoints omit the config)",
              flush=True)
    module = MarkovHead(int(state["w1.weight"].shape[0]),
                        int(state["w2.weight"].shape[0]),
                        rank=rank)
    module.load_state_dict(state)
    module = module.to(device).eval()
    for param in module.parameters():
        param.requires_grad_(False)
    print(f"[markov] loaded rank {rank}, trained input={input_mode!r}, "
          f"w1 {tuple(state['w1.weight'].shape)} "
          f"w2 {tuple(state['w2.weight'].shape)}", flush=True)
    return module
