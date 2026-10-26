#!/usr/bin/env python3
"""Unit tests for the DSpark-style Markov bias.

The load-bearing test is equivalence. Every claim about "the Markov head adds X"
is worthless unless rank 0 -- or a zeroed `w2` -- reproduces the unmodified block
head bit for bit, on both the training loss and the decoding path. If that test
fails, any measured gain could just as well be a change in the baseline.

    python test_markov.py
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import torch

from shared import common
from shared import markov as M
from historical import train_head as T


def _tiny_head(device, block=16, vocab=64):
    """A block head small enough to forward on CPU, shaped like the real one."""
    from dflash.model import DFlashDraftModel
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(
        __import__("os").path.join(common.WORKSPACE, "scripts", "head_ref", "config.json"))
    cfg.vocab_size = vocab
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.head_dim = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_hidden_layers = 2
    cfg.num_target_layers = 6
    cfg.block_size = block
    cfg.dflash_config = {"mask_token_id": vocab - 1, "target_layer_ids": [1, 3, 5]}
    torch.manual_seed(0)
    return DFlashDraftModel(cfg).to(device).eval(), cfg


class _FakeTarget:
    def __init__(self, cfg, device):
        torch.manual_seed(1)
        self.embed_weight = torch.randn(cfg.vocab_size, cfg.hidden_size,
                                        device=device) * 0.02
        self.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size,
                                       bias=False).to(device)

    def get_input_embeddings(self):
        return self


def test_prev_token_indexing():
    """Slot k must be conditioned on the block's slot k-1, slot 0 on itself."""
    blocks = torch.arange(1, 17).reshape(1, 16)
    prev = M.block_prev_tokens(blocks)
    assert prev.shape == blocks.shape
    assert prev[0, 0].item() == blocks[0, 0].item(), "slot 0 conditions on itself"
    for k in range(1, 16):
        assert prev[0, k].item() == blocks[0, k - 1].item(), f"slot {k} shift wrong"
    print("PASS: block_prev_tokens shifts by one, slot 0 self-referential")


def test_zero_bias_is_a_no_op(device="cpu"):
    """A zeroed Markov head must not move the loss by a single bit."""
    head, cfg = _tiny_head(device)
    target = _FakeTarget(cfg, device)
    block = cfg.block_size
    total = 40
    torch.manual_seed(2)
    n_layers = len(cfg.dflash_config["target_layer_ids"])
    ctx = torch.randn(1, total, n_layers * cfg.hidden_size, device=device)
    seq = torch.randint(0, cfg.vocab_size - 1, (total,), device=device)
    anchors = list(range(8, total - block))
    reveals = torch.tensor([1, 3, 5] * len(anchors), device=device)[: len(anchors)]

    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    base = T.sequence_loss(head, target, ctx, seq, anchors, reveals,
                           target.embed_weight, cfg.dflash_config["mask_token_id"],
                           block, device)[0]

    zeroed = M.MarkovHead(cfg.vocab_size, cfg.vocab_size, rank=8).to(device)
    with torch.no_grad():
        zeroed.w2.weight.zero_()
    with_markov = T.sequence_loss(head, target, ctx, seq, anchors, reveals,
                                  target.embed_weight, cfg.dflash_config["mask_token_id"],
                                  block, device, markov=zeroed)[0]

    diff = abs(base.item() - with_markov.item())
    print(f"--- zeroed Markov head: loss {base.item():.6f} vs {with_markov.item():.6f} "
          f"(|diff| {diff:.3e})")
    assert diff == 0.0, f"a zeroed Markov head changed the loss by {diff:.3e}"
    print("PASS: a zeroed Markov head is bit-identical to no Markov head")


def test_bias_shape_and_grad(device="cpu"):
    head, cfg = _tiny_head(device)
    markov = M.MarkovHead(cfg.vocab_size, cfg.vocab_size, rank=8).to(device)
    tok = torch.randint(0, cfg.vocab_size, (5, 16), device=device)
    bias = markov.bias(tok)
    assert bias.shape == (5, 16, cfg.vocab_size), bias.shape
    bias.sum().backward()
    grads = [p.grad for p in markov.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    print(f"PASS: bias shape {tuple(bias.shape)} and both parameters receive finite gradients")


def test_steering_reduces_to_greedy_at_zero(device="cpu"):
    """The semi-autoregressive loop must reproduce plain greedy when the bias is 0.

    This is the inference-side counterpart of the loss equivalence test: it says
    the decode change is a strict generalisation, not a different decoder.
    """
    head, cfg = _tiny_head(device)
    block = cfg.block_size
    torch.manual_seed(3)
    base_logits = torch.randn(block - 1, cfg.vocab_size, device=device)
    root = torch.randint(0, cfg.vocab_size, (1,), device=device)

    zeroed = M.MarkovHead(cfg.vocab_size, cfg.vocab_size, rank=8).to(device)
    with torch.no_grad():
        zeroed.w2.weight.zero_()
    steered, spine = M.steer_block_greedy(base_logits, root, zeroed)

    assert torch.equal(steered, base_logits), "zero bias must leave the marginals alone"
    assert torch.equal(spine, base_logits.argmax(dim=-1)), "spine must be plain greedy"
    print("PASS: semi-autoregressive steering degenerates to greedy at zero bias")


def test_steering_is_path_dependent(device="cpu"):
    """With a live bias, offset k's marginal must depend on the spine before it."""
    _, cfg = _tiny_head(device)
    block = cfg.block_size
    torch.manual_seed(4)
    base_logits = torch.randn(block - 1, cfg.vocab_size, device=device)
    root = torch.randint(0, cfg.vocab_size, (1,), device=device)
    live = M.MarkovHead(cfg.vocab_size, cfg.vocab_size, rank=8).to(device)
    # At DSpark's initialisation the bias has std ~0.01 against base logits of
    # std ~1, so an untrained head is almost a no-op and would never flip an
    # argmax -- a real property, and one worth knowing before reading any
    # training curve. This test is about the mechanism, so scale the projection
    # until the bias dominates and then demand path dependence.
    with torch.no_grad():
        live.w2.weight.mul_(50.0)

    _, spine_a = M.steer_block_greedy(base_logits, root, live)
    steered_b, spine_b = M.steer_block_greedy(base_logits, torch.zeros_like(root), live)

    assert not torch.equal(spine_a, spine_b), \
        "a different root must produce a different conditioned block"
    print("PASS: steering is path-dependent on the root, as a sequential model should be")
    print("      (note: at DSpark's default init the bias std is ~0.01 vs logits ~1, "
          "so an untrained head is nearly a no-op)")


def main():
    test_prev_token_indexing()
    test_zero_bias_is_a_no_op()
    test_bias_shape_and_grad()
    test_steering_reduces_to_greedy_at_zero()
    test_steering_is_path_dependent()
    print("\nall Markov tests passed")


if __name__ == "__main__":
    main()
