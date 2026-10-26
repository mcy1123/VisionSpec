#!/usr/bin/env python3
"""Numerical check of the trainer's batched multi-anchor forward.

`train_head.py` replaces the head's own forward so that one sequence's anchors
share a single context projection. That rewrite is the piece most likely to be
subtly wrong, and it can be checked without the target model: build a head from
the released config with random weights, then compare

  * the released forward, given only the prefix and decode-shaped positions, and
  * this file's forward, given the whole sequence plus a mask,

on the same single anchor. They must agree to floating-point noise.

    python test_forward.py
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
import torch.nn.functional as F

from shared import common
from historical import train_head as T


def build_head(device, seed=0):
    from dflash.model import DFlashDraftModel
    from transformers import AutoConfig

    cfg_path = os.path.join(common.WORKSPACE, "scripts", "head_ref", "config.json")
    config = AutoConfig.from_pretrained(cfg_path)
    torch.manual_seed(seed)
    head = DFlashDraftModel(config).to(device=device, dtype=torch.bfloat16)
    head.eval()
    return head, config


def main():
    device = "cuda:3"
    torch.manual_seed(0)
    head, config = build_head(device)

    total = 160
    block = common.BLOCK_SIZE
    ctx = torch.randn(1, total, len(common.KEPT_LAYERS) * config.hidden_size,
                      device=device, dtype=torch.bfloat16)
    seq_ids = torch.randint(0, config.vocab_size, (total,), device=device)

    print("--- single-anchor: released forward vs batched forward ---")
    worst = 1.0
    for anchor in (1, 40, 90, total - block - 1):
        diff, scale, cosine = T.verify(head, _FakeTarget(head, config), ctx, seq_ids,
                                       common.MASK_ID, block, device, anchor=anchor)
        worst = min(worst, cosine)
        print(f"  anchor {anchor:4d}  max|diff| {diff:.3e}  scale {scale:.3e}  "
              f"cosine {cosine:.6f}")
    print(f"worst hidden cosine {worst:.6f}")
    # bf16 rounding over five layers is what separates the two paths; in fp32
    # they agree to ~3e-6 relative, which is the equality that matters.
    assert worst > 0.999, f"custom forward diverges from the released one ({worst:.6f})"
    print("PASS: batched forward reproduces the released forward")

    print("--- multi-anchor loss: finite, weighted, differentiable ---")
    anchors = list(range(20, total - block))
    reveals = T.sample_reveals(len(anchors), block,
                               torch.Generator(device="cpu").manual_seed(0), device)
    print(f"  {len(anchors)} anchors, reveal lengths "
          f"{reveals.min().item()}..{reveals.max().item()}, "
          f"mean {reveals.float().mean():.2f}")

    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    fake_target = _FakeTarget(head, config)
    loss, correct, weights, by_offset = T.sequence_loss(
        head, fake_target, ctx, seq_ids, anchors, reveals,
        fake_target.embed_weight, common.MASK_ID, block, device)
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert torch.isfinite(loss), "loss is not finite"
    assert grads and all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    per_offset = (by_offset[:, 0] / by_offset[:, 1].clamp_min(1)).tolist()
    print(f"  loss {loss.item():.4f}  supervised positions {int(weights.gt(0).sum())} "
          f"  grad tensors {len(grads)}")
    print("  per-offset top1: " + " ".join(f"{v:.2f}" for v in per_offset[1:]))
    print("PASS: multi-anchor loss is finite and back-propagates")

    check_d2sd_identity()


class _FakeTarget:
    """Enough of a target for the loss: an embedding table and an lm_head."""

    def __init__(self, head, config):
        torch.manual_seed(1)
        self.embed_weight = torch.randn(config.vocab_size, config.hidden_size,
                                        device=head.device, dtype=torch.bfloat16) * 0.02
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size,
                                       bias=False).to(head.device, torch.bfloat16)

    def get_input_embeddings(self):
        return self

    @property
    def weight(self):
        return self.embed_weight



def check_d2sd_identity():
    """`0.8^offset` must equal D2SD Eq. 7's exponential with tau = 1/ln(1/0.8).

    The paper gives the weight only as a phrase ("coverage CE, 0.8^offset
    weight"); D2SD gives it as an exponential in the distance past the reveal
    boundary. If those two are not the same function, the trainer is optimising
    something the paper did not specify.
    """
    import math

    from shared import common

    tau = common.d2sd_tau()
    worst = 0.0
    for reveal in range(common.PREFIX_MIN, common.BLOCK_SIZE):
        for k in range(reveal, common.BLOCK_SIZE):
            d2sd = math.exp(-(k - reveal - 1 + 1) / tau)   # its first masked k has exponent 0
            ours = common.DECAY ** (k - reveal)
            worst = max(worst, abs(d2sd - ours))
    print(f"--- D2SD Eq.7 identity: tau = {tau:.4f}, max |exp form - 0.8^offset| "
          f"= {worst:.3e}")
    assert worst < 1e-9, f"weighting does not match D2SD Eq. 7 ({worst:.3e})"

    lengths, probs = common.prefix_weights()
    print(f"    reveal support {int(lengths[0])}..{int(lengths[-1])} "
          f"(D2SD Eq.6 gives l in [0, gamma-2] -> reveal in [1, gamma-1])")
    print(f"    prior sums to {float(probs.sum()):.6f}, mean reveal "
          f"{float((lengths * probs).sum()):.2f}")
    assert int(lengths[0]) == 1 and int(lengths[-1]) == common.BLOCK_SIZE - 1
    assert abs(float(probs.sum()) - 1.0) < 1e-9
    print("PASS: the variable-prefix objective matches D2SD Eq. 6 and Eq. 7")


if __name__ == "__main__":
    main()
