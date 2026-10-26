
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

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, os.path.join(str(_REPRO_ROOT.parent),
                               "GLANCE_repo"))

from historical.opd_online_rollout import anchor_from_logits


def test_chain_accepts_only_matching_prefix():
    logits = torch.tensor([
        [0.0, 5.0, 1.0, 0.0],
        [0.0, 1.0, 5.0, 0.0],
        [0.0, 0.0, 1.0, 5.0],
    ])
    got = anchor_from_logits(logits, [1, 0, 3], verify="chain")
    assert got.prop == [1, 2, 3]
    assert got.n_acc == 1


def test_tree_can_accept_non_argmax_path_and_uses_argmax_after_reject():
    logits = torch.tensor([
        [0.0, 5.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 5.0, 4.0],
    ])
    got = anchor_from_logits(
        logits, [2, 3, 0], verify="tree", tree_budget=20, branch_k=2)
    assert got.n_acc == 2
    assert got.prop == [2, 3, 4]


def test_width_one_tree_matches_chain():
    torch.manual_seed(0)
    logits = torch.randn(5, 11)
    want = [int(x) for x in logits.argmax(-1)]
    want[3] = (want[3] + 1) % logits.shape[-1]
    chain = anchor_from_logits(logits, want, verify="chain")
    tree = anchor_from_logits(
        logits, want, verify="tree", tree_budget=5, branch_k=1)
    assert tree.prop == chain.prop
    assert tree.n_acc == chain.n_acc
