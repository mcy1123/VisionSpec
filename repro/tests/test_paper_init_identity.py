"""Regress paper-head provenance before expensive training."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_h100.init_paper_transfer import check_backbone


def test_verifies_plain_bytes_and_ignores_only_added_readout():
    plain = {"weights":torch.tensor([-0., 1.], dtype=torch.bfloat16)}
    head = {**plain, "evidence_readout.value":torch.tensor([2.])}
    assert check_backbone(head, plain) == 1
    with pytest.raises(ValueError, match="tensor differs"):
        check_backbone({**head, "weights":torch.tensor([+0.,1.],dtype=torch.bfloat16)}, plain)
    with pytest.raises(ValueError, match="tensor names differ"):
        check_backbone({**head, "unexpected":torch.tensor([1.])}, plain)
