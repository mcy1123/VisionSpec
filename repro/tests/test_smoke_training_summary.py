"""Small CPU tests of full-parameter drift and fp32-to-bf16 audit semantics."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100.summarize_smoke_training import compare_parameters, finish_stats, comparisons


def test_master_can_change_while_bf16_remains_identical():
    base = torch.tensor([1., 2., 3., 4.], dtype=torch.bfloat16)
    master = base.float() + torch.tensor([0.0001, 0.0001, 0.05, -0.1])
    saved = master.bfloat16()
    stats = finish_stats(compare_parameters(base, saved, master, chunk=2))
    assert stats['elements'] == 4 and stats['changed'] == 2
    assert stats['master_changed'] == 4
    assert stats['master_changed_but_saved_unchanged'] == 2
    assert stats['master_to_saved_mismatches'] == 0
    expected = (saved.double() - base.double()).norm() / base.double().norm()
    assert stats['relative_l2'] == pytest.approx(float(expected))
    wrong = saved.clone(); wrong[0] = 2.
    assert compare_parameters(base, wrong, master)['master_to_saved_mismatches'] == 1


def test_pair_matching_changes_only_one_factor():
    rows=[]
    for seed in (0, 1):
        for lr in (1e-5, 3e-6):
            name=f'{seed}:{lr}'
            rows.append({'name':name,'config':{'pool':'same','seed':seed,'lr':lr},
                         'parameter_content_sha256':name,'manifest_verified':{'model.safetensors':name}})
    pairs=comparisons(rows)
    assert len(pairs)==4
    assert all(not p['identical_parameter_contents'] for p in pairs)
