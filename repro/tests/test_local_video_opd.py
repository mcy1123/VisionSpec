"""CPU regressions for the unwrapped single-GPU local OPD entry."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import train_video_opd as local
from shared import training_resume


def test_single_uuid_and_balanced_map_are_required():
    assert local.validate_execution(SimpleNamespace(device_map='balanced'), 'GPU-single') == ['GPU-single']
    for bad in ('', '0', '0,1', 'GPU-a,GPU-b', ' GPU-a', 'GPU-a '):
        with pytest.raises(ValueError):
            local.validate_execution(SimpleNamespace(device_map='balanced'), bad)
    with pytest.raises(ValueError, match='balanced'):
        local.validate_execution(SimpleNamespace(device_map='auto'), 'GPU-single')


def test_noop_preserves_real_module_forward_and_rope_write():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rope_deltas = torch.tensor(7)
        def forward(self, x):
            return x + 1
    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()
            self.lm_head = torch.nn.Linear(1, 1)
            self.hf_device_map = {'': 0}
        def forward(self, x):
            return self.model(x)
    target = Target()
    original_model, original_lm, original_forward = target.model, target.lm_head, target.forward
    assert local.preserve_single_gpu_target(target, 'cuda:0') is target
    assert target.model is original_model and target.lm_head is original_lm
    assert target.forward == original_forward
    assert 'forward' not in target._modules
    target.model.rope_deltas = None
    assert original_model.rope_deltas is None
    assert target(torch.tensor(1)).item() == 2
    target.hf_device_map = {'a': 0, 'b': 'cpu'}
    with pytest.raises(ValueError, match='offloaded'):
        local.preserve_single_gpu_target(target, 'cuda:0')


def test_protocol_records_adapter_and_restores_shared_functions_on_failure(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-single')
    old_wrap = lambda target, device: 'old'
    def old_protocol(*args, **kwargs):
        return {'schema': 3, 'evidence': {'mode': 'none'}, 'protocol_sha256': 'old'}
    trainer = SimpleNamespace(wrap_target=old_wrap, make_protocol=old_protocol)
    old_validator = training_resume.validate_output
    with pytest.raises(RuntimeError, match='test stop'):
        with local.local_runtime(trainer):
            assert trainer.wrap_target is local.preserve_single_gpu_target
            result = trainer.make_protocol()
            assert result['local_execution']['visual_weighting_enabled'] is False
            expected = dict(result)
            expected.pop('protocol_sha256')
            assert result['protocol_sha256'] == hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
            raise RuntimeError('test stop')
    assert trainer.wrap_target is old_wrap and trainer.make_protocol is old_protocol
    assert training_resume.validate_output is old_validator
