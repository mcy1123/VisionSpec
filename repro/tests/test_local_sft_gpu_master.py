"""CPU-only placement, wrapper composition and atomic identity regressions."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import train_video_sft as base
from local_h100 import train_video_sft_pilot as pilot
from local_h100 import train_video_sft_gpu_master as entry
from local_h100 import train_video_evidence_sft as evidence_entry
from evaluation.eval_video_opd import evaluation_artifacts
from shared import training_resume


def fake_head(*devices):
    return SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device(d)) for d in devices]))


def protocol_value():
    return dict(entry='local_h100/train_video_sft_pilot.py', checkpoint_storage='weights_only_pilot',
                resume_supported=False, optimizer_state_saved=False, pilot_only=True,
                config={'qa_per_update': 4}, optimizer={'where': 'cpu', 'clip': 1.0},
                hashes={'initial_head_weights': {'model': 'a'}, 'target_weights': {'model': 'b'},
                        'pool': 'p', 'schedule': 's'}, protocol_sha256='old')


def fake_optimizer():
    return SimpleNamespace(where='gpu', _dev=torch.device('cuda:0'),
                           master={'w': SimpleNamespace(device=torch.device('cuda:0'))})


def test_factory_preserves_arguments_freeze_order_and_restores_on_error(monkeypatch):
    calls = []
    model = fake_head('cuda:0', 'cuda:0')
    model.frozen = False
    def optimizer(head, **kwargs):
        assert head.frozen
        calls.append((head, kwargs))
        return SimpleNamespace(where=kwargs['where'], _dev=torch.device('cuda:0'))
    def freeze(head):
        head.frozen = True
    monkeypatch.setattr(base, 'MasterWeightAdamW', optimizer)
    monkeypatch.setattr(evidence_entry.evidence, 'freeze_time_after_unfreeze', freeze)
    before = (base.main, base.hidden_loss, base.anchor_plan, base.get_teacher,
              base.MasterWeightAdamW, base.build_protocol, base.checkpoint, base.write_json_exclusive)
    options = dict(lr=3e-6, betas=(.9, .999), weight_decay=.01, where='cpu', clip=1.0, eps=1e-8)
    with pytest.raises(RuntimeError, match='injected'):
        with pilot.pilot_runtime(), entry.gpu_master_runtime():
            assert (base.main, base.hidden_loss, base.anchor_plan, base.get_teacher) == before[:4]
            assert base.MasterWeightAdamW.__module__ == optimizer.__module__
            with evidence_entry.evidence_runtime():
                result = base.MasterWeightAdamW(model, **options)
                assert result.where == 'gpu'
                assert calls == [(model, dict(options, where='gpu'))]
                raise RuntimeError('injected')
    assert (base.main, base.hidden_loss, base.anchor_plan, base.get_teacher,
            base.MasterWeightAdamW, base.build_protocol, base.checkpoint, base.write_json_exclusive) == before
    for model in (fake_head(), fake_head('cpu'), fake_head('cuda:0', 'cuda:1')):
        with pytest.raises(ValueError, match='one CUDA'):
            entry.head_cuda_device(model)


def test_protocol_and_checkpoint_identity_sha_no_optimizer_state(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'ALLOWED_ROOTS', (tmp_path,))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 123)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 456)
    original = protocol_value()
    with pilot.pilot_runtime(), monkeypatch.context() as scoped:
        scoped.setattr(base, 'build_protocol', lambda *args: copy.deepcopy(original))
        with entry.gpu_master_runtime():
            protocol = base.build_protocol(None, {}, {}, ['GPU-test'])
            assert protocol['optimizer_where'] == protocol['config']['optimizer_where'] == 'gpu'
            assert protocol['optimizer'] == dict(original['optimizer'], where='gpu')
            assert protocol['hashes']['gpu_master_wrapper'] == training_resume.file_hash(entry.__file__)
            assert protocol['gpu_master_policy']['cpu_gpu_bitwise_equivalence_claimed'] is False
            payload = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
            assert protocol['protocol_sha256'] == base.digest(payload)
            class Head(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.weight = torch.nn.Parameter(torch.ones(2))
                def save_pretrained(self, path):
                    (path / 'config.json').write_text('{}')
                    save_file(self.state_dict(), str(path / 'model.safetensors'))
            class Optimizer:
                where = 'gpu'
                _dev = torch.device('cuda:0')
                master = {'w': SimpleNamespace(device=torch.device('cuda:0'))}
                def state_dict(self):
                    raise AssertionError('must never serialize optimizer state')
            root = tmp_path / 'run'; root.mkdir()
            writer = base.write_json_exclusive
            checkpoint = Path(base.checkpoint(root, 2, Head(), Optimizer(), protocol))
            assert base.write_json_exclusive is writer
            assert not (checkpoint / 'training_state.pt').exists()
            assert json.loads((checkpoint / 'TRAINING_PROTOCOL.json').read_text()) == protocol
            sha = json.loads((checkpoint / 'SHA256.json').read_text())
            assert all(training_resume.file_hash(checkpoint / name) == value for name, value in sha.items())
            target = tmp_path / 'target'; target.mkdir(); (target / 'config.json').write_text('{}')
            identity = evaluation_artifacts(target, checkpoint)['checkpoint_identity']
            assert identity['optimizer_where'] == 'gpu'
            assert identity['gpu_master_diagnostics']['master_devices'] == ['cuda:0']
            assert identity['gpu_master_diagnostics']['cuda']['cuda:0'] == {
                'max_memory_allocated_bytes': 123, 'max_memory_reserved_bytes': 456}
            assert identity['protocol_sha256'] == protocol['protocol_sha256']
            assert identity['optimizer_state_saved'] is False and identity['resume_supported'] is False
            with pytest.raises(FileExistsError):
                base.checkpoint(root, 2, Head(), Optimizer(), protocol)
            assert base.write_json_exclusive is writer
    assert original['optimizer']['where'] == 'cpu'


def test_saver_exception_restores_writer_and_missing_pilot_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 123)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 456)
    def failure(*args):
        raise RuntimeError('save failed')
    monkeypatch.setattr(base, 'checkpoint', failure)
    monkeypatch.setattr(base, 'build_protocol', lambda *args: {})
    before = (base.MasterWeightAdamW, base.build_protocol, base.checkpoint, base.write_json_exclusive)
    with entry.gpu_master_runtime():
        with pytest.raises(ValueError, match='pilot_runtime'):
            base.build_protocol(None)
        with pytest.raises(RuntimeError, match='save failed'):
            base.checkpoint('unused', 1, None, fake_optimizer(),
                            {'optimizer_where': 'gpu', 'optimizer': {'where': 'gpu'}})
        assert base.write_json_exclusive is before[-1]
    assert (base.MasterWeightAdamW, base.build_protocol, base.checkpoint, base.write_json_exclusive) == before


def test_real_protocol_dry_run_preserves_target_hashes_without_cuda_or_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'ALLOWED_ROOTS', (tmp_path,))
    def forbidden(*args, **kwargs):
        raise AssertionError('protocol dry run must not touch CUDA')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', forbidden)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    for name in ('target', 'head'):
        path = tmp_path / name; path.mkdir()
        (path / 'config.json').write_text('{}')
        (path / 'model.safetensors').write_bytes(name.encode())
    for name in ('pool', 'schedule'):
        (tmp_path / name).write_text(name)
    args = SimpleNamespace(target=str(tmp_path / 'target'), init_from=str(tmp_path / 'head'),
                           pool=str(tmp_path / 'pool'), schedule=str(tmp_path / 'schedule'),
                           out=str(tmp_path / 'run'), teacher_cache=str(tmp_path / 'cache'),
                           dry_run=True, clip=1.0, max_new_tokens=1024, qa_per_update=4)
    expected = base.build_protocol(args, {}, {}, ['GPU-test'])
    with pilot.pilot_runtime(), entry.gpu_master_runtime():
        actual = base.build_protocol(args, {}, {}, ['GPU-test'])
    assert actual['hashes']['target_weights'] == expected['hashes']['target_weights']
    assert actual['hashes']['master_optim'] == expected['hashes']['master_optim']
    assert actual['objective'] == expected['objective']
    assert actual['optimizer']['where'] == 'gpu'
    assert not (tmp_path / 'run').exists() and not (tmp_path / 'cache').exists()
