"""CPU tests for read-only OPD GPU-memory accounting and atomic metadata."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import train_video_opd_gpu_memory as entry
from shared import training_resume
from evaluation.eval_video_opd import evaluation_artifacts


def tensor(device='cuda:0', *, grad=None):
    return SimpleNamespace(device=torch.device(device), dtype=torch.float32,
                           grad=grad, numel=lambda: 2)


def optimizer(initialized=True):
    states = {'w': {'exp_avg': tensor(), 'exp_avg_sq': tensor(), 'step': torch.tensor(1.)}} if initialized else {}
    return SimpleNamespace(where='gpu', _dev=torch.device('cuda:0'),
                           master={'w': tensor(grad=tensor() if initialized else None)},
                           _opt=SimpleNamespace(state=states))


def protocol(args, *rest, **kwargs):
    return dict(optimizer={'name': 'AdamW'}, hashes={}, config={'master_weights': args.master_weights},
                protocol_sha256='before')


def metrics(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 123456)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 234567)


def test_factory_only_captures_unchanged_arguments_and_restores_on_exception(monkeypatch):
    calls = []
    value = optimizer()
    head = SimpleNamespace(parameters=lambda: iter([tensor()]))
    def factory(module, **kwargs):
        calls.append((module, kwargs))
        return value
    trainer = SimpleNamespace(MasterWeightAdamW=factory, make_protocol=protocol)
    original_save = training_resume.save_atomic_checkpoint
    options = dict(where='gpu', lr=1e-5, betas=(.9, .999), weight_decay=.01, clip=1.)
    with pytest.raises(RuntimeError, match='injected'):
        with entry.memory_runtime(trainer, 'none') as captured:
            assert trainer.MasterWeightAdamW(head, **options) is value
            assert calls == [(head, options)] and captured['optimizer'] is value
            with pytest.raises(RuntimeError, match='exactly one'):
                trainer.MasterWeightAdamW(head, **options)
            raise RuntimeError('injected')
    assert trainer.MasterWeightAdamW is factory and trainer.make_protocol is protocol
    assert training_resume.save_atomic_checkpoint is original_save


def test_protocol_dry_run_never_reads_cuda_and_keeps_final_evidence_hash(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run protocol must not inspect CUDA')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', forbidden)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', forbidden)
    trainer = SimpleNamespace(MasterWeightAdamW=forbidden, make_protocol=protocol)
    with entry.memory_runtime(trainer, 'visual') as captured:
        result = trainer.make_protocol(SimpleNamespace(master_weights='gpu'))
        assert result['optimizer']['where'] == 'gpu'
        assert result['gpu_memory_policy']['optimizer_state_saved'] is True
        assert result['gpu_memory_policy']['resume_supported'] is False
        assert result['hashes']['local_gpu_memory_entry'] == training_resume.file_hash(entry.__file__)
        result['protocol_sha256'] = 'inner evidence wrapper final hash'
        assert captured['protocol']['protocol_sha256'] == result['protocol_sha256']
        with pytest.raises(ValueError, match='master-weights gpu'):
            trainer.make_protocol(SimpleNamespace(master_weights='cpu'))


def test_atomic_full_checkpoint_includes_actual_optimizer_and_peak_metadata(tmp_path, monkeypatch):
    metrics(monkeypatch)
    monkeypatch.setattr(training_resume, 'validate_output', lambda path: None)
    value = optimizer()
    trainer = SimpleNamespace(MasterWeightAdamW=lambda *args, **kwargs: value, make_protocol=protocol)
    fake_head = SimpleNamespace(parameters=lambda: iter([tensor()]))
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(2))
        def save_pretrained(self, path):
            (path / 'config.json').write_text('{}')
            save_file(self.state_dict(), str(path / 'model.safetensors'))
    original_save = training_resume.save_atomic_checkpoint
    with entry.memory_runtime(trainer, 'none'):
        result = trainer.make_protocol(SimpleNamespace(master_weights='gpu'))
        trainer.MasterWeightAdamW(fake_head, where='gpu')
        state = {'protocol_sha256': result['protocol_sha256'], 'head': Head().state_dict(),
                 'master_optimizer': {'sentinel': torch.ones(2)}, 'unchanged': 42}
        metadata = {'protocol_sha256': result['protocol_sha256'], 'optimizer_step': 2}
        dest = tmp_path / 'step2'
        training_resume.save_atomic_checkpoint(dest, Head(), state, metadata=metadata)
        assert (dest / 'training_state.pt').exists()
        saved = torch.load(dest / 'training_state.pt', weights_only=False)
        assert saved['unchanged'] == 42
        torch.testing.assert_close(saved['master_optimizer']['sentinel'], state['master_optimizer']['sentinel'])
        assert 'gpu_memory_diagnostics' not in metadata  # caller identity was not mutated
        target = tmp_path / 'target'; target.mkdir(); (target / 'config.json').write_text('{}')
        identity = evaluation_artifacts(target, dest)['checkpoint_identity']
        diag = identity['gpu_memory_diagnostics']
        assert diag['master_devices'] == diag['adam_moment_devices'] == ['cuda:0']
        assert diag['adam_step_devices'] == ['cpu']  # normal non-capturable Adam step scalars
        assert diag['master_gradient_tensor_count'] == diag['adam_initialized_parameter_count'] == 1
        assert diag['adam_moment_tensor_count'] == 2
        assert diag['cuda']['cuda:0']['max_memory_allocated_bytes'] == 123456
        assert identity['gpu_memory_protocol_sha256'] == result['protocol_sha256']
        hashes = json.loads((dest / 'SHA256.json').read_text())
        assert all(training_resume.file_hash(dest / name) == sha for name, sha in hashes.items())
        with pytest.raises(FileExistsError):
            training_resume.save_atomic_checkpoint(dest, Head(), state, metadata=metadata)
        with pytest.raises(ValueError, match='identity mismatch'):
            training_resume.save_atomic_checkpoint(tmp_path / 'bad', Head(), state,
                                                  metadata={'protocol_sha256': 'wrong'})
        assert not (tmp_path / 'bad').exists()
    assert training_resume.save_atomic_checkpoint is original_save


def test_lazy_adam_states_are_reported_and_cpu_master_is_rejected(monkeypatch):
    metrics(monkeypatch)
    cold = entry.optimizer_snapshot(optimizer(initialized=False))
    assert cold['adam_initialized_parameter_count'] == cold['adam_moment_tensor_count'] == 0
    assert cold['master_gradient_tensor_count'] == 0
    value = optimizer(); value.master['w'].device = torch.device('cpu')
    with pytest.raises(ValueError, match='CUDA master'):
        entry.optimizer_snapshot(value)


def test_evidence_selection_requires_explicit_time_off(tmp_path):
    path = tmp_path / 'head'; path.mkdir()
    config = {'dflash_config': {'evidence_mode': 'none'}}
    def save():
        (path / 'config.json').write_text(json.dumps(config))
    save()
    assert entry.select_evidence(path, 'auto') == entry.select_evidence(path, 'plain') == 'none'
    with pytest.raises(ValueError, match='disagrees'):
        entry.select_evidence(path, 'visual')
    config['dflash_config'].update(evidence_mode='visual', target_layer_ids=[1, 9, 17, 25, 33])
    save()
    with pytest.raises(ValueError, match='time_encoding=off'):
        entry.select_evidence(path, 'auto')
    config['dflash_config']['evidence_time_encoding'] = 'off'; save()
    assert entry.select_evidence(path, 'visual') == 'visual'


@pytest.mark.parametrize('mode', ['none', 'visual'])
def test_main_selects_plain_or_time_off_evidence_without_touching_cuda(tmp_path, monkeypatch, mode):
    from video_caption import train_video_9k as trainer
    path = tmp_path / 'head'; path.mkdir()
    config = {'dflash_config': dict(evidence_mode=mode, evidence_time_encoding='off',
                                   target_layer_ids=[1, 9, 17, 25, 33])}
    (path / 'config.json').write_text(json.dumps(config))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test')
    monkeypatch.setattr(trainer, 'make_protocol', protocol)
    def forbidden(*args, **kwargs):
        raise AssertionError('no CUDA access in this CPU protocol/composition test')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', forbidden)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', forbidden)
    events = []
    monkeypatch.setattr(entry.evidence.evidence, 'freeze_time_after_unfreeze', lambda head: events.append('freeze'))
    def factory(head, **kwargs):
        assert events == (['freeze'] if mode == 'visual' else [])
        return optimizer()
    monkeypatch.setattr(trainer, 'MasterWeightAdamW', factory)
    original = (trainer.MasterWeightAdamW, trainer.make_protocol, trainer.wrap_target,
                trainer.adapters.collect, trainer.score_training_trace, training_resume.save_atomic_checkpoint)
    def inherited_main(argv):
        args = trainer.parser().parse_args(argv)  # custom CLI flag was removed
        result = trainer.make_protocol(args, {}, {})
        trainer.MasterWeightAdamW(SimpleNamespace(parameters=lambda: iter([tensor()])), where='gpu')
        return result
    monkeypatch.setattr(trainer, 'main', inherited_main)
    result = entry.main(['--pool', 'unused', '--schedule', 'unused', '--init-from', str(path),
                         '--out', str(tmp_path / 'run'), '--updates', '2', '--master-weights', 'gpu',
                         '--evidence-mode', 'auto', '--dry-run'])
    assert result['gpu_memory_policy']['selected_evidence_mode'] == mode
    assert ('evidence_architecture' in result) == (mode == 'visual')
    assert (trainer.MasterWeightAdamW, trainer.make_protocol, trainer.wrap_target,
            trainer.adapters.collect, trainer.score_training_trace, training_resume.save_atomic_checkpoint) == original
    assert not (tmp_path / 'run').exists()
