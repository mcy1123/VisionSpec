"""CPU-only test of the final evidence/GPU-master context order and cleanup."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import train_video_sft as base
from local_h100 import train_video_evidence_sft as evidence
from local_h100 import train_video_evidence_sft_gpu_master as entry
from qa_opd_fair import videoencode
from shared import training_resume


def symbols():
    return (base.MasterWeightAdamW, base.build_protocol, base.checkpoint,
            base.validate_head, base.hidden_loss, base.write_json_exclusive,
            videoencode.encode_inputs, training_resume.file_hash)


@pytest.mark.parametrize('fail', [False, True])
def test_main_uses_freeze_then_gpu_factory_and_restores_every_layer(monkeypatch, fail):
    events = []
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device('cuda:0'))]))
    def freeze(head):
        assert head is model
        events.append('freeze')
    def optimizer(head, **kwargs):
        assert events == ['freeze'] and head is model and kwargs['where'] == 'gpu'
        events.append('gpu factory')
        return SimpleNamespace(where='gpu', _dev=torch.device('cuda:0'))
    monkeypatch.setattr(evidence.evidence, 'freeze_time_after_unfreeze', freeze)
    monkeypatch.setattr(base, 'MasterWeightAdamW', optimizer)
    original = symbols()
    argv = ['--updates', '2', '--dry-run']
    def main(arguments):
        assert arguments is argv
        assert base.validate_head is evidence.validate_head
        assert base.checkpoint is not original[2] and base.hidden_loss is not original[4]
        base.MasterWeightAdamW(model, where='cpu', lr=1e-5)
        if fail:
            raise RuntimeError('injected main failure')
        return 'inherited main result'
    monkeypatch.setattr(base, 'main', main)
    if fail:
        with pytest.raises(RuntimeError, match='injected main failure'):
            entry.main(argv)
    else:
        assert entry.main(argv) == 'inherited main result'
    assert events == ['freeze', 'gpu factory']
    assert symbols() == original


def test_composed_protocol_keeps_evidence_gpu_policy_and_its_source_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'ALLOWED_ROOTS', (tmp_path,))
    for name in ('target', 'head'):
        path = tmp_path / name; path.mkdir()
        config = {'dflash_config': dict(evidence_mode='visual', evidence_time_encoding='off',
                                       evidence_dim=256, evidence_max_tokens=128,
                                       evidence_spatial_bins=2)} if name == 'head' else {}
        (path / 'config.json').write_text(json.dumps(config))
        (path / 'model.safetensors').write_bytes(name.encode())
    for name in ('pool', 'schedule'):
        (tmp_path / name).write_text(name)
    args = SimpleNamespace(target=str(tmp_path / 'target'), init_from=str(tmp_path / 'head'),
                           pool=str(tmp_path / 'pool'), schedule=str(tmp_path / 'schedule'),
                           out=str(tmp_path / 'run'), teacher_cache=str(tmp_path / 'cache'),
                           dry_run=True, clip=1.0, max_new_tokens=1024, qa_per_update=4)
    original = symbols()
    with entry.combined_runtime():
        protocol = base.build_protocol(args, {}, {}, ['GPU-test'])
    assert symbols() == original
    assert protocol['entry'] == 'local_h100/train_video_evidence_sft_gpu_master.py'
    assert protocol['hashes']['evidence_gpu_master_entry'] == training_resume.file_hash(entry.__file__)
    assert protocol['optimizer']['where'] == 'gpu'
    assert protocol['evidence_architecture']['time_encoding'] == 'off'
    assert protocol['resume_supported'] is False and protocol['optimizer_state_saved'] is False
    assert protocol['protocol_sha256'] == base.digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'})
    assert not (tmp_path / 'run').exists() and not (tmp_path / 'cache').exists()
