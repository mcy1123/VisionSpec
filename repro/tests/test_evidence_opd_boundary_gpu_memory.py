"""CPU-only composition and final protocol identity checks."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_h100 import train_video_evidence_opd_boundary_gpu_memory as entry
from local_h100 import train_video_evidence_opd as evidence
from shared import training_resume
from video_caption import train_video_9k as trainer
from test_local_opd_gpu_memory import tensor, optimizer, protocol


@pytest.mark.parametrize('mode', ['text', 'visual'])
@pytest.mark.parametrize('fail', [False, True])
def test_composition_final_sha_freeze_before_gpu_factory_and_restore(tmp_path, monkeypatch, mode, fail):
    path = tmp_path / 'head'; path.mkdir()
    (path / 'config.json').write_text(json.dumps({'dflash_config': dict(
        evidence_mode=mode, evidence_time_encoding='off', target_layer_ids=[1, 9, 17, 25, 33])}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test')
    monkeypatch.setattr(trainer, 'make_protocol', protocol)
    events = []
    monkeypatch.setattr(evidence.evidence, 'freeze_time_after_unfreeze', lambda head: events.append('freeze'))
    def factory(head, **kwargs):
        assert events == ['freeze']
        assert kwargs == {'where': 'gpu', 'lr': 1e-5}
        events.append('gpu_factory')
        return optimizer()
    monkeypatch.setattr(trainer, 'MasterWeightAdamW', factory)
    def forbidden(*args, **kwargs): raise AssertionError('CPU composition must not query CUDA')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', forbidden)
    before = (trainer.make_protocol, trainer.MasterWeightAdamW, trainer.wrap_target,
              trainer.score_training_trace, trainer.adapters.draft, trainer.adapters.collect,
              training_resume.save_atomic_checkpoint)
    def main(argv):
        args = trainer.parser().parse_args(argv)  # --evidence-mode stripped exactly once.
        result = trainer.make_protocol(args, {}, {})
        trainer.MasterWeightAdamW(SimpleNamespace(parameters=lambda: iter([tensor()])), where='gpu', lr=1e-5)
        assert result['gpu_memory_policy']['selected_evidence_mode'] == mode
        assert result['evidence_architecture']['time_encoding'] == 'off'
        assert result['evidence_prompt_boundary']['request_mask_and_identity'] == 'unchanged complete prompt'
        assert result['hashes']['combined_evidence_opd_entry'] == training_resume.file_hash(entry.__file__)
        check = dict(result); recorded = check.pop('protocol_sha256')
        assert recorded == hashlib.sha256(json.dumps(check, sort_keys=True).encode()).hexdigest()
        if fail: raise RuntimeError('injected')
        return result
    monkeypatch.setattr(trainer, 'main', main)
    argv = ['--pool', 'unused', '--schedule', 'unused', '--init-from', str(path),
            '--out', str(tmp_path / 'run'), '--updates', '2', '--master-weights', 'gpu',
            '--evidence-mode', mode, '--dry-run']
    if fail:
        with pytest.raises(RuntimeError, match='injected'): entry.main(argv)
    else:
        entry.main(argv)
    after = (trainer.make_protocol, trainer.MasterWeightAdamW, trainer.wrap_target,
             trainer.score_training_trace, trainer.adapters.draft, trainer.adapters.collect,
             training_resume.save_atomic_checkpoint)
    assert before == after and events == ['freeze', 'gpu_factory']
    assert not (tmp_path / 'run').exists()
