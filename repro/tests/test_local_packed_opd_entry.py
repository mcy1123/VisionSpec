"""CPU-only future-entry composition and cleanup; no real training launched."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import train_video_opd_packed_experimental as entry
from shared import training_resume
from video_caption import train_video_9k as trainer


@pytest.mark.parametrize('mode', ['none', 'visual'])
@pytest.mark.parametrize('fail', [False, True])
def test_experimental_main_protocol_no_cuda_and_restores_every_layer(tmp_path, monkeypatch, mode, fail):
    path = tmp_path / 'head'; path.mkdir()
    (path / 'config.json').write_text(json.dumps({'dflash_config': dict(
        evidence_mode=mode, evidence_time_encoding='off', target_layer_ids=[1, 9, 17, 25, 33])}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test')
    monkeypatch.setattr(trainer, 'make_protocol', lambda *args, **kwargs: {'optimizer': {}, 'hashes': {}})
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU entry proof must not inspect CUDA')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', forbidden)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', forbidden)
    original = (trainer.wrap_target, trainer.MasterWeightAdamW, trainer.make_protocol,
                trainer.score_training_trace, trainer.backward_update, trainer.adapters.collect,
                training_resume.save_atomic_checkpoint, training_resume.validate_output)
    def main(argv):
        args = trainer.parser().parse_args(argv)
        assert args.updates == 2 and trainer.backward_update is original[4]
        if mode == 'none':
            assert trainer.adapters.collect is original[5]
        result = trainer.make_protocol(args, {}, {})
        assert result['packed_opd_score']['pack_size'] == 4
        assert result['optimizer']['where'] == 'gpu'
        assert ('evidence_architecture' in result) == (mode == 'visual')
        assert result['protocol_sha256'] == hashlib.sha256(json.dumps(
            {k: v for k, v in result.items() if k != 'protocol_sha256'}, sort_keys=True).encode()).hexdigest()
        if fail:
            raise RuntimeError('injected entry failure')
        return result
    monkeypatch.setattr(trainer, 'main', main)
    argv = ['--pool', 'unused', '--schedule', 'unused', '--init-from', str(path),
            '--out', str(tmp_path / 'run'), '--updates', '2', '--master-weights', 'gpu',
            '--evidence-mode', 'auto', '--packed-anchors', '4', '--e2e-tv-weight', '0',
            '--loss-normalization', 'global_pooled', '--dry-run']
    if fail:
        with pytest.raises(RuntimeError, match='injected entry failure'):
            entry.main(argv)
    else:
        result = entry.main(argv)
        assert result['hashes']['experimental_packed_entry'] == training_resume.file_hash(entry.__file__)
    assert not (tmp_path / 'run').exists()
    assert (trainer.wrap_target, trainer.MasterWeightAdamW, trainer.make_protocol,
            trainer.score_training_trace, trainer.backward_update, trainer.adapters.collect,
            training_resume.save_atomic_checkpoint, training_resume.validate_output) == original
