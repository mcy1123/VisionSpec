"""CPU tests of weights-only I/O with unchanged SFT training functions."""
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
from shared import training_resume
from evaluation.eval_video_opd import evaluation_artifacts


def test_only_io_functions_are_replaced_and_everything_restored():
    original = (base.main, base.hidden_loss, base.anchor_plan, base.get_teacher,
                base.MasterWeightAdamW, base.build_protocol, base.checkpoint)
    with pytest.raises(RuntimeError):
        with pilot.pilot_runtime():
            assert (base.main, base.hidden_loss, base.anchor_plan, base.get_teacher,
                    base.MasterWeightAdamW) == original[:5]
            assert base.checkpoint is pilot.weights_only_checkpoint
            raise RuntimeError('stop')
    assert (base.main, base.hidden_loss, base.anchor_plan, base.get_teacher,
            base.MasterWeightAdamW, base.build_protocol, base.checkpoint) == original


def test_checkpoint_has_eval_identity_no_optimizer_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'ALLOWED_ROOTS', (tmp_path,))
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(2))
        def save_pretrained(self, path):
            (path / 'config.json').write_text('{}')
            save_file(self.state_dict(), str(path / 'model.safetensors'))
    class ForbiddenOptimizer:
        def state_dict(self):
            raise AssertionError('weights-only checkpoint must not touch optimizer state')
    root = tmp_path / 'run'; root.mkdir()
    protocol = dict(protocol_sha256='p', config={'qa_per_update':4},
                    hashes={'initial_head_weights':{'model':'a'}, 'target_weights':{'model':'b'},
                            'pool':'pool','schedule':'schedule'}, optimizer_state_saved=False)
    ckpt = Path(pilot.weights_only_checkpoint(root,20,Head(),ForbiddenOptimizer(),protocol))
    assert not (ckpt/'training_state.pt').exists()
    assert json.loads((ckpt/'TRAINING_PROTOCOL.json').read_text()) == protocol
    target=tmp_path/'target'; target.mkdir(); (target/'config.json').write_text('{}')
    artifacts=evaluation_artifacts(target,ckpt)
    identity=artifacts['checkpoint_identity']
    assert identity['optimizer_state_saved'] is False and identity['resume_supported'] is False
    assert identity['optimizer_step']==20 and identity['qa_exposures']==80
    with pytest.raises(FileExistsError):
        pilot.weights_only_checkpoint(root,20,Head(),ForbiddenOptimizer(),protocol)


def test_target_hash_reuse_preserves_exact_dictionary_and_hashes_other_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(base,'ALLOWED_ROOTS',(tmp_path,))
    target=tmp_path/'target'; target.mkdir(); (target/'config.json').write_text('{}')
    weight=target/'model.safetensors'; weight.write_bytes(b'target-bytes')
    head=tmp_path/'head'; head.mkdir(); (head/'config.json').write_text('{}')
    (head/'model.safetensors').write_bytes(b'head-bytes')
    pool=tmp_path/'pool';pool.write_bytes(b'pool')
    schedule=tmp_path/'schedule';schedule.write_bytes(b'schedule')
    args=SimpleNamespace(target=str(target),init_from=str(head),pool=str(pool),schedule=str(schedule),
                         teacher_cache=str(tmp_path/'cache'),out=str(tmp_path/'run'),dry_run=False,
                         clip=1.0,max_new_tokens=1024,qa_per_update=4)
    original_hash=training_resume.file_hash
    direct=base.build_protocol(args,{}, {},['GPU-test'])
    actual=pilot.pilot_protocol(base.build_protocol,args,{}, {},['GPU-test'])
    assert actual['hashes']['target_weights']==direct['hashes']['target_weights']
    assert actual['hashes']['initial_head_weights']==direct['hashes']['initial_head_weights']
    assert actual['hashes']['pool']==direct['hashes']['pool']
    assert actual['hashes']['trainer']==direct['hashes']['trainer']
    assert actual['objective']==direct['objective'] and actual['optimizer']==direct['optimizer']
    assert actual['optimizer_state_saved'] is False and actual['pilot_only'] is True
    second=pilot.pilot_protocol(base.build_protocol,args,{}, {},['GPU-test'])
    assert second['target_weight_verification']['verified_identity_reused'] is True
    assert training_resume.file_hash is original_hash
    def failure(*args):
        raise RuntimeError('protocol failure')
    with pytest.raises(RuntimeError):
        pilot.pilot_protocol(failure,args,{}, {},['GPU-test'])
    assert training_resume.file_hash is original_hash
    args.teacher_cache=str(tmp_path/'dry_cache');args.dry_run=True
    pilot.pilot_protocol(base.build_protocol,args,{}, {},['GPU-test'])
    assert not Path(args.teacher_cache).exists()
