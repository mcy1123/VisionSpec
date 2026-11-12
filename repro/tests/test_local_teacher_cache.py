"""CPU tests for train-only partitioning and exact SFT teacher cache reuse."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import generate_teacher_cache as collector
from local_h100 import train_video_sft as sft


def rows():
    return [dict(task=str(i), split='train', prompt='Describe.') for i in range(9)]


def test_shards_are_disjoint_complete_stable_and_train_only():
    partitions = [collector.partition_train(rows()[::-1], 4, i) for i in range(4)]
    flattened = [r['task'] for p in partitions for r in p]
    assert sorted(flattened) == [str(i) for i in range(9)]
    assert len(set(flattened)) == 9
    assert partitions[0] == collector.partition_train(rows(), 4, 0)
    for split in ('dev', 'test', None):
        bad = rows()
        bad[-1]['split'] = split
        with pytest.raises(ValueError, match='train-only'):
            collector.partition_train(bad, 4, 0)
    with pytest.raises(ValueError, match='unique'):
        collector.partition_train(rows() + [rows()[0]], 4, 0)
    for count, shard in ((0, 0), (2, -1), (2, 2), (True, 0)):
        with pytest.raises(ValueError):
            collector.partition_train(rows(), count, shard)


def test_weights_hashed_once_and_changed_inode_or_contents_invalidates(tmp_path, monkeypatch):
    target, identities = tmp_path / 'model', tmp_path / 'identity'
    target.mkdir()
    weight = target / 'model.safetensors'
    weight.write_bytes(b'original')
    called = []
    original = collector.training_resume.file_hash
    def counted(path):
        if Path(path).name == weight.name:
            called.append(str(path))
        return original(path)
    monkeypatch.setattr(collector.training_resume, 'file_hash', counted)
    first, provenance = collector.target_weight_hashes(target, identities)
    again, provenance2 = collector.target_weight_hashes(target, identities)
    assert first == again and len(called) == 1
    assert not provenance['verified_identity_reused'] and provenance2['verified_identity_reused']
    weight.write_bytes(b'changed!')
    changed, _ = collector.target_weight_hashes(target, identities)
    assert changed != first and len(called) == 2
    replacement = target / 'replacement'
    replacement.write_bytes(b'changed!')
    replacement.replace(weight)
    equal_content, p = collector.target_weight_hashes(target, identities)
    assert equal_content == changed and not p['verified_identity_reused'] and len(called) == 3


def test_unverified_local_hash_cache_is_rejected(tmp_path):
    target = tmp_path / 'model'
    target.mkdir()
    (target / 'model.safetensors').write_bytes(b'weights')
    _, info = collector.target_weight_hashes(target, tmp_path / 'identity')
    record_path = Path(info['path'])
    record = json.loads(record_path.read_text())
    record['weight_sha256']['model.safetensors'] = 'not-a-sha'
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='identity cache'):
        collector.target_weight_hashes(target, tmp_path / 'identity')


def test_collector_key_is_exactly_sft_key_and_cached_result_needs_no_target(tmp_path):
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'config.json').write_text('{"model_type":"test"}')
    protocol = collector.teacher_protocol(target, {'model.safetensors': 'weights'}, 1024)
    training_protocol = {'config': {'max_new_tokens': 1024, 'lr': 1e-5},
                         'hashes': dict(protocol['hashes'], trainer='train-code')}
    inputs = {'input_ids': torch.tensor([[1, 2, 3]]),
              'pixel_values_videos': torch.arange(12).reshape(3, 4).float(),
              'video_grid_thw': torch.tensor([[1, 2, 2]])}
    key, identity = sft.teacher_key(inputs, protocol, {9})
    assert sft.teacher_key(inputs, training_protocol, {9}) == (key, identity)
    tokens = [4, 5, 9]
    sft.write_json_exclusive(tmp_path / f'{key}.json',
        dict(identity=identity, prompt_ids=[1, 2, 3], gen_ids=tokens,
             gen_ids_sha256=sft.digest(tokens)))
    result, hit, actual = sft.get_teacher(None, inputs, training_protocol, tmp_path, {9})
    assert (result, hit, actual) == (tokens, True, key)


def test_eos_truncation_and_first_token_diagnostics():
    assert collector.output_summary([1, 2, 9], {9}, 1024) == {
        'generated_tokens': 3, 'hit_eos': True, 'eos_token_id': 9,
        'truncated': False, 'usable_for_sft': True}
    assert collector.output_summary([9], {9}, 1024)['usable_for_sft'] is False
    assert collector.output_summary([1, 2], {9}, 2)['truncated'] is True
    for bad, budget in (([], 10), ([1], 10), ([9, 1], 10), ([1, 2, 3], 2)):
        with pytest.raises(ValueError):
            collector.output_summary(bad, {9}, budget)
