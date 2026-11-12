"""CPU checks that audit evidence is exact and missing logs stay missing."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import audit_evidence_engineering as audit


def test_scalar_tensor_digest_uses_same_bytes():
    assert audit.tensor_sha(torch.tensor(0.5)) == audit.tensor_sha(torch.tensor([0.5]))


def test_actual_prompt_sha_and_full_visual_mask(tmp_path):
    (tmp_path / 'teacher_cache').mkdir()
    target = tmp_path / 'target'; target.mkdir()
    (target / 'config.json').write_text(json.dumps({'video_token_id': 7}))
    ids = torch.tensor([[1, 7, 7, 2]])
    value = dict(prompt_ids=ids[0].tolist(), identity={'encoded_inputs': {
        'input_ids': {'shape': [1, 4], 'sha256': audit.tensor_sha(ids)}}})
    path = tmp_path / 'teacher_cache/key.json'
    path.write_text(json.dumps(value))
    result = audit.cached_prompt_audit(tmp_path, 'key', target)
    assert result['visual_tokens'] == 2
    assert result['last_visual_position'] == 2
    assert result['all_visual_known_before_last_prompt_root'] is True
    assert result['visual_mask_sha256'] == audit.tensor_sha(ids[0].eq(7))
    value['prompt_ids'][-1] = 7
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='actual input SHA'):
        audit.cached_prompt_audit(tmp_path, 'key', target)


def test_sft_forward_stats_never_become_gradient_evidence():
    result = audit.log_audit([dict(update=2, loss=1.0, gradient_norm=3.0,
                                  qa=[dict(task='a', evidence={'residual_rms': .1})])])
    assert result[0]['per_parameter_gradient_observation'] == 'NOT_RECORDED'
    assert 'gradient_norms' not in result[0]


def test_opd_explicit_gradient_record_preserved():
    stats = {'gradient_norms': {'query.weight': .1}, 'missing_gradients': ['time.weight']}
    result = audit.log_audit([dict(update=2, loss=1.0, gradient_norm=3.0,
        tasks=['a'], input_tokens_per_qa=[4], visual_tokens_per_qa=[2], evidence=stats)])
    assert result[0]['evidence'] == stats
    assert result[0]['per_parameter_gradient_observation'] == 'RECORDED_IN_SHARED_OPD_DIAGNOSTICS'
