import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import summarize_long_dev as summary


@pytest.fixture
def encoding(tmp_path):
    path = tmp_path / 'profile.json'
    profile = {'video_size': {'shortest_edge': 4096, 'longest_edge': 50331648}}
    path.write_text(json.dumps(profile))
    digest = summary.warm.file_sha(path)
    audit = dict(video='one', n_frames=512, input_ids_len=24072,
        visual_token_count=21504, video_grid_thw=[[256,14,24]],
        processor_profile_sha256=digest, processor_size=profile['video_size'])
    value = dict(config=dict(processor_profile=str(path), processor_profile_sha256=digest),
        protocol=dict(processor_profile=dict(sha256=digest, profile=profile,
            actual_processor_instantiated=True, actual_encodings=[audit]),
            cuda_process_peak_memory=dict(status='COLLECTED',
                max_memory_allocated_bytes=100, max_memory_reserved_bytes=110)),
        rows={'one':dict(processor_profile_sha256=digest, processor_size=profile['video_size'],
            n_visual_tokens=21504, prompt_len=24072)})
    return path, value


def test_actual_profile_is_required(encoding):
    path, value = encoding
    summary.validate_profile(value, path, 512)
    value['protocol']['processor_profile']['actual_processor_instantiated'] = False
    with pytest.raises(ValueError, match='actual processor'):
        summary.validate_profile(value, path, 512)


def test_automatic_spatial_shrink_is_rejected(encoding):
    path, value = encoding
    value['protocol']['processor_profile']['actual_encodings'][0]['video_grid_thw'] = [[256,10,18]]
    with pytest.raises(ValueError, match='actual long-input'):
        summary.validate_profile(value, path, 512)


def test_duplicate_encoding_audit_is_rejected(encoding):
    path, value = encoding
    value['protocol']['processor_profile']['actual_encodings'] *= 2
    with pytest.raises(ValueError, match='missing or duplicate'):
        summary.validate_profile(value, path, 512)


def test_process_peak_is_required(encoding):
    path, value = encoding
    value['protocol']['cuda_process_peak_memory']['status'] = 'NOT_COLLECTED'
    with pytest.raises(ValueError, match='peak memory'):
        summary.validate_profile(value, path, 512)


def test_path_alone_does_not_bind_checkpoint():
    left = dict(config=dict(head='/same/path'), protocol=dict(artifacts=dict(
        head_weights_sha256={'model.safetensors':'a'}, head_config_sha256='b',
        checkpoint_identity_sha256='c', checkpoint_identity={'optimizer_step':80})))
    right = copy.deepcopy(left)
    summary.same_head(left,right)
    right['protocol']['artifacts']['head_weights_sha256']['model.safetensors'] = 'changed'
    with pytest.raises(ValueError, match='checkpoint identity'):
        summary.same_head(left,right)
