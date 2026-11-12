import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('prompt_set', Path(__file__).with_name('audit_prompt_set.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_prompt_list_is_copied_and_preserved():
    rows = [{'label':'detail', 'prompt':'Describe visible details.'}]
    result = audit.validate_prompts(rows)
    assert result == rows and result is not rows and result[0] is not rows[0]


@pytest.mark.parametrize('value', [[], {}, [{'label':'x','prompt':''}],
                                  [{'label':'x','prompt':'a'}, {'label':'x','prompt':'b'}],
                                  [{'label':'x','prompt':'a','temperature':1}]])
def test_invalid_prompt_sets_rejected(value):
    with pytest.raises(ValueError):audit.validate_prompts(value)


def test_same_pixels_allow_different_text():
    first = {'encoded_sha256': {'pixel_values_videos':'pixels', 'video_grid_thw':'grid', 'input_ids':'first'}}
    second = {'encoded_sha256': {'pixel_values_videos':'pixels', 'video_grid_thw':'grid', 'input_ids':'second'}}
    audit.validate_visual_identity(first, second)
    second['encoded_sha256']['pixel_values_videos'] = 'different'
    with pytest.raises(ValueError, match='changed visual input'):
        audit.validate_visual_identity(first, second)


def test_shared_native_helper_retains_unpenalized_greedy():
    config = audit.native.generation_config(2, 0, 1536)
    assert config.max_new_tokens == 1536 and config.do_sample is False
    assert config.repetition_penalty == 1 and config.no_repeat_ngram_size == 0
    assert config.min_length == 0 and config.min_new_tokens is None
