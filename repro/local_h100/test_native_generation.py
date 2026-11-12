import copy
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import pytest
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin

spec = importlib.util.spec_from_file_location('audit_native', Path(__file__).with_name('audit_native_generation.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_fresh_config_cannot_inherit_sampling_or_repeat_penalty():
    fake = SimpleNamespace(generation_config=GenerationConfig(do_sample=True, repetition_penalty=1.2,
                                                              temperature=.7, top_p=.8))
    config, _ = GenerationMixin._prepare_generation_config(fake, audit.generation_config(2, 0, 512), use_model_defaults=False)
    assert config.do_sample is False
    assert config.repetition_penalty == 1.0
    assert config.num_beams == 1 and config.use_cache is True
    assert config.max_new_tokens == 512
    assert config.min_length == 0 and config.min_new_tokens is None
    assert config.eos_token_id == 2 and config.pad_token_id == 0


def test_native_call_resets_rope_and_disables_defaults():
    class Fake:
        model = SimpleNamespace(rope_deltas='stale')
        def generate(self, **kwargs):
            assert self.model.rope_deltas is None
            assert kwargs['use_model_defaults'] is False
            assert kwargs['generation_config'].do_sample is False
            return 'output'
    assert audit.native_generate(Fake(), {'input_ids': 'fixed'}, audit.generation_config(2, 0, 512)) == 'output'


def test_prompts_copy_without_touching_pool_row():
    row = {'split': 'train', 'question': 'Original prompt', 'prompt': 'Original prompt'}
    before = copy.deepcopy(row)
    prompts = audit.prompts(row)
    assert row == before
    assert len(prompts) == 3 and prompts[0][1] == 'Original prompt'
    assert 'at most eight' in prompts[1][1]


def test_heldout_is_rejected():
    with pytest.raises(ValueError, match='train video'):
        audit.prompts({'split': 'dev', 'question': 'Do not use'})


@pytest.mark.parametrize('value', ['', '0', 'GPU-a,GPU-b', ' GPU-a'])
def test_multiple_or_implicit_gpus_rejected(value):
    with pytest.raises(ValueError):audit.single_uuid(value)
