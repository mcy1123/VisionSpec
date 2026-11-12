"""CPU tests for public initialization identity and compatibility rejection."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("fresh_init", Path(__file__).with_name("fresh_init.py"))
fresh_init = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh_init)


def configs():
    source = {"hidden_size": 4096, "vocab_size": 151936, "head_dim": 128,
              "num_attention_heads": 32, "num_key_value_heads": 8,
              "num_target_layers": 36, "num_hidden_layers": 5, "block_size": 16,
              "rope_theta": 1000000,
              "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33], "mask_token_id": 151669}}
    target = {"model_type": "qwen3_vl", "text_config": dict(source, num_hidden_layers=36,
                                                                rope_theta=5000000)}
    return source, target


def test_contract_preserves_public_rope_and_does_not_mutate():
    source, target = configs()
    before = copy.deepcopy((source, target))
    result = fresh_init.validate_contract(source, target)
    assert result["source_rope_theta"] == 1000000
    assert result["target_rope_theta"] == 5000000
    assert (source, target) == before


@pytest.mark.parametrize("name,value", [("hidden_size", 5120), ("vocab_size", 152064),
                                         ("num_hidden_layers", 64), ("head_dim", 64)])
def test_reject_target_mismatch(name, value):
    source, target = configs()
    target["text_config"][name] = value
    with pytest.raises(ValueError):
        fresh_init.validate_contract(source, target)


def test_reject_feature_layers():
    source, target = configs()
    source["dflash_config"]["target_layer_ids"][-1] = 35
    with pytest.raises(ValueError, match="taps"):
        fresh_init.validate_contract(source, target)


def test_no_overwrite_verified_report(tmp_path):
    path = tmp_path / "report.json"
    fresh_init.write_exclusive(path, {"good": True})
    with pytest.raises(FileExistsError):
        fresh_init.write_exclusive(path, {"good": False})
    assert '"good": true' in path.read_text()


def test_sha_identity_rejection(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"known")
    fresh_init.verify_file(path, fresh_init.sha256(path), 5)
    with pytest.raises(ValueError, match="SHA256"):
        fresh_init.verify_file(path, "0" * 64)
    with pytest.raises(ValueError, match="Size"):
        fresh_init.verify_file(path, fresh_init.sha256(path), 10)
