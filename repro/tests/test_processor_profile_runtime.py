"""CPU contract tests: configure the processor eval actually uses, retain hashes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100.processor_profile_runtime import (apply_profile, profiled_evaluation,
                                                 validate_encoded)


class Qwen3VLVideoProcessor:
    patch_size = 16
    merge_size = 2
    temporal_patch_size = 2

    def __init__(self):
        self.size = {"shortest_edge": 4096, "longest_edge": 25165824}


def processor():
    return SimpleNamespace(video_processor=Qwen3VLVideoProcessor(),
                           tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 7))


@pytest.fixture
def case():
    profile = {"path": "/explicit/test-profile.json", "sha256": "1" * 64,
               "profile": {"video_processor_class": "Qwen3VLVideoProcessor",
                           "video_size": {"shortest_edge": 4096, "longest_edge": 50331648},
                           "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
                           "expected_spatial_grid_hw": [14, 24], "frame_counts": [256, 512]}}
    row = {"video_id": "train-video", "n_frames": 256,
           "engineering_long_input_probe": True,
           "processor_profile_sha256": profile["sha256"],
           "input_ids_len": 10753, "visual_token_count": 10752}
    inputs = {"video_grid_thw": torch.tensor([[128, 14, 24]]),
              "input_ids": torch.tensor([[1] + [7] * 10752])}
    return profile, row, inputs


def test_actual_factory_twice_and_identity_unchanged(monkeypatch, case):
    from transformers import AutoProcessor
    profile, row, inputs = case
    factory = lambda *args, **kwargs: processor()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", staticmethod(factory))
    memory_calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                        lambda device: memory_calls.append(("allocated", device)) or 123456)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved",
                        lambda device: memory_calls.append(("reserved", device)) or 234567)
    def forbid_reset(*args, **kwargs):
        pytest.fail("process lifetime peak must never be reset by the wrapper")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", forbid_reset)
    output = []
    base_encode = lambda proc, row, mode, **kwargs: inputs
    base_validate = lambda frame, *args, **kwargs: ["validated"]
    base_write = lambda path, payload: output.append(payload)
    module = SimpleNamespace(encode_inputs=base_encode, validate_rows=base_validate,
                             write_result=base_write)
    identity = {"input_ids": "actual-ids-hash", "pixel_values_videos": "actual-pixels-hash"}
    payload = {"config": {"gpu": 3}, "protocol": {}, "rows": [
        {"video": "train-video", "L": 10753, "input_identity": deepcopy(identity)}]}
    with profiled_evaluation(module, profile):
        discarded = AutoProcessor.from_pretrained("common-loader")
        used = AutoProcessor.from_pretrained("actual-eval-construction")
        assert discarded is not used
        assert used.video_processor.size == profile["profile"]["video_size"]
        assert module.validate_rows(pd.DataFrame([row]), "real") == ["validated"]
        assert module.encode_inputs(used, row, "real") is inputs
        module.write_result("out", payload)
    assert AutoProcessor.from_pretrained is factory
    assert module.encode_inputs is base_encode
    assert module.validate_rows is base_validate
    assert module.write_result is base_write
    result = output[0]
    assert result["rows"][0]["input_identity"] == identity
    assert "processor_profile" not in payload["protocol"]
    assert len(result["protocol"]["processor_profile"]["factory_calls"]) == 2
    assert len(result["protocol"]["processor_profile"]["actual_encodings"]) == 1
    assert result["rows"][0]["processor_size"] == profile["profile"]["video_size"]
    memory = result["protocol"]["cuda_process_peak_memory"]
    assert memory["status"] == "COLLECTED"
    assert memory["device"] == "cuda:3"
    assert memory["max_memory_allocated_bytes"] == 123456
    assert memory["max_memory_reserved_bytes"] == 234567
    assert memory["counters_reset_by_wrapper"] is False
    assert "not an individual SPEC peak" in memory["scope"]
    assert memory_calls == [("allocated", 3), ("reserved", 3)]


def test_foreign_processor_and_mutation_fail_and_restore(monkeypatch, case):
    from transformers import AutoProcessor
    profile, row, inputs = case
    factory = lambda *args, **kwargs: processor()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", staticmethod(factory))
    module = SimpleNamespace(encode_inputs=lambda *a, **k: inputs,
                             validate_rows=lambda *a, **k: [], write_result=lambda *a: None)
    with profiled_evaluation(module, profile):
        foreign = processor()
        apply_profile(foreign, profile)
        with pytest.raises(ValueError, match="outside the profiled"):
            module.encode_inputs(foreign, row, "real")
        actual = AutoProcessor.from_pretrained("eval")
        actual.video_processor.size["longest_edge"] = 25165824
        with pytest.raises(ValueError, match="size changed"):
            module.encode_inputs(actual, row, "real")
    assert AutoProcessor.from_pretrained is factory


def test_wrong_grid_and_row_binding_rejected(case):
    profile, row, inputs = case
    proc = processor()
    apply_profile(proc, profile)
    with pytest.raises(ValueError, match="grid"):
        validate_encoded(proc, row, dict(inputs, video_grid_thw=torch.tensor([[128, 10, 18]])), profile)
    with pytest.raises(ValueError, match="explicitly bound"):
        validate_encoded(proc, dict(row, processor_profile_sha256="wrong"), inputs, profile)


def test_preflight_records_no_claim_of_instantiation(monkeypatch, case):
    profile, row, _ = case
    output = []
    def forbid_cuda(*args, **kwargs):
        pytest.fail("CPU preflight must not call CUDA APIs")
    for name in ("is_initialized", "max_memory_allocated", "max_memory_reserved",
                 "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, forbid_cuda)
    module = SimpleNamespace(encode_inputs=lambda *a: None, validate_rows=lambda *a, **k: [],
                             write_result=lambda path, payload: output.append(payload))
    with profiled_evaluation(module, profile):
        module.validate_rows(pd.DataFrame([row]), "real")
        module.write_result("out", {"config": {"preflight_only": True},
                                    "protocol": {}, "preflight": []})
    audit = output[0]["protocol"]["processor_profile"]
    assert audit["actual_processor_instantiated"] is False
    assert audit["actual_encodings"] == []
    memory = output[0]["protocol"]["cuda_process_peak_memory"]
    assert memory["status"] == "NOT_COLLECTED_CPU_ONLY"
    assert "max_memory_allocated_bytes" not in memory
