"""Wrong-video banks preserve source identity and reject incompatible evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.evidence import build_evidence_request, evidence_settings
from shared import evidence_memory as memory
from evaluation.eval_video_opd import evaluation_artifacts, validate_evidence_eval


def prepared_fixture(number=0, *, grid=(2, 4, 4)):
    row = dict(n_frames=4, source_fps=1., video_duration=8., source_frame_count=8,
               sample_frame_ids=[number % 2 + x for x in (0, 2, 4, 6)],
               sample_timestamps=[float(number % 2 + x) for x in (0, 2, 4, 6)],
               width=64, height=64, source_sha256=hashlib.sha256(f"source-{number}".encode()).hexdigest())
    t, h, w = grid
    ins = dict(input_ids=torch.tensor([[1] + [9] * (t * h * w // 4) + [2]]),
               pixel_values_videos=torch.ones(t * h * w, 12) * number,
               video_grid_thw=torch.tensor([grid]))
    proc = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 9),
                           video_processor=SimpleNamespace(merge_size=2, temporal_patch_size=2))
    request = build_evidence_request(ins, row, proc)
    request["prepared_visual"] = dict(identity=request["identity"], config=(grid, 2, 8, 1, 4),
                                      features=torch.ones(2, 4) * number,
                                      times=request["time_values"].clone(), groups=2, spatial_bins=1)
    return row, request


def contract_fixture():
    settings = evidence_settings({"dflash_config": dict(evidence_mode="visual", evidence_max_tokens=8,
                                                       evidence_spatial_bins=1, target_layer_ids=[1, 9, 17, 25, 33])})
    artifacts = dict(target_config_sha256="a" * 64, head_config_sha256="b" * 64,
                     head_weights_sha256={"model.safetensors": "c" * 64},
                     checkpoint_identity=None, checkpoint_identity_sha256=None,
                     source_sha256={"helper.py": "d" * 64}, target_weight_verification="external_manifest")
    return memory.bank_contract(settings, artifacts, {"hidden_size": 4}, "float32")


def bank_fixture():
    entries, rows, requests = {}, {}, {}
    for i, name in enumerate(("a", "b", "c")):
        row, request = prepared_fixture(i)
        entry = memory.memory_entry(name, row, request)
        entries[memory.entry_key(entry)] = entry
        rows[name], requests[name] = row, request
    contract = contract_fixture()
    return memory.validate_bank(dict(contract=contract, entries=entries), contract), rows, requests


def test_normal_export_clones_frozen_memory_and_keeps_real_times():
    row, request = prepared_fixture(1)
    entry = memory.memory_entry("a", row, request)
    assert entry["prepared"]["features"].data_ptr() != request["prepared_visual"]["features"].data_ptr()
    assert not entry["prepared"]["features"].requires_grad
    torch.testing.assert_close(entry["prepared"]["times"], torch.tensor([0.25, 0.75]))
    request["intervention"] = "bypass"
    with pytest.raises(ValueError, match="normal"):
        memory.memory_entry("a", row, request)


def test_sorted_cycle_replaces_only_donor_and_preserves_target_request_identity():
    bank, rows, requests = bank_fixture()
    for current, expected in (("a", "b"), ("b", "c"), ("c", "a")):
        identity = requests[current]["identity"]
        donor, audit = memory.select_donor(bank, current, rows[current], requests[current])
        assert audit["video"] == expected
        assert donor["identity"] == requests[expected]["identity"]
        assert donor["identity"] != identity
        assert requests[current]["identity"] == identity
        torch.testing.assert_close(donor["times"], requests[expected]["prepared_visual"]["times"])


def test_donor_with_different_source_grid_and_same_pooled_capacity_is_valid():
    bank, rows, requests = bank_fixture()
    row, request = prepared_fixture(1, grid=(2, 2, 8))
    entry = memory.memory_entry("b", row, request)
    bank["entries"] = {key: value for key, value in bank["entries"].items() if value["video"] != "b"}
    bank["entries"][memory.entry_key(entry)] = entry
    memory.validate_bank(bank, contract_fixture())
    donor, audit = memory.select_donor(bank, "a", rows["a"], requests["a"])
    assert audit["video"] == "b"
    assert donor["config"][0] == (2, 2, 8)


def test_loaded_bank_donor_reaches_model_without_overwriting_current_memory():
    from dflash.evidence import EvidenceReadout
    from transformers import DynamicCache
    bank, rows, requests = bank_fixture()
    current = requests["a"]
    original_identity = current["identity"]
    original_features = current["prepared_visual"]["features"].clone()
    donor, audit = memory.select_donor(bank, "a", rows["a"], current)
    current.update(intervention="wrong_video", donor_visual=donor)
    module = EvidenceReadout(4, dim=3, max_tokens=8, spatial_bins=1, mode="visual")
    module.set_request(current)
    n = len(current["prompt_visual_mask"])
    features, times, _ = module.prepare(torch.randn(1, n, 4), torch.arange(n + 4)[None], DynamicCache(), False)
    torch.testing.assert_close(features, requests[audit["video"]]["prepared_visual"]["features"])
    torch.testing.assert_close(times, requests[audit["video"]]["prepared_visual"]["times"])
    assert current["identity"] == original_identity
    torch.testing.assert_close(current["prepared_visual"]["features"], original_features)


def test_same_source_or_same_encoded_pixels_never_become_wrong_video_donors():
    bank, rows, requests = bank_fixture()
    for entry in bank["entries"].values():
        entry["source_sha256"] = rows["a"]["source_sha256"]
    with pytest.raises(ValueError, match="no compatible different-video"):
        memory.select_donor(bank, "a", rows["a"], requests["a"])
    bank, rows, requests = bank_fixture()
    for entry in bank["entries"].values():
        entry["metadata"]["encoded_sha256"]["pixel_values_videos"] = requests["a"]["metadata"]["encoded_sha256"]["pixel_values_videos"]
    with pytest.raises(ValueError, match="no compatible different-video"):
        memory.select_donor(bank, "a", rows["a"], requests["a"])


def test_wrong_target_code_dtype_or_readout_capacity_contract_fails():
    bank, _, _ = bank_fixture()
    for change in ("target", "code", "capacity", "dtype"):
        expected = copy.deepcopy(bank["contract"])
        if change == "target":
            expected["artifacts"]["target_config_sha256"] = "e" * 64
        elif change == "code":
            expected["artifacts"]["source_sha256"]["helper.py"] = "f" * 64
        elif change == "capacity":
            expected["settings"]["max_tokens"] += 1
        else:
            expected["target_dtype"] = "bfloat16"
        with pytest.raises(ValueError, match="contract mismatch"):
            memory.validate_bank(bank, expected)


def test_tampered_times_features_or_capacity_fail_before_use():
    bank, _, _ = bank_fixture()
    for field in ("times", "features", "capacity"):
        changed = copy.deepcopy(bank)
        entry = next(iter(changed["entries"].values()))
        if field == "times":
            entry["prepared"]["times"].add_(0.01)
        elif field == "features":
            entry["prepared"]["features"].add_(0.1)
        else:
            cfg = list(entry["prepared"]["config"])
            cfg[2] = 9
            entry["prepared"]["config"] = tuple(cfg)
        with pytest.raises(ValueError):
            memory.validate_bank(changed, bank["contract"])


def test_current_input_must_match_the_normal_export_exactly():
    bank, rows, requests = bank_fixture()
    wrong = dict(requests["a"], identity="0" * 64)
    with pytest.raises(ValueError, match="absent"):
        memory.select_donor(bank, "a", rows["a"], wrong)
    with pytest.raises(ValueError, match="source/encoded identity"):
        memory.select_donor(bank, "a", rows["b"], requests["a"])


def test_storage_resolution_atomic_exclusive_and_weights_only_load(tmp_path):
    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "escaped").symlink_to(outside, target_is_directory=True)
    bank, _, _ = bank_fixture()
    with mock.patch.object(memory, "STORAGE_ROOTS", (allowed,)):
        with pytest.raises(ValueError, match="physically reside"):
            memory.checked_bank_path(outside / "bank.pt", writing=True)
        with pytest.raises(ValueError, match="physically reside"):
            memory.checked_bank_path(allowed / "escaped/bank.pt", writing=True)
        path = memory.save_bank(allowed / "bank.pt", bank["contract"], bank["entries"])
        with pytest.raises(FileExistsError):
            memory.save_bank(path, bank["contract"], bank["entries"])
        with mock.patch.object(memory.torch, "load", wraps=torch.load) as loader:
            loaded = memory.load_bank(path, bank["contract"])
            assert loader.call_args.kwargs["weights_only"] is True
        assert loaded["contract"] == bank["contract"]
        assert not list(allowed.glob(".evidence-bank-*.tmp"))


def test_cli_bank_options_require_the_correct_enabled_probe():
    settings = contract_fixture()["settings"]
    validate_evidence_eval(settings, time_mode="real", intervention="wrong_video", donor_memory="bank.pt")
    for options in ({"intervention": "wrong_video"}, {"donor_memory": "bank.pt"},
                    {"intervention": "zero", "export_memory": "bank.pt"},
                    {"export_memory": "bank.pt", "shards": 2}):
        with pytest.raises(ValueError):
            validate_evidence_eval(settings, time_mode="real", **options)
    with pytest.raises(ValueError):
        validate_evidence_eval(dict(settings, mode="none"), time_mode="real", export_memory="bank.pt")
    with pytest.raises(ValueError):
        validate_evidence_eval(dict(settings, mode="text"), time_mode="real", intervention="wrong_video", donor_memory="bank.pt")


def test_checkpoint_sidecar_is_read_only_when_manifest_checksum_agrees(tmp_path):
    target, head = tmp_path / "target", tmp_path / "head"
    target.mkdir()
    head.mkdir()
    (target / "config.json").write_text("{}")
    (head / "config.json").write_text("{}")
    (head / "model.safetensors").write_bytes(b"not loaded")
    assert evaluation_artifacts(target, head)["checkpoint_identity"] is None
    identity = {"optimizer_step": 60, "schema": "fixture"}
    path = head / "CHECKPOINT_IDENTITY.json"
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="sidecar"):
        evaluation_artifacts(target, head)
    manifest = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in (path, head / "config.json", head / "model.safetensors")}
    (head / "SHA256.json").write_text(json.dumps(manifest))
    artifacts = evaluation_artifacts(target, head)
    assert artifacts["checkpoint_identity"] == identity
    (head / "model.safetensors").write_bytes(b"silently replaced checkpoint weights")
    with pytest.raises(ValueError, match="config/weights"):
        evaluation_artifacts(target, head)
    (head / "model.safetensors").write_bytes(b"not loaded")
    path.write_text(json.dumps(dict(identity, optimizer_step=20)))
    with pytest.raises(ValueError, match="sidecar"):
        evaluation_artifacts(target, head)
