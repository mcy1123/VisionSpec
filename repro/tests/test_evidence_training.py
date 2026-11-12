"""CPU checks for matched initialization and QA-isolated evidence scoring."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from video_caption import init_evidence_heads as initialization
from video_caption import train_video_9k as trainer
from shared import training_resume
from dflash.model import DFlashDraftModel


def tiny_config(mode="none"):
    return Qwen3Config(
        hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32, num_target_layers=36, block_size=4,
        dflash_config={"target_layer_ids": [1, 9, 17, 25, 33],
                       "mask_token_id": 31, "evidence_mode": mode,
                       "evidence_dim": 8, "evidence_max_tokens": 8,
                       "evidence_spatial_bins": 2})


def save_source(path):
    config = tiny_config()
    config._attn_implementation = "sdpa"
    head = DFlashDraftModel(config).to(torch.bfloat16)
    head.save_pretrained(path)
    return {name: value.clone() for name, value in head.state_dict().items()}


def test_saved_arms_preserve_every_base_tensor_and_match_added_weights(tmp_path):
    source = tmp_path / "source"
    original = save_source(source)
    with patch.object(initialization, "ALLOWED_ROOTS", (tmp_path,)):
        result = initialization.initialize(source, tmp_path / "arms", seed=11,
                                            dim=8, max_tokens=8, spatial_bins=2)
    assert result["matched_saved_weights_verified"]
    assert result["base_tensors_verified_equal"]
    assert result["resume_supported"] is False
    states = []
    for arm, mode in (("B_text", "text"), ("C_visual", "visual")):
        directory = tmp_path / "arms" / arm
        state = initialization.read_weights(initialization.weight_files(directory))
        states.append(state)
        config = json.loads((directory / "config.json").read_text())
        assert config["dflash_config"]["evidence_mode"] == mode
        for name, value in original.items():
            assert state[name].dtype == value.dtype
            assert torch.equal(state[name], value)
        added = {name: value for name, value in state.items() if name.startswith(initialization.PREFIX)}
        assert added and all(value.dtype == torch.float32 for value in added.values())
        assert not torch.count_nonzero(state["evidence_readout.output.weight"])
        assert not (directory / "training_state.pt").exists()
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])
    source_after = initialization.read_weights(initialization.weight_files(source))
    assert all(torch.equal(original[name], source_after[name]) for name in original)


def test_initialization_seed_is_reproducible_without_changing_cpu_rng():
    base = DFlashDraftModel(tiny_config()).state_dict()
    config = tiny_config().to_dict()
    before = torch.get_rng_state().clone()
    left, _ = initialization.make_matched_head(config, base, seed=23, dim=8,
                                                max_tokens=8, spatial_bins=2)
    assert torch.equal(torch.get_rng_state(), before)
    right, _ = initialization.make_matched_head(config, base, seed=23, dim=8,
                                                 max_tokens=8, spatial_bins=2)
    assert initialization.state_digest(left.evidence_readout.state_dict()) == \
        initialization.state_digest(right.evidence_readout.state_dict())


def test_missing_old_tensor_cannot_be_silently_initialized():
    base = DFlashDraftModel(tiny_config()).state_dict()
    base.pop(next(iter(base)))
    with pytest.raises(ValueError, match="beyond the evidence_readout"):
        initialization.make_matched_head(tiny_config().to_dict(), base, seed=0,
                                         dim=8, max_tokens=8, spatial_bins=2)


def test_dry_run_reads_no_weights_and_never_creates_output(tmp_path):
    source = tmp_path / "source"
    save_source(source)
    with patch.object(initialization, "ALLOWED_ROOTS", (tmp_path,)), \
         patch.object(initialization, "read_weights", side_effect=AssertionError("no weights in dry-run")):
        result = initialization.initialize(source, tmp_path / "arms", dry_run=True)
    assert result["dry_run"]
    assert not (tmp_path / "arms").exists()


def test_output_storage_and_no_overwrite_contract(tmp_path):
    source = tmp_path / "source"
    save_source(source)
    with pytest.raises(ValueError, match="existing /mnt/local"):
        initialization.initialize(source, tmp_path / "bad", dry_run=True)
    with patch.object(initialization, "ALLOWED_ROOTS", (tmp_path,)), \
         pytest.raises(FileExistsError, match="overwrite"):
        initialization.initialize(source, source, dry_run=True)


def test_evidence_heads_require_real_globally_pooled_training(tmp_path):
    config = tiny_config("visual").to_dict()
    (tmp_path / "config.json").write_text(json.dumps(config))
    for time_mode, normalization in (("legacy", "global_pooled"), ("real", "legacy_per_qa")):
        with pytest.raises(ValueError, match="require --time-mode real"):
            trainer.validate_training_head(tmp_path, time_mode, normalization)
    assert trainer.validate_training_head(tmp_path, "real", "global_pooled")["mode"] == "visual"
    config["dflash_config"]["evidence_mode"] = "none"
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert trainer.validate_training_head(tmp_path)["mode"] == "none"


def test_request_staging_detaches_nested_prepared_features():
    feature = torch.randn(2, 3, requires_grad=True) * 2
    request = {"identity": "videoA", "prepared_visual": {"features": feature,
                "config": (1, [2, 3])}, "time_values": torch.tensor([0.2, 0.8])}
    trace = {"evidence_request": request, "rounds": [{"fused": torch.ones(1)}]}
    staged = trainer.move_training_trace(trace, "cpu")
    assert not staged["evidence_request"]["prepared_visual"]["features"].requires_grad
    assert staged["evidence_request"] is not request
    assert staged["evidence_request"]["prepared_visual"] is not request["prepared_visual"]
    staged["evidence_request"]["identity"] = "other"
    assert request["identity"] == "videoA"
    assert isinstance(staged["evidence_request"]["prepared_visual"]["config"], tuple)


class FakeEvidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = tiny_config("visual")
        self.evidence_mode = "visual"
        self.evidence_readout = nn.Linear(1, 1, bias=False)
        self.request = None

    def set_evidence_request(self, request):
        self.request = request

    def evidence_diagnostics(self):
        return {"identity": self.request["identity"]}


def test_four_qa_global_score_rebinds_each_staged_request_not_last_rollout():
    head = FakeEvidenceHead()
    scored = []
    diagnostics = []

    def collect():
        for index in range(4):
            request = {"identity": f"video{index}", "prepared_visual": {
                "features": torch.tensor([[float(index + 1)]])}}
            head.set_evidence_request(request)
            yield {"denominator": 1.0, "evidence_request": request}

    def score(target, actual_head, trace, **kwargs):
        assert actual_head.request["identity"] == trace["evidence_request"]["identity"]
        scored.append(actual_head.request["identity"])
        yield SimpleNamespace(numerator=actual_head.evidence_readout(
            actual_head.request["prepared_visual"]["features"]).sum())

    with patch.object(trainer.runtime, "score", side_effect=score):
        trainer.backward_update(collect(), lambda trace: trainer.score_training_trace(
            None, head, trace, e2e_tv_weight=10, diagnostics=diagnostics), "global_pooled",
            denominator_fn=lambda trace: trace["denominator"],
            stage_trace=lambda trace: trainer.move_training_trace(trace, "cpu"),
            restore_trace=lambda trace: trainer.move_training_trace(trace, "cpu"))
    assert scored == [f"video{i}" for i in range(4)]
    assert [item["identity"] for item in diagnostics] == scored
    assert head.evidence_readout.weight.grad.item() == 2.5
    assert trainer.evidence_gradient_diagnostics(head)["gradient_l2"] == 2.5


def test_disconnected_evidence_gradient_is_rejected():
    with pytest.raises(RuntimeError, match="received no gradients"):
        trainer.evidence_gradient_diagnostics(FakeEvidenceHead())


def test_nonfinite_evidence_gradient_is_rejected():
    head = FakeEvidenceHead()
    head.evidence_readout.weight.grad = torch.full_like(head.evidence_readout.weight, float("nan"))
    with pytest.raises(RuntimeError, match="non-finite evidence"):
        trainer.evidence_gradient_diagnostics(head)


class TinySavedHead:
    def save_pretrained(self, path):
        (path / "config.json").write_text('{"tiny": true}\n')
        (path / "model.safetensors").write_bytes(b"tiny fixture, never loaded")


def test_checkpoint_identity_is_hashed_before_atomic_publication(tmp_path):
    destination = tmp_path / "step20"
    metadata = {"schema": "video_opd_checkpoint_identity_v1", "optimizer_step": 20,
                "data_cursor": 20, "resume_supported": False}
    original_rename = training_resume.os.rename
    observed = []

    def inspect_publish(source, target):
        source, target = Path(source), Path(target)
        assert not destination.exists()
        identity = source / "CHECKPOINT_IDENTITY.json"
        assert json.loads(identity.read_text()) == metadata
        sha = json.loads((source / "SHA256.json").read_text())
        assert sha[identity.name] == training_resume.file_hash(identity)
        observed.append(True)
        return original_rename(source, target)

    with patch.object(training_resume, "validate_output"), \
         patch.object(training_resume.os, "rename", side_effect=inspect_publish):
        hashes = training_resume.save_atomic_checkpoint(destination, TinySavedHead(),
                                                        {"update": 20}, metadata=metadata)
    assert observed == [True]
    assert hashes["CHECKPOINT_IDENTITY.json"] == training_resume.file_hash(
        destination / "CHECKPOINT_IDENTITY.json")


def test_legacy_checkpoint_default_has_no_identity_sidecar(tmp_path):
    destination = tmp_path / "legacy"
    with patch.object(training_resume, "validate_output"):
        hashes = training_resume.save_atomic_checkpoint(destination, TinySavedHead(), {"update": 1})
    assert "CHECKPOINT_IDENTITY.json" not in hashes
    assert not (destination / "CHECKPOINT_IDENTITY.json").exists()
    assert set(hashes) == {"config.json", "model.safetensors", "training_state.pt"}


@pytest.mark.parametrize("bad", [{"tensor": torch.ones(1)}, {"value": float("nan")}, ["not an object"]])
def test_invalid_checkpoint_metadata_fails_before_any_output(tmp_path, bad):
    destination = tmp_path / "bad"
    with patch.object(training_resume, "validate_output"), pytest.raises((TypeError, ValueError)):
        training_resume.save_atomic_checkpoint(destination, TinySavedHead(), {}, metadata=bad)
    assert not list(tmp_path.iterdir())


def sample_training_protocol(tmp_path, mode="none"):
    return {
        "config": {"init_from": str(tmp_path), "seed": 7, "lr": 1e-5,
                   "qa_per_update": 4, "time_mode": "real",
                   "loss_normalization": "global_pooled", "e2e_tv_weight": 10},
        "hashes": {"initial_head_weights": {"model.safetensors": "a" * 64},
                   "pool": "p" * 64, "schedule": "s" * 64, "target_config": "t" * 64},
        "evidence": {"mode": mode, "dim": 256, "max_tokens": 256, "spatial_bins": 2},
        "protocol_sha256": "f" * 64, "target_dtype": "float32", "head_dtype": "bfloat16",
        "master_dtype": "float32", "evidence_parameter_dtype": None if mode == "none" else "float32",
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.0}}


def test_checkpoint_identity_uses_common_original_head_for_both_added_arms(tmp_path):
    a_protocol = sample_training_protocol(tmp_path)
    source = a_protocol["hashes"]["initial_head_weights"]
    manifest = {"source_weights_sha256": source, "base_tensors_verified_equal": True}
    manifest_path = tmp_path / "INIT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    records = [trainer.checkpoint_identity(a_protocol, 60)]
    for mode in ("text", "visual"):
        protocol = sample_training_protocol(tmp_path, mode)
        protocol["hashes"]["initial_head_weights"] = {"model.safetensors": "b" * 64}
        protocol["hashes"]["initialization_manifest"] = training_resume.file_hash(manifest_path)
        records.append(trainer.checkpoint_identity(protocol, 60))
    assert all(record["common_initial_head_weights_sha256"] == source for record in records)
    assert all(record["optimizer_step"] == record["data_cursor"] == 60 for record in records)
    assert [record["evidence"]["mode"] for record in records] == ["none", "text", "visual"]


def test_checkpoint_identity_rejects_changed_initialization_manifest(tmp_path):
    protocol = sample_training_protocol(tmp_path, "visual")
    manifest = tmp_path / "INIT_MANIFEST.json"
    manifest.write_text("{}")
    protocol["hashes"]["initialization_manifest"] = training_resume.file_hash(manifest)
    manifest.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed since training preflight"):
        trainer.checkpoint_identity(protocol, 20)
