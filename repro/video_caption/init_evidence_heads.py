#!/usr/bin/env python3
"""CPU-only, audited weights-only initialization of matched text/visual arms.

The A arm references --source unchanged. B_text and C_visual have identical
base and added tensors; only evidence_mode differs. No optimizer state is read
or created, and no target model is loaded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import torch

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from shared import common, training_resume
from shared.evidence import evidence_settings

PREFIX = "evidence_readout."
ALLOWED_ROOTS = tuple(Path(path) for path in ("/mnt/local", "/mnt/local2", "/mnt/local3"))


def validate_destination(path):
    resolved = Path(path).expanduser().resolve()
    roots = [root.resolve() for root in ALLOWED_ROOTS if root.is_dir()]
    if not any(root in resolved.parents for root in roots):
        raise ValueError("output must resolve below an existing /mnt/local{,2,3} storage root")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite initialization directory: {resolved}")
    return resolved


def weight_files(source):
    source = Path(source)
    safe = sorted(source.glob("*.safetensors"))
    binary = sorted(source.glob("pytorch_model*.bin"))
    if safe and binary:
        raise ValueError("ambiguous source has both safetensors and pytorch_model binaries")
    paths = safe or binary
    if not paths:
        raise ValueError("source must contain head weights")
    return paths


def read_weights(paths):
    result = {}
    for path in paths:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            part = load_file(str(path), device="cpu")
        else:
            part = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(part, dict) or not all(torch.is_tensor(value) for value in part.values()):
            raise ValueError(f"not a plain head tensor state: {path}")
        overlap = set(part) & set(result)
        if overlap:
            raise ValueError(f"duplicate tensor names across weight files: {sorted(overlap)}")
        result.update(part)
    return result


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checked_base_load(head, source_state):
    """Never silently random-initialize or cast a single source tensor."""
    if any(name.startswith(PREFIX) for name in source_state):
        raise ValueError("source already contains evidence weights; use the common original head")
    model_keys = set(head.state_dict())
    expected_new = {name for name in model_keys if name.startswith(PREFIX)}
    if not expected_new:
        raise ValueError("constructed model has no evidence parameters")
    if model_keys - set(source_state) != expected_new or set(source_state) - model_keys:
        raise ValueError("source/base tensor keys differ beyond the evidence_readout prefix")
    # assign preserves the original tensor dtypes, avoiding a hidden fp32/bf16 cast.
    result = head.load_state_dict(source_state, strict=False, assign=True)
    if set(result.missing_keys) != expected_new or result.unexpected_keys:
        raise ValueError("unexpected missing/unexpected tensors while loading the common head")
    after = head.state_dict()
    for name, expected in source_state.items():
        actual = after[name]
        if actual.dtype != expected.dtype or not torch.equal(actual, expected):
            raise ValueError(f"base weight was changed during initialization: {name}")
    return expected_new


def make_matched_head(source_config, source_state, *, seed, dim, max_tokens, spatial_bins):
    from dflash.model import DFlashDraftModel
    from transformers import Qwen3Config

    config = copy.deepcopy(source_config)
    draft = dict(config.get("dflash_config", {}))
    if evidence_settings(config)["mode"] != "none":
        raise ValueError("source must have evidence_mode=none")
    for name in ("use_visual_fovea", "use_visual_film", "use_visual_xattn", "use_visual_perceiver",
                 "use_xattn", "use_layer_gate"):
        if draft.get(name, config.get(name, False)):
            raise ValueError(f"unsupported external-memory source: {name}")
    draft.update(evidence_mode="text", evidence_dim=dim,
                 evidence_max_tokens=max_tokens, evidence_spatial_bins=spatial_bins)
    config["dflash_config"] = draft
    model_config = Qwen3Config.from_dict(config)
    model_config._attn_implementation = "sdpa"
    # Isolate CPU initialization RNG; no CUDA RNG or target model is needed.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = DFlashDraftModel(model_config).cpu().eval()
    new_keys = checked_base_load(head, source_state)
    head.evidence_readout.float()
    head.evidence_readout.reset_output()
    if torch.count_nonzero(head.evidence_readout.output.weight).item():
        raise ValueError("evidence residual output must initialize to zero")
    return head, new_keys


def initialize(source, out, *, seed=20261112, dim=256, max_tokens=256,
               spatial_bins=2, dry_run=False):
    source = Path(source).expanduser().resolve()
    out = validate_destination(out)
    config = json.loads((source / "config.json").read_text())
    settings = evidence_settings(config)
    if settings["mode"] != "none":
        raise ValueError("source must be the common head without an evidence readout")
    for name in ("use_visual_fovea", "use_visual_film", "use_visual_xattn", "use_visual_perceiver",
                 "use_xattn", "use_layer_gate"):
        if config.get("dflash_config", {}).get(name, config.get(name, False)):
            raise ValueError(f"unsupported external-memory source: {name}")
    if dim < 1 or max_tokens < 1 or spatial_bins < 1:
        raise ValueError("evidence dimensions and token budgets must be positive")
    taps = config.get("dflash_config", {}).get("target_layer_ids", config.get("target_layer_ids"))
    if taps != [1, 9, 17, 25, 33]:
        raise ValueError("initialization requires audited target_layer_ids [1,9,17,25,33]")
    paths = weight_files(source)
    summary = dict(schema=1, initialization="weights_only_fresh_optimizer",
                   resume_supported=False, source=str(source), A_base=str(source),
                   output=str(out), seed=seed, evidence_dim=dim,
                   evidence_max_tokens=max_tokens, evidence_spatial_bins=spatial_bins,
                   evidence_parameter_dtype="torch.float32", base_dtype="preserved_per_tensor",
                   source_config_sha256=training_resume.file_hash(source / "config.json"),
                   source_weight_files=[path.name for path in paths], dry_run=dry_run)
    if dry_run:
        return summary

    source_hashes = {path.name: training_resume.file_hash(path) for path in paths}
    state = read_weights(paths)
    head, new_keys = make_matched_head(config, state, seed=seed, dim=dim,
                                       max_tokens=max_tokens, spatial_bins=spatial_bins)
    new_state = {name: head.state_dict()[name] for name in sorted(new_keys)}
    new_hash = state_digest(new_state)
    summary.update(source_weights_sha256=source_hashes,
                   base_tensors_verified_equal=True, base_tensor_count=len(state),
                   base_dtypes=sorted({str(value.dtype) for value in state.values()}),
                   added_tensor_count=len(new_keys), added_tensor_sha256=new_hash,
                   added_parameter_count=sum(p.numel() for p in head.evidence_readout.parameters()),
                   output_projection_zero=True,
                   model_source_sha256=training_resume.file_hash(
                       REPRO.parent / "scripts/dflash_pkg/dflash-0.1.0/dflash/model.py"),
                   readout_source_sha256=training_resume.file_hash(
                       REPRO.parent / "scripts/dflash_pkg/dflash-0.1.0/dflash/evidence.py"),
                   helper_source_sha256=training_resume.file_hash(REPRO / "shared/evidence.py"))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    arm_hashes = {}
    for arm, mode in (("B_text", "text"), ("C_visual", "visual")):
        head.config.dflash_config["evidence_mode"] = mode
        # Runtime mode must follow config if a caller inspects this object.
        if hasattr(head, "evidence_mode"):
            head.evidence_mode = mode
        head.evidence_readout.mode = mode
        arm_path = temporary / arm
        head.save_pretrained(arm_path, safe_serialization=True)
        hashes = {path.name: training_resume.file_hash(path) for path in weight_files(arm_path)}
        arm_hashes[arm] = hashes
        if state_digest({name: head.state_dict()[name] for name in new_keys}) != new_hash:
            raise ValueError("added weights changed while saving matched arms")
        manifest = dict(summary, arm=arm, evidence=evidence_settings(arm_path),
                        output_weights_sha256=hashes)
        (arm_path / "INIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if arm_hashes["B_text"] != arm_hashes["C_visual"]:
        raise ValueError("saved B/C weight files differ; only their configs may differ")
    summary.update(arms={arm: str(out / arm) for arm in arm_hashes},
                   matched_saved_weights_verified=True, arm_weights_sha256=arm_hashes)
    (temporary / "INIT_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Refuse an intervening creator, including an empty directory.
    if out.exists():
        raise FileExistsError(f"destination appeared while initializing: {out}")
    os.rename(temporary, out)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="common original head; A uses this directly")
    parser.add_argument("--out", required=True, help="new directory under existing /mnt/local{,2,3}")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--evidence-dim", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--spatial-bins", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="CPU config/path check only; no weight load or writes")
    args = parser.parse_args(argv)
    result = initialize(args.source, args.out, seed=args.seed, dim=args.evidence_dim,
                        max_tokens=args.max_tokens, spatial_bins=args.spatial_bins,
                        dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
