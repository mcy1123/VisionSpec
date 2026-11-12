"""Frozen visual-memory export and identity-checked wrong-video probes.

Banks contain tensors and plain metadata only, and are loaded with weights_only.
They are diagnostic artifacts: normal SPEC timing excludes export, and donor
loading is reported separately. No additional target forward is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from shared.evidence import FEATURE_PROTOCOL, _tensor_digest


STORAGE_ROOTS = tuple(Path(p) for p in ("/mnt/local", "/mnt/local2", "/mnt/local3"))
BANK_VERSION = "frozen_visual_evidence_bank_v1"


def checked_bank_path(path, *, writing=False):
    """Require an existing real storage parent; never create mount aliases."""
    value = Path(path).expanduser()
    if writing and (value.exists() or value.is_symlink()):
        raise FileExistsError(f"memory bank output already exists: {value}")
    parent = value.parent.resolve(strict=True)
    resolved = parent / value.name if writing else value.resolve(strict=True)
    allowed = [root.resolve() for root in STORAGE_ROOTS if root.is_dir()]
    if not any(resolved.is_relative_to(root) for root in allowed):
        raise ValueError("evidence memory must physically reside under an existing /mnt/local, /mnt/local2, or /mnt/local3")
    if not writing and not resolved.is_file():
        raise ValueError("evidence memory bank must be a regular file")
    return resolved


def source_identity(row):
    value = row.get("source_sha256", row.get("hash"))
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("evidence memory export/probe requires source_sha256 (or source hash)")
    return value


def bank_contract(settings, artifacts, model_contract, dtype):
    if settings["mode"] != "visual":
        raise ValueError("frozen evidence banks require visual mode")
    result = {"version": BANK_VERSION, "feature_protocol": FEATURE_PROTOCOL,
              "settings": settings, "model_contract": model_contract,
              "target_dtype": dtype,
              "artifacts": {key: artifacts[key] for key in (
                  "target_config_sha256", "head_config_sha256", "head_weights_sha256",
                  "checkpoint_identity", "checkpoint_identity_sha256",
                  "source_sha256", "target_weight_verification")}}
    # Enforce a plain, finite metadata contract (no custom pickled classes).
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def memory_entry(video, row, request):
    if request.get("intervention") != "normal":
        raise ValueError("only unmodified normal visual memory can be exported")
    if not isinstance(video, str) or not video:
        raise ValueError("memory export requires a nonempty video name")
    prepared = request.get("prepared_visual")
    if not isinstance(prepared, dict) or prepared.get("identity") != request["identity"]:
        raise ValueError("normal evaluation did not produce identity-bound visual memory")
    with torch.inference_mode(False), torch.no_grad():
        frozen = {key: prepared[key] for key in ("identity", "config", "groups", "spatial_bins")}
        for key in ("features", "times"):
            frozen[key] = prepared[key].detach().cpu().clone()
    return {"video": video, "identity": request["identity"],
            "source_sha256": source_identity(row),
            "metadata": json.loads(json.dumps(request["metadata"], allow_nan=False)),
            "prepared": frozen,
            "tensor_sha256": {key: _tensor_digest(frozen[key]) for key in ("features", "times")}}


def entry_key(entry):
    return entry["video"] + ":" + entry["identity"]


def _validate_entry(entry, contract):
    if not isinstance(entry, dict) or not isinstance(entry.get("video"), str) or not entry["video"]:
        raise ValueError("invalid memory-bank video entry")
    source_identity(entry)
    identity, prepared, metadata = entry.get("identity"), entry.get("prepared"), entry.get("metadata")
    if (not isinstance(identity, str) or len(identity) != 64 or not isinstance(prepared, dict)
            or prepared.get("identity") != identity or not isinstance(metadata, dict)):
        raise ValueError("memory bank entry identity/metadata mismatch")
    cfg = prepared.get("config")
    settings = contract["settings"]
    hidden = contract["model_contract"]["hidden_size"]
    if not isinstance(cfg, (tuple, list)) or len(cfg) != 5:
        raise ValueError("invalid prepared visual pooling config")
    grid, merge, budget, bins, stored_hidden = cfg
    if (not isinstance(grid, (tuple, list)) or len(grid) != 3
            or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in grid)
            or not isinstance(merge, int) or isinstance(merge, bool) or merge < 1
            or grid[1] % merge or grid[2] % merge):
        raise ValueError("invalid prepared visual source grid")
    if (budget, bins, stored_hidden) != (settings["max_tokens"], settings["spatial_bins"], hidden):
        raise ValueError("memory-bank readout capacity/config mismatch")
    spatial = min(bins, grid[1] // merge) * min(bins, grid[2] // merge)
    groups = min(grid[0], budget // spatial)
    count = spatial * groups
    if groups < 1 or prepared.get("groups") != groups or prepared.get("spatial_bins") != spatial:
        raise ValueError("prepared memory temporal/spatial groups mismatch")
    features, times = prepared.get("features"), prepared.get("times")
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32
            or features.shape != (count, hidden) or features.requires_grad
            or features.device.type != "cpu" or not bool(torch.isfinite(features).all())):
        raise ValueError("invalid frozen CPU visual features")
    if (not isinstance(times, torch.Tensor) or times.dtype != torch.float32
            or times.shape != (count,) or times.requires_grad or times.device.type != "cpu"
            or not bool(torch.isfinite(times).all()) or bool(((times < 0) | (times > 1)).any())):
        raise ValueError("invalid frozen CPU visual times")
    if metadata.get("feature_protocol") != FEATURE_PROTOCOL:
        raise ValueError("memory-bank feature protocol mismatch")
    patch_times = metadata.get("patch_times_s", [])
    duration = metadata.get("duration_s", 0)
    if len(patch_times) != grid[0] or duration <= 0:
        raise ValueError("memory-bank timeline metadata mismatch")
    raw_times = torch.tensor(patch_times, dtype=torch.float32) / duration
    expected = F.adaptive_avg_pool1d(raw_times[None, None], groups)[0, 0].repeat_interleave(spatial)
    if not torch.allclose(times, expected, atol=1e-6, rtol=1e-5):
        raise ValueError("pooled memory times differ from the real source timeline")
    encoded = metadata.get("encoded_sha256", {})
    if set(encoded) != {"input_ids", "pixel_values_videos", "video_grid_thw"} or any(
            not isinstance(value, str) or len(value) != 64 for value in encoded.values()):
        raise ValueError("memory bank lacks encoded input identity")
    if entry.get("tensor_sha256") != {key: _tensor_digest(prepared[key]) for key in ("features", "times")}:
        raise ValueError("memory-bank tensor SHA256 mismatch")


def validate_bank(bank, expected_contract):
    if not isinstance(bank, dict) or bank.get("contract") != expected_contract:
        raise ValueError("memory-bank target/head/code/feature/readout contract mismatch")
    entries = bank.get("entries")
    if not isinstance(entries, dict) or len(entries) < 2:
        raise ValueError("wrong-video memory bank needs at least two videos")
    videos = set()
    for key, entry in entries.items():
        _validate_entry(entry, expected_contract)
        if key != entry_key(entry) or entry["video"] in videos:
            raise ValueError("memory bank key mismatch or repeated video")
        videos.add(entry["video"])
    return bank


def save_bank(path, contract, entries):
    path = checked_bank_path(path, writing=True)
    payload = validate_bank({"contract": contract, "entries": entries}, contract)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".evidence-bank-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses an existing destination.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_bank(path, expected_contract):
    path = checked_bank_path(path)
    bank = torch.load(path, map_location="cpu", weights_only=True)
    return validate_bank(bank, expected_contract)


def select_donor(bank, video, row, request):
    """Deterministically choose the next compatible, distinct source in a cycle."""
    key = video + ":" + request["identity"]
    entries = bank["entries"]
    if key not in entries:
        raise ValueError("current video/input identity is absent from the normal memory bank")
    current = entries[key]
    if (current["source_sha256"] != source_identity(row)
            or current["metadata"]["encoded_sha256"] != request["metadata"]["encoded_sha256"]):
        raise ValueError("current source/encoded identity differs from normal memory bank")
    ordered = sorted(entries, key=lambda item: (entries[item]["video"], entries[item]["identity"]))
    start = ordered.index(key)
    prepared = current["prepared"]
    for offset in range(1, len(ordered)):
        donor_key = ordered[(start + offset) % len(ordered)]
        donor = entries[donor_key]
        candidate = donor["prepared"]
        if (donor["video"] == video or donor["identity"] == request["identity"]
                or donor["source_sha256"] == current["source_sha256"]
                or donor["metadata"]["encoded_sha256"]["pixel_values_videos"]
                == current["metadata"]["encoded_sha256"]["pixel_values_videos"]):
            continue
        # Raw source H/W may differ; compressed token/group capacity must match.
        if (tuple(candidate["config"][1:]) != tuple(prepared["config"][1:])
                or candidate["features"].shape != prepared["features"].shape
                or candidate["groups"] != prepared["groups"]
                or candidate["spatial_bins"] != prepared["spatial_bins"]):
            continue
        return ({**candidate, "donor_video": donor["video"]},
                {"video": donor["video"], "identity": donor["identity"],
                 "source_sha256": donor["source_sha256"],
                 "encoded_sha256": donor["metadata"]["encoded_sha256"],
                 "bank_key": donor_key, "memory_tokens": len(candidate["times"]),
                 "time_values": candidate["times"].tolist()})
    raise ValueError("no compatible different-video/source/pixels donor exists in memory bank")
