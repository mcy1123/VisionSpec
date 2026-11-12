"""Strict, CPU-side requests for the draft-only temporal evidence readout.

The feature is the last *retained* target layer (33), not the target's final
layer. Requests are per encoded input and may acquire detached pooled visual
features during rollout. Keep the request with its trace when scoring another
QA; a head-global request from the last rollout is not interchangeable.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from qa_opd_fair.videoencode import validate_real_metadata


FEATURE_LAYER = 33
FEATURE_PROTOCOL = "target_retained_layer_33_prefc_hidden_v1"
REQUEST_VERSION = "real_video_evidence_v1"
INTERVENTIONS = ("normal", "bypass", "time_shuffle", "zero", "wrong_video")


def evidence_settings(config_or_path_or_head):
    """Read the evidence configuration without loading model weights."""
    value = config_or_path_or_head
    if isinstance(value, (str, Path)):
        path = Path(value)
        value = json.loads((path / "config.json" if path.is_dir() else path).read_text())
    if hasattr(value, "config"):
        value = value.config
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("evidence settings require a config, head, or config path")
    cfg = value.get("dflash_config", {}) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError("dflash_config must be a mapping")
    mode = cfg.get("evidence_mode", value.get("evidence_mode", "none"))
    if mode not in ("none", "text", "visual"):
        raise ValueError(f"unknown evidence_mode: {mode!r}")
    result = {"mode": mode, "feature_layer": FEATURE_LAYER,
              "feature_protocol": FEATURE_PROTOCOL}
    for key, default in (("dim", 256), ("max_tokens", 256), ("spatial_bins", 2)):
        raw = cfg.get(f"evidence_{key}", value.get(f"evidence_{key}", default))
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"evidence_{key} must be a positive integer")
        result[key] = raw
    if mode != "none":
        taps = cfg.get("target_layer_ids", value.get("target_layer_ids"))
        if not isinstance(taps, (list, tuple)) or not taps or taps[-1] != FEATURE_LAYER:
            raise ValueError("evidence readout requires last retained target layer 33")
    return result


def _tensor_digest(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("encoded evidence inputs must be tensors")
    tensor = value.detach().cpu().contiguous()
    # Byte view also supports bf16, although current processor pixels are fp32.
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _positive_processor_int(processor, name):
    video = getattr(processor, "video_processor", processor)
    raw = getattr(video, name, None)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(f"processor.video_processor.{name} must be an explicit positive integer")
    return raw


def build_evidence_request(inputs, row, processor, identity=None,
                           intervention="normal", seed=0):
    """Bind real timeline metadata to the exact single-video processor output.

    ``identity`` may be the freshly computed encoded-input SHA256 mapping used
    by eval. This avoids hashing large pixels twice; the caller owns freshness
    of that mapping. A string is an extra label and does not replace hashing.
    The returned identity also binds tensor shapes/dtypes, frame sampling,
    duration, processor grouping, and the feature-layer contract. No target
    input or request tensor is modified by this function.
    """
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unsupported evidence intervention: {intervention!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("evidence seed must be an integer")
    metadata = validate_real_metadata(row, require_cache=False, require_exact=True)
    n_frames = int(row["n_frames"])
    if row.get("n_unique_frames") is not None and int(row["n_unique_frames"]) != n_frames:
        raise ValueError("n_unique_frames differs from the unique real-frame manifest")
    required = ("input_ids", "pixel_values_videos", "video_grid_thw")
    if any(key not in inputs or not isinstance(inputs[key], torch.Tensor) for key in required):
        raise ValueError("evidence request needs tensor input_ids, pixels, and video_grid_thw")
    ids = inputs["input_ids"].detach().cpu()
    grid = inputs["video_grid_thw"].detach().cpu()
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("evidence supports one nonempty integer input_ids sequence")
    if grid.shape != (1, 3) or grid.dtype not in (torch.int32, torch.int64):
        raise ValueError("evidence requires one integer video_grid_thw shaped (1,3)")
    t, height, width = [int(x) for x in grid[0].tolist()]
    if min(t, height, width) < 1:
        raise ValueError("video_grid_thw dimensions must be positive")
    temporal = _positive_processor_int(processor, "temporal_patch_size")
    merge = _positive_processor_int(processor, "merge_size")
    if t * temporal != n_frames:
        raise ValueError("video temporal grid differs from true frames / temporal_patch_size; padding is unsupported")
    if height % merge or width % merge:
        raise ValueError("video spatial grid must be divisible by spatial merge_size")
    pixels = inputs["pixel_values_videos"]
    if pixels.ndim != 2 or pixels.shape[0] != t * height * width:
        raise ValueError("encoded video patch count differs from video_grid_thw")
    video_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    if isinstance(video_id, bool) or not isinstance(video_id, int) or video_id < 0:
        raise ValueError("processor tokenizer has no valid video_pad token")
    if video_id == getattr(processor.tokenizer, "unk_token_id", None):
        raise ValueError("processor tokenizer maps video_pad to its unknown token")
    visual_mask = ids[0].eq(video_id)
    visual_count = t * (height // merge) * (width // merge)
    if int(visual_mask.sum()) != visual_count:
        raise ValueError("prompt visual token count differs from merged video grid")
    duration = float(row["video_duration"])
    frame_times = torch.tensor(list(row["sample_timestamps"]), dtype=torch.float64)
    time_seconds = frame_times.reshape(t, temporal).mean(dim=1)
    times = (time_seconds / duration).to(torch.float32)
    if isinstance(identity, Mapping):
        hashes = {key: identity.get(key) for key in required}
        if any(not isinstance(digest, str) or len(digest) != 64
               or any(c not in "0123456789abcdef" for c in digest) for digest in hashes.values()):
            raise ValueError("encoded input identity must contain SHA256 for ids, pixels, and grid")
    elif identity is None or isinstance(identity, str):
        hashes = {key: _tensor_digest(inputs[key]) for key in required}
    else:
        raise ValueError("identity must be an encoded SHA256 mapping, string label, or None")
    identity_payload = {
        "version": REQUEST_VERSION, "feature_protocol": FEATURE_PROTOCOL,
        "encoded_sha256": hashes,
        "encoded_layout": {key: {"shape": list(inputs[key].shape), "dtype": str(inputs[key].dtype)}
                           for key in required},
        "sample_frame_ids": [int(x) for x in row["sample_frame_ids"]],
        "sample_timestamps": [float(x) for x in row["sample_timestamps"]],
        "source_fps": float(row["source_fps"]), "duration": duration,
        "temporal_patch_size": temporal, "spatial_merge_size": merge,
        "external_label": identity if isinstance(identity, str) else None,
    }
    digest = hashlib.sha256(json.dumps(identity_payload, sort_keys=True, allow_nan=False,
                                      separators=(",", ":")).encode()).hexdigest()
    return {
        "identity": digest, "prompt_visual_mask": visual_mask.clone(),
        "grid_thw": [t, height, width], "spatial_merge_size": merge,
        "time_values": times, "intervention": intervention, "seed": seed,
        "metadata": {"version": REQUEST_VERSION, "feature_layer": FEATURE_LAYER,
                     "feature_protocol": FEATURE_PROTOCOL, "prompt_length": ids.shape[1],
                     "visual_tokens": visual_count, "n_unique_frames": n_frames,
                     "temporal_patch_size": temporal, "duration_s": duration,
                     "patch_times_s": time_seconds.tolist(), "normalized_time": True,
                     "encoded_sha256": hashes, **metadata},
    }


def configure_evidence(head, request):
    """Bind one request for one rollout / scoring trace / deployment request."""
    if evidence_settings(head)["mode"] == "none":
        return
    if not isinstance(request, dict) or not request.get("identity"):
        raise ValueError("enabled evidence readout requires an identity-bound request")
    setter = getattr(head, "set_evidence_request", None)
    if not callable(setter):
        raise ValueError("head config enables evidence but model lacks set_evidence_request")
    setter(request)
