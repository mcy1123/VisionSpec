"""Explicit, process-local Qwen3-VL video processor profiles and eval auditing.

No checkpoint, model configuration, shared loader or encoded-identity function is
modified. This opt-in context patches the actual AutoProcessor factory used by
eval_video_opd as well as checking its concrete processor at each encode call.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    required = {
        "schema": "qwen3_vl_fixed_spatial_engineering_v1",
        "video_processor_class": "Qwen3VLVideoProcessor",
        "video_size": {"shortest_edge": 4096, "longest_edge": 50331648},
        "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
        "expected_spatial_grid_hw": [14, 24],
        "frame_counts": [256, 512], "do_sample_frames": False,
    }
    for key, value in required.items():
        if data.get(key) != value:
            raise ValueError(f"unsupported processor profile {key}: {data.get(key)!r}")
    source = Path(data["original_video_preprocessor_config"])
    if sha256(source) != data["original_video_preprocessor_config_sha256"]:
        raise ValueError("original video preprocessor configuration changed")
    return {"path": str(path), "sha256": sha256(path), "profile": data}


def processor_size(processor):
    return {str(k): int(v) for k, v in processor.video_processor.size.items()}


def apply_profile(processor, profile):
    data = profile["profile"]
    video = processor.video_processor
    if type(video).__name__ != data["video_processor_class"]:
        raise ValueError("processor profile requires the audited Qwen3VL video processor")
    for key in ("patch_size", "merge_size", "temporal_patch_size"):
        if getattr(video, key) != data[key]:
            raise ValueError(f"processor profile mismatch for {key}")
    before = processor_size(processor)
    video.size = dict(data["video_size"])
    return {"before_size": before, "actual_size": processor_size(processor),
            "processor_class": type(processor).__name__,
            "video_processor_class": type(video).__name__}


def validate_row_profile(row, profile):
    if row.get("processor_profile_sha256") != profile["sha256"]:
        raise ValueError("row must be explicitly bound to this processor profile SHA256")
    if int(row["n_frames"]) not in profile["profile"]["frame_counts"]:
        raise ValueError("profile supports only the audited 256/512-frame probe")
    if not row.get("engineering_long_input_probe"):
        raise ValueError("profile is restricted to engineering long-input probe rows")


def validate_encoded(processor, row, inputs, profile):
    validate_row_profile(row, profile)
    data = profile["profile"]
    if processor_size(processor) != data["video_size"]:
        raise ValueError("actual video processor size differs from explicit profile")
    grid = inputs["video_grid_thw"].tolist()
    expected = [[int(row["n_frames"]) // data["temporal_patch_size"],
                 *data["expected_spatial_grid_hw"]]]
    if grid != expected:
        raise ValueError(f"actual processor grid {grid} differs from {expected}")
    token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    visual = int((inputs["input_ids"] == token_id).sum())
    if visual != expected[0][0] * expected[0][1] * expected[0][2] // data["merge_size"] ** 2:
        raise ValueError("actual visual-token count differs from grid/merge calculation")
    length = int(inputs["input_ids"].shape[-1])
    for key, value in (("input_ids_len", length), ("visual_token_count", visual)):
        if row.get(key) is not None and int(row[key]) != value:
            raise ValueError(f"actual encoded {key} differs from audited row metadata")
    return {"video": str(row.get("video_name", row.get("video_id"))),
            "n_frames": int(row["n_frames"]), "input_ids_len": length,
            "visual_token_count": visual, "video_grid_thw": grid,
            "processor_profile_sha256": profile["sha256"],
            "processor_size": processor_size(processor)}


@contextmanager
def profiled_evaluation(module, profile):
    """Scope actual factory, row checks, encoding audit and result provenance.

    common.load_target constructs one processor whose result eval ignores; eval
    constructs another. Both are configured here, and the concrete instance
    passed to encode_inputs must have been seen by this patched factory.
    """
    from transformers import AutoProcessor
    factory = AutoProcessor.from_pretrained
    encode = module.encode_inputs
    validate_rows = module.validate_rows
    write = module.write_result
    instances = []  # Retain objects so id reuse cannot fake factory provenance.
    factories, encoded = [], []

    def configured_factory(*args, **kwargs):
        processor = factory(*args, **kwargs)
        audit = apply_profile(processor, profile)
        instances.append(processor)
        factories.append({"factory_call": len(factories), **audit})
        return processor

    def checked_rows(frame, *args, **kwargs):
        for _, row in frame.iterrows():
            validate_row_profile(row, profile)
        return validate_rows(frame, *args, **kwargs)

    def checked_encode(processor, row, time_mode, **kwargs):
        if time_mode != "real":
            raise ValueError("profiled evaluation requires actual source timestamps")
        if not any(processor is instance for instance in instances):
            raise ValueError("eval used a processor outside the profiled AutoProcessor factory")
        if processor_size(processor) != profile["profile"]["video_size"]:
            raise ValueError("processor size changed after its profiled construction")
        validate_row_profile(row, profile)
        inputs = encode(processor, row, time_mode, **kwargs)
        encoded.append(validate_encoded(processor, row, inputs, profile))
        return inputs

    def audited_write(path, payload):
        payload = deepcopy(payload)
        rows = payload.get("rows", [])
        if len(rows) != len(encoded):
            raise ValueError("result rows do not match actual profiled encoding calls")
        for row, audit in zip(rows, encoded):
            if row["video"] != audit["video"] or row["L"] != audit["input_ids_len"]:
                raise ValueError("result row identity differs from encoding audit")
            row["processor_profile_sha256"] = profile["sha256"]
            row["processor_size"] = audit["processor_size"]
        payload["protocol"]["processor_profile"] = {
            **profile, "factory_calls": factories, "actual_encodings": encoded,
            "actual_processor_instantiated": bool(instances),
            "scope": "process-local factory only; original model/config/cache unchanged",
            "input_identity": "unchanged eval_video_opd.encoded_identity of actual tensors",
            "runtime_source_sha256": sha256(__file__),
        }
        payload["config"]["processor_profile"] = profile["path"]
        payload["config"]["processor_profile_sha256"] = profile["sha256"]
        memory = {
            "scope": "process lifetime on the selected eval GPU up to result write; "
                     "includes model loading and AR/SPEC decoding; not an individual SPEC peak",
            "counter_source": "torch CUDA caching allocator; not total nvidia-smi device usage",
            "counters_reset_by_wrapper": False,
        }
        config = payload["config"]
        # Do not even query CUDA initialization in CPU preflight/audit paths.
        if (config.get("preflight_only") or config.get("cpu_only")
                or str(payload.get("status", "")).startswith("CPU_")):
            memory["status"] = "NOT_COLLECTED_CPU_ONLY"
        else:
            import torch
            if torch.cuda.is_initialized():
                device = int(config.get("gpu", 0))
                memory.update(
                    status="COLLECTED", device=f"cuda:{device}",
                    max_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    max_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                )
            else:
                memory["status"] = "NOT_COLLECTED_CUDA_NOT_INITIALIZED"
        payload["protocol"]["cuda_process_peak_memory"] = memory
        return write(path, payload)

    with patch.object(AutoProcessor, "from_pretrained", staticmethod(configured_factory)), \
         patch.object(module, "validate_rows", checked_rows), \
         patch.object(module, "encode_inputs", checked_encode), \
         patch.object(module, "write_result", audited_write):
        yield {"factory_calls": factories, "actual_encodings": encoded}
