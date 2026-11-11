#!/usr/bin/env python3
"""D2: keep the sampled frames and their pixels fixed, correct only the timeline.

The historical pipeline samples 16 frames evenly from the source video, resizes
them to long side 384, re-encodes them as a 16-frame / 30 fps MP4 and hands that
file to the processor. Qwen3-VL builds its `<X.Y seconds>` text markers and its
`get_rope_index` temporal layout from the video metadata the *processor* sees, so
those markers describe the CACHE (0.0 .. 0.5 s), not the source clip. That is the
defect: the visual tensor is right, the timeline is wrong.

This module changes ONLY the timeline:

* `legacy` -- the historical call: `videos=[cache_path], fps=30`. Reproduces the
  existing input bit for bit (asserted by the D2 gate, not assumed).
* `real` -- the same cache file, but handed to the same processor with explicit
  `VideoMetadata(total_num_frames=16, fps=source_fps, frames_indices=<the 16 real
  source frame ids>, duration, height, width)` and `do_sample_frames=False`, so
  the frame count, frame pixels, grid and M-RoPE layout are untouched while the
  processor's `_calculate_timestamps` computes `idx / source_fps` for the REAL
  frame indices.

`_calculate_timestamps` averages the timestamps of the `merge_size` (=2) frames
that share one temporal patch, which is exactly what the model's time tokens are
built from, so the emitted markers are patch means, not raw PTS. The gate checks
monotonic growth against the source PTS at that resolution rather than demanding
one marker per frame.

VFR caveat: for a variable-frame-rate source, `frame_index / source_fps` is not
the true PTS. The per-frame PyAV PTS is available in the manifest, so the gate
reports the deviation instead of hiding it, and any sample whose real timeline
cannot be represented this way is flagged `real_time_exact: false`. Strict
`encode_inputs(..., 'real')` now rejects those samples. New real-frame pilots
may use lossless NPZ caches, which are only supported by this strict real mode.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
ROOT = os.path.dirname(REPRO)
for _p in (REPRO, os.path.join(ROOT, "GLANCE_repo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ["build_video_text", "encode_inputs", "input_key", "encode_signature",
           "validate_real_metadata"]


def validate_real_metadata(row, *, require_cache=True, require_exact=False,
                           tolerance_s=1e-3):
    """Validate a real-frame manifest without loading a model or importing torch.

    Frame ids refer to the source decode order, timestamps to source PTS relative
    to its first decoded frame. Qwen's current metadata interface represents time
    as ``frame_id / source_fps``. An irregular source may have valid PTS yet fail
    this representation; ``require_exact`` rejects that case explicitly.
    """
    import math
    from pathlib import Path

    required = ("n_frames", "source_fps", "video_duration", "sample_frame_ids",
                "sample_timestamps", "height", "width")
    missing = [key for key in required if key not in row or row[key] is None]
    if missing:
        raise ValueError(f"missing real video metadata: {missing}")
    n = int(row["n_frames"])
    if n < 1 or n != row["n_frames"]:
        raise ValueError("n_frames must be a positive integer")
    fps, duration = float(row["source_fps"]), float(row["video_duration"])
    if not all(math.isfinite(x) and x > 0 for x in (fps, duration)):
        raise ValueError("source_fps and video_duration must be finite and positive")
    if any(int(row[key]) <= 0 for key in ("height", "width")):
        raise ValueError("video dimensions must be positive")
    raw_ids = list(row["sample_frame_ids"])
    ids = [int(x) for x in raw_ids]
    ts = [float(x) for x in row["sample_timestamps"]]
    if len(ids) != n or len(ts) != n:
        raise ValueError("frame ids and timestamps must both have n_frames entries")
    if any(i != raw or i < 0 for i, raw in zip(ids, raw_ids)):
        raise ValueError("source frame ids must be non-negative integers")
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ValueError("source frame ids must be unique and strictly increasing")
    if any(not math.isfinite(t) or t < 0 or t > duration + tolerance_s for t in ts):
        raise ValueError("source timestamps must be finite and within video duration")
    if any(a >= b for a, b in zip(ts, ts[1:])):
        raise ValueError("source timestamps must be strictly increasing")
    source_count = row.get("source_frame_count")
    if source_count is not None and (int(source_count) <= ids[-1]):
        raise ValueError("sampled frame id exceeds source_frame_count")
    exact, error = real_time_exact(row, tol=tolerance_s)
    if require_exact and not exact:
        raise ValueError(f"source PTS cannot be represented exactly by frame_id/fps "
                         f"(max error {error:.6f}s); strict real mode refuses this input")
    if require_cache:
        path = row.get("cache_path")
        if not path or not Path(str(path)).is_file():
            raise ValueError("real input needs an existing cache_path; metadata-only "
                             "manifests must first be materialized with --cache-root")
    return {"real_time_exact": exact, "max_timestamp_error_s": error,
            "n_unique_frames": n, "time_origin": row.get("time_origin", "unspecified")}


def build_video_text(proc, question_prompt):
    """The chat-template string with the single video placeholder."""
    return proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": question_prompt}]}],
        add_generation_prompt=True, tokenize=False)


def _video_metadata(row):
    from transformers.video_utils import VideoMetadata

    return VideoMetadata(
        total_num_frames=int(row["n_frames"]),
        fps=float(row["source_fps"]),
        duration=float(row["video_duration"]),
        frames_indices=[int(x) for x in row["sample_frame_ids"]],
        height=int(row["height"]),
        width=int(row["width"]),
    )


def _cache_frames(row):
    """All preselected cache frames as an (N, C, H, W) uint8 tensor.

    MEASURED, and the reason this function exists: passing the cache PATH plus
    `video_metadata` does NOT work with transformers 4.57.3. The processor
    forwards `do_sample_frames=False` correctly, but `fetch_videos` re-infers
    metadata from the file, so the real frame indices never reach
    `_calculate_timestamps` and the markers stay at the cache's own fps
    (verified: `<0.0 .. 0.5 seconds>` for a 39 s clip). Passing the DECODED
    FRAMES as a tensor keeps `make_batched_metadata`'s use of the supplied
    `video_metadata`, and then the markers come out as
    `<0.2, 0.8, 1.5, ...>` for the same clip. The pixel tensor is byte-identical
    to the path-based call in either case, which is exactly the property the
    adapter needs: same frames, corrected timeline.
    """
    import numpy as np
    from pathlib import Path
    import torch

    if Path(str(row["cache_path"])).suffix == ".npz":
        with np.load(str(row["cache_path"]), allow_pickle=False) as data:
            pixels = data["frames"]
            if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1] != 3:
                raise ValueError("frame cache must contain uint8 frames shaped (N,3,H,W)")
            frames = torch.from_numpy(pixels.copy())
    else:
        from torchvision.io import read_video
        frames, _, _ = read_video(str(row["cache_path"]), pts_unit="sec",
                                  output_format="TCHW")
    n = int(frames.shape[0])
    if n != int(row["n_frames"]):
        raise RuntimeError(f"{row.get('video_id', 'video')}: cache decodes to {n} frames but "
                           f"the manifest says {row['n_frames']}")
    if tuple(frames.shape[-2:]) != (int(row["height"]), int(row["width"])):
        raise ValueError("decoded frame dimensions differ from the manifest")
    return frames


def encode_inputs(proc, row, time_mode, *, text=None):
    """Encode one (video, question) row under `legacy` or `real` time semantics.

    Both modes decode the SAME cache file and let the processor do the same
    resize/normalise/patchify, so `pixel_values_videos` and `video_grid_thw` are
    byte-identical between them. The only difference is the metadata the
    processor reads for its time markers and `get_rope_index`.
    """
    if time_mode not in ("legacy", "real"):
        raise ValueError(f"time_mode must be 'legacy' or 'real', got {time_mode!r}")
    text = text or build_video_text(proc, row["prompt"])
    if time_mode == "legacy":
        if str(row["cache_path"]).endswith(".npz"):
            raise ValueError("NPZ real-frame caches require --time-mode real")
        return proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                    fps=float(row["fps"]))
    validate_real_metadata(row, require_exact=True)
    return proc(text=[text], videos=[_cache_frames(row)],
                video_metadata=[_video_metadata(row)],
                do_sample_frames=False, return_tensors="pt")


def real_time_exact(row, tol=1e-3):
    """Whether `frame_index / source_fps` reproduces the manifest PTS.

    Returns `(exact, max_abs_error_s)`. A CFR source passes with error at the
    float-rounding level; a VFR source does not, and its rows must be marked as
    approximations rather than reported as true timelines.
    """
    ids = [int(x) for x in row["sample_frame_ids"]]
    ts = [float(x) for x in row["sample_timestamps"]]
    fps = float(row["source_fps"])
    errs = [abs(i / fps - t) for i, t in zip(ids, ts)]
    return (max(errs) <= tol if errs else True), (max(errs) if errs else 0.0)


def encode_signature(x, tokenizer=None):
    """Content signature of an encoded input: ids, grid, pixel digest, markers."""
    import hashlib
    import re
    import torch

    def digest(t):
        t = t.detach().cpu().contiguous()
        # The processor returns float32 pixels; keep them as they are so the
        # digest is a pure byte identity check between the two modes.
        return hashlib.sha256(t.numpy().tobytes()).hexdigest()

    sig = {"input_ids": x["input_ids"].tolist(),
           "video_grid_thw": x["video_grid_thw"].tolist(),
           "pixel_values_videos": digest(x["pixel_values_videos"]),
           "n_video_pad_tokens": int((x["input_ids"] ==
                                      tokenizer.convert_tokens_to_ids("<|video_pad|>")
                                      ).sum()) if tokenizer else None}
    if tokenizer is not None:
        decoded = tokenizer.decode(x["input_ids"][0].tolist())
        sig["time_tokens"] = re.findall(r"<(\d+\.\d) seconds>", decoded)
    if "image_grid_thw" in x:
        sig["image_grid_thw"] = x["image_grid_thw"].tolist()
    return sig


def input_key(row, time_mode, processor_hash, prompt_override=None):
    """Content key for caching: pixels, timeline, prompt, processor, mode.

    The mode is part of the key because legacy and real produce different token
    streams for the same video; mixing their fused contexts would be a silent
    correctness bug.
    """
    import hashlib
    import json

    payload = {
        "video_sha256": row["hash"],
        "cache_sha256": row["cache_hash"],
        "sample_frame_ids": [int(x) for x in row["sample_frame_ids"]],
        "sample_timestamps": [float(x) for x in row["sample_timestamps"]],
        "source_fps": float(row["source_fps"]),
        "prompt": prompt_override or row["prompt"],
        "processor": processor_hash,
        "time_mode": time_mode,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()
