#!/usr/bin/env python3
"""Image-input adapter for the frozen QA trainers.

WHY A SEPARATE MODULE. The QA position contract that this round must reuse was
validated on video (`repro/qa_opd_fair/`). The task allows changing *only* the
image input adaptation when moving to ScienceQA, and forbids touching the
production tree/decode/target code. This module is that adaptation and nothing
else: it produces the same kinds of objects the video path produced
(``input_ids``, a vision tensor, a vision grid, a bound ``VisionPositions``) from
a single real image, and it changes no objective, no mask, no position rule and
no decode path.

The one asymmetry worth naming: `videoencode.encode_inputs` had to reconstruct a
video timeline because the historical cache re-encoded 16 frames at 30 fps. A
still image has no timeline, so the `--time-mode` flag of the video trainers is
irrelevant here; the image path is a single, unambiguous encoding. The copied
trainers keep accepting the flag only so their CLI matches the frozen scripts and
so a stray `--time-mode real` cannot silently change anything.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
ROOT = os.path.dirname(REPRO)
for _p in (REPRO, os.path.join(ROOT, "GLANCE_repo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ["build_image_text", "encode_image_inputs",
           "prefix_image_vision_positions", "image_input_key"]


def build_image_text(proc, question_prompt):
    """The chat-template string with exactly one image placeholder.

    Mirrors the repo's image generation path: one user turn, image first, then
    text (`common.build_processor_messages`). Keeping that order is what makes the
    text stream and the `<|image_pad|>` count agree with `glance.generate`.
    """
    messages = [{"role": "user", "content": [{"type": "image"},
                                            {"type": "text", "text": question_prompt}]}]
    return proc.apply_chat_template(messages, add_generation_prompt=True,
                                    tokenize=False)


def encode_image_inputs(proc, row, *, max_pixels=None):
    """Encode one (image, prompt) row into the tensors the target consumes.

    `row` needs `image_path` and `prompt` (both present in the frozen
    `records_manifest.csv` / pool parquet). Returns `input_ids`,
    `pixel_values`, `image_grid_thw` and the attention mask, as CPU tensors.

    The image is resized only when `max_pixels` is given, by the same
    longest-side rule `common.load_image` uses, so training and evaluation agree
    on the pixels. No padding, no placeholder strings, no synthetic images: a
    missing or unreadable file raises instead of being substituted.
    """
    from PIL import Image

    from shared import common

    path = str(row["image_path"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"image for {row.get('pid', '?')} is missing: {path}")
    with Image.open(path) as im:
        image = im.convert("RGB")
    if max_pixels and max(image.size) > int(max_pixels):
        w, h = image.size
        s = int(max_pixels) / max(w, h)
        image = image.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                             Image.BICUBIC)
    # Delegate to `common.encode`: it is the exact call the repo's image
    # generation path (`eval_chartqa.py`) uses, so training and evaluation cannot
    # drift apart in binding, template or tensor layout.
    ids, pixels, grid = common.encode(proc, row["prompt"], image)
    if pixels is None:
        raise RuntimeError("processor produced no pixel_values for an image row")
    return {"input_ids": ids, "pixel_values": pixels, "image_grid_thw": grid}


def prefix_image_vision_positions(target, ids, pixel_values, image_grid_thw):
    """A bound `VisionPositions` for an image prefix.

    `opd_fullkl.prefix_vision_positions` is the validated factory but it is typed
    for video (`pixel_values_videos` / `video_grid_thw`). `glance.vl.VisionPositions`
    already supports the image arguments and `prefix_vision_positions` does
    nothing beyond constructing it and calling `bind`, so this reproduces exactly
    that construction for the image case instead of editing the validated module.
    """
    from glance.vl import VisionPositions

    if ids.ndim == 1:
        ids = ids[None]
    pos = VisionPositions(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    pos.bind(target, ids)
    return pos


def image_input_key(row, prompt, processor_hash):
    """Content key for caching: pixels, prompt, processor, resize policy."""
    import json

    payload = {"image_sha256": row["image_sha256"], "prompt": prompt,
               "processor": processor_hash,
               "image_path": os.path.basename(str(row["image_path"]))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()
