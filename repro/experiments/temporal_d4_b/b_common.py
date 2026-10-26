#!/usr/bin/env python3
"""Task B (parallel 2026-10-12) shared constants and pure-CPU helpers.

Direction B owns `repro/experiments/temporal_d4_b/`,
`results/parallel_B_temporal/<timestamp>/` and
`docs/reports/2026-10-12-parallel-B-temporal.md`.  Nothing here writes inside
`glance/`, `repro/shared/common.py` or any other shared file; the module only imports
them.

Every frozen choice used by the analysis lives in this file so the prereg can
point at concrete code.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

# ---------------------------------------------------------------- identities

C6_RUN = ("/mnt/local/mcy/GLANCE/results/phaseC6/"
          "current_ar_closed_loop_20261012T221104_gpu_scores")
BASELINE_EVAL = "/mnt/local/mcy/GLANCE/results/phaseA4/sft439_step1000_fresh30_tree_float32.json"
C0_PATH = "/mnt/local/mcy/GLANCE/checkpoints/head_video439_aligned/step1000"
C0_MODEL_SHA = "e2526a249a8ef7e1db3a9990589b727541486093372011e1a0ff6a16e06d23d2"
CACHE_DIR = "/mnt/local/mcy/GLANCE/data/video_cache"
SRC_DIR = "/mnt/local2/mcy/datasets/VideoDetailCaption/Test_Videos"
FRESH30 = "/mnt/local/mcy/GLANCE/data/video_fresh30.parquet"

# Frozen C6 six-video sample, in C6 manifest order.
VIDEOS = ["v_-D1gdv_gQyw", "v_1kwIYXB4b8s", "v_0X0u_3xUBPE",
          "v_1KONAdZGdWU", "v_0DFz3sgfda0", "v_0x6ZV9Y8g_c"]

# ---------------------------------------------------------------- conditions

CONDITIONS = ["O", "T", "K"]
# O = original cached clip (bit-identical to C6's input file)
# T = temporal-order intervention: the 16 sampled frames in reverse order
# K = order-preserving control: the same 16 frames in the same order,
#     mirrored horizontally (a general spatial perturbation)
FRAME_ROTATION = None  # unused; kept to document that no rotation is applied

# ------------------------------------------------------- response thresholds

# Pre-registered: a target L1 change below this many nats is treated as "target
# did not respond" and every ratio that divides by it is NA, never a number.
TARGET_L1_NA = 0.05
# Pre-registered probability floor for the candidate set.  A token whose
# probability stays below this under O, T and K contributes to a log-prob L1
# only through its (meaningless) tail magnitude; the 2-video pilot measured
# that this swamps the real mass movement.  Floored tokens are dropped from the
# primary candidate set and kept in the secondary unfiltered one.
PROB_FLOOR = 0.005
# Pre-registered: vectors shorter than this norm are degenerate; cosine is NA.
VEC_NORM_EPS = 1e-8
K_TOP = 8
# Bootstrap
BOOT_N = 2000
BOOT_SEED = 20261012

# Ordering vocabulary.  A caption prefix that contains any of these asserts
# something about *when* things happen, so a full time reversal makes the
# prefix false rather than merely off-distribution.  Such units are excluded
# from the main causal comparison and reported separately.
ORDER_MARKERS = [
    " then ", " after ", " before ", " next ", " finally ", " followed ",
    " following ", " subsequently ", " progresses", " progress", " begins ",
    " begins,", " begins.", " starts ", " started ", " start,", " start.",
    " initially", " at first", " first,", " first.", " first ", " last ",
    " ends ", " ends,", " ends.", " concludes", " transitions", " turns ",
    " turning ", " becomes", " turning into", " as the video",
    " as the scene", " as the clip", " once ", " meanwhile", " later",
    " earlier", " continue", " continues", " throughout", " eventually",
    " sequence", " the video opens", " the video begins", " the clip opens",
    " the clip begins", " opens with", " opens in", " begins with",
    " begins in", " ends with", " closes with", " culminates",
]


def has_order_marker(prefix_text: str) -> tuple[bool, list[str]]:
    """Return (flag, hits) for the prefix text (caption part only)."""
    low = " " + prefix_text.lower().replace("\n", " ") + " "
    hits = [m.strip() for m in ORDER_MARKERS if m in low]
    return (len(hits) > 0), hits


# ---------------------------------------------------------------- utilities

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_json(path: str, value) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def emit_jsonl(path: str, value) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: str):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def log_softmax(values):
    """fp64 log-softmax of a python sequence (list/np array), returns list."""
    m = max(values)
    s = math.log(sum(math.exp(v - m) for v in values)) + m
    return [v - s for v in values]
