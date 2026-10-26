#!/usr/bin/env python3
"""Normalize video files into a fixed-geometry cache, and build the pool index.

Why normalize at all. Two independent reasons, both measured:

1. The processor only honours frame count through the FILE-PATH + fps route
   ("pre-decoded frames + metadata" silently yields a fixed grid regardless of
   how many frames are passed). So the file has to exist on disk.
2. If fps is fixed, the number of frames the processor samples scales with the
   SOURCE DURATION: the same `fps=30` gave 16 frames on a 0.53 s clip but 240
   frames (10560 tokens) on an 8 s one. Writing a clip whose duration is exactly
   `n_frames / fps` decouples frame count from source length, so "16 frames"
   means 16 frames for every row.

Secondary benefit: it also unifies resolution (the 150 sources run 640x480 to
1280x720 and mix h264 with vp9), which is what makes token counts comparable
across rows at all.

Failures are NOT swallowed: each per-row exception is logged with its full
traceback, retried up to `--retries` times, and counted in the final report.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(_REPRO_ROOT))

DEFAULT_PARQUET = ("/mnt/local2/mcy/datasets/VideoDetailCaption/data/"
                   "test-00000-of-00001.parquet")
DEFAULT_VIDEO_ROOT = "/mnt/local2/mcy/datasets/VideoDetailCaption/Test_Videos"


def resolve(root, name):
    """The parquet stores `video_name` without an extension."""
    for cand in (name, name + ".mp4", name + ".mkv", name + ".webm"):
        p = os.path.join(root, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no video for {name!r} under {root}")


def normalize(src, dst, n_frames, fps, long_side):
    """Read `n_frames` frames spread over `src` and write a fixed-geometry clip.

    Returns (width, height) actually written.
    """
    import cv2
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {src}")
    sw, sh = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = long_side / max(sw, sh)
    w, h = max(int(sw * scale) // 2 * 2, 2), max(int(sh * scale) // 2 * 2, 2)
    idx = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    frames = []
    for i in idx:
        # cv2 commonly reports one frame too many, so seeking to the last index
        # returns False. Step back until a frame is actually delivered instead of
        # failing the whole row -- 13 of 150 rows died on exactly this.
        got = None
        for back in range(0, 25):
            j = int(i) - back
            if j < 0:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            ok, f = cap.read()
            if ok and f is not None:
                got = f
                break
        if got is None:
            raise RuntimeError(
                f"cv2 could not deliver a frame near index {i} of {total} "
                f"(tried 25 back)")
        frames.append(cv2.resize(got, (w, h), interpolation=cv2.INTER_AREA))
    cap.release()
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cv2 VideoWriter cannot open {dst}")
    for f in frames:
        vw.write(f)
    vw.release()
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError(f"wrote nothing to {dst}")
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", required=True, help="output pool parquet")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--heldout", type=int, default=30,
                    help="exactly this many rows go to the held-out split; "
                         "extra successful rows beyond train+heldout are dropped "
                         "so that per-row failures cannot shrink the eval set")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--long-side", type=int, default=384)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    # Growing the pool must not re-encode what is already on disk (the first 150
    # took ~75 min) and must not re-cut the splits: re-deriving them from a new
    # `--limit` would silently move the 30 rows every earlier number is anchored
    # to into training. `--reuse-split-from` keeps the prior parquet's assignment
    # verbatim and only routes genuinely new rows.
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse a cache file that is already on disk (only the "
                         "tokenizer length and grid are re-measured)")
    ap.add_argument("--reuse-split-from", default=None,
                    help="pool parquet whose video_name->split map is kept as-is")
    # Decoding is ~30 s/clip and pure CPU (96 cores here), so the pool is built
    # as N disjoint shards over the SAME precomputed plan, then concatenated.
    # Sharding cannot change the split: `assign` is computed before the slice.
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    import cv2  # noqa: F401  (fail early if the decoder is missing)
    import pandas as pd
    from transformers import AutoProcessor
    from shared import common

    log_path = args.log or os.path.join(os.path.dirname(args.index),
                                        "predecode_cache.log")
    log = open(log_path, "a")

    def say(m):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    rng = np.random.default_rng(args.seed)
    picked = rng.permutation(len(df))[: args.limit] if args.limit else np.arange(len(df))
    rows = df.iloc[sorted(picked)].reset_index(drop=True)
    say(f"selected {len(rows)}/{len(df)} rows (seed {args.seed})")

    prior = {}
    if args.reuse_split_from:
        pr = pd.read_parquet(args.reuse_split_from)
        prior = {str(n): str(s) for n, s in zip(pr["video_name"], pr["split"])}
        say(f"reuse-split-from {args.reuse_split_from}: {len(prior)} rows "
            f"({sum(v == 'train' for v in prior.values())} train / "
            f"{sum(v == 'heldout' for v in prior.values())} heldout)")

    # Plan the whole split up front, so a row's split never depends on how many
    # earlier rows happened to fail. Prior rows keep their split verbatim; fresh
    # rows fill the held-out quota first, then train; anything past both quotas
    # is dropped rather than silently promoted.
    order = [str(n) for n in rows["video_name"]]
    assign, n_t, n_h = {}, 0, 0
    for name in order:
        if name in prior:
            assign[name] = prior[name]
            n_t += prior[name] == "train"
            n_h += prior[name] == "heldout"
    for name in order:
        if name in assign:
            continue
        if n_h < args.heldout:
            assign[name], n_h = "heldout", n_h + 1
        elif n_t < args.train:
            assign[name], n_t = "train", n_t + 1
        else:
            assign[name] = None
    say(f"planned splits: {sum(v == 'train' for v in assign.values())} train / "
        f"{sum(v == 'heldout' for v in assign.values())} heldout / "
        f"{sum(v is None for v in assign.values())} dropped "
        f"(target {args.train}+{args.heldout})")

    # Processor only -- the 8B model is not needed to measure input_ids length.
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    # The prompt is the pool's own `question` column -- one value for the whole
    # video pool. Using it here (rather than a second hardcoded copy of the
    # template) is what makes `input_ids_len` the length the trainer will
    # actually see; the two spellings of this prompt differ by one word
    # ("overall scene" vs "background scenes") and that was a real mismatch.
    if df["question"].nunique() != 1:
        raise RuntimeError(f"video pool expects one prompt, got "
                           f"{df['question'].nunique()} distinct questions")

    it = [r for _, r in rows.iterrows()
          if assign.get(str(r["video_name"])) is not None]
    if args.shards > 1:
        it = it[args.shard::args.shards]
        say(f"shard {args.shard}/{args.shards}: {len(it)} rows "
            f"(of {sum(v is not None for v in assign.values())} planned)")
    recs, failures = [], []
    retry_hist = {}
    for i, row in enumerate(it):
        name = str(row["video_name"])
        dst = os.path.join(args.out_dir, f"{name}.mp4")
        attempts, err = 0, None
        for attempts in range(1, args.retries + 1):
            try:
                if args.skip_existing and os.path.exists(dst):
                    import cv2 as _cv2
                    _c = _cv2.VideoCapture(dst)
                    wh = (int(_c.get(3)), int(_c.get(4)))
                    _c.release()
                    if wh[0] <= 0 or wh[1] <= 0:
                        raise RuntimeError(f"cached clip unreadable: {dst}")
                else:
                    wh = normalize(resolve(args.video_root, name), dst,
                                   args.frames, args.fps, args.long_side)
                text = proc.apply_chat_template(
                    [{"role": "user", "content": [{"type": "video"},
                                                  {"type": "text", "text": row["question"]}]}],
                    add_generation_prompt=True, tokenize=False)
                ins = proc(text=[text], videos=[dst], return_tensors="pt", fps=args.fps)
                L = int(ins["input_ids"].shape[1])
                grid = ins["video_grid_thw"].tolist()
                err = None
                break
            except Exception as e:                      # logged, never swallowed
                err = f"{type(e).__name__}: {e}"
                say(f"[retry {attempts}/{args.retries}] {name}: {err}")
                say(traceback.format_exc())
                time.sleep(1.0)
        retry_hist[attempts] = retry_hist.get(attempts, 0) + 1
        if err is not None:
            failures.append({"video_name": name, "error": err, "attempts": attempts})
            say(f"[FAIL] {name} after {attempts} attempts")
            continue
        recs.append({"video_name": name, "question": row["question"],
                     "answer": row["answer"], "cache_path": dst,
                     "fps": args.fps, "n_frames": args.frames,
                     "input_ids_len": L, "grid": json.dumps(grid),
                     "width": wh[0], "height": wh[1],
                     "split": assign[name]})
        if (i + 1) % 25 == 0:
            say(f"  {i+1}/{len(it)} done, {len(failures)} failed")

    out = pd.DataFrame(recs)
    out.to_parquet(args.index, index=False)
    say(f"wrote {args.index}: {len(out)} rows ({len(failures)} failed)")
    say(f"retry histogram (attempts->count): {json.dumps(retry_hist)}")
    if len(out):
        L = out["input_ids_len"]
        say(f"input_ids_len: min {L.min()} p25 {int(L.quantile(.25))} "
            f"median {int(L.median())} p75 {int(L.quantile(.75))} max {L.max()}")
        ts = out[out.split == "train"]
        say(f"train {len(ts)} / heldout {len(out)-len(ts)}")
    if failures:
        with open(args.index + ".failures.json", "w") as fh:
            json.dump(failures, fh, indent=2)
        say(f"failure details -> {args.index}.failures.json")


if __name__ == "__main__":
    main()
