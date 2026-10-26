#!/usr/bin/env python3
"""Build the three harder MLVU task pools (plan -> sharded normalize -> merge).

Why the pools are built this way (measured facts, not preferences):

  * The local MLVU release is `/mnt/local2/mcy/datasets/MLVU/mlvu_test/
    test-*.parquet` (502 rows over the 396 mp4s in `MLVU/videos/`). Its video
    names match the local files exactly, and **all 502 rows are multiple-choice
    with one-word answers**.
  * The free-form MLVU annotations (`MLVU/json/9_summary.json`,
    `8_sub_scene.json`) reference videos that are NOT on this machine
    (`in_local = 0/216` and `0/201`, checked), so they cannot be used.
  * The task therefore has to come from the real MLVU questions rendered as
    FREE-FORM tasks: keeping the question (and its options, since the options
    are part of the question text) and appending a reasoning instruction. That
    is what makes the answer long enough for tau to be meaningful -- the task
    explicitly excludes the short-answer MCQ form for exactly that reason.

Three disjoint tasks by MLVU task_type, each targeting ~180-195 rows (the total
row count is 502, so 200-400 rows PER TASK is impossible; variant counts are
chosen per task to equalise the row budget instead):

    temporal   (时序推理)   order + tutorialQA                97 q x 2 variants = 194
    videoqa    (视频问答)   needleQA + ego + sportsQA          96 q x 2 variants = 192
    multistep  (多步推理)   topic_reasoning + plotQA + anomaly_reco + count
                                                             181 q x 1 variant  = 181

The split is a GLOBAL video-level 70/30 cut with seed 0 over the union of used
videos, so a held-out row's video is never trained by any arm of any task (and
never by the multi-task mix).

Stages:
    --stage plan                      write the plan parquet (no video I/O)
    --stage normalize --shards 6 --shard i
    --stage merge
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
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(_REPRO_ROOT))
from data.predecode_cache import normalize  # the verified cv2 fallback lives there

ROOT = "/mnt/local2/mcy/GLANCE_v2"
MLVU = "/mnt/local2/mcy/datasets/MLVU"
SRC_PARQUET = f"{MLVU}/mlvu_test/test-00000-of-00001.parquet"
VIDEO_ROOT = f"{MLVU}/videos"

TASKS = {
    "temporal": {
        "types": ["order", "tutorialQA"],
        "variants": [
            "{q}\n\nList the steps of what happens in the video in the correct "
            "order, and explain how you determined the order.",
            "{q}\n\nDescribe, step by step, the sequence of actions in this video, "
            "then state the correct order and why.",
        ],
    },
    "videoqa": {
        "types": ["needleQA", "ego", "sportsQA"],
        "variants": [
            "{q}\n\nAnswer in three or four complete sentences, describing what you "
            "see in the video and how it supports your answer.",
            "{q}\n\nExplain in a short paragraph what happens in the video that "
            "answers this question.",
        ],
    },
    "multistep": {
        "types": ["topic_reasoning", "plotQA", "anomaly_reco", "count"],
        "variants": [
            "{q}\n\nThink step by step: describe the evidence you see in the video, "
            "reason about it, then give your final answer.",
        ],
    },
}


def plan(out_plan, seed=0, train_frac=0.70):
    df = pd.read_parquet(SRC_PARQUET)
    local = {os.path.basename(p) for p in glob.glob(f"{VIDEO_ROOT}/*.mp4")}
    df = df[df.video_name.isin(local)].copy()
    # one question per (task_type, video): keeps rows and videos 1:1 inside a task
    df = df.sort_values("question_id").drop_duplicates(["task_type", "video_name"])
    print(f"source rows usable: {len(df)} over {df.video_name.nunique()} videos "
          f"({df.task_type.nunique()} task types)")

    rows = []
    for task, spec in TASKS.items():
        sub = df[df.task_type.isin(spec["types"])]
        assert len(sub) > 0, task
        for _, r in sub.iterrows():
            for vi, tmpl in enumerate(spec["variants"]):
                rows.append({
                    "task": f"{task}__{r.task_type}__{r.video_name[:-4]}__v{vi}",
                    "video_name": r.video_name,
                    "task_name": task,
                    "src_task_type": r.task_type,
                    "question": tmpl.format(q=str(r["question"]).strip()),
                    "base_question": str(r["question"]).strip(),
                    "prompt_kind": vi,
                    "answer_key": str(r["answer"]),
                    "duration": float(r["duration"]),
                })
    plan_df = pd.DataFrame(rows)
    assert plan_df["task"].is_unique, "task keys are not unique"

    vids = sorted(plan_df.video_name.unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vids))
    n_tr = int(round(len(vids) * train_frac))
    tr = {vids[i] for i in perm[:n_tr]}
    plan_df["split"] = np.where(plan_df.video_name.isin(tr), "train", "heldout")
    os.makedirs(os.path.dirname(out_plan), exist_ok=True)
    plan_df.to_parquet(out_plan, index=False)

    print(f"\nvideos: {len(vids)} = {n_tr} train + {len(vids)-n_tr} heldout "
          f"(video-level, seed {seed}, train_frac {train_frac})")
    print(f"{'task':<10} {'rows':>5} {'train':>6} {'heldout':>8} {'videos':>7} "
          f"{'p_kinds':>8}")
    for task in TASKS:
        g = plan_df[plan_df.task_name == task]
        print(f"{task:<10} {len(g):>5} {(g.split=='train').sum():>6} "
              f"{(g.split=='heldout').sum():>8} {g.video_name.nunique():>7} "
              f"{g.prompt_kind.nunique():>8}")
    # the exact row budget is whatever the deduplicated MLVU rows give; the
    # per-task counts are printed above and carried into the report
    assert len(plan_df) > 0
    # held-out videos must not appear in ANY task's training rows
    tr_v = set(plan_df[plan_df.split == "train"].video_name)
    ho_v = set(plan_df[plan_df.split == "heldout"].video_name)
    assert not (tr_v & ho_v), "video-level split leaked"
    print(f"plan written: {out_plan} ({len(plan_df)} rows)")
    for task in TASKS:
        p = f"{ROOT}/data/pool_{task}.parquet"
        plan_df[plan_df.task_name == task].to_parquet(p, index=False)
        print(f"  wrote {p}: {(plan_df.task_name==task).sum()} rows")


def normalize_stage(args):
    plan_df = pd.read_parquet(args.plan)
    vids = sorted(plan_df.video_name.unique())
    mine = vids[args.shard::args.shards]
    cache = args.cache_dir
    os.makedirs(cache, exist_ok=True)
    print(f"shard {args.shard}/{args.shards}: {len(mine)} videos -> {cache}", flush=True)
    out = []
    for i, v in enumerate(mine):
        dst = os.path.join(cache, v)
        if os.path.exists(dst) and args.skip_existing:
            w, h = 0, 0
        else:
            src = os.path.join(VIDEO_ROOT, v)
            assert os.path.exists(src), src
            w, h = normalize(src, dst, args.frames, args.fps, args.long_side)
        rows = plan_df[plan_df.video_name == v].copy()
        rows["cache_path"] = dst
        rows["fps"] = args.fps
        rows["n_frames"] = args.frames
        rows["width"] = w
        rows["height"] = h
        out.append(rows)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(mine)} videos done", flush=True)
    part = pd.concat(out, ignore_index=True)
    p = f"{ROOT}/data/pools_shard{args.shard}.parquet"
    part.to_parquet(p, index=False)
    print(f"wrote {p}: {len(part)} rows over {part.video_name.nunique()} videos "
          f"({(part.split=='train').sum()} train / {(part.split=='heldout').sum()} heldout)",
          flush=True)


def merge_stage(args):
    parts = sorted(glob.glob(f"{ROOT}/data/pools_shard*.parquet"))
    assert parts, "no shard parquets"
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    assert df["task"].is_unique, "duplicate task keys after merge"
    full = pd.read_parquet(args.plan)
    assert len(df) == len(full), f"merged {len(df)} != plan {len(full)}"
    missing = set(df.video_name) ^ set(full.video_name)
    assert not missing, f"video set drifted: {sorted(missing)[:5]}"
    for task in TASKS:
        p = f"{ROOT}/data/pool_{task}.parquet"
        sub = df[df.task_name == task].copy()
        sub.to_parquet(p, index=False)
        print(f"{p}: {len(sub)} rows = {(sub.split=='train').sum()} train + "
              f"{(sub.split=='heldout').sum()} heldout, {sub.video_name.nunique()} videos")
    # multi-task mix: up to 200 rows per task, train rows only for the mix train
    mix = []
    for task in TASKS:
        sub = df[(df.task_name == task) & (df.split == "train")].head(200)
        mix.append(sub)
        print(f"  mix takes {len(sub)} train rows from {task}")
    mix_df = pd.concat(mix, ignore_index=True)
    mix_df.to_parquet(f"{ROOT}/data/pool_mix.parquet", index=False)
    print(f"{ROOT}/data/pool_mix.parquet: {len(mix_df)} train rows "
          f"({mix_df.video_name.nunique()} videos, {mix_df.task_name.nunique()} tasks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["plan", "normalize", "merge"])
    ap.add_argument("--plan", default=f"{ROOT}/data/mlvu_task_plan.parquet")
    ap.add_argument("--cache-dir", default=f"{ROOT}/video_cache_mlvu")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--long-side", type=int, default=384)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    if a.stage == "plan":
        plan(a.plan)
    elif a.stage == "normalize":
        normalize_stage(a)
    else:
        merge_stage(a)


if __name__ == "__main__":
    main()
