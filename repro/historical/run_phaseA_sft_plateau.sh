#!/usr/bin/env bash
# Phase A of 后续计划_视频OPD_2026-10-08.md (A1): rebuild the CLEAN video-SFT
# plateau curve.
#
# Why retrain at all: `head_video439` was trained 6000 steps with save-every 500,
# but only `step1000` survived; the step500/1500 checkpoints behind the old curve
# are gone. The old curve was measured on `video_heldout_30`, whose 30 rows are
# INSIDE pool_499's train split (i.e. head_video439 trained on them), so it is a
# memorisation curve, not a generalisation curve. The plan requires the curve on
# `video_fresh30`, whose 30 rows are all in the heldout split.
#
# Recipe is byte-for-byte the one that produced `head_video439`
# (scripts/76_video_sft439.sh) except for --steps: same init head, same 439 train
# rows, same objective/reveal prior/block layout, same max-anchors/lr/seed.
# historical/train_head_video.py has no step-count-dependent schedule (fixed lr, reveals
# seeded by seed+step, permutation drawn once), so the first 1500 steps of a
# 1500-step run must reproduce the original run's step500/1000/1500 exactly.
# `step1000` is then md5-compared against the surviving `head_video439/step1000`
# as the reproduction gate.
set -euo pipefail

R=/mnt/local/mcy/GLANCE
REPRO=/home/mcy/Projects/GLANCE/repro
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
GPU="${GPU:-0}"
STEPS="${STEPS:-1500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
OUT="${OUT:-$R/checkpoints/head_video439_plateau}"

cd "$REPRO"
mkdir -p "$R/logs" "$R/results/phaseA"
echo "=== PHASE A SFT  out=$OUT steps=$STEPS save_every=$SAVE_EVERY gpu=$GPU $(date --iso-8601=seconds) ===" \
  | tee "$R/logs/phaseA_sft439.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u historical/train_head_video.py \
  --records "$R/data/video_greedy_499.pt" \
  --pool    "$R/data/pool_499.parquet" \
  --init-from "$R/checkpoints/head" \
  --out "$OUT" \
  --steps "$STEPS" --save-every "$SAVE_EVERY" --log-every 50 \
  --max-anchors 96 --lr 3e-4 --seed 0 --gpu 0 \
  --log-file "$R/logs/phaseA_sft439.jsonl" 2>&1 | tee -a "$R/logs/phaseA_sft439.log"

echo "=== PHASE A SFT done $(date --iso-8601=seconds) ===" \
  | tee -a "$R/logs/phaseA_sft439.log"
echo "--- reproduction gate: retrained step1000 vs surviving head_video439/step1000 ---"
md5sum "$OUT/step1000/model.safetensors" "$R/checkpoints/head_video439/step1000/model.safetensors" \
  | tee -a "$R/logs/phaseA_sft439.log"
