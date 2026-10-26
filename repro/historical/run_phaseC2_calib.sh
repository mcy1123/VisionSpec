#!/usr/bin/env bash
# Step 3: small-step continue-SFT calibration (2026-10-10 correction round).
#
# Goal: find a continue-SFT setting that HOLDS the aligned start point, before any
# C1/C2 comparison is re-discussed. The Phase C C1 arm degraded by -0.0798 tau,
# but that run changed four things at once relative to A4 (lr, fresh Adam, anchor
# cap 96 -> 24, and a fresh sampling order), so the cause is not identified.
#
# This run changes ONE thing at a time away from the A4 recipe:
#   * anchor cap back to 96 (A4's value) -- removes the "budget" confound and is
#     the regime step1000 was actually produced in. The audit measured
#     46848 supervised positions / weight sum 19767 over 50 such steps.
#   * lr swept over {3e-5, 1e-5}, both far below A4's 3e-4.
#   * 25 updates, checkpoints at 5/10/25.
# Everything else is held identical and identical between the two arms:
#   same init, same seed (so the same video order AND the same anchor subsample),
#   same --feature-path cached, same block/reveal prior, same AdamW/clip.
set -euo pipefail

R=/mnt/local/mcy/GLANCE
REPRO=/home/mcy/Projects/GLANCE/repro
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
GPU="${GPU:-0}"
INIT="$R/checkpoints/head_video439_aligned/step1000"
STEPS="${STEPS:-25}"
SAVE_EVERY="${SAVE_EVERY:-5}"
ANCHORS="${ANCHORS:-96}"
mkdir -p "$R/results/phaseC2" "$R/logs"

cd "$REPRO"

for LR in 3e-5 1e-5; do
  TAG="C1cal_lr${LR}"
  OUT="$R/checkpoints/phaseC2_${TAG}"
  echo "=== $TAG  lr=$LR anchors=$ANCHORS steps=$STEPS -> $OUT  $(date --iso-8601=seconds) ===" \
    | tee "$R/results/phaseC2/${TAG}.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u historical/train_head_video.py \
    --records "$R/data/video_greedy_499.pt" \
    --pool    "$R/data/pool_499.parquet" \
    --init-from "$INIT" \
    --out "$OUT" \
    --steps "$STEPS" --save-every "$SAVE_EVERY" --log-every 1 \
    --max-anchors "$ANCHORS" --lr "$LR" --seed 0 --gpu 0 \
    --feature-path cached \
    --log-file "$R/results/phaseC2/${TAG}.jsonl" \
    2>&1 | tee -a "$R/results/phaseC2/${TAG}.log"
  echo "=== $TAG done $(date --iso-8601=seconds) ===" | tee -a "$R/results/phaseC2/${TAG}.log"
done
echo "=== CALIBRATION TRAINING DONE $(date --iso-8601=seconds) ==="
