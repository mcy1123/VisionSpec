#!/usr/bin/env bash
# Phase C seed-0 three-arm run (plan section 10.6).
#
#   C0  head_video439_aligned/step1000, no training (already evaluated by A4)
#   C1  aligned continue-SFT   : historical/train_head_video.py --feature-path cached
#   C2  paper-faithful OPD     : historical/opd_train_video_fullkl.py (online chain rollout)
#
# Both training arms start from the SAME checkpoint, use the SAME 439 train rows,
# the SAME 24 anchors/update, the SAME AdamW(lr 3e-4, betas .9/.95, wd 0, clip 1.0)
# and the SAME checkpoint nodes (10/25/50). What CANNOT be matched is the meaning
# of "anchor": C1 supervises one position per anchor, C2 supervises a whole block
# (15 offsets) per anchor. That is recorded in the frozen-config report.
#
# serialised on purpose: the plan forbids concurrent heavy jobs so the wall-clock
# numbers are comparable.
set -euo pipefail

R=/mnt/local/mcy/GLANCE
REPRO=/home/mcy/Projects/GLANCE/repro
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
GPU="${GPU:-0}"
UPDATES="${UPDATES:-50}"
SAVE_EVERY="${SAVE_EVERY:-5}"
INIT="$R/checkpoints/head_video439_aligned/step1000"
mkdir -p "$R/results/phaseC" "$R/logs"

cd "$REPRO"
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_c1 () {
  local OUT="$R/checkpoints/phaseC_C1_continue_sft"
  echo "=== C1 aligned continue-SFT -> $OUT  $(date --iso-8601=seconds) ===" \
    | tee "$R/results/phaseC/C1_continue_sft.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u historical/train_head_video.py \
    --records "$R/data/video_greedy_499.pt" \
    --pool    "$R/data/pool_499.parquet" \
    --init-from "$INIT" \
    --out "$OUT" \
    --steps "$UPDATES" --save-every "$SAVE_EVERY" --log-every 5 \
    --max-anchors 24 --lr 3e-4 --seed 0 --gpu 0 \
    --feature-path cached \
    --log-file "$R/results/phaseC/C1_continue_sft.jsonl" \
    2>&1 | tee -a "$R/results/phaseC/C1_continue_sft.log"
  echo "=== C1 done $(date --iso-8601=seconds) ===" | tee -a "$R/results/phaseC/C1_continue_sft.log"
}

run_c2 () {
  local OUT="$R/checkpoints/phaseC_C2_opd_fullkl"
  echo "=== C2 paper-faithful full-KL OPD -> $OUT  $(date --iso-8601=seconds) ===" \
    | tee "$R/results/phaseC/C2_opd_fullkl.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u historical/opd_train_video_fullkl.py --rollout-mode legacy-reference \
    --records "$R/data/video_greedy_499.pt" \
    --pool    "$R/data/pool_499.parquet" \
    --init-from "$INIT" \
    --out "$OUT" \
    --updates "$UPDATES" --save-every "$SAVE_EVERY" --log-every 1 \
    --anchors-per-update 24 --anchors-per-video 4 --max-rounds 4 \
    --teacher-dtype bfloat16 --head-dtype bfloat16 \
    --gamma 0.8 --lr 3e-4 --clip 1.0 --seed 0 --gpu 0 \
    --log-file "$R/results/phaseC/C2_opd_fullkl.jsonl" \
    2>&1 | tee -a "$R/results/phaseC/C2_opd_fullkl.log"
  echo "=== C2 done $(date --iso-8601=seconds) ===" | tee -a "$R/results/phaseC/C2_opd_fullkl.log"
}

run_c1
run_c2
echo "=== PHASE C TRAINING DONE $(date --iso-8601=seconds) ==="
