#!/usr/bin/env bash
# Phase A of 后续计划_视频OPD_2026-10-08.md (A2/A3): score the rebuilt clean SFT
# curve on the genuinely independent `video_fresh30`.
#
# Protocol is deliberately the SAME niche as the 2x2 arms in
# VIDEO_OPD_2X2_REPORT_2026-10-08.md so the plateau curve and yesterday's arms are
# directly comparable: 30 rows, first row dropped as warmup (n=29), fp32,
# GLANCE tree b63/k8, max_new 1024, greedy, run to EOS, `--ar` on so every
# checkpoint gets the fp32 speculative == AR token-for-token lossless gate that
# A2 makes a validity condition.
#
# `head_video439/step1000` is the warm start the plan's Phase C will branch from;
# the retrained `step1000` is md5-gated against it in historical/run_phaseA_sft_plateau.sh,
# so its score here doubles as the protocol-matched warm-start reference.
set -euo pipefail

R=/mnt/local/mcy/GLANCE
REPRO=/home/mcy/Projects/GLANCE/repro
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
GPU="${GPU:-0}"
OUT="${OUT:-$R/checkpoints/head_video439_plateau}"
RES="${RES:-$R/results/phaseA}"
STEPS="${STEPS:-500 1000 1500}"

cd "$REPRO"
mkdir -p "$RES"

run_one () {  # $1 = result tag, $2 = head dir
  local NAME="$1" JAIL="$2"
  [ -d "$JAIL" ] || { echo "MISSING head: $JAIL" >&2; exit 2; }
  echo "=== EVAL $NAME  head=$JAIL  $(date --iso-8601=seconds) ===" \
    | tee "$RES/${NAME}_fresh30_tree_float32.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluation/eval_video_opd.py \
    --rows "$R/data/video_fresh30.parquet" \
    --head "$JAIL" \
    --n 0 --max-new 1024 --budget 63 --branch 8 \
    --ar --warmup 1 --dtype float32 --gpu 0 --tag "$NAME" \
    --out "$RES/${NAME}_fresh30_tree_float32.json" \
    2>&1 | tee -a "$RES/${NAME}_fresh30_tree_float32.log"
}

for STEP in $STEPS; do
  run_one "sft439_step${STEP}" "$OUT/step${STEP}"
done

# Protocol-matched warm-start reference. The retrained lineage reproduces the
# original run only up to float-reduction noise (GPU/kernel nondeterminism), so
# score the surviving `head_video439/step1000` -- the exact checkpoint Phase C
# branches from -- under this same protocol instead of borrowing the older
# 2-shard (n=28) fp32 audit number.
if [ "${WITH_WARM_REF:-1}" = "1" ]; then
  run_one "warmstart_v439_step1000" "$R/checkpoints/head_video439/step1000"
fi
echo "=== PHASE A EVAL done $(date --iso-8601=seconds) ==="
