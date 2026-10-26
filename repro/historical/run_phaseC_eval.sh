#!/usr/bin/env bash
# Phase C evaluation (plan section 10.6).
#
# Monitoring subset (the 8 fresh30 videos frozen in subsets.json, score-free):
#   C1/C2 at update 10 and 25, scored with --warmup 0 so all 8 rows enter the
#   aggregate. C0's subset numbers come from the existing A4 eval of the same
#   checkpoint (row 0 of fresh30 is NOT in the subset, so nothing is dropped and
#   the two protocols agree on the same 8 videos).
#
# Full validation at update 50: all 30 fresh30 rows with --warmup 1, matching the
# A4 protocol exactly (n=29, row 0 dropped). C0 therefore needs no re-eval.
#
# Every run is fp32, GLANCE tree b63/k8, max_new 1024, greedy to EOS, --ar on for
# the token-for-token lossless gate.
set -euo pipefail

R=/mnt/local/mcy/GLANCE
REPRO=/home/mcy/Projects/GLANCE/repro
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
GPU="${GPU:-0}"
RES="$R/results/phaseC"
MON="$R/data/video_fresh30_monitor8.parquet"
FULL="$R/data/video_fresh30.parquet"
C1="$R/checkpoints/phaseC_C1_continue_sft"
C2="$R/checkpoints/phaseC_C2_opd_fullkl"

cd "$REPRO"
mkdir -p "$RES"
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_one () {  # tag, head dir, rows parquet, warmup
  local TAG="$1" HEAD="$2" ROWS="$3" WARM="$4"
  [ -d "$HEAD" ] || { echo "MISSING head $HEAD" >&2; exit 2; }
  echo "=== EVAL $TAG  head=$HEAD rows=$ROWS warmup=$WARM  $(date --iso-8601=seconds) ===" \
    | tee "$RES/${TAG}.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluation/eval_video_opd.py \
    --rows "$ROWS" --head "$HEAD" --n 0 --max-new 1024 --budget 63 --branch 8 \
    --ar --warmup "$WARM" --dtype float32 --gpu 0 --tag "$TAG" \
    --out "$RES/${TAG}.json" 2>&1 | tee -a "$RES/${TAG}.log"
}

# --- monitoring nodes (8 videos, all rows kept) ---
run_one "C1_update10_monitor8" "$C1/step10" "$MON" 0
run_one "C2_update10_monitor8" "$C2/step10" "$MON" 0
run_one "C1_update25_monitor8" "$C1/step25" "$MON" 0
run_one "C2_update25_monitor8" "$C2/step25" "$MON" 0

# --- full validation at update 50 (30 rows, row 0 warmup, n=29) ---
run_one "C1_update50_fresh30" "$C1/step50" "$FULL" 1
run_one "C2_update50_fresh30" "$C2/step50" "$FULL" 1

echo "=== PHASE C EVAL DONE $(date --iso-8601=seconds) ==="
