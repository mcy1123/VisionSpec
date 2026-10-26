#!/usr/bin/env bash
# Evaluate the fp32-master-weight calibration candidates on the monitoring subset.
# Same protocol and same frozen rule as phaseC2: keep if mean paired Delta-tau vs
# C0 >= -0.02; ties -> smaller lr, then earlier update. No wait loop: the trainer
# has already finished, and a `pgrep -f <name>` wait can self-match the shell that
# created this file.
set -euo pipefail
R=/mnt/local/mcy/GLANCE
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
RES="$R/results/phaseC3"
MON="$R/data/video_fresh30_monitor8.parquet"
cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$RES"
for LR in 3e-5 1e-5; do
  for NODE in 5 10 25; do
    TAG="mw_lr${LR}_u${NODE}_monitor8"
    HEAD="$R/checkpoints/phaseC3_C1mw_lr${LR}/step${NODE}"
    [ -d "$HEAD" ] || { echo "MISSING $HEAD" >&2; continue; }
    echo "=== EVAL $TAG $(date --iso-8601=seconds) ===" | tee "$RES/${TAG}.log"
    CUDA_VISIBLE_DEVICES="${GPU:-2}" "$PY" -u evaluation/eval_video_opd.py \
      --rows "$MON" --head "$HEAD" --n 0 --max-new 1024 --budget 63 --branch 8 \
      --ar --warmup 0 --dtype float32 --gpu 0 --tag "$TAG" \
      --out "$RES/${TAG}.json" 2>&1 | tee -a "$RES/${TAG}.log"
  done
done
echo "=== PHASE C3 EVAL DONE $(date --iso-8601=seconds) ==="
