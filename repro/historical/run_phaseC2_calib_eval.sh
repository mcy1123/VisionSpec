#!/usr/bin/env bash
# Evaluate the 6 pre-registered calibration candidates on the monitoring subset.
# Rule frozen in docs/reports/2026-10-10-phase-c2-prereg.md BEFORE any run:
#   keep if mean paired Delta-tau (candidate - C0) >= -0.02 on the 8-video subset.
# Protocol identical to Phase C: fp32, tree b63/k8, max_new 1024, to EOS, --ar,
# --warmup 0 so all 8 rows enter (C0's subset numbers come from the A4 eval of the
# same checkpoint, whose dropped warmup row is NOT in the subset).
set -euo pipefail
R=/mnt/local/mcy/GLANCE
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
RES="$R/results/phaseC2"
MON="$R/data/video_fresh30_monitor8.parquet"
cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$RES"

for LR in 3e-5 1e-5; do
  for NODE in 5 10 25; do
    TAG="cal_lr${LR}_u${NODE}_monitor8"
    HEAD="$R/checkpoints/phaseC2_C1cal_lr${LR}/step${NODE}"
    [ -d "$HEAD" ] || { echo "MISSING $HEAD" >&2; exit 2; }
    echo "=== EVAL $TAG $(date --iso-8601=seconds) ===" | tee "$RES/${TAG}.log"
    CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PY" -u evaluation/eval_video_opd.py \
      --rows "$MON" --head "$HEAD" --n 0 --max-new 1024 --budget 63 --branch 8 \
      --ar --warmup 0 --dtype float32 --gpu 0 --tag "$TAG" \
      --out "$RES/${TAG}.json" 2>&1 | tee -a "$RES/${TAG}.log"
  done
done
echo "=== CALIB EVAL DONE $(date --iso-8601=seconds) ==="
