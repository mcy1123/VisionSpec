#!/usr/bin/env bash
# Step 2 of the 2026-10-10 correction round: re-calibrate the continue-SFT control
# arm AFTER fixing optimizer precision.
#
# Why re-run at all: `diagnostics/diag_update_precision.py` showed that at lr 3e-5 the bf16
# parameter + bf16 AdamW path realises only 0.223 of the intended update norm and
# leaves 96.9% of elements with a sub-half-ulp (i.e. discarded) update. The
# earlier calibration therefore measured mostly "no optimisation". With fp32
# master weights (CPU offload, `--master-weights cpu`) the fp32 update accumulates
# and the realized step matches the fp32 ideal.
#
# Only the optimizer precision changes relative to historical/run_phaseC2_calib.sh: same
# start point, same 439 rows, same --feature-path cached, same seed 0 (hence the
# same video order and the same anchor subsample), same anchor cap 96, same
# AdamW hyperparameters, same 25 updates and 5/10/25 saves. lr stays in
# {3e-5, 1e-5} -- the task says not to widen the LR search.
set -euo pipefail
R=/mnt/local/mcy/GLANCE
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
GPU="${GPU:-2}"
INIT="$R/checkpoints/head_video439_aligned/step1000"
mkdir -p "$R/results/phaseC3" "$R/logs"
cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True
for LR in 3e-5 1e-5; do
  TAG="C1mw_lr${LR}"
  OUT="$R/checkpoints/phaseC3_${TAG}"
  echo "=== $TAG lr=$LR master=cpu -> $OUT $(date --iso-8601=seconds) ===" | tee "$R/results/phaseC3/${TAG}.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u historical/train_head_video.py \
    --records "$R/data/video_greedy_499.pt" --pool "$R/data/pool_499.parquet" \
    --init-from "$INIT" --out "$OUT" \
    --steps 25 --save-every 5 --log-every 1 --max-anchors 96 \
    --lr "$LR" --seed 0 --gpu 0 --feature-path cached --master-weights cpu \
    --log-file "$R/results/phaseC3/${TAG}.jsonl" 2>&1 | tee -a "$R/results/phaseC3/${TAG}.log"
  echo "=== $TAG done $(date --iso-8601=seconds) ===" | tee -a "$R/results/phaseC3/${TAG}.log"
done
echo "=== PHASE C3 CALIB TRAINING DONE $(date --iso-8601=seconds) ==="
