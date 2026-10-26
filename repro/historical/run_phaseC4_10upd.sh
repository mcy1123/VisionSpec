#!/usr/bin/env bash
# Frozen 10-update C1/C2 comparison (user spec, 2026-10-10 round 4).
#
# Both arms start from the SAME checkpoint (aligned step1000). `phaseC3/.../step10`
# is C1's RESULT and is deliberately NOT used as any initialisation.
#
# Unified between the arms:
#   * head context numerical path: C1 --target-dtype float32 == C2 --teacher-dtype
#     float32. Without this the two arms differ in their context FEATURES as well
#     as in their loss, and the comparison is not "only the loss differs".
#   * fp32 master weights (CPU offload) in BOTH, so the update-precision defect is
#     out of the comparison.
#   * lr 3e-5 constant, AdamW betas (0.9,0.95), wd 0, clip 1.0, seed 0.
#   * updates: 10.  video exposure: 1 video/update (10 each).
#   * anchors/update: 24.
#   * checkpoint nodes: 5 and 10.  Formal evaluation is FIXED at update 10.
#
# What CANNOT be matched and is therefore REPORTED: the meaning of "anchor".
# C1 supervises `super_mask` block positions per anchor with weight sum w; C2
# supervises accepted+rejected offsets of a draft block. See
# docs/reports/2026-10-10-phase-c-corrections.md for the measured counts.
set -euo pipefail
R=/mnt/local/mcy/GLANCE
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
GPU="${GPU:-0}"
INIT="$R/checkpoints/head_video439_aligned/step1000"
mkdir -p "$R/results/phaseC4"
cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== C1 (continue-SFT, fp32 target context, master weights) $(date --iso-8601=seconds) ===" \
  | tee "$R/results/phaseC4/C1.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u historical/train_head_video.py \
  --records "$R/data/video_greedy_499.pt" --pool "$R/data/pool_499.parquet" \
  --init-from "$INIT" --out "$R/checkpoints/phaseC4_C1" \
  --steps 10 --save-every 5 --log-every 1 --max-anchors 24 \
  --lr 3e-5 --seed 0 --gpu 0 --feature-path cached \
  --target-dtype float32 --master-weights cpu \
  --log-file "$R/results/phaseC4/C1.jsonl" 2>&1 | tee -a "$R/results/phaseC4/C1.log"
echo "=== C1 done $(date --iso-8601=seconds) ===" | tee -a "$R/results/phaseC4/C1.log"

echo "=== C2 (paper-faithful OPD, fp32 teacher, master weights) $(date --iso-8601=seconds) ===" \
  | tee "$R/results/phaseC4/C2.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u historical/opd_train_video_fullkl.py --rollout-mode legacy-reference \
  --records "$R/data/video_greedy_499.pt" --pool "$R/data/pool_499.parquet" \
  --init-from "$INIT" --out "$R/checkpoints/phaseC4_C2" \
  --updates 10 --save-every 5 --log-every 1 \
  --anchors-per-update 24 --anchors-per-video 24 --max-rounds 24 \
  --teacher-dtype float32 --head-dtype bfloat16 --master-weights cpu \
  --gamma 0.8 --lr 3e-5 --clip 1.0 --seed 0 --gpu 0 \
  --log-file "$R/results/phaseC4/C2.jsonl" 2>&1 | tee -a "$R/results/phaseC4/C2.log"
echo "=== C2 done $(date --iso-8601=seconds) ===" | tee -a "$R/results/phaseC4/C2.log"
