#!/usr/bin/env bash
set -euo pipefail

# Matched factorial diagnostic for the Draft-OPD/GLANCE discrepancy.
# Only rollout freshness (frozen vs online) and verifier topology
# (chain vs tree) change between arms.

GPU_PHYSICAL="${GPU_PHYSICAL:-7}"
STEPS="${STEPS:-200}"
SEED="${SEED:-0}"
MAX_ROUNDS="${MAX_ROUNDS:-32}"
MAX_ANCHORS="${MAX_ANCHORS:-24}"
TAG="${TAG:-v2}"
ROOT="${GLANCE_ROOT:-/mnt/local/mcy/GLANCE}"
PYTHON="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
STAMP="s${STEPS}_seed${SEED}_r${MAX_ROUNDS}_a${MAX_ANCHORS}_${TAG}"

mkdir -p "${ROOT}/checkpoints/opd_video_2x2" "${ROOT}/results/opd_video_2x2"

for MODE in frozen online; do
  for VERIFY in chain tree; do
    NAME="${MODE}_${VERIFY}_${STAMP}"
    OUT="${ROOT}/checkpoints/opd_video_2x2/${NAME}"
    LOG="${ROOT}/results/opd_video_2x2/${NAME}.log"
    JSONL="${ROOT}/results/opd_video_2x2/${NAME}.jsonl"
    echo "[$(date --iso-8601=seconds)] START ${NAME}" | tee "${LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL}" GLANCE_ROOT="${ROOT}" "${PYTHON}" \
      repro/historical/opd_train_video.py \
      --records "${ROOT}/data/video_opd_pool_warm_499.pt" \
      --pool "${ROOT}/data/pool_499.parquet" \
      --holdout-parquet "${ROOT}/data/video_heldout_30.parquet" \
      --init-from "${ROOT}/checkpoints/head_video439/step1000" \
      --out "${OUT}" \
      --steps "${STEPS}" --save-every 0 --log-every 10 \
      --max-anchors "${MAX_ANCHORS}" --max-rounds "${MAX_ROUNDS}" \
      --rollout-mode "${MODE}" --verify "${VERIFY}" \
      --tree-budget 63 --branch-k 8 \
      --reject-target topk --alpha-rev 0 --beta-fwd 1 --decay 0.9 \
      --lr 3e-4 --seed "${SEED}" --aligned-forward --deterministic \
      --gpu 0 --log-file "${JSONL}" 2>&1 | tee -a "${LOG}"
    echo "[$(date --iso-8601=seconds)] DONE ${NAME}" | tee -a "${LOG}"
  done
done
