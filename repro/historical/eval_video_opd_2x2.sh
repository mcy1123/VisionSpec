#!/usr/bin/env bash
set -euo pipefail

GPU_PHYSICAL="${GPU_PHYSICAL:-7}"
ROOT="${GLANCE_ROOT:-/mnt/local/mcy/GLANCE}"
PYTHON="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
STAMP="${STAMP:-s200_seed0_r32_a24_v2}"
N="${N:-0}"
DTYPE="${DTYPE:-float32}"

mkdir -p "${ROOT}/results/opd_video_2x2"
for MODE in frozen online; do
  for VERIFY in chain tree; do
    NAME="${MODE}_${VERIFY}_${STAMP}"
    CUDA_VISIBLE_DEVICES="${GPU_PHYSICAL}" GLANCE_ROOT="${ROOT}" "${PYTHON}" \
      repro/evaluation/eval_video_opd.py \
      --rows "${ROOT}/data/video_fresh30.parquet" \
      --head "${ROOT}/checkpoints/opd_video_2x2/${NAME}" \
      --n "${N}" --max-new 1024 --budget 63 --branch 8 \
      --warmup 1 --dtype "${DTYPE}" --gpu 0 --tag "${NAME}" \
      --out "${ROOT}/results/opd_video_2x2/${NAME}_fresh30_tree_${DTYPE}.json" \
      2>&1 | tee "${ROOT}/results/opd_video_2x2/${NAME}_fresh30_tree_${DTYPE}.log"
  done
done
