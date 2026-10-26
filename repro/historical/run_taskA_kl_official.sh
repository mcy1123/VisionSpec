#!/usr/bin/env bash
# Task A (AGENTS.md 2026-10-15): the official KL objective, P1 only.
#
# Baseline entry: Draft-OPD `run_qwen_gsm8k.sh` (`d1476f5`, 2026-08-08), the
# newest KL-version DFlash entry. It sets `DISTILLATION_LOSS_MODE=k3` with
# `REJECTED_DRAFT_USE_REVERSE_KL=True`, i.e. the rejected suffix is the k3
# single-sample estimator, not a full-vocabulary reverse KL.
#
# P1 is the ONLY change in the arm: `--rejected-kl k3`. Frozen in both arms:
# start point (`head_video439_aligned/step1000`, C0), 10 updates, at most 24
# anchors/update from at most 24 anchors of one video, rollout mode `online` (the
# current closed-loop entry), fp32 target + bf16 head + fp32 CPU master weights,
# lr 3e-5 constant, AdamW betas (0.9,0.95), wd 0, clip 1.0, seed 0, gamma 0.8.
# The control arm is the SAME code with the default `--rejected-kl full_reverse`,
# so "every other number is identical" is the point of the comparison.
#
# Evaluation is fixed at update 10 (no checkpoint selection), fp32, tree b63/k8,
# video_fresh30, --ar (per-token SPEC == AR) with --warmup 1 so n=29 matches the
# phase-A4 / phase-C4 protocol exactly.
#
# GPU policy: ONE card, the emptiest one with >=39 GiB free (an A6000), waiting
# rather than starting an under-provisioned run. `nvidia-smi` cannot talk to the
# driver on this host, so `tools/pick_gpu.py` (torch runtime) chooses. Arms are serial.
set -euo pipefail
R=/mnt/local/mcy/GLANCE
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
TAG="${TAG:-$(date +%Y%m%dT%H%M%S)}"
OUT="${OUT:-$R/results/taskA_kl_official_$TAG}"
INIT="$R/checkpoints/head_video439_aligned/step1000"
MIN_FREE="${MIN_FREE:-39}"
mkdir -p "$OUT"/{logs,training,eval}

cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "$OUT" > "$OUT/run_dir.txt"

if [ "${GPU:-auto}" = "auto" ]; then
  while true; do
    if GPU=$("$PY" tools/pick_gpu.py --min-free-gib "$MIN_FREE"); then
      break
    fi
    echo "[wait] no card with ${MIN_FREE} GiB free; sleeping 300s" >&2
    sleep 300
  done
fi
echo "[gpu] using physical GPU $GPU" | tee -a "$OUT/logs/train.log"

COMMON=(--records "$R/data/video_greedy_499.pt" --pool "$R/data/pool_499.parquet"
        --init-from "$INIT" --updates 10 --save-every 5 --log-every 1
        --anchors-per-update 24 --anchors-per-video 24 --max-rounds 24
        --rollout-mode online
        --teacher-dtype float32 --head-dtype bfloat16 --master-weights cpu
        --gamma 0.8 --lr 3e-5 --clip 1.0 --seed 0 --gpu 0)

for ARM in control p1_k3; do
  if [ "$ARM" = "p1_k3" ]; then EXTRA=(--rejected-kl k3); else EXTRA=(); fi
  echo "=== TRAIN $ARM start $(date --iso-8601=seconds) ===" | tee -a "$OUT/logs/train.log"
  echo "CUDA_VISIBLE_DEVICES=$GPU $PY -u historical/opd_train_video_fullkl.py ${COMMON[*]} ${EXTRA[*]} --out $OUT/training/$ARM" >> "$OUT/commands.txt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u historical/opd_train_video_fullkl.py \
    "${COMMON[@]}" "${EXTRA[@]}" \
    --out "$OUT/training/$ARM" --log-file "$OUT/training/$ARM.jsonl" \
    > "$OUT/logs/train_$ARM.log" 2>&1
  echo "=== TRAIN $ARM done $(date --iso-8601=seconds) ===" | tee -a "$OUT/logs/train.log"
done

for ARM in control p1_k3; do
  echo "=== EVAL $ARM start $(date --iso-8601=seconds) ===" | tee -a "$OUT/logs/eval.log"
  echo "CUDA_VISIBLE_DEVICES=$GPU $PY -u evaluation/eval_video_opd.py --rows $R/data/video_fresh30.parquet --head $OUT/training/$ARM/step10 --n 0 --max-new 1024 --budget 63 --branch 8 --ar --warmup 1 --dtype float32 --gpu 0 --tag taskA_${ARM}_u10_fresh30 --out $OUT/eval/${ARM}_u10_fresh30.json" >> "$OUT/commands.txt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluation/eval_video_opd.py \
    --rows "$R/data/video_fresh30.parquet" --head "$OUT/training/$ARM/step10" \
    --n 0 --max-new 1024 --budget 63 --branch 8 --ar --warmup 1 \
    --dtype float32 --gpu 0 --tag "taskA_${ARM}_u10_fresh30" \
    --out "$OUT/eval/${ARM}_u10_fresh30.json" > "$OUT/logs/eval_$ARM.log" 2>&1
  echo "=== EVAL $ARM done $(date --iso-8601=seconds) ===" | tee -a "$OUT/logs/eval.log"
done

echo "=== TASK A ARMS DONE $(date --iso-8601=seconds) -> $OUT ===" | tee -a "$OUT/logs/eval.log"
