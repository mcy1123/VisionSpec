#!/usr/bin/env bash
# Task B / video QA fair redo (AGENTS.md 2026-10-15).
#
# What was wrong with 2026-10-14 and is fixed here:
#   * seed20261013 ran QA-SFT 120 updates but QA-OPD only 60 (the step-60 save died
#     on a full disk and was resumed 30 -> 60), so that seed is not a paired
#     replication. This run uses ONE seed and gives BOTH arms all 120 updates.
#   * seed20261014 trained in `real` timeline mode but every evaluation in the
#     campaign ran `legacy`, so the training inputs and the evaluated inputs were
#     not the same encoding. This run uses `real` in BOTH training and eval.
#   * seed20261014 trained both arms in bf16 (a GPU-contention fallback). This run
#     is fp32 for both arms, so the SPEC == AR gate can actually be claimed.
#
# Frozen: the 2026-10-14 seed-20261014 schedule (480 QA slots, sha256 checked
# against the frozen copy before anything runs), 4 QA/update, 120 updates,
# lr 3e-5 constant, AdamW betas (0.9,0.95) wd 0, clip 1.0, fp32 CPU master
# weights, eos_stop "immediate" (hardcoded by both trainers), 6 anchors/QA.
# The only difference between the arms is the training signal; the same
# `--run` schedule file is read by both trainers.
#
# Usage: historical/run_taskB_video_qa.sh <taskB-run-dir> [sft|opd|both]
set -euo pipefail
RUN="$1"
WHICH="${2:-both}"
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
SRC=/mnt/local/mcy/GLANCE/results/qa_opd_fair_20261013_195500/dataset
START=/mnt/local/mcy/GLANCE/checkpoints/head_video439_aligned/step1000
SCHED_SHA=eeafe78eb6352449e90c9459812ee7fd9fcc85fd9e58fafaa2cec582ca73fd63
SEED=20261014
MIN_FREE="${MIN_FREE:-39}"

mkdir -p "$RUN/video/dataset" "$RUN/video/training"
cp -n "$SRC/train_schedule_seed_${SEED}.jsonl" "$RUN/video/dataset/" 2>/dev/null || true
got=$(sha256sum "$RUN/video/dataset/train_schedule_seed_${SEED}.jsonl" | cut -d' ' -f1)
[ "$got" = "$SCHED_SHA" ] || { echo "FATAL: schedule sha $got != frozen $SCHED_SHA" >&2; exit 4; }
n_slots=$(wc -l < "$RUN/video/dataset/train_schedule_seed_${SEED}.jsonl")
echo "[schedule] seed $SEED: $n_slots slots, sha256 $got (frozen copy verified)"

cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ "${GPU:-auto}" = "auto" ]; then
  while true; do
    if GPU=$("$PY" tools/pick_gpu.py --min-free-gib "$MIN_FREE"); then break; fi
    echo "[wait] no card with ${MIN_FREE} GiB free; sleeping 300s" >&2
    sleep 300
  done
fi
echo "[gpu] using physical GPU $GPU" | tee -a "$RUN/logs/train_video.log"

for ARM in sft opd; do
  if [ "$WHICH" != "both" ] && [ "$WHICH" != "$ARM" ]; then continue; fi
  SCRIPT="train_qa_${ARM}.py"
  OUT="$RUN/video/training/$ARM"
  echo "=== VIDEO QA-${ARM} start $(date --iso-8601=seconds) ===" | tee -a "$RUN/logs/train_video.log"
  echo "CUDA_VISIBLE_DEVICES=$GPU $PY -u repro/qa_opd_fair/$SCRIPT --run $RUN/video --seed $SEED --gpu 0 --pool $SRC/qa_pool_df${SEED}_real.parquet --records $SRC/target_ar_df${SEED}_real.pt --init-from $START --out $OUT --updates 120 --save-every 120 --log-every 1 --target-dtype float32 --time-mode real --master-weights cpu" >> "$RUN/commands.txt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "qa_opd_fair/$SCRIPT" \
    --run "$RUN/video" --seed "$SEED" --gpu 0 \
    --pool "$SRC/qa_pool_df${SEED}_real.parquet" \
    --records "$SRC/target_ar_df${SEED}_real.pt" \
    --init-from "$START" --out "$OUT" \
    --updates 120 --save-every 120 --log-every 1 \
    --target-dtype float32 --time-mode real --master-weights cpu \
    > "$RUN/logs/train_video_$ARM.log" 2>&1
  echo "=== VIDEO QA-${ARM} done $(date --iso-8601=seconds) ===" | tee -a "$RUN/logs/train_video.log"
done
