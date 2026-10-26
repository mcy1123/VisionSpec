#!/usr/bin/env bash
# Task B / video QA evaluation: the three arms of the fair redo on the SAME
# frozen dev inputs, in the SAME time mode the arms were TRAINED in.
#
# Why this is a separate step: the 2026-10-14 campaign trained seed20261014 in
# `real` timeline mode but evaluated everything in `legacy`, so the evaluated
# inputs were not the inputs the head was adapted to. Here training and eval are
# both `real`, and the same manifest is used for all three arms.
#
# `--allow-foreign` is passed deliberately: the A6000s on this host carry foreign
# processes, so the GPU resource gate would refuse. Every row is therefore marked
# `timing_contaminated` by the evaluator and NO speed claim may be made from this
# run; the primary metric (per-video tau, SPEC == AR) is unaffected.
#
# Usage: historical/run_taskB_video_eval.sh <taskB-run-dir>
set -euo pipefail
RUN="$1"
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
START=/mnt/local/mcy/GLANCE/checkpoints/head_video439_aligned/step1000
MANIFEST=/mnt/local/mcy/GLANCE/results/videoqa_transfer/20261013T134000/dataset_manifest_20videos.jsonl
OUTDIR="$RUN/video/eval/dev_real_fp32_b"
[ -d "$RUN/video/training/sft" ] || { echo "QA-SFT checkpoint missing" >&2; exit 3; }
[ -d "$RUN/video/training/opd" ] || { echo "QA-OPD checkpoint missing" >&2; exit 3; }
[ -e "$OUTDIR/per_question.jsonl" ] && { echo "$OUTDIR already has results; refusing to reuse" >&2; exit 4; }
cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True
HEADS="C0=$START,QA-SFT=$RUN/video/training/sft/step120,QA-OPD=$RUN/video/training/opd/step120"
GATE_MIB="${GATE_MIB:-38500}"

# The evaluator has its OWN free-memory gate, and it is stricter than the picker's
# GiB threshold (38500 MiB vs 38 GiB = 38912 MiB), which is exactly how the first
# attempt died at 12:06 with "gpu 0 has 35956 MiB free, need 39000": the card was
# momentarily busy. So: pick a card, run, and if the gate refuses, wait and retry
# on a freshly picked card instead of failing the whole arm.
# NOTE on GPU selection (2026-10-15, launcher bug found and fixed here):
# `eval_qa_models.py` reads the free-memory gate with `nvidia-smi`, and
# `nvidia-smi` IGNORES `CUDA_VISIBLE_DEVICES`. So passing the chosen card through
# `CUDA_VISIBLE_DEVICES` and `--gpu 0` made torch use physical card N while the
# gate checked physical card 0 -- the first attempt died with
# "GPU_RESOURCE_GATE: gpu 0 has 35956 MiB free" while the chosen card had 46 GiB.
# The evaluator is therefore invoked with the PHYSICAL index and NO
# CUDA_VISIBLE_DEVICES, so torch and nvidia-smi agree on which card is meant.
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if [ "${GPU:-auto}" = "auto" ]; then
    while ! GPUSEL=$("$PY" tools/pick_gpu.py --min-free-gib 40); do
      echo "[wait] no card with 40 GiB free; sleeping 300s" >> "$RUN/logs/video_eval.log"
      sleep 300
    done
  else
    GPUSEL="$GPU"
  fi
  echo "=== video eval attempt $attempt on GPU $GPUSEL $(date --iso-8601=seconds) ===" | tee -a "$RUN/logs/video_eval.log"
  echo "$PY -u qa_opd_fair/eval_qa_models.py --run $RUN/video --stage dev --gpu $GPUSEL --manifest $MANIFEST --heads $HEADS --time-mode real --formats B --max-videos 20 --target-dtype float32 --gpu-gate-mib $GATE_MIB --allow-foreign --append --out-subdir dev_real_fp32_b" >> "$RUN/commands.txt"
  env -u CUDA_VISIBLE_DEVICES "$PY" -u qa_opd_fair/eval_qa_models.py \
    --run "$RUN/video" --stage dev --gpu "$GPUSEL" --manifest "$MANIFEST" \
    --heads "$HEADS" --time-mode real --formats B --max-videos 20 \
    --target-dtype float32 --gpu-gate-mib "$GATE_MIB" --allow-foreign --append \
    --out-subdir dev_real_fp32_b > "$RUN/logs/video_eval_attempt$attempt.log" 2>&1
  rc=$?
  cat "$RUN/logs/video_eval_attempt$attempt.log" >> "$RUN/logs/video_eval.log"
  if [ $rc -eq 0 ]; then break; fi
  if grep -q "GPU_RESOURCE_GATE" "$RUN/logs/video_eval_attempt$attempt.log"; then
    echo "=== attempt $attempt refused by the memory gate; waiting" | tee -a "$RUN/logs/video_eval.log"
    sleep 300; continue
  fi
  echo "=== attempt $attempt failed rc=$rc (not the memory gate); stopping" | tee -a "$RUN/logs/video_eval.log"
  exit $rc
done
echo "=== video eval done $(date --iso-8601=seconds)" | tee -a "$RUN/logs/video_eval.log"
"$PY" -u qa_opd_fair/aggregate_eval.py --run "$RUN/video" --runs dev_real_fp32_b \
  --label taskB 2>&1 | tee "$RUN/logs/video_aggregate.log"
