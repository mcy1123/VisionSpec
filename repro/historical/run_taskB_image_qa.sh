#!/usr/bin/env bash
# Task B / image QA fair redo (AGENTS.md 2026-10-15).
#
# What was wrong with 2026-09-23/24 and what this run fixes:
#   * every image arm checkpoint AND the OPD pool were deleted, so the comparison
#     could not be re-derived: the pool is rebuilt (see image/build_pool.sh) and
#     every arm is retrained here.
#   * the report compared the 2000-row arms with the 16000-row `head` number
#     (tau 5.199). Here the warm start IS `checkpoints/head` and it is re-evaluated
#     in THIS run with the same script/data/args, so no number is imported from a
#     different budget. Every training arm is a continuation of that same head.
#   * one seed per arm is a single observation: the SFT and OPD arms are run at a
#     fixed step count with no checkpoint selection, and the warm start is
#     deterministic, so the comparison is paired by prompt.
#
# Frozen in every arm: start `checkpoints/head`, 2000 TextVQA rows, 1000 optimizer
# steps, lr 1e-4, accum 4, micro-batch 1, AdamW betas (0.9,0.95) wd 0 clip 1.0,
# bf16 (the historical image setting), the same eval (ChartQA plain template,
# n=100 after warm-up, pooled per-round tau, budgets 15/31/47/63, branch-k 8,
# max-pixels 896, max-new 256, bf16).
#
# Usage: historical/run_taskB_image_qa.sh <taskB-run-dir> <arm>
#   arm = warm | sft | opd | opd_ce | eval_all
set -euo pipefail
RUN="$1"
ARM="${2:-warm}"
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
SRC=/mnt/local/mcy/GLANCE
START="$SRC/checkpoints/head"
DATA="$SRC/data/target_greedy_textvqa2000.jsonl"
POOL="$RUN/image/data/opd_pool_textvqa_2000.pt"
LR=1e-4
STEPS=1000
mkdir -p "$RUN/image/training" "$RUN/image/eval" "$RUN/logs"
cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True

eval_arm() {   # $1 = tag, $2 = head dir
  local tag="$1" head="$2"
  echo "=== IMAGE EVAL $tag start $(date --iso-8601=seconds) ===" | tee -a "$RUN/logs/image_eval.log"
  echo "$PY -u evaluation/eval_chartqa.py --head $head --data $SRC/data/vl_eval_plain --task chartqa --budgets 15,31,47,63 --branch-k 8 --n 100 --max-new 256 --max-pixels 896 --gpu $GPUSEL --out $RUN/image/eval/chartqa_${tag}.json" >> "$RUN/commands.txt"
  env -u CUDA_VISIBLE_DEVICES "$PY" -u evaluation/eval_chartqa.py \
    --head "$head" --data "$SRC/data/vl_eval_plain" --task chartqa \
    --budgets 15,31,47,63 --branch-k 8 --n 100 --max-new 256 --max-pixels 896 \
    --gpu "$GPUSEL" --out "$RUN/image/eval/chartqa_${tag}.json" \
    > "$RUN/logs/image_eval_${tag}.log" 2>&1
  echo "=== IMAGE EVAL $tag done $(date --iso-8601=seconds) ===" | tee -a "$RUN/logs/image_eval.log"
}

pick() {
  local want="${MIN_FREE:-36}"
  if [ "${GPU:-auto}" != "auto" ]; then
    # A pinned card is a PREFERENCE, not a guarantee: this host is multi-tenant
    # and a foreign process can take the card between two arms of this pipeline
    # (measured: GPU 1 went from 48 GiB free at 12:54 to 28 GiB free minutes
    # later). If the pinned card no longer has room, fall back to the emptiest
    # card that does rather than starting a run that will OOM at step 6.
    if "$PY" tools/pick_gpu.py --min-free-gib "$want" --only "$GPU" >/dev/null 2>&1; then
      GPUSEL="$GPU"
    else
      echo "[gpu] pinned GPU $GPU lacks ${want} GiB; falling back to the emptiest card" \
        | tee -a "$RUN/logs/image.log"
      while ! GPUSEL=$("$PY" tools/pick_gpu.py --min-free-gib "$want"); do
        echo "[wait] no card with ${want} GiB free; sleeping 300s" >&2
        sleep 300
      done
    fi
  else
    while ! GPUSEL=$("$PY" tools/pick_gpu.py --min-free-gib "$want"); do
      echo "[wait] no card with ${want} GiB free; sleeping 300s" >&2
      sleep 300
    done
  fi
  echo "[gpu] using physical GPU $GPUSEL" | tee -a "$RUN/logs/image.log"
}

case "$ARM" in
  warm)
    pick
    [ -f "$RUN/image/eval/chartqa_warm_start.json" ] || eval_arm warm_start "$START"
    ;;
  sft)
    pick
    OUT="$RUN/image/training/sft_control"
    echo "=== IMAGE SFT start $(date --iso-8601=seconds) ===" | tee -a "$RUN/logs/image.log"
    echo "CUDA_VISIBLE_DEVICES=$GPUSEL $PY -u historical/train_head.py --data $DATA --init-from $START --out $OUT --gpus 0 --lr $LR --accum 4 --epochs 3 --steps $STEPS --micro-batch 1 --seed 0 --save-every $STEPS --log-file $RUN/image/training/sft_control.jsonl" >> "$RUN/commands.txt"
    CUDA_VISIBLE_DEVICES="$GPUSEL" "$PY" -u historical/train_head.py \
      --data "$DATA" --init-from "$START" --out "$OUT" --gpus 0 \
      --lr $LR --accum 4 --epochs 3 --steps $STEPS --micro-batch 1 --seed 0 \
      --save-every $STEPS --log-file "$RUN/image/training/sft_control.jsonl" \
      > "$RUN/logs/image_sft.log" 2>&1
    echo "=== IMAGE SFT done $(date --iso-8601=seconds) ===" | tee -a "$RUN/logs/image.log"
    eval_arm sft_control "$OUT"
    ;;
  opd|opd_ce)
    pick
    [ -f "$POOL" ] || { echo "pool $POOL missing; run image/build_pool.sh first" >&2; exit 5; }
    EXTRA=()
    [ "$ARM" = "opd_ce" ] && EXTRA=(--objective ce --reject-mode suffix --decay 0.8 --beta-fwd 0.5 --alpha-rev 0.5 --cache-ctx)
    OUT="$RUN/image/training/$ARM"
    echo "=== IMAGE $ARM start $(date --iso-8601=seconds) ===" | tee -a "$RUN/logs/image.log"
    echo "CUDA_VISIBLE_DEVICES=$GPUSEL $PY -u historical/opd_train.py --pool $POOL --data $DATA --init-from $START --out $OUT --lr $LR --accum 4 --steps $STEPS --max-anchors 8 --alpha-rev 0.1 --seed 0 --save-every $STEPS --log-file $RUN/image/training/$ARM.jsonl --gpu 0 ${EXTRA[*]}" >> "$RUN/commands.txt"
    CUDA_VISIBLE_DEVICES="$GPUSEL" "$PY" -u historical/opd_train.py \
      --pool "$POOL" --data "$DATA" --init-from "$START" --out "$OUT" \
      --lr $LR --accum 4 --steps $STEPS --max-anchors 8 --alpha-rev 0.1 --seed 0 \
      --save-every $STEPS --log-file "$RUN/image/training/$ARM.jsonl" --gpu 0 \
      "${EXTRA[@]}" > "$RUN/logs/image_$ARM.log" 2>&1
    echo "=== IMAGE $ARM done $(date --iso-8601=seconds) ===" | tee -a "$RUN/logs/image.log"
    eval_arm "$ARM" "$OUT"
    ;;
  eval_all)
    pick
    for pair in "warm_start=$START" "sft_control=$RUN/image/training/sft_control" \
                "opd=$RUN/image/training/opd" "opd_ce=$RUN/image/training/opd_ce"; do
      tag="${pair%%=*}"; head="${pair#*=}"
      [ -d "$head" ] || { echo "[skip] $tag: $head missing"; continue; }
      [ -f "$RUN/image/eval/chartqa_${tag}.json" ] && { echo "[skip] $tag already evaluated"; continue; }
      eval_arm "$tag" "$head"
    done
    ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
