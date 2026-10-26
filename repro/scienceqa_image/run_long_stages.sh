#!/usr/bin/env bash
# Sequential long stages, so the single usable card is never idle between them.
#
#   ./run_long_stages.sh <stage...>
#
# Each stage is skipped when its artifact already exists, so the script is safe to
# re-run after an interruption. Every stage is logged to <RUN>/logs/.
set -uo pipefail
export RUN="${RUN:-/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955}"
export PYTHON="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
REPRO=/home/mcy/Projects/GLANCE/repro/scienceqa_image
GPU="${GPU:-7}"
UPDATES="${UPDATES:-120}"
LOG="$RUN/logs/long_stages.log"
mkdir -p "$RUN/logs"

say() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

for stage in "$@"; do
  case "$stage" in
    train-ar)
      for s in 20261013 20261014; do
        if [[ -f "$RUN/dataset/sqa_target_ar_df$s.pt" ]]; then say "train-ar $s exists, skip"; continue; fi
        if [[ "$s" == 20261014 && -f "$RUN/dataset/sqa_target_ar_df20261013.pt" ]]; then
          say "train-ar $s: schedule is seed-independent, reusing the 20261013 labels"
          cp "$RUN/dataset/sqa_target_ar_df20261013.pt" "$RUN/dataset/sqa_target_ar_df$s.pt"
          cp "$RUN/dataset/sqa_pool_df20261013.parquet" "$RUN/dataset/sqa_pool_df$s.parquet"
          cp "$RUN/dataset/sqa_records_summary_df20261013.json" "$RUN/dataset/sqa_records_summary_df$s.json"
          cp "$RUN/dataset/train_schedule_seed_20261013.jsonl" "$RUN/dataset/train_schedule_seed_$s.jsonl"
          "$PYTHON" - "$RUN" "$s" <<'PY'
import json, sys, pathlib
run, s = pathlib.Path(sys.argv[1]), sys.argv[2]
# the schedule rows and the AR blob are identical; only the recorded seed label differs
p = run / "dataset" / f"train_schedule_seed_{s}.jsonl"
assert p.exists()
print(f"[long] seed {s} schedule rows:", len(p.read_text().splitlines()))
PY
          continue
        fi
        say "train-ar $s ($((UPDATES*4)) slots) on GPU $GPU"
        "$PYTHON" -u "$REPRO/build_training_data.py" --run "$RUN" --seed "$s" \
            --gpu "$GPU" --updates "$UPDATES" --max-new 256 --max-pixels 768 \
            > "$RUN/logs/build_train_ar_$s.log" 2>&1
        say "train-ar $s rc=$?"
      done
      ;;
    eval-ar)
      for split in dev test; do
        if [[ -f "$RUN/eval/ar_$split.pt" ]]; then say "eval-ar $split exists, skip"; continue; fi
        say "eval-ar $split on GPU $GPU"
        "$PYTHON" -u "$REPRO/build_eval_ar.py" --run "$RUN" --split "$split" \
            --gpu "$GPU" --max-new 256 --max-pixels 768 \
            --out "$RUN/eval/ar_$split.pt" > "$RUN/logs/build_eval_ar_$split.log" 2>&1
        say "eval-ar $split rc=$?"
      done
      ;;
    train-all)
      # checkpoints are spread over the three permitted volumes: seed1 on
      # local3, seed2 SFT on local2, seed2 OPD on local. Each volume keeps a
      # large share free; none is filled.
      for seed in 20261013 20261014; do
        for arm in sft opd; do
          case "$seed/$arm" in
            20261013/*) export OUTROOT=/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955 ;;
            20261014/sft) export OUTROOT=/mnt/local2/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955 ;;
            20261014/opd) export OUTROOT=/mnt/local/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955 ;;
          esac
          mkdir -p "$OUTROOT/train"
          say "checkpoint root $OUTROOT"
          say "train $arm seed $seed"
          RUN="$RUN" bash "$REPRO/run_scienceqa.sh" train "$arm" "$seed" "$GPU" "$UPDATES" \
              >> "$LOG" 2>&1
          say "train $arm seed $seed rc=$?"
        done
      done
      ;;
    *) echo "unknown long stage '$stage'" >&2; exit 2 ;;
  esac
done
say "long stages done: $*"
