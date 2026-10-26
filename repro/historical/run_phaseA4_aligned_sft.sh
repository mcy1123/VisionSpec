#!/usr/bin/env bash
# Phase A4 (2026-10-09 decision): rebuild video SFT on the decode-aligned feature
# path and re-find the peak.
#
# Why this exists. Phase A's curve (3.5389 / 3.5785 / 3.5246 at step500/1000/1500)
# was trained through `common.target_context` -- `use_cache=False` with no explicit
# mask. `diagnostics/check_context_interface.py` measures that path's TEXT-position hidden
# states to be bit-identical under a blanked video (max|delta| exactly 0.000e+00
# on 3 rows), while the decode-aligned path moves by 2.4e2..9.8e3. So every video
# head so far was trained on text rows that carry no video information and
# evaluated on text rows that do. Those checkpoints cannot be the strong SFT
# start for paper-faithful OPD.
#
# A4 changes EXACTLY ONE thing relative to Phase A -- `--feature-path cached` --
# and nothing else: same image warm head `checkpoints/head`, same 439 train rows,
# same objective/reveal prior/block layout, same max-anchors/lr/seed/steps.
# `target_context_cached` was verified bit-for-bit identical to
# `aligned_forward.target_context_aligned`, i.e. to the decode prefill, so both
# arms of Phase C now share one feature path.
#
# The old `head_video439_plateau/*` checkpoints stay on disk as the historical
# path baseline; they are not overwritten.
set -euo pipefail

R=/mnt/local/mcy/GLANCE
REPRO=/home/mcy/Projects/GLANCE/repro
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
GPU="${GPU:-0}"
STEPS="${STEPS:-1500}"
SAVE_EVERY="${SAVE_EVERY:-250}"
OUT="${OUT:-$R/checkpoints/head_video439_aligned}"
FEATURE_PATH="${FEATURE_PATH:-cached}"

cd "$REPRO"
mkdir -p "$R/logs" "$R/results/phaseA4"
echo "=== PHASE A4 aligned SFT  out=$OUT steps=$STEPS save_every=$SAVE_EVERY feature_path=$FEATURE_PATH gpu=$GPU $(date --iso-8601=seconds) ===" \
  | tee "$R/logs/phaseA4_sft439.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u historical/train_head_video.py \
  --records "$R/data/video_greedy_499.pt" \
  --pool    "$R/data/pool_499.parquet" \
  --init-from "$R/checkpoints/head" \
  --out "$OUT" \
  --steps "$STEPS" --save-every "$SAVE_EVERY" --log-every 50 \
  --max-anchors 96 --lr 3e-4 --seed 0 --gpu 0 \
  --feature-path "$FEATURE_PATH" \
  --log-file "$R/logs/phaseA4_sft439.jsonl" 2>&1 | tee -a "$R/logs/phaseA4_sft439.log"

echo "=== PHASE A4 aligned SFT done $(date --iso-8601=seconds) ===" \
  | tee -a "$R/logs/phaseA4_sft439.log"
