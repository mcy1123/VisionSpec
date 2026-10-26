#!/usr/bin/env bash
# ScienceQA image-reasoning experiment driver.
#
# Every stage writes into its own directory, appends the exact command to
# commands.txt, and never overwrites an existing artifact: re-running a stage
# that already produced its output is a no-op. Every stage is resumable and
# every stage is logged.
#
#   ./run_scienceqa.sh <stage> [args...]
#
# stages:
#   debug-ar      build a 40-slot debug AR set + debug schedule
#   smoke         both arms, 10 real updates each, from the debug schedule
#   train-ar      build the formal train AR set (needs --updates/--slots decided)
#   eval-ar       build the dev and test AR oracles (shared by every arm)
#   train         formal paired training: <arm> <seed> <gpu>
#   eval          lossless online evaluation: <tag> <head> <split> <gpu>
#   coverage      fixed-anchor candidate/tree trace: <tag> <head> <gpu>
set -uo pipefail

RUN="${RUN:-/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955}"
REPRO=/home/mcy/Projects/GLANCE/repro/scienceqa_image
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
I0=/mnt/local/mcy/GLANCE/checkpoints/head
LR=3e-5
QPU=4
APQ=6
MAXNEW=256
MAXPIX=768
BUDGET=63
BRANCH=8
OUTROOT="${OUTROOT:-$RUN}"
mkdir -p "$RUN/train" "$RUN/eval" "$RUN/logs" "$RUN/scripts" "$OUTROOT/train"
CMD="$RUN/commands.txt"

say() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$RUN/logs/driver.log"; }
rec() { echo "$*" >> "$CMD"; }

stage="${1:-}"; shift || true

case "$stage" in
# ---------------------------------------------------------------------------
debug-ar)
  [[ -f "$RUN/dataset/sqa_target_ar_df20261013_debug.pt" ]] && { say "debug AR exists, skip"; exit 0; }
  GPU="${1:-7}"
  say "debug AR build on GPU $GPU"
  rec "$PY -u $REPRO/build_training_data.py --run $RUN --seed 20261013 --gpu $GPU --updates 10 --limit 40 --max-new $MAXNEW --max-pixels $MAXPIX"
  "$PY" -u "$REPRO/build_training_data.py" --run "$RUN" --seed 20261013 \
      --gpu "$GPU" --updates 10 --limit 40 --max-new "$MAXNEW" --max-pixels "$MAXPIX" \
      > "$RUN/logs/build_debug_ar.log" 2>&1
  say "debug AR done rc=$?"
  ;;

# ---------------------------------------------------------------------------
smoke)
  ARM="${1:?arm sft|opd}"; GPU="${2:-7}"
  OUT="$RUN/train/smoke10/$ARM"
  SCHED="$RUN/dataset/train_schedule_seed_20261013_debug.jsonl"
  POOL="$RUN/dataset/sqa_pool_df20261013_debug.parquet"
  RECS="$RUN/dataset/sqa_target_ar_df20261013_debug.pt"
  [[ -d "$OUT/step10" ]] && { say "smoke $ARM already has step10, skip"; exit 0; }
  mkdir -p "$OUT"
  if [[ "$ARM" == sft ]]; then ENTRY=train_sqa_sft.py; EXTRA=();
  else ENTRY=train_sqa_opd.py; EXTRA=(--gamma 0.8); fi
  say "smoke $ARM on GPU $GPU"
  rec "CUDA_VISIBLE_DEVICES=$GPU $PY -u $REPRO/$ENTRY --run $RUN --seed 20261013 --gpu 0 --pool $POOL --records $RECS --init-from $I0 --out $OUT --updates 10 --qa-per-update $QPU --anchors-per-qa $APQ --lr $LR --master-weights cpu --target-dtype float32 --max-pixels $MAXPIX --schedule $SCHED --save-at 10 ${EXTRA[*]}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$REPRO/$ENTRY" --run "$RUN" --seed 20261013 \
      --gpu 0 --pool "$POOL" --records "$RECS" --init-from "$I0" --out "$OUT" \
      --updates 10 --qa-per-update "$QPU" --anchors-per-qa "$APQ" --lr "$LR" \
      --master-weights cpu --target-dtype float32 --max-pixels "$MAXPIX" \
      --schedule "$SCHED" --save-at 10 "${EXTRA[@]}" \
      > "$RUN/logs/smoke10_$ARM.log" 2>&1
  say "smoke $ARM done rc=$?"
  ;;

# ---------------------------------------------------------------------------
train-ar)
  SEED="${1:?seed}"; GPU="${2:-7}"; UPDATES="${3:?updates}"
  [[ -f "$RUN/dataset/sqa_target_ar_df${SEED}.pt" ]] && { say "train AR seed $SEED exists, skip"; exit 0; }
  say "train AR build seed $SEED, $((UPDATES*QPU)) slots, GPU $GPU"
  rec "$PY -u $REPRO/build_training_data.py --run $RUN --seed $SEED --gpu $GPU --updates $UPDATES --max-new $MAXNEW --max-pixels $MAXPIX"
  "$PY" -u "$REPRO/build_training_data.py" --run "$RUN" --seed "$SEED" \
      --gpu "$GPU" --updates "$UPDATES" --max-new "$MAXNEW" --max-pixels "$MAXPIX" \
      > "$RUN/logs/build_train_ar_$SEED.log" 2>&1
  say "train AR seed $SEED done rc=$?"
  ;;

# ---------------------------------------------------------------------------
eval-ar)
  SPLIT="${1:?dev|test}"; GPU="${2:-7}"
  [[ -f "$RUN/eval/ar_$SPLIT.pt" ]] && { say "eval AR $SPLIT exists, skip"; exit 0; }
  say "eval AR oracle for $SPLIT on GPU $GPU"
  rec "$PY -u $REPRO/build_eval_ar.py --run $RUN --split $SPLIT --gpu $GPU --max-new $MAXNEW --max-pixels $MAXPIX --out $RUN/eval/ar_$SPLIT.pt"
  "$PY" -u "$REPRO/build_eval_ar.py" --run "$RUN" --split "$SPLIT" --gpu "$GPU" \
      --max-new "$MAXNEW" --max-pixels "$MAXPIX" --out "$RUN/eval/ar_$SPLIT.pt" \
      > "$RUN/logs/build_eval_ar_$SPLIT.log" 2>&1
  say "eval AR $SPLIT done rc=$?"
  ;;

# ---------------------------------------------------------------------------
train)
  ARM="${1:?sft|opd}"; SEED="${2:?seed}"; GPU="${3:-7}"; UPDATES="${4:?updates}"
  OUT="$OUTROOT/train/seed$SEED/$ARM"
  [[ -f "$OUT/step$UPDATES/model.safetensors" ]] && { say "train $ARM/$SEED final exists, skip"; exit 0; }
  mkdir -p "$OUT"
  if [[ "$ARM" == sft ]]; then ENTRY=train_sqa_sft.py; EXTRA=();
  else ENTRY=train_sqa_opd.py; EXTRA=(--gamma 0.8); fi
  if [[ -f "$OUT/training_state.pt" ]]; then
      RESUME=(--resume-from "$OUT" --resume-update "$(cat "$OUT/resume_update" 2>/dev/null || echo 0)")
  else RESUME=(); fi
  say "train $ARM seed $SEED updates $UPDATES GPU $GPU"
  rec "CUDA_VISIBLE_DEVICES=$GPU $PY -u $REPRO/$ENTRY --run $RUN --seed $SEED --gpu 0 --pool $RUN/dataset/sqa_pool_df$SEED.parquet --records $RUN/dataset/sqa_target_ar_df$SEED.pt --init-from $I0 --out $OUT --updates $UPDATES --qa-per-update $QPU --anchors-per-qa $APQ --lr $LR --master-weights cpu --target-dtype float32 --max-pixels $MAXPIX --save-at 30,60,120 ${EXTRA[*]} ${RESUME[*]}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$REPRO/$ENTRY" --run "$RUN" --seed "$SEED" \
      --gpu 0 --pool "$RUN/dataset/sqa_pool_df$SEED.parquet" \
      --records "$RUN/dataset/sqa_target_ar_df$SEED.pt" --init-from "$I0" \
      --out "$OUT" --updates "$UPDATES" --qa-per-update "$QPU" --anchors-per-qa "$APQ" \
      --lr "$LR" --master-weights cpu --target-dtype float32 --max-pixels "$MAXPIX" \
      --save-at 30,60,120 "${EXTRA[@]}" "${RESUME[@]}" \
      > "$RUN/logs/train_${ARM}_seed${SEED}.log" 2>&1
  say "train $ARM seed $SEED done rc=$?"
  ;;

# ---------------------------------------------------------------------------
eval)
  TAG="${1:?tag}"; HEAD="${2:?head}"; SPLIT="${3:?dev|test}"; GPU="${4:-7}"
  LIMIT="${5:-0}"
  OUT="$RUN/eval/${TAG}_${SPLIT}.json"
  [[ -f "$OUT" ]] && { say "eval $TAG/$SPLIT exists, skip"; exit 0; }
  say "eval $TAG on $SPLIT GPU $GPU head $HEAD"
  rec "$PY -u $REPRO/eval_sqa.py --run $RUN --head $HEAD --tag $TAG --split $SPLIT --gpu $GPU --budget $BUDGET --branch-k $BRANCH --max-new $MAXNEW --max-pixels $MAXPIX --head-dtype float32 --ar-cache $RUN/eval/ar_$SPLIT.pt --repeat-spec 2 --repeat-n 24 --limit $LIMIT --out $OUT"
  "$PY" -u "$REPRO/eval_sqa.py" --run "$RUN" --head "$HEAD" --tag "$TAG" \
      --split "$SPLIT" --gpu "$GPU" --budget "$BUDGET" --branch-k "$BRANCH" \
      --max-new "$MAXNEW" --max-pixels "$MAXPIX" --head-dtype float32 \
      --ar-cache "$RUN/eval/ar_$SPLIT.pt" --repeat-spec 2 --repeat-n 24 \
      --limit "$LIMIT" --out "$OUT" > "$RUN/logs/eval_${TAG}_${SPLIT}.log" 2>&1
  say "eval $TAG/$SPLIT done rc=$?"
  ;;

# ---------------------------------------------------------------------------
coverage)
  TAG="${1:?tag}"; HEAD="${2:?head}"; GPU="${3:-7}"
  OUT="$RUN/eval/coverage_${TAG}.json"
  [[ -f "$OUT" ]] && { say "coverage $TAG exists, skip"; exit 0; }
  say "coverage $TAG GPU $GPU head $HEAD"
  rec "$PY -u $REPRO/trace_tree_coverage.py --run $RUN --head $HEAD --tag $TAG --gpu $GPU --split dev --budget $BUDGET --branch-k $BRANCH --max-new $MAXNEW --max-pixels $MAXPIX --anchors 3 --out $OUT"
  "$PY" -u "$REPRO/trace_tree_coverage.py" --run "$RUN" --head "$HEAD" --tag "$TAG" \
      --gpu "$GPU" --split dev --budget "$BUDGET" --branch-k "$BRANCH" \
      --max-new "$MAXNEW" --max-pixels "$MAXPIX" --anchors 3 --out "$OUT" \
      > "$RUN/logs/coverage_${TAG}.log" 2>&1
  say "coverage $TAG done rc=$?"
  ;;

*) echo "unknown stage '$stage'" >&2; exit 2 ;;
esac
