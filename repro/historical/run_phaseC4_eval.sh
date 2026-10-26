#!/usr/bin/env bash
# Formal evaluation of the frozen 10-update comparison. FIXED at update 10 -- no
# checkpoint is chosen after seeing a result. Full video_fresh30, fp32, tree
# b63/k8, max_new 1024, to EOS, --ar lossless gate, --warmup 1 (n=29) so it
# matches the C0 numbers taken from the phase-A4 evaluation of the same start
# checkpoint. Waits for a GPU with room for an fp32 eval.
set -uo pipefail
R=/mnt/local/mcy/GLANCE
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
RES="$R/results/phaseC4"; FULL="$R/data/video_fresh30.parquet"
cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$RES"
while pgrep -f 'phaseC4_1[0]upd' >/dev/null; do sleep 30; done
for ARM in C1 C2; do
  TAG="phaseC4_${ARM}_update10_fresh30"
  [ -f "$RES/${TAG}.json" ] && { echo "[skip] $TAG"; continue; }
  HEAD="$R/checkpoints/phaseC4_${ARM}/step10"
  [ -d "$HEAD" ] || { echo "[skip] missing $HEAD" >&2; continue; }
  while true; do
    G=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2-$3 >= 40000 {print $1; exit}')
    [ -n "$G" ] && break; sleep 300
  done
  echo "=== EVAL $TAG on GPU $G $(date --iso-8601=seconds) ===" | tee -a "$RES/eval.log"
  CUDA_VISIBLE_DEVICES="$G" "$PY" -u evaluation/eval_video_opd.py \
    --rows "$FULL" --head "$HEAD" --n 0 --max-new 1024 --budget 63 --branch 8 \
    --ar --warmup 1 --dtype float32 --gpu 0 --tag "$TAG" \
    --out "$RES/${TAG}.json" > "$RES/${TAG}.log" 2>&1 \
    && echo "[ok] $TAG" | tee -a "$RES/eval.log" \
    || echo "[fail] $TAG" | tee -a "$RES/eval.log"
done
echo "=== PHASE C4 EVAL DONE $(date --iso-8601=seconds) ===" | tee -a "$RES/eval.log"
