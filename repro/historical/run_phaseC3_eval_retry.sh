#!/usr/bin/env bash
# Same evals as historical/run_phaseC3_eval.sh, but GPU-aware: an fp32 eval needs ~40 GiB
# free (fp32 target 32.7 GiB + fp32 head 3.9 GiB + activations, and the load-time
# dtype conversion transiently holds both copies). If no A6000 has that, wait and
# retry instead of OOM-ing or disturbing another user's job.
set -uo pipefail
R=/mnt/local/mcy/GLANCE
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
RES="$R/results/phaseC3"; MON="$R/data/video_fresh30_monitor8.parquet"
NEED_MIB=40000
cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$RES"

pick_gpu () {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
  | awk -F', ' -v need="$NEED_MIB" '$2-$3 >= need {print $1; exit}'
}

for LR in 3e-5 1e-5; do
  for NODE in 5 10 25; do
    TAG="mw_lr${LR}_u${NODE}_monitor8"
    OUT="$RES/${TAG}.json"
    [ -f "$OUT" ] && { echo "[skip] $TAG done"; continue; }
    HEAD="$R/checkpoints/phaseC3_C1mw_lr${LR}/step${NODE}"
    [ -d "$HEAD" ] || { echo "[skip] $TAG missing head"; continue; }
    while true; do
      G=$(pick_gpu)
      if [ -n "$G" ]; then
        echo "=== EVAL $TAG on physical GPU $G $(date --iso-8601=seconds) ===" | tee -a "$RES/retry.log"
        CUDA_VISIBLE_DEVICES="$G" "$PY" -u evaluation/eval_video_opd.py \
          --rows "$MON" --head "$HEAD" --n 0 --max-new 1024 --budget 63 --branch 8 \
          --ar --warmup 0 --dtype float32 --gpu 0 --tag "$TAG" \
          --out "$OUT" > "$RES/${TAG}.log" 2>&1 && { echo "[ok] $TAG" | tee -a "$RES/retry.log"; break; }
        echo "[retry] $TAG failed on GPU $G (likely OOM); waiting" | tee -a "$RES/retry.log"
      fi
      sleep 300
    done
  done
done
echo "=== PHASE C3 EVAL DONE $(date --iso-8601=seconds) ===" | tee -a "$RES/retry.log"
