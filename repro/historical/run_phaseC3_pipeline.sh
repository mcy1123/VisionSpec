#!/usr/bin/env bash
# Completes the phaseC3 flow that the six evals alone do not:
#   wait for the 6 monitor evals -> apply the FROZEN selection rule ->
#   run the full validation (30 rows, warmup 1, n=29) on the single winner ->
#   score the winner against C0 with the same >= -0.02 threshold.
# The rule is unchanged from docs/reports/2026-10-10-phase-c2-prereg.md.
set -uo pipefail
R=/mnt/local/mcy/GLANCE
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
RES="$R/results/phaseC3"; MON="$R/data/video_fresh30_monitor8.parquet"
FULL="$R/data/video_fresh30.parquet"
cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 1. wait until all six monitor JSONs exist and no eval process is running
while true; do
  n=0
  for LR in 3e-5 1e-5; do for U in 5 10 25; do
    [ -f "$RES/mw_lr${LR}_u${U}_monitor8.json" ] && n=$((n+1))
  done; done
  [ "$n" -eq 6 ] && break
  echo "[pipeline] $n/6 monitor JSONs" >> "$RES/pipeline.log"
  sleep 120
done

# 2. frozen selection rule
"$PY" evaluation/analyze_phaseC2_calib.py --dir "$RES" --tag 'mw_lr{lr}_u{node}_monitor8' \
  --out "$RES/summary_calib_monitor8.json" 2>&1 | tee -a "$RES/pipeline.log"

# 3. full validation on the winner, on a GPU with room for an fp32 eval
WIN=$("$PY" -c "
import json;d=json.load(open('$RES/summary_calib_monitor8.json'));w=d['winner']
print('' if w is None else f\"{w['lr']}_{w['update']}\")")
if [ -z "$WIN" ]; then
  echo "[pipeline] no candidate passed the -0.02 threshold -> STOP, do not run full validation" | tee -a "$RES/pipeline.log"
  exit 0
fi
LR=${WIN%%_*}; U=${WIN##*_}
TAG="mw_lr${LR}_u${U}_fresh30"
echo "[pipeline] winner lr=$LR update=$U -> $TAG" | tee -a "$RES/pipeline.log"
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
      | awk -F', ' '$2-$3 >= 40000 {print $1; exit}')
  [ -n "$G" ] && break; sleep 300
done
CUDA_VISIBLE_DEVICES="$G" "$PY" -u evaluation/eval_video_opd.py \
  --rows "$FULL" --head "$R/checkpoints/phaseC3_C1mw_lr${LR}/step${U}" \
  --n 0 --max-new 1024 --budget 63 --branch 8 --ar --warmup 1 --dtype float32 --gpu 0 \
  --tag "$TAG" --out "$RES/${TAG}.json" > "$RES/${TAG}.log" 2>&1
echo "[pipeline] full validation done -> $RES/${TAG}.json" | tee -a "$RES/pipeline.log"
echo "=== PHASE C3 PIPELINE DONE ===" | tee -a "$RES/pipeline.log"
