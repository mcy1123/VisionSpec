#!/usr/bin/env bash
# Wait for all six Phase A4 eval JSONs, then run the pre-registered A3/A4.4
# plateau judgement on the aligned curve and write the summary.
set -euo pipefail
R=/mnt/local/mcy/GLANCE
RES="$R/results/phaseA4"
STEPS="250 500 750 1000 1250 1500"
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"

for i in $(seq 1 480); do   # up to ~8h at 60s
  n=0
  for s in $STEPS; do
    [ -f "$RES/sft439_step${s}_fresh30_tree_float32.json" ] && n=$((n+1))
  done
  echo "[$(date '+%H:%M:%S')] $n/6 A4 result JSONs present"
  if [ "$n" -eq 6 ] && ! pgrep -f "eval_video_opd" >/dev/null; then
    echo "all present and eval process gone -> summarising"
    break
  fi
  sleep 60
done

cd /home/mcy/Projects/GLANCE/repro
"$PY" evaluation/summarize_phaseA.py --dir "$RES" --steps $STEPS \
  --out "$RES/summary_phaseA4.json" 2>&1 | tee "$RES/summary_phaseA4.log"
echo "WATCHER DONE"
