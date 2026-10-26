#!/usr/bin/env bash
# End-to-end task-B run for a NEW timestamp directory.
#
#   bash repro/experiments/temporal_d4_b/run_b.sh 20261012T233231
#
# It never writes into another direction's directory and never touches
# GLANCE_repo/glance, repro/shared/common.py or the shared AGENTS.md.  The GPU leg runs
# inside the shared flock and re-checks the actual free memory before starting.
set -euo pipefail

TS="${1:?usage: run_b.sh <timestamp>}"
ROOT=/home/mcy/Projects/GLANCE
ART=/mnt/local/mcy/GLANCE
RUN="$ART/results/parallel_B_temporal/$TS"
LOCK="$ART/.locks/glance_parallel_gpu.lock"
PY=/home/mcy/.conda/envs/fin_research/bin/python
SRC="$ROOT/repro/experiments/temporal_d4_b"
GPU="${GPU:-0}"
VIDEOS="${VIDEOS:-}"

mkdir -p "$RUN" "$ART/.locks"

echo "=== stage A: C6 audit and unit selection (CPU) $(date -Is)"
"$PY" "$SRC/audit_select.py" --out "$RUN/stageA" | tee "$RUN/stageA_select.log"

echo "=== stage B: freeze the prereg (CPU) $(date -Is)"
"$PY" "$SRC/write_prereg.py" --out "$RUN"

echo "=== stage C: build and verify T/K clips (CPU) $(date -Is)"
"$PY" "$SRC/build_conditions.py" --out "$RUN" | tee "$RUN/build_conditions.log"

echo "=== unit tests (CPU) $(date -Is)"
( cd "$ROOT" && B_RUN="$RUN" "$PY" -m unittest discover \
    -s repro/experiments/temporal_d4_b -p "test_*.py" -v ) 2>&1 | tee "$RUN/tests.log"

echo "=== GPU leg under flock $(date -Is)"
VARG=""
[ -n "$VIDEOS" ] && VARG="--videos $VIDEOS"
flock -x "$LOCK" bash -c "
set -e
echo '=== lock acquired' \$(date -Is)
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
FREE=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
echo \"GPU$GPU free MiB: \$FREE\"
if [ \"\$FREE\" -lt 40000 ]; then echo 'GPU too small; aborting'; exit 3; fi
cd $ROOT
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True $PY -u \
  $SRC/run_forward.py --run '$RUN' --out '$RUN/main' --gpu $GPU $VARG
echo '=== analysis' \$(date -Is)
$PY $SRC/analyze_b.py --run '$RUN' --out '$RUN/main/analysis'
" 2>&1 | tee "$RUN/main.log"

echo "=== run manifest $(date -Is)"
"$PY" "$SRC/package_b.py" --run "$RUN" --main main
echo "done: $RUN"
