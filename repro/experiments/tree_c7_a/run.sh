#!/usr/bin/env bash
# Task A (parallel_20261012) -- one-command CPU reproduction of Stage 0+1.
#
# Creates a fresh timestamped result directory, copies the frozen
# preregistration into it, runs the unit tests, the Stage 0 replay/Stage 1
# counterfactual and the post-hoc mechanism analysis, and writes the logs.
# No GPU is used.
#
# Usage:  bash repro/experiments/tree_c7_a/run.sh [results_root]
set -euo pipefail

ROOT=/home/mcy/Projects/GLANCE
PY=${PY:-/home/mcy/.conda/envs/fin_research/bin/python}
OUTROOT=${1:-/mnt/local/mcy/GLANCE/results/parallel_A_tree}
TS=$(date +%Y%m%dT%H%M%S)
RUN="$OUTROOT/$TS"

mkdir -p "$RUN"
cp "$ROOT/repro/experiments/tree_c7_a/prereg.json" "$RUN/prereg.json"
cp "$ROOT/repro/experiments/tree_c7_a/fixed_quota_tree.py" "$RUN/source_fixed_quota_tree.py"
cp "$ROOT/repro/experiments/tree_c7_a/run_stage01.py" "$RUN/source_run_stage01.py"

echo "== run dir: $RUN"
cd "$ROOT"

echo "== unit tests"
"$PY" -m unittest -v repro.experiments.tree_c7_a.test_fixed_quota_tree \
    > "$RUN/tests.log" 2>&1
tail -3 "$RUN/tests.log"

echo "== stage 0 + stage 1"
"$PY" -u repro/experiments/tree_c7_a/run_stage01.py --out "$RUN" \
    > "$RUN/stage01.log" 2>&1
cat "$RUN/stage01.log"

echo "== mechanism analysis"
"$PY" -u repro/experiments/tree_c7_a/analyse_mechanism.py --out "$RUN" \
    > "$RUN/mechanism_analysis.log" 2>&1
head -20 "$RUN/mechanism_analysis.json"

echo "== done: $RUN"
