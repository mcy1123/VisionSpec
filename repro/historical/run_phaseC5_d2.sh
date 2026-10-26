#!/usr/bin/env bash
# phaseC5 D2: one (topology, arm, row-list) evaluation of the gain-stuck
# diagnostic. READ-ONLY with respect to parameters: this only ever calls
# `evaluation/eval_video_opd.py`, which loads a checkpoint and decodes from it.
#
# WHY a wrapper instead of a raw command line: the two topologies must differ in
# EXACTLY budget/branch and nothing else, or the chain-vs-tree comparison is not
# one comparison. This script is the single place that mapping lives.
#
#   tree  = the existing evaluation tree, budget 63 / branch 8
#   chain = --budget 1 --branch 1, i.e. ONE proposed token per round. This is
#           "component-wise chain" (block = 1) and is NOT the same object as the
#           block-16 chain rollout used in training: the drafter still runs its
#           16-slot block head, but the verifier commits at most one token per
#           round. It is the closest deployment-side probe of "does the head's
#           first-proposal token get accepted" that the existing decode path can
#           express without changing `glance.generate`.
#
# The `--ar` lossless gate is on for EVERY run and its pass/fail is in the log
# (`LOSSLESS n/30 exact`) and in the output JSON's per-row `lossless` field.
set -euo pipefail

usage() {
  echo "usage: $0 <tree|chain> <C0|C1|C2> <rows.parquet> <tag> <gpu>" >&2
  exit 2
}
[ $# -eq 5 ] || usage

TOPO="$1"; ARM="$2"; ROWS="$3"; TAG="$4"; GPU="$5"
R=/mnt/local/mcy/GLANCE
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
RES="$R/results/phaseC5"
mkdir -p "$RES"

case "$TOPO" in
  tree)  BUDGET=63; BRANCH=8 ;;
  chain) BUDGET=1;  BRANCH=1 ;;
  *) usage ;;
esac
case "$ARM" in
  C0) HEAD="$R/checkpoints/head_video439_aligned/step1000" ;;
  C1) HEAD="$R/checkpoints/phaseC4_C1/step10" ;;
  C2) HEAD="$R/checkpoints/phaseC4_C2/step10" ;;
  *) usage ;;
esac
[ -d "$HEAD" ] || { echo "missing checkpoint $HEAD" >&2; exit 1; }
[ -f "$ROWS" ] || { echo "missing rows $ROWS" >&2; exit 1; }

cd /home/mcy/Projects/GLANCE/repro
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "=== $TAG topo=$TOPO b$BUDGET k$BRANCH arm=$ARM gpu=$GPU $(date --iso-8601=seconds) ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluation/eval_video_opd.py \
  --rows "$ROWS" --head "$HEAD" --n 0 --max-new 1024 \
  --budget "$BUDGET" --branch "$BRANCH" --ar --warmup 1 --dtype float32 \
  --gpu 0 --tag "$TAG" --out "$RES/${TAG}.json" 2>&1 | tee "$RES/${TAG}.log"
echo "=== $TAG done $(date --iso-8601=seconds) ==="
