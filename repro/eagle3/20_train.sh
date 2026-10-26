#!/usr/bin/env bash
# Train the EAGLE-3 draft head on the video rows, inside the patched SpecForge.
#
#   GPU=7 bash 20_train.sh            # standard recipe: ttt_length 3, lr 1e-4
#   GPU=7 bash 20_train.sh --ttt 7    # repair 1, if the standard run does not converge
#   ROUND=2 GPU=7 bash 20_train.sh --lr 5e-5   # repair 2
#
# The run config (/data3/eagle3_video/code/run_eagle3_video.yaml) is the standard
# EAGLE-3 recipe for this target and is NOT tuned toward GLANCE's numbers:
#   draft layers 1, ttt_length 3, lr 1e-4, AdamW-style, 40 epochs, EAGLE-3's own
#   3 aux layers [1, 17, 32] rather than GLANCE's 5 kept layers.
# Overrides are applied as dotted keys so every repair is recorded in the log
# beside the config it modified.
set -euo pipefail

R=/mnt/local/mcy/GLANCE
SPECFORGE=${SPECFORGE_ROOT:-/home/mcy/Projects/SpecForge}
STAGE=${STAGE:-/mnt/local3/mcy/specforge/eagle3_video}
CONTAINER=${CONTAINER:-sf-glance}
GPU=${GPU:-7}
ROUND=${ROUND:-1}
LOG=${LOG:-/mnt/local/mcy/GLANCE/logs/eagle3_train_round${ROUND}.log}
CONFIG=/data3/eagle3_video/code/run_eagle3_video.yaml

TTT=${TTT:-}
LR=${LR:-}
ATTN=${ATTN:-}
EPOCHS=${EPOCHS:-}
OVER=""
[ -n "$TTT" ]    && OVER="$OVER training.ttt_length=$TTT"
[ -n "$LR" ]     && OVER="$OVER training.learning_rate=$LR"
[ -n "$ATTN" ]   && OVER="$OVER training.attention_backend=$ATTN"
[ -n "$EPOCHS" ] && OVER="$OVER training.num_epochs=$EPOCHS"

# The trainer must not run before the CPU-only gates have passed; the patches in
# particular fail silently, so re-check them here rather than trusting that
# someone ran 00_validate.sh.
echo "=== preflight: patches"
/home/mcy/.conda/envs/fin_research/bin/python \
  "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/specforge_patches.py" --check

echo "=== preflight: captured features present"
for split in train select; do
  n=$(find "$STAGE/$split" -name '*.ckpt' 2>/dev/null | wc -l)
  echo "  $split: $n .ckpt"
  if [ "$n" -eq 0 ]; then
    echo "REFUSING: no features for split '$split'; run 10_capture.sh first" >&2
    exit 1
  fi
done

echo "=== container $CONTAINER"
if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker run -d --name "$CONTAINER" \
    --gpus "device=$GPU" \
    -v "$SPECFORGE":/workspace \
    -v "$R":/glance:ro \
    -v "$(dirname "$STAGE")":/data3 \
    -w /workspace python:3.11-slim sleep infinity >/dev/null
  echo "  created on GPU $GPU"
else
  echo "  reusing; confirm it was created with --gpus device=$GPU"
fi

echo "=== training (round $ROUND) -> $LOG"
echo "    config $CONFIG $OVER"
mkdir -p "$(dirname "$LOG")"
docker exec "$CONTAINER" bash -lc \
  "cd /workspace && .venv/bin/specforge train --config $CONFIG $OVER" \
  2>&1 | tee "$LOG"

echo "=== checkpoints under $STAGE/out"
find "$STAGE/out" -maxdepth 2 -name '*.safetensors' -o -maxdepth 2 -type d | head -20
echo "TRAIN EXIT ${PIPESTATUS[0]}"
