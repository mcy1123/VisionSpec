#!/usr/bin/env bash
# Capture EAGLE-3 offline features for the video rows from Qwen3-VL-8B.
#
# This is the step SpecForge cannot do for itself: its capture path is text-only,
# SGLang-only, and forces ForwardMode.DECODE, which skips SGLang's multimodal
# embedding branch entirely. So the target forward is run here, on the host, with
# the same processor call GLANCE uses -- which also means EAGLE-3's features come
# from the identical prompt encoding GLANCE trained on.
#
#   GPU=7 bash 10_capture.sh train      # 409 rows, the GLANCE OPD training set
#   GPU=7 bash 10_capture.sh select     # 60 rows, checkpoint selection
#   GPU=7 bash 10_capture.sh holdout30  # 30 rows, the independent holdout
#   GPU=7 bash 10_capture.sh all
#
# Cost: one teacher-forced prefill per row, ~1-2 s/row on an A6000 once the
# visual tower's Conv3d is patched (common.use_fast_patch_embed, ~62x there).
# Storage: ~36 MB/row => ~15 GB for train, ~2 GB select, ~1 GB holdout30.
set -euo pipefail

R=/mnt/local/mcy/GLANCE
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/mcy/.conda/envs/fin_research/bin/python
STAGE=${STAGE:-/mnt/local3/mcy/specforge/eagle3_video}
GPU=${GPU:-7}
SPLIT=${1:?usage: 10_capture.sh train|select|holdout30|all}

# The GPUs are shared with other users' jobs. Capture needs the 8B target in
# bf16 (~17 GB) plus activations, so refuse to start on a card that cannot hold
# it rather than OOM halfway through a 20-minute run.
free_mib=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU $GPU free: ${free_mib} MiB"
if [ "$free_mib" -lt 22000 ]; then
  echo "REFUSING: need >= 22000 MiB free for Qwen3-VL-8B + activations" >&2
  exit 1
fi

capture() {
  local split=$1
  local out="$STAGE/$split"
  echo "=== capture $split -> $out"
  mkdir -p "$out"
  "$PY" -u "$HERE/capture_features.py" \
    --split "$split" --out "$out" --gpu "$GPU" --resume
  du -sh "$out"
}

case "$SPLIT" in
  all)
    capture train; capture select; capture holdout30 ;;
  train|select|holdout30)
    capture "$SPLIT" ;;
  *)
    echo "unknown split $SPLIT" >&2; exit 2 ;;
esac

echo "done. next: bash 20_train.sh"
