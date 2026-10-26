#!/usr/bin/env bash
# EAGLE-3 vs GLANCE video comparison -- stage 0, the zero-GPU gate.
#
# Everything here runs on CPU and blocks the GPU phase. If any step fails, the
# training run would have been wrong in a way that costs hours to notice:
#
#   patches   SpecForge's EAGLE-3 path needs interleaved mRoPE (Qwen3-VL) and a
#             position_ids route through the offline reader/normalizer. Both fail
#             SILENTLY if missing.
#   mrope     the interleaved frequency layout is checked numerically against
#             transformers, with a negative control so the test cannot pass blind.
#   config    the run YAML against SpecForge's own pydantic schema (extra="forbid"),
#             plus the registry check, the draft/target dimension match, and the
#             target shard-key lookup.
#   ingest    reader -> normalizer -> collator over stub features with the real
#             shapes, asserting position_ids survives as [3, B, L].
#
#   bash 00_validate.sh
#
# Validation runs in throwaway containers (--rm, no --gpus), so it leaves nothing
# behind and cannot be mistaken for the GPU container 20_train.sh needs.
set -euo pipefail

R=/mnt/local/mcy/GLANCE
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/mcy/.conda/envs/fin_research/bin/python
SPECFORGE=${SPECFORGE_ROOT:-/home/mcy/Projects/SpecForge}
STAGE=${STAGE:-/mnt/local3/mcy/specforge/eagle3_video}   # == /data3 in-container
STUB_DIR=${STUB_DIR:-train_stub}
IMAGE=${IMAGE:-python:3.11-slim}
MOUNTS=(-v "$SPECFORGE":/workspace -v "$R":/glance:ro -v "$(dirname "$STAGE")":/data3)

echo "=== [0] stage the code + metadata the container needs"
mkdir -p "$STAGE/code" "$STAGE/target_meta"
for f in draft_qwen3vl_eagle3.json run_eagle3_video.yaml validate_offline_ingest.py \
         test_mrope_interleaved.py test_offline_ingest.py; do
  cp "$HERE/$f" "$STAGE/code/"
done
cp "$R/models/Qwen3-VL-8B-Instruct/config.json" "$STAGE/target_meta/"
cp "$R/models/Qwen3-VL-8B-Instruct/model.safetensors.index.json" "$STAGE/target_meta/"
cp "$R/models/Qwen3-VL-8B-Instruct/config.json" "$STAGE/qwen3vl8b_config.json"
echo "  staged under $STAGE"

echo
echo "=== [1] stub features (real shapes, zero GPU) -> $STAGE/$STUB_DIR"
"$PY" -u "$HERE/capture_features.py" --split train --stub --limit 6 \
  --out "$STAGE/$STUB_DIR" 2>&1 | tail -4

run_in_container() {
  docker run --rm "${MOUNTS[@]}" -w /workspace "$IMAGE" \
    bash -lc "cd /workspace && $*"
}

echo
echo "=== [2] SpecForge patches are applied"
"$PY" "$HERE/specforge_patches.py" --apply

echo
echo "=== [3] interleaved mRoPE == transformers (negative control included)"
run_in_container ".venv/bin/python /data3/eagle3_video/code/test_mrope_interleaved.py \
  --target-config /data3/eagle3_video/qwen3vl8b_config.json" 2>&1 | grep -v "UserWarning\|warnings.warn"

echo
echo "=== [4] run config vs SpecForge schema + registry + target keys"
run_in_container ".venv/bin/python /data3/eagle3_video/code/validate_offline_ingest.py \
  --config /data3/eagle3_video/code/run_eagle3_video.yaml \
  --target-config /data3/eagle3_video/target_meta/config.json \
  --target-dir /data3/eagle3_video/target_meta \
  --features /data3/eagle3_video/$STUB_DIR" 2>&1 | grep -v "UserWarning\|warnings.warn"

echo
echo "=== [5] offline ingest: reader -> normalizer -> collator"
run_in_container ".venv/bin/python /data3/eagle3_video/code/test_offline_ingest.py \
  --features /data3/eagle3_video/$STUB_DIR" 2>&1 | grep -v "UserWarning\|warnings.warn"

echo
echo "=== [6] training data (JSONL) round-trip gate"
"$PY" -u "$HERE/../build_eagle3_data.py" --verify 4 2>&1 | grep -v "UserWarning\|warnings.warn"

echo
echo "ALL STAGE-0 GATES PASSED -- next: 10_capture.sh, then 20_train.sh"
