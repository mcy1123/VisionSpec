#!/usr/bin/env bash
# Run only on the execution server. One seed, A/B/C x 60 updates, no retries.
set -euo pipefail
: "${EV_PY:?Set EV_PY to the verified Python interpreter}"
: "${EV_RUN:?Set EV_RUN to a NEW directory under an existing /mnt/local mount}"
: "${EV_TARGET:?Set EV_TARGET to the verified matching 8B target}"
: "${EV_BASE:?Set EV_BASE to the common evidence-free head}"
: "${EV_DATA:?Set EV_DATA to the existing real-video data_parallel directory}"
: "${CUDA_VISIBLE_DEVICES:?Select exclusively available GPU UUIDs}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export EV_PY EV_RUN EV_TARGET EV_BASE EV_DATA
EV_SEED=${EV_SEED:-20261112}
export EV_SEED
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# Validate before creating outputs. No allocation or process termination here.
"$EV_PY" - <<'PY'
import os
from pathlib import Path
import torch
p = Path(os.environ['EV_RUN']).expanduser().resolve()
roots = [Path(x).resolve() for x in ('/mnt/local', '/mnt/local2', '/mnt/local3') if Path(x).is_dir()]
assert any(r in p.parents for r in roots), 'EV_RUN must be below an existing approved mount'
assert not p.exists(), 'EV_RUN must be new; no overwrite or implicit resume'
assert hasattr(torch.nn.utils, 'get_total_norm'), 'use the known-working training environment: get_total_norm is required'
ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(ids) >= 3 and len(ids) == len(set(ids)) and all(x.startswith('GPU-') for x in ids), 'select >=3 distinct full GPU UUIDs'
for name in ('EV_TARGET', 'EV_BASE', 'EV_DATA'):
    assert Path(os.environ[name]).is_dir(), name + ' is not a directory'
for name in ('train_128.parquet', 'train_128.jsonl', 'dev_128.parquet'):
    assert (Path(os.environ['EV_DATA']) / name).is_file(), name + ' is missing'
PY
mkdir -p "$EV_RUN/logs" "$EV_RUN/eval"

finish() {
  code=$?
  trap - EXIT
  "$EV_PY" - "$code" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
root=Path(os.environ['EV_RUN'])
arms={}
for name in ('A', 'B', 'C'):
    status=root / ('train_' + name) / 'RUN_STATUS.json'
    if status.is_file():
        arms[name]=json.loads(status.read_text())
    else:
        log=root / ('train_' + name) / 'log.jsonl'
        last=json.loads(log.read_text().splitlines()[-1]) if log.is_file() and log.stat().st_size else {}
        arms[name]={'status':'NOT_STARTED_OR_FAILED','last_logged_update':last.get('update',0)}
record={'status':'COMPLETED' if sys.argv[1]=='0' else ('STOPPED' if sys.argv[1]=='3' else 'FAILED'),
        'exit_code':int(sys.argv[1]), 'finished_utc':datetime.now(timezone.utc).isoformat(),
        'training_update_cap':180, 'arms':arms, 'resume_supported':False}
with (root/'SUITE_STATUS.json').open('x') as f: json.dump(record,f,indent=2)
PY
  exit "$code"
}
trap finish EXIT

run() {
  local label=$1
  shift
  echo "[$(date -u +%FT%TZ)] $label -> $EV_RUN/logs/$label.log"
  "$@" >"$EV_RUN/logs/$label.log" 2>&1
}
check_gpus() {
  "$EV_PY" - <<'PY'
import os, subprocess
selected=set(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
out=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
free={r.split(',')[0].strip():int(r.split(',')[1]) for r in out.splitlines() if r.strip()}
assert selected <= free.keys() and all(free[x]>=40*1024 for x in selected), 'need >=40 GiB free on each selected GPU'
out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
busy=[r for r in out.splitlines() if r.split(',')[0].strip() in selected]
assert not busy, 'selected GPUs already have compute processes: '+str(busy)
PY
}
run cpu_tests env CUDA_VISIBLE_DEVICES='' "$EV_PY" -m pytest -q \
  repro/tests/test_real_video_pilot.py repro/tests/test_pilot_schedule.py \
  repro/tests/test_video_9k_protocol.py repro/tests/test_video_visual_probe.py \
  repro/tests/test_video_eval_merge.py repro/tests/test_visual_context_cache.py \
  repro/tests/test_video_training_stop.py repro/tests/test_evidence_readout.py \
  repro/tests/test_evidence_training.py repro/tests/test_evidence_eval.py \
  repro/tests/test_evidence_analysis.py repro/tests/test_evidence_decoder_cpu.py \
  repro/tests/test_evidence_memory.py
run init_heads env CUDA_VISIBLE_DEVICES='' "$EV_PY" repro/video_caption/init_evidence_heads.py \
  --source "$EV_BASE" --out "$EV_RUN/initial_heads" --seed "$EV_SEED" \
  --evidence-dim 256 --max-tokens 256 --spatial-bins 2
run schedule "$EV_PY" repro/video_caption/build_pilot_schedule.py \
  --rows "$EV_DATA/train_128.jsonl" --epochs 3 --qa-per-update 4 \
  --seed "$EV_SEED" --out "$EV_RUN/schedule.jsonl"

declare -A HEADS=([A]="$EV_BASE" [B]="$EV_RUN/initial_heads/B_text" [C]="$EV_RUN/initial_heads/C_visual")
TRAIN=(repro/video_caption/train_video_9k.py --pool "$EV_DATA/train_128.parquet" \
  --schedule "$EV_RUN/schedule.jsonl" --target "$EV_TARGET" --updates 60 \
  --qa-per-update 4 --seed "$EV_SEED" --time-mode real --loss-normalization global_pooled \
  --device-map balanced --master-weights cpu --max-new-tokens 1024 --max-seq-len 32768 \
  --e2e-tv-weight 10 --lr 1e-5 --save-every 20 --stop-file "$EV_RUN/STOP_TRAINING")
if [[ -n "${EV_TRAIN_STOP:-}" ]]; then TRAIN+=(--deadline-utc "$EV_TRAIN_STOP"); fi
EVAL=(repro/evaluation/eval_video_opd.py --rows "$EV_DATA/dev_128.parquet" \
  --target "$EV_TARGET" --time-mode real --require-exact-time --eos-stop immediate \
  --dtype float32 --device-map balanced --gpu 0 --max-new 1024 --budget 63 --branch 8 --ar)
for arm in A B C; do
  run "preflight_$arm" env CUDA_VISIBLE_DEVICES='' "$EV_PY" "${TRAIN[@]}" \
    --init-from "${HEADS[$arm]}" --out "$EV_RUN/train_$arm" --dry-run
done
for arm in A B C; do
  check_gpus
  run "gate_$arm" "$EV_PY" "${EVAL[@]}" --head "${HEADS[$arm]}" --n 4 --warmup 0 \
    --out "$EV_RUN/eval/gate_$arm.json"
done
for arm in A B C; do
  check_gpus
  run "train_$arm" "$EV_PY" "${TRAIN[@]}" --init-from "${HEADS[$arm]}" --out "$EV_RUN/train_$arm"
  "$EV_PY" - "$EV_RUN/train_$arm/RUN_STATUS.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
if s['status']!='COMPLETED' or s['completed_updates']!=60:
    print('Training stopped before fixed budget; preserve all artifacts and do not auto-resume.',flush=True)
    sys.exit(3)
PY
done
# Full development set, same head step. First row is a matched timing warmup;
# the analyzer retains it for lossless and acceptance metrics, not timing.
for arm in A B C; do
  check_gpus
  EXTRA=()
  if [[ "$arm" == C ]]; then EXTRA=(--export-evidence-memory "$EV_RUN/C_normal_memory.pt"); fi
  run "eval_$arm" "$EV_PY" "${EVAL[@]}" --head "$EV_RUN/train_$arm/step60" --warmup 1 \
    "${EXTRA[@]}" --out "$EV_RUN/eval/$arm.json"
done
for mode in bypass zero time_shuffle; do
  check_gpus
  run "eval_C_$mode" "$EV_PY" "${EVAL[@]}" --head "$EV_RUN/train_C/step60" --warmup 1 \
    --evidence-intervention "$mode" --evidence-intervention-seed "$EV_SEED" \
    --out "$EV_RUN/eval/C_$mode.json"
done
check_gpus
run eval_C_wrong_video "$EV_PY" "${EVAL[@]}" --head "$EV_RUN/train_C/step60" --warmup 1 \
  --evidence-intervention wrong_video --evidence-donor-memory "$EV_RUN/C_normal_memory.pt" \
  --out "$EV_RUN/eval/C_wrong_video.json"
run analysis "$EV_PY" repro/evaluation/summarize_evidence_eval.py \
  --baseline "$EV_RUN/eval/A.json" --context "$EV_RUN/eval/B.json" --visual "$EV_RUN/eval/C.json" \
  --bypass "$EV_RUN/eval/C_bypass.json" --zero "$EV_RUN/eval/C_zero.json" \
  --time-shuffle "$EV_RUN/eval/C_time_shuffle.json" --wrong-video "$EV_RUN/eval/C_wrong_video.json" \
  --out "$EV_RUN/ANALYSIS.json"
echo "Completed fixed 180-update architecture pilot: $EV_RUN/ANALYSIS.json"
