#!/usr/bin/env bash
# QA-SFT / QA-OPD training+memory/gradient/resume gate pilot.
# 3 real updates each, then a resume check that continues 2 updates from a
# non-final checkpoint, all on the frozen schedule with the frozen budget.
set -euo pipefail
R=/mnt/local/mcy/GLANCE/results/qa_opd_fair_20261013_195500
PY=/home/mcy/.conda/envs/fin_research/bin/python
GPU=${1:?usage: run_qa_pilot.sh <gpu>}
SEED=20261013
POOL=$R/dataset/qa_pool_df${SEED}.parquet
REC=$R/dataset/target_ar_df${SEED}.pt
C0=/mnt/local/mcy/GLANCE/checkpoints/head_video439_aligned/step1000
OUT=$R/training/seed_${SEED}
mkdir -p "$OUT/pilot"
cd /home/mcy/Projects/GLANCE

if [ "${2:-}" != "resume" ]; then
  $PY repro/qa_opd_fair/train_qa_sft.py --run $R --seed $SEED --gpu $GPU \
    --pool $POOL --records $REC --init-from $C0 --out "$OUT/pilot/sft" \
    --updates 3 --save-every 3 --log-every 1
  $PY repro/qa_opd_fair/train_qa_opd.py --run $R --seed $SEED --gpu $GPU \
    --pool $POOL --records $REC --init-from $C0 --out "$OUT/pilot/opd" \
    --updates 2 --save-every 2 --log-every 1
else
  $PY repro/qa_opd_fair/train_qa_sft.py --run $R --seed $SEED --gpu $GPU \
    --pool $POOL --records $REC --init-from $C0 --out "$OUT/pilot/sft_resume" \
    --updates 5 --save-every 5 --log-every 1 \
    --resume-from "$OUT/pilot/sft/step2" --resume-update 2
fi
