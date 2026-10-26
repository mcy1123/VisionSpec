#!/usr/bin/env bash
# Run the frozen evaluation matrix (amendments.jsonl, `analysis_plan_addendum`).
#
# Order matters for honesty: the dev120 disaster gate runs FIRST for every
# trained arm, so a catastrophic arm is caught before the expensive full-dev and
# frozen-test passes. Each pass is skipped when its JSON already exists, so the
# script is resumable and never overwrites an existing measurement.
set -uo pipefail
export RUN="${RUN:-/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955}"
GPU="${GPU:-7}"
PY="${PYTHON:-/home/mcy/.conda/envs/fin_research/bin/python}"
REPRO=/home/mcy/Projects/GLANCE/repro/scienceqa_image
LOG="$RUN/logs/eval_matrix.log"
I0=/mnt/local/mcy/GLANCE/checkpoints/head
S1="$RUN/train/seed20261013"
S2_SFT="/mnt/local2/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/train/seed20261014/sft"
S2_OPD="/mnt/local/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/train/seed20261014/opd"
mkdir -p "$RUN/eval" "$RUN/logs"

say() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

run_eval() {  # tag head split limit label
  local tag="$1" head="$2" split="$3" limit="$4" label="${5:-$3}"
  if [[ ! -d "$head" ]]; then say "SKIP $tag ($label): head missing ($head)"; return; fi
  if [[ -f "$RUN/eval/${tag}_${label}.json" ]]; then say "SKIP $tag ($label): exists"; return; fi
  say "eval $tag on $label (limit=$limit) head=$head"
  "$PY" -u "$REPRO/eval_sqa.py" --run "$RUN" --head "$head" --tag "$tag" \
      --split "$split" --split-label "$label" --gpu "$GPU" --budget 63 --branch-k 8 \
      --max-new 256 --max-pixels 768 --head-dtype float32 \
      --ar-cache "$RUN/eval/ar_$split.pt" --repeat-spec 2 --repeat-n 24 \
      --limit "$limit" --out "$RUN/eval/${tag}_${label}.json" \
      >> "$RUN/logs/eval_${tag}_${label}.log" 2>&1
  say "rc=$? $tag/$label"
}

# 1. the disaster gate, first, on the frozen dev120 subset
run_eval i0 "$I0" dev 120 dev120
for s in sft opd; do run_eval "${s}s20261013u30" "$S1/$s/step30" dev 120 dev120; done
for s in sft opd; do
  head="$S2_SFT/step30"; [[ "$s" == opd ]] && head="$S2_OPD/step30"
  run_eval "${s}s20261014u30" "$head" dev 120 dev120
done

# 2. intermediate trend at step60, dev120
for s in sft opd; do run_eval "${s}s20261013u60" "$S1/$s/step60" dev 120 dev120; done
for s in sft opd; do
  head="$S2_SFT/step60"; [[ "$s" == opd ]] && head="$S2_OPD/step60"
  run_eval "${s}s20261014u60" "$head" dev 120 dev120
done

# 3. full dev (240) at step120 -- primary dev endpoint
run_eval i0 "$I0" dev 0
for s in sft opd; do run_eval "${s}s20261013u120" "$S1/$s/step120" dev 0; done
for s in sft opd; do
  head="$S2_SFT/step120"; [[ "$s" == opd ]] && head="$S2_OPD/step120"
  run_eval "${s}s20261014u120" "$head" dev 0
done

# 4. frozen test (240) at step120 -- opened once, after the plan above is fixed
run_eval i0 "$I0" test 0
for s in sft opd; do run_eval "${s}s20261013u120" "$S1/$s/step120" test 0; done
for s in sft opd; do
  head="$S2_SFT/step120"; [[ "$s" == opd ]] && head="$S2_OPD/step120"
  run_eval "${s}s20261014u120" "$head" test 0
done

# 5. fixed-anchor coverage probe on dev120 for the final arms
run_cov() {  # tag head
  local tag="$1" head="$2"
  [[ -d "$head" ]] || { say "SKIP coverage $tag: head missing"; return; }
  [[ -f "$RUN/eval/coverage_${tag}.json" ]] && { say "SKIP coverage $tag: exists"; return; }
  say "coverage $tag"
  "$PY" -u "$REPRO/trace_tree_coverage.py" --run "$RUN" --head "$head" --tag "$tag" \
      --gpu "$GPU" --split dev --budget 63 --branch-k 8 --max-new 256 \
      --max-pixels 768 --anchors 2 --limit 120 --head-dtype float32 \
      --ar-cache "$RUN/eval/ar_dev.pt" --out "$RUN/eval/coverage_${tag}.json" \
      >> "$RUN/logs/coverage_${tag}.log" 2>&1
  say "rc=$? coverage $tag"
}
run_cov i0 "$I0"
run_cov sfts20261013u120 "$S1/sft/step120"
run_cov opds20261013u120 "$S1/opd/step120"
run_cov sfts20261014u120 "$S2_SFT/step120"
run_cov opds20261014u120 "$S2_OPD/step120"

say "evaluation matrix done"
