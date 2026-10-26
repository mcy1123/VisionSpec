#!/usr/bin/env python3
"""Read the training log and say whether the head is learning the right thing.

Loss alone cannot answer that. The paper's own checkpoint criterion is
"held-out top-1 accuracy at each offset" (App.D), and it quotes 0.78 for a
well-trained adapter's top-1@1 -- so offset-wise accuracy is the quantity with
an external reference point, and it is the one to watch.

Two failure modes it separates:

* offset-1 accuracy flat near chance -> the head is not learning at all
  (data, learning rate or objective problem), and no amount of waiting helps;
* offset-1 high but later offsets near chance -> the head learned only the
  first prediction, which is the fixed-anchor regime rather than the
  variable-prefix one, i.e. the reveal schedule is not doing its job.

    python watch_training.py --log $GLANCE_ROOT/logs/train_head.jsonl
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import argparse
import json
import os

from shared import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=os.path.join(common.ROOT, "logs", "train_head.jsonl"))
    parser.add_argument("--total-steps", type=int, default=1000,
                        help="16K rows / 2 ranks / accumulation 8")
    parser.add_argument("--tail", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"no training log yet at {args.log}")
        return

    rows = [json.loads(l) for l in open(args.log) if l.strip()]
    if not rows:
        print("training log is empty")
        return

    # The log is append-only, so a restarted run stacks on top of the previous
    # one. Read only the last contiguous run, otherwise step numbers repeat and
    # the loss trend spans two different training runs.
    starts = [i for i, r in enumerate(rows) if r["step"] == 1]
    if len(starts) > 1:
        print(f"note: {len(starts)} runs in this log; reading the last one "
              f"({len(rows) - starts[-1]} steps)")
        rows = rows[starts[-1]:]

    last = rows[-1]
    step = last["step"]
    pct = 100.0 * step / max(args.total_steps, 1)
    print(f"{step}/{args.total_steps} steps ({pct:.0f}%), {last['elapsed_min']:.1f} min elapsed")
    if step > 0 and last["elapsed_min"] > 0:
        per_step = last["elapsed_min"] / step
        print(f"  {per_step*60:.1f} s/step -> {per_step*(args.total_steps-step)/60:.1f} h remaining")

    losses = [r["loss"] for r in rows]
    window = min(10, len(losses))
    print(f"  loss: first{window} {sum(losses[:window])/window:.4f} -> "
          f"last{window} {sum(losses[-window:])/window:.4f}")

    print(f"  last {args.tail} steps:")
    for r in rows[-args.tail:]:
        per = r["per_offset_top1"]
        print(f"    step {r['step']:4d}  loss {r['loss']:.4f}  "
              f"off1 {per[1]:.3f}  off2 {per[2]:.3f}  off4 {per[4]:.3f}  "
              f"off8 {per[8]:.3f}  off15 {per[15]:.3f}")

    per = last["per_offset_top1"]
    print("\n  versus the paper's reference points:")
    print(f"    offset-1 top-1 = {per[1]:.3f}   (paper quotes 0.78 for a trained adapter)")
    print(f"    offset-4 top-1 = {per[4]:.3f}")
    print(f"    offset-15 top-1 = {per[15]:.3f}")

    if step < 50:
        print("  VERDICT: too early to judge")
    elif per[1] < 0.45:
        print("  VERDICT: offset-1 near chance -- the head is not learning; "
              "diagnose before letting the epoch run out")
    elif per[4] < 0.08:
        print("  VERDICT: only the first offset is learned -- the reveal "
              "schedule may not be reaching later offsets")
    elif per[1] > 0.65 and per[4] > 0.25:
        print("  VERDICT: learning across offsets, in the paper's neighbourhood")
    else:
        print("  VERDICT: learning, but below the paper's level so far")


if __name__ == "__main__":
    main()
