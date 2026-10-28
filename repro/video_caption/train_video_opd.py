#!/usr/bin/env python3
"""Explicit video-caption OPD pilot (three ranks, four logical slots, fresh start).

This is NOT the default entry and NOT a silent switch of the video recipe: the
historical `repro/train_video_official_kl.py` still refuses to run. This launcher
exists because the user explicitly asked to train the current pure-block method
on the long-generation video set, and the machine has three free >=40 GiB cards
rather than four.

What is reused unchanged:
  * the pure-block-native rollout/verification/loss in opd_official_kl;
  * the three-rank x four-logical-slot topology from the ScienceQA continuation,
    including its slot map {0:[0,3],1:[1],2:[2]} and ordered rank0 scoring.

What is new and explicit:
  * `run(..., allow_fresh_start=True)` (no migrated checkpoint needed);
  * `--modality video` through the existing qa_opd_fair video encoder;
  * `time_mode=legacy` by default, because the warm head was trained on the
    legacy (cache-clip) timeline. `real` is available but is a different input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))

from opd_official_kl import cli, three_card_trainer as tct  # noqa: E402


def ensure_ledger(path, case, updates, entry, seed, config_sha256):
    """Create the attempt ledger once; never widen an existing authorization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text())
        spec = data.get("cases", {}).get(case)
        if spec is None:
            raise ValueError(f"ledger exists without case {case!r}; refusing to add budget")
        expected = dict(updates=updates, entry=entry, seed=seed, config_sha256=config_sha256)
        mismatch = {k: (spec.get(k), v) for k, v in expected.items() if spec.get(k) != v}
        if mismatch:
            raise ValueError(f"ledger case {case!r} disagrees with this run: {mismatch}")
        return path
    data = dict(cases={case: dict(updates=updates, entry=entry, seed=seed,
                                  config_sha256=config_sha256)},
                max_attempts=updates + 5, attempts=[], completed=[])
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def parser():
    ap = cli.parser()
    for action in ap._actions:
        if action.dest == "parallel_rollout":
            action.choices = [3]
            action.default = 3
    ap.add_argument("--experiment-entry", default="video_caption_pure_block",
                    help="ledger entry name (ignored by the checkpoint identity)")
    ap.set_defaults(modality="video", time_mode="legacy", max_new_tokens=512,
                    max_seq_len=2048, updates=110, save_every=10,
                    lr=1e-5, lr_schedule="constant", max_pixels=768)
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.modality != "video":
        ap.error("this launcher is video-only; use repro/scienceqa_image/ for image")
    if not (args.pool and args.schedule and args.init_from and args.out):
        ap.error("--pool --schedule --init-from --out are required")
    metadata = tct.identity(args)
    prepared = dict(modality=args.modality, time_mode=args.time_mode,
                    updates=args.updates, max_new_tokens=args.max_new_tokens,
                    max_seq_len=args.max_seq_len, lr=args.lr, seed=args.seed,
                    slots=4, world_size=3, parallel_rollout=3,
                    init_from=args.init_from, pool=args.pool, schedule=args.schedule,
                    out=args.out, config_sha256=metadata["config_sha256"],
                    fresh_start=True, experiment_entry=args.experiment_entry)
    if args.dry_run:
        print(json.dumps(prepared, indent=2))
        return
    if not (args.experiment_budget and args.experiment_case):
        ap.error("real updates require --update-budget and --update-case")
    ensure_ledger(args.experiment_budget, args.experiment_case, args.updates,
                  args.experiment_entry, args.seed, metadata["config_sha256"])
    tct.run(args, allow_fresh_start=True)


if __name__ == "__main__":
    main()
