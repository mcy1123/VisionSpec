#!/usr/bin/env python3
"""Existing time-off evidence OPD plus the audited first-response boundary adapter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
from local_h100 import train_video_opd as base
from local_h100.train_video_evidence_opd import evidence_runtime
from local_h100.evidence_opd_prompt_boundary import prompt_boundary_runtime


def main(argv=None):
    from video_caption import train_video_9k as trainer
    args = trainer.parser().parse_args(argv)
    base.validate_execution(args)
    with base.local_runtime(trainer), evidence_runtime(trainer), prompt_boundary_runtime(trainer):
        return trainer.main(argv)


if __name__ == '__main__':
    main()
