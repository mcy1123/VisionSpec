#!/usr/bin/env python3
"""Explicit local storage adapter for the existing real-video OPD trainer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
from shared import training_resume

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
ORIGINAL_VALIDATE = training_resume.validate_output


def validate_output(path):
    resolved = Path(path).resolve()
    root = ROOT.resolve()
    if root.exists() and resolved != root and resolved.is_relative_to(root):
        return
    ORIGINAL_VALIDATE(path)


def main():
    # Process-local explicit extension. Historical trainer defaults stay intact.
    training_resume.validate_output = validate_output
    from video_caption.train_video_9k import main as train
    train()


if __name__ == '__main__':
    main()
