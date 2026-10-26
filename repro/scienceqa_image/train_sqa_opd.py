#!/usr/bin/env python3
"""Canonical ScienceQA image OPD: approximate pure block, four GPU rollout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opd_official_kl.cli import main

if __name__ == '__main__':
    main()
