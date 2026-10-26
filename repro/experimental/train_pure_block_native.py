#!/usr/bin/env python3
"""Compatibility alias for the canonical pure-block ScienceQA entry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opd_official_kl.cli import main

if __name__ == '__main__':
    main()
