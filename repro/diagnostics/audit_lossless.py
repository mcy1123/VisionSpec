#!/usr/bin/env python3
"""Run the repository's fp32 lossless audit with this box's conv fix applied.

`experiments/audit_lossless.py` is the paper's own losslessness gate: it
re-decodes every prompt with no drafter at all and requires the speculative
walk to reproduce it token for token, at fp32, where a bf16 tie cannot excuse a
mismatch. Nothing about the audit changes here -- only the vision patch-embed
workaround, which is applied globally and verified at import.

    python audit_lossless.py --head <ckpt> --data $GLANCE_ROOT/data/vl_eval \
        --tasks chartqa --n 20 --out $GLANCE_ROOT/results/audit.json
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import os
import runpy
import sys

from shared import common

common.use_fast_patch_embed()

REPO = os.path.join(common.WORKSPACE, "GLANCE_repo")
EXPERIMENTS = os.path.join(REPO, "experiments")
# `experiments/audit_lossless.py` does `from main_table import load_task`, which
# only resolves when its own directory is importable -- true when it is run as
# a script from there, false under runpy. Without this the audit dies on import
# after a multi-hour training run, which is precisely what it is meant to gate.
for path in (REPO, EXPERIMENTS):
    if path not in sys.path:
        sys.path.insert(0, path)
target = os.path.join(EXPERIMENTS, "audit_lossless.py")

# Default the target to the local checkpoint so the audit never reaches the
# network mid-run; an explicit --target still wins.
if "--target" not in sys.argv:
    sys.argv += ["--target", common.TARGET_PATH]

runpy.run_path(target, run_name="__main__")
