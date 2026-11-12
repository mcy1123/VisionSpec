#!/usr/bin/env python3
"""Explicit fixed-spatial long-input SPEC/AR engineering gate, natural EOS.

Forwards to the unchanged eval_video_opd with the actual AutoProcessor factory
profiled process-locally. No max-sequence cap is imposed by this entrypoint.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from local_h100.processor_profile_runtime import load_profile, profiled_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--processor-profile", required=True)
    custom, remaining = parser.parse_known_args()
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument("--time-mode", default="legacy")
    check.add_argument("--eos-stop", default="legacy")
    check.add_argument("--dtype", default="bfloat16")
    check.add_argument("--max-new", type=int, default=1024)
    check.add_argument("--ar", action="store_true")
    check.add_argument("--preflight-only", action="store_true")
    check.add_argument("--export-evidence-memory")
    check.add_argument("--evidence-donor-memory")
    flags, _ = check.parse_known_args(remaining)
    if flags.time_mode != "real" or flags.eos_stop != "immediate":
        parser.error("long-input gate requires --time-mode real --eos-stop immediate")
    if flags.dtype != "float32" or flags.max_new != 1024:
        parser.error("long-input gate requires --dtype float32 --max-new 1024")
    if not flags.ar and not flags.preflight_only:
        parser.error("long-input gate requires --ar to verify exact SPEC=AR")
    if flags.export_evidence_memory or flags.evidence_donor_memory:
        parser.error("cross-input memory banks need a separate profile-bound contract")
    profile = load_profile(custom.processor_profile)
    from evaluation import eval_video_opd
    old_argv = sys.argv
    try:
        sys.argv = [str(REPRO / "evaluation/eval_video_opd.py"), *remaining]
        with profiled_evaluation(eval_video_opd, profile):
            eval_video_opd.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
