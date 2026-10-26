#!/usr/bin/env python3
"""Assemble `run_manifest.json` and a delivery index for direction B."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/repro")

import b_common as B  # noqa: E402


def file_entry(path, note=""):
    return {"path": path, "sha256": B.sha256_file(path) if os.path.exists(path)
            else None, "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--main", default="main")
    ap.add_argument("--outcome", default=None,
                    help="one of the four decision reads or blocked")
    ap.add_argument("--outcome-detail", default=None,
                    help="free-text explanation recorded next to outcome_type")
    ap.add_argument("--status", default="completed")
    ap.add_argument("--limitations", default="")
    ap.add_argument("--next-step", default="")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    main = os.path.join(run, a.main)

    analysis = os.path.join(main, "analysis")
    summary_path = os.path.join(analysis, "summary.json")
    summary = B.read_json(summary_path) if os.path.exists(summary_path) else {}
    gates = B.read_json(os.path.join(main, "gates.json")) \
        if os.path.exists(os.path.join(main, "gates.json")) else {}
    calls = B.read_json(os.path.join(main, "calls.json")) \
        if os.path.exists(os.path.join(main, "calls.json")) else {}
    sel = B.read_json(os.path.join(run, "stageA", "selection.json"))

    fam = summary.get("families", {})
    n_fam = fam.get("N", {})
    reads = {}
    inferred = "undecidable"
    detail = None
    if n_fam:
        rt = n_fam.get("mean_R_T", {})
        rk = n_fam.get("mean_R_K", {})
        diff = n_fam.get("mean_R_T_minus_K", {})
        prt = n_fam.get("mean_prob_R_T", {})
        prk = n_fam.get("mean_prob_R_K", {})
        pdiff = n_fam.get("mean_prob_R_T_minus_K", {})
        reads = {"mean_R_T_logprob": rt, "mean_R_K_logprob": rk,
                 "mean_R_T_minus_K_logprob": diff,
                 "mean_prob_R_T": prt, "mean_prob_R_K": prk,
                 "mean_prob_R_T_minus_K": pdiff,
                 "n_units_with_R_T": n_fam.get("n_units_with_R_T")}
        # The probability-space contrast is the bounded, tail-insensitive one,
        # so the automatic read keys off it; the log-probability contrast is
        # recorded beside it rather than used to choose the label.
        detail = (
            f"probability-space R_T={prt.get('point')} R_K={prk.get('point')} "
            f"R_T-R_K={pdiff.get('point')} CI[{pdiff.get('lo')},"
            f"{pdiff.get('hi')}]; log-probability R_T={rt.get('point')} "
            f"R_K={rk.get('point')} R_T-R_K={diff.get('point')} "
            f"CI[{diff.get('lo')},{diff.get('hi')}]")
        if pdiff.get("hi") is not None and pdiff["hi"] < 0:
            inferred = ("supports_temporal_specific_draft_weakness"
                        if (prt.get("point") or 1) < 1
                        else "draft_under_responds_to_temporal_relative_to_control")
        elif pdiff.get("lo") is not None and pdiff["lo"] > 0:
            inferred = "draft_over_responds_to_temporal_relative_to_control"
        elif (pdiff.get("lo") is not None and pdiff.get("hi") is not None
                and pdiff["lo"] <= 0 <= pdiff["hi"]):
            inferred = "no_stable_difference_observed"
        else:
            inferred = "undecidable"

    existing = os.path.join(run, "run_manifest.json")
    if not a.outcome and os.path.exists(existing):
        prev = B.read_json(existing)
        inferred = prev.get("outcome_type", inferred)
        detail = prev.get("outcome_detail", detail)

    tests_log = os.path.join(run, "tests.log")
    tests_passed = None
    if os.path.exists(tests_log):
        text = open(tests_log).read()
        tests_passed = {"ran": None, "ok": "\nOK" in text or text.rstrip().endswith("OK")}
        for line in text.splitlines():
            if line.startswith("Ran ") and " test" in line:
                tests_passed["ran"] = int(line.split()[1])
                break
    manifest = {
        "direction": "B",
        "task": "docs/tasks/parallel_20261012/02_task_B_temporal_response.md",
        "created": datetime.now().astimezone().isoformat(),
        "status": a.status,
        "outcome_type": a.outcome or inferred,
        "outcome_detail": a.outcome_detail or detail,
        "outcome_reads": reads,
        "family_N_units": sel["family_N_count"],
        "family_C_units": sel["family_C_count"],
        "evidence_paths": {
            "prereg": file_entry(os.path.join(run, "prereg.json")),
            "stageA_selection": file_entry(os.path.join(run, "stageA", "selection.json")),
            "stageA_candidates": file_entry(os.path.join(run, "stageA", "candidates.csv")),
            "stageA_audit": file_entry(os.path.join(run, "stageA", "audit_c6.json")),
            "conditions": file_entry(os.path.join(run, "conditions", "conditions.json")),
            "frame_manifest": file_entry(os.path.join(run, "frames", "frame_manifest.json")),
            "forward_records": file_entry(os.path.join(main, "forward_records.jsonl")),
            "gates": file_entry(os.path.join(main, "gates.json")),
            "calls": file_entry(os.path.join(main, "calls.json")),
            "per_unit": file_entry(os.path.join(analysis, "per_unit.csv")),
            "per_video": file_entry(os.path.join(analysis, "per_video.csv")),
            "summary": file_entry(summary_path),
            "per_offset_exploratory": file_entry(
                os.path.join(analysis, "per_offset_exploratory.csv")),
            "prefix_provenance": file_entry(
                os.path.join(analysis, "prefix_provenance.csv")),
            "prereg_v1_superseded": file_entry(
                os.path.join(run, "prereg_v1_superseded.json")),
            "prereg_amendments": file_entry(
                os.path.join(run, "prereg_amendments.json")),
            "report": file_entry(
                "/home/mcy/Projects/GLANCE/docs/reports/"
                "2026-10-12-parallel-B-temporal.md"),
            "pilot_log": file_entry(os.path.join(run, "pilot.log")),
            "main_log": file_entry(os.path.join(run, "main.log")),
            "tests_log": file_entry(os.path.join(run, "tests.log")),
        },
        "tests_passed": tests_passed,
        "ar_gate": {
            "note": ("this direction performs no decoding; the only token-level "
                     "gate is G2, reproducing C6's captured draft top-8 and "
                     "target argmax on the original condition"),
            "G1": gates.get("G1_summary"),
            "G2": gates.get("G2_summary"),
            "G3": gates.get("G3_summary"),
            "pilot": gates.get("pilot_gate"),
        },
        "call_accounting": calls,
        "limitations": a.limitations,
        "next_step": a.next_step,
        "requires_shared_update": True,
        "shared_files_touched": [],
    }
    B.write_json(os.path.join(run, "run_manifest.json"), manifest)
    print(json.dumps({"outcome_type": manifest["outcome_type"],
                      "reads": reads}, indent=1))
    print("wrote", os.path.join(run, "run_manifest.json"))


if __name__ == "__main__":
    main()
