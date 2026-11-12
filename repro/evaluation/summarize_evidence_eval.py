#!/usr/bin/env python3
"""Paired, same-input architecture pilot analysis; no model execution.

Bootstrap units are videos. Timing excludes the same declared warmup videos in
all arms. A single run remains diagnostic timing, even when its CI excludes 1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SAME_CONFIG = ("target", "dtype", "time_mode", "eos_stop", "max_new", "budget", "branch")


def indexed(payload):
    if payload.get("status") == "LOSSLESS_FAILED":
        raise ValueError("refusing failed lossless run")
    records = payload.get("rows", [])
    if not records:
        raise ValueError("empty evaluation")
    result = {}
    for row in records:
        key = row["video"]
        if key in result:
            raise ValueError(f"duplicate video {key}")
        if row.get("lossless") is not True or row.get("spec_ids") != row.get("ar_ids"):
            raise ValueError(f"{key}: require explicit token-exact AR/SPEC evidence")
        if not row.get("input_identity") or not row.get("ar_ids"):
            raise ValueError(f"{key}: missing input identity or AR tokens")
        if len(row["spec_ids"]) != row.get("gen_tokens") or row.get("rounds", 0) < 1:
            raise ValueError(f"{key}: inconsistent output token/round counts")
        for field in ("tau", "spec_wall_s", "spec_end_to_end_s"):
            if not math.isfinite(float(row[field])) or float(row[field]) <= 0:
                raise ValueError(f"{key}: invalid {field}")
        result[key] = row
    return result


def paired_inputs(payloads):
    names = list(payloads)
    maps = {name: indexed(payload) for name, payload in payloads.items()}
    ref = names[0]
    keys = sorted(maps[ref])
    for name in names[1:]:
        for field in SAME_CONFIG:
            values = [payloads[n].get("config", {}).get(field) for n in (ref, name)]
            if any(v is None for v in values) or values[0] != values[1]:
                raise ValueError(f"{name}: different/missing config {field}")
        if set(maps[name]) != set(keys):
            raise ValueError(f"{name}: video sets differ")
        for video in keys:
            a, b = maps[ref][video], maps[name][video]
            for field in ("input_identity", "ar_ids", "warmup"):
                if field not in a or field not in b or a[field] != b[field]:
                    raise ValueError(f"{name}/{video}: different/missing {field}")
    return maps, keys


def validate_arms(payloads):
    reference = payloads["C"]["protocol"]["evidence"]
    for key in ("dim", "max_tokens", "spatial_bins", "feature_layer", "feature_protocol"):
        if reference.get(key) is None or payloads["B"].get("protocol", {}).get("evidence", {}).get(key) != reference[key]:
            raise ValueError(f"B/C: evidence capacity/source mismatch: {key}")
    for name, payload in payloads.items():
        protocol = payload.get("protocol", {})
        evidence = protocol.get("evidence", {})
        expected_mode = {"A": "none", "B": "text"}.get(name, "visual")
        expected_intervention = "normal" if name in ("A", "B", "C") else name
        if evidence.get("mode") != expected_mode or evidence.get("intervention") != expected_intervention:
            raise ValueError(f"{name}: wrong evidence mode/intervention for this comparison")
        if name not in ("A", "B", "C"):
            source = payloads["C"].get("protocol", {}).get("artifacts", {})
            actual = protocol.get("artifacts", {})
            for key in ("head_config_sha256", "head_weights_sha256"):
                if not source.get(key) or source.get(key) != actual.get(key):
                    raise ValueError(f"{name}: intervention must use the identical C checkpoint")


def validate_training_identity(payloads):
    identities = {name: payloads[name]["protocol"].get("artifacts", {}).get("checkpoint_identity")
                  for name in ("A", "B", "C")}
    if not any(value is not None for value in identities.values()):
        return {"verified": False, "reason": "checkpoint sidecars absent; training budget not verified"}
    if any(not isinstance(value, dict) for value in identities.values()):
        raise ValueError("cannot mix checkpoints with and without training identity sidecars")
    fields = ("schema", "optimizer_step", "data_cursor", "qa_exposures", "common_initial_head_weights_sha256",
              "pool_sha256", "schedule_sha256", "target_config_sha256", "seed", "lr",
              "qa_per_update", "time_mode", "loss_normalization", "e2e_tv_weight", "optimizer",
              "max_new_tokens", "max_seq_len")
    for field in fields:
        values = [identities[name].get(field) for name in ("A", "B", "C")]
        if any(value is None for value in values) or any(value != values[0] for value in values[1:]):
            raise ValueError(f"training identity mismatch/missing: {field}")
    reference = identities["A"]
    if reference["schema"] != "video_opd_checkpoint_identity_v1":
        raise ValueError("unsupported checkpoint training identity schema")
    if reference["optimizer_step"] != reference["data_cursor"] or reference["optimizer_step"] < 1:
        raise ValueError("checkpoint step/cursor mismatch")
    if reference["qa_exposures"] != reference["optimizer_step"] * reference["qa_per_update"]:
        raise ValueError("checkpoint QA exposure count mismatch")
    for precision in ("target_dtype", "head_dtype", "master_dtype"):
        values = [identities[name].get("precision", {}).get(precision) for name in ("A", "B", "C")]
        if any(value is None for value in values) or any(value != values[0] for value in values[1:]):
            raise ValueError(f"training precision mismatch/missing: {precision}")
    for name in ("B", "C"):
        if identities[name].get("precision", {}).get("evidence_parameter_dtype") != "float32":
            raise ValueError(f"{name}: evidence parameters must train in float32")
    return {"verified": True, "optimizer_step": reference["optimizer_step"],
            "common_initial_head_weights_sha256": reference["common_initial_head_weights_sha256"],
            "pool_sha256": reference["pool_sha256"], "schedule_sha256": reference["schedule_sha256"]}


def interval(samples):
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize(payloads, *, resamples=10000, seed=20261112):
    if resamples < 1 or not {"A", "B", "C"}.issubset(payloads):
        raise ValueError("need A/B/C and positive bootstrap resamples")
    validate_arms(payloads)
    training_identity = validate_training_identity(payloads)
    maps, keys = paired_inputs(payloads)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(resamples, len(keys)))
    timing_indices = np.array([i for i, k in enumerate(keys) if not maps["A"][k]["warmup"]])
    timing_draws = (rng.integers(0, len(timing_indices), size=(resamples, len(timing_indices)))
                    if len(timing_indices) else None)
    arrays, arms = {}, {}
    for name, records in maps.items():
        rs = [records[k] for k in keys]
        arrays[name] = {
            "tau": np.array([r["tau"] for r in rs], dtype=float),
            "effective_tokens_per_round": np.array([r["gen_tokens"] / r["rounds"] for r in rs]),
            "spec_wall_s": np.array([r["spec_wall_s"] for r in rs]),
            "spec_end_to_end_s": np.array([r["spec_end_to_end_s"] for r in rs]),
        }
        arms[name] = {
            "n_videos": len(rs), "n_timing_videos": len(timing_indices),
            "mean_tau": float(arrays[name]["tau"].mean()),
            "mean_effective_tokens_per_round": float(arrays[name]["effective_tokens_per_round"].mean()),
            "total_output_tokens_over_total_rounds": sum(r["gen_tokens"] for r in rs) / sum(r["rounds"] for r in rs),
            "mean_output_tokens": float(np.mean([r["gen_tokens"] for r in rs])),
            "hit_eos": sum(bool(r.get("hit_eos")) for r in rs),
            "truncated": sum(bool(r.get("truncated")) for r in rs),
            "all_lossless": True,
        }
    pairs = [("C", "A"), ("C", "B"), ("B", "A")]
    pairs += [("C", name) for name in payloads if name not in ("A", "B", "C")]
    comparisons = {}
    for new, old in pairs:
        record = {}
        for metric in ("tau", "effective_tokens_per_round"):
            delta = arrays[new][metric] - arrays[old][metric]
            record[metric] = {"paired_mean_difference": float(delta.mean()),
                              "ci95": interval(delta[draws].mean(axis=1)),
                              "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum())}
        if timing_draws is not None and old != "wrong_video":
            for metric in ("spec_wall_s", "spec_end_to_end_s"):
                a, b = arrays[old][metric][timing_indices], arrays[new][metric][timing_indices]
                record[metric] = {"speedup_old_over_new": float(a.sum() / b.sum()),
                                  "ci95": interval(a[timing_draws].sum(axis=1) / b[timing_draws].sum(axis=1))}
        if old == "wrong_video":
            record["timing_omitted"] = "Donor-memory intervention is a mechanism probe, not a deployment speed comparison."
        comparisons[f"{new}_minus_{old}"] = record
    return {
        "schema": "evidence_pilot_analysis_v1", "diagnostic_only": True,
        "bootstrap_unit": "source_video", "resamples": resamples, "seed": seed,
        "training_identity": training_identity,
        "timing_scope": "paired evaluation fields, excluding declared warmup rows",
        "limitations": ["One seed does not measure training randomness.",
                        "Video bootstrap does not remove shared-load or execution-order timing bias.",
                        "Repeated use of development videos is not independent final confirmation."],
        "arms": arms, "comparisons": comparisons,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("baseline", "context", "visual"):
        ap.add_argument(f"--{flag}", required=True)
    for flag in ("bypass", "zero", "time-shuffle", "wrong-video"):
        ap.add_argument(f"--{flag}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args(argv)
    files = {"A": args.baseline, "B": args.context, "C": args.visual}
    files.update({name: getattr(args, name) for name in ("bypass", "zero", "time_shuffle", "wrong_video") if getattr(args, name)})
    payloads = {name: json.loads(Path(path).read_text()) for name, path in files.items()}
    result = summarize(payloads, resamples=args.resamples, seed=args.seed)
    result["source_files"] = files
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
