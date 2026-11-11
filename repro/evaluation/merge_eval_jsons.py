#!/usr/bin/env python3
"""Concatenate sharded `eval_video_opd.py` outputs, asserting they are one run.

Protocol fields must agree; shard/output paths and GPU placement may differ.
Duplicate row sets and missing declared shards are rejected. Historical outputs
without protocol fields can still merge with other equally unversioned outputs,
but cannot be mixed with newly versioned evaluations.
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
from pathlib import Path


CONFIG_KEYS = ("rows", "head", "budget", "branch", "max_new", "ar", "warmup", "dtype",
               "n", "time_mode", "eos_stop", "target", "zero_visual", "visual_memory",
               "require_exact_time")
PROTOCOL_KEYS = ("version", "time_mode", "eos_stop", "visual_mask", "rows_sha256",
                 "target", "model_contract", "artifacts", "lossless_checked", "timing_scope",
                 "spec_prefill_definition", "tau_definition",
                 "effective_tokens_per_round_definition", "diagnostic_only",
                 "uses_cpu_or_disk_offload")


def _require_equal(items, key, label):
    values = {json.dumps(item.get(key), sort_keys=True) for item in items}
    if len(values) != 1:
        raise ValueError(f"shards disagree on {label}.{key}: {sorted(values)}")


def merge_results(parts, *, expect=0):
    """Validate loaded shard payloads and return a merged payload; no file I/O."""
    if not parts:
        raise ValueError("at least one shard is required")
    if expect < 0:
        raise ValueError("expect must be nonnegative")
    if any(not isinstance(part.get("rows"), list) or not isinstance(part.get("config"), dict)
           for part in parts):
        raise ValueError("each shard must contain evaluation rows and config")
    cfgs = [part["config"] for part in parts]
    for key in CONFIG_KEYS:
        _require_equal(cfgs, key, "config")
    has_protocol = ["protocol" in part for part in parts]
    if any(has_protocol) != all(has_protocol):
        raise ValueError("cannot mix versioned and legacy unversioned evaluation protocols")
    if all(has_protocol):
        protocols = [part["protocol"] for part in parts]
        if any(not isinstance(p, dict) for p in protocols):
            raise ValueError("protocol must be a mapping")
        for key in PROTOCOL_KEYS:
            _require_equal(protocols, key, "protocol")

    declared = [cfg.get("shards") for cfg in cfgs]
    if any(value is not None for value in declared):
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in declared) or len(set(declared)) != 1:
            raise ValueError("shards disagree on declared shard count")
        ids = [cfg.get("shard") for cfg in cfgs]
        if (any(not isinstance(i, int) or isinstance(i, bool) for i in ids)
                or len(parts) != declared[0] or sorted(ids) != list(range(declared[0]))):
            raise ValueError("missing, repeated, or out-of-range shard indices")

    rows = [row for part in parts for row in part["rows"]]
    seen, duplicate = set(), set()
    for row in rows:
        video = row.get("video")
        if not isinstance(video, str) or not video:
            raise ValueError("each evaluation row needs a nonempty video identity")
        if video in seen:
            duplicate.add(video)
        seen.add(video)
    if duplicate:
        raise ValueError(f"rows scored twice across shards: {sorted(duplicate)[:5]}")
    if expect and len(rows) != expect:
        raise ValueError(f"expected {expect} rows, got {len(rows)}")
    cfg = dict(cfgs[0])
    cfg.pop("shard", None)
    cfg["shards"] = len(parts)
    result = {"config": cfg, "rows": rows}
    if all(has_protocol):
        protocol = dict(parts[0]["protocol"])
        # GPU indices may legitimately differ across processes; do not present
        # the first shard's placement as the placement of the entire merge.
        protocol.pop("target_device_map", None)
        protocol["shard_target_device_maps"] = [p["protocol"].get("target_device_map")
                                                for p in parts]
        result["protocol"] = protocol
    if cfg.get("ar"):
        result["status"] = ("LOSSLESS_PASSED" if rows and all(r.get("lossless") is True
                                                                for r in rows)
                            else "LOSSLESS_FAILED")
    else:
        result["status"] = "LOSSLESS_NOT_CHECKED"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--expect", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if Path(a.out).exists():
        ap.error(f"output already exists; choose a new --out: {a.out}")
    parts = []
    for p in a.parts:
        with open(p) as stream:
            d = json.load(stream)
        print(f"{os.path.basename(p)}: {len(d['rows'])} rows")
        parts.append(d)
    result = merge_results(parts, expect=a.expect)
    result["config"]["merged_from"] = a.parts
    result["config"]["out"] = a.out
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as stream:
        json.dump(result, stream, indent=1, allow_nan=False)
    print(f"wrote {a.out}: {len(result['rows'])} rows; {result['status']}")
    if result["status"] == "LOSSLESS_FAILED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
