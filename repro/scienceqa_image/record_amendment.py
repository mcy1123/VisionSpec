#!/usr/bin/env python3
"""Register the uniform resource-budget amendment BEFORE any formal training.

The pre-registration (`prereg.json` / `experiment_protocol.json`) planned 300
updates per arm per seed with a checkpoint grid of 30/100/200/300. The task
allows -- and requires -- a single, uniform, verifiable budget revision when local
compute limits make that plan unsafe, provided it is written down BEFORE the
formal runs and applies to BOTH arms of BOTH seeds. This script writes that
record, with the measurements it is based on, and refuses to run if formal
training has already started (so the amendment cannot be a post-hoc excuse).

    python record_amendment.py --run <run>

The amendment reduces updates, not the comparison's validity: both arms still
start from the same I0, read the same frozen schedule, use the same optimizer,
precision, prompt, verifier and evaluation, and the primary endpoint is still
"final checkpoint of OPD minus final checkpoint of SFT, paired by seed".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time


def df(path):
    out = subprocess.run(["df", "-B1", path], capture_output=True, text=True).stdout
    line = out.strip().splitlines()[-1].split()
    return {"filesystem": line[0], "size_bytes": int(line[1]),
            "used_bytes": int(line[2]), "free_bytes": int(line[3]),
            "use_percent": line[4]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--qa-per-update", type=int, default=4)
    ap.add_argument("--checkpoint-grid", default="30,60,120")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run = pathlib.Path(a.run)

    # refuse if any formal training output already exists
    started = sorted(glob.glob(str(run / "train" / "seed*" / "*" / "log.jsonl")))
    if started:
        print(f"REFUSING: formal training already started ({started[:3]})", file=sys.stderr)
        return 2

    debug = run / "logs" / "ar_replay_diagnosis.json"
    debug_meta = {}
    bd = run / "training" / "seed_20261013" / "data" / "build_log.jsonl"
    if bd.exists():
        rows = [json.loads(l) for l in bd.read_text().splitlines() if l.strip()]
        if rows:
            debug_meta["ar_build_records_measured"] = len(rows)
            debug_meta["ar_build_seconds_per_record_mean"] = round(
                sum(r["seconds"] for r in rows) / len(rows), 2)
            debug_meta["ar_build_seconds_per_record_max"] = round(
                max(r["seconds"] for r in rows), 2)

    disks = {p: df(p) for p in ("/mnt/local", "/mnt/local2", "/mnt/local3")}
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()

    rec = {
        "kind": "resource_budget_amendment",
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "registered_before_formal_training": True,
        "applies_to": "BOTH arms (I-SFT, I-OPD) x BOTH seeds (20261013, 20261014); "
                      "it is not an arm-specific reduction",
        "original_plan": {"updates": 300, "qa_per_update": 4,
                          "qa_exposures_per_arm_per_seed": 1200,
                          "checkpoint_grid": [30, 100, 200, 300]},
        "amended_plan": {"updates": a.updates, "qa_per_update": a.qa_per_update,
                         "qa_exposures_per_arm_per_seed": a.updates * a.qa_per_update,
                         "checkpoint_grid": [int(x) for x in a.checkpoint_grid.split(",")],
                         "train_pool_records_drawn": a.updates * a.qa_per_update,
                         "primary_endpoint": f"step{a.updates} (the final checkpoint)"},
        "unchanged_by_this_amendment": [
            "start point I0 (the same image warm head for every arm and both seeds)",
            "the frozen schedule permutation (identical for both arms AND both seeds)",
            "the QA position contract, the objective, gamma=0.8, the anchor rule",
            "lr 3e-5 constant, AdamW (0.9,0.95), wd 0, clip 1.0, fp32 CPU master weights",
            "fp32 target, bf16 head, max_pixels 768 train and eval",
            "prompt template, gold isolation, b63k8 verifier, EOS immediate",
            "max_new_tokens 256 (from the length gate) and the lossless gate",
            "the dev/test sets, the image-group bootstrap and the claim limits",
        ],
        "reasons": [],
        "measurements": {"ar_build": debug_meta, "disk_at_registration": disks,
                         "gpu_at_registration": gpus,
                         "free_bytes_total_mnt": sum(d["free_bytes"] for d in disks.values()),
                         "free_bytes_total_mnt_gib": round(
                             sum(d["free_bytes"] for d in disks.values()) / 2**30, 1)},
    }

    r = rec["reasons"]
    r.append({
        "reason": "fp32 target autoregressive labelling is the dominant single cost",
        "evidence": (f"measured {debug_meta.get('ar_build_seconds_per_record_mean', '?')} s "
                     f"per record on real ScienceQA rows (fp32 target, deployment decode "
                     f"path, mean answer length 153 tokens)"),
        "arithmetic": ("300 updates x 4 QA = 1200 labelled train records -> about "
                       f"{1200 * debug_meta.get('ar_build_seconds_per_record_mean', 13) / 3600:.1f} h, "
                       "plus 480 dev/test records -> about "
                       f"{480 * debug_meta.get('ar_build_seconds_per_record_mean', 13) / 3600:.1f} h, "
                       "of exclusive fp32 GPU before a single update; the amended budget "
                       f"needs {a.updates * a.qa_per_update} + 480 = "
                       f"{a.updates * a.qa_per_update + 480} records"),
    })
    r.append({
        "reason": "all three permitted /mnt volumes are 98-99% full",
        "evidence": {k: {"free_gib": round(v["free_bytes"] / 2**30, 1),
                         "use_percent": v["use_percent"]} for k, v in disks.items()},
        "arithmetic": ("a full head + fp32 master + Adam checkpoint measures about 16 GiB; "
                       "the original grid is 4 checkpoints x 4 arm-runs = 16 checkpoints "
                       "= about 256 GiB against "
                       f"{rec['measurements']['free_bytes_total_mnt_gib']} GiB free in "
                       "total across the three shared volumes, the largest single volume "
                       "holding only about 193 GiB. The amended grid is 3 x 4 = 12 "
                       "checkpoints = about 192 GiB, split across two volumes."),
    })
    r.append({
        "reason": "no idle accelerator can hold the fp32 target, which itself needs ~33 GiB",
        "evidence": ("8 visible GPUs; at registration the largest free block was on GPU7 "
                     "(a card this session already holds), and every other card either had "
                     "less than 33 GiB free or was busy with another user's job"),
        "consequence": ("wall-clock must be assumed to be shared and contended, so the "
                        "budget must leave room for the evaluation sweep rather than "
                        "spending all of it on labelling"),
    })

    if a.dry_run:
        rec["dry_run"] = True
        print(json.dumps(rec, indent=2, ensure_ascii=False))
        return 0

    with open(run / "amendments.jsonl", "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # keep an exact copy of the pre-training data/source seal
    seal = {
        "registered_at": rec["registered_at"],
        "amended_plan": rec["amended_plan"],
        "dataset_readiness_sha256": _sha(run / "dataset" / "readiness_gate.json"),
        "prereg_sha256": _sha(run / "prereg.json"),
        "experiment_protocol_sha256": _sha(run / "experiment_protocol.json"),
        "length_pilot_sha256": _sha(run / "preflight" / "length_pilot.json"),
        "image_binding_sha256": _sha(run / "preflight" / "image_processor_binding.json"),
        "ar_lossless_gate_sha256": _sha(run / "preflight" / "ar_lossless_gate.json"),
        "source_sha256": {str(p.relative_to(pathlib.Path(
            "/home/mcy/Projects/GLANCE"))): _sha(p) for p in sorted(
                (pathlib.Path("/home/mcy/Projects/GLANCE/repro/scienceqa_image")).glob("*"))
            if p.is_file()},
    }
    json.dump(seal, open(run / "training_seal.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"amended_plan": rec["amended_plan"],
                      "reasons": [x["reason"] for x in r]}, indent=2,
                     ensure_ascii=False))
    print("wrote", run / "amendments.jsonl", "and", run / "training_seal.json")
    return 0


def _sha(p):
    import hashlib
    p = pathlib.Path(p)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
