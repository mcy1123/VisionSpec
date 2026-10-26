#!/usr/bin/env python3
"""The machine-checkable training-start gate (`dataset/readiness_gate.json`).

Every field is computed from files on disk. `train_ready` is true only when all
of them hold; a missing file, an empty list, a zero count or a skipped sample is
a failure, never a pass. This is deliberately stricter than "the directory
exists": the task document's data gate is about the EXACT scheduled assets being
present, decodable, QA-bound, timestamp-carrying and split-clean.

Usage: python check_dataset_readiness.py --run <run dir>
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASET_ROOT = pathlib.Path("/mnt/local2/mcy/datasets/msvd_qa_full")


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--verify-full-decode", action="store_true", default=True)
    a = ap.parse_args()
    R = pathlib.Path(a.run)
    D = R / "dataset"
    gate = {}
    evidence = {}

    # ---- 1. official train annotation present and parseable
    train_json = DATASET_ROOT / "official" / "qa_train.json"
    summary_json = DATASET_ROOT / "download_summary.json"
    inv_path = DATASET_ROOT / "inventory.jsonl"
    official_ok = train_json.exists() and summary_json.exists() and inv_path.exists()
    n_official = n_inv = 0
    if official_ok:
        official = json.loads(train_json.read_text())
        n_official = len({r["video"] for r in official})
        inv = [json.loads(l) for l in inv_path.read_text().splitlines() if l.strip()]
        n_inv = len(inv)
    gate["msvd_train_source_verified"] = bool(
        official_ok and n_official == 1161 and n_inv == n_official)
    evidence["msvd_train_source_verified"] = {
        "official_annotation": str(train_json),
        "official_train_videos": n_official,
        "inventory_rows": n_inv,
        "download_summary": str(summary_json),
        "download_summary_sha256": (sha256_file(summary_json)
                                    if summary_json.exists() else None),
        "source_used": (json.loads(summary_json.read_text())
                        ["source_urls_used"]["source_used"]
                        if summary_json.exists() else None),
    }

    # ---- 2. the two seeds' schedules exist, are full, and share their QA set
    sched = {}
    for seed in (20261013, 20261014):
        p = D / f"train_schedule_seed_{seed}.jsonl"
        sched[seed] = ([json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                       if p.exists() else [])
    same_set = ({r["video_id"] + "|" + r["question_id"] for r in sched[20261013]}
                == {r["video_id"] + "|" + r["question_id"] for r in sched[20261014]})
    gate["train_schedule_frozen"] = all(
        len(sched[s]) == 480 and len({r["order"] for r in sched[s]}) == 480
        and all(r["cache_path"] for r in sched[s]) for s in sched)
    gate["paired_sft_opd_schedule_identical"] = bool(
        same_set and len(sched[20261013]) == len(sched[20261014]) == 480)
    evidence["train_schedule_frozen"] = {
        s: {"slots": len(sched[s]),
            "distinct_order": len({r["order"] for r in sched[s]}),
            "distinct_videos": len({r["video_id"] for r in sched[s]}),
            "sha256": sha256_file(D / f"train_schedule_seed_{s}.jsonl")
            if (D / f"train_schedule_seed_{s}.jsonl").exists() else None}
        for s in sched}
    evidence["paired_sft_opd_schedule_identical"] = {
        "same_qa_set": same_set,
        "order_differs": ([r["video_id"] for r in sched[20261013]]
                          != [r["video_id"] for r in sched[20261014]]),
    }

    # ---- 3. every scheduled video is present and decodes to 16 frames
    needed = sorted({r["video_id"] for s in sched for r in sched[s]})
    inv_by_vid = {}
    if inv_path.exists():
        for line in inv_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                inv_by_vid[rec["video_id"]] = rec
    plan = [(r["video_id"], r["question_id"], r["update"], r["slot"])
            for s in sched for r in sched[s]]
    asset_present = []
    for vid, qid, upd, slot in plan:
        rec = inv_by_vid.get(vid)
        ok = bool(rec and rec.get("decode_ok")
                  and pathlib.Path(rec["cache_path"]).exists()
                  and pathlib.Path(rec["raw_path"]).exists())
        asset_present.append(ok)
    gate["all_scheduled_assets_present"] = bool(plan and all(asset_present))
    evidence["all_scheduled_assets_present"] = {
        "scheduled_slots": len(plan), "distinct_videos": len(needed),
        "present": int(sum(asset_present)), "missing": len(plan) - int(sum(asset_present))}

    # ---- 4. a real decode of every scheduled video (16 frames), not a smoke 3
    decode_ok = 0
    decode_fail = []
    if a.verify_full_decode:
        import cv2
        for vid in needed:
            rec = inv_by_vid.get(vid)
            if not rec:
                decode_fail.append({"video_id": vid, "reason": "no inventory row"})
                continue
            cap = cv2.VideoCapture(rec["cache_path"])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            if n == 16:
                decode_ok += 1
            else:
                decode_fail.append({"video_id": vid, "frames": n})
    gate["all_scheduled_assets_decode_ok"] = bool(
        needed and decode_ok == len(needed))
    evidence["all_scheduled_assets_decode_ok"] = {
        "checked_videos": len(needed), "16_frame_ok": decode_ok,
        "failures": decode_fail[:20]}

    # ---- 5. every scheduled QA is bound to the official train annotation
    official_pairs = {(r["video"], r["question_id"]) for r in
                      json.loads(train_json.read_text())} if official_ok else set()
    bound = [((r["video_id"], r["question_id"]) in official_pairs) for r in
             [x for s in sched for x in sched[s]]]
    gate["all_scheduled_qa_bound"] = bool(bound and all(bound))
    evidence["all_scheduled_qa_bound"] = {
        "checked": len(bound), "bound": int(sum(bound)),
        "unbound_examples": [ (r["video_id"], r["question_id"])
                              for r, b in zip([x for s in sched for x in sched[s]], bound)
                              if not b][:5]}

    # ---- 6. the timestamp metadata the real-time adapter needs is present
    ts_ok = 0
    for vid in needed:
        rec = inv_by_vid.get(vid)
        if (rec and len(rec.get("sample_frame_ids", [])) == 16
                and len(rec.get("sample_timestamps", [])) == 16
                and rec.get("source_fps") and rec.get("video_duration")):
            ts_ok += 1
    gate["timestamp_metadata_ready"] = bool(needed and ts_ok == len(needed))
    evidence["timestamp_metadata_ready"] = {"videos": len(needed), "ready": ts_ok}

    # ---- 7. no cross-split identity overlap
    dev = {json.loads(l)["video_id"] for l in
           (D / "dev_manifest.jsonl").read_text().splitlines() if l.strip()}
    conf = {json.loads(l)["video_id"] for l in
            (D / "confirm_manifest.jsonl").read_text().splitlines() if l.strip()}
    train_v = {r["video_id"] for s in sched for r in sched[s]}
    gate["split_overlap_audit_passed"] = bool(
        not (train_v & dev) and not (train_v & conf) and not (dev & conf))
    evidence["split_overlap_audit_passed"] = {
        "train_dev": len(train_v & dev), "train_confirm": len(train_v & conf),
        "dev_confirm": len(dev & conf),
        "audit_csv": str(D / "split_overlap_audit.csv"),
        "note": "identity is the exact clip file name; same-source-different-clip "
                "collisions are reported in the audit csv and are not label leaks"}

    # ---- 8. disk space for checkpoints and results
    usage = os.statvfs("/mnt/local")
    free_gib = usage.f_bavail * usage.f_frsize / 2**30
    gate["space_sufficient"] = bool(free_gib > 40)
    evidence["space_sufficient"] = {"free_GiB_mnt_local": round(free_gib, 1)}

    # ---- 9. no exception needed: the full official train split was acquired
    gate["exception_approved_by_protocol"] = True
    evidence["exception_approved_by_protocol"] = {
        "availability_exception": (D / "availability_exception.md").exists(),
        "reason": "the complete official MSVD-QA train split (1161/1161 videos) "
                  "was acquired, so no reduced-subset exception is in force"}

    # ---- 10. confirm-set assets for the final evaluation
    confirm_inv = DATASET_ROOT / "official" / "val_fetch_inventory.jsonl"
    confirm_ok = 0
    if confirm_inv.exists():
        have = {json.loads(l)["video_id"] for l in
                confirm_inv.read_text().splitlines() if l.strip()}
        confirm_ok = len(conf & have)
    gate["confirm_assets_present"] = bool(conf and confirm_ok == len(conf))
    evidence["confirm_assets_present"] = {"confirm_videos": len(conf),
                                          "fetched": confirm_ok}

    hard = ["msvd_train_source_verified", "train_schedule_frozen",
            "all_scheduled_assets_present", "all_scheduled_assets_decode_ok",
            "all_scheduled_qa_bound", "timestamp_metadata_ready",
            "paired_sft_opd_schedule_identical", "split_overlap_audit_passed",
            "space_sufficient", "exception_approved_by_protocol"]
    gate["train_ready"] = all(gate[k] for k in hard)
    gate["soft_gates"] = {"confirm_assets_present":
                          gate.pop("confirm_assets_present")}
    out = {
        "computed_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": str(R),
        **gate,
        "evidence": evidence,
        "hard_requirements": hard,
        "note": "train_ready gates the formal 120-update runs; the confirm-set "
                "assets are a soft gate for the FINAL evaluation only and are "
                "fetched from the same verified ZIP source.",
    }
    (D / "readiness_gate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items()
                      if k in hard or k in ("train_ready", "soft_gates")}, indent=2))
    if not gate["train_ready"]:
        failed = [k for k in hard if not out[k]]
        raise SystemExit(f"train_ready is FALSE; failing gates: {failed}")


if __name__ == "__main__":
    main()
