#!/usr/bin/env python3
"""Assemble the confirmation-set evaluation manifest from the frozen rows.

The confirmation set was frozen before any model output was read
(`dataset/confirm_manifest.jsonl`: 60 official-val MSVD videos, up to 2 questions
each, video-disjoint from dev and from every training video). This script turns
those frozen rows into the same manifest shape the evaluation consumes, pulling
the cache path, source PTS and frame ids from the acquisition inventory that
`fetch_msvd_subset.py` wrote.

Both prompt formats are emitted, because the evaluation needs B for the main
comparison and A for the EOS-sensitivity view. Faithful to the historical run:
the prompt strings come from `repro/videoqa_transfer/protocol.py`.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "videoqa_transfer")]

from protocol import prompt as make_prompt  # noqa: E402

INV = pathlib.Path("/mnt/local2/mcy/datasets/msvd_qa_full/official/"
                   "val_fetch_inventory.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = pathlib.Path(a.run)
    D = R / "dataset"
    inv = {}
    for line in INV.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            inv[rec["video_id"]] = rec
    rows = [json.loads(l) for l in (D / "confirm_manifest.jsonl").read_text().splitlines()
            if l.strip()]
    missing = sorted({r["video_id"] for r in rows} - set(inv))
    if missing:
        raise SystemExit(f"confirm videos without a fetched asset: {missing[:5]}")
    out = []
    for r in rows:
        m = inv[r["video_id"]]
        for mode in ("A", "B"):
            out.append({
                "dataset": "MSVD-QA",
                "split": "val",
                "video_id": r["video_id"],
                "source_id": r.get("source_id") or r["video_id"][:11],
                "question_id": r["question_id"],
                "question": r["question"],
                "options": None,
                "gold_answer": r["gold_answer"],
                "question_type": r["question_type"],
                "overlap_status": "overlap_unknown",
                "selection": "confirm",
                "prompt_mode": mode,
                "prompt": make_prompt(r, mode),
                "raw_path": m["raw_path"],
                "cache_path": m["cache_path"],
                "hash": m["raw_sha256"],
                "cache_hash": m["cache_sha256"],
                "fps": float(m["fps"]),
                "source_fps": float(m["source_fps"]),
                "video_duration": float(m["video_duration"]),
                "source_frame_count": int(m["source_frame_count"]),
                "sample_frame_ids": m["sample_frame_ids"],
                "sample_timestamps": m["sample_timestamps"],
                "n_frames": int(m["n_frames"]),
                "width": int(m["width"]),
                "height": int(m["height"]),
            })
    path = D / "confirm_eval_manifest.jsonl"
    with path.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "videos": len({r["video_id"] for r in out}),
        "questions": len({(r["video_id"], r["question_id"]) for r in out}),
        "rows": len(out),
        "formats": ["A", "B"],
        "path": str(path),
        "frozen_source": str(D / "confirm_manifest.jsonl"),
        "asset_source": str(INV),
    }
    (D / "confirm_eval_manifest_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
