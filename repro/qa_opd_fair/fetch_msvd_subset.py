#!/usr/bin/env python
"""Acquire + cache-16 any MSVD videos this run needs beyond the train split.

The task document's data section requires the evaluation assets to be present and
verified too, not just the training ones. The 2026-10-13 transfer run already
froze 20 MSVD val videos; this script fetches the additional official-val videos
the frozen `confirm_manifest.jsonl` references, from the same HF-mirror ZIP the
train acquisition used, with the same cache-16 convention and the same
verification fields.

Writes:
  <dataset-root>/val_cache16/<stem>.cache16.mp4
  <dataset-root>/val_raw/<video_id>
  <dataset-root>/official/val_fetch_inventory.jsonl   (one row per requested id)
  <dataset-root>/official/val_fetch_summary.json

Idempotent: a video whose raw + cache exist with the recorded sizes is skipped.

Usage:
  python fetch_msvd_subset.py --run /path/to/run [--workers 6] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

DATASET = pathlib.Path("/mnt/local2/mcy/datasets/msvd_qa_full")
OFFICIAL = DATASET / "official"
RAW_DIR = DATASET / "val_raw"
CACHE_DIR = DATASET / "val_cache16"
ZIP_INDEX = DATASET / "zipmeta" / "msvd_zip_index.json"
ZIP_URL = ("https://hf-mirror.com/datasets/friedrichor/MSVD/resolve/main/"
           "MSVD_Videos.zip")


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_cache16(raw_path, cache_path):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(raw_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {raw_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sw, sh = int(cap.get(3)), int(cap.get(4))
    if total <= 0 or fps <= 0 or sw <= 0 or sh <= 0:
        raise RuntimeError(f"bad probe {raw_path}: {total} {fps} {sw}x{sh}")
    scale = 384 / max(sw, sh)
    w, h = max(int(sw * scale) // 2 * 2, 2), max(int(sh * scale) // 2 * 2, 2)
    ids, frames, cv_ts = [], [], []
    for i in np.linspace(0, total - 1, 16).astype(int):
        ok = False
        for back in range(25):
            j = int(i) - back
            if j < 0:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            ok, f = cap.read()
            if ok:
                break
        if not ok:
            cap.release()
            raise RuntimeError(f"cannot decode frame {i} of {raw_path}")
        ids.append(j)
        cv_ts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
        frames.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA))
    cap.release()
    wr = cv2.VideoWriter(str(cache_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    if not wr.isOpened():
        raise RuntimeError(f"cannot write {cache_path}")
    for f in frames:
        wr.write(f)
    wr.release()
    import av
    container = av.open(str(raw_path))
    duration = (container.duration / av.time_base
                if container.duration is not None else total / fps)
    pts = [float(fr.pts * fr.time_base) for fr in container.decode(video=0)]
    container.close()
    ts = [pts[j] for j in ids]
    if not all(t >= 0 for t in ts) or any(ts[i] > ts[i + 1] for i in range(15)):
        raise RuntimeError(f"non-monotonic PTS for {raw_path}")
    return {"video_duration": float(duration), "source_fps": float(fps),
            "source_frame_count": total, "sample_frame_ids": [int(x) for x in ids],
            "sample_timestamps": ts, "width": int(w), "height": int(h),
            "n_frames": 16, "opencv_timestamps": cv_ts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default=None,
                    help="default: the run's dataset/confirm_manifest.jsonl")
    a = ap.parse_args()
    R = pathlib.Path(a.run)
    manifest = pathlib.Path(a.manifest or (R / "dataset" / "confirm_manifest.jsonl"))
    want = []
    seen = set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        vid = json.loads(line)["video_id"]
        if vid not in seen:
            seen.add(vid)
            want.append(vid)
    if a.limit:
        want = want[:a.limit]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    members = {m["name"].split("/")[-1]: m
               for m in json.loads(ZIP_INDEX.read_text())}
    missing = [v for v in want if v not in members]
    if missing:
        raise SystemExit(f"{len(missing)} requested videos are absent from the ZIP "
                         f"index: {missing[:5]}")

    inv_path = OFFICIAL / "val_fetch_inventory.jsonl"
    prior = {}
    if inv_path.exists():
        for line in inv_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                prior[rec["video_id"]] = rec

    from fetch_msvd_train import ZipSource, RateLimiter, fetch_member

    limiter = RateLimiter(3.8e6)
    source = ZipSource(limiter, workers=2)

    def one(vid):
        raw = RAW_DIR / vid
        cache = CACHE_DIR / (pathlib.Path(vid).stem + ".cache16.mp4")
        old = prior.get(vid)
        if (old and raw.exists() and cache.exists()
                and raw.stat().st_size == old.get("size_bytes")
                and cache.stat().st_size == old.get("cache_size_bytes")):
            return {**old, "reused": True}
        t0 = time.time()
        if not (raw.exists() and old and raw.stat().st_size == old.get("size_bytes")):
            entry = members[vid]
            fetch_member(source, entry, raw)
        raw_sha = sha256_file(raw)
        info = build_cache16(raw, cache)
        rec = {
            "video_id": vid, "source_id": vid[:11],
            "raw_path": str(raw), "cache_path": str(cache),
            "raw_sha256": raw_sha, "cache_sha256": sha256_file(cache),
            "size_bytes": raw.stat().st_size,
            "cache_size_bytes": cache.stat().st_size,
            "download_source": "hf_mirror_zip_selective",
            "zip_name": members[vid]["name"],
            **info, "fps": 30, "decode_ok": True, "failure_reason": None,
            "seconds": time.time() - t0, "reused": False,
        }
        return rec

    out, errors = [], []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, v): v for v in want}
        for fut in cf.as_completed(futs):
            v = futs[fut]
            try:
                rec = fut.result()
                out.append(rec)
                print("ok" if not rec.get("reused") else "reuse", v,
                      rec.get("size_bytes"), flush=True)
            except Exception as err:                    # noqa: BLE001
                errors.append({"video_id": v, "error": repr(err)})
                print("FAIL", v, repr(err), flush=True)
    out.sort(key=lambda r: want.index(r["video_id"]))
    with inv_path.open("w") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "requested": len(want), "ok": len(out), "failed": len(errors),
        "reused": sum(1 for r in out if r.get("reused")),
        "errors": errors, "manifest": str(manifest),
        "raw_dir": str(RAW_DIR), "cache_dir": str(CACHE_DIR),
        "zip_url": ZIP_URL,
        "note": "cache-16 convention identical to repro/videoqa_transfer/prepare.py",
    }
    (OFFICIAL / "val_fetch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2)[:1200], flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} videos failed")


if __name__ == "__main__":
    main()
