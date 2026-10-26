#!/usr/bin/env python3
"""Download only the COCO train2017 images the training prompts actually use.

The full train2017 archive is 18 GB; the reproduction needs 8000 images, which
is about 1.3 GB fetched one file at a time. Parallelism is what makes that
faster than the archive.

    python fetch_coco_images.py --prompts $GLANCE_ROOT/data/train_prompts.jsonl
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
import concurrent.futures as futures
import os
import subprocess
import sys

from shared import common


def fetch_one(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for attempt in range(6):
        proc = subprocess.run(
            ["curl", "-sS", "--noproxy", "*", "-L", "--connect-timeout", "30",
             "--max-time", "180", "--retry", "2", "-o", part, url],
            capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(part) and os.path.getsize(part) > 0:
            os.replace(part, dest)
            return 0
        if os.path.exists(part):
            os.remove(part)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=os.path.join(common.DATA, "train_prompts.jsonl"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--base", default="http://images.cocodataset.org/train2017")
    args = parser.parse_args()

    rows = common.read_jsonl(args.prompts)
    targets = []
    for row in rows:
        if row["task"] != "coco":
            continue
        if os.path.exists(row["image"]):
            continue
        targets.append((f"{args.base}/{os.path.basename(row['image'])}", row["image"]))

    print(f"{len(targets)} COCO images to fetch, {args.workers} workers", flush=True)
    done = failed = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch_one, url, dest): dest for url, dest in targets}
        for job in futures.as_completed(jobs):
            code = job.result()
            done += 1
            failed += code
            if done % 200 == 0 or done == len(targets):
                print(f"  {done}/{len(targets)} done, {failed} failed", flush=True)

    print(f"finished: {done - failed} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
