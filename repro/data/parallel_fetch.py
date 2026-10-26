#!/usr/bin/env python3
"""Parallel chunked download, because a single connection here runs ~3 MB/s.

Splits the file into N byte ranges, fetches them concurrently with curl, then
concatenates. Used for archives where one stream is the bottleneck and the
server honours range requests.

    python parallel_fetch.py URL OUT [--workers 12]
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


def head_size(url):
    proc = subprocess.run(
        ["curl", "-sSI", "--noproxy", "*", "--connect-timeout", "30",
         "--max-time", "90", url], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


def get_range(url, start, end, dest):
    part = dest + f".{start}-{end}"
    for attempt in range(8):
        proc = subprocess.run(
            ["curl", "-sS", "--noproxy", "*", "-L", "--connect-timeout", "30",
             "--max-time", "1800", "--retry", "3", "--retry-delay", "5",
             "-r", f"{start}-{end}", "-o", part, url],
            capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(part):
            have = os.path.getsize(part)
            if have == end - start + 1:
                return start, part
            if have > end - start + 1:
                with open(part, "r+b") as handle:
                    handle.truncate(end - start + 1)
                return start, part
        if os.path.exists(part):
            os.remove(part)
    raise RuntimeError(f"range {start}-{end} failed after 8 attempts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    size = head_size(args.url)
    if size is None:
        print("could not read Content-Length; server may not support ranges")
        return 1
    print(f"{args.url}\n  {size/2**30:.2f} GiB -> {args.out}", flush=True)

    chunk = size // args.workers
    ranges = []
    for i in range(args.workers):
        start = i * chunk
        end = size - 1 if i == args.workers - 1 else (i + 1) * chunk - 1
        ranges.append((start, end))

    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(get_range, args.url, s, e, args.out) for s, e in ranges]
        for job in futures.as_completed(jobs):
            start, _ = job.result()
            done += 1
            print(f"  chunk {done}/{len(ranges)} (from byte {start})", flush=True)

    with open(args.out, "wb") as out:
        for start, end in ranges:
            part = args.out + f".{start}-{end}"
            with open(part, "rb") as handle:
                while True:
                    block = handle.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
            os.remove(part)
    print(f"assembled {os.path.getsize(args.out)} bytes -> {args.out}")

    import zipfile
    if zipfile.is_zipfile(args.out):
        with zipfile.ZipFile(args.out) as zf:
            bad = zf.testzip()
        print("CRC check:", "OK" if bad is None else f"FAILED on {bad}")
        return 0 if bad is None else 2
    print("not a zip; skipping CRC check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
