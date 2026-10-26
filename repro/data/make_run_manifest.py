#!/usr/bin/env python3
"""Write a delivery manifest for a run directory: every file, size, sha256.

WHY: AGENTS.md requires each run to ship a manifest so a reader can verify that
the numbers in a report come from the files that are actually on disk. Hashing
the big checkpoint weights is worth the seconds; hashing nothing and claiming
"verified" is what the project keeps having to correct.

Usage: make_run_manifest.py <run-dir> [--out manifest.json] [--skip PATTERN ...]
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
import datetime
import hashlib
import json
import os
import sys


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", action="append", default=[],
                    help="substring; a file whose relative path contains it is "
                         "listed with its size but NOT hashed (e.g. training_state.pt)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    run = os.path.abspath(a.run)
    if not os.path.isdir(run):
        raise SystemExit(f"not a directory: {run}")
    files = []
    for root, dirs, names in os.walk(run):
        dirs.sort()
        for name in sorted(names):
            path = os.path.join(root, name)
            if os.path.islink(path):
                files.append({"path": os.path.relpath(path, run),
                              "symlink_to": os.readlink(path)})
                continue
            rel = os.path.relpath(path, run)
            entry = {"path": rel, "size": os.path.getsize(path)}
            if any(s in rel for s in a.skip):
                entry["sha256"] = None
                entry["note"] = "skipped by --skip"
            else:
                entry["sha256"] = sha256(path)
            files.append(entry)

    manifest = {
        "run": run,
        "label": a.label,
        "created": datetime.datetime.now().astimezone().isoformat(),
        "n_files": len(files),
        "total_bytes": sum(f.get("size", 0) for f in files),
        "files": files,
    }
    out = a.out or os.path.join(run, "delivery_manifest.json")
    with open(out, "w") as handle:
        json.dump(manifest, handle, indent=1)
    print(f"wrote {out}: {len(files)} files, "
          f"{manifest['total_bytes']/2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
