#!/usr/bin/env python3
"""Verify and extract the TextVQA train images archive.

The archive was fetched by a retrying downloader that, for part of its run, had
two processes resuming the same `.part` file. Both wrote the same bytes at the
same offsets, so the result should be intact -- but "should be" is not a
verification, so the central directory and every entry's CRC are checked before
anything is extracted.

    python extract_textvqa.py
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import os
import sys
import zipfile

from shared import common


def main():
    root = os.path.join(common.DATASETS, "textvqa")
    archive = os.path.join(root, "train_val_images.zip")
    # the archive stores entries as `train_images/<id>.jpg`
    target = os.path.join(root, "train_images")

    if not os.path.exists(archive):
        print(f"missing archive: {archive}")
        return 1

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        print(f"{len(names)} entries, e.g. {names[1:3]}")
        bad = zf.testzip()
        if bad is not None:
            print(f"CRC FAILED on {bad} -- the archive is corrupt, re-download it")
            return 2
        print("CRC check: OK")
        zf.extractall(root)

    count = sum(1 for _ in os.scandir(target))
    print(f"extracted {count} images -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
