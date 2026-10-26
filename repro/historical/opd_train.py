#!/usr/bin/env python3
"""Image entry for the shared online Bernoulli/k3 runtime.

Historical objectives are available only through legacy/replay.py.
"""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import sys


def main(argv=None):
    from opd_official_kl.cli import main as official_main

    args = sys.argv[1:] if argv is None else argv
    return official_main(["--modality", "image", *args])


if __name__ == "__main__":
    main()
