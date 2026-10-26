"""Workspace paths for scripts executed directly from a repro subdirectory."""
from pathlib import Path
import sys

REPRO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPRO_ROOT.parent

def configure():
    for path in [REPRO_ROOT, PROJECT_ROOT / "GLANCE_repo",
                 PROJECT_ROOT / "scripts/dflash_pkg/dflash-0.1.0"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def source_path(name):
    """Resolve pre-reorganization source names for diagnostic provenance."""
    import json
    mapping = json.loads((REPRO_ROOT / "layout.json").read_text())
    old = name if name.startswith("repro/") else "repro/" + name
    return PROJECT_ROOT / mapping.get(old, old)
