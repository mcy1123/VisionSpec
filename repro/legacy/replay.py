"""Explicit historical replay; never an official-KL training entry."""

import argparse
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--entry",
        required=True,
        choices=[
            "repro/scienceqa_image/train_sqa_opd.py",
            "repro/qa_opd_fair/train_qa_opd.py",
            "repro/opd_train_video_fullkl.py",
            "repro/opd_train_video.py",
            "repro/opd_train.py",
        ],
    )
    ap.add_argument(
        "--acknowledge-historical-objective", action="store_true", required=True
    )
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = Path(__file__).parent / "pre20261016" / a.entry
    original = root / a.entry
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    print(
        json.dumps(
            dict(
                objective="legacy_historical_replay",
                official_kl=False,
                entry=a.entry,
                archived_source=str(path),
            )
        ),
        flush=True,
    )
    sys.path[:0] = [str(original.parent), str(root / "repro")]
    # Archived sources keep their original bare imports; only explicit replay
    # exposes the categorized compatibility directories.
    sys.path[:0] = [str(root / "repro" / name) for name in
                   ("shared", "historical", "diagnostics", "evaluation", "data", "tools", "tests")]
    sys.argv = [str(original), *args]
    if "--help" in args or "-h" in args:
        # The archived full-KL parser contains literal "% of" help strings.
        # Escape bare percent signs for display only; never alter frozen source
        # or the training branch's semantics.
        import re

        original_help = argparse.HelpFormatter._get_help_string
        argparse.HelpFormatter._get_help_string = lambda self, action: re.sub(
            r"%(?![%(])", "%%", original_help(self, action)
        )
    exec(
        compile(path.read_bytes(), str(path), "exec"),
        {"__name__": "__main__", "__file__": str(original)},
    )


if __name__ == "__main__":
    main()
