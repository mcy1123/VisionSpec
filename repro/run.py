#!/usr/bin/env python3
"""Resolve an old repro filename to its categorized location.

Examples: python repro/run.py --list
          python repro/run.py eval_chartqa.py -- --help
This resolves paths only; it never converts a historical recipe into a new one.
"""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='show old-to-new paths without executing')
    parser.add_argument('name', nargs='?')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    repro = Path(__file__).resolve().parent
    paths = json.loads((repro / 'layout.json').read_text())
    if args.list:
        for old, new in sorted(paths.items()):
            print(f'{old} -> {new}')
        return
    old = args.name if args.name and args.name.startswith('repro/') else 'repro/' + (args.name or '')
    if old not in paths:
        parser.error('unknown former filename; see --list')
    path = repro.parent / paths[old]
    forwarded = args.args[1:] if args.args[:1] == ['--'] else args.args
    if path.suffix == '.sh':
        raise SystemExit(subprocess.call(['bash', str(path), *forwarded]))
    if path.suffix != '.py':
        parser.error('this entry is a document: ' + str(path))
    sys.argv = [str(path), *forwarded]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
