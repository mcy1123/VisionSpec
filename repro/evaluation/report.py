#!/usr/bin/env python3
"""Collect the reproduction's numbers next to the paper's.

Reads the eval JSON written by `eval_chartqa.py` (one file per budget setting,
or one file with several arms) and prints the comparison against the targets
recorded in CONFIG.md §7. It deliberately prints the deviations too: a table
that only shows the headline number is how a reproduction quietly becomes a
demo.

    python report.py --eval $GLANCE_ROOT/results/chartqa.json \
                     --audit $GLANCE_ROOT/results/audit.json
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
import json
import os
import sys

# CONFIG.md §7, all three taken from the paper's own tables.
PAPER = {
    "chartqa_table10_b15": {"tau": 4.43, "speedup": 3.07, "source": "Table 10, N=15"},
    "chartqa_table10_b31": {"tau": 4.81, "speedup": 3.30, "source": "Table 10, N=31"},
    "chartqa_table10_b47": {"tau": 5.00, "speedup": 3.40, "source": "Table 10, N=47"},
    "chartqa_table10_b63": {"tau": 5.16, "speedup": 3.42, "source": "Table 10, N=63"},
    "chartqa_table2": {"tau": 5.16, "speedup": None, "source": "Table 2, budget-63 tree"},
    "chartqa_table1": {"tau": 4.62, "speedup": 2.93, "source": "Table 1, SGLang, 32 draft tokens"},
}


def load(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="append", default=[],
                        help="eval JSON; repeatable")
    parser.add_argument("--audit", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    lines = ["# GLANCE reproduction: measured vs. paper", ""]

    for path in args.eval:
        result = load(path)
        task = result.get("task", "?")
        lines.append(f"## {task}  (`{os.path.basename(path)}`)")
        lines.append("")
        lines.append(f"- prompts: {result['n_prompts']} (one more read as warm-up)")
        lines.append(f"- autoregressive baseline: {result['ar_ms_per_token']:.2f} ms/token")
        cfg = result.get("config", {})
        lines.append(f"- max-pixels: {cfg.get('max_pixels')}, max-new: {cfg.get('max_new')}")
        lines.append(f"- prompt set: `{cfg.get('data')}`  (this is the only axis that "
                     f"separates the template variants)")
        lines.append(f"- head: `{cfg.get('head')}`")
        lines.append("")
        lines.append("| arm | budget | tau | tau 95% CI | paper tau | ms/token | speedup | speedup 95% CI | paper speedup | rounds |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, arm in sorted(result["arms"].items()):
            budget = name.rsplit("b", 1)[-1] if "b" in name else "-"
            paper = PAPER.get(f"{task}_table10_b{budget}", {})
            tau_ci = arm.get("tau_ci95")
            sp_ci = arm.get("speedup_ci95")
            lines.append(
                f"| {name} | {budget} | **{arm['tau']:.2f}** | "
                f"{f'[{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]' if tau_ci else '-'} | "
                f"{paper.get('tau', '-')} | {arm['ms_per_token']:.2f} | "
                f"**{arm['speedup']:.2f}x** | "
                f"{f'[{sp_ci[0]:.2f}, {sp_ci[1]:.2f}]' if sp_ci else '-'} | "
                f"{paper.get('speedup', '-')} | {arm['rounds']} |")
        lines.append("")
        # [App.D "Statistics"] a ratio is worth reporting when its interval
        # excludes the paper's, so say plainly which arms clear 1.0.
        for name, arm in sorted(result["arms"].items()):
            sp_ci = arm.get("speedup_ci95")
            if sp_ci:
                verdict = ("beats autoregression" if sp_ci[0] > 1.0
                           else "slower than autoregression" if sp_ci[1] < 1.0
                           else "indistinguishable from autoregression")
                lines.append(f"- {name}: speedup 95% CI [{sp_ci[0]:.2f}, {sp_ci[1]:.2f}] "
                             f"-> {verdict}")
        lines.append("")
        for name, arm in sorted(result["arms"].items()):
            budget = name.rsplit("b", 1)[-1] if "b" in name else "-"
            paper = PAPER.get(f"{task}_table10_b{budget}")
            if not paper:
                continue
            lines.append(f"- {name}: tau {arm['tau'] - paper['tau']:+.2f} vs paper "
                         f"({paper['source']}), speedup {arm['speedup'] - paper['speedup']:+.2f}x")
        lines.append("")

    if args.audit and os.path.exists(args.audit):
        audit = load(args.audit)
        lines.append("## fp32 lossless gate")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(audit, indent=2)[:2000])
        lines.append("```")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
