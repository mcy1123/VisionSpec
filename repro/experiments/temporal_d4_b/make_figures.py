#!/usr/bin/env python3
"""Optional figures: probability mass moved by each condition, per video."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/repro")

import b_common as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--main", default="main")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    main = os.path.join(run, a.main)
    recs = B.read_jsonl(os.path.join(main, "forward_records.jsonl"))
    out = os.path.join(main, "analysis")
    os.makedirs(out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fam, name in (
            ("N", "family N: main causal set (order-neutral prefix)"),
            ("C", "family C: secondary set (prefix asserts timing)")):
        per = {}
        for v in B.VIDEOS:
            us = [r for r in recs if r["video"] == v and r["family"] == fam]
            if not us:
                continue
            per[v] = {
                "T_tgt": float(np.mean([r["response_T"]["prob_tv_target"]
                                        for r in us])),
                "T_drf": float(np.mean([r["response_T"]["prob_tv_draft"]
                                        for r in us])),
                "K_tgt": float(np.mean([r["response_K"]["prob_tv_target"]
                                        for r in us])),
                "K_drf": float(np.mean([r["response_K"]["prob_tv_draft"]
                                        for r in us])),
                "N_tgt": float(np.mean([r["response_Orec"]["prob_tv_target"]
                                        for r in us])),
                "n": len(us),
                "R_T": float(np.mean([r["response_T"]["prob_R"] for r in us
                                      if r["response_T"]["prob_R"] is not None])),
                "R_K": float(np.mean([r["response_K"]["prob_R"] for r in us
                                      if r["response_K"]["prob_R"] is not None]))
                if any(r["response_K"]["prob_R"] is not None for r in us) else None,
            }
        vids = list(per)
        x = np.arange(len(vids))
        w = 0.2
        fig, ax = plt.subplots(figsize=(11, 4.4))
        ax.bar(x - 1.5 * w, [per[v]["T_tgt"] for v in vids], w,
               label="O$\\to$T target", color="#1f77b4")
        ax.bar(x - 0.5 * w, [per[v]["T_drf"] for v in vids], w,
               label="O$\\to$T draft", color="#9ecae1")
        ax.bar(x + 0.5 * w, [per[v]["K_tgt"] for v in vids], w,
               label="O$\\to$K target", color="#d62728")
        ax.bar(x + 1.5 * w, [per[v]["K_drf"] for v in vids], w,
               label="O$\\to$K draft", color="#ff9896")
        ax.plot(x, [per[v]["N_tgt"] for v in vids], "k_", ms=14,
                label="O$\\to$O$_{rec}$ target (pipeline noise floor)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v[2:13]}\n(n={per[v]['n']})" for v in vids],
                           fontsize=8)
        ax.set_ylabel("probability mass moved (total variation)")
        ax.set_title(name)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out, f"fig_tv_{fam}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
