#!/usr/bin/env python3
"""Does an fp32 teacher forward make B0's G1 gate pass literally?

B0 (bf16) reported `max|dlogit| = 0.375` between the incremental (prefill + draft
proposal through a forked prefix KV) and the independent single full forward. Its
root-cause probe says bf16 GEMM row-count sensitivity. A standalone GEMM check
agrees and shows the effect is EXACTLY zero in fp32:

    bf16 M=16 vs M=1 : max|d| = 3.125e-02   (|logit|max 6.4)
    fp32 M=16 vs M=1 : max|d| = 0.000e+00

That only covered `lm_head`. This script re-runs B0's own `score_anchor` with the
WHOLE target in fp32, reusing the exact anchor `pos`/`prop` recorded in the bf16
JSON, and reports whether the two paths then agree bit-for-bit.

Decides a real B1 question: does paper-faithful teacher scoring need an fp32
target forward (2x memory/time) to be reproducible against an offline full
forward, or is bf16's ~0.4% objective drift acceptable?

No try/except: a failure surfaces with its traceback.
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

import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from diagnostics import proto_fullkl_b0 as proto
from transformers import AutoProcessor


def stats(a, b):
    d = (a.float() - b.float()).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "exact": bool(torch.equal(a, b)), "ref_max_abs": b.float().abs().max().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", default="/mnt/local/mcy/GLANCE/results/phaseB0/proto_fullkl_b0.json")
    ap.add_argument("--rows", default="/mnt/local/mcy/GLANCE/data/video_fresh30.parquet")
    ap.add_argument("--records", default="/mnt/local/mcy/GLANCE/data/video_opd_pool_warm_499.pt")
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    b0 = json.load(open(a.b0))
    anc = b0["anchors"][a.anchor]
    p, prop = int(anc["pos"]), list(anc["prop"])
    row_index = int(b0["config"]["row_index"])

    target, _ = common.load_target(device=dev, dtype=torch.float32)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)

    import pandas as pd
    df = pd.read_parquet(a.rows)
    row = df.iloc[row_index]
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": row["question"]}]}],
        add_generation_prompt=True, tokenize=False)
    ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt", fps=float(row["fps"]))
    pv = ins["pixel_values_videos"].to(dev)
    vg = ins["video_grid_thw"].to(dev)

    recs = torch.load(a.records, weights_only=False)["records"]
    rec = next(r for r in recs if r["task"] == str(row["video_name"]))
    prefix_ids = rec["input_ids"][: p + 1].to(dev).long()[None]
    assert int(prefix_ids.shape[1]) == p + 1, (prefix_ids.shape, p)

    torch.cuda.reset_peak_memory_stats(dev)
    out = proto.score_anchor(target, pv, vg, prefix_ids, prop, p,
                             with_chunking_control=True)

    inc_vs_ref = stats(out["inc"], out["ref"])
    hid = stats(out["h_inc"], out["ref_h"])
    ctrl = stats(out["inc_stepwise"], out["inc"])
    # Offset-0 logits come from the prefill itself in both paths, so they are
    # trivially equal; report the appended-only rows too.
    inc_vs_ref_tail = stats(out["inc"][1:], out["ref"][1:])

    print(f"dtype float32 | anchor {a.anchor} pos={p} len_prop={len(prop)}")
    print(f"  inc vs ref  logits : max|d|={inc_vs_ref['max_abs']:.4e} "
          f"exact={inc_vs_ref['exact']}  (ref max|logit|={inc_vs_ref['ref_max_abs']:.3f})")
    print(f"  inc vs ref  hidden : max|d|={hid['max_abs']:.4e} exact={hid['exact']}")
    print(f"  inc vs ref  logits[appended rows only]: max|d|={inc_vs_ref_tail['max_abs']:.4e} "
          f"exact={inc_vs_ref_tail['exact']}")
    print(f"  chunking control (chunk vs stepwise): max|d|={ctrl['max_abs']:.4e} exact={ctrl['exact']}")
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    print(f"  peak allocated: {peak:.2f} GiB")
    print(f"\nVERDICT: fp32 teacher forward is "
          f"{'BIT-IDENTICAL' if inc_vs_ref['exact'] else 'still not bit-identical'} "
          f"on the incremental-vs-full-forward gate")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"dtype": "float32", "anchor": anc, "inc_vs_ref_logits": inc_vs_ref,
                   "inc_vs_ref_logits_appended_only": inc_vs_ref_tail,
                   "inc_vs_ref_hidden": hid, "chunking_control": ctrl,
                   "peak_allocated_gib": peak,
                   "bit_identical": inc_vs_ref["exact"]}, open(a.out, "w"), indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
