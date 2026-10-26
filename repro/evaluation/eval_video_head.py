#!/usr/bin/env python3
"""tau of one block head on cached video rows, run to EOS.

The video counterpart of `eval_chartqa.py`. Two deliberate differences from the
image evaluator:

1. The stopping condition is the token budget, not a round budget. `glance.
   generate` has no `max_rounds`; 13/30 of the held-out rows need >120 rounds, so
   a round cap would truncate 43% of the pool and bias every length statistic.
   Rows that hit `--max-new` without EOS are flagged, never dropped.
2. Every row is one prompt template, taken from the pool parquet's `question`
   column, which `predecode_cache.py` fills from the dataset (one unique value
   for the whole pool). It must be the same string the head was trained under.

No try/except: a failure surfaces with its traceback rather than shrinking the
denominator.
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
import time

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

import pandas as pd
import torch

from shared import common
import glance
from glance.vl import VisionPositions
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--head", required=True,
                    help="head dir; pass checkpoints/head for the zero-shot arm")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--restrict-to", default=None,
                    help="parquet whose video_name column limits the rows; used "
                         "to score the 30 anchor and the 30 fresh held-out rows "
                         "separately, since they are interleaved in pool order")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=31)
    ap.add_argument("--branch", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--prompt", default=None,
                    help="override the pool's question column")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.pool)
    use = df[df.split == a.split].reset_index(drop=True)
    if a.restrict_to:
        keep = set(pd.read_parquet(a.restrict_to)["video_name"].astype(str))
        use = use[use["video_name"].astype(str).isin(keep)].reset_index(drop=True)
    if a.offset:
        use = use.iloc[a.offset:]
    use = use.head(a.n)
    print(f"{a.split}: {len(df[df.split == a.split])} rows, running {len(use)} "
          f"(offset {a.offset}), max_new={a.max_new} b{a.budget}k{a.branch} "
          f"head={a.head}", flush=True)

    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev).eval()
    eos = proc.tokenizer.eos_token_id
    print(f"head block {head.block_size}", flush=True)

    res = []
    for i, row in use.iterrows():
        text = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": a.prompt or row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        ids = ins["input_ids"].to(dev)
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, st = glance.generate(head, target, ids, a.max_new, [eos],
                                  budget=a.budget, branch_k=a.branch,
                                  temperature=0.0, positions=pos,
                                  return_stats=True, timing="decode_only")
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        cont = out[0, ids.shape[1]:].tolist()
        hit_eos = bool(cont and cont[-1] == eos)
        res.append({"video": str(row["video_name"]), "gen_tokens": len(cont),
                    "rounds": st.rounds, "tau": float(st.tau),
                    "Ea": float(st.tau) - 1.0, "wall_s": el,
                    "hit_eos": hit_eos, "L": int(ids.shape[1]),
                    "prompt_len": int(ids.shape[1])})
        print(f"  [{i+1}/{len(use)}] {str(row['video_name'])[:16]:16s} "
              f"gen={len(cont):4d} rounds={st.rounds:3d} tau={st.tau:.3f} "
              f"wall={el:6.1f}s eos={hit_eos}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": res}, open(a.out, "w"), indent=1)

    import statistics as stt
    taus = [r["tau"] for r in res]
    print(f"\nrows {len(res)}  tau median {stt.median(taus):.4f}  "
          f"mean {stt.fmean(taus):.4f}  min {min(taus):.4f}  max {max(taus):.4f}  "
          f"sd {stt.stdev(taus):.4f}")
    print(f"Ea median {stt.median(r['Ea'] for r in res):.4f}")
    print(f"gen_tokens median {stt.median(r['gen_tokens'] for r in res):.0f}  "
          f"max {max(r['gen_tokens'] for r in res)}")
    print(f"no-EOS {sum(1 for r in res if not r['hit_eos'])}/{len(res)}")
    print(f"wall median {stt.median(r['wall_s'] for r in res):.1f}s  "
          f"total {sum(r['wall_s'] for r in res):.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
