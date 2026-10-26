#!/usr/bin/env python3
"""Autoregressive oracle: the target's own greedy continuation, no drafter.

This is the reference the speculative arms are supposed to reproduce. GLANCE is
lossless at temperature 0 in exact arithmetic, so every arm's committed tokens
should equal this sequence; comparing against it turns "the arms disagree" into
"how far each arm drifts from the target's own output", which is the confound the
main table cannot separate on its own.

Writes the token ids (not just their count) so divergence can be localised to a
position rather than merely counted.

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
import statistics as st
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from glance.decode import greedy_generate
from glance.vl import VisionPositions
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.rows)
    if a.n:
        df = df.head(a.n)
    if a.shards > 1:
        df = df.iloc[a.shard::a.shards].reset_index(drop=True)
    print(f"AR oracle: {len(df)} rows max_new {a.max_new}", flush=True)

    target, proc = common.load_target(device=dev)
    eos = proc.tokenizer.eos_token_id

    res = []
    for i, row in df.reset_index(drop=True).iterrows():
        t0 = time.perf_counter()
        text = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        ids = ins["input_ids"].to(dev)
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
        out, stats = greedy_generate(target, ids, a.max_new, [eos], positions=pos,
                                     return_stats=True)
        n_in = int(ids.shape[1])
        cont = out[0, n_in:].tolist()
        res.append({"video": str(row["video_name"]), "L": n_in,
                    "ar_ids": cont, "gen_tokens": len(cont),
                    "hit_eos": bool(cont and cont[-1] == eos),
                    "wall_s": time.perf_counter() - t0})
        print(f"  [{i+1}/{len(df)}] {str(row['video_name'])[:16]:16s} "
              f"gen={len(cont):4d} eos={res[-1]['hit_eos']} "
              f"wall={res[-1]['wall_s']:6.1f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": res}, open(a.out, "w"), indent=1)
    print(f"\nAR gen median {st.median(r['gen_tokens'] for r in res):.0f}  "
          f"no-EOS {sum(1 for r in res if not r['hit_eos'])}/{len(res)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
