#!/usr/bin/env python3
"""Phase 3: which part of the verify pass breaks bf16 losslessness?

Two candidate causes were left open: (a) the ATTENTION BRANCH -- the tree verify
passes a 4-D ancestor mask, so `create_causal_mask` materialises a mask, while the
AR decode steps carry a KV cache and take SDPA's `is_causal` branch; (b) the BATCH
LAYOUT -- a tree verifies 64 packed tokens in one forward, AR verifies one.

`glance.decode.chain_generate` separates them, using only existing repo code:
it verifies a 16-token block in ONE forward but with NO attention mask (cache
present) -- i.e. the `is_causal` branch with a batched layout. So

    AR (batch 1, is_causal)  vs  chain (batch 16, is_causal)  -> isolates LAYOUT
    chain (batch 16, is_causal) vs tree (batch 64, 4-D mask)  -> isolates BRANCH
    AR (batch 1, is_causal)  vs  tree (batch 64, 4-D mask)    -> what we report today

All three also run in fp32 as the control: in fp32 every pair must agree, or the
disagreement is not arithmetic.

    python diagnose_packed_bf16.py --dtype bfloat16 --n 10 --gpu 4
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
import glance
from glance.decode import chain_generate, greedy_generate
from glance.vl import VisionPositions
from transformers import AutoProcessor


def first_diff(a, b):
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="/mnt/local/mcy/GLANCE/data/video_fresh30.parquet")
    ap.add_argument("--head", default="/mnt/local/mcy/GLANCE/checkpoints/head_video439/step1000")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    dt = torch.float32 if a.dtype == "float32" else None
    df = pd.read_parquet(a.rows).head(a.n)
    target = common.load_target(device=dev, dtype=dt)[0]
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev, dtype=dt).eval()
    eos = proc.tokenizer.eos_token_id

    res = []
    for i, row in df.reset_index(drop=True).iterrows():
        ins = proc(text=[proc.apply_chat_template(
                    [{"role": "user", "content": [{"type": "video"},
                                                  {"type": "text", "text": row["question"]}]}],
                    add_generation_prompt=True, tokenize=False)],
                   videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        ids = ins["input_ids"].to(dev)
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        n_in = int(ids.shape[1])

        def vp():
            return VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)

        t0 = time.perf_counter()
        ar, _ = greedy_generate(target, ids, a.max_new, [eos], positions=vp(),
                                return_stats=True)
        t1 = time.perf_counter()
        ch, chst = chain_generate(head, target, ids, a.max_new, [eos],
                                  temperature=0.0, positions=vp(), return_stats=True)
        t2 = time.perf_counter()
        tr, trst = glance.generate(head, target, ids, a.max_new, [eos],
                                   budget=a.budget, branch_k=a.branch,
                                   temperature=0.0, positions=vp(),
                                   return_stats=True, timing="decode_only")
        t3 = time.perf_counter()
        ar_t = ar[0, n_in:].tolist()
        ch_t = ch[0, n_in:].tolist()
        tr_t = tr[0, n_in:].tolist()
        rec = {"video": str(row["video_name"]), "L": n_in,
               "ar_len": len(ar_t), "chain_len": len(ch_t), "tree_len": len(tr_t),
               "chain_tau": float(chst.tau), "tree_tau": float(trst.tau),
               "wall_ar": t1 - t0, "wall_chain": t2 - t1, "wall_tree": t3 - t2,
               "diff_ar_chain": first_diff(ar_t, ch_t),
               "diff_ar_tree": first_diff(ar_t, tr_t),
               "diff_chain_tree": first_diff(ch_t, tr_t)}
        rec["ok_ar_chain"] = rec["diff_ar_chain"] is None
        rec["ok_ar_tree"] = rec["diff_ar_tree"] is None
        rec["ok_chain_tree"] = rec["diff_chain_tree"] is None
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:14]:14s} L={n_in:4d} "
              f"AR {len(ar_t):4d} ch {len(ch_t):4d} tree {len(tr_t):4d} | "
              f"AR-chain {rec['diff_ar_chain']} AR-tree {rec['diff_ar_tree']} "
              f"chain-tree {rec['diff_chain_tree']} | tau ch {rec['chain_tau']:.3f} "
              f"tree {rec['tree_tau']:.3f} | wall {rec['wall_ar']:.1f}/"
              f"{rec['wall_chain']:.1f}/{rec['wall_tree']:.1f}s", flush=True)

    def rate(key):
        return sum(1 for r in res if r[key]) / len(res)
    smy = {k: rate(k) for k in ("ok_ar_chain", "ok_ar_tree", "ok_chain_tree")}
    print(f"\n[{a.dtype}] n={len(res)} exact-token agreement: "
          f"AR-chain {smy['ok_ar_chain']:.2f} | AR-tree {smy['ok_ar_tree']:.2f} | "
          f"chain-tree {smy['ok_chain_tree']:.2f}")
    print("  read: AR-chain isolates the batching (both is_causal); "
          "chain-tree isolates the attention branch (both batched).")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "summary": smy, "rows": res}, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
