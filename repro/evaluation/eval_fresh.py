#!/usr/bin/env python3
"""Score one (head, arm) pair on a clean row parquet, with the standard protocol.

Same protocol as `eval_video_opd.py` / `diag_arms.py` -- explicit row parquet,
budget 63, branch 8, max_new 1024, greedy, per-row paired -- so numbers are
comparable with the recorded A-arm baseline. What differs is only how the head is
assembled:

  --arm mlp      the head as saved (fc + hidden_norm), context width n_concat*D
  --arm vispec   the head plus its saved ViSpec adapter; the context is the slim
                 wide tensor (n_concat*D + 1) and the adapter is fed the visual mask

For `vispec` the decode path is left completely alone: the map from `fused` to KV
still goes through `head.hidden_norm(head.fc(...))`, which is the seam the adapter
occupies, and the context row count is unchanged, so positions, the tree and the
rotary tables are untouched. `SlimWideContext` only appends one channel to what
`glance.decode.context_feature` returns, and that channel survives the per-round
gather because the gather indexes positions.

`--capture-feats` additionally reports the projection-fidelity metrics on the
CACHED forward path (the one decode uses), not the uncached ablation path.

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
import glance
from historical import vispec_adapter as va
from glance.vl import VisionPositions
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["mlp", "vispec"])
    ap.add_argument("--head", required=True, help="head dir (for vispec, also holds the adapter)")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--store-ids", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.rows)
    if a.n:
        df = df.head(a.n)
    print(f"arm {a.arm}: {len(df)} rows head {a.head} max_new {a.max_new} "
          f"b{a.budget}k{a.branch}", flush=True)

    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev).eval()
    ctx = None
    if a.arm == "vispec":
        va.load_vispec(head, a.head)
        n_ad = sum(p.numel() for p in head.fc.adapter.parameters())
        print(f"loaded ViSpec adapter {n_ad/1e6:.2f}M params", flush=True)
        ctx = va.SlimWideContext(target)
    total = sum(p.numel() for p in head.parameters())
    print(f"head params {total/1e6:.1f}M target_layer_ids {head.target_layer_ids}", flush=True)
    eos = proc.tokenizer.eos_token_id

    res = []
    for i, row in df.reset_index(drop=True).iterrows():
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
        n_in = int(ids.shape[1])
        pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
        rounds = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, stats = glance.generate(head, target, ids, a.max_new, [eos],
                                     budget=a.budget, branch_k=a.branch,
                                     temperature=0.0, positions=pos,
                                     return_stats=True, timing="decode_only",
                                     on_round=(rounds.append if a.store_ids else None))
        torch.cuda.synchronize()
        spec_s = time.perf_counter() - t0
        cont = out[0, n_in:].tolist()
        rec = {"video": str(row["video_name"]), "arm": a.arm, "L": n_in,
               "gen_tokens": len(cont), "rounds": int(stats.rounds),
               "tau": float(stats.tau), "Ea": float(stats.tau) - 1.0,
               "hit_eos": bool(cont and cont[-1] == eos),
               "spec_wall_s": spec_s, "warmup": i < a.warmup,
               "n_frames": int(row.get("n_frames", 0)),
               "fps": float(row.get("fps", 0.0)),
               "width": int(row.get("width", 0)), "height": int(row.get("height", 0))}
        if a.store_ids:
            rec["spec_ids"] = cont
            rec["accepted"] = [int(r["accepted"]) for r in rounds]
            rec["anchor_rel"] = [int(r["pos"]) - n_in for r in rounds]
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:16]:16s} L={n_in:4d} "
              f"gen={len(cont):4d} rounds={rec['rounds']:3d} tau={rec['tau']:.3f} "
              f"spec={spec_s:6.2f}s eos={rec['hit_eos']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "arm": a.arm, "rows": res}, open(a.out, "w"), indent=1)
    keep = [r for r in res if not r["warmup"]]
    taus = [r["tau"] for r in keep]
    print(f"\nARM {a.arm} ({a.tag})  n={len(keep)}  tau median {st.median(taus):.4f} "
          f"mean {st.fmean(taus):.4f} min {min(taus):.4f} max {max(taus):.4f} "
          f"sd {st.stdev(taus):.4f}")
    print(f"Ea median {st.median(r['Ea'] for r in keep):.4f}  "
          f"no-EOS {sum(1 for r in keep if not r['hit_eos'])}/{len(keep)}")
    print(f"-> {a.out}")
    if ctx is not None:
        ctx.close()


if __name__ == "__main__":
    main()
