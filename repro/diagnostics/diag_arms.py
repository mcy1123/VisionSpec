#!/usr/bin/env python3
"""Stage 1/3 of the visual-adaptation diagnosis: score one KV-source arm.

Same protocol as `eval_video_opd.py` -- explicit row parquet, budget 63, branch
8, max_new 1024, per-row paired tau -- so an arm's numbers are directly
comparable with the recorded H-8 baseline. The only difference is which KV
source the frozen head is fed (`diag_common.bind_arm`).

    python diag_arms.py --arm C --rows <holdout>.parquet --head <h8> \
        --gpu 0 --tag C --out results/diag/C.json

`--capture-attn` additionally records, for the first draft round only, how the
block's attention mass splits between visual and text context rows and how
concentrated it is (stage 3). Capture is a read-only hook, so it cannot change
the arm's tokens.

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
import statistics as st
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
import glance
from diagnostics import diag_common as dc
from glance.vl import VisionPositions
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(dc.ARMS))
    ap.add_argument("--rows", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--capture-attn", action="store_true")
    ap.add_argument("--capture-attn-rows", type=int, default=3,
                    help="how many rows get attention capture (cost grows with ctx_len)")
    ap.add_argument("--store-ids", action="store_true",
                    help="also store the committed token ids and per-round accepted "
                         "lengths, so cross-arm divergence can be localised instead of "
                         "only counted (bf16 is not bit-lossless, so two arms can commit "
                         "different continuations)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    dt = torch.float32 if a.dtype == "float32" else None
    df = pd.read_parquet(a.rows)
    if a.n:
        df = df.head(a.n)
    if a.shards > 1:
        df = df.iloc[a.shard::a.shards].reset_index(drop=True)
    print(f"ARM {a.arm}: {dc.ARM_DOC[a.arm]}", flush=True)
    print(f"rows {a.rows}: {len(df)} (shard {a.shard}/{a.shards}) head {a.head} "
          f"dtype {a.dtype} max_new {a.max_new} b{a.budget}k{a.branch} "
          f"attn={a.capture_attn}", flush=True)

    target, _ = common.load_target(device=dev, dtype=dt)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev, dtype=dt).eval()
    eos = proc.tokenizer.eos_token_id

    ctx = dc.WideContext(target)
    th_stats = []
    dc.bind_arm(head, a.arm, stats=th_stats)
    cap = dc.AttnCapture(head) if a.capture_attn else None
    print(f"target_layer_ids {head.target_layer_ids} block {head.block_size} "
          f"params {sum(p.numel() for p in head.parameters())/1e6:.1f}M", flush=True)

    res = []
    for i, row in df.reset_index(drop=True).iterrows():
        t_row0 = time.perf_counter()
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
        vis_row = dc.visual_mask_for_row(ids.cpu(), (ctx.video_id, ctx.image_id))
        torch.cuda.synchronize()

        do_cap = cap is not None and i < a.capture_attn_rows
        if do_cap:
            cap.vis_ctx = vis_row
            cap.rec = []
            cap._n_calls = 0
            cap.enabled = True
        th_stats.clear()

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
        if do_cap:
            cap.enabled = False
        t_row1 = time.perf_counter()

        n_spec = int(out.shape[1]) - n_in
        spec_ids = out[0, n_in:].tolist()
        rec = {"video": str(row["video_name"]), "arm": a.arm,
               "L": n_in, "prompt_len": n_in, "gen_tokens": n_spec,
               "rounds": int(stats.rounds), "tau": float(stats.tau),
               "Ea": float(stats.tau) - 1.0,
               "hit_eos": bool(spec_ids and spec_ids[-1] == eos),
               "tree_size": float(stats.tree_size),
               "spec_wall_s": spec_s, "row_wall_s": t_row1 - t_row0,
               "warmup": i < a.warmup,
               "n_frames": int(row.get("n_frames", 0)),
               "width": int(row.get("width", 0)), "height": int(row.get("height", 0)),
               "fps": float(row.get("fps", 0.0)),
               "n_visual_tokens": int(vis_row.sum()),
               "vis_frac": float(vis_row.float().mean()),
               "kv_round0": dict(th_stats[0]) if th_stats else None}
        if a.store_ids:
            rec["spec_ids"] = spec_ids
            rec["accepted"] = [int(r["accepted"]) for r in rounds]
            rec["anchor_rel"] = [int(r["pos"]) - n_in for r in rounds]
        if do_cap:
            rec["attn"] = cap.take()
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:16]:16s} L={n_in:4d} "
              f"vis={rec['n_visual_tokens']:4d} gen={n_spec:4d} "
              f"rounds={rec['rounds']:3d} tau={rec['tau']:.3f} "
              f"spec={spec_s:6.2f}s eos={rec['hit_eos']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "arm": a.arm, "arm_doc": dc.ARM_DOC[a.arm],
               "rows": res}, open(a.out, "w"), indent=1)

    keep = [r for r in res if not r["warmup"]]
    taus = [r["tau"] for r in keep]
    print(f"\nARM {a.arm}  n={len(keep)} (warmup dropped {len(res)-len(keep)})  "
          f"tau median {st.median(taus):.4f} mean {st.fmean(taus):.4f} "
          f"min {min(taus):.4f} max {max(taus):.4f} sd {st.stdev(taus):.4f}")
    print(f"Ea median {st.median(r['Ea'] for r in keep):.4f}  "
          f"gen median {st.median(r['gen_tokens'] for r in keep):.0f}")
    print(f"no-EOS {sum(1 for r in keep if not r['hit_eos'])}/{len(keep)}  "
          f"spec wall median {st.median(r['spec_wall_s'] for r in keep):.1f}s")
    print(f"-> {a.out}")
    ctx.close()
    if cap is not None:
        cap.close()


if __name__ == "__main__":
    main()
