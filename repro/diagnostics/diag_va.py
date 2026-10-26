#!/usr/bin/env python3
"""Stage 2 of the visual-adaptation diagnosis: per-row visual dependency (VA).

VA is the difference the visual input makes to the TARGET's own output
distribution, measured on the target only, so it is a property of the row and
not of any draft arm:

  1. greedily decode `--cont` tokens with the real video (the project's own path);
  2. re-run the SAME real trajectory twice under a KV cache -- once with the real
     `pixel_values_videos` and once with that tensor zeroed -- collecting the
     next-token distribution at every step. Both runs are teacher-forced on the
     identical token prefix, so the two distributions at step t are directly
     comparable and the only thing that differs is the video content;
  3. VA = mean Jensen-Shannon divergence (bits) over those steps, plus the
     fraction of the continuation that changes when the video is blanked.

Why the cache matters, and why this file does NOT simply run one
`use_cache=False` forward over prompt+continuation: that shortcut is silently
wrong on this model. With `use_cache=False` the target builds its mask from the
M-RoPE `position_ids`, and the resulting text-position logits come out
BIT-IDENTICAL whether the video is real, zeroed, or negated -- even though the
visual positions' hidden states move by ~2e3. Under `use_cache=True` (the path
`glance.decode.generate` actually uses at prefill) the same comparison moves the
logits by ~15. A VA built on the uncached forward therefore reports 0 for every
row and measures nothing. This file uses the cached path throughout.

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
from transformers import AutoProcessor, DynamicCache

LOG2 = 0.6931471805599453


def jsd_bits(p, q):
    """Jensen-Shannon divergence in bits between two logit rows."""
    lp = torch.log_softmax(p.float(), dim=-1)
    lq = torch.log_softmax(q.float(), dim=-1)
    m = torch.logaddexp(lp, lq) - LOG2                 # log((p+q)/2)
    kl_pm = (lp.exp() * (lp - m)).sum(-1)
    kl_qm = (lq.exp() * (lq - m)).sum(-1)
    return (0.5 * (kl_pm + kl_qm)) / LOG2


@torch.no_grad()
def logits_along(target, ids, vg, pixels, cont_ids, dev):
    """Teacher-forced next-token logits at each step of `cont_ids`, with a cache."""
    pos = VisionPositions(pixel_values_videos=pixels, video_grid_thw=vg)
    pos.bind(target, ids)
    n_in = int(ids.shape[1])
    cache = DynamicCache()
    out = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                 **pos.prefill_inputs(ids, None))
    steps = [out.logits[0, -1]]
    for t in range(int(cont_ids.shape[1]) - 1):
        tok = cont_ids[:, t:t + 1]
        pid = pos(torch.tensor([[n_in + t]], device=dev))
        o = target(tok, position_ids=pid, past_key_values=cache, use_cache=True,
                   logits_to_keep=1)
        steps.append(o.logits[0, -1])
    return torch.stack(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--cont", type=int, default=64,
                    help="continuation length the divergence is averaged over")
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
    print(f"VA (cached path): rows {a.rows} n={len(df)} cont={a.cont} "
          f"shard {a.shard}/{a.shards}", flush=True)

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
        n_in = int(ids.shape[1])

        pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
        real_out, _ = greedy_generate(target, ids, a.cont, [eos], positions=pos,
                                      return_stats=True)
        cont_ids = real_out[:, n_in:n_in + a.cont]
        n_cont = int(cont_ids.shape[1])
        blank_out, _ = greedy_generate(
            target, ids, a.cont, [eos],
            positions=VisionPositions(pixel_values_videos=torch.zeros_like(pv),
                                      video_grid_thw=vg),
            return_stats=True)
        blank_ids = blank_out[:, n_in:]
        lcp = 0
        for x, y in zip(cont_ids[0].tolist(), blank_ids[0].tolist()):
            if x != y:
                break
            lcp += 1

        l_real = logits_along(target, ids, vg, pv, cont_ids, dev)
        l_blank = logits_along(target, ids, vg, torch.zeros_like(pv), cont_ids, dev)

        d = jsd_bits(l_real, l_blank)
        agree = (l_real.argmax(-1) == l_blank.argmax(-1)).float()
        rec = {"video": str(row["video_name"]),
               "L": n_in, "n_frames": int(row.get("n_frames", 0)),
               "fps": float(row.get("fps", 0.0)),
               "width": int(row.get("width", 0)), "height": int(row.get("height", 0)),
               "n_cont": n_cont,
               "VA": float(d.mean()), "VA_median": float(d.median()),
               "VA_step1": float(d[0]),
               "VA_frac_pos_gt_0.01": float((d > 0.01).float().mean()),
               "top1_agree": float(agree.mean()),
               "prefix_lcp": lcp,
               "prefix_div": 1.0 - lcp / max(n_cont, 1),
               "blank_distinct": bool(cont_ids[0].tolist() != blank_ids[0].tolist()),
               "wall_s": time.perf_counter() - t0}
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:16]:16s} L={n_in:4d} "
              f"cont={n_cont:3d} VA={rec['VA']:.4f} step1={rec['VA_step1']:.4f} "
              f"top1agree={rec['top1_agree']:.2f} lcp={lcp:3d} "
              f"changed={rec['blank_distinct']} wall={rec['wall_s']:5.1f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": res}, open(a.out, "w"), indent=1)
    vas = [r["VA"] for r in res]
    print(f"\nVA median {st.median(vas):.4f} mean {st.fmean(vas):.4f} "
          f"min {min(vas):.4f} max {max(vas):.4f} sd {st.stdev(vas):.4f}")
    print(f"top1 agree median {st.median(r['top1_agree'] for r in res):.4f}  "
          f"rows whose greedy continuation changes when blanked: "
          f"{sum(1 for r in res if r['blank_distinct'])}/{len(res)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
