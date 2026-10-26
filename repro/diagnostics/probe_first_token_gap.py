#!/usr/bin/env python3
"""Is the first generated token actually near-tied on video prompts?

`control_ar_selfconsistency.py` found that two pure-AR paths (KV-cache
incremental vs full re-prefill) disagree at generated token 0 on 4/4 video rows
in bf16, with no drafter in the loop. That is only credible if the first token is
genuinely flat: the pool's single prompt ("Please provide a detailed description
of the video, ...") opens a free-form caption, where the first token is a choice
among many near-equal continuations.

This measures it directly: one forward per row, report the top-1/top-2 logit gap
and the top-5, then mark which of the three path choices (SPEC, AR-KV, AR
re-prefill) fall inside that top-5. A small gap plus all three choices inside the
top-5 is the near-tie explanation; a large gap or a choice outside the top-5 is
not, and would be a bug.
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

import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
import glance
from glance.decode import greedy_generate
from glance.vl import VisionPositions
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.rows).head(a.n)
    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev).eval()
    eos = proc.tokenizer.eos_token_id

    out = []
    for i, row in df.reset_index(drop=True).iterrows():
        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"},
                                          {"type": "text", "text": row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        ids = ins["input_ids"].to(dev)
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        n_in = int(ids.shape[1])

        # Canonical forward for the prompt's next-token distribution.
        pos, _ = target.model.get_rope_index(ids, None, vg, None)
        with torch.no_grad():
            o = target(ids, position_ids=pos, pixel_values_videos=pv,
                       video_grid_thw=vg, use_cache=False, logits_to_keep=1)
        lg = o.logits[0, -1].float()
        top = torch.topk(lg, 5)
        top5 = [int(x) for x in top.indices.tolist()]
        vals = [float(x) for x in top.values.tolist()]

        # First committed token by each path.
        sp, _ = glance.generate(head, target, ids, 4, [eos], budget=63, branch_k=8,
                                temperature=0.0,
                                positions=VisionPositions(pixel_values_videos=pv,
                                                          video_grid_thw=vg),
                                return_stats=True, timing="decode_only")
        ar, _ = greedy_generate(target, ids, 4, [eos],
                                positions=VisionPositions(pixel_values_videos=pv,
                                                          video_grid_thw=vg),
                                return_stats=True)
        full = ids
        pos2, _ = target.model.get_rope_index(full, None, vg, None)
        with torch.no_grad():
            o2 = target(full, position_ids=pos2, pixel_values_videos=pv,
                        video_grid_thw=vg, use_cache=False, logits_to_keep=1)
        rep = int(o2.logits[0, -1].argmax())

        rec = {"video": str(row["video_name"]), "L": n_in,
               "top5": top5, "top5_logits": vals,
               "gap_top1_top2": float(top.values[0] - top.values[1]),
               "gap_top1_top5": float(top.values[0] - top.values[4]),
               "spec_first": int(sp[0, n_in]), "ar_kv_first": int(ar[0, n_in]),
               "ar_reprefill_first": rep,
               "spec_in_top5": int(sp[0, n_in]) in top5,
               "ar_kv_in_top5": int(ar[0, n_in]) in top5,
               "reprefill_in_top5": rep in top5,
               "reprefill_agrees_with_ref": rep == top5[0]}
        out.append(rec)
        print(f"  {rec['video'][:16]:16s} L={n_in:4d} top5={top5} "
              f"gap(1-2)={rec['gap_top1_top2']:.3f} gap(1-5)={rec['gap_top1_top5']:.3f} | "
              f"spec={rec['spec_first']}{'' if rec['spec_in_top5'] else ' OUT'} "
              f"ar_kv={rec['ar_kv_first']}{'' if rec['ar_kv_in_top5'] else ' OUT'} "
              f"reprefill={rep}{'' if rec['reprefill_in_top5'] else ' OUT'}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": out}, open(a.out, "w"), indent=1)
    ing = sum(1 for r in out if r["spec_in_top5"] and r["ar_kv_in_top5"]
              and r["reprefill_in_top5"])
    gaps = [r["gap_top1_top2"] for r in out]
    print(f"\nall three path choices inside the top-5: {ing}/{len(out)}; "
          f"top1-top2 gap median {sorted(gaps)[len(gaps)//2]:.3f} "
          f"(min {min(gaps):.3f}, max {max(gaps):.3f})")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
