#!/usr/bin/env python3
"""Do the target's cached and uncached prefills agree on a video row?

`probe_first_token_gap.py` left one thing unexplained: on the same encoded video
row, the first-token argmax from a prefill run with `use_cache=False` differed
from the argmax of the prefill that `glance.generate` and `greedy_generate`
actually run (`use_cache=True`), by more than the top1-top2 gap. Every training
path in this project (`common.target_context`) uses `use_cache=False`, while every
decode path uses a KV cache, so if the two disagree the head is trained on states
it never sees.

This isolates it: four forwards on the same row, varying only
`use_cache` and whether `position_ids` is passed explicitly.

  A: position_ids=explicit, use_cache=True
  B: position_ids=explicit, use_cache=False
  C: position_ids=None (model computes from video_grid_thw), use_cache=True
  D: position_ids=None, use_cache=False

Reports each argmax, the top-5, and the last-position hidden state's max|Δ| and
cosine against A. No try/except: a shape error is a finding, not a nuisance.
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
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from transformers import AutoProcessor


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(F.cosine_similarity(a, b, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.rows).head(a.n)
    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    from transformers.cache_utils import DynamicCache

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
        pos, deltas = target.model.get_rope_index(ids, None, vg, None)
        cp = torch.arange(n_in, device=dev)

        variants = {}
        for name, (use_pos, use_cache) in {
                "A_explicit_cache": (True, True),
                "B_explicit_nocache": (True, False),
                "C_internal_cache": (False, True),
                "D_internal_nocache": (False, False)}.items():
            kw = {}
            if use_pos:
                kw["position_ids"] = pos
            if use_cache:
                kw["past_key_values"] = DynamicCache()
                kw["cache_position"] = cp
            target.model.rope_deltas = None
            with torch.no_grad():
                o = target(ids, pixel_values_videos=pv, video_grid_thw=vg,
                           use_cache=use_cache, logits_to_keep=1,
                           output_hidden_states=True, **kw)
            lg = o.logits[0, -1].float()
            top = torch.topk(lg, 5)
            variants[name] = {
                "argmax": int(top.indices[0]),
                "top5": [int(x) for x in top.indices.tolist()],
                "top5_logits": [float(x) for x in top.values.tolist()],
                "hidden_last": o.hidden_states[-1][0, -1].float().cpu(),
            }
        ref = variants["A_explicit_cache"]
        rec = {"video": str(row["video_name"]), "L": n_in,
               "rope_delta": int(deltas[0, 0]), "variants": {}}
        for name, v in variants.items():
            rec["variants"][name] = {
                "argmax": v["argmax"], "top5": v["top5"],
                "top5_logits": v["top5_logits"],
                "hidden_maxabs_vs_A": float((v["hidden_last"] - ref["hidden_last"]).abs().max()),
                "hidden_cos_vs_A": cosine(v["hidden_last"], ref["hidden_last"])}
        rec["argmax_all_equal"] = len({v["argmax"] for v in variants.values()}) == 1
        rec["max_logit_gap_between_variants"] = max(
            abs(ref["top5_logits"][0] - v["top5_logits"][0]) for v in variants.values())
        out.append(rec)
        print(f"  {rec['video'][:16]:16s} L={n_in} rope_delta={rec['rope_delta']} "
              f"argmax A/B/C/D = {[variants[k]['argmax'] for k in ('A_explicit_cache','B_explicit_nocache','C_internal_cache','D_internal_nocache')]} "
              f"all_equal={rec['argmax_all_equal']} "
              f"maxΔlogit={rec['max_logit_gap_between_variants']:.3f}", flush=True)
        for name, v in rec["variants"].items():
            print(f"      {name:22s} argmax {v['argmax']:6d} top5 {v['top5']} "
                  f"hidden max|Δ| {v['hidden_maxabs_vs_A']:.4f} cos {v['hidden_cos_vs_A']:.6f}",
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": out}, open(a.out, "w"), indent=1)
    n_eq = sum(1 for r in out if r["argmax_all_equal"])
    print(f"\nall four variants agree on the argmax: {n_eq}/{len(out)}")
    for name in ("B_explicit_nocache", "C_internal_cache", "D_internal_nocache"):
        cos = [r["variants"][name]["hidden_cos_vs_A"] for r in out]
        mx = [r["variants"][name]["hidden_maxabs_vs_A"] for r in out]
        print(f"  {name:22s} vs A: cosine min {min(cos):.6f}  max|Δhidden| max {max(mx):.4f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
