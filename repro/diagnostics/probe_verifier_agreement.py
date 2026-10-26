#!/usr/bin/env python3
"""Which forward actually tracks the *verifier's* decisions?

Phase 0 found the two attention branches. But a real generation uses BOTH: the
prefill takes `is_causal` (KV cache present, no mask) while the tree verify takes
the *materialised* branch (an explicit 4-D ancestor mask). So "align training to
decode" is not one configuration, and which one the teacher should use is an
empirical question.

This measures it directly. Take the token sequence a real bf16 SPEC run committed
(recorded in the lossless JSONs), teacher-force it through single-sequence
forwards, and ask how often each forward's argmax reproduces the token the
verifier actually committed. The tree pass is the reference, so:

  * high agreement  -> the single-sequence forward is a faithful teacher for the
                       verifier's decisions, and its branch choice barely matters;
  * low agreement   -> the verify pass's packed layout is a first-order effect and
                       NO single-sequence teacher can be trusted.

Both branch variants are measured, because that decides whether the phase-0 fix
points the teacher at the right branch.
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
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="/mnt/local/mcy/GLANCE/data/video_fresh30.parquet")
    ap.add_argument("--lossless-json", default="/mnt/local/mcy/GLANCE/results/opd_fix/"
                    "lossless_bf16_fresh30_shard0.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    spec = {r["video"]: r for r in json.load(open(a.lossless_json))["rows"]
            if r.get("spec_ids")}
    df = pd.read_parquet(a.rows)
    df = df[df["video_name"].astype(str).isin(spec)].head(a.n)
    target = common.load_target(device=dev)[0]
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)

    res = []
    for i, row in df.reset_index(drop=True).iterrows():
        rec = spec[str(row["video_name"])]
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
        committed = rec["spec_ids"]
        full = torch.cat([ids, torch.tensor([committed[:-1]], device=dev)], dim=1)
        pos, _ = target.model.get_rope_index(full, None, vg, None)
        out = {}
        for name, (use_cache, mask2d) in {"materialised(nocache,nomask)": (False, False),
                                          "is_causal(2D mask)": (False, True),
                                          "is_causal(cache)": (True, False)}.items():
            kw = dict(pixel_values_videos=pv, video_grid_thw=vg, position_ids=pos,
                      logits_to_keep=0)
            if use_cache:
                kw.update(past_key_values=DynamicCache(),
                          cache_position=torch.arange(full.shape[1], device=dev))
            if mask2d:
                kw["attention_mask"] = torch.ones_like(full)
            target.model.rope_deltas = None
            with torch.no_grad():
                o = target(full, **kw)
            lg = o.logits[0].float()
            # row n_in-1+j predicts committed[j]; the verifier committed exactly that token
            pred = lg[n_in - 1: n_in - 1 + len(committed)]
            am = pred.argmax(-1).tolist()
            hit = sum(1 for j, t in enumerate(committed) if am[j] == t)
            # also: is the committed token inside this forward's top-5?
            top5 = torch.topk(pred, 5, dim=-1).indices.tolist()
            in5 = sum(1 for j, t in enumerate(committed) if t in top5[j])
            out[name] = {"argmax_agreement": hit / len(committed),
                         "in_top5": in5 / len(committed),
                         "n": len(committed)}
        res.append({"video": str(row["video_name"]), "n_gen": len(committed),
                    "forwards": out})
        print(f"  {str(row['video_name'])[:16]:16s} gen={len(committed):4d} " +
              " | ".join(f"{k.split('(')[0]} argmax {v['argmax_agreement']:.4f} "
                         f"top5 {v['in_top5']:.4f}" for k, v in out.items()), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": res}, open(a.out, "w"), indent=1)
    import statistics as st
    for k in res[0]["forwards"]:
        print(f"mean {k:26s} argmax-agreement "
              f"{st.fmean(r['forwards'][k]['argmax_agreement'] for r in res):.4f}  "
              f"top5 {st.fmean(r['forwards'][k]['in_top5'] for r in res):.4f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
