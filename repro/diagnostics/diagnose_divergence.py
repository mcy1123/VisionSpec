#!/usr/bin/env python3
"""Why bf16 AR and bf16 SPEC disagree: near-ties, or a real bug?

The repository's own audit runs the losslessness gate at fp32 and says why
(`experiments/audit_lossless.py`): in bf16 the target does not reproduce *itself*,
because a tree pass and a single-token pass reduce in different orders and
near-ties land on either side. This script turns that claim into a measurement on
the video held-out rows instead of quoting it.

For every row whose AR and SPEC streams diverge it does a THIRD, independent
forward -- teacher-forced on the AR prefix, full-sequence prefill -- and reports
the target's top-2 logits at the divergence position, the gap, and which token
each path committed. A near-tie is a gap far below the bf16 resolution of these
logits; a systematic difference between `argmax` and both paths would be a bug.

No try/except.
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
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch", type=int, default=8)
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
        ar, _ = greedy_generate(target, ids, a.max_new, [eos],
                                positions=VisionPositions(pixel_values_videos=pv,
                                                          video_grid_thw=vg),
                                return_stats=True)
        sp, _ = glance.generate(head, target, ids, a.max_new, [eos], budget=a.budget,
                                branch_k=a.branch, temperature=0.0,
                                positions=VisionPositions(pixel_values_videos=pv,
                                                          video_grid_thw=vg),
                                return_stats=True, timing="decode_only")
        ar_ids = ar[0, n_in:].tolist()
        sp_ids = sp[0, n_in:].tolist()
        m = min(len(ar_ids), len(sp_ids))
        j = next((t for t in range(m) if ar_ids[t] != sp_ids[t]), None)
        rec = {"video": str(row["video_name"]), "L": n_in, "ar_len": len(ar_ids),
               "spec_len": len(sp_ids), "first_mismatch": j}
        if j is None:
            rec["verdict"] = "agree" if len(ar_ids) == len(sp_ids) else "prefix-agree"
            out.append(rec)
            print(f"  {rec['video'][:16]:16s} AGREE (ar {len(ar_ids)} spec {len(sp_ids)})",
                  flush=True)
            continue
        # Independent third forward: teacher-forced on the AR prefix.
        full = torch.cat([ids, ar[0, n_in:n_in + j][None]], dim=1)
        pos, _ = target.model.get_rope_index(full, None, vg, None)
        with torch.no_grad():
            o = target(full, position_ids=pos, pixel_values_videos=pv,
                       video_grid_thw=vg, use_cache=False, logits_to_keep=1)
        lg = o.logits[0, -1].float()
        top = torch.topk(lg, 4)
        rec.update({
            "ar_tok": int(ar_ids[j]), "spec_tok": int(sp_ids[j]) if j < len(sp_ids) else None,
            "ref_argmax": int(top.indices[0]),
            "ref_top1": float(top.values[0]), "ref_top2": float(top.values[1]),
            "gap_top1_top2": float(top.values[0] - top.values[1]),
            "ref_logit_ar": float(lg[int(ar_ids[j])]),
            "ref_logit_spec": float(lg[int(sp_ids[j])]) if j < len(sp_ids) else None,
            "ref_top4": [int(x) for x in top.indices.tolist()],
            "ref_top4_vals": [float(x) for x in top.values.tolist()],
        })
        rec["verdict"] = ("near-tie (both in top-4)" if int(ar_ids[j]) in rec["ref_top4"]
                          and rec["spec_tok"] in rec["ref_top4"] else "systematic?")
        out.append(rec)
        print(f"  {rec['video'][:16]:16s} diverge@{j:4d}  ar={rec['ar_tok']:6d} "
              f"spec={rec['spec_tok']}  ref_argmax={rec['ref_argmax']:6d} "
              f"gap(top1-top2)={rec['gap_top1_top2']:.3f} "
              f"|logp_ar-logp_spec|={abs(rec['ref_logit_ar']-rec['ref_logit_spec']):.3f} "
              f"{rec['verdict']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": out}, open(a.out, "w"), indent=1)
    ties = [r for r in out if r.get("verdict", "").startswith("near-tie")]
    bad = [r for r in out if r.get("verdict") == "systematic?"]
    print(f"\nnear-tie {len(ties)}/{len(out)}   systematic? {len(bad)}/{len(out)}   "
          f"agree {sum(1 for r in out if r.get('verdict') in ('agree','prefix-agree'))}")
    if bad:
        print(f"  !! rows needing a real look: {[r['video'] for r in bad]}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
