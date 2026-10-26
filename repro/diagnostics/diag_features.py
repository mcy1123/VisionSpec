#!/usr/bin/env python3
"""Stage 4 of the visual-adaptation diagnosis: what the MLP projection preserves.

The draft head's only vision path is

    proj = hidden_norm(fc(cat(hidden_states[1,9,17,25,33])))

so this script asks what `proj` at VISUAL positions is still related to, on the
frozen target and one prefill per row (no decoding, no draft head involved):

  * to the raw visual tokens the target itself received (`hidden_states[0]`,
    i.e. the vision tower + merger output at those positions) -- if this is high,
    the projection still carries the visual embedding;
  * to the target's own final-layer hidden at those positions -- if this is high,
    the projection tracks the target's deep visual semantics;
  * the same two numbers at TEXT positions, as a within-model control: whatever
    the projection does to vision, it must not degrade text by comparison;
  * alignment of the projected visual mean with the text manifold, which is the
    "the MLP mapped vision into text space" hypothesis.

Both mean cosine and linear CKA are reported. CKA is invariant to invertible
linear maps and isotropic scaling, so it isolates shared structure rather than
shared basis; cosine is basis-sensitive, so the pair distinguishes "rotated but
intact" from "actually collapsed".

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
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
import glance
from diagnostics import diag_common as dc
from glance.vl import VisionPositions
from transformers import AutoProcessor


def cka(X, Y):
    """Linear CKA between two (n, d) float32 matrices."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    num = (xty * xty).sum()
    den = (torch.linalg.matrix_norm(xtx, "fro")
           * torch.linalg.matrix_norm(yty, "fro"))
    if float(den) == 0.0:
        return float("nan")
    return float(num / den)


def cos_mean(A, B):
    a = F.normalize(A.float(), dim=-1)
    b = F.normalize(B.float(), dim=-1)
    return float((a * b).sum(-1).mean())


def cos_vec(a, b):
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-vis", type=int, default=2048,
                    help="cap on visual positions used for the (d x d) CKA blocks")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.rows)
    if a.n:
        df = df.head(a.n)
    print(f"features: rows {a.rows} n={len(df)} head {a.head} max_vis {a.max_vis}",
          flush=True)

    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev).eval()
    ctx = dc.WideContext(target)
    D = head.config.hidden_size
    n_layers = len(head.target_layer_ids)
    cls = proc.tokenizer

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
        pos.bind(target, ids)
        with torch.no_grad():
            out = target(ids, position_ids=pos._prefill_positions,
                         pixel_values_videos=pv, video_grid_thw=vg,
                         use_cache=False, output_hidden_states=True,
                         logits_to_keep=1)
            wide = glance.decode.context_feature(out.hidden_states,
                                                 head.target_layer_ids)
            base = wide[..., : n_layers * D]
            raw = wide[..., n_layers * D:(n_layers + 1) * D]
            last = wide[..., (n_layers + 1) * D:(n_layers + 2) * D]
            vis = wide[..., (n_layers + 2) * D:(n_layers + 2) * D + 1] > 0.5
            proj = head.hidden_norm(head.fc(base))

            v = vis[0, :, 0]
            x = ~v
            pv_v = proj[0][v]
            pv_x = proj[0][x]
            raw_v = raw[0][v]
            raw_x = raw[0][x]
            last_v = last[0][v]
            last_x = last[0][x]

            # deterministic subsample of visual rows for the d x d CKA blocks
            sub = None
            if pv_v.shape[0] > a.max_vis:
                g = torch.Generator(device="cpu").manual_seed(0)
                sub = torch.randperm(pv_v.shape[0], generator=g)[:a.max_vis].to(pv_v.device)

            def ck(A, B):
                if sub is not None:
                    A, B = A[sub], B[sub]
                return cka(A, B)

            mt_pv = pv_v.mean(0)
            mt_px = pv_x.mean(0)
            mt_lx = last_x.mean(0)
            mt_raw = raw_v.mean(0)
            rec = {
                "video": str(row["video_name"]), "L": n_in,
                "n_vis": int(v.sum()), "n_txt": int(x.sum()),
                "n_frames": int(row.get("n_frames", 0)),
                "fps": float(row.get("fps", 0.0)),
                # cosine
                "cos_proj_raw_vis": cos_mean(pv_v, raw_v),
                "cos_proj_last_vis": cos_mean(pv_v, last_v),
                "cos_proj_last_txt": cos_mean(pv_x, last_x),
                "cos_raw_last_vis": cos_mean(raw_v, last_v),
                "cos_projvis_projtxt_mean": cos_vec(mt_pv, mt_px),
                "cos_projvis_lasttxt_mean": cos_vec(mt_pv, mt_lx),
                "cos_rawvis_lasttxt_mean": cos_vec(mt_raw, mt_lx),
                # CKA
                "cka_proj_raw_vis": ck(pv_v, raw_v),
                "cka_proj_last_vis": ck(pv_v, last_v),
                "cka_proj_last_txt": ck(pv_x, last_x),
                "cka_proj_raw_txt": ck(pv_x, raw_x),
                # scale
                "proj_rms_vis": float(pv_v.pow(2).mean().sqrt()),
                "proj_rms_txt": float(pv_x.pow(2).mean().sqrt()),
                "last_rms_vis": float(last_v.pow(2).mean().sqrt()),
                "raw_rms_vis": float(raw_v.pow(2).mean().sqrt()),
                "wall_s": time.perf_counter() - t0,
            }
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:16]:16s} L={n_in:4d} "
              f"vis={rec['n_vis']:4d} cos(proj,raw)_vis={rec['cos_proj_raw_vis']:+.3f} "
              f"cos(proj,last)_vis={rec['cos_proj_last_vis']:+.3f} "
              f"cos(proj,last)_txt={rec['cos_proj_last_txt']:+.3f} "
              f"CKA(proj,raw)_vis={rec['cka_proj_raw_vis']:.3f} "
              f"CKA(proj,last)_vis={rec['cka_proj_last_vis']:.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": res}, open(a.out, "w"), indent=1)

    def agg(k):
        vals = [r[k] for r in res if r[k] == r[k]]
        return f"{st.median(vals):.4f}"

    print("\nmedian over rows:")
    for k in ["cos_proj_raw_vis", "cos_proj_last_vis", "cos_proj_last_txt",
              "cos_raw_last_vis", "cos_projvis_projtxt_mean",
              "cos_projvis_lasttxt_mean", "cos_rawvis_lasttxt_mean",
              "cka_proj_raw_vis", "cka_proj_last_vis", "cka_proj_last_txt",
              "cka_proj_raw_txt"]:
        print(f"  {k:32s} {agg(k)}")
    ctx.close()
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
