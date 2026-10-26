#!/usr/bin/env python3
"""Projection-fidelity metrics for one (head, arm), on the path decode uses.

This is the pitfall-3 fix. The first version of this measurement ran a single
`use_cache=False` full-sequence forward. On Qwen3-VL that path makes the TEXT
positions independent of the video (see `probe_context_cache_path.py`: blanking
the video leaves text rows bit-identical, relL2 0.0000, cos 1.0000), so the
numbers it produced described a forward pass that decoding never performs. The
default here is `--feature-path cached`, which reproduces exactly what
`glance.decode.generate` does at prefill -- `use_cache=True`, a fresh
`DynamicCache`, explicit M-RoPE positions, `output_hidden_states=True`, and the
context taken through `glance.head.context_feature`. `uncached` is kept only so
the size of the discrepancy can be measured.

Metrics, all reported per row and summarised as medians:

  * `proj` is arm A's KV source `hidden_norm(fc(cat(layers)))`, or arm V's
    adapter output -- whatever the arm actually feeds to `k_proj`/`v_proj`.
  * cosine and linear CKA between `proj` at VISUAL positions and (a) the raw
    visual embeddings the target received, (b) the target's final-layer hidden.
  * the same at TEXT positions, as a within-model control.
  * alignment of the projected visual mean with the text manifold.

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
from historical import vispec_adapter as va
from glance.head import context_feature
from transformers import AutoProcessor, DynamicCache


def cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    den = torch.linalg.matrix_norm(xtx, "fro") * torch.linalg.matrix_norm(yty, "fro")
    if float(den) == 0.0:
        return float("nan")
    return float((xty * xty).sum() / den)


def cos_mean(A, B):
    a = F.normalize(A.float(), dim=-1)
    b = F.normalize(B.float(), dim=-1)
    return float((a * b).sum(-1).mean())


def cos_vec(a, b):
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


@torch.no_grad()
def forward_states(target, ids, pv, vg, path):
    """The five-layer fused context, plus the embedding and final-layer states."""
    if path == "cached":
        out = target(ids, use_cache=True, past_key_values=DynamicCache(),
                     output_hidden_states=True, logits_to_keep=1,
                     pixel_values_videos=pv, video_grid_thw=vg,
                     position_ids=target.model.get_rope_index(
                         ids, None, vg, None)[0])
    else:
        out = target(ids, use_cache=False, output_hidden_states=True,
                     logits_to_keep=1, pixel_values_videos=pv, video_grid_thw=vg,
                     position_ids=target.model.get_rope_index(
                         ids, None, vg, None)[0])
    return out.hidden_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["mlp", "vispec"])
    ap.add_argument("--head", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--feature-path", default="cached", choices=["cached", "uncached"])
    ap.add_argument("--max-vis", type=int, default=2048)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.rows)
    if a.n:
        df = df.head(a.n)
    print(f"CKA arm={a.arm} path={a.feature_path} head={a.head} rows={len(df)}",
          flush=True)

    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev).eval()
    if a.arm == "vispec":
        va.load_vispec(head, a.head)
    cfg = json.load(open(os.path.join(common.TARGET_PATH, "config.json")))
    vid, iid = int(cfg["video_token_id"]), int(cfg["image_token_id"])
    lid = head.target_layer_ids
    D = head.config.hidden_size

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
        hs = forward_states(target, ids, pv, vg, a.feature_path)
        fused = context_feature(hs, lid)[0]                 # (T, 5D)
        raw = hs[0][0]
        last = hs[-1][0]
        v = ((ids == vid) | (ids == iid))[0]
        x = ~v
        if a.arm == "vispec":
            ctx_in = va.build_wide(fused[None], v[None])
        else:
            ctx_in = fused[None]
        proj = head.hidden_norm(head.fc(ctx_in))[0]         # (T, D) -- the arm's KV source

        pv_v, pv_x = proj[v], proj[x]
        raw_v, raw_x = raw[v], raw[x]
        last_v, last_x = last[v], last[x]
        sub = None
        if pv_v.shape[0] > a.max_vis:
            g = torch.Generator().manual_seed(0)
            sub = torch.randperm(pv_v.shape[0], generator=g)[:a.max_vis].to(pv_v.device)

        def ck(A, B):
            return cka(A[sub] if sub is not None else A, B[sub] if sub is not None else B)

        rec = {
            "video": str(row["video_name"]), "L": int(ids.shape[1]),
            "n_vis": int(v.sum()), "n_txt": int(x.sum()),
            "n_frames": int(row.get("n_frames", 0)),
            "cos_proj_raw_vis": cos_mean(pv_v, raw_v),
            "cos_proj_last_vis": cos_mean(pv_v, last_v),
            "cos_proj_last_txt": cos_mean(pv_x, last_x),
            "cos_raw_last_vis": cos_mean(raw_v, last_v),
            "cos_projvis_projtxt_mean": cos_vec(pv_v.mean(0), pv_x.mean(0)),
            "cos_projvis_lasttxt_mean": cos_vec(pv_v.mean(0), last_x.mean(0)),
            "cos_rawvis_lasttxt_mean": cos_vec(raw_v.mean(0), last_x.mean(0)),
            "cka_proj_raw_vis": ck(pv_v, raw_v),
            "cka_proj_last_vis": ck(pv_v, last_v),
            "cka_proj_last_txt": ck(pv_x, last_x),
            "cka_proj_raw_txt": ck(pv_x, raw_x),
            "proj_rms_vis": float(pv_v.pow(2).mean().sqrt()),
            "proj_rms_txt": float(pv_x.pow(2).mean().sqrt()),
            "wall_s": time.perf_counter() - t0,
        }
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:16]:16s} L={rec['L']:4d} "
              f"vis={rec['n_vis']:4d} | CKA vis: raw {rec['cka_proj_raw_vis']:.3f} "
              f"last {rec['cka_proj_last_vis']:.3f} | CKA txt: raw "
              f"{rec['cka_proj_raw_txt']:.3f} last {rec['cka_proj_last_txt']:.3f}",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "arm": a.arm, "feature_path": a.feature_path,
               "rows": res}, open(a.out, "w"), indent=1)
    print("\nmedian over rows:")
    for k in ["cos_proj_raw_vis", "cos_proj_last_vis", "cos_proj_last_txt",
              "cos_projvis_projtxt_mean", "cos_projvis_lasttxt_mean",
              "cka_proj_raw_vis", "cka_proj_last_vis", "cka_proj_last_txt",
              "cka_proj_raw_txt", "proj_rms_vis", "proj_rms_txt"]:
        print(f"  {k:30s} {st.median(r[k] for r in res):+.4f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
