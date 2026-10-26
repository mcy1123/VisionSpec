#!/usr/bin/env python3
"""Phase 0.1/0.2: quantify and localise the target's cache-dependent forward.

Fact to explain: on the same input, `use_cache=True` and `use_cache=False` give
different target hidden states. Every training/teacher path in this project
(`common.target_context`, the pool's `target_topk`) is uncached while every decode
path uses a KV cache, so the head is trained on states it does not see at decode.

This runs a fixed variant ladder on real rows of three different model/task
combinations and reports, per variant, the layer-by-layer difference against the
reference (which is exactly the call `glance.generate` makes for its prefill:
explicit position_ids + KV cache + no attention_mask):

  ref            explicit positions, use_cache=True            (decode prefill)
  exp_nocache    explicit positions, use_cache=False           (training path)
  int_cache      positions computed by the model, cache on
  int_nocache    positions computed by the model, cache off
  exp_nocache_2d explicit positions, no cache, 2-D all-visible attention mask
  exp_cache_2d   explicit positions, cache on, 2-D all-visible mask
  exp_cache_nocp explicit positions, cache on, no cache_position

The 2-D-mask variants force the mask-materialisation branch; if they move a
no-cache run onto the cache run's numbers (or vice versa) the root cause is the
attention-mask branch, not precision in general.

Cases: `image8b` (GLANCE target, COCO image), `video8b` (GLANCE target, cached
video row), `sqa2b` (Qwen3-VL-2B, a real ScienceQA row). `--dtype float32` tests
whether the whole thing is a bf16 artefact.

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

import torch
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache


def build_case(kind, proc, dev):
    """Returns (label, model_kwargs, extra) for one real row."""
    if kind == "video8b":
        import pandas as pd
        pool = pd.read_parquet("/mnt/local/mcy/GLANCE/data/pool_499.parquet").head(1)
        row = pool.iloc[0]
        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"},
                                          {"type": "text", "text": row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        return (f"video8b/{row['video_name']}",
                dict(input_ids=ins["input_ids"].to(dev),
                     pixel_values_videos=ins["pixel_values_videos"].to(dev),
                     video_grid_thw=ins["video_grid_thw"].to(dev)), None)
    if kind == "image8b":
        import glob
        from PIL import Image
        path = sorted(glob.glob("/mnt/local/mcy/GLANCE/datasets/coco/train2017/*.jpg"))[0]
        image = common.load_image(path, 768)
        ids, pix, grid = common.encode(proc, common.PROMPT_CAPTION, image)
        return (f"image8b/{os.path.basename(path)}",
                dict(input_ids=ids.to(dev), pixel_values=pix.to(dev),
                     image_grid_thw=grid.to(dev)), None)
    if kind == "sqa2b":
        from PIL import Image
        base = "/home/mcy/Projects/Draft-OPD/opd_multimodal"
        line = open(f"{base}/data_sqa/qa.jsonl").readline()
        row = json.loads(line)
        image = Image.open(os.path.join(base, row["image"])).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": row["question"]}]}]
        ins = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt")
        kw = {"input_ids": ins["input_ids"].to(dev)}
        if ins.get("pixel_values") is not None:
            kw["pixel_values"] = ins["pixel_values"].to(dev)
            kw["image_grid_thw"] = ins["image_grid_thw"].to(dev)
        return f"sqa2b/{os.path.basename(row['image'])}", kw, {"question": row["question"][:80]}
    raise ValueError(f"unknown case {kind!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["image8b", "video8b", "sqa2b"])
    ap.add_argument("--model", default=None, help="default: the GLANCE 8B target")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    path = a.model or common.TARGET_PATH
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    target = common.load_target(path, device=dev, dtype=dt,
                                attn_implementation=a.attn)[0]
    proc = AutoProcessor.from_pretrained(path)
    label, kw, extra = build_case(a.case, proc, dev)
    ids = kw["input_ids"]
    L = int(ids.shape[1])
    print(f"case {label}  L={L}  dtype={a.dtype} attn={a.attn} model={path}",
          flush=True)

    grid_v = kw.get("video_grid_thw")
    grid_i = kw.get("image_grid_thw")
    pos, deltas = target.model.get_rope_index(ids, grid_i, grid_v, None)
    ones2d = torch.ones((1, L), dtype=torch.long, device=dev)

    def run(explicit_pos, use_cache, cache_position, mask2d, use_cache_kw):
        k = dict(kw)
        if explicit_pos:
            k["position_ids"] = pos
        if use_cache_kw:
            k["past_key_values"] = DynamicCache()
            if cache_position:
                k["cache_position"] = torch.arange(L, device=dev)
        if mask2d:
            k["attention_mask"] = ones2d
        target.model.rope_deltas = None
        with torch.no_grad():
            o = target(use_cache=use_cache, logits_to_keep=1,
                       output_hidden_states=True, **k)
        return o

    ref = run(True, True, False, False, True)      # exactly glance.generate's prefill
    variants = {
        "ref_decode_prefill": ref,
        "exp_nocache": run(True, False, False, False, False),
        "int_cache": run(False, True, True, False, True),
        "int_nocache": run(False, False, False, False, False),
        "exp_nocache_2d": run(True, False, False, True, False),
        "exp_cache_2d": run(True, True, True, True, True),
        "exp_cache_nocp": run(True, True, False, False, True),
    }

    n_layers = len(ref.hidden_states) - 1
    keep = [i for i in common.KEPT_LAYERS if i + 1 < len(ref.hidden_states)]
    ref_ctx = torch.cat([ref.hidden_states[i + 1][0] for i in keep], dim=-1).float()

    rep = {"case": label, "L": L, "dtype": a.dtype, "attn": a.attn, "model": path,
           "n_layers": n_layers, "rope_delta": int(deltas[0, 0]),
           "keep_layers": keep, "variants": {}}
    for name, o in variants.items():
        ha = o.hidden_states[-1][0].float()
        hb = ref.hidden_states[-1][0].float()
        ctx = torch.cat([o.hidden_states[i + 1][0] for i in keep], dim=-1).float()
        lg = o.logits[0, -1].float()
        top2 = torch.topk(lg, 2)
        rec = {"last_pos_max_abs": float((ha[-1] - hb[-1]).abs().max()),
               "last_pos_cos": float(F.cosine_similarity(ha[-1], hb[-1], dim=0)),
               "all_pos_max_abs": float((ha - hb).abs().max()),
               "all_pos_cos": float(F.cosine_similarity(ha.flatten(), hb.flatten(), dim=0)),
               "ctx_last_max_abs": float((ctx[-1] - ref_ctx[-1]).abs().max()),
               "ctx_last_cos": float(F.cosine_similarity(ctx[-1], ref_ctx[-1], dim=0)),
               "argmax": int(top2.indices[0]),
               "gap_top1_top2": float(top2.values[0] - top2.values[1]),
               "layer_profile": {str(i): float((o.hidden_states[i][0, -1].float()
                                                - ref.hidden_states[i][0, -1].float()).abs().max())
                                 for i in [0, 1, 2, 4, 7, 11, 17, 25, 33, n_layers]
                                 if i <= n_layers}}
        rep["variants"][name] = rec
        print(f"  {name:20s} lastpos max|Δ| {rec['last_pos_max_abs']:9.3f} "
              f"cos {rec['last_pos_cos']:.6f} | allpos max|Δ| {rec['all_pos_max_abs']:10.2f} "
              f"cos {rec['all_pos_cos']:.4f} | ctx cos {rec['ctx_last_cos']:.6f} "
              f"| argmax {rec['argmax']} gap {rec['gap_top1_top2']:.3f}", flush=True)
    print("  layer profile (last-pos max|Δ| vs ref):", flush=True)
    for name in ("exp_nocache", "exp_nocache_2d", "exp_cache_2d", "int_nocache"):
        r = rep["variants"][name]
        print(f"    {name:18s} " + " ".join(
            f"L{k}:{v:.2f}" for k, v in r["layer_profile"].items()), flush=True)

    if extra:
        rep["extra"] = extra
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
