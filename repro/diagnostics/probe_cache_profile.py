#!/usr/bin/env python3
"""Where exactly do the cached and uncached target forwards differ?

`probe_cache_path_equivalence.py` found the last-position hidden state on a video
row differing by max|Δ| 74-126 (cosine 0.72-0.74) between `use_cache=True` (what
every decode path uses) and `use_cache=False` (what `common.target_context` -- the
training path for both SFT and OPD -- uses), with explicit and internally-computed
position_ids agreeing inside each group.

That result is only interpretable with the positional profile, so this measures:
  * the per-position max|Δ| profile over the whole sequence for one video row,
    plus the share of positions whose max|Δ| exceeds 1;
  * the same four variants on an IMAGE row, to see whether this is a video-specific
    property (the image side is a reproduced, paper-matching pipeline).

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


def fwd(target, ids, explicit_pos, use_cache, pv=None, vg=None, pix=None, grid=None):
    pos = None
    if explicit_pos:
        pos, _ = target.model.get_rope_index(ids, grid, vg, None)
    kw = {}
    if pos is not None:
        kw["position_ids"] = pos
    if use_cache:
        kw["past_key_values"] = DynamicCache()
        kw["cache_position"] = torch.arange(ids.shape[1], device=ids.device)
    if pv is not None:
        kw.update(pixel_values_videos=pv, video_grid_thw=vg)
    if pix is not None:
        kw.update(pixel_values=pix, image_grid_thw=grid)
    target.model.rope_deltas = None
    with torch.no_grad():
        o = target(ids, use_cache=use_cache, logits_to_keep=1,
                   output_hidden_states=True, **kw)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/mnt/local/mcy/GLANCE/data/pool_499.parquet")
    ap.add_argument("--image", default="/mnt/local/mcy/GLANCE/datasets/coco/train2017")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    target, proc = common.load_target(device=dev)
    rep = {}

    # ---------------- video rows: positional profile ----------------
    import pandas as pd
    pool = pd.read_parquet(a.pool).head(2)
    vtext = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": pool.iloc[0]["question"]}]}],
        add_generation_prompt=True, tokenize=False)
    vids = []
    for _, row in pool.iterrows():
        ins = proc(text=[vtext], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        vids.append((str(row["video_name"]), ins["input_ids"].to(dev),
                     ins["pixel_values_videos"].to(dev), ins["video_grid_thw"].to(dev)))
    prof = []
    for name, ids, pv, vg in vids:
        oA = fwd(target, ids, True, True, pv=pv, vg=vg)
        oB = fwd(target, ids, True, False, pv=pv, vg=vg)
        hA = oA.hidden_states[-1][0].float()
        hB = oB.hidden_states[-1][0].float()
        d = (hA - hB).abs().max(dim=-1).values
        n = d.shape[0]
        lgA = oA.logits[0, -1].float()
        lgB = oB.logits[0, -1].float()
        rec = {"video": name, "L": n,
               "max_abs_delta_all": float(d.max()),
               "frac_pos_delta_gt1": float((d > 1).float().mean()),
               "delta_last10": [float(x) for x in d[-10:]],
               "delta_first5": [float(x) for x in d[:5]],
               "delta_mid": float(d[n // 2]),
               "argmax_cache": int(lgA.argmax()), "argmax_nocache": int(lgB.argmax()),
               "cos_last": float(F.cosine_similarity(hA[-1], hB[-1], dim=0)),
               "frac_hidden_cos_gt_0.999": float(
                   F.cosine_similarity(hA, hB, dim=-1).gt(0.999).float().mean())}
        prof.append(rec)
        print(f"  VIDEO {name[:14]:14s} L={n} max|Δh|={rec['max_abs_delta_all']:.1f} "
              f"share(pos Δh>1)={rec['frac_pos_delta_gt1']:.3f} "
              f"last10={[round(x,1) for x in rec['delta_last10']]} "
              f"argmax {rec['argmax_cache']} vs {rec['argmax_nocache']}", flush=True)
    rep["video_profile"] = prof

    # ---------------- image row: same four variants ----------------
    import glob
    from PIL import Image
    img_path = sorted(glob.glob(os.path.join(a.image, "*.jpg")))[0]
    image = common.load_image(img_path, 768)
    ids, pix, grid = common.encode(proc, common.PROMPT_CAPTION, image)
    ids, pix, grid = ids.to(dev), pix.to(dev), grid.to(dev)
    img = {}
    for label, (ep, uc) in {"A_explicit_cache": (True, True),
                            "B_explicit_nocache": (True, False),
                            "C_internal_cache": (False, True),
                            "D_internal_nocache": (False, False)}.items():
        o = fwd(target, ids, ep, uc, pix=pix, grid=grid)
        img[label] = {"argmax": int(o.logits[0, -1].argmax()),
                      "hidden_last": o.hidden_states[-1][0, -1].float().cpu()}
    hA = img["A_explicit_cache"]["hidden_last"]
    for label, v in img.items():
        v["hidden_maxabs_vs_A"] = float((v["hidden_last"] - hA).abs().max())
        v["hidden_cos_vs_A"] = float(F.cosine_similarity(v["hidden_last"], hA, dim=0))
        del v["hidden_last"]
    rep["image"] = {"path": img_path, "L": int(ids.shape[1]),
                    "variants": img,
                    "argmax_all_equal": len({v["argmax"] for v in img.values()}) == 1}
    print(f"  IMAGE {os.path.basename(img_path)} L={int(ids.shape[1])} argmax "
          f"A/B/C/D = {[img[k]['argmax'] for k in ('A_explicit_cache','B_explicit_nocache','C_internal_cache','D_internal_nocache')]} "
          f"all_equal={rep['image']['argmax_all_equal']} "
          f"max|Δh| nocache-vs-cache {img['B_explicit_nocache']['hidden_maxabs_vs_A']:.4f} "
          f"cos {img['B_explicit_nocache']['hidden_cos_vs_A']:.6f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
