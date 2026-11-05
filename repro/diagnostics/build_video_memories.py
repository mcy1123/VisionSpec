#!/usr/bin/env python3
"""Cache raw Qwen3-VL vision-encoder tokens for video rows (fp16, CPU).

Used by the video visual-adaptor / FOVEA experiments. The vision encoder runs in
bf16 to fit on a 24 GB card; no gradients and no target LLM forward.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["train", "fresh"], required=True)
    ap.add_argument("--pool", default="/mnt/local3/mcy/GLANCE/results/video_caption_opd_20261026/data_stage2/pool_video_caption.parquet")
    ap.add_argument("--fresh30", default="/mnt/local/mcy/GLANCE/data/video_fresh30.parquet")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"
    torch.set_grad_enabled(False)
    rows = pd.read_parquet(a.pool if a.kind == "train" else a.fresh30)
    if a.limit:
        rows = rows.iloc[:a.limit]
    target, proc = common.load_target(device=dev, dtype=torch.bfloat16)
    target.eval().requires_grad_(False)
    print(f"[vvid] target bf16 loaded; {len(rows)} rows", flush=True)
    out, t0 = {}, time.time()
    for i, row in enumerate(rows.to_dict("records")):
        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"},
                                          {"type": "text",
                                           "text": row.get("prompt") or row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        with torch.inference_mode():
            v = target.model.visual(
                hidden_states=ins["pixel_values_videos"].to(dev),
                grid_thw=ins["video_grid_thw"].to(dev))[0]
        key = row.get("video_name") or row["task"]
        out[key] = v.detach().to(torch.float16).cpu()
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[vvid] {i+1}/{len(rows)} {key} shape={tuple(v.shape)} "
                  f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
        del v, ins
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 1, "kind": a.kind, "features": out}, path)
    print("wrote", path, "rows", len(out), "minutes", (time.time() - t0) / 60,
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
