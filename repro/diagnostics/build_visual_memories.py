#!/usr/bin/env python3
"""Cache raw Qwen3-VL vision-encoder features for the visual-adaptor head.

For each scheduled image QA (or the test split), run the frozen target's vision
encoder on the same resized image the target sees and save the post-merger visual
tokens [N_vis, 4096] (fp16, CPU). Data preparation only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402


def load_rows(run: Path, kind: str, split: str, schedule: str | None):
    if kind == "pool":
        pool = pd.read_parquet(run / "dataset" / "pool.parquet").set_index("task")
        if schedule:
            tasks = []
            for line in Path(schedule).read_text().splitlines():
                if line.strip():
                    tasks.append(json.loads(line)["task"])
            # preserve order, unique
            seen = set()
            tasks = [t for t in tasks if not (t in seen or seen.add(t))]
            return [(t, pool.loc[t].to_dict()) for t in tasks]
        return [(t, pool.loc[t].to_dict()) for t in pool.index]
    rows = []
    with open(run / "dataset" / "records_manifest.csv") as fh:
        for r in csv.DictReader(fh):
            if r["canonical_split"] == split:
                rows.append(r)
    rows.sort(key=lambda r: int(r["official_row_index"]))
    return [(r["pid"], r) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020")
    ap.add_argument("--kind", choices=["pool", "split"], default="pool")
    ap.add_argument("--split", default="test")
    ap.add_argument("--schedule", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"
    torch.set_grad_enabled(False)
    run = Path(a.run)
    rows = load_rows(run, a.kind, a.split, a.schedule)
    if a.limit:
        rows = rows[: a.limit]
    target, proc = common.load_target(device=dev, dtype=torch.float32)
    target.eval().requires_grad_(False)
    print(f"[vmem] target loaded; {len(rows)} rows", flush=True)
    out, t0 = {}, time.time()
    for i, (key, row) in enumerate(rows):
        img = Image.open(row["image_path"]).convert("RGB")
        if max(img.size) > 768:
            w, h = img.size
            s = 768 / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))),
                             Image.BICUBIC)
        _, pix, grid = common.encode(proc, row["prompt"], img)
        with torch.inference_mode():
            v = target.model.visual(hidden_states=pix.to(dev),
                                    grid_thw=grid.to(dev))[0]
        out[key] = v.detach().to(torch.float16).cpu()
        if (i + 1) % 25 == 0 or i == 0:
            print(f"[vmem] {i+1}/{len(rows)} {key} shape={tuple(v.shape)} "
                  f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
        del v
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 1, "kind": a.kind, "split": a.split,
                "features": out}, path)
    print("wrote", path, "rows", len(out), "minutes", (time.time() - t0) / 60,
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
