#!/usr/bin/env python3
"""Build target visual-attention-guided per-token OPD weights.

For every scheduled QA, teacher-force the target's own greedy answer with eager
attention, measure the attention mass on image tokens for each answer token
(mean over heads and the last few layers, excluding the first image-token sink),
then turn the relative mass into w = 1 + alpha * clip(mass/mean - 1, 0, cap).

Data preparation only; no optimizer updates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import DynamicCache

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
from glance.vl import VisionPositions  # noqa: E402

IMAGE_TOKEN_ID = 151655


def encode_prompt(proc, prompt, image, dev):
    ids, pix, grid = common.encode(proc, prompt, image)
    return (ids.to(dev),
            pix.to(dev) if pix is not None else None,
            grid.to(dev) if grid is not None else None)


def attention_mass(target, ids, ans, pix, grid, dev, layers):
    pos = VisionPositions(pix, grid)
    pos.bind(target, ids)
    cache = DynamicCache()
    with torch.inference_mode():
        out = target.model(ids, past_key_values=cache, use_cache=True,
                           output_attentions=True,
                           **pos.ar_prefill_inputs(ids))
        cache = out.past_key_values
        index = torch.arange(ids.shape[1], ids.shape[1] + int(ans.numel()),
                             device=dev).unsqueeze(0)
        block = target.model(ans.unsqueeze(0), past_key_values=cache,
                             use_cache=True, output_attentions=True,
                             position_ids=pos(index), cache_position=index[0])
    vis = (ids[0] == IMAGE_TOKEN_ID).nonzero().squeeze(1)
    if vis.numel() < 2:
        raise RuntimeError("need at least two image tokens")
    keep = vis[1:]  # drop the first image-token attention sink
    masses = []
    for layer in layers:
        attn = block.attentions[layer]
        if attn is None:
            raise RuntimeError(f"layer {layer} returned no attention")
        a = attn[0, :, :, keep].mean(dim=0)  # [q, n_keep]
        masses.append(a.sum(dim=-1))
    return torch.stack(masses, dim=0).mean(dim=0)  # [n_answer]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020")
    ap.add_argument("--schedule", default="/mnt/local/mcy/GLANCE/results/visual_opd_20261102/schedule_u60.jsonl")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--layers", default="32,33,34,35",
                    help="comma-separated target layer indices (0-based transformer layers)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="debug only; 0 means all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"
    torch.set_grad_enabled(False)
    run = Path(a.run)
    labels = torch.load(run / "dataset" / "labels.pt", map_location="cpu",
                        weights_only=False)["records"]
    by_task = {r["task"]: r for r in labels}
    pool = pd.read_parquet(run / "dataset" / "pool.parquet").set_index("task")

    schedule = {}
    for line in Path(a.schedule).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        u = int(item["update"])
        if 0 <= u < a.updates:
            schedule.setdefault(u, []).append(item["task"])
    if sorted(schedule) != list(range(a.updates)):
        raise ValueError("schedule does not cover the requested updates")
    tasks = [task for u in range(a.updates) for task in schedule[u]]
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"[attnw] {len(tasks)} tasks layers={layers}", flush=True)

    target, proc = common.load_target(device=dev, dtype=torch.float32,
                                      attn_implementation="eager")
    target.eval().requires_grad_(False)
    print("[attnw] target loaded", flush=True)

    masses, weights = {}, {}
    t0 = time.time()
    for i, task in enumerate(tasks):
        row = pool.loc[task].to_dict()
        rec = by_task[task]
        ans = rec["input_ids"][rec["prompt_len"]:rec["prompt_len"] + rec["n_answer_tokens"]]
        ans = ans.to(dev, dtype=torch.long)
        img = Image.open(row["image_path"]).convert("RGB")
        if max(img.size) > 768:
            w, h = img.size
            scale = 768 / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.BICUBIC)
        ids, pix, grid = encode_prompt(proc, row["prompt"], img, dev)
        if ids.shape[1] != rec["prompt_len"]:
            raise RuntimeError(f"prompt mismatch {task}: {ids.shape[1]} vs {rec['prompt_len']}")
        mass = attention_mass(target, ids, ans, pix, grid, dev, layers).float().cpu()
        masses[task] = mass
        mean = mass.mean()
        if float(mean) > 0:
            rel = (mass / mean - 1.0).clamp(0.0, a.cap)
            w = 1.0 + a.alpha * rel
        else:
            w = torch.ones_like(mass)
        weights[task] = w
        del ids, pix, grid, ans, mass
        if (i + 1) % 5 == 0 or i == 0:
            m = masses[task]
            print(f"[attnw] {i+1}/{len(tasks)} {task} mass mean/min/max="
                  f"{float(m.mean()):.4f}/{float(m.min()):.4f}/{float(m.max()):.4f} "
                  f"w mean/min/max={float(weights[task].mean()):.3f}/"
                  f"{float(weights[task].min()):.3f}/{float(weights[task].max()):.3f} "
                  f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1,
        "run": str(run),
        "schedule": str(a.schedule),
        "updates": a.updates,
        "method": "target_eager_attention_mass_last_layers_excl_sink",
        "layers": layers,
        "alpha": a.alpha,
        "cap": a.cap,
        "weights": weights,
        "mass": masses,
        "wall_minutes": (time.time() - t0) / 60,
    }, out)
    print("wrote", out, "tasks", len(weights), "minutes", (time.time() - t0) / 60,
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
