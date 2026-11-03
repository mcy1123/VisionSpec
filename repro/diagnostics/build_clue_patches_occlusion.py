#!/usr/bin/env python3
"""Occlusion-based clue patches for clue-targeted cutout.

For each scheduled QA, measure the target's mean log-prob of its own greedy
answer under a G x G grid of gray occlusions and keep the cell with the largest
drop. The selected cell is the visual region the frozen target relies on most.

Data preparation only: no optimizer updates, no checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import DynamicCache

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
from glance.vl import VisionPositions  # noqa: E402


def encode_prompt(proc, prompt, image, dev):
    ids, pix, grid = common.encode(proc, prompt, image)
    return (ids.to(dev),
            pix.to(dev) if pix is not None else None,
            grid.to(dev) if grid is not None else None)


def answer_logits(target, ids, pix, grid, ans, dev):
    pos = VisionPositions(pix, grid)
    pos.bind(target, ids)
    cache = DynamicCache()
    with torch.inference_mode():
        out = target.model(ids, past_key_values=cache, use_cache=True,
                           output_hidden_states=False,
                           **pos.ar_prefill_inputs(ids))
        cache = out.past_key_values
        first = target.lm_head(out.last_hidden_state[0, -1:])[0]
        n = int(ans.numel())
        if n > 1:
            index = torch.arange(ids.shape[1], ids.shape[1] + n - 1,
                                 device=dev).unsqueeze(0)
            block = target.model(ans[:-1].unsqueeze(0), past_key_values=cache,
                                 use_cache=True, output_hidden_states=False,
                                 position_ids=pos(index), cache_position=index[0])
            rest = target.lm_head(block.last_hidden_state[0])
        else:
            rest = torch.empty(0, first.numel(), device=dev, dtype=first.dtype)
    return torch.cat([first.unsqueeze(0), rest], dim=0)


def logprob_at(logits, ans):
    return F.log_softmax(logits.float(), dim=-1).gather(1, ans[:, None]).squeeze(1)


def occluded(image, r, c, grid):
    im = image.copy()
    w, h = im.size
    x0, x1 = int(c * w / grid), int((c + 1) * w / grid)
    y0, y1 = int(r * h / grid), int((r + 1) * h / grid)
    ImageDraw.Draw(im).rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)],
                                 fill=(128, 128, 128))
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020")
    ap.add_argument("--schedule", default="/mnt/local/mcy/GLANCE/results/visual_opd_20261102/schedule_u60.jsonl")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

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
    if len(tasks) != 4 * a.updates:
        raise ValueError("expected four QA slots per update")
    print(f"[occl] {len(tasks)} tasks, grid {a.grid}x{a.grid}", flush=True)

    target, proc = common.load_target(device=dev, dtype=torch.float32)
    target.eval().requires_grad_(False)
    print("[occl] target loaded", flush=True)

    patches, drops = {}, {}
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
        base = logprob_at(answer_logits(target, ids, pix, grid, ans, dev), ans).mean()
        base = float(base)
        best = None
        for r in range(a.grid):
            for c in range(a.grid):
                occ = occluded(img, r, c, a.grid)
                oids, opix, ogrid = encode_prompt(proc, row["prompt"], occ, dev)
                if oids.shape[1] != ids.shape[1] or ogrid.shape != grid.shape:
                    raise RuntimeError(f"occlusion changed prompt/grid {task}")
                lp = float(logprob_at(answer_logits(target, oids, opix, ogrid, ans, dev), ans).mean())
                drop = base - lp
                if best is None or drop > best[0]:
                    best = (drop, r, c)
        drop, r, c = best
        patches[task] = [c / a.grid, r / a.grid, (c + 1) / a.grid, (r + 1) / a.grid]
        drops[task] = float(drop)
        if a.debug_dir and i < 8:
            dbg = Path(a.debug_dir); dbg.mkdir(parents=True, exist_ok=True)
            im = occluded(img, r, c, a.grid)
            im.save(dbg / f"{i:02d}_{task.replace(':','_')}.png")
        del ids, pix, grid, ans, base
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"[occl] {i+1}/{len(tasks)} {task} drop={drop:.3f} "
                  f"cell=({r},{c}) elapsed={elapsed/60:.1f}min "
                  f"eta={(elapsed/(i+1)*(len(tasks)-i-1))/60:.1f}min", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1,
        "run": str(run),
        "schedule": str(a.schedule),
        "updates": a.updates,
        "method": "target_occlusion_grid_best_cell",
        "grid": a.grid,
        "patches": patches,
        "drop": drops,
        "wall_minutes": (time.time() - t0) / 60,
    }, out)
    print("wrote", out, "tasks", len(patches), "minutes", (time.time() - t0) / 60,
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
