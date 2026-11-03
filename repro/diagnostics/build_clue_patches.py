#!/usr/bin/env python3
"""Build clue-targeted cutout patches for the OPD training schedule.

For each scheduled QA, teacher-force the target on its own greedy answer with the
real image, take the target's fused hidden at the answer tokens and at the image
tokens, and pick the visual token with the highest cosine similarity to the mean
answer representation. Store a 25%-of-image box centered on that token.

This is a data-preparation script only: no optimizer updates, no checkpoint.
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

IMAGE_TOKEN_ID = 151655


def encode_prompt(proc, prompt, image, dev):
    ids, pix, grid = common.encode(proc, prompt, image)
    return (ids.to(dev),
            pix.to(dev) if pix is not None else None,
            grid.to(dev) if grid is not None else None)


def fused_states(target, ids, ans, pix, grid, dev):
    pos = VisionPositions(pix, grid)
    pos.bind(target, ids)
    cache = DynamicCache()
    with torch.inference_mode():
        out = target.model(ids, past_key_values=cache, use_cache=True,
                           output_hidden_states=True,
                           **pos.ar_prefill_inputs(ids))
        fused_prompt = common.context_feature(out.hidden_states,
                                              layer_ids=common.KEPT_LAYERS)[0]
        cache = out.past_key_values
        index = torch.arange(ids.shape[1], ids.shape[1] + int(ans.numel()),
                             device=dev).unsqueeze(0)
        block = target.model(ans.unsqueeze(0), past_key_values=cache,
                             use_cache=True, output_hidden_states=True,
                             position_ids=pos(index), cache_position=index[0])
        fused_answer = common.context_feature(block.hidden_states,
                                             layer_ids=common.KEPT_LAYERS)[0]
    return fused_prompt, fused_answer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020")
    ap.add_argument("--schedule", default="/mnt/local/mcy/GLANCE/results/visual_opd_20261102/schedule_u60.jsonl")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--box", type=float, default=0.25,
                    help="fraction of the shorter image side to cover")
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
    print(f"[clue] {len(tasks)} tasks", flush=True)

    target, proc = common.load_target(device=dev, dtype=torch.float32)
    target.eval().requires_grad_(False)
    print("[clue] target loaded", flush=True)

    patches, sims = {}, {}
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
        fused_prompt, fused_answer = fused_states(target, ids, ans, pix, grid, dev)
        vis = ids[0] == IMAGE_TOKEN_ID
        vis_h = fused_prompt[vis]
        mean_answer = fused_answer.mean(dim=0)
        sim = F.cosine_similarity(mean_answer[None], vis_h, dim=-1)
        top = int(sim.argmax())
        sims[task] = float(sim[top])
        t, hp, wp = [int(x) for x in grid[0].tolist()]
        hv, wv = hp // 2, wp // 2
        if vis_h.shape[0] != hv * wv:
            raise RuntimeError(f"visual token count {vis_h.shape[0]} != {hv}x{wv} for {task}")
        row_i, col_i = divmod(top, wv)
        cx = (col_i + 0.5) / wv
        cy = (row_i + 0.5) / hv
        half = a.box / 2.0
        x0, x1 = max(0.0, cx - half), min(1.0, cx + half)
        y0, y1 = max(0.0, cy - half), min(1.0, cy + half)
        patches[task] = [x0, y0, x1, y1]
        if a.debug_dir and i < 8:
            dbg = Path(a.debug_dir); dbg.mkdir(parents=True, exist_ok=True)
            im = img.copy()
            draw = ImageDraw.Draw(im)
            w, h = im.size
            draw.rectangle([int(x0*w), int(y0*h), int(x1*w), int(y1*h)],
                           outline=(255, 0, 0), width=3)
            im.save(dbg / f"{i:02d}_{task.replace(':','_')}.png")
        del ids, pix, grid, fused_prompt, fused_answer, vis, vis_h, mean_answer, sim
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"[clue] {i+1}/{len(tasks)} {task} sim={sims[task]:.3f} "
                  f"patch={patches[task]} elapsed={elapsed/60:.1f}min "
                  f"eta={(elapsed/(i+1)*(len(tasks)-i-1))/60:.1f}min", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1,
        "run": str(run),
        "schedule": str(a.schedule),
        "updates": a.updates,
        "method": "target_fused_answer_visual_cosine_top1_box",
        "image_token_id": IMAGE_TOKEN_ID,
        "box": a.box,
        "patches": patches,
        "similarity": sims,
        "wall_minutes": (time.time() - t0) / 60,
    }, out)
    print("wrote", out, "tasks", len(patches), "minutes", (time.time() - t0) / 60,
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
