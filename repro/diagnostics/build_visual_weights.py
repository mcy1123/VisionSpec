#!/usr/bin/env python3
"""Build per-answer-token visual-dependence weights for an OPD schedule.

For each scheduled task, teacher-force the target's own greedy answer under the
real image and a same-size blank image, then store

    w_i = 1 + alpha * clip(max(0, log p_real(y_i) - log p_blank(y_i)), 0, cap)

for every answer token. The weighted OPD trainer consumes the ``weights`` dict.
This is a data-preparation script: no optimizer updates, no checkpoint writes.
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
from PIL import Image
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020")
    ap.add_argument("--schedule", default="/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020/dataset/schedule.jsonl")
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cap", type=float, default=2.0)
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
        raise ValueError("schedule does not cover exactly the requested updates")
    tasks = [task for u in range(a.updates) for task in schedule[u]]
    if len(tasks) != 4 * a.updates:
        raise ValueError("expected four QA slots per update")
    print(f"[weights] {len(tasks)} tasks from {a.updates} updates", flush=True)

    target, proc = common.load_target(device=dev, dtype=torch.float32)
    target.eval().requires_grad_(False)
    print("[weights] target loaded", flush=True)

    weights, gmeans = {}, {}
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
        blank = Image.new("RGB", img.size, (128, 128, 128))

        ids, pix, grid = encode_prompt(proc, row["prompt"], img, dev)
        bids, bpix, bgrid = encode_prompt(proc, row["prompt"], blank, dev)
        if ids.shape[1] != rec["prompt_len"]:
            raise RuntimeError(f"prompt mismatch {task}: {ids.shape[1]} vs {rec['prompt_len']}")
        if bids.shape[1] != ids.shape[1] or bgrid.shape != grid.shape:
            raise RuntimeError(f"blank changed prompt/grid {task}")

        logits = answer_logits(target, ids, pix, grid, ans, dev)
        blogits = answer_logits(target, bids, bpix, bgrid, ans, dev)
        g = torch.clamp(logprob_at(logits, ans) - logprob_at(blogits, ans), min=0.0)
        w = 1.0 + a.alpha * torch.clamp(g, max=a.cap)
        weights[task] = [float(x) for x in w]
        gmeans[task] = float(g.mean())
        del ids, pix, grid, bids, bpix, bgrid, logits, blogits, g, w
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"[weights] {i+1}/{len(tasks)} {task} g={gmeans[task]:.3f} "
                  f"elapsed={elapsed/60:.1f}min eta={(elapsed/(i+1)*(len(tasks)-i-1))/60:.1f}min",
                  flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1,
        "run": str(run),
        "schedule": str(a.schedule),
        "updates": a.updates,
        "corrupt": "blank_same_size",
        "alpha": a.alpha,
        "cap": a.cap,
        "weights": weights,
        "g_mean": gmeans,
        "wall_minutes": (time.time() - t0) / 60,
    }, out)
    print("wrote", out, "tasks", len(weights), "minutes", (time.time() - t0) / 60,
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
