#!/usr/bin/env python3
"""Blank-video visual-dependence diagnostic for the video OPD checkpoint.

For every fresh30 row, teacher-force the target on the arm's AR answer with the
real 16-frame clip and with an all-gray 16-frame clip of the same size/fps.
g_i = max(0, log p_real(y_i) - log p_blank(y_i)); per-video mean g is joined to
the existing warm/OPD fresh30 tau.

Diagnostic only: no optimizer updates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, DynamicCache

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
from glance.vl import VisionPositions  # noqa: E402


def encode_video(proc, row, path, dev):
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": row["question"]}]}],
        add_generation_prompt=True, tokenize=False)
    ins = proc(text=[text], videos=[path], return_tensors="pt",
               fps=float(row["fps"]))
    return (ins["input_ids"].to(dev), ins["pixel_values_videos"].to(dev),
            ins["video_grid_thw"].to(dev))


def answer_logits(target, ids, pv, vg, ans, dev):
    pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
    pos.bind(target, ids)
    cache = DynamicCache()
    with torch.inference_mode():
        out = target.model(ids, past_key_values=cache, use_cache=True,
                           **pos.ar_prefill_inputs(ids))
        cache = out.past_key_values
        first = target.lm_head(out.last_hidden_state[0, -1:])[0]
        n = int(ans.numel())
        if n > 1:
            index = torch.arange(ids.shape[1], ids.shape[1] + n - 1,
                                 device=dev).unsqueeze(0)
            block = target.model(ans[:-1].unsqueeze(0), past_key_values=cache,
                                 use_cache=True, position_ids=pos(index),
                                 cache_position=index[0])
            rest = target.lm_head(block.last_hidden_state[0])
        else:
            rest = torch.empty(0, first.numel(), device=dev, dtype=first.dtype)
    return torch.cat([first.unsqueeze(0), rest], dim=0)


def logprob_at(logits, ans):
    return F.log_softmax(logits.float(), dim=-1).gather(1, ans[:, None]).squeeze(1)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    vals = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh30", default="/mnt/local/mcy/GLANCE/data/video_fresh30.parquet")
    ap.add_argument("--blank-dir", default="/mnt/local3/mcy/GLANCE/results/video_opd_visual_20261105/blank_videos")
    ap.add_argument("--warm-eval", default="/mnt/local3/mcy/GLANCE/results/video_caption_opd_20261026/eval/warm_fp32.json")
    ap.add_argument("--opd-eval", default="/mnt/local3/mcy/GLANCE/results/video_caption_opd_20261026/eval/opd330.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"
    torch.set_grad_enabled(False)
    df = pd.read_parquet(a.fresh30)
    if a.limit:
        df = df.head(a.limit)
    warm = {r["video"]: r for r in json.load(open(a.warm_eval))["rows"]}
    opd = {r["video"]: r for r in json.load(open(a.opd_eval))["rows"]}
    target, _ = common.load_target(device=dev, dtype=torch.float32)
    target.eval().requires_grad_(False)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    print(f"[vg] target loaded, {len(df)} rows", flush=True)

    rows = []
    t0 = time.time()
    for i, row in df.reset_index(drop=True).iterrows():
        real = str(row["cache_path"])
        blank = str(Path(a.blank_dir) / f"{row['video_name']}.mp4")
        ids_r, pv_r, vg_r = encode_video(proc, row, real, dev)
        ids_b, pv_b, vg_b = encode_video(proc, row, blank, dev)
        if ids_r.shape != ids_b.shape:
            raise RuntimeError(f"{row['video_name']}: prompt shape {ids_r.shape} vs {ids_b.shape}")
        if tuple(vg_r[0].tolist()) != tuple(vg_b[0].tolist()):
            raise RuntimeError(f"{row['video_name']}: grid {vg_r.tolist()} vs {vg_b.tolist()}")
        ans = torch.tensor(opd[row["video_name"]]["ar_ids"], device=dev, dtype=torch.long)
        lr_ = logprob_at(answer_logits(target, ids_r, pv_r, vg_r, ans, dev), ans)
        lb_ = logprob_at(answer_logits(target, ids_b, pv_b, vg_b, ans, dev), ans)
        g = torch.clamp(lr_ - lb_, min=0.0).cpu()
        rows.append({
            "video": row["video_name"], "n_answer": int(ans.numel()),
            "prompt_len": int(ids_r.shape[1]), "g_mean": float(g.mean()),
            "g_median": float(g.median()), "g_max": float(g.max()),
            "logp_real_mean": float(lr_.mean()), "logp_blank_mean": float(lb_.mean()),
            "tau_warm": float(warm[row["video_name"]]["tau"]),
            "tau_opd330": float(opd[row["video_name"]]["tau"]),
        })
        print(f"[vg] {i+1}/{len(df)} {row['video_name']} n={ans.numel()} "
              f"g={float(g.mean()):.4f} tau_w={rows[-1]['tau_warm']:.3f} "
              f"tau_o={rows[-1]['tau_opd330']:.3f} elapsed={(time.time()-t0)/60:.1f}min",
              flush=True)
        del ids_r, pv_r, vg_r, ids_b, pv_b, vg_b, ans, lr_, lb_, g

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["delta_opd_warm"] = frame["tau_opd330"] - frame["tau_warm"]
    frame.to_csv(out.with_suffix(".csv"), index=False)
    g = frame["g_mean"].to_numpy()
    delta = frame["delta_opd_warm"].to_numpy()
    med = float(np.median(g))
    hi = delta[g >= med]; lo = delta[g < med]
    summary = {
        "n": int(len(frame)),
        "g_mean": float(g.mean()), "g_median": med,
        "tau_warm": float(frame["tau_warm"].mean()),
        "tau_opd330": float(frame["tau_opd330"].mean()),
        "delta_opd_warm": float(delta.mean()),
        "delta_ci95": boot_ci(delta),
        "high_g": {"n": int(hi.size), "g_mean": float(g[g >= med].mean()),
                   "delta": float(hi.mean()), "ci95": boot_ci(hi)},
        "low_g": {"n": int(lo.size), "g_mean": float(g[g < med].mean()),
                  "delta": float(lo.mean()), "ci95": boot_ci(lo)},
        "spearman_g_delta": float(pd.Series(g).corr(pd.Series(delta), method="spearman")),
        "wall_minutes": (time.time() - t0) / 60,
    }
    json.dump(summary, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, out.with_suffix(".csv"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
