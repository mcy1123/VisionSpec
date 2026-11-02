#!/usr/bin/env python3
"""Visual-dependence diagnostic for ScienceQA SFT/OPD heads.

For each test question, teacher-force the target's own greedy answer under
(a) the real image and (b) a same-size blank image, then measure how much the
target's log-prob of each committed token drops:

    g_i = max(0, log p_real(y_i) - log p_blank(y_i))

The answer logits are computed through the same prefill + cached block path
used by training, so M-RoPE and the image state match the deployed model.
The per-question mean g is joined with the existing SFT/OPD per-question
tokens-per-round to test whether OPD's gain is concentrated on visually
dependent questions.

This is a diagnostic only: no training, no checkpoint writes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
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


def boot_ci(values, n=5000, seed=20261102):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, values.size, values.size)].mean())
             for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def load_rows(run: Path, split: str):
    rows = list(csv.DictReader(open(run / "dataset" / "records_manifest.csv")))
    rows = [r for r in rows if r["canonical_split"] == split]
    rows.sort(key=lambda r: int(r["official_row_index"]))
    return rows


def load_ar(run: Path, split: str):
    blob = torch.load(run / "eval" / f"ar_{split}.pt", map_location="cpu",
                      weights_only=False)
    return {r["pid"]: r for r in blob["records"]}


def load_head(path: Path):
    head = common.load_head(str(path), device="cuda:0", dtype=torch.bfloat16).eval()
    head.requires_grad_(False)
    return head


def encode_prompt(proc, row, image, dev):
    ids, pix, grid = common.encode(proc, row["prompt"], image)
    return (ids.to(dev),
            pix.to(dev) if pix is not None else None,
            grid.to(dev) if grid is not None else None)


def answer_logits(target, ids, pix, grid, ans, dev):
    """Teacher-forced logits for the answer tokens, through prefill + cache."""
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
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(1, ans[:, None]).squeeze(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corrupt", default="blank", choices=["blank"])
    a = ap.parse_args()

    torch.cuda.set_device(a.gpu)
    dev = f"cuda:{a.gpu}"
    torch.set_grad_enabled(False)
    run = Path(a.run)
    rows = load_rows(run, a.split)
    if a.limit:
        rows = rows[:a.limit]
    ar = load_ar(run, a.split)
    print(f"[diag] {len(rows)} {a.split} rows, {a.corrupt} corruption", flush=True)

    target, proc = common.load_target(device=dev, dtype=torch.float32)
    target.eval().requires_grad_(False)
    print("[diag] target loaded", flush=True)

    records = []
    t0 = time.time()
    for i, row in enumerate(rows):
        rec = ar[row["pid"]]
        ans = torch.tensor(rec["tokens"], dtype=torch.long, device=dev)
        img = Image.open(row["image_path"]).convert("RGB")
        if max(img.size) > 768:
            w, h = img.size
            scale = 768 / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.BICUBIC)
        if a.corrupt == "blank":
            corrupt = Image.new("RGB", img.size, (128, 128, 128))
        else:
            raise ValueError(a.corrupt)

        ids, pix, grid = encode_prompt(proc, row, img, dev)
        bids, bpix, bgrid = encode_prompt(proc, row, corrupt, dev)
        if ids.shape[1] != rec["prompt_len"]:
            raise RuntimeError(
                f"prompt length mismatch for {row['pid']}: "
                f"{ids.shape[1]} vs AR {rec['prompt_len']}")
        if bids.shape[1] != ids.shape[1] or bgrid.shape != grid.shape:
            raise RuntimeError(f"corruption changed prompt/grid for {row['pid']}")

        logits = answer_logits(target, ids, pix, grid, ans, dev)
        blogits = answer_logits(target, bids, bpix, bgrid, ans, dev)
        correct = logprob_at(logits, ans)
        corrupt_lp = logprob_at(blogits, ans)
        g = torch.clamp(correct - corrupt_lp, min=0.0)

        # target entropy at the real-image positions (diagnostic only)
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)

        records.append({
            "pid": row["pid"], "subject": row["subject"],
            "grade": row["grade"], "n_choices": int(row["n_choices"]),
            "gold_letter": row["gold_letter"], "n_tokens": int(ans.numel()),
            "prompt_len": int(ids.shape[1]),
            "g_mean": float(g.mean()), "g_max": float(g.max()),
            "g_median": float(g.median()),
            "g_frac_gt_0p5": float((g > 0.5).float().mean()),
            "g_frac_gt_1p0": float((g > 1.0).float().mean()),
            "target_entropy_mean": float(entropy.mean()),
            "target_entropy_median": float(entropy.median()),
        })
        del ids, pix, grid, bids, bpix, bgrid, logits, blogits, correct, corrupt_lp, g, probs, entropy
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(rows) - i - 1) / 60
            print(f"[diag] {i+1}/{len(rows)} {row['pid']} "
                  f"g={records[-1]['g_mean']:.3f} elapsed={elapsed/60:.1f}min "
                  f"eta={eta:.1f}min", flush=True)

    df = pd.DataFrame(records)
    sft = pd.read_csv(run / "eval" / f"sft_3546_{a.split}_per_question.csv")
    opd = pd.read_csv(run / "eval" / f"opd_3546_{a.split}_per_question.csv")
    sft = sft[["pid", "tokens_per_round", "tau", "answer_correct"]].rename(
        columns={"tokens_per_round": "tau_sft", "tau": "accepted_plus_one_sft",
                 "answer_correct": "correct_sft"})
    opd = opd[["pid", "tokens_per_round", "tau", "answer_correct"]].rename(
        columns={"tokens_per_round": "tau_opd", "tau": "accepted_plus_one_opd",
                 "answer_correct": "correct_opd"})
    df = df.merge(sft, on="pid", how="left").merge(opd, on="pid", how="left")
    df["delta"] = df["tau_opd"] - df["tau_sft"]

    med = float(df["g_mean"].median())
    high = df["g_mean"] >= med
    low = ~high
    q75 = float(df["g_mean"].quantile(0.75))
    q25 = float(df["g_mean"].quantile(0.25))

    def block(mask):
        sub = df[mask]
        return {
            "n": int(len(sub)),
            "g_mean": float(sub["g_mean"].mean()) if len(sub) else float("nan"),
            "tau_sft": float(sub["tau_sft"].mean()) if len(sub) else float("nan"),
            "tau_opd": float(sub["tau_opd"].mean()) if len(sub) else float("nan"),
            "delta": float(sub["delta"].mean()) if len(sub) else float("nan"),
            "delta_ci95": boot_ci(sub["delta"].tolist()) if len(sub) else [float("nan")] * 2,
            "correct_sft": float(sub["correct_sft"].mean()) if len(sub) else float("nan"),
            "correct_opd": float(sub["correct_opd"].mean()) if len(sub) else float("nan"),
        }

    summary = {
        "split": a.split, "n": int(len(df)), "corrupt": a.corrupt,
        "g_mean": float(df["g_mean"].mean()),
        "g_median": float(df["g_mean"].median()),
        "g_q25": q25, "g_q75": q75,
        "tau_sft": float(df["tau_sft"].mean()),
        "tau_opd": float(df["tau_opd"].mean()),
        "delta": float(df["delta"].mean()),
        "delta_ci95": boot_ci(df["delta"].tolist()),
        "spearman_g_tau_sft": float(df["g_mean"].corr(df["tau_sft"], method="spearman")),
        "spearman_g_tau_opd": float(df["g_mean"].corr(df["tau_opd"], method="spearman")),
        "spearman_g_delta": float(df["g_mean"].corr(df["delta"], method="spearman")),
        "high_median": block(high),
        "low_median": block(low),
        "top_quartile": block(df["g_mean"] >= q75),
        "bottom_quartile": block(df["g_mean"] <= q25),
        "wall_minutes": (time.time() - t0) / 60,
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
