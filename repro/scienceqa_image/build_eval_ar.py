#!/usr/bin/env python3
"""Pre-generate the independent greedy AR oracle for a split.

The AR continuation depends only on the frozen target, the picture and the
prompt -- NOT on the draft head. Every arm evaluated on a split therefore needs
the *same* oracle, and re-deriving it once per arm would multiply the dominant
cost of the whole evaluation by the number of checkpoints. This script derives it
once, records the token ids and the full per-position argmax, and lets
``eval_sqa.py`` consume it.

It also records, for each record, the target's own multiple-choice answer parsed
from the visible text, so the target-accuracy appendix needs no extra pass.

    python build_eval_ar.py --run <run> --split dev --gpu 7 --out <run>/eval/ar_dev.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
import glance  # noqa: E402
from glance.vl import VisionPositions  # noqa: E402

ANSWER_RE = re.compile(r"Answer:\s*\(?\s*([A-Ea-e])\s*\)?")


def load_split(run, split, limit=None):
    rows = []
    with open(pathlib.Path(run) / "dataset" / "records_manifest.csv") as fh:
        for r in csv.DictReader(fh):
            if r["canonical_split"] == split:
                rows.append(r)
    rows.sort(key=lambda r: r["official_row_index"])
    return rows[:limit] if limit else rows


def encode(proc, row, max_pixels, dev):
    from PIL import Image
    img = Image.open(row["image_path"]).convert("RGB")
    if max_pixels and max(img.size) > max_pixels:
        w, h = img.size
        s = max_pixels / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    ids, pixels, grid = common.encode(proc, row["prompt"], img)
    ids = ids.to(dev)
    pix = pixels.to(dev) if pixels is not None else None
    g = grid.to(dev) if grid is not None else None
    pos = VisionPositions(pix, g)
    return ids, pos, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)
    torch.set_grad_enabled(False)

    target, proc = common.load_target(device=dev, dtype=torch.float32)
    eos = int(proc.tokenizer.eos_token_id)
    rows = load_split(a.run, a.split, a.limit or None)
    print(f"[ar] split {a.split}: {len(rows)} records, fp32 target, "
          f"max_new {a.max_new}, max_pixels {a.max_pixels}", flush=True)

    out_rows, t_all = [], time.time()
    for i, r in enumerate(rows):
        ids, pos, grid = encode(proc, r, a.max_pixels, dev)
        t0 = time.perf_counter()
        final, st = glance.greedy_generate(target, ids, a.max_new, [eos],
                                          positions=pos, return_stats=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        toks = final[0, ids.shape[1]:].tolist()
        text = proc.tokenizer.decode(toks, skip_special_tokens=True)
        m = ANSWER_RE.findall(text)
        pred = m[-1].upper() if m else None
        out_rows.append({
            "pid": r["pid"], "image_group_id": r["image_group_id"],
            "canonical_split": r["canonical_split"],
            "prompt_len": int(ids.shape[1]), "tokens": toks,
            "n_tokens": len(toks), "text": text,
            "ends_with_eos": bool(toks and toks[-1] == eos),
            "hit_cap": len(toks) >= a.max_new,
            "n_rounds": st.rounds, "ms_per_token": st.ms_per_token,
            "prefill_ms": st.prefill_ms,
            "wall_ms": dt * 1000.0,
            "gold_letter": r["gold_letter"], "pred_letter": pred,
            "answer_correct": (pred == r["gold_letter"]) if pred else False,
            "answer_parse_ok": pred is not None,
        })
        if (i + 1) % 10 == 0 or i == 0:
            avg = sum(x["wall_ms"] for x in out_rows) / len(out_rows)
            print(f"[ar] {i+1}/{len(rows)} {r['pid']} n={len(toks)} "
                  f"eos={out_rows[-1]['ends_with_eos']} "
                  f"{dt*1000:.0f}ms avg {avg:.0f}ms "
                  f"eta {(len(rows)-i-1)*avg/1000/60:.1f}min", flush=True)
        del ids, pos, final
        torch.cuda.empty_cache()

    blob = {"split": a.split, "target": common.TARGET_PATH, "target_dtype": "float32",
            "max_new": a.max_new, "max_pixels": a.max_pixels,
            "eos_token_id": eos, "n": len(out_rows), "records": out_rows,
            "wall_seconds": time.time() - t_all}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, a.out)
    n_eos = sum(x["ends_with_eos"] for x in out_rows)
    n_cap = sum(x["hit_cap"] for x in out_rows)
    n_ans = sum(x["answer_parse_ok"] for x in out_rows)
    acc = sum(x["answer_correct"] for x in out_rows)
    lens = sorted(x["n_tokens"] for x in out_rows)
    print(json.dumps({
        "split": a.split, "n": len(out_rows),
        "n_tokens_p10": lens[len(lens)//10], "n_tokens_p50": lens[len(lens)//2],
        "n_tokens_p90": lens[9*len(lens)//10],
        "natural_eos": n_eos, "cap_hits": n_cap,
        "answer_parsed": n_ans, "answer_correct": acc,
        "answer_accuracy": acc / max(n_ans, 1),
        "wall_minutes": (time.time() - t_all) / 60,
    }, indent=2), flush=True)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
