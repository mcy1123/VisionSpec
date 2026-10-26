#!/usr/bin/env python3
"""Build the QA training records and pool for one seed from the frozen schedule.

Outputs, under the run's `dataset/`:

  `qa_pool_df<seed>.parquet`  -- one row per scheduled QA slot, carrying the
      video's `cache_path` / `fps` / `source_fps` and the frozen question and
      prompt. The trainer reads ONLY this file for inputs, so the training data
      is exactly the frozen schedule.
  `target_ar_df<seed>.pt`     -- for every slot, the FROZEN TARGET's own greedy
      autoregressive continuation on that exact prompt, produced here once with
      the target in fp32 on the deployment prefill path (`use_cache=True`) so
      the trainer never has to re-derive it. Contains the full token sequence
      and the prompt length; no logits are stored (they are re-derived at
      training time from the same forward the loss uses).
  `qa_records_summary_df<seed>.json` -- per-slot length, EOS position and the
      number of usable anchors, so the budget is auditable before training.

This is a data step: it calls the frozen target, and it never calls the draft
head, never computes a loss and never updates a parameter.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
from videoencode import build_video_text  # noqa: E402


def _argmax_gap(logits):
    """Top-1 id and top1-top2 gap of one logits row, in float32."""
    top = torch.topk(logits, 2)
    return int(top.indices[0]), float(top.values[0] - top.values[1])


def append(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--time-mode", default="legacy", choices=["legacy", "real"])
    ap.add_argument("--inventory", default="/mnt/local2/mcy/datasets/msvd_qa_full/inventory.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="debug only: first N slots")
    ap.add_argument("--target-dtype", default="float32",
                    choices=["float32", "bfloat16"])
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the artifact names, so a small smoke build can "
                         "run beside the full one without touching its files")
    a = ap.parse_args()

    R = pathlib.Path(a.run)
    D = R / "dataset"
    outdir = R / "training" / f"seed_{a.seed}" / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "build_log.jsonl"

    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)

    inv = {}
    for line in pathlib.Path(a.inventory).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            inv[rec["video_id"]] = rec
    # Map every acquired train video onto the official file name used by the
    # schedule: cache_path/raw_path/fps come from the inventory record.
    sched = [json.loads(l) for l in
             (D / f"train_schedule_seed_{a.seed}.jsonl").read_text().splitlines()
             if l.strip()]
    if a.limit:
        sched = sched[:a.limit]

    from transformers import AutoProcessor
    target, proc = common.load_target(
        device=dev,
        dtype=torch.float32 if a.target_dtype == "float32" else torch.bfloat16)
    eos = int(proc.tokenizer.eos_token_id)
    print(f"[build] seed {a.seed} slots {len(sched)} eos {eos} time {a.time_mode}",
          flush=True)

    pool_rows, records, summary = [], [], []
    missing_videos = []
    for i, slot in enumerate(sched):
        vid = slot["video_id"]
        rec = inv.get(vid)
        if rec is None or not rec.get("decode_ok"):
            missing_videos.append(vid)
            continue
        row_task = f"{vid}|{slot['question_id']}"
        row = {
            "task": row_task,
            "video_id": vid,
            "question_id": slot["question_id"],
            "question": slot["question"],
            "prompt": slot["prompt"],
            "prompt_mode": "B",
            "split": "train",
            "update": slot["update"], "slot_in_update": slot["slot"],
            "order": slot["order"],
            "cache_path": rec["cache_path"],
            "raw_path": rec["raw_path"],
            "fps": float(rec.get("fps", 30.0)),
            "source_fps": float(rec["source_fps"]),
            "video_duration": float(rec["video_duration"]),
            "source_frame_count": int(rec["source_frame_count"]),
            "sample_frame_ids": rec["sample_frame_ids"],
            "sample_timestamps": rec["sample_timestamps"],
            "n_frames": int(rec["n_frames"]),
            "width": int(rec["width"]), "height": int(rec["height"]),
            "hash": rec["raw_sha256"], "cache_hash": rec["cache_sha256"],
        }
        text = build_video_text(proc, row["prompt"])
        from videoencode import encode_inputs
        x = encode_inputs(proc, row, a.time_mode, text=text)
        ids = x["input_ids"].to(dev)
        prompt_len = int(ids.shape[1])
        t0 = time.time()
        # Greedy AR continuation on the cached path, which is exactly the
        # deployment prefill/step sequence. The full logits row of every answer
        # position is captured here, because this single decode IS the target's
        # own forward: re-deriving its argmax with a second pass would double the
        # cost of the whole data build and could disagree in the last ulp.
        from transformers import DynamicCache
        cache = DynamicCache()
        out = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                     **{k: v.to(dev) for k, v in x.items() if k != "input_ids"})
        argmax, gap = _argmax_gap(out.logits[0, -1].float())
        toks = [argmax]
        verifier = [argmax]          # index 0: what the last prompt token predicts
        margins = [gap]
        while len(toks) < a.max_new and toks[-1] != eos:
            step = torch.tensor([[toks[-1]]], device=dev)
            out = target(step, past_key_values=cache, use_cache=True,
                         logits_to_keep=1,
                         cache_position=torch.tensor([cache.get_seq_length()],
                                                     device=dev))
            argmax, gap = _argmax_gap(out.logits[0, -1].float())
            toks.append(argmax)
            verifier.append(argmax)
            margins.append(gap)
        full = torch.cat([ids[0].cpu(), torch.tensor(toks, dtype=torch.long)])
        eos_pos = (int(np.argmax([t == eos for t in toks]))
                   if eos in toks else None)
        n_ans = len(toks)
        usable = []
        if eos_pos is not None:
            usable = list(range(prompt_len - 1, prompt_len + eos_pos))
        else:
            usable = list(range(prompt_len - 1, prompt_len + n_ans - 1))
        usable = [p for p in usable if 1 <= p <= int(full.shape[0]) - 2]
        records.append({"task": row_task, "input_ids": full,
                        "prompt_len": prompt_len, "eos_pos_in_answer": eos_pos,
                        "n_answer_tokens": n_ans,
                        # The frozen target's own next-token argmax at every
                        # answer position (index i belongs to answer token i),
                        # plus the top1-top2 logit gap so the Q1 one-hot
                        # assumption can be audited after the fact.
                        "verifier_argmax": verifier, "verifier_gap": margins})
        pool_rows.append({**row, "input_len": int(full.shape[0])})
        summary.append({"task": row_task, "video_id": vid,
                        "question_id": slot["question_id"],
                        "update": slot["update"], "order": slot["order"],
                        "prompt_len": prompt_len, "n_answer_tokens": n_ans,
                        "eos_pos_in_answer": eos_pos,
                        "hit_eos": eos_pos is not None,
                        "n_usable_anchors": len(usable),
                        "answer_text": proc.tokenizer.decode(toks,
                                                             skip_special_tokens=True),
                        "seconds": time.time() - t0})
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[build] {i+1}/{len(sched)} {row_task} prompt {prompt_len} "
                  f"ans {n_ans} gapmax {max(margins):.2f} "
                  f"{time.time() - t0:.2f}s", flush=True)
        append(log, summary[-1])
        del x, ids

    sfx = a.out_suffix
    df = pd.DataFrame(pool_rows)
    df.to_parquet(D / f"qa_pool_df{a.seed}{sfx}.parquet", index=False)
    blob = {"records": records,
            "pool": str(D / f"qa_pool_df{a.seed}{sfx}.parquet"),
            "seed": a.seed, "time_mode": a.time_mode,
            "max_new": a.max_new, "eos_token_id": eos,
            "target": common.TARGET_PATH}
    torch.save(blob, D / f"target_ar_df{a.seed}{sfx}.pt")
    meta = {
        "seed": a.seed, "slots": len(sched), "built": len(records),
        "missing_videos": sorted(set(missing_videos)),
        "n_missing_slots": len(sched) - len(records),
        "time_mode": a.time_mode, "max_new": a.max_new,
        "prompt_len": {"min": int(np.min([r["prompt_len"] for r in summary])),
                       "max": int(np.max([r["prompt_len"] for r in summary]))} if summary else {},
        "answer_tokens": {"min": int(np.min([r["n_answer_tokens"] for r in summary])),
                          "median": float(np.median([r["n_answer_tokens"] for r in summary])),
                          "max": int(np.max([r["n_answer_tokens"] for r in summary])),
                          "mean": float(np.mean([r["n_answer_tokens"] for r in summary]))} if summary else {},
        "hit_eos_fraction": (sum(r["hit_eos"] for r in summary) / len(summary)) if summary else None,
        "usable_anchors_total": int(sum(r["n_usable_anchors"] for r in summary)),
        "usable_anchors_per_qa": {"min": int(np.min([r["n_usable_anchors"] for r in summary])),
                                  "median": float(np.median([r["n_usable_anchors"] for r in summary])),
                                  "max": int(np.max([r["n_usable_anchors"] for r in summary]))} if summary else {},
    }
    (D / f"qa_records_summary_df{a.seed}{sfx}.json").write_text(
        json.dumps({"meta": meta, "per_slot": summary}, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
