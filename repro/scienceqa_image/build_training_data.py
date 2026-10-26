#!/usr/bin/env python3
"""Build the frozen ScienceQA training schedule plus the target's own AR labels.

Outputs, under the run's `dataset/`:

  `train_schedule_seed_<seed>.jsonl` -- the QA schedule. Both arms of one seed
      read this exact file, so they see the same questions in the same order.
      The permutation is a PURE function of the frozen record list (it does not
      depend on the seed), so seed1 and seed2 also see the same data in the same
      order and the seed axis carries only RNG, not a data-subset confound.
  `sqa_pool_df<seed>.parquet`       -- one row per scheduled QA: the image path,
      the frozen prompt, and the identity fields. The trainer reads inputs from
      here, so what was trained on is exactly what is frozen here.
  `sqa_target_ar_df<seed>.pt`      -- for every slot, the FROZEN TARGET's own
      greedy AR continuation on that exact prompt, produced once with the target
      in fp32 on the deployment cached path (`use_cache=True`). Carries the full
      token sequence, the prompt length and the target's per-answer-position
      argmax (`verifier_argmax`).
  `sqa_records_summary_df<seed>.json` -- per-slot lengths, EOS position and the
      number of usable anchors, so the budget is auditable before training.

This is a data step: it calls the frozen target, never the draft head, never a
loss, and never an optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
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
from imageencode import encode_image_inputs  # noqa: E402

QA_PER_UPDATE = 4


def append(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=768)
    ap.add_argument("--target-dtype", default="float32",
                    choices=["float32", "bfloat16"])
    ap.add_argument("--limit", type=int, default=0, help="debug: first N slots only")
    ap.add_argument("--order-seed", default="sqa-image-order-20261015",
                    help="the schedule permutation salt (deliberately NOT the "
                         "training seed, so both seeds share one QA order)")
    a = ap.parse_args()

    R = pathlib.Path(a.run)
    D = R / "dataset"
    outdir = R / "training" / f"seed_{a.seed}" / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    # Truncate: this log records ONE build. Appending would silently mix a debug
    # build with the formal one, which is how an earlier attempt at the formal run
    # reported a 43-line "progress" reading after 3 real records.
    log = outdir / "build_log.jsonl"
    log.write_text("")

    torch.set_grad_enabled(False)
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)

    # ---- the frozen record list, deterministically permuted
    records = []
    with open(D / "records_manifest.csv") as fh:
        import csv
        for r in csv.DictReader(fh):
            if r["canonical_split"] == "train":
                records.append(r)
    records.sort(key=lambda r: r["official_row_index"])
    if len(records) != 1200:
        raise SystemExit(f"expected the frozen 1200 train records, found {len(records)}")
    order = sorted(records,
                   key=lambda r: hashlib.sha256(
                       f"{a.order_seed}:{r['pid']}".encode()).hexdigest())
    n_slots = a.updates * QA_PER_UPDATE
    if n_slots > len(order):
        raise SystemExit(f"{n_slots} slots requested but only {len(order)} records")
    scheduled = order[:n_slots]
    if a.limit:
        scheduled = scheduled[:a.limit]

    sched_rows = []
    for i, r in enumerate(scheduled):
        sched_rows.append({
            "order": i, "update": i // QA_PER_UPDATE, "slot": i % QA_PER_UPDATE,
            "task": r["pid"],
            "video_id": r["pid"], "question_id": r["pid"],
            "pid": r["pid"], "image_group_id": r["image_group_id"],
            "prompt_sha256": r["prompt_sha256"], "question": r["question"],
            "prompt": r["prompt"],
        })
    # A limited build is a debug/smoke build and gets its own schedule name, so it
    # can never be mistaken for (or overwrite) the formal 4*updates schedule.
    sched_name = (f"train_schedule_seed_{a.seed}_debug.jsonl" if a.limit
                  else f"train_schedule_seed_{a.seed}.jsonl")
    with open(D / sched_name, "w") as fh:
        for r in sched_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_pid = {r["pid"]: r for r in records}
    target, proc = common.load_target(
        device=dev,
        dtype=torch.float32 if a.target_dtype == "float32" else torch.bfloat16)
    eos = int(proc.tokenizer.eos_token_id)
    print(f"[build] seed {a.seed} slots {len(sched_rows)} eos {eos} "
          f"max_new {a.max_new} max_pixels {a.max_pixels}", flush=True)

    pool_rows, recs, summary = [], [], []
    t_all = time.time()
    for i, slot in enumerate(sched_rows):
        man = by_pid[slot["pid"]]
        row = {**man, "task": slot["pid"], "video_id": slot["pid"],
               "question_id": slot["pid"], "prompt_mode": "B", "split": "train",
               "update": slot["update"], "slot_in_update": slot["slot"],
               "order": slot["order"]}
        x = encode_image_inputs(proc, man, max_pixels=a.max_pixels)
        ids = x["input_ids"].to(dev)
        prompt_len = int(ids.shape[1])
        pix = x["pixel_values"].to(dev)
        grid = x["image_grid_thw"].to(dev)
        t0 = time.time()
        from transformers import DynamicCache
        cache = DynamicCache()
        # MEASURED TRAP, do not "simplify" this back. Passing explicit
        # `position_ids` to the prefill suppresses `Qwen3VLModel`'s
        # `rope_deltas` computation (it is only set when `position_ids is None`),
        # so every subsequent single-token step positions itself as if it were
        # token 0 and the trajectory degenerates into a 2-token repetition.
        # The deployment path (`glance.decode.greedy_generate` via
        # `VisionPositions.ar_prefill_inputs`) deliberately passes only
        # `pixel_values` / `image_grid_thw` / `cache_position`, and that is what
        # this reproduces. Measured on the first 12 debug records: the corrected
        # path agrees with the teacher-forced full-sequence replay at EVERY answer
        # position (0/12 records disagree), while the explicit-position variant
        # diverged from token 2 onward.
        out = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                     pixel_values=pix, image_grid_thw=grid)
        toks = [int(out.logits[0, -1].argmax())]
        while len(toks) < a.max_new and toks[-1] != eos:
            step = torch.tensor([[toks[-1]]], device=dev)
            out = target(step, past_key_values=cache, use_cache=True,
                         logits_to_keep=1,
                         cache_position=torch.tensor([cache.get_seq_length()],
                                                     device=dev))
            toks.append(int(out.logits[0, -1].argmax()))
        full = torch.cat([ids[0].cpu(), torch.tensor(toks, dtype=torch.long)])
        # The labels the trainer consumes are the per-answer-position argmax of ONE
        # teacher-forced forward over the whole trajectory, which is exactly the
        # check the trainer replays. Deriving them here (instead of from the
        # incremental logits) makes a disagreement impossible to miss rather than
        # merely unlikely: it is asserted below.
        del out, cache
        torch.cuda.empty_cache()
        full_dev = full.to(dev)
        hidden = []
        hook = target.model.register_forward_hook(
            lambda m, i, o: hidden.append(o.last_hidden_state.detach()))
        try:
            target(full_dev[None], past_key_values=DynamicCache(), use_cache=True,
                   logits_to_keep=1, pixel_values=pix, image_grid_thw=grid,
                   position_ids=common.target_positions(target, full_dev[None], grid))
        finally:
            hook.remove()
        rows_lg = target.lm_head(hidden[0][0, prompt_len - 1:-1]).float()
        verifier = rows_lg.argmax(-1).tolist()
        margins = [float(v) for v in
                   torch.topk(rows_lg, 2, dim=-1).values.diff(dim=-1)[:, 0].tolist()]
        del hidden, rows_lg, full_dev
        torch.cuda.empty_cache()
        if verifier != toks:
            bad = next(i for i, (x, y) in enumerate(zip(verifier, toks)) if x != y)
            raise RuntimeError(
                f"{slot['pid']}: the incremental greedy AR and the teacher-forced "
                f"replay disagree at answer position {bad} "
                f"({toks[bad]} vs {verifier[bad]}); refusing to freeze labels that "
                f"the trainer would reject")
        eos_pos = (int(np.argmax([t == eos for t in toks])) if eos in toks else None)
        n_ans = len(toks)
        usable = (list(range(prompt_len - 1, prompt_len + eos_pos))
                  if eos_pos is not None
                  else list(range(prompt_len - 1, prompt_len + n_ans - 1)))
        usable = [p for p in usable if 1 <= p <= int(full.shape[0]) - 2]
        recs.append({"task": slot["pid"], "input_ids": full,
                     "prompt_len": prompt_len, "eos_pos_in_answer": eos_pos,
                     "n_answer_tokens": n_ans, "verifier_argmax": verifier,
                     "verifier_gap": margins})
        pool_rows.append({**row, "input_len": int(full.shape[0])})
        summary.append({"task": slot["pid"], "image_group_id": man["image_group_id"],
                        "update": slot["update"], "order": slot["order"],
                        "prompt_len": prompt_len, "n_answer_tokens": n_ans,
                        "eos_pos_in_answer": eos_pos, "hit_eos": eos_pos is not None,
                        "n_usable_anchors": len(usable),
                        "answer_text": proc.tokenizer.decode(toks, skip_special_tokens=True),
                        "seconds": time.time() - t0})
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[build] {i+1}/{len(sched_rows)} {slot['pid']} prompt {prompt_len} "
                  f"ans {n_ans} {time.time() - t0:.2f}s "
                  f"eta {(len(sched_rows)-i-1)*np.mean([s['seconds'] for s in summary])/60:.1f}min",
                  flush=True)
        append(log, summary[-1])
        del x, ids, pix, grid
        torch.cuda.empty_cache()

    sfx = "_debug" if a.limit else ""
    df = pd.DataFrame(pool_rows)
    df.to_parquet(D / f"sqa_pool_df{a.seed}{sfx}.parquet", index=False)
    blob = {"records": recs, "pool": str(D / f"sqa_pool_df{a.seed}{sfx}.parquet"),
            "seed": a.seed, "max_new": a.max_new, "eos_token_id": eos,
            "max_pixels": a.max_pixels, "modality": "image",
            "target": common.TARGET_PATH, "target_dtype": a.target_dtype}
    torch.save(blob, D / f"sqa_target_ar_df{a.seed}{sfx}.pt")
    meta = {
        "seed": a.seed, "slots_requested": len(sched_rows),
        "records_built": len(recs), "modality": "image",
        "max_new": a.max_new, "max_pixels": a.max_pixels,
        "order_seed": a.order_seed,
        "unique_records": len({r["task"] for r in recs}),
        "unique_image_groups": len({r["image_group_id"] for r in summary}),
        "prompt_len": {"min": int(np.min([r["prompt_len"] for r in summary])),
                       "median": float(np.median([r["prompt_len"] for r in summary])),
                       "max": int(np.max([r["prompt_len"] for r in summary]))},
        "answer_tokens": {"min": int(np.min([r["n_answer_tokens"] for r in summary])),
                          "median": float(np.median([r["n_answer_tokens"] for r in summary])),
                          "max": int(np.max([r["n_answer_tokens"] for r in summary])),
                          "mean": float(np.mean([r["n_answer_tokens"] for r in summary]))},
        "hit_eos_fraction": sum(r["hit_eos"] for r in summary) / len(summary),
        "usable_anchors_total": int(sum(r["n_usable_anchors"] for r in summary)),
        "usable_anchors_per_qa": {
            "min": int(np.min([r["n_usable_anchors"] for r in summary])),
            "median": float(np.median([r["n_usable_anchors"] for r in summary])),
            "max": int(np.max([r["n_usable_anchors"] for r in summary]))},
        "wall_seconds": time.time() - t_all,
    }
    (D / f"sqa_records_summary_df{a.seed}{sfx}.json").write_text(
        json.dumps({"meta": meta, "per_slot": summary}, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
