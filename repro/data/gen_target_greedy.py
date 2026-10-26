#!/usr/bin/env python3
"""Generate the target's own greedy continuations for the training prompts.

[§4.3] "The training data are the target's own greedy generations on 8000
COCO-Caption2017 and 8000 TextVQA prompts".
[Table 5] "source: target greedy, 256 new tokens".

Batched greedy is only a speed-up, not a change of answer: the script asserts
that a batched row reproduces the single-row encoding exactly, and every row
records the ids it produced so the trainer never re-generates.

Sharding lets one process per GPU share the 16K prompts:

    python gen_target_greedy.py --shard 0 --num-shards 2 --gpu 3 --batch 16
"""

from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import argparse
import json
import os
import time

import torch

from shared import common


def collate(processor, batch):
    """Encode a batch of (row, image) with left padding for generation."""
    convs = [common.build_processor_messages(row["prompt"], image)
             for row, image in batch]
    inputs = processor.apply_chat_template(
        convs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True)
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=os.path.join(common.DATA, "train_prompts.jsonl"))
    parser.add_argument("--out", default=os.path.join(common.DATA, "target_greedy.jsonl"))
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=common.GEN_NEW_TOKENS)
    parser.add_argument("--long-side", type=int, default=common.IMAGE_LONG_SIDE)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap rows")
    parser.add_argument("--resume", action="store_true",
                        help="skip rows already present in the output file")
    parser.add_argument("--verify-single", type=int, default=2,
                        help="rows to cross-check against single-row encoding")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.manual_seed(common.SEED)

    rows = common.read_jsonl(args.prompts)
    if args.limit:
        rows = rows[: args.limit]
    rows = [r for r in rows if os.path.exists(r["image"])]
    shard = rows[args.shard :: args.num_shards]
    out_path = args.out if args.num_shards == 1 else \
        args.out.replace(".jsonl", f".shard{args.shard}.jsonl")

    if args.resume and os.path.exists(out_path):
        # Skipping finished rows is what makes a killed shard cheap to restart.
        # A partially written last line is tolerated, since a kill can land
        # mid-write.
        done = set()
        bad = 0
        with open(out_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                done.add((row["image"], row["prompt"]))
        before = len(shard)
        shard = [r for r in shard if (r["image"], r["prompt"]) not in done]
        print(f"[resume] {len(done)} rows already done, {bad} unreadable line(s); "
              f"{before} -> {len(shard)} rows left", flush=True)
        handle_mode = "a"
    else:
        handle_mode = "w"

    print(f"shard {args.shard}/{args.num_shards}: {len(shard)} prompts "
          f"(of {len(rows)} with images) -> {out_path} [{handle_mode}]", flush=True)
    if not shard:
        print("[done] nothing left to generate", flush=True)
        return

    target, processor = common.load_target(device=device)
    processor.tokenizer.padding_side = "left"
    eos = processor.tokenizer.eos_token_id

    # --- correctness guard: batched encoding must equal single-row encoding ---
    if args.verify_single:
        for row in shard[: args.verify_single]:
            image = common.load_image(row["image"], args.long_side)
            single_ids, _, _ = common.encode(processor, row["prompt"], image)
            batched = collate(processor, [(row, image)])
            batched_ids = batched["input_ids"]
            mask = batched["attention_mask"][0].bool()
            assert torch.equal(batched_ids[0][mask], single_ids[0]), \
                f"batched encoding differs from single-row encoding for {row['image']}"
        print(f"encoding guard passed on {args.verify_single} rows", flush=True)

    started = time.time()
    written = 0
    done = 0
    with open(out_path, handle_mode) as handle:
        for start in range(0, len(shard), args.batch):
            chunk = shard[start : start + args.batch]
            loaded = []
            for row in chunk:
                try:
                    loaded.append((row, common.load_image(row["image"], args.long_side)))
                except Exception as exc:                       # unreadable image
                    print(f"  skip {row['image']}: {exc}", flush=True)
            if not loaded:
                continue

            inputs = collate(processor, loaded).to(device)
            with torch.inference_mode():
                generated = target.generate(
                    **inputs, do_sample=False, max_new_tokens=args.max_new,
                    use_cache=True)
            prompt_len = inputs["input_ids"].shape[1]
            new_ids = generated[:, prompt_len:]

            mask = inputs["attention_mask"]
            for i, (row, _) in enumerate(loaded):
                real = int(mask[i].sum())
                input_ids = inputs["input_ids"][i, mask[i].bool()].tolist()
                gen = new_ids[i].tolist()
                if eos is not None and eos in gen:
                    gen = gen[: gen.index(eos) + 1]
                record = dict(row)
                record.update(input_ids=input_ids, gen_ids=gen,
                              n_prompt=len(input_ids),
                              finish="eos" if (eos is not None and gen and gen[-1] == eos)
                                     else "length")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            done += len(loaded)
            if (start // args.batch) % 10 == 0:
                rate = done / max(1e-9, time.time() - started)
                eta = (len(shard) - done) / max(1e-9, rate)
                print(f"  {done}/{len(shard)} rows, {rate:.2f} rows/s, "
                      f"eta {eta/60:.1f} min", flush=True)

    print(f"done: {written} rows in {(time.time()-started)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
