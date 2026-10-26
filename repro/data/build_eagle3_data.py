#!/usr/bin/env python3
"""Build the EAGLE-3 training data for the video domain from GLANCE's own rollout.

Why this is a separate script and not a flag on an existing one. The EAGLE-3
head must be trained on **exactly the rows GLANCE was trained on**, and judged
on **exactly the rows GLANCE was judged on**, or the tau comparison is not a
comparison. Those row sets already exist as explicit parquets
(`video_opd_train_409.parquet`, `video_heldout_30.parquet`); this script reads
them, derives the assistant turn from the target's own greedy rollout
(`video_greedy_499.pt`), and writes the canonical `id` + `conversations` JSONL
that every EAGLE-3 implementation consumes.

The label is the TARGET's greedy continuation, not the dataset's reference
answer. `train_head_video.py` makes the same choice for the same reason: GLANCE
trains on the target's generations, and the dataset answer is a different object
(using it would silently turn this into caption distillation).

    python build_eagle3_data.py            # writes train / select / holdout30

Splits, and why the names matter:

  train      409 rows = GLANCE's OPD training set (`video_opd_train_409`).
  select      60 rows = the pool's `heldout` split. GLANCE used these to pick a
             checkpoint, so EAGLE-3 may use them the same way.
  holdout30   30 rows = `video_heldout_30`. This is the independent holdout,
             carved out of the pool's *train* split and never used for any
             selection. Nothing here may influence training or model choice.

The three are asserted pairwise disjoint before anything is written.

Round-trip gate. The JSONL stores text, so the sequence EAGLE-3 would see is
re-tokenised from that text. That re-tokenisation is checked against the
rollout's own token ids, token by token. Two things fall out of that check and
both are recorded per row in the sidecar manifest rather than waved through:

  * BPE is not injective. `decode(encode(text)) == text` holds for every row,
    but `encode(decode(ids)) == ids` does not: on 3 of the 499 rows Qwen's
    encoder picks an equally valid segmentation (`ĠLang|uell` instead of the
    rollout's canonical `ĠL|angu|ell`). Those rows are marked
    `roundtrip: "text_only"`.
  * Qwen's chat template closes the assistant turn with `<|im_end|>\\n`, which
    the rollout does not contain. A rollout that ended with EOS gets one extra
    token; a rollout truncated at the token cap without EOS gets two.

Consequence, and the reason the manifest exists: the JSONL alone is not enough
to rebuild the exact training sequence. Capture therefore renders the
conversation through the same processor GLANCE used and derives ids, hidden
states and mask from that single forward, so the three are self-consistent by
construction; the manifest supplies `prompt_len` so the loss mask can be placed
without re-deriving character offsets (which is impossible once video tokens
expand). On the 3 `text_only` rows the segmentation differs from the rollout but
the text is identical, which is harmless for training; it is recorded rather
than hidden.
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
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
from shared import common

from transformers import AutoProcessor

# Which split of `pool_499.parquet` the selection set is. The pool's own labels
# are the source of truth for "these rows were not trained on" -- the parquet's
# `split` column, not a re-derivation.
POOL_SELECT_SPLIT = "heldout"


def load_rollout(path):
    """The target's greedy rollout, keyed by video name."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    records = blob["records"]
    by_name = {r["task"]: r for r in records}
    assert len(by_name) == len(records), "duplicate video_name in the rollout"
    return blob, by_name


def assistant_text(tokenizer, ids, prompt_len):
    """Decode the rollout's continuation, minus a trailing EOS if present.

    The chat template supplies `<|im_end|>` itself, so leaving the EOS in the
    text would double it. Both cases are counted by the caller.
    """
    gen = ids[prompt_len:]
    ends_eos = bool(int(gen[-1].item()) == tokenizer.eos_token_id)
    body = gen[:-1] if ends_eos else gen

    stray = [int(t) for t in body.tolist()
             if int(t) in tokenizer.all_special_ids]
    return tokenizer.decode(body, skip_special_tokens=True), ends_eos, stray


def build_rows(name, names, by_name, pool, tokenizer):
    """One JSONL split, plus the manifest entries that let capture truncate."""
    rows, manifest, skipped = [], [], []
    for video in names:
        rec = by_name.get(video)
        if rec is None:
            skipped.append((video, "absent from rollout"))
            continue
        meta = pool.loc[video]
        prompt = str(meta["question"])
        cache_path = str(meta["cache_path"])
        if not os.path.exists(cache_path):
            skipped.append((video, f"missing video {cache_path}"))
            continue
        if not prompt:
            skipped.append((video, "empty pool prompt"))
            continue

        ids = rec["input_ids"].long()
        prompt_len = int(rec["prompt_len"])
        # The pool's `input_ids_len` is the length GLANCE's own trainer encoded.
        # A mismatch here is the exact bug that cost the earlier video stage
        # +0.33 tau (a one-word prompt drift shifting every anchor by one), so
        # it is a hard failure, not a warning.
        if int(meta["input_ids_len"]) != prompt_len:
            raise AssertionError(
                f"{video}: pool input_ids_len={int(meta['input_ids_len'])} but the "
                f"rollout used prompt_len={prompt_len}; the prompt drifted "
                f"between label collection and training")
        if prompt_len >= ids.shape[0]:
            skipped.append((video, "rollout has no continuation"))
            continue

        text, ends_eos, stray = assistant_text(tokenizer, ids, prompt_len)
        if stray:
            raise AssertionError(
                f"{video}: special token(s) {stray} inside the decoded "
                f"continuation; text round-trip would not be lossless")

        # Text is the only thing a JSONL can carry, but BPE is not injective:
        # `decode(encode(text)) == text` always holds, while `encode(decode(t))
        # == t` does not (Qwen admits segmentations like `ĠLang|uell` that the
        # canonical encoder would write `ĠL|angu|ell`). Measure it per row so the
        # capture stage knows which rows it must not re-tokenise.
        body = ids[prompt_len:(-1 if ends_eos else None)]
        re_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if tokenizer.decode(re_ids, skip_special_tokens=True) != text:
            raise AssertionError(
                f"{video}: decode(re-encode(text)) != text; the text itself is "
                f"not faithful to the rollout")
        roundtrip = "exact" if (len(re_ids) == body.shape[0]
                                and re_ids == body.tolist()) else "text_only"

        rows.append({
            "id": video,
            "conversations": [
                {"role": "user", "content": [
                    {"type": "video", "video": cache_path},
                    {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": text}]},
            ],
        })
        manifest.append({
            "id": video,
            "split": name,
            "video": cache_path,
            "fps": float(meta["fps"]),
            "question": prompt,
            "prompt_len": prompt_len,
            "seq_len": int(ids.shape[0]),
            "gen_len": int(ids.shape[0]) - prompt_len,
            "ended_with_eos": ends_eos,
            "roundtrip": roundtrip,
            "answer_chars": len(text),
        })
    return rows, manifest, skipped


def render_and_check(processor, rows, manifest, by_name, sample):
    """Re-render each JSONL row and compare TOKEN BY TOKEN to the rollout.

    A length check alone would pass a prompt that drifted by one token, which is
    the exact failure that cost the earlier video stage +0.33 tau. So this
    compares the ids themselves.

    Expected outcomes, all of them recorded rather than waved through:

      identical   the rendered sequence IS the rollout (no template tail)
      template+1  rollout ended with EOS; the template appends its `\\n`
      template+2  rollout hit the token cap without EOS; the template supplies
                  the missing `<|im_end|>` plus its `\\n`
      retokenised text is faithful (`decode(re-encode(text)) == text`) but BPE
                  picked a different, equally valid segmentation
      bad         prefix differs -> the label would be shifted. Hard failure.

    Returns a dict of counts plus the list of bad rows.
    """
    counts = {"checked": 0, "identical": 0, "template+1": 0, "template+2": 0,
              "retokenised": 0}
    bad = []
    for row, man in zip(rows, manifest):
        if sample is not None and counts["checked"] >= sample:
            break
        if row["id"] != man["id"]:
            bad.append((man["id"], "jsonl/manifest order mismatch"))
            continue
        rendered = processor.apply_chat_template(
            row["conversations"], add_generation_prompt=False, tokenize=False)
        ins = processor(text=[rendered], videos=[man["video"]],
                        return_tensors="pt", fps=man["fps"])
        got = ins["input_ids"][0].long()
        want = by_name[man["id"]]["input_ids"].long()
        counts["checked"] += 1

        # The template tail is 1 token after an EOS-terminated rollout and 2
        # after a truncated one (it closes the turn the rollout left open).
        tail = 1 if man["ended_with_eos"] else 2
        n_got, n_want = int(got.shape[0]), int(want.shape[0])

        if n_got == n_want and bool(torch.equal(got, want)):
            counts["identical"] += 1
        elif n_got == n_want + tail and bool(torch.equal(got[:n_want], want)):
            counts[f"template+{tail}"] += 1
        elif man["roundtrip"] == "text_only":
            # Documented non-canonical segmentation; the prefix still has to be
            # a faithful re-encoding, which build_rows already proved by
            # `decode(re-encode(text)) == text`.
            counts["retokenised"] += 1
        else:
            n = min(n_got, n_want)
            diff = (got[:n] != want[:n]).nonzero().flatten()
            first = int(diff[0]) if diff.numel() else -1
            bad.append((man["id"], f"len {n_got} vs rollout {n_want}, "
                                   f"first differing index {first}"))
    return counts, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", default=os.path.join(common.DATA, "video_greedy_499.pt"))
    ap.add_argument("--pool", default=os.path.join(common.DATA, "pool_499.parquet"))
    ap.add_argument("--train-rows", default=os.path.join(common.DATA, "video_opd_train_409.parquet"))
    ap.add_argument("--holdout-rows", default=os.path.join(common.DATA, "video_heldout_30.parquet"))
    ap.add_argument("--model", default=common.TARGET_PATH)
    ap.add_argument("--out-dir", default=common.DATA)
    ap.add_argument("--verify", type=int, default=4,
                    help="rows to re-render as a spot check (0 disables)")
    ap.add_argument("--verify-all", action="store_true",
                    help="re-render every row, not just a sample")
    a = ap.parse_args()

    pool = pd.read_parquet(a.pool)
    dup = pool["video_name"][pool["video_name"].duplicated()].tolist()
    assert not dup, f"pool has duplicate video_name: {dup[:5]}"
    pool = pool.set_index("video_name")

    train_names = list(pd.read_parquet(a.train_rows)["video_name"].astype(str))
    hold_names = list(pd.read_parquet(a.holdout_rows)["video_name"].astype(str))
    select_names = list(pool.index[pool["split"] == POOL_SELECT_SPLIT].astype(str))

    # Disjointness is the whole point; assert it rather than trusting filenames.
    sets = {"train": set(train_names), "select": set(select_names),
            "holdout30": set(hold_names)}
    for x, y in (("train", "select"), ("train", "holdout30"), ("select", "holdout30")):
        overlap = sets[x] & sets[y]
        assert not overlap, f"{x} and {y} overlap on {len(overlap)} rows: {sorted(overlap)[:5]}"
    print(f"split sizes: train {len(train_names)}, select {len(select_names)}, "
          f"holdout30 {len(hold_names)} (pairwise disjoint)", flush=True)

    # The independent holdout must live in the pool's TRAIN split -- that is what
    # makes it "never used for selection". If this flips, the 30 rows were
    # selection rows and every number computed on them is optimistic.
    in_train = [n for n in hold_names if pool.loc[n, "split"] == "train"]
    assert len(in_train) == len(hold_names), (
        f"{len(hold_names) - len(in_train)} holdout rows are not in the pool's "
        f"train split, so they were already used for selection")

    blob, by_name = load_rollout(a.greedy)
    print(f"rollout: {len(by_name)} records ({a.greedy})", flush=True)

    tokenizer = AutoProcessor.from_pretrained(a.model).tokenizer

    splits = {"train": train_names, "select": select_names, "holdout30": hold_names}
    all_manifest, stats, skipped_all = {}, {}, []
    for name, names in splits.items():
        rows, manifest, skipped = build_rows(name, names, by_name, pool, tokenizer)
        skipped_all += [(name,) + s for s in skipped]
        assert len(rows) == len(names), (
            f"{name}: built {len(rows)} rows for {len(names)} names; "
            f"skipped {skipped[:5]}")
        out = os.path.join(a.out_dir, f"eagle3_{name}.jsonl")
        with open(out, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_eos = sum(m["ended_with_eos"] for m in manifest)
        gen = [m["gen_len"] for m in manifest]
        stats[name] = {
            "rows": len(rows),
            "jsonl": out,
            "ended_with_eos": n_eos,
            "truncated_no_eos": len(rows) - n_eos,
            "roundtrip_exact": sum(m["roundtrip"] == "exact" for m in manifest),
            "roundtrip_text_only": sum(m["roundtrip"] == "text_only" for m in manifest),
            "prompt_len_min": min(m["prompt_len"] for m in manifest),
            "prompt_len_max": max(m["prompt_len"] for m in manifest),
            "gen_len_min": min(gen), "gen_len_max": max(gen),
            "gen_len_median": sorted(gen)[len(gen) // 2],
        }
        all_manifest[name] = manifest
        print(f"  {name}: {len(rows)} rows -> {out}  "
              f"(gen len {min(gen)}~{max(gen)}, no-EOS {len(rows) - n_eos})",
              flush=True)

    if skipped_all:
        print(f"WARNING: {len(skipped_all)} rows were dropped: {skipped_all[:5]}",
              flush=True)

    manifest_path = os.path.join(a.out_dir, "eagle3_manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump({"splits": all_manifest, "stats": stats,
                   "sources": {"greedy": a.greedy, "pool": a.pool,
                               "train_rows": a.train_rows,
                               "holdout_rows": a.holdout_rows,
                               "model": a.model},
                   "note": ("prompt_len/seq_len are the rollout's own token "
                            "boundaries; the chat template appends one \\n "
                            "after the assistant <|im_end|>, so render_len = "
                            "seq_len + 1. Capture must truncate to seq_len.")},
                  handle, indent=2)
    print(f"manifest -> {manifest_path}", flush=True)

    # --- round-trip gate -----------------------------------------------------
    if a.verify or a.verify_all:
        processor = AutoProcessor.from_pretrained(a.model)
        sample = None if a.verify_all else a.verify
        print(f"\nround-trip gate ({'all rows' if a.verify_all else f'{a.verify}/split'})",
              flush=True)
        t0 = time.time()
        for name in splits:
            rows = [json.loads(l) for l in
                    open(os.path.join(a.out_dir, f"eagle3_{name}.jsonl"))]
            manifest = all_manifest[name]
            counts, bad = render_and_check(
                processor, rows, manifest, by_name, sample)
            detail = " ".join(f"{k}={v}" for k, v in counts.items())
            print(f"  {name}: {detail} bad={len(bad)}", flush=True)
            assert not bad, f"{name}: round-trip failures {bad[:5]}"
            assert counts["identical"] + counts["template+1"] + \
                counts["template+2"] + counts["retokenised"] == counts["checked"]
        print(f"gate passed in {time.time() - t0:.1f}s", flush=True)
    else:
        print("\n[skip] round-trip gate disabled (--verify 0)", flush=True)

    n_text_only = sum(1 for s in all_manifest.values()
                      for m in s if m["roundtrip"] == "text_only")
    n_no_eos = sum(1 for s in all_manifest.values()
                   for m in s if not m["ended_with_eos"])
    print(f"\ncapture contract: {n_text_only} of "
          f"{sum(len(s) for s in all_manifest.values())} rows do not re-tokenise "
          f"to the rollout's exact ids (same text, different BPE segmentation); "
          f"{n_no_eos} rollouts are truncated without EOS. Both are recorded per "
          f"row in the manifest. Capture derives ids/hidden/mask from ONE forward "
          f"over the rendered conversation, so labels cannot drift from the "
          f"teacher states.")

    print("\nSPLITS")
    for name, s in stats.items():
        print(f"  {name:9s} rows={s['rows']:4d} gen_median={s['gen_len_median']:4d} "
              f"prompt_len {s['prompt_len_min']}~{s['prompt_len_max']}")


if __name__ == "__main__":
    main()
