#!/usr/bin/env python3
"""Recompute the OPD pool's teacher top-k on the decode attention branch.

The shipped pools (`video_greedy_499.pt`, `video_opd_pool_warm_499.pt`) store the
teacher distribution at every position, computed by `opd_rollout.target_topk`
with `use_cache=False` and no mask -- the materialised-mask branch, which phase 0
measured to be a different (and worse) forward than the one decode uses. The
committed tokens and the anchors are untouched: this rewrites ONLY
`topk_ids`/`topk_val`, and asserts that the row's `input_ids` still reproduce
byte-for-byte from the cache before doing so.

    python recollect_topk_aligned.py --pool ... --shards 3 --shard 0 --gpu 6 \
        --out ..._aligned_shard0.pt

Per shard, the first row is gated with `verify_aligned_to_decode`: the aligned
context must match the decode prefill's context bit-exactly, otherwise the shard
aborts (no silent fallback to the old numbers).
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
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from shared.aligned_forward import target_context_aligned, target_topk_aligned, verify_aligned_to_decode
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/mnt/local/mcy/GLANCE/data/video_opd_pool_warm_499.pt")
    ap.add_argument("--index", default="/mnt/local/mcy/GLANCE/data/pool_499.parquet")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    blob = torch.load(a.pool, weights_only=False)
    recs = blob["records"]
    _df = pd.read_parquet(a.index)
    # single-prompt pools key by video_name; multi-prompt pools by the unique
    # "<video>__p<pid>" task, since video_name repeats there.
    _key = "task" if ("task" in _df.columns and _df["task"].is_unique) else "video_name"
    idx = _df.set_index(_key)
    print(f"index key: {_key!r}", flush=True)
    if a.shards > 1:
        recs = recs[a.shard::a.shards]
    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    prompts = idx["question"].astype(str).unique()
    print(f"shard {a.shard}/{a.shards}: {len(recs)} rows, {len(prompts)} distinct "
          f"prompt(s): {[p[:40] for p in prompts[:4]]}", flush=True)
    _vtext_cache = {}

    def vtext_for(task):
        """Per-row prompt: multi-prompt pools repeat a video with different
        questions, so a single pool-wide template would silently encode the wrong
        prompt (and the per-row prompt_len assert below would be the only signal)."""
        if task not in _vtext_cache:
            q = str(idx.loc[task]["question"])
            _vtext_cache[task] = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"},
                                              {"type": "text", "text": q}]}],
                add_generation_prompt=True, tokenize=False)
        return _vtext_cache[task]

    started = time.time()
    out = []
    for i, r in enumerate(recs):
        row = idx.loc[r["task"]]
        ins = proc(text=[vtext_for(r["task"])], videos=[row["cache_path"]],
                   return_tensors="pt", fps=float(row["fps"]))
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        full = r["input_ids"][: a.max_seq_len].to(dev).long()
        ins_ids = ins["input_ids"].to(dev)
        assert int(ins_ids.shape[1]) == int(r["prompt_len"]), (
            f"{r['task']}: prompt_len {ins_ids.shape[1]} != pool {r['prompt_len']} "
            f"-> the index row's prompt does not match the one this record was "
            f"rolled out under")
        assert torch.equal(ins_ids[0], full[: r["prompt_len"]]), (
            f"{r['task']}: prompt token ids drifted from the pool")
        if i == 0:
            d, cos = verify_aligned_to_decode(target, ins_ids, None, None, pv, vg)
            print(f"  gate on {r['task']}: aligned-vs-decode-prefill ctx "
                  f"max|Δ| {d:.6f} cos {cos:.8f}", flush=True)
            assert d == 0.0, f"aligned forward is not bit-identical to decode ({d})"
        tk_ids, tk_val = target_topk_aligned(target, full[None], a.topk, None, None, pv, vg)
        new = dict(r)
        new["topk_ids"] = tk_ids
        new["topk_val"] = tk_val
        out.append(new)
        if (i + 1) % 25 == 0:
            rate = (i + 1) / (time.time() - started)
            print(f"  {i+1}/{len(recs)} rows, {rate:.2f} rows/s, "
                  f"eta {(len(recs)-i-1)/max(rate,1e-9)/60:.1f} min", flush=True)

    cfg = dict(blob.get("config", {}))
    cfg["aligned_forward"] = True
    cfg["topk"] = a.topk
    cfg["shard"] = a.shard
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"records": out, "config": cfg}, a.out)
    print(f"wrote {a.out}: {len(out)} records "
          f"({os.path.getsize(a.out)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
