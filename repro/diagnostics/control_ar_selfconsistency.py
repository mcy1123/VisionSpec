#!/usr/bin/env python3
"""The repository's own lossless control, measured here: does the target
reproduce ITSELF in bf16?

`experiments/audit_lossless.py` says the fp32 gate exists because "in bf16 the
target does not reproduce *itself* on every prompt with no drafter in the loop at
all, because a tree pass and a single-token pass reduce in different orders".
That statement is about ARITHMETIC, and it is testable without any drafter: decode
the same prompt twice by two different AR paths and compare.

  A. `greedy_generate` -- KV-cache incremental, one token per step.
  B. re-prefill      -- every step re-runs the whole sequence from scratch.

Both are plain autoregressive greedy decodes of the same model at the same dtype;
neither involves the draft head. If A and B diverge, then bf16 SPEC-vs-AR
divergence cannot be attributed to the drafter, and the fp32 audit is the right
gate. If they agree everywhere while SPEC diverges, the divergence is specific to
the tree pass and deserves a real investigation.

Run in bf16 (the deployment dtype). No try/except.
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

import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from glance.decode import greedy_generate
from glance.vl import VisionPositions
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=192,
                    help="kept short on purpose: the first divergences observed on "
                         "video rows are at tokens 12-181, and path B costs a full "
                         "prefill per step")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    df = pd.read_parquet(a.rows).head(a.n)
    target, _ = common.load_target(device=dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    eos = proc.tokenizer.eos_token_id

    out = []
    for i, row in df.reset_index(drop=True).iterrows():
        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"},
                                          {"type": "text", "text": row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        ids = ins["input_ids"].to(dev)
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        n_in = int(ids.shape[1])

        a_ids, _ = greedy_generate(target, ids, a.max_new, [eos],
                                   positions=VisionPositions(pixel_values_videos=pv,
                                                             video_grid_thw=vg),
                                   return_stats=True)
        a_tokens = a_ids[0, n_in:].tolist()

        # B: re-prefill every step, no KV cache.
        b_tokens = []
        for _ in range(a.max_new):
            full = torch.cat([ids, torch.tensor([b_tokens], device=dev)], dim=1) \
                if b_tokens else ids
            pos, _ = target.model.get_rope_index(full, None, vg, None)
            with torch.no_grad():
                o = target(full, position_ids=pos, pixel_values_videos=pv,
                           video_grid_thw=vg, use_cache=False, logits_to_keep=1)
            tok = int(o.logits[0, -1].argmax())
            b_tokens.append(tok)
            if tok == eos:
                break

        m = min(len(a_tokens), len(b_tokens))
        j = next((t for t in range(m) if a_tokens[t] != b_tokens[t]), None)
        rec = {"video": str(row["video_name"]), "L": n_in,
               "kv_len": len(a_tokens), "reprefill_len": len(b_tokens),
               "first_mismatch": j,
               "agree": bool(j is None and len(a_tokens) == len(b_tokens))}
        out.append(rec)
        print(f"  {rec['video'][:16]:16s} KV-cache {len(a_tokens):4d} tok vs "
              f"re-prefill {len(b_tokens):4d} tok -> "
              f"{'AGREE' if rec['agree'] else f'diverge@{j}'}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": out}, open(a.out, "w"), indent=1)
    n_ok = sum(1 for r in out if r["agree"])
    verdict = (
        "The target does not reproduce itself in bf16 either -- the SPEC/AR "
        "divergence is arithmetic, not the drafter."
        if n_ok < len(out) else
        "The two AR paths agree here; the divergence is specific to the tree pass "
        "and needs a real investigation.")
    print(f"\nAR self-consistency (bf16): {n_ok}/{len(out)} identical. {verdict}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
