#!/usr/bin/env python3
"""One video head, one explicit row list: tau + per-round anchors + AR/SPEC timing.

This is the eval vehicle for the video-OPD long task. It is additive: it does not
touch `eval_video_head.py`, whose numbers the earlier video stages were reported
with. What it adds is what this task needs and that script does not produce:

  * the rows come from an EXPLICIT parquet (the frozen 30-row held-out), not from
    a split filter -- so a split re-shuffle cannot silently change the denominator;
  * `on_round` is captured, so every arm's anchor positions, accepted lengths and
    rounds-to-60%-coverage are recorded (task sections 6.4/6.5);
  * `--ar` additionally runs the repository's own autoregressive greedy oracle
    (`glance.decode.greedy_generate`) on the SAME encoded inputs, interleaved with
    the speculative run, and checks the two token streams for exact equality --
    which is both the lossless gate and the speedup baseline (task sections 4/5).

Row timing is wall-clock around the whole row (video decode + processor + prefill
+ decode) and, separately, the decode-only window the repository's DecodeStats
reports. The first `--warmup` rows are excluded from aggregates but never from the
lossless check. No try/except: a failure surfaces with its traceback.
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
import statistics as st
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
import glance
from glance.decode import greedy_generate
from glance.vl import VisionPositions
from transformers import AutoProcessor


def coverage_at(rounds, k, prompt_len, gen_tokens):
    """Fraction of the generated response reached after `k` draft rounds.

    Same definition the image-side diagnosis used (`多模态OPD诊断2_2026-09-24.md`
    section 3.1): a rollout truncated to k rounds covers this share of the
    response. `rounds[i]["pos"]` is the absolute block start, and a round commits
    `accepted + 1` tokens, so the end of round k is round k's pos + committed.
    """
    if k <= 0:
        return 0.0
    if len(rounds) <= k:
        return 1.0
    end = rounds[k - 1]["pos"] + rounds[k - 1]["accepted"] + 1
    return max(0.0, (end - prompt_len) / max(gen_tokens, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True,
                    help="parquet whose rows are exactly the rows to score")
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch", type=int, default=8)
    ap.add_argument("--ar", action="store_true",
                    help="also run the AR oracle, check losslessness, time it")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--shards", type=int, default=1,
                    help="split the row list into N strided shards (one process "
                         "per GPU); merge the JSONs with merge_eval_jsons.py")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="fp32 is the losslessness-gate dtype: the repository's own "
                         "audit runs fp32 because a bf16 tree pass and a bf16 "
                         "single-token pass reduce in different orders and near-ties "
                         "land on either side (audit_lossless.py docstring).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    dt = torch.float32 if a.dtype == "float32" else None
    df = pd.read_parquet(a.rows)
    if a.n:
        df = df.head(a.n)
    if a.shards > 1:
        df = df.iloc[a.shard::a.shards].reset_index(drop=True)
    print(f"rows {a.rows}: scoring {len(df)} rows (shard {a.shard}/{a.shards}), "
          f"head {a.head}, dtype {a.dtype}, max_new {a.max_new} "
          f"b{a.budget}k{a.branch} ar={a.ar}", flush=True)

    target, _ = common.load_target(device=dev, dtype=dt)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=dev, dtype=dt).eval()
    eos = proc.tokenizer.eos_token_id

    res = []
    for i, row in df.reset_index(drop=True).iterrows():
        t_row0 = time.perf_counter()
        text = proc.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": row["question"]}]}],
            add_generation_prompt=True, tokenize=False)
        ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
                   fps=float(row["fps"]))
        ids = ins["input_ids"].to(dev)
        pv = ins["pixel_values_videos"].to(dev)
        vg = ins["video_grid_thw"].to(dev)
        n_in = int(ids.shape[1])
        torch.cuda.synchronize()

        def run_spec():
            pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
            rounds = []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, stats = glance.generate(head, target, ids, a.max_new, [eos],
                                         budget=a.budget, branch_k=a.branch,
                                         temperature=0.0, positions=pos,
                                         return_stats=True, timing="decode_only",
                                         on_round=rounds.append)
            torch.cuda.synchronize()
            return out, stats, time.perf_counter() - t0, rounds

        def run_ar():
            pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, stats = greedy_generate(target, ids, a.max_new, [eos],
                                         positions=pos, return_stats=True)
            torch.cuda.synchronize()
            return out, stats, time.perf_counter() - t0

        # Interleave the order so a drifting clock cannot favour one arm.
        if i % 2 == 0:
            spec_out, spec_st, spec_s, spec_rounds = run_spec()
            ar_out, ar_st, ar_s = (run_ar() if a.ar else (None, None, None))
        else:
            ar_out, ar_st, ar_s = (run_ar() if a.ar else (None, None, None))
            spec_out, spec_st, spec_s, spec_rounds = run_spec()
        t_row1 = time.perf_counter()

        n_spec = int(spec_out.shape[1]) - n_in
        spec_ids = spec_out[0, n_in:].tolist()
        anchor_rel = [int(r["pos"]) - n_in for r in spec_rounds]
        accepted = [int(r["accepted"]) for r in spec_rounds]
        n_rounds = len(anchor_rel)
        rec = {"video": str(row["video_name"]), "L": n_in,
               "prompt_len": n_in, "gen_tokens": n_spec,
               "rounds": int(spec_st.rounds), "tau": float(spec_st.tau),
               "Ea": float(spec_st.tau) - 1.0,
               "hit_eos": bool(spec_ids and spec_ids[-1] == eos),
               "tree_size": float(spec_st.tree_size),
               "spec_wall_s": spec_s, "spec_decode_ms_per_tok": float(spec_st.ms_per_token),
               "row_wall_s": t_row1 - t_row0, "warmup": i < a.warmup,
               "width": int(row.get("width", 0)), "height": int(row.get("height", 0)),
               "n_frames": int(row.get("n_frames", 0)),
               "anchor_rel": anchor_rel, "accepted": accepted,
               "cov60": coverage_at(spec_rounds, 60, n_in, n_spec),
               "front32_frac": (sum(1 for p in anchor_rel if p < 32) / max(n_rounds, 1)),
               "third_frac": [
                   sum(1 for p in anchor_rel if p < n_spec / 3) / max(n_rounds, 1),
                   sum(1 for p in anchor_rel if n_spec / 3 <= p < 2 * n_spec / 3) / max(n_rounds, 1),
                   sum(1 for p in anchor_rel if p >= 2 * n_spec / 3) / max(n_rounds, 1)]}
        if a.ar:
            ar_ids = ar_out[0, n_in:].tolist()
            m = min(len(ar_ids), len(spec_ids))
            first_bad = next((j for j in range(m) if ar_ids[j] != spec_ids[j]), None)
            rec.update({
                "ar_tokens": len(ar_ids),
                "ar_hit_eos": bool(ar_ids and ar_ids[-1] == eos),
                "ar_wall_s": ar_s,
                "ar_decode_ms_per_tok": float(ar_st.ms_per_token),
                "ar_ms_per_tok_wall": float(ar_st.ms_per_token_wall),
                "ar_prefill_ms": float(ar_st.prefill_ms),
                "lossless": bool(first_bad is None and len(ar_ids) == len(spec_ids)),
                "first_mismatch": first_bad,
                "spec_ids": spec_ids, "ar_ids": ar_ids,
            })
        res.append(rec)
        print(f"  [{i+1}/{len(df)}] {rec['video'][:16]:16s} L={n_in:4d} "
              f"gen={n_spec:4d} rounds={rec['rounds']:3d} tau={rec['tau']:.3f} "
              f"spec={spec_s:6.2f}s"
              + (f" ar={ar_s:6.2f}s lossless={rec.get('lossless')} "
                 f"mm={rec.get('first_mismatch')}" if a.ar else ""), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"config": vars(a), "rows": res}, open(a.out, "w"), indent=1)

    keep = [r for r in res if not r["warmup"]]
    taus = [r["tau"] for r in keep]
    sd = f"{st.stdev(taus):.4f}" if len(taus) > 1 else "n/a"
    print(f"\nn={len(keep)} (warmup dropped {len(res)-len(keep)})  "
          f"tau median {st.median(taus):.4f} mean {st.fmean(taus):.4f} "
          f"min {min(taus):.4f} max {max(taus):.4f} sd {sd}")
    print(f"Ea median {st.median(r['Ea'] for r in keep):.4f}  "
          f"gen median {st.median(r['gen_tokens'] for r in keep):.0f}")
    print(f"no-EOS {sum(1 for r in keep if not r['hit_eos'])}/{len(keep)}  "
          f"spec wall median {st.median(r['spec_wall_s'] for r in keep):.1f}s")
    if a.ar:
        bad = [r["video"] for r in res if not r.get("lossless")]
        print(f"SPEC wall total {sum(r['spec_wall_s'] for r in res):.0f}s  "
              f"AR wall total {sum(r['ar_wall_s'] for r in res):.0f}s  "
              f"row-wall total {sum(r['row_wall_s'] for r in res):.0f}s")
        print(f"LOSSLESS {len(res)-len(bad)}/{len(res)} exact; mismatched: {bad}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
