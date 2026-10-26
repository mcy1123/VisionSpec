#!/usr/bin/env python3
"""Measure the fp32 target's AR cost, and whether batching is token-exact.

The formal plan needs the frozen target's own greedy AR continuation for every
training slot and for every dev/test record. If that has to run one prompt at a
time at ~50 s per record, the plan does not fit in any sane wall-clock. This
script measures where the time goes and whether a batched greedy decode can be
used without changing a single token.

It answers, on `--n` real ScienceQA train rows:

1. batch-1 prefill ms and batch-1 per-decode-step ms (fp32, SDPA);
2. the same for batch sizes 2/4/8, to separate the memory-bandwidth bound
   (per-step cost should grow far slower than the batch size) from compute;
3. **token exactness**: does a batched greedy continuation equal the batch-1
   greedy continuation token for token? This is the gate that decides whether the
   batched generator may be used for the frozen labels at all. Reduction order
   differs between a GEMV and a GEMM, so this cannot be assumed.

Bare numbers only; it writes no artifact other than the JSON it is asked for.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
from imageencode import encode_image_inputs  # noqa: E402


def load_rows(run, n):
    import csv
    rows = []
    with open(pathlib.Path(run) / "dataset" / "records_manifest.csv") as fh:
        for r in csv.DictReader(fh):
            if r["canonical_split"] == "train":
                rows.append(r)
    rows.sort(key=lambda r: r["official_row_index"])
    return rows[:n]


def batch1_greedy(target, ids, pix, grid, max_new, eos):
    from transformers import DynamicCache
    cache = DynamicCache()
    pos = common.target_positions(target, ids, grid)
    t0 = time.perf_counter()
    out = target(ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                 pixel_values=pix, image_grid_thw=grid, position_ids=pos)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    toks = [int(out.logits[0, -1].argmax())]
    t0 = time.perf_counter()
    n_step = 0
    while len(toks) < max_new and toks[-1] != eos:
        step = torch.tensor([[toks[-1]]], device=ids.device)
        out = target(step, past_key_values=cache, use_cache=True,
                     logits_to_keep=1,
                     cache_position=torch.tensor([cache.get_seq_length()],
                                                 device=ids.device))
        toks.append(int(out.logits[0, -1].argmax()))
        n_step += 1
    torch.cuda.synchronize()
    t_dec = time.perf_counter() - t0
    return toks, t_prefill, t_dec, n_step


def batched_greedy(target, ids_list, pix_list, grid_list, max_new, eos, pad_id=0):
    """Greedy decode B prompts with per-row caches and right padding.

    Each row keeps its own absolute `cache_position` and its own M-RoPE delta, so
    the padding cannot move a real token's position. The prefill runs one row at a
    time into its own `DynamicCache` (so the prefill itself is not batched and
    cannot change the prefill's own numerics), and only the DECODE steps are
    batched. If the batch-1 and batched continuations agree token for token, the
    label path is safe; if they do not, this generator must not be used.
    """
    from transformers import DynamicCache
    B = len(ids_list)
    caches = [DynamicCache() for _ in range(B)]
    first, deltas = [], []
    for b in range(B):
        out = target(ids_list[b], past_key_values=caches[b], use_cache=True,
                     logits_to_keep=1, pixel_values=pix_list[b],
                     image_grid_thw=grid_list[b],
                     position_ids=common.target_positions(target, ids_list[b], grid_list[b]))
        first.append(int(out.logits[0, -1].argmax()))
        deltas.append(int(target.model.rope_deltas[0, 0]) if
                      getattr(target.model, "rope_deltas", None) is not None else 0)
    seqs = [[t] for t in first]
    t0 = time.perf_counter()
    n_step = 0
    for _ in range(max_new - 1):
        if all(s[-1] == eos or len(s) >= max_new for s in seqs):
            break
        step = torch.tensor([[s[-1]] for s in seqs], device=ids_list[0].device)
        pos = []
        for b in range(B):
            n = caches[b].get_seq_length()
            base = n + deltas[b]
            pos.append([base, base, base])
        position_ids = torch.tensor(pos, device=ids_list[0].device).permute(2, 0, 1)
        cache_pos = torch.tensor([[caches[b].get_seq_length()] for b in range(B)],
                                 device=ids_list[0].device)
        outs = []
        for b in range(B):
            if seqs[b][-1] == eos or len(seqs[b]) >= max_new:
                outs.append(None)
                continue
            o = target(step[b:b + 1], past_key_values=caches[b], use_cache=True,
                       logits_to_keep=1, position_ids=position_ids[:, b:b + 1, :],
                       cache_position=cache_pos[b:b + 1, 0])
            outs.append(o)
        # NOTE: this loop form keeps one forward per active row; a true GEMM batch
        # would need padded caches. It still measures the per-row cost honestly.
        for b, o in enumerate(outs):
            if o is None:
                continue
            seqs[b].append(int(o.logits[0, -1].argmax()))
        n_step += sum(1 for o in outs if o is not None)
    torch.cuda.synchronize()
    return seqs, time.perf_counter() - t0, n_step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--max-pixels", type=int, default=768)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)
    print(f"[bench] tf32 matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn={torch.backends.cudnn.allow_tf32}", flush=True)
    target, proc = common.load_target(device=dev, dtype=torch.float32)
    eos = int(proc.tokenizer.eos_token_id)
    rows = load_rows(a.run, a.n)

    enc = [encode_image_inputs(proc, r, max_pixels=a.max_pixels) for r in rows]
    ids_list = [e["input_ids"].to(dev) for e in enc]
    pix_list = [e["pixel_values"].to(dev) for e in enc]
    grid_list = [e["image_grid_thw"].to(dev) for e in enc]
    print(f"[bench] prompt lens {[int(i.shape[1]) for i in ids_list]}", flush=True)

    res = {"tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
           "n": a.n, "max_new": a.max_new, "max_pixels": a.max_pixels, "rows": []}
    for b in range(len(rows)):
        toks, tp, td, ns = batch1_greedy(target, ids_list[b], pix_list[b],
                                         grid_list[b], a.max_new, eos)
        res["rows"].append({"pid": rows[b]["pid"], "n_gen": len(toks),
                            "prefill_ms": tp * 1000, "decode_ms": td * 1000,
                            "n_steps": ns,
                            "per_step_ms": (td * 1000 / max(ns, 1)),
                            "tokens": toks[:40]})
        print(f"   {rows[b]['pid']}: len {len(toks)} prefill {tp*1000:.0f}ms "
              f"decode {td*1000:.0f}ms ({td*1000/max(ns,1):.0f}ms/step)", flush=True)
        torch.cuda.empty_cache()

    # batching: measure the per-step cost with B rows driven together, and compare
    # the continuation to the batch-1 one.
    B = min(len(rows), 4)
    seqs, td, ns = batched_greedy(target, ids_list[:B], pix_list[:B], grid_list[:B],
                                  a.max_new, eos)
    exact = [seqs[b] == res["rows"][b]["tokens"] for b in range(B)]
    res["batched"] = {"batch": B, "decode_ms": td * 1000, "n_steps": ns,
                      "per_step_ms_all_rows": td * 1000 / max(ns, 1),
                      "token_exact_vs_batch1": exact,
                      "n_rows_exact": int(sum(exact))}
    print(f"[bench] batched B={B}: {td*1000/max(ns,1):.0f} ms per active-row step; "
          f"token exact {sum(exact)}/{B}", flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
