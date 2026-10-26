#!/usr/bin/env python3
"""Does in-block progressive refinement help a head that was trained for it?

The observation this tests. GLANCE trains the block head with a variable-mask
block prefix -- reveal length l uniform-ish over [1, B-1] -- precisely so the
head "learns every reveal length" (paper §4.3). Decode then uses exactly one of
them: `glance/decode.py` sets `block_ids[:, 1:] = mask_id`, i.e. l = 1, forever.
So a capability that was paid for during training is never spent at inference.

Why that should matter, measured. On our own round log, the lowest entropy
decile (target next-token distribution is a point mass) still commits only 6.85
of the 15 block offsets. With l = 1 the query at offset 15 sees the root token
and fourteen MASKs inside its own block; everything it knows about offsets 2..15
has to arrive through cross-attention to the target context. Under progressive
refinement the query at offset 15 can instead attend to the head's own
predictions at offsets 1..l -- the same shape it saw in training at reveal l+1.

This script is inference only. It does not retrain anything, so it measures a
lower bound: any benefit seen here is benefit obtained without a single
gradient step, and a head retrained to expect its own revealed tokens should do
better still.

    python refine_decode.py --head $GLANCE_ROOT/checkpoints/head \
        --n 20 --schedules "1,4,9,15" "1,15" "1,3,7,15"
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
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from shared import common

sys.path.insert(0, os.path.join(common.WORKSPACE, "GLANCE_repo"))

import glance
from glance.decode import gather_cache
from glance.tree import build_budget_tree, children_of, tree_mask
from glance.vl import VisionPositions


def draft_pass(head, target, fused, out_ids, stream, kv_head, start, block,
               mask_id, embed, reveal, pos_start):
    """One draft forward with `reveal` visible positions in the block.

    `reveal` is the l of the training objective: block positions [0, l) hold
    real tokens -- the root and, for a refinement pass, the head's own previous
    proposals -- and [l, block) hold MASK. Identical in shape to
    `train_head.build_blocks` at the same l.
    """
    block_ids = out_ids[:, start:start + block].clone()
    block_ids[:, reveal:] = mask_id
    # Positions start where the freshly supplied context starts, NOT where the
    # draft cache happens to sit: the first pass already cropped the cache to
    # `start`, so reading the cache length on a second pass would hand the head
    # 16 positions for a key of length len(fused) + 16. The invariant that holds
    # across rounds is start - len(fused) == the cache length before any pass.
    hidden = head(target_hidden=fused,
                  noise_embedding=embed(block_ids),
                  position_ids=stream[:, pos_start:start + block],
                  past_key_values=kv_head, use_cache=True, is_causal=False)
    logits = target.lm_head(hidden[:, 1 - block:, :])[0]
    kv_head.crop(start)
    return block_ids, logits


@torch.inference_mode()
def refine_generate(head, target, input_ids, max_new_tokens, stop_token_ids,
                    budget, branch_k, positions, schedule, on_round=None):
    """`glance.generate` with a refinement schedule instead of a single l = 1 pass.

    schedule[0] must be 1: the first pass is the normal decode-time draft.
    Each later entry l re-runs the head with the first l block positions holding
    its own current best guess, then re-reads the offset marginals.
    """
    device = input_ids.device
    block = head.block_size
    mask_id = head.mask_token_id
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    dtype = next(target.parameters()).dtype
    embed = target.get_input_embeddings()

    out_ids = torch.full((1, max_len + 2 * block + 4), mask_id, dtype=torch.long,
                         device=device)
    stream = torch.arange(out_ids.shape[1], device=device).unsqueeze(0)
    from transformers import DynamicCache

    kv_target = DynamicCache()
    kv_head = DynamicCache()

    positions.bind(target, input_ids)
    out = target(input_ids, past_key_values=kv_target, use_cache=True,
                 logits_to_keep=1, output_hidden_states=True,
                 **positions.prefill_inputs(input_ids, stream[:, :n_in]))
    out_ids[:, :n_in] = input_ids
    out_ids[:, n_in] = int(out.logits[0, -1].argmax())
    fused = common.context_feature(out.hidden_states, list(head.target_layer_ids))

    start = n_in
    taus = []
    passes = []
    while start < max_len:
        pos_start = start - fused.shape[1]
        # pass 1 at l = 1, exactly as decode does
        block_ids, logits = draft_pass(head, target, fused, out_ids, stream,
                                       kv_head, start, block, mask_id, embed, 1,
                                       pos_start)
        n_pass = 1
        for l in schedule[1:]:
            l = min(l, block - 1)
            block_ids[:, 1:l] = logits[:l - 1].argmax(dim=-1)
            block_ids, logits = draft_pass(head, target, fused, out_ids, stream,
                                           kv_head, start, block, mask_id, embed, l,
                                           pos_start)
            n_pass += 1

        top = torch.topk(F.log_softmax(logits.float(), dim=-1), branch_k, dim=-1)
        top_ids, top_logp = top.indices.cpu(), top.values.cpu()
        nodes = build_budget_tree(top_ids, top_logp, budget, branch_k)

        packed = torch.tensor([[int(out_ids[0, start])] + [n.token for n in nodes]],
                              device=device)
        packed_pos = torch.tensor([[start] + [start + n.depth for n in nodes]],
                                  device=device)
        n_prefix = kv_target.get_seq_length()
        out = target(packed, position_ids=positions(packed_pos),
                     past_key_values=kv_target, use_cache=True,
                     attention_mask=tree_mask(n_prefix, nodes, device, dtype),
                     output_hidden_states=True)
        packed_logits = out.logits[0]

        children = children_of(nodes)
        path, cursor = [], -1
        while True:
            token = int(packed_logits[0 if cursor == -1 else cursor + 1].argmax())
            hit = next((c for c in children.get(cursor, []) if nodes[c].token == token), None)
            if hit is None:
                bonus = token
                break
            path.append(hit)
            cursor = hit

        accepted = len(path)
        taus.append(accepted + 1)
        passes.append(n_pass)
        if on_round is not None:
            on_round({"round": len(taus) - 1, "accepted": accepted,
                      "tau": accepted + 1, "passes": n_pass})

        for offset, node_index in enumerate(path):
            out_ids[0, start + 1 + offset] = nodes[node_index].token
        out_ids[0, start + accepted + 1] = bonus

        keep = [0] + [i + 1 for i in path]
        gather_cache(kv_target, n_prefix, keep)
        fused = common.context_feature(out.hidden_states, list(head.target_layer_ids))[0][keep][None]
        start += accepted + 1
        if stop_token_ids and any(t in out_ids[0, n_in:start].tolist()
                                 for t in stop_token_ids):
            break

    final = out_ids[:, :min(start + 1, max_len)]
    if stop_token_ids:
        stops = torch.tensor(stop_token_ids, device=device)
        hit = torch.isin(final[0, n_in:], stops).nonzero(as_tuple=True)[0]
        if hit.numel() > 0:
            final = final[:, :n_in + hit[0] + 1]
    return final, taus, passes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", default=os.path.join(common.CKPT, "head"))
    parser.add_argument("--data", default=os.path.join(common.DATA, "vl_eval_plain"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--budget", type=int, default=63)
    parser.add_argument("--branch-k", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=896)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--schedules", nargs="+", default=["1", "1,4,9,15", "1,15", "1,3,7,15"])
    parser.add_argument("--out", default=os.path.join(common.RESULTS, "refine_decode.json"))
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    target, processor = common.load_target(device=device)
    head = common.load_head(args.head, device=device).eval()
    stop = [processor.tokenizer.eos_token_id]

    path = os.path.join(args.data, "chartqa.jsonl")
    rows = []
    for row in common.read_jsonl(path):
        img = os.path.join(args.data, row["image"])
        if os.path.exists(img):
            rows.append((img, row["prompt"]))
        if len(rows) >= args.n + 1:
            break
    print(f"{len(rows)} prompts (first is warm-up), head {args.head}", flush=True)

    results = {}
    for spec in args.schedules:
        schedule = [int(x) for x in spec.split(",")]
        assert schedule[0] == 1, "the first pass must be the decode-time l = 1"
        taus, per_round, ms = [], [], []
        for i, (img_path, prompt) in enumerate(rows):
            image = common.load_image(img_path, args.max_pixels)
            ids, pix, grid = common.encode(processor, prompt, image)
            ids = ids.to(device)
            pos = VisionPositions(pix.to(device) if pix is not None else None,
                                  grid.to(device) if grid is not None else None)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            final, rt, passes = refine_generate(
                head, target, ids, args.max_new, stop, args.budget, args.branch_k,
                pos, schedule)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if i == 0:
                continue                       # warm-up, dropped as everywhere else
            n_out = final.shape[1] - ids.shape[1]
            taus += rt
            per_round += passes
            ms.append(dt * 1000 / max(n_out, 1))
        tau = statistics.fmean(taus)
        results[spec] = {
            "schedule": schedule,
            "tau": tau,
            "rounds": len(taus),
            "draft_passes_per_round": statistics.fmean(per_round),
            "ms_per_token": statistics.fmean(ms),
        }
        r = results[spec]
        print(f"  schedule {spec:>10s}: tau {tau:.3f}  rounds {len(taus):4d}  "
              f"passes/round {r['draft_passes_per_round']:.2f}  "
              f"{r['ms_per_token']:.2f} ms/tok", flush=True)

    base = results.get("1", {}).get("tau")
    if base:
        print(f"\n  baseline (schedule 1, i.e. plain GLANCE decode): tau {base:.3f}")
        for spec, r in results.items():
            if spec != "1":
                print(f"  {spec:>10s}: {r['tau'] - base:+.3f} tau "
                      f"({100*(r['tau']/base - 1):+.1f}%), "
                      f"cost {r['draft_passes_per_round']:.2f} draft passes/round")

    with open(args.out, "w") as h:
        json.dump(results, h, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
