#!/usr/bin/env python3
"""Per-offset acceptance at the decode condition, for any head, with a paired CI.

Why this exists. Arm M's pre-registered *secondary* criterion was written as
"per-offset ratio 0.6 -> >= 0.72". Re-deriving it from the probe output in
`results/prefix_conditioning.json` shows 0.6 was never a per-offset ratio. It was
the ratio between *sampled checkpoints* of the curve (off1 -> off4 -> off8 ->
off15), which are 3, 4 and 7 offsets apart, so it cannot be a per-offset factor.
The decode-time per-offset decay actually measured by probe_prefix_conditioning.py
is

    (0.083 / 0.608) ** (1 / 14) = 0.867

and a 0.6-per-offset decay would predict off8 = 0.017 and off15 = 0.000 against
observed 0.216 and 0.083. The 0.72 bar was therefore already cleared before any
training happened and carries no decision weight. Rather than silently retune a
threshold, this script replaces the statistic with one that is well defined and
computed *identically* on both heads:

    g = (acc[15] / acc[1]) ** (1 / 14)

the geometric-mean per-offset retention. The decision still rests on the primary
criterion (tau @ budget 63) plus the falsification clause; `g` only answers "did
the curve move", and it answers it with an interval.

What "decode condition" means here. l=1 block: slot 0 is the committed root, slots
1..15 are MASK, exactly as at inference. If the head ships a `markov_head.pt`, the
block is steered semi-autoregressively (`markov.steer_block_greedy`), so the
reported per-offset accuracies are the *conditioned* marginals the tree is built
from -- not the unconditioned ones the baseline produces. Offsets are scored by
argmax against the target's greedy continuation, which is the token verification
would have to accept.

Per-anchor outcomes are persisted so two heads can be compared with a paired
cluster bootstrap over prompts (the conservative unit: offsets within a prompt are
strongly correlated).

    python measure_offset_curve.py --head $GLANCE_ROOT/checkpoints/head \
        --tag baseline --gpu 2 --out $GLANCE_ROOT/results/offset_curve_baseline.json
    python measure_offset_curve.py --head $GLANCE_ROOT/checkpoints/head_markov \
        --tag armM --gpu 2 --out $GLANCE_ROOT/results/offset_curve_armM.json
    python measure_offset_curve.py --compare \
        $GLANCE_ROOT/results/offset_curve_baseline.json \
        $GLANCE_ROOT/results/offset_curve_armM.json
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
import random
import sys

import torch

from shared import common
from shared import markov as markov_mod

sys.path.insert(0, os.path.join(common.WORKSPACE, "GLANCE_repo"))

import glance
from glance.vl import VisionPositions


@torch.no_grad()
def decode_block_logits(head, target, seq_ids, fused, anchor):
    """The l=1 block exactly as decode builds it: [root, MASK...] -> offsets 1..15."""
    block = head.block_size
    ids = torch.full((1, block), head.mask_token_id, dtype=torch.long,
                     device=fused.device)
    ids[0, 0] = seq_ids[anchor]
    ctx = fused[:, :anchor]
    position_ids = torch.arange(anchor + block, device=fused.device).unsqueeze(0)
    hidden = head(target_hidden=ctx,
                  noise_embedding=target.get_input_embeddings()(ids),
                  position_ids=position_ids)
    return target.lm_head(hidden[:, 1 - block:, :])[0]      # (block-1, vocab)


def run(args):
    device = f"cuda:{args.gpu}"
    target, processor = common.load_target(device=device)
    head = common.load_head(args.head, device=device).eval()
    mk = markov_mod.load_markov(args.head, device)
    stop = [processor.tokenizer.eos_token_id]
    block = head.block_size

    rows = []
    for row in common.read_jsonl(os.path.join(args.data, "chartqa.jsonl")):
        img = os.path.join(args.data, row["image"])
        if os.path.exists(img):
            rows.append((img, row["prompt"]))
        if len(rows) >= args.n:
            break
    print(f"{len(rows)} prompts, block {block}, stride {args.stride}, "
          f"markov={'yes' if mk is not None else 'no'}", flush=True)

    per_anchor = []           # [prompt_idx, anchor, [0/1] * (block-1)]
    spine_matches = [0, 0]    # Markov spine vs unconditioned argmax, diagnostic

    for i, (img_path, prompt) in enumerate(rows):
        image = common.load_image(img_path, args.max_pixels)
        ids, pix, grid = common.encode(processor, prompt, image)
        ids = ids.to(device)
        pos = VisionPositions(pix.to(device) if pix is not None else None,
                              grid.to(device) if grid is not None else None)
        final = glance.greedy_generate(target, ids, args.gen, stop, positions=pos)
        cont = final[0, ids.shape[1]:]
        full = torch.cat([ids[0], cont])
        prompt_len = ids.shape[1]
        if full.shape[0] < prompt_len + block + 2:
            continue
        fused = common.target_context(target, full[None])

        last_anchor = full.shape[0] - block - 1
        for anchor in range(prompt_len, last_anchor + 1, args.stride):
            base = decode_block_logits(head, target, full, fused, anchor)
            if mk is not None:
                steered, spine = markov_mod.steer_block_greedy(
                    base, full[anchor].reshape(1), mk, scale=args.markov_scale)
                pred = steered.argmax(dim=-1)
                spine_matches[0] += int((spine == base.argmax(dim=-1)).sum())
                spine_matches[1] += spine.numel()
            else:
                pred = base.argmax(dim=-1)
            hits = [int(pred[k - 1]) == int(full[anchor + k])
                    for k in range(1, block)]
            per_anchor.append([i, anchor, hits])
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(rows)} prompts", flush=True)

    curve = []
    for k in range(1, block):
        c = sum(a[2][k - 1] for a in per_anchor)
        curve.append([c, len(per_anchor)])

    acc1 = curve[0][0] / curve[0][1]
    acc15 = curve[block - 2][0] / curve[block - 2][1]
    g = (acc15 / acc1) ** (1.0 / (block - 2))

    print(f"\n=== {args.tag}: per-offset top-1 at the decode condition ===")
    for k, (c, t) in enumerate(curve, start=1):
        print(f"  off{k:>2}  {c/t:.3f}  ({c}/{t})")
    print(f"\n  acc[1] {acc1:.3f} -> acc[{block-1}] {acc15:.3f}"
          f"   geometric per-offset retention g = {g:.4f}")
    if spine_matches[1]:
        print(f"  Markov spine == unconditioned argmax: "
              f"{spine_matches[0]/spine_matches[1]:.3f} "
              f"({spine_matches[0]}/{spine_matches[1]})")

    payload = {"tag": args.tag, "head": args.head, "n_prompts": len(rows),
               "block": block, "stride": args.stride,
               "markov": mk is not None, "gen": args.gen,
               "max_pixels": args.max_pixels, "curve": curve,
               "g": g, "per_anchor": per_anchor}
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    print(f"\nwrote {args.out}")


def compare(paths):
    """Paired cluster bootstrap over prompts for acc[k] and for g."""
    a = json.load(open(paths[0]))
    b = json.load(open(paths[1]))
    block = a["block"]

    def keyed(d):
        return {(p, anc): hits for p, anc, hits in d["per_anchor"]}

    ka, kb = keyed(a), keyed(b)
    shared = sorted(set(ka) & set(kb))
    if not shared:
        raise SystemExit("no shared (prompt, anchor) keys -- runs are not comparable")
    prompts = sorted({p for p, _ in shared})
    print(f"{a['tag']} vs {b['tag']}: {len(shared)} anchors over {len(prompts)} prompts")

    def stats(keyed, keys):
        # `keyed` must be a parameter, not captured from the enclosing scope:
        # an earlier version closed over the first head's dict and silently
        # reported the first curve twice, which made every comparison read as
        # "+0.0000, curve did not move".
        c = [0] * (block - 1)
        for k_ in keys:
            for j, h in enumerate(keyed[k_]):
                c[j] += h
        n = len(keys)
        acc = [x / n for x in c]
        return acc, (acc[block - 2] / acc[0]) ** (1.0 / (block - 2))

    acc_a, g_a = stats(ka, shared)
    acc_b, g_b = stats(kb, shared)
    print(f"\n  {'off':>4} {a['tag']:>10} {b['tag']:>10} {'delta':>8}  {'95% CI':>18}")
    for k in range(block - 1):
        print(f"  {k+1:>4} {acc_a[k]:>10.3f} {acc_b[k]:>10.3f}"
              f" {acc_b[k]-acc_a[k]:>+8.3f}")
    print(f"\n  g: {a['tag']} {g_a:.4f} -> {b['tag']} {g_b:.4f}  ({g_b-g_a:+.4f})")

    rng = random.Random(0)
    B = 2000
    boots = []
    by_prompt = {p: [k_ for k_ in shared if k_[0] == p] for p in prompts}
    for _ in range(B):
        pick = [rng.choice(prompts) for _ in prompts]
        keys = [k_ for p in pick for k_ in by_prompt[p]]
        try:
            _, ga = stats(ka, keys)
            _, gb = stats(kb, keys)
        except ZeroDivisionError:
            continue
        boots.append(gb - ga)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    print(f"  paired bootstrap 95% CI on delta g: [{lo:+.4f}, {hi:+.4f}]  ({B} reps)")
    moved = not (lo <= 0.0 <= hi)
    print(f"  curve moved: {'YES' if moved else 'NO'} "
          f"(interval {'excludes' if moved else 'contains'} zero)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=os.path.join(common.CKPT, "head"))
    ap.add_argument("--data", default=os.path.join(common.DATA, "vl_eval_plain"))
    ap.add_argument("--tag", default="head")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--max-pixels", type=int, default=896)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--markov-scale", type=float, default=1.0,
                    help="multiplier on the Markov bias (1.0 = as trained)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
        return
    if args.out is None:
        args.out = os.path.join(common.RESULTS, f"offset_curve_{args.tag}.json")
    run(args)


if __name__ == "__main__":
    main()
