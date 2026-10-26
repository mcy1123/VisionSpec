#!/usr/bin/env python3
"""WHY: the B-reference alignment experiment -- can the offline metric explain the real commit?

The phaseC5 audit (docs/reports/2026-10-12-gain-stuck.md sections 3, 8, 11, 12) left
three blockers in front of every mechanistic reading of the phaseC4 result:

  * the stored old trajectories diverge from the current fp32 AR deployment in
    69/70 rows, so the offline diagnostic's anchors and prefixes are NOT the
    prefixes deployment actually visits (section 3);
  * on g2 the tree's real first-token acceptance rises (+0.0072) while the
    offline `rank<=8` coverage is exactly flat (+0.0000), and restricting to the
    common prefix does not remove it (sections 11.4, 12.4). Section 12.2 also
    corrected a sign error: the offline `rank<=8` delta is +0.0014 while the
    tree's measured P(commit>=1) delta is -0.0020, so "offline is calibrated to
    deployment" holds only for the b1k1 single-candidate probe;
  * the b1k1 "chain" is a single-candidate probe, not a real block-16 chain
    (section 5).

This file runs the ONE experiment that was left as the next step (sections 8,
12.4): a small, strictly-scoped, READ-ONLY B-reference alignment diagnostic.

WHAT IS DIFFERENT FROM EVERY EARLIER DIAGNOSTIC. Earlier diagnostics built their
anchors from the stored trajectory or from a C0-policy rollout detached from a
real decode, then scored the head offline against a re-derived teacher label.
Here the anchors ARE the rounds of the current fp32 deployment decode:

  1. `glance.generate` (fp32, C0, tree b63k8) is run unmodified and its per-round
     `(pos, accepted, path_tokens)` are recorded -- that is the reference;
  2. the same decode is re-run by an instrumented copy of the SAME round loop
     (`instrumented_decode`) which, at each of the first `--max-anchors` rounds,
     freezes the exact decode state: the target's KV cache covering `[0, start)`,
     the drafter's accumulated context features covering `[0, start)`, the root
     token and the block mask. The copy is validated against the unmodified
     `generate` token-for-token and round-for-round on every video, so drift in
     the copy is a hard failure and not a silent one;
  3. at each frozen anchor the THREE phaseC4 students (C0/C1/C2) are replayed on
     the IDENTICAL frozen state -- one draft pass each, one candidate tree each,
     one fp32 target verify pass each. The only thing that varies between arms is
     the head, so this is a genuinely paired same-context comparison;
  4. the same pass also replays the head's top-1 spine alone (budget 15, branch 1
     == the repo's real block-16 chain on the same context), so the tree/chain
     question is answered per anchor instead of per differently-diverged
     trajectory.

WHY A COPY OF THE ROUND LOOP AND NOT A MONKEYPATCH. `generate` is frozen and must
not be modified (AGENTS.md, task constraints), and it does not expose the target
KV cache, the accumulated `fused` features or the packed target logits at a round
boundary. Wrapping internals would have to intercept a closure-local `out_ids`,
which is strictly more fragile than a copy. The copy is therefore guarded by an
equality gate against the unmodified `generate`, and by a second equality gate
that replays C0's own round from the frozen state and compares the committed
tokens, commit length and the actual draft top-8 with what the real decode did.

THE SINGLE QUESTION. Can the offline metric explain the actual commit at the same
anchor? The offline and the actual quantity have to be defined on the same object
or the comparison is vacuous. The head's per-offset marginals are conditioned on
the context only (block diffusion over independent offsets), so for arm H at an
anchor we record, per offset k = 1..15, the rank of the target's own AR token at
absolute position `start + k` inside H's offset-k distribution. From that:

  * `rank1_first_rej` -- first k whose rank > 1. This is the "chain" prediction
    under independent marginals;
  * `top8_first_rej`  -- first k whose rank > 8. This is the candidate-coverage
    prediction, i.e. the widest statement the head's top-8 can support;
  * `rank1_agree_len`, `top8_agree_len` -- the leading agreement lengths.

and from the real fp32 tree verify:

  * `tree_first_rej` / `commit_len` -- the actual first rejected offset and the
    number of tokens really committed that round (accepted draft tokens plus the
    bonus token);
  * `chain_first_rej` / `chain_accepted` -- the same for the top-1 spine, i.e. the
    real block-16 chain on the identical context;
  * `rej_cause` -- whether the token the target would have taken at the rejected
    offset was (a) not in the head's top-8 at all (candidate coverage), (b) in the
    top-8 but not present in the budget tree (selection / budget pruning), or
    (c) present in the tree (which would contradict the walk and is therefore
    reported explicitly).

TWO LEVELS OF HONESTY. (1) Whether a round is a same-state comparison across arms
is decided by TOKEN equality of the committed prefix, not by absolute position:
after the first round in which an arm commits a different token, that arm's own
decode would be standing at a different prefix, so later anchors are labelled
`diverged` and are NOT pooled with the common-prefix ones. (2) fp32 reductions are
not associative, and the deployment's cached drafter pass is not the same
reduction order as a one-shot full-prefix pass. The head applies no causal mask
(`is_causal=False`, no sliding window), so every query sees the same key set and
the per-row keys/values are identical either way -- but the attention reduction
can still differ in the last bits. That is measured, not assumed: the C0 replay
gate counts anchors where the replayed commit differs AND records the exact fp32
bit-sum of the drafted logits on both sides, so a bit-identical pass is
distinguishable from a near-tie.

This module never builds an optimizer, never calls backward, never writes a
checkpoint and never touches the frozen test-set candidates.
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
import hashlib
import json
import os
import statistics as stats
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common  # noqa: E402
from glance.decode import _pick, gather_cache, greedy_generate  # noqa: E402
from glance.head import context_feature  # noqa: E402
from glance.tree import build_budget_tree, children_of, tree_mask  # noqa: E402
from glance.vl import VisionPositions  # noqa: E402

#: The three phaseC4 students. C0 is the aligned SFT start checkpoint, C1 the
#: continue-SFT control (10 updates), C2 the paper-faithful full-vocab online OPD
#: arm (10 updates). All three start from `head_video439_aligned/step1000`.
STUDENTS = {
    "C0": "checkpoints/head_video439_aligned/step1000",
    "C1": "checkpoints/phaseC4_C1/step10",
    "C2": "checkpoints/phaseC4_C2/step10",
}
ARMS = ("C0", "C1", "C2")

#: The deployment tree of every phaseC4/phaseC5 measurement, held fixed so the
#: anchors are the real tree's anchors.
BUDGET = 63
BRANCH = 8

#: `glance.generate --budget 15 --branch 1` builds the single-column spine, i.e.
#: the real block-16 chain through the same code path (15 draft offsets + root).
CHAIN_BUDGET = 15
CHAIN_BRANCH = 1

#: Anchor selection, fixed in advance: at most this many real C0 decode rounds per
#: video, taken as the FIRST rounds of the decode. No score-based choice, which is
#: what keeps the selection free of the outcome.
MAX_ANCHORS = 8

#: Pre-registration constants. The salt is part of the pre-registration: changing
#: it changes the selected videos, so it is written to disk before any scoring.
PREREG_SALT = "phaseC5-bref-align-2026-10-12"
PREREG_SEED = 20261012
#: Three input-length terciles, two videos each.
N_STRATA = 3
PER_STRATUM = 2

#: The eval convention used everywhere in this project: `--warmup 1` drops the
#: first row of the row list. Row 0 of video_fresh30 is that warmup row, so it is
#: excluded before any selection happens (still a valid video, just not part of the
#: n=29 validation aggregate).
WARMUP_ROW = 0

#: Paired bootstrap, resampling unit = video (a video's rounds are not independent
#: draws). Same convention as `diag_gain_stuck.py`.
BOOT_N = 10_000
BOOT_SEED = 20261012


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _config(a) -> dict:
    """The argparse namespace minus the subcommand callable (not JSON)."""
    return {k: v for k, v in vars(a).items() if not callable(v)}


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _jsonl_write(handle, obj) -> None:
    handle.write(json.dumps(obj) + "\n")
    handle.flush()


def head_path(arm: str) -> str:
    return os.path.join(common.ROOT, STUDENTS[arm])


def load_rows(pool_path: str):
    import pandas as pd

    return pd.read_parquet(pool_path).reset_index(drop=True)


def encode_row(proc, row, device):
    """Exactly the encoding `eval_video_opd.py` uses, so the tokens match."""
    text = proc.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": row["question"]}]}],
        add_generation_prompt=True, tokenize=False)
    ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
               fps=float(row["fps"]))
    return (ins["input_ids"].to(device),
            ins["pixel_values_videos"].to(device),
            ins["video_grid_thw"].to(device))


def new_cache():
    from transformers import DynamicCache

    return DynamicCache()


def snapshot_cache(cache):
    """Freeze a DynamicCache to CPU in whichever layout transformers is using."""
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        entries = [(layer.keys.detach().to("cpu", copy=True),
                    layer.values.detach().to("cpu", copy=True))
                   for layer in cache.layers]
        return {"layout": "layers", "entries": entries,
                "len": int(cache.layers[0].keys.shape[-2])}
    entries = [(k.detach().to("cpu", copy=True), v.detach().to("cpu", copy=True))
               for k, v in zip(cache.key_cache, cache.value_cache)]
    return {"layout": "legacy", "entries": entries,
            "len": int(entries[0][0].shape[-2])}


def restore_cache(snap, device):
    """Rebuild the frozen cache. One clone per arm: the verify pass appends to it."""
    from transformers.cache_utils import DynamicLayer

    cache = new_cache()
    if snap["layout"] == "layers":
        for k, v in snap["entries"]:
            layer = DynamicLayer()
            layer.keys = k.to(device, copy=True)
            layer.values = v.to(device, copy=True)
            layer.is_initialized = True
            cache.layers.append(layer)
    else:
        for i, (k, v) in enumerate(snap["entries"]):
            cache.key_cache[i] = k.to(device, copy=True)
            cache.value_cache[i] = v.to(device, copy=True)
    return cache


def _walk(nodes, packed_logits, temperature=0.0):
    """Copy of `glance.decode.generate` step 4: walk the tree along the target's
    own greedy choices. Returns (path node indices, bonus token)."""
    children = children_of(nodes)
    path = []
    cursor = -1
    while True:
        token = _pick(packed_logits[0 if cursor == -1 else cursor + 1], temperature)
        hit = next((c for c in children.get(cursor, []) if nodes[c].token == token), None)
        if hit is None:
            return path, token
        path.append(hit)
        cursor = hit


def _packed_stats(packed_logits):
    """fp32 target decision detail per packed row: argmax id, top1-top2 margin."""
    top2 = torch.topk(packed_logits.float(), 2, dim=-1)
    return (top2.indices[:, 0].tolist(),
            [float(x) for x in (top2.values[:, 0] - top2.values[:, 1]).tolist()])


def _logits_bitsum(logits) -> int:
    """Exact fp32 bit pattern sum; an equality test for two drafts, not a hash."""
    return int(logits.detach().float().view(torch.int32).to(torch.int64).sum().item())


# --------------------------------------------------------------------------
# pre-registration
# --------------------------------------------------------------------------

def cmd_prereg(a):
    """Fix the six videos (and the two gate videos) BEFORE any scoring.

    Score-free by construction: the only row properties used are the video id and
    the tokenised input length. Nothing from any model output is consulted.
    """
    df = load_rows(a.pool)
    ordered = df.reset_index(drop=True)
    if a.warmup_row is not None and a.warmup_row >= 0:
        ordered = ordered.drop(index=a.warmup_row).reset_index(drop=True)

    order = np.argsort(ordered["input_ids_len"].to_numpy(), kind="stable")
    ranked = ordered.iloc[order].reset_index(drop=True)
    n = len(ranked)
    bounds = [(i * n) // N_STRATA for i in range(N_STRATA + 1)]

    strata = []
    for s in range(N_STRATA):
        block = ranked.iloc[bounds[s]:bounds[s + 1]].reset_index(drop=True)
        hashed = sorted(range(len(block)),
                        key=lambda i: hashlib.sha256(
                            f"{PREREG_SALT}:{block.loc[i, 'video_name']}".encode()
                        ).hexdigest())
        picked = [str(block.loc[i, "video_name"]) for i in hashed[:PER_STRATUM]]
        strata.append({
            "stratum": s,
            "len_range": [int(block["input_ids_len"].min()),
                          int(block["input_ids_len"].max())],
            "n_in_stratum": int(len(block)),
            "members_sorted_by_len": [str(x) for x in block["video_name"]],
            "members_len": [int(x) for x in block["input_ids_len"]],
            "hash_order": [str(block.loc[i, "video_name"]) for i in hashed],
            "picked": picked,
        })

    videos = [v for s in strata for v in s["picked"]]
    # The gate takes the first pick of the shortest and of the longest stratum, so
    # it exercises both ends of the input-length range before the other four run.
    gate = [strata[0]["picked"][0], strata[N_STRATA - 1]["picked"][0]]

    out = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": ("B-reference alignment: score C0/C1/C2 on the frozen state of "
                    "the real fp32 C0 tree decode's own rounds. Written BEFORE any "
                    "model was run."),
        "pool": a.pool,
        "warmup_row_dropped": a.warmup_row,
        "n_candidates": int(n),
        "salt": PREREG_SALT,
        "seed": PREREG_SEED,
        "rules": {
            "stratification": (f"{N_STRATA} terciles of the input-length-sorted "
                               f"{n} non-warmup rows; {PER_STRATUM} picks per tercile"),
            "within_stratum": "sha256(f'{salt}:{video_name}') ascending",
            "gate": "the first pick of the shortest and of the longest stratum",
            "anchors": (f"the first {MAX_ANCHORS} rounds of the real fp32 C0 tree "
                        f"decode (b{BUDGET}k{BRANCH}); no score-based choice"),
            "gate_criterion": ("C0 replay must reproduce the real C0 commit length "
                               "AND committed path at EVERY gate anchor"),
            "gate_action_on_failure": ("STOP; report the mismatch as the finding and "
                                       "do not compare C1/C2 on an unreproducible "
                                       "reference"),
        },
        "strata": strata,
        "videos": videos,
        "videos_len": [int(ranked[ranked.video_name == v]["input_ids_len"].iloc[0])
                       for v in videos],
        "gate_videos": gate,
        "rest_videos": [v for v in videos if v not in gate],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"pre-registration -> {a.out}")
    print(f"  gate : {gate}")
    print(f"  rest : {out['rest_videos']}")
    for s in strata:
        print(f"  stratum {s['stratum']} len {s['len_range']} "
              f"({s['n_in_stratum']} rows) picked {s['picked']}")


# --------------------------------------------------------------------------
# the instrumented decode (a guarded copy of glance.decode.generate's round loop)
# --------------------------------------------------------------------------

@torch.inference_mode()
def instrumented_decode(head, target, input_ids, max_new_tokens, stop_token_ids,
                        positions, anchor_rounds, budget, branch_k):
    """`glance.decode.generate`'s loop, plus a frozen state per selected anchor.

    Only the recording is new. Every model call, position tensor, mask and commit
    step is the original; `cmd_run` asserts the resulting token stream and
    per-round accepted lengths against the unmodified `generate`, so a mistake in
    this copy is caught before any of its anchors are used.

    The frozen state is taken BEFORE the round's verify pass, because that is the
    state a replay needs: target KV over `[0, start)`, drafter context features
    over `[0, start)`, the block (root + masks) and the root position.
    """
    device = input_ids.device
    block = head.block_size
    mask_id = head.mask_token_id
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    dtype = next(target.parameters()).dtype
    embed = target.get_input_embeddings()

    out_ids = torch.full((1, max_len + 2 * block + 4), mask_id,
                         dtype=torch.long, device=device)
    stream = torch.arange(out_ids.shape[1], device=device).unsqueeze(0)
    kv_target = new_cache()
    kv_head = new_cache()

    positions.bind(target, input_ids)
    out = target(input_ids, past_key_values=kv_target, use_cache=True,
                 logits_to_keep=1, output_hidden_states=True,
                 **positions.prefill_inputs(input_ids, stream[:, :n_in]))
    out_ids[:, :n_in] = input_ids
    out_ids[:, n_in] = _pick(out.logits[0, -1], 0.0)
    fused = context_feature(out.hidden_states, head.target_layer_ids)
    # Concatenated in order these chunks are exactly the rows the cached head pass
    # saw, in order, covering [0, start) at every later round.
    fused_chunks = [fused.detach().clone()]

    start = n_in
    rounds = []
    anchors = []
    while start < max_len:
        block_ids = out_ids[:, start:start + block].clone()
        block_ids[:, 1:] = mask_id
        hidden = head(target_hidden=fused,
                      noise_embedding=embed(block_ids),
                      position_ids=stream[:, kv_head.get_seq_length():start + block],
                      past_key_values=kv_head, use_cache=True, is_causal=False)
        draft_logits = target.lm_head(hidden[:, 1 - block:, :])[0]
        kv_head.crop(start)

        top = torch.topk(F.log_softmax(draft_logits.float(), dim=-1), branch_k, dim=-1)
        top_ids, top_logp = top.indices.cpu(), top.values.cpu()
        nodes = build_budget_tree(top_ids, top_logp, budget, branch_k)

        packed = torch.tensor(
            [[int(out_ids[0, start])] + [n.token for n in nodes]], device=device)
        packed_pos = torch.tensor(
            [[start] + [start + n.depth for n in nodes]], device=device)
        n_prefix = kv_target.get_seq_length()
        if int(n_prefix) != int(start):
            raise RuntimeError(
                f"target KV length {int(n_prefix)} != round start {int(start)}; "
                "the frozen-anchor state assumption does not hold")

        round_index = len(rounds)
        take_anchor = round_index in anchor_rounds
        frozen = None
        if take_anchor:
            # BEFORE the verify pass: kv_target still covers [0, start).
            frozen = {
                "round": round_index,
                "start": int(start),
                "n_prefix": int(n_prefix),
                "root": int(out_ids[0, start]),
                "block_ids": block_ids.detach().to("cpu", copy=True),
                "prefix_ids": [int(x) for x in out_ids[0, :start + 1].tolist()],
                "prefix_len": int(start + 1),
                "kv": snapshot_cache(kv_target),
                "fused_all": torch.cat(fused_chunks, dim=1).detach().to("cpu", copy=True),
            }

        out = target(packed, position_ids=positions(packed_pos),
                     past_key_values=kv_target, use_cache=True,
                     attention_mask=tree_mask(n_prefix, nodes, device, dtype),
                     output_hidden_states=True)
        packed_logits = out.logits[0]

        path, bonus = _walk(nodes, packed_logits)
        accepted = len(path)

        if take_anchor:
            argmax, margin = _packed_stats(packed_logits)
            frozen["actual"] = {
                "top_ids": top_ids.tolist(),
                "top_logp": top_logp.tolist(),
                "nodes": [[n.token, n.depth, n.parent, n.rank] for n in nodes],
                "packed_pos": [int(x) for x in packed_pos[0].tolist()],
                "packed_argmax": argmax,
                "packed_margin": margin,
                "node_target_ok": [bool(argmax[0 if nodes[i].parent == -1
                                               else nodes[i].parent + 1] == nodes[i].token)
                                   for i in range(len(nodes))],
                "path_nodes": path,
                "path_tokens": [int(nodes[i].token) for i in path],
                "bonus": int(bonus),
                "accepted": int(accepted),
                "commit_len": int(accepted + 1),
                "commit_tokens": [int(nodes[i].token) for i in path] + [int(bonus)],
                "draft_logits_bitsum": _logits_bitsum(draft_logits),
                "ctx_hash": _sha(frozen["prefix_ids"],
                                 frozen["block_ids"].tolist()),
                "vispos_hash": _sha([int(x) for x in packed_pos[0].tolist()]),
                "tree_mask_hash": _sha(tree_mask(n_prefix, nodes, device, dtype)
                                       .to("cpu").to(torch.int8).numpy().tobytes().hex()),
            }
            anchors.append(frozen)

        for offset, node_index in enumerate(path):
            out_ids[0, start + 1 + offset] = nodes[node_index].token
        out_ids[0, start + accepted + 1] = bonus

        keep = [0] + [i + 1 for i in path]
        gather_cache(kv_target, n_prefix, keep)
        fused = context_feature(out.hidden_states, head.target_layer_ids)[0][keep][None]
        fused_chunks.append(fused.detach().clone())

        rounds.append({"round": round_index, "pos": int(start),
                       "accepted": int(accepted), "commit_len": int(accepted + 1),
                       "path_tokens": [int(nodes[i].token) for i in path],
                       "bonus": int(bonus),
                       "argmax_tokens": [int(t) for t in top_ids[:, 0]],
                       "n_nodes": int(len(nodes))})
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
    return final, rounds, anchors


# --------------------------------------------------------------------------
# the per-anchor replay of one arm and one topology
# --------------------------------------------------------------------------

@torch.inference_mode()
def arm_draft_logits(head, target, embed, frozen, device, block):
    """The arm's first-layer candidates: one draft pass over the frozen context.

    One-shot over the full frozen prefix instead of the deployment's chunked
    cache. `Qwen3DFlashAttention` applies no causal mask (is_causal=False, no
    sliding window), so every query row sees the same key set either way and every
    row's key/value projection is the same function of the same row; only the fp32
    attention reduction order can differ. The C0 gate measures that difference.
    """
    start = frozen["start"]
    fused_all = frozen["fused_all"].to(device)
    block_ids = frozen["block_ids"].to(device)
    stream = torch.arange(start + block, device=device).unsqueeze(0)
    hidden = head(target_hidden=fused_all,
                  noise_embedding=embed(block_ids),
                  position_ids=stream,
                  past_key_values=new_cache(), use_cache=True, is_causal=False)
    return target.lm_head(hidden[:, 1 - block:, :])[0]


@torch.inference_mode()
def verify_topology(target, frozen, positions, draft_logits, branch_k, topology,
                    device, block):
    """Build one candidate tree from `draft_logits` and run the real fp32 verify.

    `topology="tree"` is the deployment tree (budget 63 / branch 8);
    `topology="chain"` is the head's top-1 spine alone (budget 15 / branch 1),
    which is the repo's real block-16 chain. Both go through
    `build_budget_tree`, the ancestor-masked target pass and the same acceptance
    walk as deployment, so the commit is the real AR check, not a re-derived label.
    """
    start = frozen["start"]
    # The M-RoPE delta is a property of the prompt's video grid, not of the round:
    # `glance.decode.generate` binds the adapter once per prompt and every verify
    # pass reuses that delta. A fresh, unbound adapter silently defaults to delta
    # 0, which is a real bug that was caught here by the replay gate and not by any
    # token comparison -- hence the explicit guard.
    if hasattr(positions, "_prefill_positions") and positions._prefill_positions is None:
        raise RuntimeError("positions adapter was not bound to the prompt; its "
                           "M-RoPE delta would default to 0")
    top = torch.topk(F.log_softmax(draft_logits.float(), dim=-1), branch_k, dim=-1)
    top_ids_all, top_logp_all = top.indices.cpu(), top.values.cpu()
    if topology == "chain":
        top_ids, top_logp = top_ids_all[:, :1], top_logp_all[:, :1]
        eff_budget, eff_branch = CHAIN_BUDGET, CHAIN_BRANCH
    else:
        top_ids, top_logp = top_ids_all, top_logp_all
        eff_budget, eff_branch = BUDGET, BRANCH

    nodes = build_budget_tree(top_ids, top_logp, eff_budget, eff_branch)
    root = frozen["root"]
    packed = torch.tensor([[root] + [n.token for n in nodes]], device=device)
    packed_pos = torch.tensor([[start] + [start + n.depth for n in nodes]],
                              device=device)
    dtype = next(target.parameters()).dtype

    kv = restore_cache(frozen["kv"], device)
    out = target(packed, position_ids=positions(packed_pos), past_key_values=kv,
                 use_cache=True,
                 attention_mask=tree_mask(start, nodes, device, dtype),
                 output_hidden_states=True)
    packed_logits = out.logits[0]

    path, bonus = _walk(nodes, packed_logits)
    accepted = len(path)
    argmax, margin = _packed_stats(packed_logits)

    # Rejection cause: what the target itself would have taken at the offset right
    # after the last committed node, and whether the head had it at all.
    cursor = path[-1] if path else -1
    rej_token = int(argmax[0 if cursor == -1 else cursor + 1])
    rej_children = children_of(nodes).get(cursor, [])
    rej_in_tree = any(nodes[c].token == rej_token for c in rej_children)
    rej_in_topk = (bool((top_ids_all[accepted] == rej_token).any().item())
                   if accepted < top_ids_all.shape[0] else None)
    if accepted >= block - 1:
        cause = "none"
    elif not rej_in_topk:
        cause = "not_in_topk"
    elif not rej_in_tree:
        cause = "in_topk_not_in_tree"
    else:
        cause = "in_tree"

    return {
        "topology": topology,
        "accepted": int(accepted),
        "commit_len": int(accepted + 1),
        "path_tokens": [int(nodes[i].token) for i in path],
        "commit_tokens": [int(nodes[i].token) for i in path] + [int(bonus)],
        "bonus": int(bonus),
        "first_rej_offset": (int(accepted) + 1) if accepted < block - 1 else None,
        "all_accepted_in_block": bool(accepted == block - 1),
        "n_nodes": int(len(nodes)),
        "top_ids": top_ids_all.tolist(),
        "top_logp": top_logp_all.tolist(),
        "nodes": [[n.token, n.depth, n.parent, n.rank] for n in nodes],
        "packed_pos": [int(x) for x in packed_pos[0].tolist()],
        "packed_argmax": argmax,
        "packed_margin": margin,
        "node_target_ok": [bool(argmax[0 if nodes[i].parent == -1
                                       else nodes[i].parent + 1] == nodes[i].token)
                           for i in range(len(nodes))],
        "root_row_argmax": int(argmax[0]),
        "rej_token": rej_token,
        "rej_in_topk": rej_in_topk,
        "rej_in_tree": bool(rej_in_tree),
        "rej_cause": cause,
    }


def offline_metrics(draft_logits, ar_tokens, block, top_ids_all):
    """Offline rank of the target's own AR token inside the head's distribution.

    Two predictions are read off the same marginals:
      * `rank1_first_rej` -- first offset whose rank > 1 (the chain prediction);
      * `top8_first_rej`  -- first offset whose rank > 8 (the coverage prediction,
        which is the widest statement the head's own top-8 can support).
    `top8_agree_len` / `rank1_agree_len` are the matching leading agreement
    lengths, so the offline numbers and the real tree/chain commits are directly
    comparable: the real commit cannot exceed `top8_agree_len + 1`.

    Conditioning caveat, stated because it is a real limitation: the head's
    offset-k marginal is conditioned on the context only, while the target's AR
    token at position `start + k` is conditioned on the true tokens before it. For
    k > 1 the two are not the same conditional, which is exactly why the
    independent-offset prediction and the chain (self-conditioned) replay are BOTH
    reported instead of one being called "the" offline metric.
    """
    n = min(block - 1, len(ar_tokens), draft_logits.shape[0])
    logp = F.log_softmax(draft_logits.float(), dim=-1)
    ranks = []
    for k in range(n):
        token = int(ar_tokens[k])
        ranks.append(int(1 + (logp[k] > logp[k, token]).sum().item()))
    rank1_first = next((k + 1 for k in range(n) if ranks[k] > 1), None)
    top8_first = next((k + 1 for k in range(n) if ranks[k] > 8), None)
    top8_agree = next((k for k in range(n) if ranks[k] > 8), n)
    rank1_agree = next((k for k in range(n) if ranks[k] > 1), n)
    return {
        "n_eval_offsets": int(n),
        "ranks": ranks,
        "rank1_first_rej": rank1_first,
        "top8_first_rej": top8_first,
        "rank1_agree_len": int(rank1_agree),
        "top8_agree_len": int(top8_agree),
        "rank_at_1": ranks[0] if ranks else None,
        "top8_at_1": bool(ranks and ranks[0] <= 8),
        "top1_at_1": bool(ranks and ranks[0] == 1),
    }


def _ar_window(ar_ids, start, n_in, block):
    """The target's own AR tokens at absolute positions start+1 .. start+block-1.

    This is the window the offline ranks are computed against, stored per anchor
    so the metric can be re-derived without re-running a model. `ar_ids` is the
    whole continuation after the prompt, so absolute position p is
    `ar_ids[p - n_in]`.
    """
    lo = start + 1 - n_in
    return [int(x) for x in ar_ids[lo:lo + block - 1]]


def _anchor_record(video, arm, frozen, actual, ar_ids, n_in, tree, chain, off,
                    draft_logits, extra_ctx=None):
    rec = {
        "video": video, "arm": arm, "round": frozen["round"],
        "start": frozen["start"], "prefix_len": frozen["prefix_len"],
        "n_in": n_in, "root": frozen["root"],
        "ctx_hash": extra_ctx or _sha(frozen["prefix_ids"],
                                      frozen["block_ids"].tolist()),
        "vispos_hash": _sha([int(x) for x in tree["packed_pos"]]),
        "tree": tree, "chain": chain, "offline": off,
        # Absolute AR token at every offset the offline metric used. Earlier
        # versions stored the whole continuation under this name; the window is
        # what the ranks are actually computed against, and `ar_first_pos`
        # pins it to absolute positions.
        "ar_window": _ar_window(ar_ids, frozen["start"], n_in, 16),
        "ar_first_pos": int(frozen["start"]) + 1,
        "ar_continuation_len": len(ar_ids),
        "ar_continuation_sha": _sha([int(x) for x in ar_ids]),
        "head_logits_bitsum": _logits_bitsum(draft_logits),
    }
    if actual is not None:
        rec["actual"] = actual
    return rec


def cmd_run(a):
    """Capture the real deployment's rounds, then replay C0/C1/C2 per anchor.

    Stage A loads the target and C0 and, per video: runs the unmodified
    `glance.generate` (the reference), the AR oracle (the lossless gate) and the
    instrumented decode (the frozen anchors), then replays C0's own rounds as the
    first half of the replay gate. Stages B/C swap in C1 and C2 and replay the same
    frozen anchors. All frozen states stay in CPU RAM, so only one head is resident
    at a time (target + one fp32 head ~= 37 GiB, the envelope this project measured
    for fp32 evaluation).
    """
    from transformers import AutoProcessor

    import glance

    device = f"cuda:{a.gpu}"
    df = load_rows(a.pool)
    prereg = json.load(open(a.prereg))
    wanted = a.videos or prereg["videos"]
    outdir = a.out
    os.makedirs(outdir, exist_ok=True)
    tag = a.tag
    anchors_path = os.path.join(outdir, f"anchors_{tag}.jsonl")
    stage_a_path = os.path.join(outdir, f"stage_a_{tag}.json")
    status_path = os.path.join(outdir, f"status_{tag}.json")

    target, _ = common.load_target(device=device, dtype=torch.float32)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    eos = proc.tokenizer.eos_token_id
    embed = target.get_input_embeddings()

    frozen_all = {}
    stage_a = {}
    rows = {str(r["video_name"]): r for _, r in df.iterrows()}

    print(f"=== stage A: C0 capture on {len(wanted)} videos b{BUDGET}k{BRANCH} fp32 ===",
          flush=True)
    head = common.load_head(head_path("C0"), device=device, dtype=torch.float32).eval()

    anchors_handle = open(anchors_path, "w")
    for video in wanted:
        row = rows[video]
        t0 = time.perf_counter()
        ids, pv, vg = encode_row(proc, row, device)
        n_in = int(ids.shape[1])
        make_pos = lambda: VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)

        ref_rounds = []
        ref_out, ref_stats = glance.generate(
            head, target, ids, a.max_new, [eos], budget=BUDGET, branch_k=BRANCH,
            temperature=0.0, positions=make_pos(), return_stats=True,
            timing="decode_only", on_round=ref_rounds.append)
        ar_out, _ = greedy_generate(target, ids, a.max_new, [eos],
                                    positions=make_pos(), return_stats=True)

        ref_ids = [int(x) for x in ref_out[0, n_in:].tolist()]
        ar_ids = [int(x) for x in ar_out[0, n_in:].tolist()]
        first_bad = next((j for j in range(min(len(ref_ids), len(ar_ids)))
                          if ref_ids[j] != ar_ids[j]), None)
        lossless = bool(first_bad is None and len(ref_ids) == len(ar_ids))

        anchor_rounds = set(range(min(MAX_ANCHORS, len(ref_rounds))))
        inst_out, inst_rounds, anchors = instrumented_decode(
            head, target, ids, a.max_new, [eos], make_pos(), anchor_rounds,
            BUDGET, BRANCH)
        inst_ids = [int(x) for x in inst_out[0, n_in:].tolist()]

        # The replay adapter must be bound to THIS prompt, because the verify
        # positions depend on the video grid's M-RoPE delta (see verify_topology).
        replay_pos = make_pos()
        replay_pos.bind(target, ids)
        rope_delta = int(replay_pos._delta)

        ref_acc = [int(r["accepted"]) for r in ref_rounds]
        inst_acc = [int(r["accepted"]) for r in inst_rounds]
        copy_exact = {
            "tokens_exact": bool(inst_ids == ref_ids),
            "n_rounds_ref": len(ref_rounds), "n_rounds_inst": len(inst_rounds),
            "accepted_exact": bool(len(ref_acc) == len(inst_acc)
                                   and all(x == y for x, y in zip(ref_acc, inst_acc))),
            "first_token_mismatch": next((j for j in range(min(len(ref_ids), len(inst_ids)))
                                          if ref_ids[j] != inst_ids[j]), None),
            "first_round_accepted_mismatch": next(
                (j for j in range(min(len(ref_acc), len(inst_acc)))
                 if ref_acc[j] != inst_acc[j]), None),
            "ref_rounds": ref_rounds, "inst_rounds": inst_rounds,
        }

        stage_a[video] = {
            "video": video, "L": n_in, "gen_tokens": len(ref_ids),
            "rounds": len(ref_rounds), "tau": float(ref_stats.tau),
            "ar_gen_tokens": len(ar_ids), "lossless": lossless,
            "first_mismatch": first_bad, "copy_exact": copy_exact,
            "rope_delta": rope_delta,
            "ar_ids": ar_ids, "ref_ids": ref_ids,
            "anchor_rounds": sorted(anchor_rounds), "anchors": [],
        }

        for frozen in anchors:
            actual = frozen["actual"]
            logits = arm_draft_logits(head, target, embed, frozen, device,
                                      head.block_size)
            tree = verify_topology(target, frozen, replay_pos, logits, BRANCH,
                                   "tree", device, head.block_size)
            chain = verify_topology(target, frozen, replay_pos, logits, BRANCH,
                                    "chain", device, head.block_size)
            ar_tokens = ar_ids[frozen["start"] + 1 - n_in:
                               frozen["start"] + 1 - n_in + head.block_size - 1]
            off = offline_metrics(logits, ar_tokens, head.block_size, tree["top_ids"])
            rec = _anchor_record(video, "C0", frozen, actual, ar_ids, n_in,
                                 tree, chain, off, logits)
            rec["c0_replay_matches"] = {
                "commit_len": bool(tree["commit_len"] == actual["commit_len"]),
                "tokens": bool(tree["path_tokens"] == actual["path_tokens"]),
                "bonus": bool(tree["bonus"] == actual["bonus"]),
                "top_ids": bool(tree["top_ids"] == actual["top_ids"]),
                "nodes": bool(tree["nodes"] == actual["nodes"]),
                "draft_logits_bitsum": bool(
                    rec["head_logits_bitsum"] == actual["draft_logits_bitsum"]),
                "actual_bitsum": actual["draft_logits_bitsum"],
            }
            _jsonl_write(anchors_handle, rec)
            stage_a[video]["anchors"].append(
                {"round": frozen["round"], "start": frozen["start"],
                 "prefix_len": frozen["prefix_len"], "ctx_hash": actual["ctx_hash"],
                 "accepted": actual["accepted"], "commit_len": actual["commit_len"],
                 "commit_tokens": actual["commit_tokens"]})
            frozen.pop("actual")
        frozen_all[video] = anchors
        print(f"  A {video[:16]:16s} L={n_in:4d} gen={len(ref_ids):4d} "
              f"rounds={len(ref_rounds):3d} tau={ref_stats.tau:.3f} "
              f"lossless={lossless} copy_tokens={copy_exact['tokens_exact']} "
              f"copy_acc={copy_exact['accepted_exact']} anchors={len(anchors)} "
              f"wall={time.perf_counter() - t0:.0f}s", flush=True)
        with open(status_path, "w") as h:
            json.dump({"stage": "A", "done": list(stage_a)}, h)

    del head
    torch.cuda.empty_cache()

    for arm in ("C1", "C2"):
        print(f"=== stage {arm}: replay on frozen anchors ===", flush=True)
        head = common.load_head(head_path(arm), device=device,
                                dtype=torch.float32).eval()
        for video in wanted:
            row = rows[video]
            ids, pv, vg = encode_row(proc, row, device)
            n_in = int(ids.shape[1])
            ar_ids = stage_a[video]["ar_ids"]
            replay_pos = VisionPositions(pixel_values_videos=pv, video_grid_thw=vg)
            replay_pos.bind(target, ids)
            for frozen in frozen_all[video]:
                logits = arm_draft_logits(head, target, embed, frozen, device,
                                          head.block_size)
                tree = verify_topology(target, frozen, replay_pos, logits, BRANCH,
                                       "tree", device, head.block_size)
                chain = verify_topology(target, frozen, replay_pos, logits, BRANCH,
                                        "chain", device, head.block_size)
                ar_tokens = ar_ids[frozen["start"] + 1 - n_in:
                                   frozen["start"] + 1 - n_in + head.block_size - 1]
                off = offline_metrics(logits, ar_tokens, head.block_size,
                                      tree["top_ids"])
                _jsonl_write(anchors_handle,
                             _anchor_record(video, arm, frozen, None, ar_ids, n_in,
                                            tree, chain, off, logits))
            print(f"  {arm} {video[:16]:16s} anchors={len(frozen_all[video])}",
                  flush=True)
            with open(status_path, "w") as h:
                json.dump({"stage": arm, "done": list(stage_a)}, h)
        del head
        torch.cuda.empty_cache()

    anchors_handle.close()
    with open(stage_a_path, "w") as h:
        json.dump({"config": _config(a), "videos": stage_a}, h, indent=1)
    print(f"-> {anchors_path}\n-> {stage_a_path}")


def cmd_chain16(a):
    """One whole-trajectory decode per arm through the UNMODIFIED decode path.

    Defaults to the real block-16 chain (budget 15, branch 1); passing
    `--budget 63 --branch 8` measures the deployment tree on the same rows, which
    is what makes the chain-vs-tree comparison a comparison of topologies rather
    than of two different row sets.

    `glance.generate` with branch 1 builds the single-column spine, so this is the
    repo's real 15-offset chain rather than the b1k1 single-candidate probe
    (section 5: b1k1 changes width AND depth and is not a chain ablation). It is a
    whole-trajectory measurement, so its prefixes diverge between arms; the
    per-anchor chain replay in `cmd_run` is the paired version.
    """
    from transformers import AutoProcessor

    import glance

    device = f"cuda:{a.gpu}"
    df = load_rows(a.pool)
    prereg = json.load(open(a.prereg))
    wanted = a.videos or prereg["videos"]
    rows = {str(r["video_name"]): r for _, r in df.iterrows()}
    topology = f"b{a.budget}k{a.branch}"
    out = {}
    target, _ = common.load_target(device=device, dtype=torch.float32)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    eos = proc.tokenizer.eos_token_id
    for arm in ARMS:
        head = common.load_head(head_path(arm), device=device,
                                dtype=torch.float32).eval()
        out[arm] = {}
        for video in wanted:
            row = rows[video]
            ids, pv, vg = encode_row(proc, row, device)
            n_in = int(ids.shape[1])
            rounds = []
            spec_out, spec_stats = glance.generate(
                head, target, ids, a.max_new, [eos],
                budget=a.budget, branch_k=a.branch, temperature=0.0,
                positions=VisionPositions(pixel_values_videos=pv, video_grid_thw=vg),
                return_stats=True, timing="decode_only", on_round=rounds.append)
            ar_out = greedy_generate(
                target, ids, a.max_new, [eos],
                positions=VisionPositions(pixel_values_videos=pv, video_grid_thw=vg))
            spec_ids = [int(x) for x in spec_out[0, n_in:].tolist()]
            ar_ids = [int(x) for x in ar_out[0, n_in:].tolist()]
            out[arm][video] = {
                "topology": topology,
                "tau": float(spec_stats.tau), "rounds": len(rounds),
                "gen_tokens": len(spec_ids), "lossless": bool(spec_ids == ar_ids),
                "accepted": [int(r["accepted"]) for r in rounds],
                "pos_rel": [int(r["pos"]) - n_in for r in rounds],
            }
            print(f"  b{a.budget}k{a.branch} {arm} {video[:16]:16s} tau={spec_stats.tau:.4f} "
                  f"rounds={len(rounds)} gen={len(spec_ids)} "
                  f"lossless={spec_ids == ar_ids}", flush=True)
        del head
        torch.cuda.empty_cache()
    with open(a.out, "w") as h:
        json.dump({"config": _config(a), "rows": out}, h, indent=1)
    print(f"-> {a.out}")


# --------------------------------------------------------------------------
# the report (CPU only)
# --------------------------------------------------------------------------

def _mean(values):
    """Mean of a possibly empty list, as nan rather than an exception."""
    return float(np.mean(values)) if len(values) else float("nan")


def _video_bootstrap(deltas, videos):
    """Paired bootstrap over videos; every round of a resampled video is included."""
    keys = [v for v in videos if deltas.get(v)]
    if not keys:
        return None
    arrs = {v: np.asarray(deltas[v], dtype=np.float64) for v in keys}
    rng = np.random.default_rng(BOOT_SEED)
    means = np.empty(BOOT_N)
    for b in range(BOOT_N):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        means[b] = np.concatenate([arrs[keys[i]] for i in pick]).mean()
    vals = np.concatenate([arrs[v] for v in keys])
    return {"mean": float(vals.mean()),
            "ci": [float(np.percentile(means, 2.5)),
                   float(np.percentile(means, 97.5))],
            "n": int(vals.size), "n_videos": len(keys)}


def cmd_report(a):
    prereg = json.load(open(a.prereg))
    # Several capture runs (the 2-video gate, then the other 4 pre-registered
    # videos) are merged here; each is a separate file so a partial run can never
    # be mistaken for the whole set.
    stage_a = {"config": [], "videos": {}}
    for path in a.stage_a:
        blob = json.load(open(path))
        stage_a["config"].append({"path": path, "config": blob["config"]})
        stage_a["videos"].update(blob["videos"])
    rows = []
    for path in a.anchors:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    by = {}
    for r in rows:
        by.setdefault((r["video"], r["round"]), {})[r["arm"]] = r
    videos = sorted({r["video"] for r in rows})

    report = {"n_anchors": len(by), "n_videos": len(videos), "videos": videos}

    # ---- context identity is an invariant, so verify it ----------------
    ctx_consistent = all(len({arm["ctx_hash"] for arm in arms.values()}) == 1
                         for arms in by.values())
    report["ctx_hash_identical_across_arms"] = bool(ctx_consistent)

    # ---- common-prefix bookkeeping -------------------------------------
    # An anchor is on a GENUINELY common prefix across all three arms iff, for
    # every EARLIER round of the same video, all three arms committed exactly the
    # tokens C0's deployment committed. The prefix of anchor r is what rounds
    # 0..r-1 produced, so the test does NOT include round r's own commit: a
    # divergence at round r is what makes round r+1 a different context.
    #
    # The per-anchor C1/C2 rows are still measured on the frozen C0 context (that
    # is the design), but past the divergence point that context is one the arm's
    # own decode would never reach, so those rows are labelled and reported
    # separately instead of pooled with the same-state ones.
    common = {}
    divergence = {}
    commit_diff = {}
    for video in videos:
        first_div = {}
        standing = True
        for rd in sorted(r for (v, r) in by if v == video):
            arms = by[(video, rd)]
            c0 = arms["C0"]["actual"]["commit_tokens"]
            common[(video, rd)] = standing
            offenders = [arm for arm in ARMS if arm != "C0"
                         and arms[arm]["tree"]["commit_tokens"] != c0]
            commit_diff[(video, rd)] = offenders
            for arm in offenders:
                first_div.setdefault(arm, rd)
            if offenders:
                standing = False
        divergence[video] = first_div
    report["first_commit_difference_round"] = divergence
    report["commit_difference_rounds"] = {
        v: int(sum(1 for (vv, rd) in commit_diff if vv == v and commit_diff[(vv, rd)]))
        for v in videos}
    report["commit_difference_rounds_per_arm"] = {
        arm: {v: int(sum(1 for (vv, rd) in commit_diff
                           if vv == v and arm in commit_diff[(vv, rd)]))
              for v in videos}
        for arm in ("C1", "C2")}
    report["common_prefix_anchors"] = sum(1 for v in common.values() if v)
    report["diverged_anchors"] = sum(1 for v in common.values() if not v)

    def sel_rows(arm, only_common):
        return [arms[arm] for (video, rd), arms in by.items()
                if arm in arms and (not only_common or common[(video, rd)])]

    summary = {}
    for arm in ARMS:
        for label, only in (("common", True), ("all", False)):
            sel = sel_rows(arm, only)
            if not sel:
                continue
            exact = sum(1 for r in sel
                        if r["offline"]["rank1_first_rej"] == r["tree"]["first_rej_offset"])
            exact8 = sum(1 for r in sel
                         if r["offline"]["top8_first_rej"] == r["tree"]["first_rej_offset"])
            summary.setdefault(arm, {})[label] = {
                "n": len(sel),
                "offline_rank1_first_rej_mean": _mean(
                    [r["offline"]["rank1_first_rej"] for r in sel
                     if r["offline"]["rank1_first_rej"] is not None]),
                "offline_top8_first_rej_mean": _mean(
                    [r["offline"]["top8_first_rej"] for r in sel
                     if r["offline"]["top8_first_rej"] is not None]),
                "tree_first_rej_mean": _mean(
                    [r["tree"]["first_rej_offset"] for r in sel
                     if r["tree"]["first_rej_offset"] is not None]),
                "tree_commit_mean": float(np.mean([r["tree"]["commit_len"] for r in sel])),
                "chain_commit_mean": float(np.mean([r["chain"]["commit_len"] for r in sel])),
                "offline_rank1_agree_mean": float(np.mean(
                    [r["offline"]["rank1_agree_len"] for r in sel])),
                "offline_top8_agree_mean": float(np.mean(
                    [r["offline"]["top8_agree_len"] for r in sel])),
                "rank1_equals_tree_first_rej": int(exact),
                "top8_equals_tree_first_rej": int(exact8),
                "top8_covers_tree": float(np.mean(
                    [r["tree"]["first_rej_offset"] is not None
                     and r["offline"]["top8_first_rej"] is not None
                     and r["offline"]["top8_first_rej"] >= r["tree"]["first_rej_offset"]
                     for r in sel])),
                "top8_at_1": float(np.mean([bool(r["offline"]["top8_at_1"]) for r in sel])),
                "top1_at_1": float(np.mean([bool(r["offline"]["top1_at_1"]) for r in sel])),
                "tree_commit_ge2": float(np.mean([r["tree"]["commit_len"] >= 2 for r in sel])),
                "tree_commit_ge4": float(np.mean([r["tree"]["commit_len"] >= 4 for r in sel])),
                "rej_cause": {c: int(sum(1 for r in sel if r["tree"]["rej_cause"] == c))
                              for c in ("not_in_topk", "in_topk_not_in_tree",
                                        "in_tree", "none")},
                "all_accepted_in_block": int(sum(1 for r in sel
                                                 if r["tree"]["all_accepted_in_block"])),
                "chain_equals_tree_commit": int(sum(
                    1 for r in sel if r["chain"]["commit_len"] == r["tree"]["commit_len"])),
                "chain_ge_tree_commit": int(sum(
                    1 for r in sel if r["chain"]["commit_len"] >= r["tree"]["commit_len"])),
            }
    report["per_arm"] = summary

    # ---- offline/actual alignment gate ---------------------------------
    # The offline ranks are recomputed from nothing here: they are CHECKED
    # against the stored candidate sets. `rank == 1` at offset k must mean the
    # head's top-1 at offset k IS the target's AR token at position start+k, and
    # `rank == r <= 8` must mean that token sits exactly at index r-1 of the
    # stored top-8. If this gate fails, the offline and the actual quantities are
    # not being compared on the same object and no other number in this report
    # may be used.
    align_ok = align_bad = 0
    top8_ok = top8_bad = 0
    align_examples = []
    for r in rows:
        win = r.get("ar_window")
        if win is None:
            sa = stage_a["videos"][r["video"]]
            win = _ar_window(sa["ar_ids"], r["start"], r["n_in"], 16)
        ranks = r["offline"]["ranks"]
        top = r["tree"]["top_ids"]
        for k in range(min(len(ranks), len(win), len(top))):
            if ranks[k] == 1:
                if top[k][0] == win[k]:
                    align_ok += 1
                else:
                    align_bad += 1
                    if len(align_examples) < 5:
                        align_examples.append([r["video"], r["round"], r["arm"], k + 1,
                                               top[k][0], win[k]])
            if ranks[k] <= 8:
                if top[k][ranks[k] - 1] == win[k]:
                    top8_ok += 1
                else:
                    top8_bad += 1
    report["offline_alignment_gate"] = {
        "rank1_matches_head_argmax": {"ok": align_ok, "bad": align_bad},
        "rank_le8_matches_top8_index": {"ok": top8_ok, "bad": top8_bad},
        "examples": align_examples,
        "pass": bool(align_bad == 0 and top8_bad == 0),
    }

    # ---- §11.4 bridging on ALIGNED contexts ---------------------------
    # "commit >= 1 draft token" is commit_len >= 2. Offline, the same event at
    # offset 1 is "the target's AR token is inside the head's top-8". Both are
    # measured here on the SAME anchor and the SAME distribution, which is the
    # only way the comparison is meaningful.
    bridging = {}
    for arm in ARMS:
        for label, only in (("common", True), ("all", False)):
            sel = sel_rows(arm, only)
            if not sel:
                continue
            ge2 = sum(1 for r in sel if r["tree"]["commit_len"] >= 2)
            t8 = sum(1 for r in sel if r["offline"]["top8_at_1"])
            disagree = [[r["video"], r["round"], r["tree"]["commit_len"],
                         r["offline"]["ranks"][0], r["tree"]["rej_cause"]]
                        for r in sel
                        if (r["tree"]["commit_len"] >= 2) != bool(r["offline"]["top8_at_1"])]
            bridging.setdefault(arm, {})[label] = {
                "n": len(sel),
                "P_tree_commit_ge1_draft": ge2 / len(sel),
                "P_offline_top8_at_1": t8 / len(sel),
                "n_tree_commit_ge1_draft": ge2,
                "n_offline_top8_at_1": t8,
                "disagreements": disagree,
                "top8_binding": _mean([(r["offline"]["top8_agree_len"] + 1)
                                       - r["tree"]["commit_len"] for r in sel]),
                "top8_selection_loss_positive": int(sum(
                    1 for r in sel
                    if (r["offline"]["top8_agree_len"] + 1) > r["tree"]["commit_len"])),
                "top8_selection_loss_max": int(max(
                    (r["offline"]["top8_agree_len"] + 1) - r["tree"]["commit_len"]
                    for r in sel)),
            }
    report["bridging_11_4"] = bridging

    # ---- whole-trajectory topologies (optional) -----------------------
    if a.decode:
        whole = {}
        for path in a.decode:
            blob = json.load(open(path))
            for arm, per_video in blob["rows"].items():
                for video, rec in per_video.items():
                    whole.setdefault(arm, {}).setdefault(video, {})[
                        rec["topology"]] = rec
        report["whole_trajectory"] = whole
        tops = sorted({t for arm in whole for v in whole[arm] for t in whole[arm][v]})
        cmp = {}
        if len(tops) >= 2:
            for arm in ARMS:
                deltas = {v: [] for v in videos}
                for v in videos:
                    recs = whole.get(arm, {}).get(v, {})
                    if len(recs) >= 2:
                        deltas[v].append(recs[tops[0]]["tau"] - recs[tops[1]]["tau"])
                res = _video_bootstrap(deltas, videos)
                if res:
                    cmp[f"{arm} tau({tops[0]}) - tau({tops[1]})"] = res
                for t in tops:
                    vals = [whole[arm][v][t]["tau"] for v in videos
                            if v in whole.get(arm, {}) and t in whole[arm][v]]
                    if vals:
                        cmp[f"{arm} tau({t}) mean"] = {"mean": float(np.mean(vals)),
                                                        "n_videos": len(vals)}
        report["whole_trajectory_compare"] = cmp

    # ---- C0 replay gate -----------------------------------------------
    gate = {}
    c0rows = [r for r in rows if r["arm"] == "C0"]
    for video in videos:
        sel = [r for r in c0rows if r["video"] == video]
        gate[video] = {
            "n_anchors": len(sel),
            "commit_len_exact": int(sum(1 for r in sel
                                        if r["c0_replay_matches"]["commit_len"])),
            "tokens_exact": int(sum(1 for r in sel if r["c0_replay_matches"]["tokens"])),
            "bonus_exact": int(sum(1 for r in sel if r["c0_replay_matches"]["bonus"])),
            "top_ids_exact": int(sum(1 for r in sel if r["c0_replay_matches"]["top_ids"])),
            "nodes_exact": int(sum(1 for r in sel if r["c0_replay_matches"]["nodes"])),
            "draft_bitsum_exact": int(sum(1 for r in sel
                                          if r["c0_replay_matches"]["draft_logits_bitsum"])),
            "in_gate": video in prereg["gate_videos"],
        }
    report["c0_replay_gate"] = gate
    gv = [v for v in prereg["gate_videos"] if v in gate]
    report["gate_totals"] = {
        "videos": gv,
        "n_anchors": sum(gate[v]["n_anchors"] for v in gv),
        "commit_len_exact": sum(gate[v]["commit_len_exact"] for v in gv),
        "tokens_exact": sum(gate[v]["tokens_exact"] for v in gv),
        "bonus_exact": sum(gate[v]["bonus_exact"] for v in gv),
        "top_ids_exact": sum(gate[v]["top_ids_exact"] for v in gv),
        "draft_bitsum_exact": sum(gate[v]["draft_bitsum_exact"] for v in gv),
    }
    report["copy_exact"] = {
        v: {"tokens": stage_a["videos"][v]["copy_exact"]["tokens_exact"],
            "accepted": stage_a["videos"][v]["copy_exact"]["accepted_exact"],
            "n_rounds": stage_a["videos"][v]["copy_exact"]["n_rounds_ref"],
            "lossless": stage_a["videos"][v]["lossless"],
            "tau": stage_a["videos"][v]["tau"]}
        for v in videos}

    # ---- paired arm vs arm on common-prefix anchors --------------------
    paired = {}
    for arm in ("C1", "C2"):
        for name, key in (("tree_commit", lambda r: r["tree"]["commit_len"]),
                          ("chain_commit", lambda r: r["chain"]["commit_len"]),
                          ("offline_rank1_agree",
                           lambda r: r["offline"]["rank1_agree_len"]),
                          ("offline_top8_agree",
                           lambda r: r["offline"]["top8_agree_len"])):
            deltas = {v: [] for v in videos}
            for (video, rd), arms in by.items():
                if not common[(video, rd)] or arm not in arms:
                    continue
                deltas[video].append(key(arms[arm]) - key(arms["C0"]))
            res = _video_bootstrap(deltas, videos)
            if res:
                paired[f"{arm}-C0_{name}"] = res
    report["paired"] = paired

    # ---- per-video table ----------------------------------------------
    report["per_video"] = {}
    for video in videos:
        entry = {"L": stage_a["videos"][video]["L"],
                 "tau_C0_deploy": stage_a["videos"][video]["tau"],
                 "lossless": stage_a["videos"][video]["lossless"],
                 "gen_tokens": stage_a["videos"][video]["gen_tokens"],
                 "first_commit_difference": divergence[video],
                 "commit_diff_rounds": report["commit_difference_rounds"][video],
                 "per_arm": {}}
        for arm in ARMS:
            sel = [arms[arm] for (v, rd), arms in by.items()
                   if v == video and arm in arms]
            if not sel:
                continue
            entry["per_arm"][arm] = {
                "rounds": [{
                    "round": r["round"], "start": r["start"],
                    "off_rank1_rej": r["offline"]["rank1_first_rej"],
                    "off_top8_rej": r["offline"]["top8_first_rej"],
                    "tree_rej": r["tree"]["first_rej_offset"],
                    "tree_commit": r["tree"]["commit_len"],
                    "chain_commit": r["chain"]["commit_len"],
                    "rej_cause": r["tree"]["rej_cause"],
                    "ranks": r["offline"]["ranks"],
                    "commit_tokens": r["tree"]["commit_tokens"],
                } for r in sorted(sel, key=lambda x: x["round"])],
                "offline_rank1_agree_mean": float(np.mean(
                    [r["offline"]["rank1_agree_len"] for r in sel])),
                "offline_top8_agree_mean": float(np.mean(
                    [r["offline"]["top8_agree_len"] for r in sel])),
                "tree_commit_mean": float(np.mean([r["tree"]["commit_len"] for r in sel])),
                "chain_commit_mean": float(np.mean([r["chain"]["commit_len"] for r in sel])),
            }
        report["per_video"][video] = entry

    with open(a.out_json, "w") as h:
        json.dump(report, h, indent=1)

    # ---- markdown ------------------------------------------------------
    L = []
    L.append(f"# B-reference alignment report ({a.tag})\n")
    L.append(f"anchors: **{report['n_anchors']}** over **{report['n_videos']}** videos; "
             f"common-prefix anchors **{report['common_prefix_anchors']}**, "
             f"diverged **{report['diverged_anchors']}**; "
             f"context hash identical across arms: "
             f"{report['ctx_hash_identical_across_arms']}\n")
    L.append("## 1. C0 replay gate (offline-state replay vs the real C0 decode)\n")
    L.append("| video | in gate | anchors | commit_len exact | tokens exact | bonus exact "
             "| nodes exact | top-8 exact | draft logits bit-sum exact |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for v in videos:
        g = gate[v]
        L.append(f"| {v} | {g['in_gate']} | {g['n_anchors']} | {g['commit_len_exact']} "
                 f"| {g['tokens_exact']} | {g['bonus_exact']} | {g['nodes_exact']} "
                 f"| {g['top_ids_exact']} | {g['draft_bitsum_exact']} |")
    gt = report["gate_totals"]
    L.append(f"\ngate videos {gt['videos']}: "
             f"{gt['commit_len_exact']}/{gt['n_anchors']} commit length, "
             f"{gt['tokens_exact']}/{gt['n_anchors']} committed tokens, "
             f"{gt['top_ids_exact']}/{gt['n_anchors']} identical draft top-8, "
             f"{gt['draft_bitsum_exact']}/{gt['n_anchors']} bit-identical draft logits\n")
    L.append("| video | instrumented decode == unmodified generate | AR lossless | rounds |")
    L.append("|---|---|---|---:|")
    for v in videos:
        c = report["copy_exact"][v]
        L.append(f"| {v} | tokens={c['tokens']} accepted={c['accepted']} | "
                 f"{c['lossless']} | {c['n_rounds']} |")
    L.append("\n## 2. per-arm means (common-prefix anchors only)\n")
    L.append("| arm | n | off rank1 first-rej | off top8 first-rej | tree first-rej "
             "| tree commit | chain commit | off rank1 agree | off top8 agree "
             "| rank1==tree | top8==tree | top8 covers tree | top8@1 | top1@1 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        s = summary.get(arm, {}).get("common")
        if not s:
            continue
        L.append(f"| {arm} | {s['n']} | {s['offline_rank1_first_rej_mean']:.3f} "
                 f"| {s['offline_top8_first_rej_mean']:.3f} | {s['tree_first_rej_mean']:.3f} "
                 f"| {s['tree_commit_mean']:.3f} | {s['chain_commit_mean']:.3f} "
                 f"| {s['offline_rank1_agree_mean']:.3f} "
                 f"| {s['offline_top8_agree_mean']:.3f} "
                 f"| {s['rank1_equals_tree_first_rej']}/{s['n']} "
                 f"| {s['top8_equals_tree_first_rej']}/{s['n']} "
                 f"| {s['top8_covers_tree']:.3f} | {s['top8_at_1']:.4f} "
                 f"| {s['top1_at_1']:.4f} |")
    L.append("\n### rejection cause at the first rejected offset\n")
    L.append("| arm | not_in_topk (coverage) | in_topk_not_in_tree (budget) "
             "| in_tree (inconsistent) | none (block fully accepted) |")
    L.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        s = summary.get(arm, {}).get("common")
        if not s:
            continue
        c = s["rej_cause"]
        L.append(f"| {arm} | {c['not_in_topk']} | {c['in_topk_not_in_tree']} "
                 f"| {c['in_tree']} | {c['none']} |")
    ag = report["offline_alignment_gate"]
    L.append("\n## 3. offline/actual alignment gate\n")
    L.append(f"rank==1 vs stored head argmax: {ag['rank1_matches_head_argmax']['ok']} ok / "
             f"{ag['rank1_matches_head_argmax']['bad']} bad; "
             f"rank<=8 vs stored top-8 index: {ag['rank_le8_matches_top8_index']['ok']} ok / "
             f"{ag['rank_le8_matches_top8_index']['bad']} bad -> pass={ag['pass']}\n")
    if ag["examples"]:
        L.append(f"counterexamples: {ag['examples']}\n")
    L.append("## 4. bridging section 11.4 on aligned contexts\n")
    L.append("`commit>=1 draft token` is `tree commit >= 2`; the offline twin at offset 1 is "
             "`rank<=8`. Both are read here from the same anchor and the same marginals.\n")
    L.append("| arm | set | n | P(tree commit>=1 draft) | P(offline top8@1) | disagreements "
             "| mean top8 ceiling - tree commit | anchors with selection loss | max loss |")
    L.append("|---|---|---:|---:|---:|---|---:|---:|---:|")
    for arm in ARMS:
        for label in ("common", "all"):
            b = report["bridging_11_4"].get(arm, {}).get(label)
            if not b:
                continue
            L.append(f"| {arm} | {label} | {b['n']} "
                     f"| {b['n_tree_commit_ge1_draft']}/{b['n']} = {b['P_tree_commit_ge1_draft']:.4f} "
                     f"| {b['n_offline_top8_at_1']}/{b['n']} = {b['P_offline_top8_at_1']:.4f} "
                     f"| {len(b['disagreements'])} "
                     f"| {b['top8_binding']:+.4f} "
                     f"| {b['top8_selection_loss_positive']}/{b['n']} "
                     f"| {b['top8_selection_loss_max']} |")
    for arm in ARMS:
        for label in ("common", "all"):
            b = report["bridging_11_4"].get(arm, {}).get(label)
            if b and b["disagreements"]:
                L.append(f"\ndisagreements {arm}/{label} "
                         f"(video, round, tree commit, offset-1 rank, rej cause): "
                         f"{b['disagreements']}")
    L.append("\n## 5. whole-trajectory topologies (unmodified decode path)\n")
    if report.get("whole_trajectory_compare"):
        for k, v in report["whole_trajectory_compare"].items():
            if "ci" in v:
                L.append(f"- {k}: {v['mean']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
                         f"n={v['n']} videos={v['n_videos']}")
            else:
                L.append(f"- {k}: {v['mean']:+.4f} (videos={v['n_videos']})")
    else:
        L.append("not run in this report invocation")
    L.append("\n## 6. paired deltas (common-prefix anchors, bootstrap over videos)\n")
    for k, v in paired.items():
        L.append(f"- {k}: {v['mean']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
                 f"n={v['n']} videos={v['n_videos']}")
    L.append("\n## 7. per-video, per-round\n")
    for v in videos:
        e = report["per_video"][v]
        L.append(f"### {v}  (L={e['L']}, deploy tau={e['tau_C0_deploy']:.4f}, "
                 f"lossless={e['lossless']}, gen={e['gen_tokens']}, "
                 f"first commit difference={e['first_commit_difference']}, "
                 f"diff rounds={e['commit_diff_rounds']})")
        L.append("| round | start | arm | off rank1 first-rej | off top8 first-rej "
                 "| tree first-rej | tree commit | chain commit | rej cause | ranks 1..15 |")
        L.append("|---:|---:|---|---:|---:|---:|---:|---:|---|---|")
        for arm in ARMS:
            pa = e["per_arm"].get(arm)
            if not pa:
                continue
            for rr in pa["rounds"]:
                L.append(f"| {rr['round']} | {rr['start']} | {arm} "
                         f"| {rr['off_rank1_rej']} | {rr['off_top8_rej']} "
                         f"| {rr['tree_rej']} | {rr['tree_commit']} "
                         f"| {rr['chain_commit']} | {rr['rej_cause']} "
                         f"| {rr['ranks']} |")
        L.append("")
    with open(a.out_md, "w") as h:
        h.write("\n".join(L) + "\n")
    print("\n".join(L[:80]))
    print(f"\n-> {a.out_md}\n-> {a.out_json}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prereg")
    p.add_argument("--pool", default=f"{common.ROOT}/data/video_fresh30.parquet")
    p.add_argument("--warmup-row", type=int, default=WARMUP_ROW)
    p.add_argument("--out",
                   default=f"{common.ROOT}/results/phaseC5_brief/prereg.json")
    p.set_defaults(func=cmd_prereg)

    p = sub.add_parser("run")
    p.add_argument("--pool", default=f"{common.ROOT}/data/video_fresh30.parquet")
    p.add_argument("--prereg",
                   default=f"{common.ROOT}/results/phaseC5_brief/prereg.json")
    p.add_argument("--videos", nargs="*", default=None,
                   help="override the pre-registered list (the gate step passes 2)")
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", default=f"{common.ROOT}/results/phaseC5_brief")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("chain16")
    p.add_argument("--pool", default=f"{common.ROOT}/data/video_fresh30.parquet")
    p.add_argument("--prereg",
                   default=f"{common.ROOT}/results/phaseC5_brief/prereg.json")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--budget", type=int, default=CHAIN_BUDGET)
    p.add_argument("--branch", type=int, default=CHAIN_BRANCH)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_chain16)

    p = sub.add_parser("report")
    p.add_argument("--prereg", required=True)
    p.add_argument("--stage-a", nargs="+", required=True)
    p.add_argument("--anchors", nargs="+", required=True)
    p.add_argument("--decode", nargs="*", default=None,
                   help="whole-trajectory decodes from `chain16` (one per topology)")
    p.add_argument("--tag", default="run")
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    p.set_defaults(func=cmd_report)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
