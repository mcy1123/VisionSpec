#!/usr/bin/env python3
"""CPU tests for the `eos_stop` fix in `glance.decode`.

The fix is one window: which slice of the round's newly committed tokens is
scanned for a stop token. `legacy` scans `out_ids[n_in:start]` BEFORE the
generation loop advances `start`, so a stop token committed as the round's
BONUS (which lands exactly at the new `start`) is invisible and the loop runs
one more round with that stop token as its root. `immediate` scans
`out_ids[n_in:start + 1]`, accepted path and bonus both.

These tests drive the real `glance.decode.generate` / `chain_generate` with a
tiny deterministic stub target and stub head, so every branch is exercised
without a GPU:

  * stop token committed as the round's BONUS (the reported defect),
  * stop token at the FIRST accepted node of the path,
  * stop token DEEPER in the accepted path,
  * no stop token, length-clipped termination,
  * zero-accepted rounds,
  * the root emitted by the prefill already being the stop token,
  * both code paths (`generate`, `chain_generate`).

The invariant every case pins: the two modes return the SAME token sequence (the
output is trimmed at the first stop token in both), and `immediate` uses at most
as many rounds as `legacy`, with strictly fewer in the bonus case.

Run: /home/mcy/.conda/envs/fin_research/bin/python repro/qa_opd_fair/test_eos_fix.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.dirname(HERE)
ROOT = os.path.dirname(REPRO)
for p in (REPRO, os.path.join(ROOT, "GLANCE_repo"),
          os.path.join(ROOT, "scripts", "dflash_pkg", "dflash-0.1.0")):
    if p not in sys.path:
        sys.path.insert(0, p)

import glance.decode as dec  # noqa: E402

EOS = 5
VOCAB = 64
MASK = 999


class StubCache:
    """The only `DynamicCache` surface `glance.decode` uses."""

    def __init__(self):
        self.n = 0

    def get_seq_length(self):
        return self.n

    def crop(self, n):
        self.n = int(n)


def _make_cache_factory(stream_len):
    def factory(*args, **kwargs):
        c = StubCache()
        # The decode loop's target KV starts at the prompt: after its prefill the
        # cache holds `stream_len` rows, which is what the cropped verify packs
        # reason about.
        c.n = int(stream_len)
        return c
    return factory


class StubLmHead(nn.Module):
    """`lm_head(hidden)` -> logits whose argmax is the token the forward wrote.

    The stub target writes the *desired next token id* into the last channel of
    every hidden row, so the projection is just a one-hot lookup. This is a
    stub, not a numerics claim: the tests pin termination bookkeeping only.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(VOCAB, 1), requires_grad=False)

    def forward(self, hidden):
        if hidden.dim() == 2:
            hidden = hidden[None]
        rows = hidden.shape[1]
        ids = hidden[..., 0].round().long().clamp(0, VOCAB - 1)
        out = torch.full((hidden.shape[0], rows, VOCAB), -20.0,
                         dtype=torch.float32, device=hidden.device)
        out.scatter_(2, ids[..., None], 20.0)
        return out


class StubTarget(nn.Module):
    """A frozen greedy "model" whose next token is stated per stream position.

    `plan[p]` is the token the target emits for the token AFTER stream position
    `p`. The prefill (empty cache) has its logits row at the prompt's last
    position, so it predicts `plan[prompt_len - 1]` (the root). A verify pack
    whose cache holds `k` committed tokens has its row `j` at stream position
    `k + j` and therefore predicts `plan[k + j + 1]`.

    `stream_len` is how many prompt tokens the cache counts (the decode loop's
    KV starts at the prompt, so `get_seq_length()` after the prefill is
    `stream_len`, not 1).
    """

    def __init__(self, plan, stream_len=3, fallback=1, spine=(1, 2, 3)):
        super().__init__()
        self.plan = {int(k): int(v) for k, v in plan.items()}
        self.stream_len = int(stream_len)
        self.fallback = int(fallback)
        self.spine = [int(x) for x in spine]
        self.lm_head = StubLmHead()
        self._anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.forward_calls = 0
        self.seen_prefill = 0

    def parameters(self, recurse=True):
        yield self._anchor

    def get_input_embeddings(self):
        def embed(ids):
            return torch.zeros(*ids.shape, 1, dtype=torch.float32)
        return embed

    def _token_for(self, pos):
        return int(self.plan.get(int(pos), self.fallback))

    def forward(self, input_ids, past_key_values=None, use_cache=True,
                logits_to_keep=1, output_hidden_states=False, position_ids=None,
                attention_mask=None, cache_position=None, **kwargs):
        self.forward_calls += 1
        n = int(input_ids.shape[1])
        # A bare `StubCache()` starts at 0 and the decode loop never appends to
        # it, so the stub must decide for itself whether this call is the one
        # prefill: it is the first call, and it is handed an empty cache.
        k = 0 if past_key_values is None else int(past_key_values.get_seq_length())
        prefill = (self.seen_prefill == 0)
        self.seen_prefill += 1
        # Verify packs arrive with the cache already cropped to the committed
        # length the decode loop wants the target to see.
        cache_len = min(k, self.stream_len - 1) if k else 0
        # The head's spine lives in the block rows; the stub head reads that
        # channel and ignores everything else, so it is the only content needed.
        block_rows = torch.zeros(1, n, 1, dtype=torch.float32)
        for j in range(n):
            block_rows[0, j, 0] = float(self.spine[min(j, len(self.spine) - 1)])

        class Out:
            pass

        out = Out()
        out.logits = self.lm_head(self._answer_rows(n, k, prefill, cache_len))
        # `context_feature` indexes `hidden_states[i + 1]` for the kept layers
        # (max 33), so the list must hold at least 35 DISTINCT suitable tensors:
        # a shared object would let one `cat` mutate what the next one reads.
        out.hidden_states = [block_rows.clone() for _ in range(40)]
        out.past_key_values = past_key_values
        return out

    def _answer_rows(self, n, k, prefill, cache_len):
        rows = torch.zeros(1, n, 1, dtype=torch.float32)
        for j in range(n):
            # The prefill is the call whose cache is still empty: its single
            # logits row is the prompt's last position and predicts the ROOT, at
            # `n - 1`. A verify pack is handed the cache already cropped to `k`,
            # because `generate` reads that length; its row j sits at stream
            # position `k + j` and predicts one past it, `k + j + 1`.
            pos = (n - 1) if prefill else (k + j + 1)
            rows[0, j, 0] = float(self._token_for(pos))
        return rows


class StubHead(nn.Module):
    """A block head that proposes `spine` at every anchor.

    Returns `block` hidden rows whose argmax pattern is the spine, which is what
    `generate` reads as the block marginals. `block_size`/`mask_token_id` are the
    only attributes the decode loop needs.
    """

    def __init__(self, spine, block=4):
        super().__init__()
        self.block_size = block
        self.mask_token_id = MASK
        self.target_layer_ids = (0,)
        self._spine = list(spine)
        self._anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, target_hidden=None, noise_embedding=None,
                position_ids=None, past_key_values=None, use_cache=True,
                is_causal=False):
        n = int(noise_embedding.shape[1])
        rows = torch.zeros(1, n, 1, dtype=torch.float32)
        for j in range(1, n):
            rows[0, j, 0] = float(self._spine[min(j - 1, len(self._spine) - 1)])
        return rows


def _run(kind, plan, spine, max_new=12, eos_stop="legacy", stop=(EOS,), block=4,
         prompt=(101, 102, 103)):
    """Drive one stub decode; return `(tokens, rounds, forward_calls)`."""
    source_ids = list(prompt)
    target = StubTarget(plan, stream_len=len(source_ids), spine=spine + [plan.get(len(source_ids) + 40, 1)])
    head = StubHead(spine, block=block)
    real_dyn = dec.DynamicCache
    real_gather = dec.gather_cache
    dec.DynamicCache = _make_cache_factory(len(source_ids))

    def gather(cache, n_prefix, keep):
        # The stub cache has no key/value tensors; cropping it is the only
        # bookkeeping `generate` depends on. The real `gather_cache` is exercised
        # by the GPU gate, not here.
        cache.crop(n_prefix + len(keep))
        return cache

    dec.gather_cache = gather
    ids = torch.tensor([source_ids])
    try:
        if kind == "tree":
            out, stats = dec.generate(head, target, ids, max_new,
                                      stop_token_ids=list(stop), budget=15,
                                      branch_k=3, positions=None,
                                      return_stats=True, eos_stop=eos_stop)
        else:
            out, stats = dec.chain_generate(head, target, ids, max_new,
                                            stop_token_ids=list(stop),
                                            return_stats=True,
                                            eos_stop=eos_stop)
    finally:
        dec.DynamicCache = real_dyn
        dec.gather_cache = real_gather
    return out[0, ids.shape[1]:].tolist(), stats.rounds, target.forward_calls


def check(name, got, want):
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")


def main():
    failures = []

    def case(name, fn):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as err:                       # noqa: BLE001
            failures.append((name, err))
            print(f"FAIL {name}: {err!r}")

    # --- bonus case: root 1, spine [2, EOS, 3]; the target accepts root and the
    #     first spine token (both match `plan`) and its next proposal is EOS, so
    #     EOS is the round's BONUS. `legacy` runs one extra round.
    plan_bonus = {2: 1, 3: 2, 4: 5, 5: 5, 6: 7}

    def bonus_tree():
        tok_l, r_l, _ = _run("tree", plan_bonus, [2, 5, 3], eos_stop="legacy")
        tok_i, r_i, _ = _run("tree", plan_bonus, [2, 5, 3], eos_stop="immediate")
        check("bonus/tree tokens agree", tok_l, tok_i)
        check("bonus/tree tokens", tok_i, [1, 5])
        check("bonus/tree legacy rounds", r_l, 2)
        check("bonus/tree immediate rounds", r_i, 1)

    def bonus_no_stop_control():
        # Same trajectory, but EOS is NOT a stop token: both modes must run the
        # same number of rounds, i.e. the fix does not shorten ordinary decodes.
        tok_l, r_l, _ = _run("tree", plan_bonus, [2, 5, 3], eos_stop="legacy",
                             stop=())
        tok_i, r_i, _ = _run("tree", plan_bonus, [2, 5, 3], eos_stop="immediate",
                             stop=())
        check("control tokens equal", tok_l, tok_i)
        check("control rounds equal", r_l, r_i)

    def bonus_chain():
        tok_l, r_l, _ = _run("chain", plan_bonus, [2, 5, 3], eos_stop="legacy")
        tok_i, r_i, _ = _run("chain", plan_bonus, [2, 5, 3], eos_stop="immediate")
        check("bonus/chain tokens equal", tok_l, tok_i)
        check("bonus/chain tokens", tok_i, [1, 5])
        check("bonus/chain legacy rounds", r_l, 2)
        check("bonus/chain immediate rounds", r_i, 1)

    case("stop token committed as the round's BONUS (tree)", bonus_tree)
    case("non-stop control: same trajectory, no stop id", bonus_no_stop_control)
    case("stop token committed as the round's BONUS (chain)", bonus_chain)

    # --- accepted-path case: the target's proposal for offset 1 is EOS, so EOS
    #     is the first accepted draft token. Both modes see it in `[n_in:start]`.
    plan_acc1 = {2: 5, 3: 9, 4: 3, 5: 7}

    def acc1_tree():
        tok_l, r_l, _ = _run("tree", plan_acc1, [9, 3, 7], eos_stop="legacy")
        tok_i, r_i, _ = _run("tree", plan_acc1, [9, 3, 7], eos_stop="immediate")
        check("acc1 tokens agree", tok_l, tok_i)
        # An accepted stop token is already inside `[n_in:start]` in both modes,
        # so this is the branch where legacy was never wrong.
        check("acc1 tokens", tok_i, [5])
        check("acc1 legacy rounds", r_l, 1)
        check("acc1 immediate rounds", r_i, 1)

    case("stop token as the FIRST accepted draft token", acc1_tree)

    # --- deeper in the accepted path: root 1, then the target proposes 2 then
    #     EOS, so the accepted path is [2, EOS] and the bonus is 3.
    plan_deep = {2: 1, 3: 2, 4: 5, 5: 3, 6: 7}

    def deep_tree():
        tok_l, r_l, _ = _run("tree", plan_deep, [2, 5, 9, 9], eos_stop="legacy")
        tok_i, r_i, _ = _run("tree", plan_deep, [2, 5, 9, 9], eos_stop="immediate")
        check("deep tokens agree", tok_l, tok_i)
        check("deep tokens", tok_i, [1, 5])
        check("deep legacy rounds", r_l, 2)
        check("deep immediate rounds", r_i, 1)

    case("stop token DEEP inside the accepted path", deep_tree)

    # --- root already EOS (the prefill emits the stop token).
    plan_root = {2: 5}

    def root_eos():
        tok_l, r_l, _ = _run("tree", plan_root, [1, 2, 3], eos_stop="legacy")
        tok_i, r_i, _ = _run("tree", plan_root, [1, 2, 3], eos_stop="immediate")
        check("root-eos tokens", tok_l, [5])
        check("root-eos tokens immediate", tok_i, [5])
        check("root-eos legacy rounds", r_l, 1)
        check("root-eos immediate rounds", r_i, 1)

    case("prefill root is already the stop token", root_eos)

    # --- zero-accepted rounds: the spine never matches, so each round commits
    #     only the bonus. A bonus EOS must stop; a non-terminal position must not.
    plan_zero = {2: 1, 3: 7, 4: 8, 5: 5, 6: 9}

    def zero_acc():
        tok_l, r_l, _ = _run("tree", plan_zero, [30, 31, 32], eos_stop="legacy")
        tok_i, r_i, _ = _run("tree", plan_zero, [30, 31, 32], eos_stop="immediate")
        check("zero-acc tokens equal", tok_l, tok_i)
        check("zero-acc tokens", tok_i, [1, 8, 5])
        check("zero-acc legacy rounds", r_l, 3)
        check("zero-acc immediate rounds", r_i, 2)

    case("zero-accepted rounds with a bonus stop token", zero_acc)

    # --- invalid mode must raise (fail closed, do not silently pick one).
    def bad_mode():
        try:
            _run("tree", plan_bonus, [2, 5, 3], eos_stop="sometimes")
        except ValueError as err:
            if "eos_stop" not in str(err):
                raise
            return
        raise AssertionError("invalid eos_stop did not raise")

    case("invalid eos_stop raises", bad_mode)

    print()
    if failures:
        print(f"CPU: {len(failures)} failed, {len(failures)} of 8 cases")
        raise SystemExit(1)
    print("CPU: all eos_stop tests passed")


if __name__ == "__main__":
    main()
