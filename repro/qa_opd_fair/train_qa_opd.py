#!/usr/bin/env python3
"""Compatibility entry to opd_official_kl.cli (Bernoulli/k3 explicit override).

Historical helper definitions remain importable by diagnostic scripts only.
The former training main is disabled; original source is under legacy/pre20261016.
"""
from __future__ import annotations

# Historical helper API retained for diagnostics. New training dispatches only to the canonical runtime.
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _repro = next(p for p in _Path(__file__).resolve().parents if p.name == "repro")
    _sys.path.insert(0, str(_repro))
    from opd_official_kl.cli import main as _official_main
    _official_main(["--modality", "video", *_sys.argv[1:]])
    raise SystemExit(0)


import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
from historical import opd_fullkl  # noqa: E402
import qa_anchors  # noqa: E402
from shared.master_optim import MasterWeightAdamW  # noqa: E402
from train_qa_sft import LmHeadShim, draft_block_logits_mixed  # noqa: E402
from videoencode import build_video_text, encode_inputs  # noqa: E402

K = common.BLOCK_SIZE - 1


def gamma_weights(offsets, gamma, device):
    idx = offsets.detach().to(device=device)
    w = torch.pow(torch.as_tensor(float(gamma), dtype=torch.float64, device=device),
                  idx.double() - 1.0)
    return w.float()


def main(argv=None):
    """New callers cannot invoke historical training through an imported main."""
    from opd_official_kl.cli import main as gate
    return gate(["--modality", "video", *(sys.argv[1:] if argv is None else argv)])


def _historical_main_disabled(*args, **kwargs):
    """The original training loop is archived under legacy/pre20261016."""
    raise RuntimeError("Use legacy/replay.py for explicitly acknowledged historical replay")


def _verifier_argmax(target, full, vispos, prefix_kv, prompt_len, n_ans, max_seq_len):
    """The frozen target's own next-token argmax over the answer positions.

    `prefix_kv` carries the prefill's per-position last hidden state under
    `opd_fullkl.PREFIX_LAST_HIDDEN` only for the LAST position, so this runs one
    extra cached forward over the whole trajectory with hidden states exposed.
    It is the only way to read the target's argmax at *every* answer position,
    and it is what acceptance is compared against.
    """
    hidden = _all_last_hidden(target, full, vispos)
    # Row `prompt_len - 1 + i` predicts the token at `prompt_len + i`; index 0 of
    # the returned list is therefore "the target's argmax predicted BY the last
    # prompt token", i.e. the root, and index i is the argmax for answer token i.
    rows = hidden[:, prompt_len - 1: prompt_len + n_ans - 1, :]
    return target.lm_head(rows)[0].argmax(-1).tolist()


def _all_last_hidden(target, full, vispos):
    from transformers import DynamicCache
    out = target.model(full[None], past_key_values=DynamicCache(),
                       use_cache=True, **vispos.prefill_inputs(full[None], None))
    return out.last_hidden_state


def _verify(prop, verifier, p, prompt_len, n_ans, eos_pos, *, eos_token_ids=None):
    """verifier[i] predicts answer i, i.e. absolute position prompt_len+i.

    Acceptance stops at the first mismatch. Validity does NOT stop at rejection:
    the entire rejected suffix is supervised, bounded by the answer horizon,
    reference EOS, and (when supplied) the first proposal EOS, inclusive.
    """
    if len(verifier) != n_ans:
        raise ValueError("verifier must have exactly one row per answer token")
    eos = set([] if eos_token_ids is None else
              [eos_token_ids] if isinstance(eos_token_ids, int) else eos_token_ids)
    valid, n_acc, accepting, ended = [], 0, True, False
    for k, token in enumerate(prop, 1):
        idx = p + k - prompt_len
        can = not ended and 0 <= idx < n_ans and (eos_pos is None or idx <= eos_pos)
        valid.append(bool(can))
        if accepting and can and token == verifier[idx]:
            n_acc += 1
        else:
            accepting = False
        if can and token in eos:
            ended = True
    return n_acc, valid


def teacher_proposal_logits(target, prep, p, prop, last, dev):
    """Row k-1 predicts p+k; consume root then proposal[:last-1]."""
    tokens = [int(prep["full"][p])] + list(prop[:last-1])
    positions = torch.arange(p, p+last, device=dev)[None]
    with torch.no_grad():
        out = target.model(torch.tensor([tokens], device=dev),
            past_key_values=_cache_from_prefix(prep["prefix_kv"], p),
            use_cache=True, position_ids=prep["vispos"](positions),
            cache_position=positions[0])
        logits = target.lm_head(out.last_hidden_state[0].to(target.lm_head.weight.dtype)).float()
    if logits.shape[0] != last:
        raise RuntimeError("teacher row count differs from prediction horizon")
    return logits


def _anchor_loss(head, target, prep, plan, gamma, dev, block, mask_id, embed,
                 n_acc_total, z_total, acc_present, rej_present, head_dtype):
    """One anchor's scaled contribution and its log record."""
    p = int(plan["pos"])
    prop = [int(t) for t in plan["prop"]]
    acc_idx, rej_idx = plan["acc_idx"], plan["rej_idx"]
    full, prompt_len = prep["full"], prep["prompt_len"]
    fused = prep["fused"]
    q_logits = draft_block_logits_mixed(
        head, target, fused[:, :p], anchor_pos=p,
        root_token=int(full[p]), head_dtype=head_dtype)
    q_ids = [int(t) for t in q_logits.argmax(-1)]
    if q_ids != prop:
        raise RuntimeError(f"anchor {p}: the gradient pass's proposal differs "
                           f"from the planning pass's; refusing to score an "
                           f"anchor whose labels moved between the two passes")
    terms = []
    stats = {"pos": p, "video": prep["slot"]["video_id"], "n_acc": int(plan["n_acc"]),
             "n_acc_sup": int(acc_idx.numel()), "n_rej_sup": int(rej_idx.numel()),
             "teacher_forwards": 0, "acc_kl_sum": 0.0, "rej_kl_sum": 0.0}
    if acc_present and int(acc_idx.numel()):
        # `KL(onehot(t) || q)` is exactly `-log q[t]`; no (V,) one-hot is built.
        tgt_tok = torch.tensor([int(full[p + int(k)]) for k in acc_idx.tolist()],
                               device=dev)
        logq = torch.log_softmax(q_logits[acc_idx - 1].float(), dim=-1)
        acc_kl = -logq.gather(1, tgt_tok[:, None])[:, 0]
        terms.append(acc_kl.sum() / float(n_acc_total))
        stats["acc_kl_sum"] = float(acc_kl.detach().sum())
    valid_offsets = acc_idx.tolist() + rej_idx.tolist()
    if valid_offsets:
        last = max(valid_offsets)
        p_logits_full = teacher_proposal_logits(target, prep, p, prop, last, dev)
        online_n = 0
        for j, token in enumerate(p_logits_full.argmax(-1).tolist()):
            if not plan["valid"][j] or token != prop[j]:
                break
            online_n += 1
        if online_n != int(plan["n_acc"]):
            raise RuntimeError(f"anchor {p}: online target acceptance {online_n} != planned {plan['n_acc']}")
        stats.update(teacher_forwards=1, teacher_input_tokens=last,
                     teacher_argmax=p_logits_full.argmax(-1).tolist(),
                     teacher_prediction_positions=list(range(p+1,p+last+1)),
                     proposal=prop, valid=list(plan["valid"]), online_n_acc=online_n)
    if rej_present and int(rej_idx.numel()):
        n_rej_sup = int(rej_idx.numel())
        stats["teacher_forwards"] = 1
        rel = (rej_idx - 1).tolist()
        p_rej = p_logits_full[rel]
        q_rej = q_logits[rej_idx - 1]
        rej_kl = opd_fullkl.full_kl_reverse(q_rej, p_rej)
        w = gamma_weights(rej_idx, gamma, dev)
        terms.append((w * rej_kl).sum() / z_total)
        stats["rej_kl_sum"] = float(rej_kl.detach().sum())
        stats["teacher_logit_gap"] = float(
            (p_logits_full[rel].argmax(-1) == torch.tensor(
                [prop[int(k) - 1] for k in rej_idx.tolist()], device=dev)
             ).float().mean()) if n_rej_sup else 0.0
    if not terms:
        return None, stats
    loss = sum(terms) / (2.0 if (acc_present and rej_present) else 1.0)
    return loss, stats


def _cache_from_prefix(prefix_kv, n):
    """A `DynamicCache` holding the first `n` rows of a shared prefix cache."""
    from transformers import DynamicCache
    data = []
    for layer in prefix_kv.layers:
        data.append((layer.keys[..., :n, :], layer.values[..., :n, :]))
    return DynamicCache(ddp_cache_data=data)


if __name__ == "__main__":
    main()
