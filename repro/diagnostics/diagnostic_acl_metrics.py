"""CPU-only diagnostics. No model loading or parameter updates.

Lengths count draft tokens, excluding the known root and bonus. The existing
GLANCE builder's budget counts candidate nodes; the packed verifier costs one
additional root. Oracles describe an individual, already reached anchor only.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import hashlib
import json
import numpy as np


def part_of(offset, n_acc, valid=True):
    if not valid:
        return None
    if offset < 1:
        raise ValueError("offset must exclude root (start at 1)")
    return "accepted" if offset <= n_acc else (
        "first_rejected" if offset == n_acc + 1 else "strict_suffix")


def paired(a, b, seed=20261012, n_boot=2000):
    """b-a, resampling videos, never offsets or anchors."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0, "mean": None, "ci95": None, "values": []}
    d = np.array([b[k] - a[k] for k in ids], dtype=float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"n": len(ids), "mean": float(d.mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist() if len(d) > 1 else None,
            "positive": int((d > 0).sum()), "negative": int((d < 0).sum()),
            "changed": int((d != 0).sum()),
            "values": [{"video": k, "delta": float(v)} for k, v in zip(ids, d)]}


def native_objective(anchors, gamma=.8):
    """Original two-stream normalisation over this video's frozen anchors.

    This is a counterfactual objective on the diagnostic anchors, not a logged
    training update, and not C1's variable-reveal CE objective.
    """
    na, sr, sa, z, nr = 0, 0., 0., 0., 0
    for a in anchors:
        for i, valid in enumerate(a["valid"]):
            if not valid:
                continue
            if i < a["n_acc_anchor"]:
                sa += a["fkl"][i]
                na += 1
            else:
                w = float(np.float32(gamma ** i))
                sr += w * a["rkl"][i]
                z += w
                nr += 1
    terms = ([sa / na] if na else []) + ([sr / z] if z else [])
    return {"loss": float(np.mean(terms)) if terms else 0., "accepted_count": na,
            "rejected_count": nr, "accepted_weight_sum": na, "rejected_weight_sum": z,
            "accepted_fkl": sa / na if na else None,
            "rejected_weighted_rkl": sr / z if z else None}


def distribution_metrics(p_logits, q_logits):
    """Full vocabulary fp32 KL/JS, actual torch.topk membership (ties explicit)."""
    import torch
    p, q = p_logits.float().log_softmax(-1), q_logits.float().log_softmax(-1)
    m = torch.logaddexp(p, q) - np.log(2.)
    arg = p.argmax(-1)
    at = q.gather(-1, arg[..., None]).squeeze(-1)
    tq = q.topk(min(32, q.shape[-1]), dim=-1).indices
    tp = p.topk(min(2, p.shape[-1]), dim=-1).values.exp()
    return {"fkl": (p.exp() * (p-q)).sum(-1),
            "rkl": (q.exp() * (q-p)).sum(-1),
            "js": .5 * ((p.exp() * (p-m)).sum(-1) + (q.exp() * (q-m)).sum(-1)),
            "rank": (q > at[..., None]).sum(-1) + 1,
            "target_logp_in_draft": at, "target_argmax": arg,
            "target_entropy": -(p.exp()*p).sum(-1),
            "draft_entropy": -(q.exp()*q).sum(-1),
            "target_margin": tp[..., 0] - tp[..., 1],
            "top1": q.argmax(-1) == arg,
            "top8": (tq[..., :8] == arg[..., None]).any(-1),
            "top32": (tq == arg[..., None]).any(-1)}


def condition_key(*, content_sha256, frame_indices, timestamps, processor, intervention):
    """New intervention caches must key pixels AND time/processor metadata."""
    fields = dict(content_sha256=content_sha256, frame_indices=frame_indices,
                  timestamps=timestamps, processor=processor, intervention=intervention)
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def local_oracles(top_ids, nodes, continuation, actual_tokens, *, budget,
                  remaining_after_root, root_token, eos_id, reference_verified):
    """Validate and score a trace from the independent-offset GLANCE builder.

    continuation must come from exact incremental AR at this anchor, not a
    teacher-forced surrogate. actual_tokens are accepted path tokens, no bonus.
    Terminal clipping is applied to all four lengths. budget excludes root.
    """
    if not reference_verified:
        raise ValueError("unverified AR reference cannot diagnose verification/commit")
    if budget < 0 or remaining_after_root < 0 or len(nodes) > budget:
        raise ValueError("invalid budget/remaining length")
    for i, node in enumerate(nodes):
        parent, depth = node["parent"], node["depth"]
        if parent < -1 or parent >= i:
            raise ValueError("tree is not prefix closed / topologically ordered")
        expect = 1 if parent == -1 else nodes[parent]["depth"] + 1
        if depth != expect or depth > len(top_ids) or node["token"] not in top_ids[depth-1]:
            raise ValueError("tree node outside original candidate set")
    cap = min(len(top_ids), remaining_after_root)
    if root_token == eos_id:
        cap = 0
    ref = list(continuation[:cap])
    if eos_id in ref:
        ref = ref[:ref.index(eos_id)+1]
        cap = len(ref)
    if len(ref) < cap:
        raise ValueError("AR continuation too short to determine raw coverage")
    raw = 0
    for i, token in enumerate(ref):
        if token not in top_ids[i]:
            break
        raw += 1
    candidate = min(raw, budget)
    cursors, tree = {-1}, 0
    for token in ref:
        cursors = {i for i, n in enumerate(nodes)
                   if n["parent"] in cursors and n["token"] == token}
        if not cursors:
            break
        tree += 1
    actual = list(actual_tokens[:cap])
    wrong = actual != ref[:len(actual)]
    if wrong or not len(actual) <= tree <= candidate <= raw:
        reason = "verification_commit_inconsistency_pending"
    elif len(actual) < tree:
        reason = "verification_commit_inconsistency_pending"
    elif len(actual) == cap:
        reason = "normal_termination"
    elif len(actual) == raw:
        reason = "raw_candidate_missing"
    elif len(actual) == candidate and candidate < raw:
        reason = "budget_constraint"
    else:
        reason = "tree_selection_loss"
    return {"L_raw": raw, "L_candidate_oracle": candidate,
            "L_tree_oracle": tree, "L_actual": len(actual),
            "reason": reason, "root_cost": 1, "candidate_budget": budget,
            "verify_nodes": len(nodes)+1, "terminal_cap": cap,
            "raw_to_tree_gap": raw-tree, "tree_to_actual_gap": tree-len(actual)}
