#!/usr/bin/env python3
"""Semantics-pinning tests for `opd_fullkl.py` (Draft-OPD, Eq. 8-13).

Every CPU test runs on tiny tensors (V=7, K=4) so the objective's semantics are
verified by arithmetic, not by a model: the hand-computed KL, the asymmetry, the
`w_k = gamma**(k-1)` schedule, the two denominators, the stream combination, the
empty-stream renormalisation, the `valid` mask, and the gradient contract. None
of them needs a GPU or the 8B target.

The one GPU test (`--gpu-smoke`) is deliberately not a re-test of the arithmetic:
it exists for the two things only the real model can settle.

  1. FORK PURITY -- `teacher_block_logits` must not pollute the shared prefix KV.
     Asserted directly: the shared cache's sequence length is unchanged after the
     call (and after a second call).
  2. TEACHER CORRECTNESS -- the incrementally-scored rows must match one
     independent full forward over `prefix_ids + prop`, and the M-RoPE position
     claims must reproduce `get_rope_index`'s. Bit-equality is NOT asserted:
     bf16 GEMM row-count noise is real and was measured at ~0.375 max abs logit
     in bf16 and ~3.4e-05 in fp32 (B0), so the bounds below are stated per dtype
     and the measured numbers are PRINTED, including the same-method
     `chunk=1` control that is the honest noise floor.

It also checks that the head-side interface agrees with the validated rollout
machinery (`opd_online_rollout.rollout` + `proto_fullkl_b0.replay_draft_logits`)
bit-for-bit on the first anchor, which is what pins `draft_block_logits`'s
position and row convention to the one the rollout actually proposed from.

No `try`/`except` beyond an explicit `except AssertionError` in the tiny runner,
so an unexpected error surfaces with its traceback instead of being scored FAIL.
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
import ast
import math
import os
import sys

import torch

sys.path.insert(0, str(_REPRO_ROOT))

from historical import opd_fullkl as M


# --------------------------------------------------------------------------
# The oracle: the objective written out as a literal loop, in float64
# --------------------------------------------------------------------------

def literal_loss(p_logits, q_logits, n_acc, gamma, lambda_acc=1.0,
                 lambda_rej=1.0, valid=None):
    """`opd_fullkl_loss` as a loop over offsets 1..K, in float64.

    This is the test's oracle, not a second implementation of the module: it uses
    `torch.log_softmax` on the whole row per offset and the paper's own indexing
    (`k` counts from 1), so any slip in the vectorised masks, the weight index or
    the denominators shows up as a disagreement.
    """
    K = int(p_logits.shape[0])
    acc, rej = [], []
    for k in range(1, K + 1):
        if valid is not None and not bool(valid[k - 1]):
            continue
        logp = torch.log_softmax(p_logits[k - 1].double(), dim=-1)
        logq = torch.log_softmax(q_logits[k - 1].double(), dim=-1)
        kl_fwd = float((logp.exp() * (logp - logq)).sum())
        kl_rev = float((logq.exp() * (logq - logp)).sum())
        if k <= n_acc:
            acc.append(kl_fwd)
        else:
            rej.append((k, kl_rev))
    l_acc = sum(acc) / len(acc) if acc else 0.0
    z = sum(gamma ** (k - 1) for k, _ in rej)
    l_rej = sum(gamma ** (k - 1) * v for k, v in rej) / z if rej else 0.0
    terms, dens = [], 0.0
    if acc:
        terms.append(lambda_acc * l_acc)
        dens += lambda_acc
    if rej:
        terms.append(lambda_rej * l_rej)
        dens += lambda_rej
    return {"loss": sum(terms) / dens if terms else 0.0, "l_acc": l_acc,
            "l_rej": l_rej, "sum_w_rej": z, "n_acc_used": len(acc),
            "n_rej_used": len(rej)}


def _rand(K=4, V=7, seed=0, scale=1.3):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(K, V, generator=gen) * scale,
            torch.randn(K, V, generator=gen) * scale)


def _close(a, b, atol, what):
    assert abs(a - b) <= atol, f"{what}: module {a!r} vs oracle {b!r} (atol {atol})"


# --------------------------------------------------------------------------
# 1./2. KL: hand-computed values, and the asymmetry is real
# --------------------------------------------------------------------------

def t01_hand_kl():
    p, q = _rand(seed=1)
    logp = torch.log_softmax(p.double(), dim=-1)
    logq = torch.log_softmax(q.double(), dim=-1)
    exp_f = (logp.exp() * (logp - logq)).sum(-1)
    exp_r = (logq.exp() * (logq - logp)).sum(-1)

    got_f = M.full_kl_forward(p, q)
    got_r = M.full_kl_reverse(q, p)
    assert got_f.shape == (4,), got_f.shape
    assert torch.allclose(got_f.double(), exp_f, atol=1e-6), (got_f, exp_f)
    assert torch.allclose(got_r.double(), exp_r, atol=1e-6), (got_r, exp_r)
    # The task's own claim: in general KL(p||q) != KL(q||p).
    assert not torch.allclose(exp_f, exp_r, atol=1e-6), (exp_f, exp_r)

    # (..., V) in, (...) out: the per-row softmax is independent, so batching is
    # just a shape change and must agree row for row.
    pa, _ = _rand(2, 7, seed=2)
    _, qa = _rand(2, 7, seed=3)
    pb = pa.reshape(2, 1, 7).expand(2, 3, 7).contiguous()
    qb = qa.reshape(2, 1, 7).expand(2, 3, 7).contiguous()
    batched = M.full_kl_forward(pb, qb)
    assert batched.shape == (2, 3), batched.shape
    flat = M.full_kl_forward(pb.reshape(-1, 7), qb.reshape(-1, 7)).reshape(2, 3)
    assert torch.equal(batched, flat)
    print(f"      KL_fwd {[round(float(x), 4) for x in exp_f]}  "
          f"KL_rev {[round(float(x), 4) for x in exp_r]}")


def t02_asymmetry_nonnegative():
    p, q = _rand(seed=5)
    fwd = M.full_kl_forward(p, q)
    rev = M.full_kl_reverse(q, p)
    for name, v in (("forward", fwd), ("reverse", rev)):
        assert float(v.min()) >= -1e-6, f"{name} KL negative: {v}"
    # Same direction twice (KL(p||q) vs KL(p||q)) must agree; opposite directions
    # must differ where the distributions do.
    assert torch.allclose(fwd, M.full_kl_forward(p, q), atol=0.0)
    assert float((fwd - rev).abs().max()) > 1e-3, (fwd, rev)
    # An identical pair has zero KL in both directions.
    assert float(M.full_kl_forward(p, p).abs().max()) < 1e-6
    assert float(M.full_kl_reverse(p, p).abs().max()) < 1e-6
    print(f"      max|KL_fwd - KL_rev| = {float((fwd - rev).abs().max()):.4f}")


def t02b_internal_fp32():
    """Rule 3: the softmax and the KL are fp32 even when the logits are bf16.

    The values below are exactly representable in bf16, so a cast to bf16 loses
    nothing and the two calls must agree to fp32 rounding. The naive path
    (`log_softmax` in bf16) is computed too, and asserted to be visibly different,
    otherwise this test would pass on an implementation that never casts.
    """
    p = torch.tensor([[0.0, 0.5, -1.0, 2.0, -0.5, 1.5, -2.0]])
    q = torch.tensor([[1.0, -1.5, 0.25, -0.5, 2.0, 0.0, 1.25]])
    ref_f = M.full_kl_forward(p, q)
    ref_r = M.full_kl_reverse(q, p)
    got_f = M.full_kl_forward(p.bfloat16(), q.bfloat16())
    got_r = M.full_kl_reverse(q.bfloat16(), p.bfloat16())
    assert float((ref_f - got_f).abs().max()) < 1e-6, (ref_f, got_f)
    assert float((ref_r - got_r).abs().max()) < 1e-6, (ref_r, got_r)

    logp = torch.log_softmax(p.bfloat16(), dim=-1).float()
    logq = torch.log_softmax(q.bfloat16(), dim=-1).float()
    naive = (logp.exp() * (logp - logq)).sum(-1)
    assert float((naive - ref_f).abs().max()) > 1e-4, (naive, ref_f)

    # The loss (not just the KL helper) must obey the same rule.
    loss32, _ = M.opd_fullkl_loss(p, q, 0)
    loss16, _ = M.opd_fullkl_loss(p.bfloat16(), q.bfloat16(), 0)
    assert abs(float(loss32) - float(loss16)) < 1e-6, (loss32, loss16)
    print(f"      fp32-internal: |loss(bf16 in) - loss(fp32 in)| = "
          f"{abs(float(loss32) - float(loss16)):.2e}; naive bf16 softmax differs "
          f"by {float((naive - ref_f).abs().max()):.2e}")


# --------------------------------------------------------------------------
# 3.7. gamma / the weight schedule
# --------------------------------------------------------------------------

def t03_gamma_weights():
    p, q = _rand(seed=7)
    K = 4
    kls = [float(M.full_kl_reverse(q[k], p[k])) for k in range(K)]
    for gamma in (0.8, 0.5, 0.99, 0.0):
        loss, m = M.opd_fullkl_loss(p, q, 0,
                                    config=M.FullKLLossConfig(gamma=gamma))
        w = [gamma ** k for k in range(K)]
        exp = sum(wi * ki for wi, ki in zip(w, kls)) / sum(w)
        _close(m["l_rej"], exp, 1e-5, f"l_rej(gamma={gamma})")
        _close(m["sum_w_rej"], sum(w), 1e-6, f"sum_w(gamma={gamma})")
        _close(float(loss), exp, 1e-5, f"loss(gamma={gamma})")  # rejected-only
    # A global rescaling of the weights cancels in the normalised mean, so the
    # SHIFT in `w_k = gamma**(k-1)` (as opposed to a `gamma**k` reading) is not
    # visible in the loss value -- it is visible in `sum_w_rej`, which is exactly
    # why the objective returns that denominator. Both facts are asserted.
    _, m = M.opd_fullkl_loss(p, q, 0)
    _close(m["sum_w_rej"], sum(0.8 ** k for k in range(K)), 1e-6,
           "Z pins the gamma**(k-1) shift")
    shifted = sum(0.8 ** (k + 1) for k in range(K))
    assert abs(m["sum_w_rej"] - shifted) > 1e-3, (m["sum_w_rej"], shifted)
    # Two rejected offsets pin the ratio w_2/w_1 = gamma, which is the only other
    # observable content of the schedule (offsets 3 and 4 here, so a shift of the
    # whole schedule would leave this number unchanged -- see the check above).
    _, m2 = M.opd_fullkl_loss(p, q, 2)
    kl1 = float(M.full_kl_reverse(q[2], p[2]))
    kl2 = float(M.full_kl_reverse(q[3], p[3]))
    _close(m2["l_rej"], (kl1 + 0.8 * kl2) / 1.8, 1e-5,
           "two rejected offsets pin w_2/w_1")
    print(f"      l_rej={m['l_rej']:.6f}  Z={m['sum_w_rej']:.6f} "
          f"(shifted reading would give {shifted:.6f})")


def t03b_gamma_direction():
    """Higher gamma puts more weight on later (larger-KL) offsets -> larger loss."""
    base = torch.randn(7, generator=torch.Generator().manual_seed(11))
    p = base.repeat(4, 1)
    q = base.repeat(4, 1) * torch.tensor([1.0, 2.0, 3.0, 4.0])[:, None]
    kls = [float(M.full_kl_reverse(q[k], p[k])) for k in range(4)]
    assert all(kls[i] < kls[i + 1] for i in range(3)), kls  # non-vacuous
    lo, _ = M.opd_fullkl_loss(p, q, 0, config=M.FullKLLossConfig(gamma=0.5))
    mid, _ = M.opd_fullkl_loss(p, q, 0, config=M.FullKLLossConfig(gamma=0.8))
    hi, _ = M.opd_fullkl_loss(p, q, 0, config=M.FullKLLossConfig(gamma=0.99))
    assert float(lo) < float(mid) < float(hi), (lo, mid, hi)
    print(f"      KL_rej by offset {[round(k, 3) for k in kls]} -> loss "
          f"{float(lo):.4f} < {float(mid):.4f} < {float(hi):.4f}")


def t07_gamma_only_on_rejected():
    p, q = _rand(seed=13)
    _, m50 = M.opd_fullkl_loss(p, q, 2, config=M.FullKLLossConfig(gamma=0.5))
    _, m80 = M.opd_fullkl_loss(p, q, 2, config=M.FullKLLossConfig(gamma=0.8))
    _, m99 = M.opd_fullkl_loss(p, q, 2, config=M.FullKLLossConfig(gamma=0.99))
    assert m50["l_acc"] == m80["l_acc"] == m99["l_acc"], (m50, m80, m99)
    assert m50["l_rej"] != m80["l_rej"] != m99["l_rej"], (m50, m80, m99)
    # Direction: with later offsets carrying larger reversed KL, a higher gamma
    # must put more weight on them and raise L_rej.
    base = torch.randn(7, generator=torch.Generator().manual_seed(53))
    p2 = base.repeat(4, 1)
    q2 = base.repeat(4, 1) * torch.tensor([1.0, 2.0, 3.0, 4.0])[:, None]
    l50, _ = M.opd_fullkl_loss(p2, q2, 0, config=M.FullKLLossConfig(gamma=0.5))
    l99, _ = M.opd_fullkl_loss(p2, q2, 0, config=M.FullKLLossConfig(gamma=0.99))
    assert float(l50) < float(l99), (l50, l99)
    # ... and the only-accepted case, where the loss IS L_acc, cannot move at all.
    p3, q3 = _rand(seed=14)
    l1, m1 = M.opd_fullkl_loss(p3, q3, 4, config=M.FullKLLossConfig(gamma=0.1))
    l2, m2 = M.opd_fullkl_loss(p3, q3, 4, config=M.FullKLLossConfig(gamma=0.9))
    assert m1["l_acc"] == m2["l_acc"] and float(l1) == float(l2)
    print(f"      l_acc frozen at {m80['l_acc']:.6f} for gamma in "
          f"{{0.5, 0.8, 0.99}}; rejected-only {float(l50):.4f} < "
          f"{float(l99):.4f}")


def t03c_accepted_has_no_position_weight():
    """The accepted stream is a PLAIN mean: permuting its offsets changes nothing.

    Swapping two accepted rows keeps the accepted SET and its multiset of KLs,
    so a plain mean is unchanged while any per-position weighting (`w_k` attached
    to offset k) changes. That difference is the discriminator.
    """
    p, q = _rand(seed=17, scale=2.0)
    perm = torch.tensor([1, 0, 2, 3])
    _, m_a = M.opd_fullkl_loss(p, q, 2)
    _, m_b = M.opd_fullkl_loss(p[perm], q[perm], 2)
    assert abs(m_a["l_acc"] - m_b["l_acc"]) < 1e-6, (m_a, m_b)
    # The two losses are not equal in general (the rejected rows moved too), but
    # l_acc is, and the position-weighted alternative is demonstrably different.
    acc_kls = [float(M.full_kl_forward(p[k], q[k])) for k in range(2)]
    weighted = (0.8 ** 0 * acc_kls[0] + 0.8 ** 1 * acc_kls[1]) / (1.0 + 0.8)
    assert abs(m_a["l_acc"] - weighted) > 1e-4, (m_a["l_acc"], weighted)
    # Doubling the accepted set changes L_acc only through the new KL values:
    # a plain mean over the 3 accepted offsets.
    _, m_c = M.opd_fullkl_loss(p, q, 3)
    want = sum(float(M.full_kl_forward(p[k], q[k])) for k in range(3)) / 3
    _close(m_c["l_acc"], want, 1e-5, "plain mean over 3 accepted offsets")
    print(f"      l_acc(perm)={m_b['l_acc']:.6f} == l_acc={m_a['l_acc']:.6f}; "
          f"weighted variant {weighted:.6f}")


# --------------------------------------------------------------------------
# 4./5. Denominators and combination, against the literal loop
# --------------------------------------------------------------------------

def t04_denominators_literal():
    p, q = _rand(seed=19)
    cases = [
        (0, None, 0.8),
        (1, None, 0.8),
        (2, None, 0.5),
        (2, None, 0.99),
        (4, None, 0.8),
        (2, [True, True, True, False], 0.9),
        (2, [True, False, True, True], 0.8),
        (3, [True, False, False, True], 0.8),
    ]
    for n_acc, valid, gamma in cases:
        cfg = M.FullKLLossConfig(gamma=gamma)
        loss, m = M.opd_fullkl_loss(p, q, n_acc, config=cfg, valid=valid)
        ref = literal_loss(p, q, n_acc, gamma, valid=valid)
        tag = f"n_acc={n_acc} valid={valid} gamma={gamma}"
        _close(float(loss), ref["loss"], 1e-5, f"loss {tag}")
        _close(m["l_acc"], ref["l_acc"], 1e-5, f"l_acc {tag}")
        _close(m["l_rej"], ref["l_rej"], 1e-5, f"l_rej {tag}")
        _close(m["sum_w_rej"], ref["sum_w_rej"], 1e-6, f"sum_w_rej {tag}")
        assert m["n_acc_used"] == ref["n_acc_used"], (m, ref)
        assert m["n_rej_used"] == ref["n_rej_used"], (m, ref)
    print(f"      {len(cases)} (n_acc, valid, gamma) cases match the literal loop")


def t05_combination():
    p, q = _rand(seed=23)
    loss, m = M.opd_fullkl_loss(p, q, 2)
    assert m["acc_present"] and m["rej_present"]
    _close(float(loss), (m["l_acc"] + m["l_rej"]) / 2.0, 1e-6, "both streams")
    _close(m["lambda_acc_used"], 1.0, 0.0, "lambda_acc_used")
    _close(m["lambda_rej_used"], 1.0, 0.0, "lambda_rej_used")
    # Unequal lambdas: the verified constants are 1/1, but the normalisation is
    # asserted too, since that is where the stream reweighting lives.
    cfg = M.FullKLLossConfig(gamma=0.8, lambda_acc=3.0, lambda_rej=1.0)
    loss2, m2 = M.opd_fullkl_loss(p, q, 2, config=cfg)
    _close(float(loss2), (3.0 * m2["l_acc"] + 1.0 * m2["l_rej"]) / 4.0, 1e-6,
           "lambda-weighted combination")
    print(f"      l_acc={m['l_acc']:.4f} l_rej={m['l_rej']:.4f} "
          f"loss={float(loss):.4f}")


# --------------------------------------------------------------------------
# 6. Empty streams
# --------------------------------------------------------------------------

def t06_empty_streams():
    p, q = _rand(seed=29)
    K = 4
    # Only accepted: L == L_acc exactly, NOT L_acc/2.
    loss_a, m_a = M.opd_fullkl_loss(p, q, K)
    assert m_a["acc_present"] and not m_a["rej_present"]
    assert m_a["streams_present"] is True
    assert m_a["n_rej_used"] == 0 and m_a["sum_w_rej"] == 0.0
    assert abs(float(loss_a) - m_a["l_acc"]) < 1e-9, (loss_a, m_a)
    assert abs(float(loss_a) - m_a["l_acc"] / 2.0) > 1e-6
    # Only rejected: L == L_rej exactly, NOT L_rej/2.
    loss_r, m_r = M.opd_fullkl_loss(p, q, 0)
    assert m_r["rej_present"] and not m_r["acc_present"]
    assert m_r["streams_present"] is True
    assert m_r["n_acc_used"] == 0 and m_r["l_acc"] == 0.0
    assert abs(float(loss_r) - m_r["l_rej"]) < 1e-9, (loss_r, m_r)
    assert abs(float(loss_r) - m_r["l_rej"] / 2.0) > 1e-6
    # Both empty: a zero loss, and the caller can see it in the flags/counts.
    loss_e, m_e = M.opd_fullkl_loss(p, q, 2, valid=[False, False, False, False])
    assert float(loss_e) == 0.0, float(loss_e)
    assert m_e["streams_present"] is False
    assert not m_e["acc_present"] and not m_e["rej_present"]
    assert m_e["n_acc_used"] == 0 and m_e["n_rej_used"] == 0
    assert m_e["sum_w_rej"] == 0.0 and m_e["l_acc"] == 0.0 and m_e["l_rej"] == 0.0
    print(f"      acc-only loss={float(loss_a):.4f} (l_acc={m_a['l_acc']:.4f}); "
          f"rej-only loss={float(loss_r):.4f} (l_rej={m_r['l_rej']:.4f}); "
          f"both-empty loss={float(loss_e)}")


# --------------------------------------------------------------------------
# 8. The valid mask
# --------------------------------------------------------------------------

def t08_valid_mask():
    p, q = _rand(seed=31)
    valid = [True, True, True, False]
    loss, m = M.opd_fullkl_loss(p, q, 2, valid=valid)
    ref = literal_loss(p, q, 2, 0.8, valid=valid)
    _close(float(loss), ref["loss"], 1e-5, "masked loss")
    assert (m["n_acc_used"], m["n_rej_used"]) == (2, 1), m
    # Row 4 is invalid and row 2 is accepted: nothing about it may reach the loss.
    p2, q2 = p.clone(), q.clone()
    p2[3] = 1e4
    q2[3] = -1e4
    loss2, m2 = M.opd_fullkl_loss(p2, q2, 2, valid=valid)
    assert float(loss2) == float(loss), (loss2, loss)
    assert (m2["n_acc_used"], m2["n_rej_used"]) == (2, 1)
    # An invalid offset inside the accepted range leaves the denominator alone
    # (n_acc_used 1, not 2) and is not silently moved into the rejected stream.
    valid2 = [True, False, True, True]
    loss3, m3 = M.opd_fullkl_loss(p, q, 2, valid=valid2)
    ref3 = literal_loss(p, q, 2, 0.8, valid=valid2)
    _close(float(loss3), ref3["loss"], 1e-5, "masked-accepted loss")
    assert (m3["n_acc_used"], m3["n_rej_used"]) == (1, 2), m3
    _close(m3["sum_w_rej"], _sum_w_rej(valid2, 2, 0.8), 1e-6, "masked Z")
    # Truncation at the END is the common real case: it must shrink both the
    # rejected set and Z, not leave a phantom offset in the denominator.
    valid4 = [True, True, False, False]
    _, m4 = M.opd_fullkl_loss(p, q, 2, valid=valid4)
    assert (m4["n_acc_used"], m4["n_rej_used"]) == (2, 0), m4
    assert m4["sum_w_rej"] == 0.0 and m4["rej_present"] is False
    print(f"      mask [T,T,T,F] -> acc {m['n_acc_used']} rej {m['n_rej_used']}; "
          f"mask [T,F,T,T] -> acc {m3['n_acc_used']} rej {m3['n_rej_used']}")


def _sum_w_rej(valid, n_acc, gamma):
    """`Z = sum w_k` over the surviving rejected offsets, from the definition."""
    return sum(gamma ** (k - 1) for k in range(1, len(valid) + 1)
               if valid[k - 1] and k > n_acc)


# --------------------------------------------------------------------------
# 9. Gradients
# --------------------------------------------------------------------------

def t09_gradient():
    p, q = _rand(seed=37)
    p = p.detach()
    assert not p.requires_grad
    q = q.clone().requires_grad_(True)
    loss, m = M.opd_fullkl_loss(p, q, 1)
    assert loss.requires_grad, "the loss must stay on the graph"
    loss.backward()
    assert p.grad is None, "the teacher must not receive a gradient"
    assert q.grad is not None
    assert torch.isfinite(q.grad).all(), q.grad
    assert float(q.grad.abs().sum()) > 0.0, "no gradient reached the student"
    # Offsets that no stream used must not move the student.
    valid = [True, True, False, False]
    q2 = _rand(seed=41)[1].clone().requires_grad_(True)
    loss2, _ = M.opd_fullkl_loss(p, q2, 1, valid=valid)
    loss2.backward()
    assert float(q2.grad[2].abs().sum()) == 0.0
    assert float(q2.grad[3].abs().sum()) == 0.0
    assert float(q2.grad[1].abs().sum()) > 0.0
    # The both-empty block still backpropagates (a zero gradient, not a crash).
    q3 = _rand(seed=43)[1].clone().requires_grad_(True)
    loss3, m3 = M.opd_fullkl_loss(p, q3, 1, valid=[False] * 4)
    assert float(loss3.detach()) == 0.0 and m3["streams_present"] is False
    loss3.backward()
    assert q3.grad is not None and float(q3.grad.abs().sum()) == 0.0
    # A finite-difference check on one coordinate: the analytic gradient of the
    # loss w.r.t. a student logit must match d(loss)/d(logit) numerically.
    idx, eps = (0, 3), 1e-3
    q4 = _rand(seed=47)[1].clone()
    def at(delta):
        qq = q4.clone()
        qq[idx] += delta
        return float(M.opd_fullkl_loss(p, qq, 1)[0])
    num = (at(eps) - at(-eps)) / (2 * eps)
    q5 = q4.clone().requires_grad_(True)
    M.opd_fullkl_loss(p, q5, 1)[0].backward()
    ana = float(q5.grad[idx])
    assert abs(num - ana) < 2e-4, (num, ana)
    print(f"      |grad|={float(q.grad.abs().sum()):.4e}  "
          f"d(loss)/d(logit) analytic {ana:.6f} vs numeric {num:.6f}")


# --------------------------------------------------------------------------
# 10. Gradient cosine / norm ratio
# --------------------------------------------------------------------------

def t10_grad_cosine_and_norm_ratio():
    a = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "b": torch.tensor([5.0, 6.0])}
    same = M.grad_cosine_and_norm_ratio(a, a)
    assert same["cosine"] == 1.0, same
    assert abs(same["norm_ratio"] - 1.0) < 1e-12, same
    assert same["n_tensors"] == 2, same
    neg = M.grad_cosine_and_norm_ratio(a, {k: -v for k, v in a.items()})
    assert neg["cosine"] == -1.0, neg
    orth = M.grad_cosine_and_norm_ratio(
        {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])},
        {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1.0, 0.0])})
    assert abs(orth["cosine"]) < 1e-12, orth
    # Scaling is orientation-free: cosine stays 1, the norm ratio carries the factor.
    scaled = M.grad_cosine_and_norm_ratio(a, {k: 2.0 * v for k, v in a.items()})
    assert scaled["cosine"] == 1.0, scaled
    assert abs(scaled["norm_ratio"] - 0.5) < 1e-12, scaled
    # None grads are skipped; a key missing on one side counts only for that norm.
    part = M.grad_cosine_and_norm_ratio({"a": torch.tensor([3.0])}, {})
    assert part["n_tensors"] == 1 and part["cosine"] == 0.0
    assert part["norm_b"] == 0.0 and math.isinf(part["norm_ratio"])
    assert abs(part["norm_a"] - 3.0) < 1e-12
    print(f"      identical {same['cosine']}/{same['norm_ratio']}, negated "
          f"{neg['cosine']}, orthogonal {orth['cosine']:.1e}")


# --------------------------------------------------------------------------
# 11a./11b. The two model-facing index conventions, pinned without a model
# --------------------------------------------------------------------------
#
# These two tests exist because the GPU smoke cannot afford to be the only
# place the offset arithmetic is exercised: a chunked loop that ran one row past
# the end of the block was found here, not on the GPU. A stub target whose hidden
# rows stamp their own stream position into the logits makes every index
# observable on the CPU.

class _StubCache:
    """`get_seq_length` is all `teacher_block_logits` asks of the cache."""

    def __init__(self, length):
        self._length = int(length)

    def get_seq_length(self):
        return self._length


class _StubPositions:
    """A bound `VisionPositions` stand-in whose `pos(index)` is the plain index."""

    def __init__(self, n):
        self._prefill_positions = torch.zeros(3, 1, int(n))

    def __call__(self, positions):
        return positions.unsqueeze(0).expand(3, -1, -1)


class _StubLmHead:
    """Channel 0 of the hidden row becomes channel 0 of the logits."""

    def __init__(self, vocab=5):
        self.vocab = int(vocab)

    def __call__(self, hidden):
        out = torch.zeros(*hidden.shape[:-1], self.vocab)
        out[..., 0] = hidden[..., 0]
        return out


class _StubOutput:
    def __init__(self, last_hidden_state, past_key_values):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values


class _StubInner:
    """`target.model`: each hidden row's channel 0 is that row's stream position.

    This mirrors what the real model guarantees for this test's purpose -- the
    hidden state at stream position `q` is the input for `lm_head` -- without
    being a model at all.
    """

    def __init__(self):
        self.seen = []            # one list of stream positions per call

    def __call__(self, tokens, position_ids=None, past_key_values=None,
                 use_cache=True, pixel_values_videos=None, video_grid_thw=None):
        pos = [int(v) for v in position_ids[0, 0]]
        assert len(pos) == int(tokens.shape[1])
        self.seen.append(pos)
        hidden = torch.zeros(1, len(pos), 3)
        hidden[0, :, 0] = torch.tensor(pos, dtype=torch.float32)
        return _StubOutput(hidden, past_key_values)


class _StubTeacherTarget:
    def __init__(self, vocab=5):
        self.model = _StubInner()
        self.lm_head = _StubLmHead(vocab)


class _StubBlockHead:
    """A block head whose returned hidden rows carry their own stream positions."""

    def __init__(self, block=16, mask_token_id=99):
        self.block_size = block
        self.mask_token_id = mask_token_id
        self.calls = []

    def __call__(self, target_hidden=None, noise_embedding=None, position_ids=None,
                 past_key_values=None, use_cache=True, is_causal=False):
        q = int(noise_embedding.shape[1])
        assert int(position_ids.shape[1]) == int(target_hidden.shape[1]) + q, (
            "the head's positions must cover the context keys plus the block, "
            "which is how the released forward applies RoPE to cat(ctx, noise)")
        self.calls.append({
            "ctx_rows": int(target_hidden.shape[1]),
            "noise_rows": q,
            "pos": [int(v) for v in position_ids[0]],
            "cache_len": int(past_key_values.get_seq_length())
            if past_key_values is not None else None,
            "is_causal": is_causal,
        })
        hidden = torch.zeros(1, q, 3)
        hidden[0, :, 0] = position_ids[0, -q:].float()   # q rows use the last q
        return hidden


class _StubDraftTarget:
    def __init__(self, vocab=5):
        self.lm_head = _StubLmHead(vocab)

    def get_input_embeddings(self):
        return lambda ids: torch.zeros(*ids.shape, 3)


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{exc.__name__} was not raised")


def t11a_teacher_row_mapping_stub():
    """Teacher row k-1 is the distribution at stream position anchor_pos+k-1.

    The `off -> pos + off - 1` rule is the single subtraction this project has
    already shipped one convention bug in (`opd_loss.teacher_rows`), so it is
    pinned here for the chunked full-vocabulary path too: the appended stream
    must be exactly `prop[:K-1]` from `anchor_pos+1`, and row k-1 must be the row
    at `anchor_pos+k-1` -- including offset 1, which comes from the carried
    prefix row rather than from a re-fed token.
    """
    V, p, K = 5, 10, 15
    prop = list(range(100, 100 + K))
    target = _StubTeacherTarget(V)
    pos = _StubPositions(p + 1)
    cache = _StubCache(p + 1)
    cache.opd_prefix_last_hidden = torch.tensor([[[float(p), 0.0, 0.0]]])
    want = torch.tensor([float(p + k - 1) for k in range(1, K + 1)])
    for chunk in (1, 2, 3, 8, 15, 64):
        target.model.seen = []
        got = M.teacher_block_logits(target, pos, cache, anchor_pos=p, prop=prop,
                                     chunk=chunk)
        assert got.shape == (K, V), got.shape
        assert torch.allclose(got[:, 0], want), (chunk, got[:, 0], want)
        feeds = [x for call in target.model.seen for x in call]
        assert feeds == list(range(p + 1, p + K)), (chunk, feeds)
        assert len(target.model.seen) == math.ceil((K - 1) / chunk), chunk
    # K = 1: the only row is the anchor's own row, and nothing is appended.
    target.model.seen = []
    one = M.teacher_block_logits(target, pos, cache, anchor_pos=p, prop=[7], chunk=8)
    assert one.shape == (1, V) and float(one[0, 0]) == p
    assert target.model.seen == []
    # The cache is a fork source, never a target.
    assert cache.get_seq_length() == p + 1
    # Errors, not silent guesses: an empty block, a cache that does not end at
    # the anchor, and a cache with no carried prefix row.
    _raises(ValueError, lambda: M.teacher_block_logits(
        target, pos, cache, anchor_pos=p, prop=[], chunk=8))
    _raises(ValueError, lambda: M.teacher_block_logits(
        target, pos, _StubCache(p), anchor_pos=p, prop=[1, 2, 3], chunk=8))
    _raises(ValueError, lambda: M.teacher_block_logits(
        target, pos, _StubCache(p + 1), anchor_pos=p, prop=[1, 2, 3], chunk=8))
    _raises(ValueError, lambda: M.teacher_block_logits(
        target, pos, cache, anchor_pos=p, prop=prop, chunk=0))
    print(f"      rows 1..{K} <- positions {p}..{p + K - 1}; appended stream "
          f"{p + 1}..{p + K - 1} for chunk in (1,2,3,8,15,64)")


def t11b_draft_block_convention_stub():
    """`draft_block_logits` reproduces rollout's block layout and positions.

    The block rows are the LAST `block-1` rows of the head's output -- the same
    slice `opd_online_rollout.rollout` proposes from -- and the head's hidden row
    for offset k sits at stream position `anchor_pos+k` (the block's slots are
    self-predicting), which is why the logits row k-1 carries position
    `anchor_pos+k`: the two rows of the KL describe the SAME target token, the
    draft's as a parallel block marginal and the teacher's as the autoregressive
    conditional on the draft-induced prefix.
    """
    V, block, p = 5, 16, 7
    head = _StubBlockHead(block=block)
    target = _StubDraftTarget(V)
    ctx = torch.zeros(1, p + 3, 3)
    got = M.draft_block_logits(head, target, ctx[:, :p], anchor_pos=p,
                               root_token=42, prompt_len=1)
    assert got.shape == (block - 1, V), got.shape
    want = torch.arange(p + 1, p + block, dtype=torch.float32)
    assert torch.allclose(got[:, 0], want), (got[:, 0], want)
    call = head.calls[-1]
    assert call["ctx_rows"] == p and call["noise_rows"] == block
    assert call["pos"] == list(range(0, p + block)), call
    assert call["is_causal"] is False and call["cache_len"] == 0
    # The carried-cache mode consumes only the newly revealed context rows and
    # starts the positions where the rollout's cache left them.
    head.calls = []
    q_rows = 2
    got2 = M.draft_block_logits(head, target, ctx[:, p:p + q_rows],
                                anchor_pos=p + q_rows, root_token=42,
                                cache=_StubCache(p))
    want2 = torch.arange(p + q_rows + 1, p + q_rows + block, dtype=torch.float32)
    assert torch.allclose(got2[:, 0], want2), (got2[:, 0], want2)
    call = head.calls[-1]
    assert call["ctx_rows"] == q_rows and call["cache_len"] == p, call
    assert call["pos"] == list(range(p, p + q_rows + block)), call
    # Refusals: the wrong number of context rows, and an anchor inside the prompt.
    _raises(ValueError, lambda: M.draft_block_logits(
        head, target, ctx[:, :p + 1], anchor_pos=p, root_token=1))
    _raises(ValueError, lambda: M.draft_block_logits(
        head, target, ctx[:, :p], anchor_pos=p, root_token=1, prompt_len=p + 1))
    print(f"      {block - 1} rows, positions {p + 1}..{p + block - 1} (fresh) / "
          f"{p + 3}..{p + q_rows + block - 1} (cache at {p})")


def t12_stream_split_matches_loss():
    """`stream_split` is the documented route for gradient-carrying pooling.

    It must name exactly the offsets `opd_fullkl_loss` split the block into, and
    rebuilding both stream means from the per-offset KLs through it must
    reproduce the loss's own `l_acc`/`l_rej`. That is what lets a trainer pool
    several anchors' offsets at update level instead of averaging per-block
    means of unequal length.
    """
    p, q = _rand(seed=59)
    K = 4
    cases = [(0, None, 0.8), (2, None, 0.8), (4, None, 0.8), (1, None, 0.5),
             (2, [True, False, True, True], 0.8)]
    for n_acc, valid, gamma in cases:
        _, m = M.opd_fullkl_loss(p, q, n_acc,
                                 config=M.FullKLLossConfig(gamma=gamma),
                                 valid=valid)
        i_acc, i_rej = M.stream_split(K, n_acc, valid)
        want_acc = [k for k in range(1, K + 1)
                    if (valid is None or valid[k - 1]) and k <= n_acc]
        want_rej = [k for k in range(1, K + 1)
                    if (valid is None or valid[k - 1]) and k > n_acc]
        assert i_acc.tolist() == want_acc, (i_acc.tolist(), want_acc)
        assert i_rej.tolist() == want_rej, (i_rej.tolist(), want_rej)
        if i_acc.numel():
            got = M.full_kl_forward(p[i_acc - 1], q[i_acc - 1]).mean()
            assert abs(float(got) - m["l_acc"]) < 1e-6, (float(got), m["l_acc"])
        if i_rej.numel():
            w = torch.tensor([gamma ** (int(k) - 1) for k in i_rej])
            got = (w * M.full_kl_reverse(q[i_rej - 1], p[i_rej - 1])).sum() / w.sum()
            assert abs(float(got) - m["l_rej"]) < 1e-6, (float(got), m["l_rej"])
    _raises(ValueError, lambda: M.stream_split(4, 5))
    _raises(ValueError, lambda: M.stream_split(4, 1, [True, False]))
    print("      split reproduces l_acc/l_rej on 5 (n_acc, valid) cases")


# --------------------------------------------------------------------------
# The official k3 rejected-stream estimator
# --------------------------------------------------------------------------

#: The checked-out official source the comparison is made against. AGENTS.md
#: names `run_qwen_gsm8k.sh` (`d1476f5`, 2026-08-08) as the KL-version baseline
#: and `core_algos.py:2183-2191` as the k3 body; `losses.py` imports the wrapper
#: `kl_penalty`, which forwards to `kl_penalty_forward` and, for a bare "k3",
#: returns that forward score unchanged.
OFFICIAL_CORE_ALGOS = os.environ.get(
    "DRAFTOPD_CORE_ALGOS",
    "/home/mcy/Projects/Draft-OPD/refs/Draft-OPD/verl/verl/trainer/ppo/core_algos.py")


def _load_official_kl_penalty():
    """`kl_penalty`/`kl_penalty_forward` EXECUTED from the official source file.

    The functions are extracted by AST and exec'd in a namespace holding only
    `torch`, so the reference cannot silently drift from the code the official
    entry actually runs and cannot be re-typed from memory into this test.
    """
    assert os.path.exists(OFFICIAL_CORE_ALGOS), (
        f"official core_algos.py not found at {OFFICIAL_CORE_ALGOS}; set "
        "DRAFTOPD_CORE_ALGOS to compare against the official k3 implementation")
    with open(OFFICIAL_CORE_ALGOS, encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)
    lines = source.splitlines()
    namespace = {"torch": torch}
    wanted = {"kl_penalty", "kl_penalty_forward"}
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            segment = "\n".join(lines[node.lineno - 1:node.end_lineno])
            exec(compile(segment, OFFICIAL_CORE_ALGOS, "exec"), namespace)
            found.add(node.name)
    assert found == wanted, f"missing official functions: {wanted - found}"
    return namespace["kl_penalty"], namespace["kl_penalty_forward"]


def t13_k3_official_bitwise():
    """`k3_from_logprobs` is bit-identical to the official k3 path.

    Covered: a dense grid over `log p - log q` in [-30, 30], the exact clamp
    boundaries (`kl = +-20`, `kld` at `+-10`), `kl = 0` (`kld = 0`), and both
    sides of the negative-side `kld` crossing near `kl = -11` -- the region where
    the single-sample estimator is actually bounded rather than asymptotic.
    """
    official, official_forward = _load_official_kl_penalty()
    student = torch.tensor(
        [-1e6, -80.0, -30.0, -20.0, -12.0, -11.0, -10.9999, -10.99, -5.0,
         -1.0, 0.0, 1.0, 5.0, 10.99, 11.0, 20.0, 30.0, -1e6],
        dtype=torch.float32)
    teacher = student + torch.tensor(
        [0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0],
        dtype=torch.float32)
    grid = torch.linspace(-30.0, 30.0, 4001, dtype=torch.float32)
    grid_student = torch.zeros_like(grid)
    grid_teacher = grid.clone()

    for stu, tea, label in ((student, teacher, "boundary table"),
                            (grid_student, grid_teacher, "dense grid")):
        want = official(logprob=stu, ref_logprob=tea, kl_penalty="k3")
        want_fp = official_forward(logprob=stu, ref_logprob=tea,
                                   kl_penalty="k3")
        got = M.k3_from_logprobs(stu, tea)
        assert torch.equal(want, want_fp), f"{label}: wrapper != forward body"
        assert torch.equal(got, want), (
            f"{label}: max |diff| {float((got - want).abs().max())}, "
            f"n_mismatch {int((got != want).sum())}")

    # Hand-computed anchors, so a wrong-but-consistent pair cannot pass.
    exact = {
        0.0: 0.0,                                   # exp(0) - 0 - 1
        -11.0: 10.0,                                # 10 + exp(-11) -> clamp
        -10.0: math.exp(-10.0) + 9.0,               # not clamped
        20.0: 10.0, -20.0: 10.0,                    # both tails clamp to 10
    }
    for kl, value in exact.items():
        got = float(M.k3_from_logprobs(torch.tensor([0.0]), torch.tensor([kl])))
        assert abs(got - value) < 1e-6, (kl, got, value)
    print(f"      bitwise equal to official kl_penalty on 18 boundary points + "
          f"4001-point grid (max kl +-30); exact anchors verified")


def t14_k3_in_loss_path():
    """`opd_fullkl_loss(rejected_kl='k3')` is the weighted mean of k3 terms.

    Also pins the two contracts: `k3` refuses to run without `sample_tokens`, and
    the DEFAULT (paper) path ignores `sample_tokens` entirely, so adding the
    argument cannot move any historical number.
    """
    p, q = _rand(K=4, V=7, seed=71)
    tokens = torch.tensor([3, 1, 5, 0])

    # Default config: the paper objective, byte-for-byte independent of tokens.
    base, base_m = M.opd_fullkl_loss(p, q, 1)
    with_tokens, with_m = M.opd_fullkl_loss(p, q, 1, sample_tokens=tokens)
    assert torch.equal(base, with_tokens), (float(base), float(with_tokens))
    assert base_m == with_m, (base_m, with_m)
    assert M.FullKLLossConfig().rejected_kl == M.REJECTED_KL_FULL_REVERSE

    # k3 path.
    cfg = M.FullKLLossConfig(rejected_kl=M.REJECTED_KL_K3)
    _raises(ValueError, lambda: M.opd_fullkl_loss(p, q, 1, config=cfg))
    _raises(ValueError, lambda: M.opd_fullkl_loss(p, q, 1, config=cfg,
                                                  sample_tokens=tokens[:2]))
    loss, m = M.opd_fullkl_loss(p, q, 1, config=cfg, sample_tokens=tokens)
    k3 = M.k3_reverse_kl(q[1:], p[1:], tokens[1:])
    w = torch.tensor([0.8 ** 0, 0.8 ** 1, 0.8 ** 2])
    want_rej = float((w * k3).sum() / w.sum())
    want = (float(M.full_kl_forward(p[:1], q[:1]).mean()) + want_rej) / 2.0
    assert abs(float(loss) - want) < 1e-6, (float(loss), want)
    assert abs(m["l_rej"] - want_rej) < 1e-6, (m["l_rej"], want_rej)
    assert m["rejected_kl"] == M.REJECTED_KL_K3 and m["n_rej_used"] == 3
    assert "k3_log_ratio_clamp_fraction" in m and "k3_kld_clamp_fraction" in m
    # Rejected-only k3 block: the loss IS the k3 stream mean, no /2, and the
    # rejected set is now all four offsets (w and tokens re-derived there).
    only, only_m = M.opd_fullkl_loss(p, q, 0, config=cfg, sample_tokens=tokens)
    w4 = torch.tensor([0.8 ** (k - 1) for k in range(1, 5)])
    k3_all = M.k3_reverse_kl(q, p, tokens)
    want_only = float((w4 * k3_all).sum() / w4.sum())
    assert abs(float(only) - want_only) < 1e-6, (float(only), want_only)
    assert only_m["n_acc_used"] == 0 and only_m["n_rej_used"] == 4
    _raises(ValueError, lambda: M.opd_fullkl_loss(
        p, q, 1, config=M.FullKLLossConfig(rejected_kl="k1")))
    print(f"      k3 loss {float(loss):.6f} == weighted mean of per-offset k3 "
          f"{want_rej:.6f}; default path ignores sample_tokens")


def t15_k3_semantics_and_gradient():
    """The k3 orientation is reverse KL(q||p), and the official clamps bias it low.

    Three checks the bit-compare cannot give:

      1. SEMANTICS -- drawing `x ~ q` and averaging `ratio - kl - 1` with
         `kl = log p(x) - log q(x)` converges to `KL(q || p)` (the reverse
         direction), not to `KL(p || q)`. Checked on the UNCLAMPED estimator so
         the clamps cannot mask a wrong orientation.
      2. BOUNDEDNESS -- the official `kld` clamp (`+-10`) is not a rounding
         detail: one rare sample with `p/q = 1000` is worth ~983 of true `kld`
         against a 10.0 ceiling, so a 1e-4 clamp fraction moves the mean by ~0.1.
         That downward bias is asserted explicitly, because it is exactly what
         makes the official estimator's SCALE incomparable to the full-vocabulary
         reverse KL.
      3. GRADIENT -- the analytic derivative of the UNCLAMPED estimator w.r.t.
         `log q(x)` is `1 - ratio`; a central finite difference of the module's
         own value, and of the full-vocabulary `k3_reverse_kl`, must match it.
    """
    V = 64
    generator = torch.Generator().manual_seed(3)
    z = torch.cat([torch.tensor([2.0]), torch.randn(V - 1, generator=generator) * 0.5])
    p = torch.softmax(z, dim=-1)
    q = p.clone()
    q[0] = p[0] / 1000.0                    # q carries (almost) no mass where p has some
    q = q / q.sum()
    logp, logq = p.log(), q.log()
    kl_qp = float((q * (logq - logp)).sum())    # KL(q || p), the reverse direction
    kl_pq = float((p * (logp - logq)).sum())    # KL(p || q), the forward direction
    assert abs(kl_qp - kl_pq) > 0.2, (kl_qp, kl_pq)

    samples = torch.distributions.Categorical(probs=q).sample((400000,))
    raw = M.k3_from_logprobs(logq[samples], logp[samples],
                             log_ratio_clamp=1e9, kld_clamp=1e9)
    clamped = M.k3_from_logprobs(logq[samples], logp[samples])
    assert abs(float(raw.mean()) - kl_qp) < 0.02, (float(raw.mean()), kl_qp)
    assert abs(float(raw.mean()) - kl_pq) > 4 * abs(float(raw.mean()) - kl_qp)
    clamp_fraction = float((clamped != raw).float().mean())
    assert 0.0 < clamp_fraction < 1e-3, clamp_fraction
    assert float(clamped.mean()) < float(raw.mean()) - 0.05, (
        float(clamped.mean()), float(raw.mean()))

    # Gradient: d/d logq(x) of (ratio - kl - 1) = 1 - ratio, for kl inside clamps.
    tok = int(samples[0])
    stu = logq[tok].clone().requires_grad_(True)
    tea = logp[tok].detach()
    value = M.k3_from_logprobs(stu.reshape(1), tea.reshape(1)).sum()
    value.backward()
    ratio = float(torch.exp(tea - logq[tok]))
    assert abs(float(stu.grad) - (1.0 - ratio)) < 1e-5, (float(stu.grad), 1 - ratio)

    logits = (z + 0.3 * torch.randn(V, generator=generator)).requires_grad_(True)
    loss = M.k3_reverse_kl(logits.unsqueeze(0), z.unsqueeze(0),
                           torch.tensor([tok]))
    analytic = torch.autograd.grad(loss.sum(), logits)[0]
    eps = 1e-3
    numeric = torch.zeros_like(logits)
    for i in range(V):
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus[i] += eps
        minus[i] -= eps
        hi = M.k3_reverse_kl(plus.unsqueeze(0), z.unsqueeze(0), torch.tensor([tok])).sum()
        lo = M.k3_reverse_kl(minus.unsqueeze(0), z.unsqueeze(0), torch.tensor([tok])).sum()
        numeric[i] = (float(hi) - float(lo)) / (2 * eps)
    assert float((analytic - numeric).abs().max()) < 1e-3, (
        float((analytic - numeric).abs().max()))
    print(f"      E_q[k3] {float(raw.mean()):.4f} ~ KL(q||p) {kl_qp:.4f} "
          f"(KL(p||q) {kl_pq:.4f}); clamp frac {clamp_fraction:.2e} biases "
          f"{float(clamped.mean()):.4f}; grads exact")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

CPU_TESTS = [
    ("01 hand-computed KL (full vocab, fp32)", t01_hand_kl),
    ("02 asymmetry is real and non-negative", t02_asymmetry_nonnegative),
    ("02b KL/loss are computed in fp32 for bf16 inputs", t02b_internal_fp32),
    ("03 w_k = gamma**(k-1), checked against a literal loop", t03_gamma_weights),
    ("03b gamma moves the loss in the expected direction", t03b_gamma_direction),
    ("03c accepted stream has no position weight", t03c_accepted_has_no_position_weight),
    ("07 gamma touches only the rejected stream", t07_gamma_only_on_rejected),
    ("04 denominators: plain mean + weighted mean, literal loop", t04_denominators_literal),
    ("05 combination with both streams: (L_acc+L_rej)/2", t05_combination),
    ("06 empty streams: drop-and-renormalise, zero when both empty", t06_empty_streams),
    ("08 valid mask excludes from both streams and denominators", t08_valid_mask),
    ("09 differentiability and the no-grad teacher contract", t09_gradient),
    ("10 grad_cosine_and_norm_ratio", t10_grad_cosine_and_norm_ratio),
    ("11a teacher row k-1 <- stream position anchor_pos+k-1 (stub)", t11a_teacher_row_mapping_stub),
    ("11b draft block rows/positions follow rollout (stub)", t11b_draft_block_convention_stub),
    ("12 stream_split matches the loss's own stream means", t12_stream_split_matches_loss),
    ("13 k3 estimator bit-identical to official kl_penalty", t13_k3_official_bitwise),
    ("14 rejected_kl='k3' loss path, default untouched", t14_k3_in_loss_path),
    ("15 k3 is reverse KL(q||p); gradient is exact", t15_k3_semantics_and_gradient),
]


def run_cpu():
    print("=" * 78)
    print("CPU semantics tests (V=7, K=4, no model)")
    print("=" * 78)
    passed, failed = 0, []
    for name, fn in CPU_TESTS:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL {name}\n     {exc}")
            failed.append(name)
        else:
            print(f"PASS {name}")
            passed += 1
    print(f"\nCPU: {passed} passed, {len(failed)} failed "
          f"({len(CPU_TESTS)} tests)")
    if failed:
        print("failed: " + ", ".join(failed))
    return not failed


# --------------------------------------------------------------------------
# GPU smoke: fork purity and teacher correctness on the real model
# --------------------------------------------------------------------------

def gpu_smoke(args):
    import pandas as pd
    from transformers import DynamicCache

    from shared import common
    from historical import opd_online_rollout
    from diagnostics import proto_fullkl_b0 as proto

    dev = f"cuda:{args.gpu}"
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tol = {"bfloat16": 2.0, "float32": 2e-4}[args.dtype]
    torch.cuda.set_device(args.gpu)
    print("=" * 78)
    print(f"GPU smoke on {dev} ({torch.cuda.get_device_name(args.gpu)}), "
          f"dtype {args.dtype}, tol max|dlogit| {tol}")
    print("=" * 78)

    target, proc = common.load_target(device=dev, dtype=dtype)
    head = common.load_head(args.head, device=dev, dtype=dtype).eval()
    head.requires_grad_(False)
    tg_req = [n for n, p in target.named_parameters() if p.requires_grad]
    assert not tg_req, f"the frozen target has trainable parameters: {tg_req[:3]}"
    n_head = sum(p.numel() for p in head.parameters())
    n_target = sum(p.numel() for p in target.parameters())
    print(f"[load] target {dtype} {n_target / 1e9:.2f}B  head "
          f"{n_head / 1e6:.1f}M params ({n_head * 4 / 2**30:.2f} GiB in fp32, "
          f"+2 Adam moments would be {n_head * 12 / 2**30:.2f} GiB)")

    df = pd.read_parquet(args.rows)
    row = df.iloc[args.row_index]
    name = str(row["video_name"])
    blob = torch.load(args.records, weights_only=False, map_location="cpu")
    recs = [r for r in blob["records"] if str(r["task"]) == name]
    assert len(recs) == 1, f"{name} appears {len(recs)} times in {args.records}"
    rec = recs[0]
    prompt_len = int(rec["prompt_len"])
    seq_ids = rec["input_ids"].to(dev).long()

    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": row["question"]}]}],
        add_generation_prompt=True, tokenize=False)
    ins = proc(text=[text], videos=[row["cache_path"]], return_tensors="pt",
               fps=float(row["fps"]))
    prompt_ids = ins["input_ids"].to(dev)
    pv = ins["pixel_values_videos"].to(dev)
    vg = ins["video_grid_thw"].to(dev)
    assert int(prompt_ids.shape[1]) == prompt_len, (
        f"prompt drifted: encoded {int(prompt_ids.shape[1])} vs stored "
        f"{prompt_len}")
    print(f"[row] {name}  prompt {prompt_len}  trajectory "
          f"{int(seq_ids.shape[0])}  video_grid {vg.tolist()}")

    # ---------------------------------------------------------------- prefill
    p = prompt_len                                   # first chain anchor
    prefix_ids = seq_ids[None, :p + 1]
    vispos = M.prefix_vision_positions(target, prefix_ids,
                                       pixel_values_videos=pv, video_grid_thw=vg)
    fused, prefix_kv = M.target_prefill_shared(
        target, prefix_ids, pixel_values_videos=pv, video_grid_thw=vg,
        max_seq_len=common.MAX_SEQ_LEN, vispos=vispos)
    assert int(prefix_kv.get_seq_length()) == p + 1, prefix_kv.get_seq_length()
    print(f"[prefill] fused {tuple(fused.shape)} {fused.dtype}  "
          f"prefix_kv {int(prefix_kv.get_seq_length())} rows")

    # The same one-forward interface must produce the context the rest of the
    # project already uses (`common.target_context_cached`), or SFT and OPD are
    # reading different features.
    ref_ctx = common.target_context_cached(target, prefix_ids,
                                           pixel_values_videos=pv,
                                           video_grid_thw=vg)
    d_ctx = float((fused.float() - ref_ctx.float()).abs().max())
    print(f"[shared-prefill] fused vs common.target_context_cached: "
          f"max|d|={d_ctx:.3e}  shape {tuple(ref_ctx.shape)}")
    assert d_ctx < 1e-3, d_ctx

    # Full-trajectory context for the head (as the trainer builds it).
    ctx = common.target_context_cached(target, seq_ids[None],
                                       pixel_values_videos=pv,
                                       video_grid_thw=vg)

    # ------------------------------------------- draft logits vs the rollout
    anchors = opd_online_rollout.rollout(head, target, ctx, seq_ids,
                                        prompt_len=prompt_len, verify="chain",
                                        max_rounds=2)
    assert len(anchors) == 2, anchors
    q_rollout = proto.replay_draft_logits(head, target, ctx, seq_ids, anchors,
                                          prompt_len)
    q_ours = []
    cache = DynamicCache()
    with torch.no_grad():
        for anc in anchors:
            ap = int(anc["pos"])
            logits = M.draft_block_logits(
                head, target, ctx[:, cache.get_seq_length():ap],
                anchor_pos=ap, root_token=int(seq_ids[ap]),
                prompt_len=prompt_len, cache=cache)
            cache.crop(ap)          # exactly what opd_online_rollout.rollout does
            q_ours.append(logits)
    for i, (anc, qa, qb) in enumerate(zip(anchors, q_ours, q_rollout)):
        d = float((qa.float() - qb.float()).abs().max())
        same_argmax = torch.equal(qa.argmax(-1)[:len(anc["prop"])],
                                  qb.argmax(-1)[:len(anc["prop"])])
        prop_ok = [int(t) for t in qa.argmax(-1)[:len(anc["prop"])]] == \
                  [int(t) for t in anc["prop"]]
        print(f"[draft {i}] pos {anc['pos']} n_acc {anc['n_acc']}"
              f"/{len(anc['prop'])}  ours vs replay max|d|={d:.3e}  "
              f"argmax==replay {same_argmax}  argmax==prop {prop_ok}")
        assert d < 1e-3, d
        assert same_argmax and prop_ok, (qa.argmax(-1), anc["prop"])

    # ------------------------------------------------------------ teacher
    anc = anchors[0]
    prop = [int(t) for t in anc["prop"]]
    n_acc = int(anc["n_acc"])
    K = len(prop)
    len_before = int(prefix_kv.get_seq_length())
    p_logits = M.teacher_block_logits(target, vispos, prefix_kv,
                                      anchor_pos=int(anc["pos"]), prop=prop,
                                      chunk=8)
    len_after = int(prefix_kv.get_seq_length())
    print(f"[fork] shared prefix_kv length {len_before} -> {len_after} "
          f"(must be unchanged);  teacher logits {tuple(p_logits.shape)} "
          f"{p_logits.dtype}")
    assert len_after == len_before, (len_before, len_after)

    # A second call must be deterministic and just as non-polluting.
    again = M.teacher_block_logits(target, vispos, prefix_kv,
                                   anchor_pos=int(anc["pos"]), prop=prop, chunk=8)
    assert int(prefix_kv.get_seq_length()) == len_before
    assert float((again - p_logits).abs().max()) == 0.0

    # Independent reference: one full forward over prefix + prop, fresh cache.
    ref_logits, ref_positions, ref_hidden = M.reference_teacher_logits(
        target, prefix_ids, prop, pixel_values_videos=pv, video_grid_thw=vg)
    # The M-RoPE claims the whole method rests on, asserted against
    # get_rope_index rather than trusted.
    prefill_positions = vispos.prefill_inputs(prefix_ids, None)["position_ids"]
    assert torch.equal(ref_positions[:, :, :p + 1], prefill_positions), \
        "prefix M-RoPE positions differ from the full-sequence positions"
    stream = torch.arange(p + 1, p + 1 + K, device=prefix_ids.device)
    assert torch.equal(vispos(stream),
                       ref_positions[:, :, p + 1:p + 1 + K]), \
        "VisionPositions(stream) does not reproduce the full-sequence positions"

    d_logit = float((p_logits - ref_logits).abs().max())
    d_tv = float((torch.softmax(p_logits, -1)
                  - torch.softmax(ref_logits, -1)).abs().sum(-1).max())
    argmax_ok = int((p_logits.argmax(-1) == ref_logits.argmax(-1)).sum())
    scale = float(ref_logits.abs().max())

    # The honest floor: the SAME method with one token per forward instead of a
    # chunk of 8. Mathematically identical, so what it moves is pure bf16/fp32
    # GEMM row-count noise.
    ctrl = M.teacher_block_logits(target, vispos, prefix_kv,
                                  anchor_pos=int(anc["pos"]), prop=prop, chunk=1)
    d_ctrl = float((ctrl - p_logits).abs().max())
    print(f"[teacher] incremental vs independent full forward: "
          f"max|dlogit|={d_logit:.3e}  maxTV={d_tv:.3e}  argmax "
          f"{argmax_ok}/{K}  (ref |logit|max {scale:.1f})")
    print(f"[teacher] chunk=1 vs chunk=8 control (same method, row count only): "
          f"max|d|={d_ctrl:.3e}")
    assert d_logit < tol, (d_logit, tol)

    # --------------------------------------------------------------- loss
    with torch.no_grad():
        loss_inc, m_inc = M.opd_fullkl_loss(p_logits, q_ours[0].float(), n_acc)
        loss_ref, m_ref = M.opd_fullkl_loss(ref_logits, q_ours[0].float(), n_acc)
    print(f"[loss] n_acc {n_acc}/{K}: L_acc {m_inc['l_acc']:.4f} "
          f"({m_inc['n_acc_used']} offsets)  L_rej {m_inc['l_rej']:.4f} "
          f"({m_inc['n_rej_used']} offsets, sum_w {m_inc['sum_w_rej']:.4f})  "
          f"L {float(loss_inc):.4f}  streams_present "
          f"{m_inc['streams_present']}")
    print(f"[loss] same q against the independent full-forward teacher: "
          f"L {float(loss_ref):.4f}  (drift "
          f"{abs(float(loss_inc) - float(loss_ref)):.3e})")

    # --------------------------------------------- end-to-end differentiability
    head.requires_grad_(True)
    cache2 = DynamicCache()
    with torch.enable_grad():
        q_train = M.draft_block_logits(
            head, target, ctx[:, :int(anc["pos"])], anchor_pos=int(anc["pos"]),
            root_token=int(seq_ids[int(anc["pos"])]), prompt_len=prompt_len,
            cache=cache2)
        loss_train, _ = M.opd_fullkl_loss(p_logits, q_train, n_acc)
        loss_train.backward()
    grads = {n: p.grad for n, p in head.named_parameters()}
    n_grad = sum(1 for v in grads.values() if v is not None)
    gnorm = math.sqrt(sum(float(v.detach().double().pow(2).sum())
                          for v in grads.values() if v is not None))
    stats = M.grad_cosine_and_norm_ratio(grads, grads)
    print(f"[grad] loss {float(loss_train.detach()):.4f}  tensors with grad {n_grad}  "
          f"||g|| {gnorm:.4e}  self-cosine {stats['cosine']}  "
          f"self-ratio {stats['norm_ratio']}")
    assert n_grad > 0 and math.isfinite(gnorm) and gnorm > 0.0
    head.requires_grad_(False)
    head.zero_grad(set_to_none=True)

    peak = torch.cuda.max_memory_allocated(args.gpu) / 2**30
    print(f"[mem] peak allocated {peak:.2f} GiB")
    print(f"GPU SMOKE PASS ({args.dtype}): fork purity, positions, teacher rows "
          f"within {tol}, draft path == rollout replay")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-smoke", action="store_true",
                    help="also run the one-model smoke (needs GPU 2 by default)")
    ap.add_argument("--skip-cpu", action="store_true",
                    help="skip the CPU suite (useful when appending a smoke run "
                         "to a log that already holds it)")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"],
                    help="model dtype for the smoke; the CPU tests are fp32 "
                         "throughout regardless")
    ap.add_argument("--head", default=f"{M.common.CKPT}/head_video439/step1000")
    ap.add_argument("--rows", default=f"{M.common.DATA}/video_fresh30.parquet")
    ap.add_argument("--records",
                    default=f"{M.common.DATA}/video_opd_pool_warm_499.pt")
    ap.add_argument("--row-index", type=int, default=0)
    args = ap.parse_args()

    ok = run_cpu() if not args.skip_cpu else True
    if args.gpu_smoke:
        ok = gpu_smoke(args) and ok
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
