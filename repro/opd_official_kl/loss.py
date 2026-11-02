"""Lightweight adaptation of Apache-2.0 VERL losses, pinned in SOURCE.json.

Recipe is an explicitly overridden base script, NOT its k3/k3 default.
Teacher tensors are detached at the scoring boundary. No vocabulary KL or CE.
"""

import math
from dataclasses import dataclass
import torch


def bernoulli_forward(student_log_probs, teacher_log_probs):
    q = student_log_probs.float().clamp_min(-10.0)
    p = teacher_log_probs.detach().float().clamp_min(-10.0)
    maximum = math.log1p(-torch.finfo(q.dtype).eps)
    q, p = q.clamp(max=maximum), p.clamp(max=maximum)
    return (
        p.exp() * (p - q)
        + (1 - p.exp()) * (torch.log1p(-p.exp()) - torch.log1p(-q.exp()))
    ).clamp(-10.0, 10.0)


def k3(student_log_probs, teacher_log_probs):
    ratio = (teacher_log_probs.detach().float() - student_log_probs.float()).clamp(
        -20.0, 20.0
    )
    return (ratio.exp() - ratio - 1).clamp(-10.0, 10.0)


@dataclass
class Pool:
    numerator: torch.Tensor
    denominator: torch.Tensor

    def __add__(self, other):
        return Pool(
            self.numerator + other.numerator, self.denominator + other.denominator
        )

    def loss(self, global_denominator=None, dp_size=1):
        # DDP averages gradients: its local numerator needs dp_size/global_count.
        denom = self.denominator if global_denominator is None else global_denominator
        return (
            self.numerator * dp_size / denom.clamp_min(1e-30)
            if denom.item() > 0
            else self.numerator * 0
        )


def pool(response_q, response_p, rejected_q, rejected_p, offsets, response_count=None,
         response_weights=None, rejected_weights=None):
    if response_q.shape != response_p.shape or rejected_q.shape != rejected_p.shape:
        raise ValueError("teacher/student selected-token rows differ")
    if offsets.shape != rejected_q.shape or (offsets <= 0).any():
        raise ValueError("rejected offsets must be one-based and aligned")
    weights = 0.9 ** (offsets.to(dtype=torch.float32) - 1)
    response_term = bernoulli_forward(response_q, response_p)
    rejected_term = k3(rejected_q, rejected_p) * weights
    if response_weights is not None:
        if response_weights.shape != response_q.shape:
            raise ValueError("response visual weights must align with response tokens")
        response_term = response_term * response_weights.to(response_term.dtype)
    if rejected_weights is not None:
        if rejected_weights.shape != rejected_q.shape:
            raise ValueError("rejected visual weights must align with rejected tokens")
        rejected_term = rejected_term * rejected_weights.to(rejected_term.dtype)

    if response_weights is None:
        response_denom = response_q.new_tensor(
            response_q.numel() if response_count is None else response_count,
            dtype=torch.float32,
        )
    else:
        # response_count may include response positions outside the plan (uncovered);
        # count them with neutral weight 1 so the official denominator is preserved.
        uncovered = (0 if response_count is None
                     else max(0, int(response_count) - int(response_weights.numel())))
        response_denom = response_weights.to(torch.float32).sum() + uncovered
    rejected_denom = (weights * (1.0 if rejected_weights is None
                                 else rejected_weights.to(weights.dtype))).sum()
    return Pool(
        response_term.sum() + 2 * rejected_term.sum(),
        response_denom + 2 * rejected_denom,
    )
