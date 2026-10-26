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


def pool(response_q, response_p, rejected_q, rejected_p, offsets, response_count=None):
    if response_q.shape != response_p.shape or rejected_q.shape != rejected_p.shape:
        raise ValueError("teacher/student selected-token rows differ")
    if offsets.shape != rejected_q.shape or (offsets <= 0).any():
        raise ValueError("rejected offsets must be one-based and aligned")
    weights = 0.9 ** (offsets.to(dtype=torch.float32) - 1)
    return Pool(
        bernoulli_forward(response_q, response_p).sum()
        + 2 * (k3(rejected_q, rejected_p) * weights).sum(),
        response_q.new_tensor(
            response_q.numel() if response_count is None else response_count,
            dtype=torch.float32,
        )
        + 2 * weights.sum(),
    )
