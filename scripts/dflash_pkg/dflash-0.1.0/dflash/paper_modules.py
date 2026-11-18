"""Published visual-memory operators adapted to a single committed video prompt."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _token_times(grid, times):
    if grid.ndim != 3 or min(grid.shape) < 1:
        raise ValueError("video grid must have shape [time, spatial bins, hidden]")
    if times.shape == grid.shape[:1]:
        return times[:, None].expand(grid.shape[:2])
    if times.shape != grid.shape[:2]:
        raise ValueError("expected one timestamp per temporal group or per visual token")
    return times


def tome_bipartite(grid, times, target_tokens):
    """ToMe-style bipartite soft matching over a budgeted target-feature grid.

    Bolya et al. (ICLR 2023), https://arxiv.org/abs/2210.09461. The frozen
    hidden states supply the similarity metric in place of ViT attention keys.
    Merging takes place at the drafter's memory interface. Features and
    timestamps preserve cluster size when a merged token is merged again.
    """
    clock = _token_times(grid, times)
    if not isinstance(target_tokens, int) or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")
    tokens = grid.flatten(0, 1)
    if len(tokens) <= target_tokens:
        return tokens, clock.reshape(-1)
    # The last coordinate tracks the mass-weighted timestamp of each cluster.
    points = torch.cat((tokens.float(), clock.reshape(-1, 1).to(tokens.device).float()), dim=-1)
    mass = torch.ones_like(points[:, :1])
    while len(points) > target_tokens:
        with torch.no_grad():
            metric = F.normalize(points[:, :-1].detach(), dim=-1)
            left, right = metric[::2], metric[1::2]
            score, neighbor = (left @ right.T).max(dim=1)
            merge_count = min(len(points) - target_tokens, len(right))
            order = torch.argsort(score, descending=True, stable=True)
            selected, keep = order[:merge_count], order[merge_count:]
            # Dense routing is small on the pooled memory and avoids CUDA
            # scatter atomics when several sources choose one destination.
            route = F.one_hot(neighbor[selected], num_classes=len(right)).T.to(points.dtype)
        left_mass, right_mass = mass[::2], mass[1::2]
        added_mass = route @ left_mass[selected]
        weighted = route @ (points[::2][selected] * left_mass[selected])
        merged_mass = right_mass + added_mass
        merged = (points[1::2] * right_mass + weighted) / merged_mass
        points = torch.cat((points[::2][keep], merged))
        mass = torch.cat((left_mass[keep], merged_mass))
    return points[:, :-1].to(grid.dtype), points[:, -1].to(times.dtype)


class FlamingoReadout(nn.Module):
    """One narrow Flamingo Perceiver Resampler and GATED XATTN-DENSE block.

    Alayrac et al. (NeurIPS 2022), https://arxiv.org/abs/2204.14198.
    A single video is already visible to every speculative position, so no
    interleaved-media mask is needed. The experiment uses one resampler layer
    and a bottleneck feedforward to fit the small parallel drafter.
    """

    def __init__(self, hidden_size, dim=256, num_latents=8, use_time_embeddings=True):
        super().__init__()
        if min(hidden_size, dim, num_latents) < 1:
            raise ValueError("Flamingo dimensions must be positive")
        self.hidden_size = hidden_size
        self.heads = math.gcd(dim, 8)
        self.head_dim = dim // self.heads
        self.latents = nn.Embedding(num_latents, dim)
        self.time_embeddings = nn.Embedding(64, dim) if use_time_embeddings else None
        self.visual_input = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim, bias=False))
        self.latent_norm = nn.LayerNorm(dim)
        self.visual_norm = nn.LayerNorm(dim)
        self.resample_query = nn.Linear(dim, dim, bias=False)
        self.resample_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.resample_output = nn.Linear(dim, dim, bias=False)
        self.resample_ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim, bias=False),
                                         nn.GELU(), nn.Linear(4 * dim, dim, bias=False))
        self.latent_kv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim, bias=False))
        self.text_query = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim, bias=False))
        self.attn_output = nn.Linear(dim, hidden_size, bias=False)
        self.text_ff = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim, bias=False),
                                     nn.GELU(), nn.Linear(dim, hidden_size, bias=False))
        self.attn_gate = nn.Parameter(torch.zeros(()))
        self.ff_gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.latents.weight, std=0.02)
        if self.time_embeddings is not None:
            nn.init.normal_(self.time_embeddings.weight, std=0.02)

    def _heads(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(-3, -2)

    def _attend(self, queries, keys, values):
        read = F.scaled_dot_product_attention(self._heads(queries), self._heads(keys), self._heads(values))
        return read.transpose(-3, -2).flatten(-2)

    def prepare_memory(self, grid, times):
        """Return projected latent keys and values, each [1, num_latents, dim]."""
        clock = _token_times(grid, times)
        visual = self.visual_input(grid.to(self.latents.weight.dtype))
        if self.time_embeddings is not None:
            # Interpolate a learned temporal table for varying sampled lengths.
            table = self.time_embeddings.weight
            position = clock.to(visual.device).float().clamp(0, 1) * (len(table) - 1)
            low = position.long()
            high = (low + 1).clamp_max(len(table) - 1)
            visual = visual + torch.lerp(table[low], table[high], (position - low).unsqueeze(-1).to(table.dtype))
        visual = self.visual_norm(visual.flatten(0, 1))[None]
        latents = self.latents.weight.unsqueeze(0)
        normalized_latents = self.latent_norm(latents)
        key, value = self.resample_kv(torch.cat((visual, normalized_latents), dim=1)).chunk(2, dim=-1)
        summary = self._attend(self.resample_query(normalized_latents), key, value)
        latents = latents + self.resample_output(summary)
        latents = latents + self.resample_ff(latents)
        return self.latent_kv(latents).chunk(2, dim=-1)

    def forward(self, hidden, prepared):
        """Inject prepared video into each independent block position."""
        key, value = prepared
        text = hidden.to(self.latents.weight.dtype)
        read = self._attend(self.text_query(text), key, value)
        text = text + self.attn_gate.tanh() * self.attn_output(read)
        text = text + self.ff_gate.tanh() * self.text_ff(text)
        return text.to(hidden.dtype)


class FoveaReadout(nn.Module):
    """FOVEA Algorithm 1 retrieval and gated correction for parallel block slots.

    Zhu et al., https://arxiv.org/html/2608.22883v1. Block-slot depth takes
    the role of tree depth. A bounded target-feature grid supplies the memory,
    with bottleneck keys and values. Normalization by the draft head and the
    training objective are supplied by the host DFlash, not by this module.
    """

    def __init__(self, hidden_size, dim=256, max_depth=16):
        super().__init__()
        if min(hidden_size, dim, max_depth) < 1:
            raise ValueError("FOVEA dimensions must be positive")
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.memory_kv = nn.Linear(hidden_size, 2 * dim, bias=False)
        self.text_norm = nn.LayerNorm(hidden_size)
        self.depth = nn.Embedding(max_depth, hidden_size)
        self.query = nn.Linear(hidden_size, dim, bias=False)
        self.correction = nn.Sequential(nn.Linear(hidden_size + dim, 512), nn.SiLU(),
                                        nn.Linear(512, hidden_size))
        self.gate = nn.Sequential(nn.Linear(hidden_size + dim, 512), nn.SiLU(), nn.Linear(512, 1))
        nn.init.normal_(self.depth.weight, std=0.02)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        nn.init.constant_(self.gate[-1].bias, -4.0)

    def prepare_memory(self, grid, times):
        """Return cached keys and values, each [1, time * spatial bins, dim]."""
        _token_times(grid, times)
        features = grid.flatten(0, 1)[None].to(self.memory_norm.weight.dtype)
        return self.memory_kv(self.memory_norm(features)).chunk(2, dim=-1)

    @staticmethod
    def _retrieve(logits, values):
        probabilities = logits.float().softmax(-1)
        limit = min(256, probabilities.shape[-1])
        floor = min(32, limit)
        top, indices = probabilities.topk(limit, dim=-1)
        sizes = ((top.cumsum(-1) < 0.8).sum(-1) + 1).clamp(floor, limit)
        included = torch.arange(limit, device=logits.device) < sizes.unsqueeze(-1)
        selected = top * included
        weights = selected / selected.sum(-1, keepdim=True)
        read = (weights.unsqueeze(-1) * values[0][indices].float()).sum(-2)
        return read.to(values.dtype)

    def forward(self, hidden, prepared):
        keys, values = prepared
        if not 0 < hidden.shape[1] <= self.depth.num_embeddings:
            raise ValueError("FOVEA needs a single complete or truncated block for depth assignment")
        text = hidden.to(self.query.weight.dtype)
        slots = torch.arange(text.shape[1], device=text.device)
        query = self.query(self.text_norm(text) + self.depth(slots))
        logits = (query.float() @ keys.float().transpose(-1, -2)) / math.sqrt(keys.shape[-1])
        read = self._retrieve(logits, values)
        fused = torch.cat((text, read), dim=-1)
        residual = self.gate(fused).sigmoid() * self.correction(fused)
        return (hidden.float() + residual.float()).to(hidden.dtype)
