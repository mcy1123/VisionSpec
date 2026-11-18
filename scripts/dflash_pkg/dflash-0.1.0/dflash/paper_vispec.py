"""ViSpec video adaptor for a DFlash block. See arXiv:2509.15235, Appendix B.2.

Each temporal group produces one prefix token and one global feature using two
learned queries, as in the authors' ImgAdaptor. DFlash inserts the prefix tokens
in its attention and broadcasts the mean global feature over the draft block.
The adaptor width is configurable independently of the backbone width.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .evidence import EvidenceReadout


class PaperViSpecReadout(EvidenceReadout):
    def __init__(self, hidden_size, dim=256, max_tokens=256, spatial_bins=2, mode="visual"):
        # Share EvidenceReadout's request, donor and cache lifecycle. This arm
        # has its own parameters because its queries operate on each frame.
        nn.Module.__init__(self)
        if mode != "visual" or min(hidden_size, dim, max_tokens, spatial_bins) < 1:
            raise ValueError("ViSpec requires visual mode and positive dimensions")
        self.mode, self.hidden_size = mode, hidden_size
        self.max_tokens, self.spatial_bins = max_tokens, spatial_bins
        self.request, self.generation, self._stats = None, 0, {}
        self.num_heads = math.gcd(dim, 8)
        self.head_dim = dim // self.num_heads

        self.frame_queries = nn.Parameter(torch.empty(2, self.num_heads, self.head_dim))
        nn.init.normal_(self.frame_queries, std=self.head_dim ** -0.5)
        self.key = nn.Linear(hidden_size, dim, bias=False)
        self.value = nn.Linear(hidden_size, dim, bias=False)
        self.prefix_output = nn.Linear(dim, hidden_size, bias=False)
        self.output = nn.Linear(dim, hidden_size, bias=False)
        # The training entry point verifies that this shared protocol field is
        # zero and frozen. Time is encoded by the positions of the prefix keys.
        self.time = nn.Linear(6, dim, bias=False)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.time.weight)
        self.time.weight.requires_grad_(False)

    def set_request(self, request):
        super().set_request(request)
        self._position_cache = None

    def prefix_positions(self, groups, device):
        if groups != self.request["grid_thw"][0]:
            raise ValueError("ViSpec requires one prefix token for every original temporal group")
        if self._position_cache is None or self._position_cache.device != device:
            # The original ViSpec code anchors a compressed visual row at the
            # trailing visual position. Preserve the actual prompt separators.
            positions = self.request["prompt_visual_mask"].nonzero(as_tuple=True)[0]
            self._position_cache = positions.reshape(groups, -1)[:, -1].contiguous().to(device)
        return self._position_cache

    def _compress(self, features, groups):
        # [groups, spatial patches, hidden] -> [groups, heads, patches, dim/head]
        def split_heads(projection):
            projected = self._linear(projection, features)
            return projected.reshape(groups, -1, self.num_heads, self.head_dim).transpose(1, 2)

        query = self.frame_queries.float().transpose(0, 1).expand(groups, -1, -1, -1).contiguous()
        tokens = F.scaled_dot_product_attention(query, split_heads(self.key), split_heads(self.value))
        tokens = tokens.transpose(1, 2).reshape(groups, 2, -1)
        prefix = self._linear(self.prefix_output, tokens[:, 0])
        global_feature = self._linear(self.output, tokens[:, 1].mean(dim=0))
        return prefix, global_feature

    def prefix_and_global(self, prepared):
        """Return compact prefix [groups,H] and global [H] using frozen request features."""
        features, times, state = prepared
        intervention = self.request.get("intervention", "normal")
        self._stats["calls"] += 1
        if intervention == "bypass":
            return None, None

        source = self.request["donor_visual" if intervention == "wrong_video" else "prepared_visual"]
        groups, bins = source["groups"], source["spatial_bins"]
        if groups < 1 or bins < 1 or features.shape != (groups * bins, self.hidden_size):
            raise ValueError("ViSpec needs all spatial rows of every prepared temporal group")
        self._stats.update(memory_tokens=groups, source_tokens=groups * bins)
        # A rollout and its later gradient-bearing OPD scoring may share the
        # frozen inputs. Trainable projected tensors may only be reused in no-grad.
        cacheable = not torch.is_grad_enabled()
        if cacheable:
            signature = (self._owner(), intervention, int(self.request.get("seed", 0)),
                         str(features.device), id(source), self._donor_signature(source),
                         tuple(parameter._version for parameter in self.parameters()))
            cached = state.get("projected")
            if cached is not None and cached[0] == signature:
                self._stats["kv_cache_hits"] += 1
                return cached[1:]

        with torch.autocast(device_type=features.device.type, enabled=False):
            prefix, global_feature = self._compress(features, groups)
            if intervention == "time_shuffle":
                # The common intervention permutes timestamps, leaving rows in
                # place. Present tokens in that order to the fixed prefix slots.
                order = times.reshape(groups, bins)[:, 0].argsort()
                prefix = prefix[order]
        self._stats["kv_recomputes"] += 1
        self._last_residual_rms = global_feature.detach().square().mean().sqrt()
        if cacheable:
            state["projected"] = signature, prefix.detach(), global_feature.detach()
            return state["projected"][1:]
        return prefix, global_feature

    def forward(self, hidden, prepared):
        # The joint-attention integration already compressed this request
        # once. Packed training can broadcast the same global to each block.
        global_feature = (self.prefix_and_global(prepared)[1]
                          if isinstance(prepared, tuple) else prepared)
        if global_feature is None:
            return hidden
        return (hidden.float() + global_feature).to(hidden.dtype)

    def diagnostics(self):
        result = dict(self._stats)
        if hasattr(self, "_last_residual_rms"):
            result["residual_rms"] = float(self._last_residual_rms.cpu())
        return result
