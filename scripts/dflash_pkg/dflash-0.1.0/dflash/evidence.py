"""Request-scoped evidence readout over frozen, already committed target features.

Only the output projection starts at zero. Visual and text modes have identical
parameters; the memory source is their only architectural difference. No target
forward, future response feature, or trainable projection is cached in requests.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class EvidenceReadout(nn.Module):
    def __init__(self, hidden_size, dim=256, max_tokens=256, spatial_bins=2, mode="visual"):
        super().__init__()
        if mode not in ("visual", "text"):
            raise ValueError("evidence mode must be visual or text")
        if min(hidden_size, dim, max_tokens, spatial_bins) < 1:
            raise ValueError("evidence dimensions must be positive")
        self.mode, self.hidden_size = mode, hidden_size
        self.max_tokens, self.spatial_bins = max_tokens, spatial_bins
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, dim, bias=False)
        self.key = nn.Linear(hidden_size, dim, bias=False)
        self.value = nn.Linear(hidden_size, dim, bias=False)
        self.time = nn.Linear(6, dim, bias=False)
        self.output = nn.Linear(dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.request = None
        self.generation = 0
        self._stats = {}
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)

    def set_request(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("identity"), str) or not request["identity"]:
            raise ValueError("evidence requires a nonempty request identity")
        mask = request.get("prompt_visual_mask")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError("evidence prompt_visual_mask must be a 1-D bool tensor")
        grid = request.get("grid_thw")
        if not isinstance(grid, (list, tuple)) or len(grid) != 3 or any(int(x) != x or x <= 0 for x in grid):
            raise ValueError("evidence grid_thw must contain three positive integers")
        merge = request.get("spatial_merge_size")
        if not isinstance(merge, int) or merge < 1 or grid[1] % merge or grid[2] % merge:
            raise ValueError("evidence grid is incompatible with spatial_merge_size")
        if int(mask.sum()) != grid[0] * (grid[1] // merge) * (grid[2] // merge):
            raise ValueError("evidence visual mask count disagrees with grid")
        times = request.get("time_values")
        if not isinstance(times, torch.Tensor) or times.ndim != 1 or len(times) != grid[0]:
            raise ValueError("evidence time_values must have one entry per temporal group")
        if not bool(torch.isfinite(times).all()) or bool(((times < 0) | (times > 1)).any()):
            raise ValueError("evidence time_values must be finite and normalized to [0,1]")
        if len(times) > 1 and not bool((times[1:] > times[:-1]).all()):
            raise ValueError("evidence source time_values must increase strictly")
        intervention = request.get("intervention", "normal")
        if intervention not in ("normal", "bypass", "zero", "time_shuffle", "wrong_video"):
            raise ValueError("unsupported evidence intervention")
        if self.mode == "text" and intervention in ("time_shuffle", "wrong_video"):
            raise ValueError(f"{intervention} is a visual evidence intervention")
        if intervention == "wrong_video":
            self._validate_donor(request)
        elif "donor_visual" in request:
            raise ValueError("donor_visual is allowed only for wrong_video")
        self.request = request
        self.generation += 1
        if hasattr(self, "_last_residual_rms"):
            del self._last_residual_rms
        self._stats = {"calls": 0, "memory_tokens": 0, "visual_prepare_hits": 0,
                       "kv_cache_hits": 0, "kv_recomputes": 0, "mode": self.mode}

    def _owner(self):
        return (id(self), self.generation, self.request["identity"])

    def _config_key(self):
        return (tuple(self.request["grid_thw"]), self.request["spatial_merge_size"],
                self.max_tokens, self.spatial_bins, self.hidden_size)

    def _validate_donor(self, request):
        donor = request.get("donor_visual")
        if not isinstance(donor, dict) or not isinstance(donor.get("identity"), str) or not donor["identity"]:
            raise ValueError("wrong_video requires donor_visual with its original identity")
        if donor["identity"] == request["identity"]:
            raise ValueError("wrong_video donor identity must differ from current request")
        expected = (tuple(request["grid_thw"]), request["spatial_merge_size"],
                    self.max_tokens, self.spatial_bins, self.hidden_size)
        config = donor.get("config")
        if (not isinstance(config, (tuple, list)) or len(config) != len(expected)
                or tuple(config[1:]) != expected[1:]):
            raise ValueError("wrong_video donor config mismatch")
        donor_grid = config[0]
        merge = expected[1]
        if (not isinstance(donor_grid, (tuple, list)) or len(donor_grid) != 3
                or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in donor_grid)
                or donor_grid[1] % merge or donor_grid[2] % merge):
            raise ValueError("wrong_video donor grid must be independently valid")
        features, times = donor.get("features"), donor.get("times")
        if not isinstance(features, torch.Tensor) or not isinstance(times, torch.Tensor):
            raise ValueError("wrong_video donor needs feature/time tensors")
        if (features.ndim != 2 or features.shape[1] != self.hidden_size or times.ndim != 1
                or not 0 < len(features) <= self.max_tokens or len(times) != len(features)):
            raise ValueError("wrong_video donor shape mismatch")
        if not features.is_floating_point() or not times.is_floating_point():
            raise ValueError("wrong_video donor tensors must be floating point")
        if features.requires_grad or times.requires_grad:
            raise ValueError("wrong_video donor must be frozen")
        groups, bins = donor.get("groups"), donor.get("spatial_bins")
        if (not isinstance(groups, int) or not isinstance(bins, int) or min(groups, bins) < 1
                or groups * bins != len(features)):
            raise ValueError("wrong_video donor group layout mismatch")
        t, h, w = expected[0]
        expected_bins = min(self.spatial_bins, h // merge) * min(self.spatial_bins, w // merge)
        if bins != expected_bins or groups != min(t, self.max_tokens // expected_bins):
            raise ValueError("wrong_video donor group layout differs from current pooled memory")
        donor_t, donor_h, donor_w = donor_grid
        donor_bins = min(self.spatial_bins, donor_h // merge) * min(self.spatial_bins, donor_w // merge)
        if bins != donor_bins or groups != min(donor_t, self.max_tokens // donor_bins):
            raise ValueError("wrong_video donor group layout disagrees with its own grid")
        if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(times).all()):
            raise ValueError("wrong_video donor must be finite")
        if bool(((times < 0) | (times > 1)).any()):
            raise ValueError("wrong_video donor times must be normalized to [0,1]")
        if features.is_inference() or times.is_inference():
            with torch.inference_mode(False), torch.no_grad():
                donor["features"], donor["times"] = features.detach().clone(), times.detach().clone()
        self._donor_validated = self._donor_signature(donor)

    @staticmethod
    def _donor_signature(donor):
        # Shape/value checks are done at setup and whenever a donor changes,
        # not as a GPU-wide finite reduction on every decode round.
        features, times = donor["features"], donor["times"]
        return (id(donor), donor.get("identity"), donor.get("config"),
                id(features), features._version, id(times), times._version,
                donor.get("groups"), donor.get("spatial_bins"))

    def _visual(self, source, prefix_len):
        request = self.request
        prompt_mask = request["prompt_visual_mask"]
        # Training scores the first block just before the final prompt root is
        # committed, so source can be one (or a few text) rows shorter than the
        # full prompt. That is safe only when every visual position is already
        # present; otherwise evidence would be silently truncated.
        if prefix_len + source.shape[1] < len(prompt_mask):
            covered = min(prefix_len + source.shape[1], len(prompt_mask))
            if bool(prompt_mask[covered:].any()):
                raise ValueError("visual evidence cannot precede the complete known prompt (missing visual rows)")
        prepared = request.get("prepared_visual")
        if prepared is not None:
            if prepared.get("identity") != request["identity"] or prepared.get("config") != self._config_key():
                raise ValueError("prepared visual evidence identity/config mismatch")
            features, times = prepared["features"], prepared["times"]
            if features.ndim != 2 or features.shape[1] != self.hidden_size or len(times) != len(features):
                raise ValueError("prepared visual evidence shape mismatch")
            if features.requires_grad or times.requires_grad:
                raise ValueError("prepared visual evidence must be frozen")
            self._stats["visual_prepare_hits"] += 1
        else:
            mask = request["prompt_visual_mask"].to(source.device)
            n_known = min(prefix_len + source.shape[1], len(mask))
            if prefix_len != 0:
                raise ValueError("first visual evidence call must start at stream position zero")
            if bool(mask[n_known:].any()):
                raise ValueError("first visual evidence call must include the complete known prompt (missing visual rows)")
            t, h, w = request["grid_thw"]
            merge = request["spatial_merge_size"]
            h, w = h // merge, w // merge
            by, bx = min(self.spatial_bins, h), min(self.spatial_bins, w)
            if by * bx > self.max_tokens:
                raise ValueError("evidence_max_tokens is smaller than one spatial group")
            groups = min(t, self.max_tokens // (by * bx))
            visual = source[0, :n_known][mask[:n_known]].float().reshape(t, h, w, self.hidden_size)
            features = F.adaptive_avg_pool3d(visual.permute(3, 0, 1, 2)[None],
                                             (groups, by, bx))[0].permute(1, 2, 3, 0).reshape(-1, self.hidden_size)
            group_times = F.adaptive_avg_pool1d(request["time_values"].to(source.device).float()[None, None], groups)[0, 0]
            times = group_times.repeat_interleave(by * bx)
            # Deployment uses inference_mode; its tensors have no version counter
            # and cannot later be saved for a gradient-bearing scoring operation.
            # Requests contain ordinary frozen tensors so rollout -> scoring works.
            with torch.inference_mode(False), torch.no_grad():
                features, times = features.detach().clone(), times.detach().clone()
            prepared = {"identity": request["identity"], "config": self._config_key(),
                        "features": features, "times": times,
                        "groups": groups, "spatial_bins": by * bx}
            request["prepared_visual"] = prepared
        if prepared["features"].is_inference() or prepared["times"].is_inference():
            with torch.inference_mode(False), torch.no_grad():
                prepared["features"] = prepared["features"].detach().clone()
                prepared["times"] = prepared["times"].detach().clone()
        intervention = request.get("intervention", "normal")
        if intervention == "wrong_video":
            donor = request.get("donor_visual")
            if not isinstance(donor, dict):
                raise ValueError("wrong_video requires donor_visual")
            try:
                changed = self._donor_signature(donor) != self._donor_validated
            except (KeyError, AttributeError, RuntimeError):
                changed = True
            if changed:
                self._validate_donor(request)
            # Keep request identity and its own prepared features unchanged.
            prepared = donor
        features = prepared["features"].to(device=source.device, dtype=torch.float32)
        times = prepared["times"].to(device=source.device, dtype=torch.float32)
        if intervention == "time_shuffle":
            groups, bins = prepared["groups"], prepared["spatial_bins"]
            if groups < 2:
                raise ValueError("time_shuffle requires at least two temporal groups")
            generator = torch.Generator(device="cpu").manual_seed(int(request.get("seed", 0)))
            order = torch.randperm(groups, generator=generator)
            if torch.equal(order, torch.arange(groups)):
                order = order.roll(1)
            # Only times move: permuting features and times together is a no-op.
            times = times.reshape(groups, bins)[order.to(times.device)].reshape(-1)
        if intervention == "zero":
            features, times = torch.zeros_like(features), torch.zeros_like(times)
        return features, times

    def prepare(self, source, position_ids, cache, use_cache):
        if self.request is None:
            raise ValueError("call set_evidence_request before an evidence-enabled forward")
        if source.ndim != 3 or source.shape[0] != 1 or source.shape[-1] != self.hidden_size:
            raise ValueError("evidence currently requires one request and one target-layer feature")
        prefix_len = int(cache.get_seq_length()) if cache is not None else 0
        if position_ids.ndim != 2 or position_ids.shape[0] != 1 or position_ids.shape[1] < source.shape[1]:
            raise ValueError("evidence needs 2-D absolute stream positions")
        expected = torch.arange(prefix_len, prefix_len + position_ids.shape[1], device=position_ids.device)[None]
        if not torch.equal(position_ids, expected):
            raise ValueError("evidence positions disagree with draft cache")
        previous = getattr(cache, "_glance_evidence_context", None) if cache is not None else None
        if prefix_len and (previous is None or previous["owner"] != self._owner()):
            raise ValueError("evidence requires its own populated request cache")
        if prefix_len and previous["end"] < prefix_len:
            raise ValueError("evidence context is shorter than committed KV; draft noise cannot become evidence")
        source = source.detach().float()
        end = prefix_len + source.shape[1]
        state = {"owner": self._owner(), "end": end}
        if self.mode == "visual":
            features, times = self._visual(source, prefix_len)
        else:
            positions = torch.arange(prefix_len, end, device=source.device)
            mask = self.request["prompt_visual_mask"].to(source.device)
            keep = torch.ones(len(positions), device=source.device, dtype=torch.bool)
            inside = positions < len(mask)
            keep[inside] = ~mask[positions[inside]]
            positions, features = positions[keep], source[0, keep]
            if prefix_len:
                old_keep = previous["positions"] < prefix_len
                positions = torch.cat((previous["positions"][old_keep], positions))
                features = torch.cat((previous["features"][old_keep], features))
            if not len(features):
                raise ValueError("text evidence needs at least one known non-visual context token")
            state.update(positions=positions, features=features)
            times = positions.float() / max(end - 1, 1)
            count = min(len(features), self.max_tokens)
            features = F.adaptive_avg_pool1d(features.T[None], count)[0].T
            times = F.adaptive_avg_pool1d(times[None, None], count)[0, 0]
            if self.request.get("intervention", "normal") == "zero":
                features, times = torch.zeros_like(features), torch.zeros_like(times)
        # Only static visual memory can reuse trainable projections in no-grad.
        if self.mode == "visual" and previous is not None and previous["owner"] == self._owner():
            state["projected"] = previous.get("projected")
        if use_cache and cache is not None:
            cache._glance_evidence_context = state
        return features, times, state

    @staticmethod
    def _norm(module, value):
        return F.layer_norm(value.float(), module.normalized_shape,
                            module.weight.float(), module.bias.float(), module.eps)

    @staticmethod
    def _linear(module, value):
        return F.linear(value.float(), module.weight.float(), None)

    def forward(self, hidden, prepared):
        features, times, state = prepared
        self._stats["calls"] += 1
        self._stats["memory_tokens"] = len(features)
        if self.request.get("intervention", "normal") == "bypass":
            return hidden
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            signature = (tuple(p._version for p in self.parameters()),
                         id(self.request.get("prepared_visual")), str(hidden.device),
                         self.request.get("intervention", "normal"), int(self.request.get("seed", 0)))
            if self.mode == "visual":
                frozen = (self.request["donor_visual"] if self.request.get("intervention") == "wrong_video"
                          else self.request["prepared_visual"])
                signature += (id(frozen), frozen["identity"], id(frozen["features"]), frozen["features"]._version,
                              id(frozen["times"]), frozen["times"]._version)
            cached = state.get("projected")
            can_cache = self.mode == "visual" and not torch.is_grad_enabled()
            if can_cache and cached is not None and cached[0] == signature:
                keys, values = cached[1:]
                self._stats["kv_cache_hits"] += 1
            else:
                memory = self._norm(self.memory_norm, features)
                time_features = torch.stack((times, torch.sin(2 * math.pi * times),
                    torch.cos(2 * math.pi * times), torch.sin(4 * math.pi * times),
                    torch.cos(4 * math.pi * times), torch.ones_like(times)), dim=-1)
                keys = self._linear(self.key, memory) + self._linear(self.time, time_features)
                values = self._linear(self.value, memory)
                self._stats["kv_recomputes"] += 1
                if can_cache:
                    state["projected"] = (signature, keys.detach(), values.detach())
            query = self._linear(self.query, self._norm(self.query_norm, hidden))
            weights = torch.softmax(torch.matmul(query, keys.T) / math.sqrt(keys.shape[-1]), dim=-1)
            residual = self._linear(self.output, torch.matmul(weights, values))
            gate = torch.sigmoid(self.gate.float())
            self._last_residual_rms = residual.detach().square().mean().sqrt()
            return (hidden.float() + gate * residual).to(hidden.dtype)

    def diagnostics(self):
        result = dict(self._stats)
        result["gate"] = float(torch.sigmoid(self.gate.detach().float()).cpu())
        if hasattr(self, "_last_residual_rms"):
            result["residual_rms"] = float(self._last_residual_rms.cpu())
        return result
