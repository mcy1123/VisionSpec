"""Video-projector feature capture and paper-module selection for the GPU6 comparison."""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from .evidence import EvidenceReadout
from .paper_modules import FlamingoReadout, tome_bipartite
from .paper_vispec import PaperViSpecReadout

METHODS = ("vispec", "fovea", "flamingo", "tome", "llava_pool", "projector_grid")
FEATURE_SOURCE = "qwen3_vl_vision_merger_output"


def avg_grid(grid, height, width):
    return F.adaptive_avg_pool2d(grid.permute(0, 3, 1, 2), (height, width)).permute(0, 2, 3, 1)


def prepare_projector(readout, raw, request):
    method = readout.paper_method
    nt, nh, nw = request["grid_thw"]
    merge = request["spatial_merge_size"]
    h, w = nh // merge, nw // merge
    if raw.shape != (nt * h * w, readout.hidden_size):
        raise ValueError("vision merger output disagrees with this video's visual grid")
    grid = raw.detach().reshape(nt, h, w, -1).float()
    clock = request["time_values"].to(raw.device).float()
    if method in ("projector_grid", "llava_pool"):
        if method == "llava_pool":
            # LLaVA-OneVision, Appendix C.1, stride-2 spatial bilinear downsampling.
            grid = F.interpolate(grid.permute(0, 3, 1, 2), size=(math.ceil(h / 2), math.ceil(w / 2)),
                                 mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        # Shared draft memory cap and the same time grouping as the historical grid arm.
        by = min(readout.spatial_bins, grid.shape[1])
        bx = min(readout.spatial_bins, grid.shape[2])
        count = min(nt, readout.max_tokens // (by * bx))
        if count < 1:
            raise ValueError("draft grid budget is smaller than one time group")
        grid = F.adaptive_avg_pool3d(grid.permute(3, 0, 1, 2)[None], (count, by, bx))[0].permute(1, 2, 3, 0)
        clock = F.adaptive_avg_pool1d(clock[None,None], count).flatten()
    elif method == "tome":
        # Bound the quadratic comparison just once during prefill.
        grid = avg_grid(grid, min(4, h), min(4, w)).flatten(1, 2)
        features, times = tome_bipartite(grid, clock, readout.max_tokens)
        order = times.argsort(stable=True)
        return features[order], times[order], len(times), 1
    grid = grid.reshape(len(clock), -1, readout.hidden_size)
    return grid.flatten(0, 1), clock.repeat_interleave(grid.shape[1]), len(clock), grid.shape[1]


@contextmanager
def capture_projector(target, head, request):
    module = getattr(head, "evidence_readout", None)
    if not getattr(module, "paper_method", None):
        yield
        return
    if request is None or request is not module.request or "prepared_visual" in request:
        raise ValueError("vision projector tap requires a new bound video request")
    if request.get("intervention") not in ("normal", "bypass"):
        raise ValueError("projector evaluation supports normal and bypass")
    captures = []
    def capture(vision, args, kwargs, output):
        if request is not module.request or captures:
            raise RuntimeError("duplicate vision encoder call or request changed during prefill")
        thw = kwargs["grid_thw"]
        if tuple(thw.shape) != (1, 3) or thw[0].tolist() != list(request["grid_thw"]):
            raise ValueError("vision encoder processed a different video grid")
        with torch.inference_mode(False), torch.no_grad():
            data, times, groups, bins = prepare_projector(module, output[0], request)
            request["prepared_visual"] = dict(identity=request["identity"], config=module._config_key(),
                    features=data.detach().clone(), times=times.detach().clone(),
                    groups=groups, spatial_bins=bins, feature_source=FEATURE_SOURCE)
        captures.append(True)
    handle = target.model.visual.register_forward_hook(capture, with_kwargs=True)
    try:
        yield
        if len(captures) != 1:
            raise RuntimeError("vision merger was not captured during target prefill")
    finally:
        handle.remove()


class GatedPaperReadout(EvidenceReadout):
    def __init__(self, method, hidden_size, dim, max_tokens, spatial_bins, mode):
        nn.Module.__init__(self)
        self.mode, self.hidden_size = mode, hidden_size
        self.max_tokens, self.spatial_bins = max_tokens, spatial_bins
        self.request, self.generation, self._stats = None, 0, {}
        self.time = nn.Linear(6, dim, bias=False)
        if method == "flamingo":
            self.adapter = FlamingoReadout(hidden_size, dim, use_time_embeddings=False)
        else:
            from .paper_modules import FoveaReadout
            self.adapter = FoveaReadout(hidden_size, dim)
        self.paper_method = method
        self.reset_output()

    def reset_output(self):
        with torch.no_grad():
            self.time.weight.zero_()
            if self.paper_method == "flamingo":
                self.adapter.attn_gate.zero_()
                self.adapter.ff_gate.zero_()
            else:
                self.adapter.reset_output()

    def forward(self, hidden, prepared):
        features, times, state = prepared
        self._stats["calls"] += 1
        if self.request.get("intervention") == "bypass":
            return hidden
        source = self.request["prepared_visual"]
        cached = state.get("projected")
        signature = (self.generation, id(source), tuple(p._version for p in self.adapter.parameters()))
        cacheable = not torch.is_grad_enabled()
        if cacheable and cached is not None and cached[0] == signature:
            kv = cached[1]
            self._stats["kv_cache_hits"] += 1
        else:
            groups, bins = source["groups"], source["spatial_bins"]
            kv = self.adapter.prepare_memory(features.reshape(groups, bins, -1), times.reshape(groups, bins))
            self._stats["kv_recomputes"] += 1
            if cacheable:
                state["projected"] = (signature, tuple(x.detach() for x in kv))
        self._stats.update(memory_tokens=kv[0].shape[1], source_tokens=len(features))
        return self.adapter(hidden, kv)

    def diagnostics(self):
        return dict(self._stats)


def paper_readout(method, **kwargs):
    if method not in METHODS or kwargs["mode"] != "visual":
        raise ValueError(f"unsupported paper readout {method}")
    if method == "vispec":
        module = PaperViSpecReadout(**kwargs)
    elif method in ("flamingo", "fovea"):
        module = GatedPaperReadout(method, **kwargs)
    else:
        module = EvidenceReadout(**kwargs)
    module.paper_method = method
    return module
