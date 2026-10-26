#!/usr/bin/env python3
"""fp32 master weights for a bf16 parameter set.

WHY THIS EXISTS (measured, 2026-10-10). Both trainers built
`torch.optim.AdamW(head.parameters())` with the head loaded in bf16 and no
master-weight copy, so Adam's `exp_avg`/`exp_avg_sq` were bf16 and the
`addcdiv_` that applies the update ran in bf16. `repro/diagnostics/diag_update_precision.py`
measured, from the real aligned checkpoint on a real training batch:

    lr 3e-5   realized update norm / fp32 ideal  =  0.223   (78% of the step lost)
              96.9% of parameter elements receive an update smaller than half a
              bf16 ulp, so they round to a bit-identical value
    lr 3e-4   ratio = 1.042   (first step lands fine)

Because bf16 parameters cannot absorb a sub-half-ulp step, and without a master
copy the discarded steps never accumulate, a small-learning-rate run can look
like "gentle optimisation" while actually being "almost no optimisation": the
low-lr calibration's tau was bit-identical to the start point on 26 of 29 videos.

`MasterWeightAdamW` keeps fp32 master parameters and fp32 Adam state, steps them
in fp32, and writes the result back into the bf16 parameters that the forward
pass uses. `where="cpu"` offloads the master and the Adam moments, which is what
fits beside the fp32 teacher; `where="gpu"` is faster when memory allows.

Usage:
    opt = MasterWeightAdamW(head, lr=..., betas=..., weight_decay=..., where="cpu")
    ... loss.backward() ...
    opt.step()          # syncs grads, steps fp32 master, writes back to bf16
    opt.zero_grad(set_to_none=True)

No try/except: a failure surfaces with its traceback.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import torch


class MasterWeightAdamW:
    """AdamW on fp32 master parameters, mirrored into the module's own params."""

    def __init__(self, module, *, lr, betas, weight_decay, where="cpu",
                 eps=1e-8, clip=None):
        if where not in ("cpu", "gpu"):
            raise ValueError(f"where must be 'cpu' or 'gpu', got {where!r}")
        self.module = module
        self.where = where
        self.clip = clip
        self._names = [n for n, _ in module.named_parameters()]
        dev = torch.device("cpu") if where == "cpu" else next(
            module.parameters()).device
        self._master = {
            n: p.detach().to(dev, torch.float32).clone()
            for n, p in module.named_parameters()}
        self._opt = torch.optim.AdamW([self._master[n] for n in self._names],
                                      lr=lr, betas=betas,
                                      weight_decay=weight_decay, eps=eps)
        self._dev = dev
        self._external_grads_ready = False

    # ------------------------------------------------------------------ state
    @property
    def master(self):
        """The fp32 parameters, for checkpointing or inspection."""
        return self._master

    def master_state_bytes(self):
        """Bytes held by master params + Adam moments (both fp32)."""
        n = sum(v.numel() for v in self._master.values())
        return 4 * n + 4 * 2 * n      # params + exp_avg + exp_avg_sq

    def learning_rates(self):
        return [g["lr"] for g in self._opt.param_groups]

    def set_lr(self, lr):
        for g in self._opt.param_groups:
            g["lr"] = lr

    # ------------------------------------------------------------------- step
    def zero_grad(self, set_to_none=True):
        self._external_grads_ready = False
        for _, p in self.module.named_parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_().zero_()

    def _module_params(self):
        return dict(self.module.named_parameters())

    @torch.no_grad()
    def sync_grads(self):
        """Copy the module's (bf16) gradients into the fp32 master's .grad.

        Separate from `step()` so a caller can read the PRE-CLIP norm in fp32
        before anything is scaled. Gradients are the only quantity read out of
        the bf16 module; the update itself accumulates in fp32.
        """
        self._external_grads_ready = False
        mods = self._module_params()
        for n in self._names:
            g = mods[n].grad
            self._master[n].grad = (None if g is None
                                    else g.detach().to(self._dev, torch.float32))

    @torch.no_grad()
    def set_external_master_grads(self, gradients):
        """Install already-normalized gradients, preserving globally absent None."""
        if set(gradients) != set(self._names):
            raise ValueError("external gradient names differ")
        for n, g in gradients.items():
            if g is not None and (g.shape != self._master[n].shape or
                                  g.dtype != torch.float32 or g.requires_grad):
                raise ValueError("external gradient shape/dtype/autograd mismatch")
        for n, g in gradients.items():
            self._master[n].grad = None if g is None else g.to(self._dev).clone()
        self._external_grads_ready = True

    def grad_norm(self):
        """Total L2 norm of the fp32 master gradients (no clipping applied)."""
        gs = [self._master[n].grad for n in self._names
              if self._master[n].grad is not None]
        if not gs:
            raise RuntimeError("no parameter received a gradient; the objective "
                               "is not connected to the head")
        return float(torch.nn.utils.get_total_norm(gs, foreach=False))

    @torch.no_grad()
    def write_back(self):
        """Mirror the fp32 master into the module's own (bf16) parameters.

        This is a lossy round-trip ON PURPOSE and it is where the single-step
        direction error comes from: a step smaller than half an ulp of the
        current value cannot be represented in the bf16 forward parameter. The
        point of the master copy is that such a step is NOT lost -- it stays in
        the fp32 master and accumulates across steps.
        """
        for n, p in self.module.named_parameters():
            p.copy_(self._master[n].to(p.device, p.dtype))

    @torch.no_grad()
    def step(self, *, collect_update_norms=True):
        """Sync grads if needed, clip in fp32, step, write back.

        Returns `{name: update norm}` where the norm is the change actually
        written into the bf16 parameter. Call `sync_grads()` first if you want
        to read `grad_norm()` before clipping; `step()` also syncs when a master
        gradient is missing, so it is safe on its own.
        """
        if not self._external_grads_ready and any(self._master[n].grad is None for n in self._names):
            self.sync_grads()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(list(self._master.values()), self.clip)
        self._opt.step()
        mods = self._module_params()
        norms = {}
        for n, p in mods.items():
            m = self._master[n]
            new = m.to(p.device, p.dtype)
            if collect_update_norms:
                norms[n] = float((new.float() - p.float()).norm())
            p.copy_(new)
        self._external_grads_ready = False
        return norms

    # ------------------------------------------------------- exact resumption
    def state_dict(self):
        """fp32 master params + Adam moments, so training can be resumed exactly.

        A bf16 `save_pretrained` of the head is enough to EVALUATE, but not to
        resume: the master's own fp32 value and Adam's moments are part of the
        optimizer state, and a step smaller than half a bf16 ulp lives only
        there. Saving just the head would silently restart the accumulation.
        """
        return {"master": {n: v.detach().cpu() for n, v in self._master.items()},
                "optimizer": self._opt.state_dict(),
                "where": self.where, "clip": self.clip}

    def save_master(self, path):
        torch.save(self.state_dict(), path)
        return path

    def load_master(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return self.load_state_dict(blob)

    def load_state_dict(self, blob):
        """Restore the fp32 master, its Adam moments and the bf16 mirror.

        `state_dict()`'s counterpart. It exists as a named method because both
        trainers' resume paths call `opt.load_state_dict(state["master_optimizer"])`
        and the earlier class only offered `load_master(path)`, so a resumed run
        died with `AttributeError` after the model had already loaded -- a
        failure that only shows up on the resume path, which is exactly the path
        that must work. The blob is validated by name set: a mismatch means the
        checkpoint belongs to a different module and silently copying it would
        corrupt the run.
        """
        if set(blob["master"]) != set(self._master):
            raise ValueError("master state does not match this module's tensors")
        with torch.no_grad():
            for n, v in blob["master"].items():
                self._master[n].copy_(v.to(self._dev, torch.float32))
        self._opt.load_state_dict(blob["optimizer"])
        if "where" in blob and blob["where"] != self.where:
            raise ValueError(f"checkpoint was written with master weights on "
                             f"{blob['where']!r}, this run uses {self.where!r}")
        self.write_back()
        return blob.get("where")


def master_change(before, after):
    """Total L2 of the fp32 master's change, for 'did the step land' checks."""
    tot = 0.0
    for n in before:
        tot += float((after[n] - before[n]).float().norm()) ** 2
    return tot ** 0.5
