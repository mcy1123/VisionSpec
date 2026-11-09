#!/usr/bin/env python3
"""Single-process 9k-target video OPD/E2E-TV trainer.

Target is fp32, sharded with accelerate ``device_map='balanced'`` across the
visible GPUs. The draft head, master weights and all trace tensors live on
cuda:0. This is the high-token diagnostic path, not the default three-rank
video launcher; it is intentionally explicit and refuses to run without
``--device-map``.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from shared import common, training_resume
from shared.master_optim import MasterWeightAdamW
from opd_official_kl import adapters, runtime


class _MoveOut(nn.Module):
    def __init__(self, inner, dev):
        super().__init__()
        self.inner = inner
        self.dev = dev

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        for name in ("last_hidden_state", "logits"):
            value = getattr(out, name, None)
            if value is not None:
                setattr(out, name, value.to(self.dev))
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            out.hidden_states = tuple(h.to(self.dev) for h in hs)
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


def wrap_target(target, dev):
    lm = target.lm_head
    class _LM(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            return self.inner(x.to(self.inner.weight.device)).to(x.device)
        @property
        def weight(self):
            return self.inner.weight
    target.lm_head = _LM(lm)
    target.model = _MoveOut(target.model, dev)
    target.forward = _MoveOut(target.forward, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--qa-per-update", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--e2e-tv-weight", type=float, default=10.0)
    ap.add_argument("--device-map", default="balanced")
    ap.add_argument("--master-weights", default="cpu")
    ap.add_argument("--save-every", type=int, default=30)
    a = ap.parse_args()
    if not a.device_map:
        raise SystemExit("9k trainer requires --device-map")

    dev = "cuda:0"
    torch.manual_seed(a.seed)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("new output must be empty")
    out.mkdir(parents=True, exist_ok=True)

    target, processor = common.load_target(device=dev, dtype=torch.float32,
                                           device_map=a.device_map)
    wrap_target(target, dev)
    target.eval().requires_grad_(False)
    print(f"[9k] target map={getattr(target, 'hf_device_map', None)}", flush=True)

    head = common.load_head(a.init_from, device=dev, dtype=torch.bfloat16).eval()
    head.requires_grad_(True)
    optimizer = MasterWeightAdamW(head, lr=a.lr, betas=common.BETAS,
                                  weight_decay=common.WEIGHT_DECAY,
                                  where=a.master_weights, clip=a.clip)

    rows = pd.read_parquet(a.pool).set_index("task", verify_integrity=True)
    schedule = {}
    for line in Path(a.schedule).read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            schedule.setdefault(int(item["update"]), []).append(item)
    if sorted(schedule) != list(range(a.updates)):
        raise SystemExit("schedule must cover exactly all updates")

    for cursor in range(a.updates):
        started = time.perf_counter()
        head.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        numerator = 0.0
        denominator = 0.0
        tasks = []
        input_token_counts = []
        for item in sorted(schedule[cursor], key=lambda x: x["slot"]):
            task = item["task"]
            tasks.append(task)
            row = rows.loc[task].to_dict()
            head.visual_memory = None
            inputs = adapters.encode(processor, row, "video", max_pixels=a.max_pixels,
                                     time_mode="legacy")
            input_token_counts.append(int(inputs["input_ids"].numel()))
            head.eval()
            trace = adapters.collect(target, head, inputs,
                                     max_new_tokens=a.max_new_tokens,
                                     max_seq_len=a.max_seq_len,
                                     teacher_verify_impl="pure_block_native")
            denom = runtime.denominator(trace)
            denominator += denom
            for stat in runtime.score(target, head, trace,
                                     draft_fn=adapters.draft,
                                     e2e_tv_weight=a.e2e_tv_weight):
                if not stat.numerator.requires_grad or stat.numerator.grad_fn is None:
                    raise RuntimeError("missing student graph")
                (stat.numerator / denom).backward()
                numerator += float(stat.numerator.detach())
            del trace
        optimizer.sync_grads()
        norm = optimizer.grad_norm()
        if not math.isfinite(norm):
            raise RuntimeError("non-finite gradient")
        optimizer.step()
        record = dict(update=cursor + 1, optimizer_step=cursor + 1,
                      tasks=tasks, loss=numerator / denominator,
                      denominator=denominator, gradient_norm=norm,
                      elapsed_seconds=time.perf_counter() - started,
                      target_input_tokens=sum(input_token_counts))
        with (out / "log.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if (cursor + 1) % a.save_every == 0 or cursor + 1 == a.updates:
            training_resume.save_atomic_checkpoint(
                out / f"step{cursor+1}", head,
                dict(head=head.state_dict(), master_optimizer=optimizer.state_dict(),
                     update=cursor + 1, data_cursor=cursor + 1,
                     diagnostic_only=True, smoke=False, opd_mode="pure_block",
                     block_native=True, teacher_verify_impl="pure_block_native"))


if __name__ == "__main__":
    main()
