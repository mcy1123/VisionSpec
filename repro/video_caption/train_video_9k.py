#!/usr/bin/env python3
"""Single-process 9k-target video OPD/E2E-TV trainer.

Target is fp32, sharded with accelerate ``device_map='balanced'`` across the
visible GPUs. The draft head and active trace live on cuda:0; global mode
stages the remaining traces on CPU, and master weights default to CPU.
This is the high-token diagnostic path, not the default three-rank
video launcher; it is intentionally explicit and refuses to run without
``--device-map``. Initialization always loads head weights only and creates a
fresh optimizer; this entry does not resume training state. Historical
``legacy_per_qa`` normalization and cache-relative time remain explicit defaults.
Use ``--loss-normalization global_pooled --time-mode real`` for the new protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from shared import common, training_resume
from shared.master_optim import MasterWeightAdamW
from shared.model_contract import validate_model_contract
from shared.evidence import build_evidence_request, configure_evidence, evidence_settings
from opd_official_kl import adapters, runtime
from opd_official_kl.trainer import move_trace


LOSS_NORMALIZATIONS = ("legacy_per_qa", "global_pooled")


def parse_deadline(value):
    """Require an absolute, timezone-aware instant; never guess server time."""
    if value is None:
        return None
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("--deadline-utc must include an explicit timezone, e.g. 2026-11-12T00:00:00Z")
    return deadline.astimezone(timezone.utc)


def stop_reason(deadline=None, stop_file=None, *, now=None):
    """An update-boundary stop, not an interrupt in backward/optimizer/save."""
    if stop_file and Path(stop_file).exists():
        return "stop_file"
    current = now if now is not None else datetime.now(timezone.utc)
    if deadline is not None and current >= deadline:
        return "deadline"
    return None


def checkpoint_and_status(out, step, budget, reason, save_fn):
    """Preserve the last completed update once, including a non-periodic stop."""
    out = Path(out)
    if step > 0 and not (out / f"step{step}").exists():
        save_fn(step)
    result = dict(status="STOPPED" if reason else "COMPLETED",
                  stop_reason=reason, completed_updates=step, planned_updates=budget,
                  resume_supported=False,
                  last_checkpoint=str(out / f"step{step}") if step else None,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    # A single final status; never repurpose an older run's output directory.
    with (out / "RUN_STATUS.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    return result


def load_schedule(path, rows, updates, qa_per_update):
    """Validate the entire finite schedule before loading any model."""
    if updates < 1 or qa_per_update < 1:
        raise ValueError("updates and qa-per-update must be positive")
    schedule = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        for field in ("update", "slot"):
            if not isinstance(item.get(field), int) or isinstance(item[field], bool):
                raise ValueError(f"schedule line {number}: {field} must be an integer")
        task = item.get("task")
        if task not in rows.index:
            raise ValueError(f"schedule line {number}: unknown task {task!r}")
        prompt_sha = hashlib.sha256(rows.loc[task, "prompt"].encode()).hexdigest()
        if item.get("prompt_sha256", prompt_sha) != prompt_sha:
            raise ValueError(f"schedule line {number}: prompt hash mismatch for {task}")
        schedule.setdefault(item["update"], []).append(item)
    if sorted(schedule) != list(range(updates)):
        raise ValueError("schedule must cover exactly all updates, numbered from zero")
    for update, items in schedule.items():
        items.sort(key=lambda item: item["slot"])
        if [item["slot"] for item in items] != list(range(qa_per_update)):
            raise ValueError(f"update {update} must contain slots 0..{qa_per_update - 1} exactly once")
    return schedule


def validate_rows(rows, schedule, time_mode):
    """Validate inputs without decoding frames or touching CUDA."""
    from qa_opd_fair.videoencode import validate_real_metadata

    diagnostics = []
    cache_hashes = {}
    tasks = list(dict.fromkeys(item["task"] for items in schedule.values() for item in items))
    for task in tasks:
        row = rows.loc[task].to_dict()
        if row.get("modality", "video") != "video":
            raise ValueError(f"{task}: this entry accepts video rows only")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{task}: nonempty prompt required")
        if time_mode == "real":
            diagnostics.append(dict(task=task, **validate_real_metadata(row, require_exact=True)))
        else:
            if not isinstance(row.get("cache_path"), str) or not Path(row["cache_path"]).is_file():
                raise ValueError(f"{task}: cache_path must identify an existing file")
            if Path(row["cache_path"]).suffix.lower() == ".npz":
                raise ValueError(f"{task}: NPZ frame caches require --time-mode real")
            fps = float(row.get("fps", 0))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"{task}: positive cache fps required")
        cache_path = str(Path(row["cache_path"]).resolve())
        if cache_path not in cache_hashes:
            cache_hashes[cache_path] = training_resume.file_hash(cache_path)
        expected_hash = row.get("cache_hash")
        if isinstance(expected_hash, str) and expected_hash:
            if expected_hash != cache_hashes[cache_path]:
                raise ValueError(f"{task}: cache_hash mismatch for {cache_path}")
    return dict(scheduled_unique_tasks=len(tasks), time_mode=time_mode,
                real_timeline_diagnostics=diagnostics, cache_sha256=cache_hashes)


def validate_training_head(head_path, time_mode=None, loss_normalization=None):
    """Keep old external-memory heads closed; explicitly gate the new readout."""
    config = json.loads((Path(head_path) / "config.json").read_text())
    draft = config.get("dflash_config", {})
    unsupported = [name for name in ("use_visual_fovea", "use_visual_film",
                                      "use_visual_xattn", "use_visual_perceiver")
                   if draft.get(name, config.get(name, False))]
    if unsupported:
        raise ValueError("9k trainer has no external visual-memory input; refusing to silently disable "
                         + ", ".join(unsupported)
                         + ". A shared real-video memory interface is required before these arms can train.")
    settings = evidence_settings(config)
    if settings["mode"] != "none":
        if any(draft.get(name, config.get(name, False)) for name in ("use_xattn", "use_layer_gate")):
            raise ValueError("evidence readout must be tested separately from legacy adapters")
        if time_mode != "real" or loss_normalization != "global_pooled":
            raise ValueError("evidence heads require --time-mode real --loss-normalization global_pooled")
    return settings


def move_evidence_tree(value, device):
    """Stage frozen request features recursively, without sharing mutable dicts."""
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: move_evidence_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_evidence_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_evidence_tree(item, device) for item in value)
    return value


def move_training_trace(trace, device):
    moved = move_trace(trace, device)
    if "evidence_request" in trace:
        moved["evidence_request"] = move_evidence_tree(trace["evidence_request"], device)
    return moved


def score_training_trace(target, head, trace, *, e2e_tv_weight, diagnostics=None):
    """Rebind each QA before scoring, never inherit the last collected video."""
    if evidence_settings(head)["mode"] != "none":
        if "evidence_request" not in trace:
            raise ValueError("evidence training trace has no request identity")
        configure_evidence(head, trace["evidence_request"])
    yield from runtime.score(target, head, trace, draft_fn=adapters.draft,
                             e2e_tv_weight=e2e_tv_weight)
    if diagnostics is not None and evidence_settings(head)["mode"] != "none":
        diagnostics.append(head.evidence_diagnostics())


def evidence_gradient_diagnostics(head):
    module = getattr(head, "evidence_readout", None)
    if module is None:
        return None
    gradients = {name: parameter.grad for name, parameter in module.named_parameters()}
    norms = {name: float(gradient.detach().float().norm())
             for name, gradient in gradients.items() if gradient is not None}
    if not norms:
        raise RuntimeError("evidence readout received no gradients")
    if not all(math.isfinite(value) for value in norms.values()):
        raise RuntimeError("non-finite evidence readout gradient")
    return dict(gradient_l2=math.sqrt(sum(value * value for value in norms.values())),
                gradient_norms=norms,
                missing_gradients=[name for name, value in gradients.items() if value is None],
                nonzero_gradient_parameters=sum(value > 0 for value in norms.values()),
                parameter_count=sum(parameter.numel() for parameter in module.parameters()),
                parameter_dtypes=sorted({str(parameter.dtype) for parameter in module.parameters()}))


def backward_update(traces, score_fn, normalization, *, denominator_fn=runtime.denominator,
                    stage_trace=lambda trace: trace, restore_trace=lambda trace: trace):
    """Accumulate the requested objective, with one rollout per QA.

    The base response/rejected denominator is intentional: the auxiliary loss
    in runtime.score is additive, as in the existing three-card trainer. Its
    augmented Pool.denominator is NOT the normalization used by this protocol.
    Global mode stages detached traces on CPU before scoring, keeping only one
    restored trace on the GPU. Legacy mode preserves collect/score interleaving.
    """
    if normalization not in LOSS_NORMALIZATIONS:
        raise ValueError(f"unknown loss normalization {normalization!r}")
    staged = None
    if normalization == "global_pooled":
        staged = []
        for trace in traces:
            denom = float(denominator_fn(trace))
            if not math.isfinite(denom) or denom <= 0:
                raise ValueError("each trace must have a positive finite base denominator")
            staged.append((stage_trace(trace), denom))
            del trace
        total_denom = sum(denom for _, denom in staged)
        if not staged:
            raise ValueError("an update must contain at least one QA")
        work = staged
    else:
        total_denom = None
        work = traces
    numerator = optimized_loss = 0.0
    denominators, qa_losses = [], []
    for item in work:
        if staged is not None:
            stored_trace, denom = item
            trace = restore_trace(stored_trace)
        else:
            trace = item
            denom = float(denominator_fn(trace))
        if not math.isfinite(denom) or denom <= 0:
            raise ValueError("each trace must have a positive finite base denominator")
        divisor = total_denom if total_denom is not None else denom
        qa_numerator = 0.0
        for stat in score_fn(trace):
            if not stat.numerator.requires_grad or stat.numerator.grad_fn is None:
                raise RuntimeError("missing student graph")
            value = float(stat.numerator.detach())
            if not math.isfinite(value):
                raise RuntimeError("non-finite objective numerator")
            (stat.numerator / divisor).backward()
            qa_numerator += value
        numerator += qa_numerator
        optimized_loss += qa_numerator / divisor
        denominators.append(denom)
        qa_losses.append(qa_numerator / denom)
        del trace, item
    if not denominators:
        raise ValueError("an update must contain at least one QA")
    denominator = sum(denominators)
    return dict(loss=optimized_loss, optimized_loss=optimized_loss,
                pooled_loss=numerator / denominator, numerator=numerator,
                denominator=denominator, qa_denominators=denominators,
                qa_normalized_losses=qa_losses, loss_normalization=normalization)


def make_protocol(args, input_diagnostics, model_dimensions):
    init = Path(args.init_from)
    weights = sorted(init.glob("*.safetensors")) + sorted(init.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("init-from must contain head weight files; no training-state resume is supported")
    config = {key: value for key, value in vars(args).items() if key not in ("out", "dry_run")}
    protocol = dict(
        schema=3, entry="video_caption_9k_diagnostic", diagnostic_only=True,
        opd_mode="pure_block", teacher_verify_impl="pure_block_native",
        initialization="weights_only_fresh_optimizer", resume_supported=False,
        optimizer_step_start=0, world_size=1,
        slot_map={"0": list(range(args.qa_per_update))},
        target_dtype="float32", head_dtype="bfloat16", master_dtype="float32",
        optimizer=dict(name="AdamW", betas=list(common.BETAS), weight_decay=common.WEIGHT_DECAY),
        loss_contract=dict(
            normalization=args.loss_normalization,
            formula=("sum_i N_i / sum_i D_i" if args.loss_normalization == "global_pooled"
                     else "sum_i (N_i / D_i); no division by QA count"),
            denominator="runtime.denominator: response count + 2 * rejected offset decay",
            auxiliary="additive top8-overlap prefix-product proxy; not full-vocabulary TV",
            score_pool_denominator_used=False),
        config=config, model_dimensions=model_dimensions,
        evidence=evidence_settings(init),
        evidence_parameter_dtype=("float32" if evidence_settings(init)["mode"] != "none" else None),
        evidence_learning_rate=(args.lr if evidence_settings(init)["mode"] != "none" else None),
        inputs=input_diagnostics,
        hashes=dict(pool=training_resume.file_hash(args.pool),
                    schedule=training_resume.file_hash(args.schedule),
                    target_config=training_resume.file_hash(Path(args.target) / "config.json"),
                    head_config=training_resume.file_hash(init / "config.json"),
                    initialization_manifest=(training_resume.file_hash(init / "INIT_MANIFEST.json")
                                             if (init / "INIT_MANIFEST.json").is_file() else None),
                    initial_head_weights={p.name: training_resume.file_hash(p) for p in weights},
                    trainer=training_resume.file_hash(__file__),
                    runtime=training_resume.file_hash(runtime.__file__),
                    draft_model=training_resume.file_hash(
                        REPRO.parent / "scripts/dflash_pkg/dflash-0.1.0/dflash/model.py"),
                    evidence_readout=training_resume.file_hash(
                        REPRO.parent / "scripts/dflash_pkg/dflash-0.1.0/dflash/evidence.py"),
                    evidence_helper=training_resume.file_hash(REPRO / "shared/evidence.py")),
    )
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def checkpoint_identity(protocol, step):
    """Small, explicit training identity; never infer a step from its directory."""
    config = protocol["config"]
    hashes = protocol["hashes"]
    common_initial = hashes["initial_head_weights"]
    initialization_manifest = hashes.get("initialization_manifest")
    if protocol["evidence"]["mode"] != "none":
        common_initial = None
        if initialization_manifest:
            manifest_path = Path(config["init_from"]) / "INIT_MANIFEST.json"
            if training_resume.file_hash(manifest_path) != initialization_manifest:
                raise ValueError("initialization manifest changed since training preflight")
            manifest = json.loads(manifest_path.read_text())
            source = manifest.get("source_weights_sha256")
            if not isinstance(source, dict) or not source or not all(
                isinstance(name, str) and isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for name, value in source.items()
            ):
                raise ValueError("initialization manifest lacks common source weight SHA256")
            if manifest.get("base_tensors_verified_equal") is not True:
                raise ValueError("initialization manifest does not verify common base tensors")
            common_initial = source
    return dict(
        schema="video_opd_checkpoint_identity_v1", optimizer_step=step, data_cursor=step,
        protocol_sha256=protocol["protocol_sha256"], evidence=protocol["evidence"],
        common_initial_head_weights_sha256=common_initial,
        initialization_manifest_sha256=initialization_manifest,
        pool_sha256=hashes["pool"], schedule_sha256=hashes["schedule"],
        target_config_sha256=hashes["target_config"], seed=config["seed"], lr=config["lr"],
        qa_per_update=config["qa_per_update"], time_mode=config["time_mode"],
        qa_exposures=step * config["qa_per_update"],
        max_new_tokens=config.get("max_new_tokens"), max_seq_len=config.get("max_seq_len"),
        loss_normalization=config["loss_normalization"], e2e_tv_weight=config["e2e_tv_weight"],
        precision={name: protocol[name] for name in (
            "target_dtype", "head_dtype", "master_dtype", "evidence_parameter_dtype")},
        optimizer=protocol["optimizer"], initialization="weights_only_fresh_optimizer",
        resume_supported=False, diagnostic_only=True)


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


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--target", default=common.TARGET_PATH,
                    help="local target model directory; its config must match the head")
    ap.add_argument("--init-from", required=True,
                    help="head weights only; starts optimizer/cursor afresh, never resumes training_state.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--qa-per-update", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=768,
                    help="legacy compatibility argument; video encoding uses the cached frame geometry")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--e2e-tv-weight", type=float, default=10.0)
    ap.add_argument("--device-map", default="balanced")
    ap.add_argument("--master-weights", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--time-mode", choices=["legacy", "real"], default="legacy",
                    help="real requires unique frame indices and original source timeline metadata")
    ap.add_argument("--loss-normalization", choices=LOSS_NORMALIZATIONS,
                    default="legacy_per_qa",
                    help="legacy sums per-QA losses; global pools all QAs with one base denominator")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate inputs/configs and print protocol on CPU; no model loads or output writes")
    ap.add_argument("--deadline-utc", default=None,
                    help="timezone-aware ISO instant; save/stop at update boundary, not a hard interrupt")
    ap.add_argument("--stop-file", default=None,
                    help="existence requests a checkpoint and stop at the next update boundary")
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    deadline = parse_deadline(a.deadline_utc)
    if not a.device_map:
        ap.error("9k trainer requires --device-map")
    for name in ("updates", "qa_per_update", "save_every", "max_new_tokens", "max_seq_len"):
        if getattr(a, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if a.max_seq_len <= a.max_new_tokens:
        ap.error("max-seq-len must leave room for prompt and max-new-tokens")
    for name in ("lr", "clip", "e2e_tv_weight"):
        value = getattr(a, name)
        if not math.isfinite(value) or value < 0 or (name in ("lr", "clip") and value == 0):
            ap.error(f"invalid --{name.replace('_', '-')}")

    out = Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit("new output must be empty")
    # This is the same storage contract enforced by save_atomic_checkpoint;
    # reject bad destinations before training rather than at the first save.
    training_resume.validate_output(out)
    rows = pd.read_parquet(a.pool).set_index("task", verify_integrity=True)
    if "prompt" not in rows or not rows["prompt"].map(lambda p: isinstance(p, str) and bool(p.strip())).all():
        raise ValueError("pool must contain nonempty string prompts")
    schedule = load_schedule(a.schedule, rows, a.updates, a.qa_per_update)
    inputs_summary = validate_rows(rows, schedule, a.time_mode)
    model_dimensions = validate_model_contract(a.target, a.init_from)
    evidence_config = validate_training_head(a.init_from, a.time_mode, a.loss_normalization)
    protocol = make_protocol(a, inputs_summary, model_dimensions)
    if a.dry_run:
        print(json.dumps(protocol, indent=2))
        return protocol

    initial_stop = stop_reason(deadline, a.stop_file)
    if initial_stop:
        raise SystemExit(f"not starting model load: {initial_stop}")

    dev = "cuda:0"
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"[9k] weights-only fresh initialization; {a.time_mode=}, {a.loss_normalization=}", flush=True)
    target, processor = common.load_target(path=a.target, device=dev, dtype=torch.float32,
                                           device_map=a.device_map)
    wrap_target(target, dev)
    target.eval().requires_grad_(False)
    print(f"[9k] target map={getattr(target, 'hf_device_map', None)}", flush=True)

    head = common.load_head(a.init_from, device=dev, dtype=torch.bfloat16).eval()
    if evidence_config["mode"] != "none":
        head.evidence_readout.float()
    head.requires_grad_(True)
    optimizer = MasterWeightAdamW(head, lr=a.lr, betas=common.BETAS,
                                  weight_decay=common.WEIGHT_DECAY,
                                  where=a.master_weights, clip=a.clip)

    def save_checkpoint(step):
        training_resume.save_atomic_checkpoint(
            out / f"step{step}", head,
            dict(head=head.state_dict(), master_optimizer=optimizer.state_dict(),
                 update=step, data_cursor=step,
                 training_protocol=protocol, protocol_sha256=protocol["protocol_sha256"],
                 world_size=1, slot_map=protocol["slot_map"],
                 head_version=dict(optimizer_step=step, schedule_cursor=step,
                                   protocol_sha256=protocol["protocol_sha256"]),
                 initialization="weights_only_fresh_optimizer", resume_supported=False,
                 loss_normalization=a.loss_normalization, time_mode=a.time_mode,
                 diagnostic_only=True, smoke=False, opd_mode="pure_block",
                 block_native=True, teacher_verify_impl="pure_block_native"),
            metadata=checkpoint_identity(protocol, step))

    for cursor in range(a.updates):
        reason = stop_reason(deadline, a.stop_file)
        if reason:
            result = checkpoint_and_status(out, cursor, a.updates, reason, save_checkpoint)
            print(json.dumps(result), flush=True)
            return result
        started = time.perf_counter()
        head.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        tasks = []
        input_token_counts = []
        visual_token_counts = []
        evidence_rollout_diagnostics = []
        evidence_score_diagnostics = []
        video_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

        def collect_traces():
            for item in schedule[cursor]:
                task = item["task"]
                tasks.append(task)
                row = rows.loc[task].to_dict()
                head.visual_memory = None
                inputs = adapters.encode(processor, row, "video", max_pixels=a.max_pixels,
                                         time_mode=a.time_mode)
                n_input = int(inputs["input_ids"].numel())
                if n_input + a.max_new_tokens > a.max_seq_len:
                    raise ValueError(f"{task}: prompt {n_input} + output budget {a.max_new_tokens} exceeds max-seq-len")
                input_token_counts.append(n_input)
                visual_token_counts.append(int((inputs["input_ids"] == video_id).sum()))
                head.eval()
                request = None
                if evidence_config["mode"] != "none":
                    request = build_evidence_request(inputs, row, processor)
                    configure_evidence(head, request)
                trace = adapters.collect(target, head, inputs,
                                         max_new_tokens=a.max_new_tokens,
                                         max_seq_len=a.max_seq_len,
                                         teacher_verify_impl="pure_block_native")
                if request is not None:
                    trace["evidence_request"] = request
                    evidence_rollout_diagnostics.append(head.evidence_diagnostics())
                yield trace

        stats = backward_update(
            collect_traces(),
            lambda trace: score_training_trace(target, head, trace,
                                               e2e_tv_weight=a.e2e_tv_weight,
                                               diagnostics=evidence_score_diagnostics),
            a.loss_normalization,
            stage_trace=lambda trace: move_training_trace(trace, "cpu"),
            restore_trace=lambda trace: move_training_trace(trace, dev))
        evidence_gradients = evidence_gradient_diagnostics(head)
        optimizer.sync_grads()
        norm = optimizer.grad_norm()
        if not math.isfinite(norm):
            raise RuntimeError("non-finite gradient")
        update_norms = optimizer.step()
        record = dict(update=cursor + 1, optimizer_step=cursor + 1,
                      tasks=tasks, **stats, gradient_norm=norm,
                      protocol_sha256=protocol["protocol_sha256"],
                      elapsed_seconds=time.perf_counter() - started,
                      target_input_tokens=sum(input_token_counts),
                      input_tokens_per_qa=input_token_counts,
                      visual_tokens_per_qa=visual_token_counts)
        if evidence_gradients is not None:
            deltas = {name: value for name, value in update_norms.items()
                      if name.startswith("evidence_readout.")}
            record["evidence"] = dict(
                **evidence_gradients, update_norms=deltas,
                update_l2=math.sqrt(sum(value * value for value in deltas.values())),
                rollout=evidence_rollout_diagnostics, scoring=evidence_score_diagnostics)
        with (out / "log.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        reason = stop_reason(deadline, a.stop_file)
        if reason or cursor + 1 == a.updates:
            # Completing the fixed budget wins over a simultaneous deadline.
            reason = reason if cursor + 1 < a.updates else None
            result = checkpoint_and_status(out, cursor + 1, a.updates, reason, save_checkpoint)
            print(json.dumps(result), flush=True)
            return result
        if (cursor + 1) % a.save_every == 0:
            save_checkpoint(cursor + 1)


if __name__ == "__main__":
    main()
