#!/usr/bin/env python3
"""Resumable finite segments of the existing local video coverage-CE SFT.

The full schedule and total budget belong to one immutable logical run. Every
invocation writes a new segment directory. Only the new long-state schema can
resume; a weights-only pilot checkpoint can be an explicit fresh initialization.
No running pilot/shared file is modified, and this entry never retries a failure.
"""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys
import time

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

import numpy as np
import torch

from local_h100 import train_video_sft as base
from local_h100 import train_video_evidence_sft as evidence_entry
from local_h100 import evidence_sft
from local_h100 import long_training_state as state
from shared.evidence import evidence_settings


def parser():
    return state.argument_parser(__doc__, 'sft')


def math_sources():
    """Bind this orchestration and every SFT-specific helper into run identity."""
    return [Path(__file__), Path(base.__file__), Path(evidence_entry.__file__),
            Path(evidence_sft.__file__)]


def train_update(prepared, args, cursor, target, processor, head, optimizer,
                 block, mask_id, teacher_cache, eos, *, device='cuda:0',
                 context_dtype=torch.bfloat16):
    """One unchanged baseline update; cursor is absolute in the full schedule.

    The explicit device/dtype keywords support tiny CPU lifecycle tests. The
    production entry always uses CUDA and bf16 target context, as the pilot did.
    """
    if not prepared['start_step'] <= cursor < prepared['end_step']:
        raise ValueError('SFT cursor is outside this segment')
    items = prepared['schedule'][cursor]
    if len(items) != args.qa_per_update:
        raise ValueError('SFT update must contain exactly qa_per_update items')
    optimizer.zero_grad(set_to_none=True)
    records = []
    for item in items:
        row = prepared['rows'].loc[item['task']].to_dict()
        from qa_opd_fair.videoencode import encode_inputs
        inputs = encode_inputs(processor, row, 'real')
        prompt_len = int(inputs['input_ids'].shape[1])
        if prompt_len + args.max_new_tokens > args.max_seq_len:
            raise ValueError(f"{item['task']}: prompt plus output budget exceeds max-seq-len")
        print(f"[teacher] update={cursor + 1} slot={item['slot']} task={item['task']} prompt_tokens={prompt_len}", flush=True)
        tokens, cached, cache_key = base.get_teacher(
            target, inputs, prepared['protocol'], teacher_cache, eos)
        if len(tokens) < 2:
            raise ValueError(f"{item['task']}: fewer than two teacher tokens")
        full = torch.tensor(inputs['input_ids'][0].tolist() + tokens, device=device)
        # The absolute cursor is essential: a resumed segment must not restart
        # the variable-reveal RNG sequence at cursor zero.
        anchors, reveals = base.anchor_plan(
            prompt_len, len(full), args.anchors_per_qa, block,
            args.seed + cursor * 100003 + int(item['slot']))
        target.model.rope_deltas = None
        ctx = base.common.target_context_cached(
            target, full[None], pixel_values_videos=inputs['pixel_values_videos'].to(device),
            video_grid_thw=inputs['video_grid_thw'].to(device)).to(
                dtype=context_dtype, device=device)
        loss, stats = base.hidden_loss(head, target, ctx, full, anchors, reveals, mask_id)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite SFT loss')
        (loss / args.qa_per_update).backward()
        if any(parameter.grad is not None for parameter in target.parameters()):
            raise RuntimeError('frozen target received a gradient')
        records.append(dict(task=item['task'], loss=float(loss.detach()), prompt_tokens=prompt_len,
                            generated_tokens=len(tokens), teacher_cache_hit=cached,
                            teacher_cache_key=cache_key, **stats))
        del ctx, full, loss
    optimizer.sync_grads()
    norm = optimizer.grad_norm()
    if not math.isfinite(norm):
        raise RuntimeError('nonfinite gradient')
    update_norms = optimizer.step()
    return dict(loss=sum(row['loss'] for row in records) / args.qa_per_update,
                gradient_norm=norm,
                written_update_norm=math.sqrt(sum(value * value for value in update_norms.values())),
                qa=records)


def run_segment(prepared, args, target, processor, head, optimizer, block,
                mask_id, teacher_cache, eos, progress, *, update_fn=train_update):
    """Control only: stop at update boundaries and publish full-state saves."""
    deadline = base.parse_deadline(args.deadline_utc)
    out = prepared['out']
    started = time.time()
    for cursor in range(prepared['start_step'], prepared['end_step']):
        reason = base.stop_reason(deadline, args.stop_file)
        if reason:
            progress['reason'] = reason
            break
        step_started = time.time()
        values = update_fn(prepared, args, cursor, target, processor, head, optimizer,
                           block, mask_id, teacher_cache, eos)
        progress['completed'] = cursor + 1
        record = dict(update=cursor + 1, planned_updates=args.total_updates,
                      segment_start_step=prepared['start_step'],
                      segment_end_step=prepared['end_step'],
                      seconds=time.time() - step_started,
                      elapsed_seconds=time.time() - started,
                      protocol_sha256=prepared['protocol']['protocol_sha256'], **values)
        with (out / 'log.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)
        if (cursor + 1) % args.save_every == 0 or cursor + 1 == prepared['end_step']:
            progress['last_checkpoint'] = str(state.save_checkpoint(
                prepared, cursor + 1, head, optimizer))
    completed = progress['completed']
    # Save an early stop only after a fully committed update. Never write a
    # checkpoint on an exception or disguise a partial update as resumable.
    if completed > prepared['start_step'] and progress['last_checkpoint'] != str(out / f'step{completed}'):
        progress['last_checkpoint'] = str(state.save_checkpoint(prepared, completed, head, optimizer))
    return state.finish_segment(prepared, completed, progress['reason'], progress['last_checkpoint'])


def main(argv=None):
    args = parser().parse_args(argv)
    prepared = state.prepare(args, 'sft', math_sources())
    gpus = base.visible_gpu_ids()
    if len(gpus) != 1:
        raise ValueError('resumable SFT requires exactly one GPU UUID')
    mode = evidence_settings(prepared['init_path'])['mode']
    runtime = evidence_entry.evidence_runtime() if mode != 'none' else nullcontext()
    # These substitutions implement the already-tested evidence forward, input
    # binding and freeze-after-unfreeze policy. Their pilot protocol/checkpoint
    # hooks are deliberately unused: all persistence belongs to long_state.
    with runtime:
        block, mask_id = base.validate_head(prepared['init_path'])
        if args.dry_run:
            print(json.dumps(prepared['protocol'], indent=2, allow_nan=False))
            return prepared['protocol']
        if not hasattr(torch.nn.utils, 'get_total_norm'):
            raise RuntimeError('select a PyTorch environment with get_total_norm')
        if base.stop_reason(base.parse_deadline(args.deadline_utc), args.stop_file):
            raise RuntimeError('stop requested before model load')
        teacher_cache = state.safe_path(args.teacher_cache or prepared['out'] / 'teacher_cache')
        state.start_segment(prepared, args)
        progress = dict(completed=prepared['start_step'], last_checkpoint=None, reason=None)
        try:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.set_device(0)
            target, processor = base.common.load_target(args.target, device='cuda:0', dtype=torch.float32)
            target.eval().requires_grad_(False)
            head = base.common.load_head(prepared['init_path'], dtype=torch.bfloat16, device='cuda:0').eval()
            head.requires_grad_(True)
            if int(head.block_size) != block:
                raise ValueError('loaded block size differs from preflight config')
            devices = {parameter.device for parameter in head.parameters()}
            if devices != {torch.device('cuda:0')}:
                raise ValueError('SFT GPU master requires all head parameters on cuda:0')
            optimizer = base.MasterWeightAdamW(
                head, lr=args.lr, betas=base.common.BETAS,
                weight_decay=base.common.WEIGHT_DECAY, where='gpu', clip=args.clip)
            from shared.draft_geometry import eos_ids
            eos = eos_ids(target)
            teacher_cache.mkdir(parents=True, exist_ok=True)
            # Restore only after all random-consuming initialization. The
            # shared function validates and restores full optimizer and RNG.
            restored = state.restore_if_needed(prepared, head, optimizer)
            if restored != prepared['start_step']:
                raise RuntimeError('restored cursor differs from prepared segment')
            state.validate_runtime(head, optimizer)
            return run_segment(prepared, args, target, processor, head, optimizer,
                               block, mask_id, teacher_cache, eos, progress)
        except Exception as exc:
            state.write_json(prepared['out'] / 'FAILED.json', dict(
                completed_updates=progress['completed'],
                segment_start_step=prepared['start_step'], segment_end_step=prepared['end_step'],
                total_updates=args.total_updates, last_checkpoint=progress['last_checkpoint'],
                error_type=type(exc).__name__, error=str(exc),
                protocol_sha256=prepared['protocol']['protocol_sha256']))
            raise


if __name__ == '__main__':
    main()
