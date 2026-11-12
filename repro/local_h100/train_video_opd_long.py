#!/usr/bin/env python3
"""Independent bounded OPD entry with complete restart state (validation candidate).

Uses the existing native rollout and serial Bernoulli/k3 scoring unchanged.
Neither old OPD snapshots nor pilot weights are accepted as resumable state.
"""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()
from shared import common
from shared.evidence import build_evidence_request, configure_evidence
from local_h100 import long_training_state as state
from local_h100 import train_video_evidence_opd as evidence
from local_h100 import evidence_opd_prompt_boundary as boundary
from local_h100.train_video_opd import preserve_single_gpu_target
from video_caption import train_video_9k as trainer


def run_update(args, prepared, cursor, target, processor, head, optimizer):
    """Original single-process collect -> global serial score -> Adam update."""
    started = time.perf_counter()
    head.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    tasks, inputs_count, visual_count, rollouts, scores = [], [], [], [], []
    mode = prepared['protocol']['evidence']['mode']
    video_id = processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')

    def traces():
        for item in prepared['schedule'][cursor]:
            task = item['task']
            row = prepared['rows'].loc[task].to_dict()
            tasks.append(task)
            head.visual_memory = None
            inputs = trainer.adapters.encode(processor, row, 'video', max_pixels=args.max_pixels,
                                             time_mode=args.time_mode)
            count = int(inputs['input_ids'].numel())
            if count + args.max_new_tokens > args.max_seq_len:
                raise ValueError('prompt plus output exceeds max-seq-len')
            inputs_count.append(count)
            visual_count.append(int((inputs['input_ids'] == video_id).sum()))
            head.eval()
            request = None
            if mode != 'none':
                request = build_evidence_request(inputs, row, processor)
                configure_evidence(head, request)
            trace = trainer.adapters.collect(target, head, inputs,
                max_new_tokens=args.max_new_tokens, max_seq_len=args.max_seq_len,
                teacher_verify_impl='pure_block_native')
            if request is not None:
                trace['evidence_request'] = request
                rollouts.append(head.evidence_diagnostics())
            yield trace

    dev = next(head.parameters()).device
    stats = trainer.backward_update(traces(),
        lambda trace: trainer.score_training_trace(target, head, trace,
            e2e_tv_weight=args.e2e_tv_weight, diagnostics=scores),
        args.loss_normalization,
        stage_trace=lambda trace: trainer.move_training_trace(trace, 'cpu'),
        restore_trace=lambda trace: trainer.move_training_trace(trace, dev))
    gradients = trainer.evidence_gradient_diagnostics(head)
    optimizer.sync_grads()
    norm = optimizer.grad_norm()
    if not math.isfinite(norm):
        raise RuntimeError('nonfinite gradient')
    update_norms = optimizer.step()
    record = dict(update=cursor + 1, optimizer_step=cursor + 1, tasks=tasks, **stats,
        gradient_norm=norm, elapsed_seconds=time.perf_counter() - started,
        input_tokens_per_qa=inputs_count, visual_tokens_per_qa=visual_count,
        protocol_sha256=prepared['protocol']['protocol_sha256'])
    if gradients is not None:
        deltas = {k: v for k, v in update_norms.items() if k.startswith('evidence_readout.')}
        record['evidence'] = dict(**gradients, update_norms=deltas,
            update_l2=math.sqrt(sum(v * v for v in deltas.values())), rollout=rollouts, scoring=scores)
    return record


def run_segment(args, prepared, target, processor, head, optimizer):
    """New owner loop only: absolute schedule cursor and complete-update saves."""
    completed = state.restore_if_needed(prepared, head, optimizer)
    last = prepared['parent_checkpoint']
    reason = None
    deadline = trainer.parse_deadline(args.deadline_utc)
    for cursor in range(completed, prepared['end_step']):
        reason = trainer.stop_reason(deadline, args.stop_file)
        if reason:
            break
        record = run_update(args, prepared, cursor, target, processor, head, optimizer)
        completed = cursor + 1
        with (prepared['out'] / 'log.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)
        reason = trainer.stop_reason(deadline, args.stop_file)
        if completed % args.save_every == 0 or completed == prepared['end_step'] or reason:
            last = state.save_checkpoint(prepared, completed, head, optimizer)
        if reason:
            break
    # A stop observed before the next iteration must not lose completed updates.
    if completed > prepared['start_step'] and (last is None or Path(last).name != f'step{completed}'):
        last = state.save_checkpoint(prepared, completed, head, optimizer)
    if completed == prepared['end_step'] and completed < args.total_updates and reason is None:
        reason = 'segment_budget'
    result = state.finish_segment(prepared, completed, reason, last)
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    args = state.argument_parser(__doc__, 'opd').parse_args(argv)
    prepared = state.prepare(args, 'opd', [__file__, evidence.__file__, boundary.__file__])
    if args.dry_run:
        print(json.dumps(prepared['protocol'], indent=2))
        return prepared['protocol']
    if trainer.stop_reason(trainer.parse_deadline(args.deadline_utc), args.stop_file):
        raise RuntimeError('stop requested before model loading')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    state.start_segment(prepared, args)
    torch.cuda.set_device(0)
    target, processor = common.load_target(args.target, device='cuda:0', dtype=torch.float32,
                                           device_map='balanced')
    preserve_single_gpu_target(target, 'cuda:0')
    target.eval().requires_grad_(False)
    with ExitStack() as stack:
        if prepared['protocol']['evidence']['mode'] != 'none':
            stack.enter_context(evidence.evidence_runtime(trainer))
            stack.enter_context(boundary.prompt_boundary_runtime(trainer))
        head = common.load_head(prepared['init_path'], device='cuda:0', dtype=torch.bfloat16).eval()
        head.requires_grad_(True)
        optimizer = trainer.MasterWeightAdamW(head, lr=args.lr, betas=common.BETAS,
            weight_decay=common.WEIGHT_DECAY, where='gpu', clip=args.clip)
        return run_segment(args, prepared, target, processor, head, optimizer)


if __name__ == '__main__':
    main()
