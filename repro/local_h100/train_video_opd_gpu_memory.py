#!/usr/bin/env python3
"""Existing single-GPU OPD with GPU-master memory accounting only.

Supports plain and explicit time-off evidence heads, keeps full-state atomic
checkpoints, and does not change OPD math, optimizer arguments or zero_grad.
"""
from contextlib import contextmanager, nullcontext
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

from local_h100 import train_video_opd as local
from local_h100 import train_video_evidence_opd as evidence
from shared import training_resume
from shared.evidence import evidence_settings


def select_evidence(init_from, requested):
    actual = evidence_settings(init_from)['mode']
    expected = 'none' if requested == 'plain' else requested
    if expected != 'auto' and actual != expected:
        raise ValueError(f'--evidence-mode {requested} disagrees with checkpoint mode {actual}')
    if actual != 'none':
        evidence.evidence.minimal_config(init_from)
    return actual


def optimizer_snapshot(optimizer):
    """JSON scalars only; do not retain or copy any gradient/state tensor."""
    master = optimizer.master
    devices = sorted({str(value.device) for value in master.values()})
    if optimizer.where != 'gpu' or not devices or any(torch.device(d).type != 'cuda' for d in devices):
        raise ValueError('OPD GPU memory entry requires actual CUDA master tensors')
    if len(devices) != 1 or str(optimizer._dev) != devices[0]:
        raise ValueError('OPD GPU master must remain on the sole head device')
    states = list(optimizer._opt.state.values())
    moments = [value for state in states for key, value in state.items()
               if key in ('exp_avg', 'exp_avg_sq')]
    moment_devices = sorted({str(value.device) for value in moments})
    if any(device not in devices for device in moment_devices):
        raise ValueError('Adam moments were unexpectedly offloaded from the master GPU')
    gradients = [value.grad for value in master.values() if value.grad is not None]
    return dict(
        optimizer_where=optimizer.where, optimizer_device=str(optimizer._dev),
        master_devices=devices, master_dtypes=sorted({str(value.dtype) for value in master.values()}),
        master_tensor_count=len(master), master_parameter_count=sum(value.numel() for value in master.values()),
        master_gradient_tensor_count=len(gradients),
        master_gradient_devices=sorted({str(value.device) for value in gradients}),
        adam_state_parameter_count=len(states),
        adam_initialized_parameter_count=sum('exp_avg' in state and 'exp_avg_sq' in state for state in states),
        adam_moment_tensor_count=len(moments), adam_moment_devices=moment_devices,
        adam_moment_dtypes=sorted({str(value.dtype) for value in moments}),
        adam_step_devices=sorted({str(state['step'].device) for state in states
                                  if torch.is_tensor(state.get('step'))}),
        peak_scope='this process since CUDA allocator statistics began; wrapper never resets peaks',
        cuda={device: dict(max_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           max_memory_reserved_bytes=torch.cuda.max_memory_reserved(device))
              for device in devices})


@contextmanager
def memory_runtime(trainer, mode):
    """Compose after local_runtime and before optional evidence_runtime."""
    original_optimizer = trainer.MasterWeightAdamW
    original_protocol = trainer.make_protocol
    original_save = training_resume.save_atomic_checkpoint
    captured = {'optimizer': None, 'protocol': None}
    wrapper_sha = training_resume.file_hash(__file__)

    def optimizer(head, **kwargs):
        if kwargs.get('where') != 'gpu':
            raise ValueError('OPD GPU memory entry requires --master-weights gpu')
        devices = {p.device for p in head.parameters()}
        if len(devices) != 1 or next(iter(devices)).type != 'cuda':
            raise ValueError('OPD GPU master requires a single CUDA head device')
        if captured['optimizer'] is not None:
            raise RuntimeError('one OPD run must create exactly one optimizer')
        result = original_optimizer(head, **kwargs)
        if result.where != 'gpu' or result._dev != next(iter(devices)):
            raise RuntimeError('original optimizer did not use the head GPU')
        captured['optimizer'] = result
        return result
    optimizer.__module__ = original_optimizer.__module__

    def protocol(args, *rest, **kwargs):
        if args.master_weights != 'gpu':
            raise ValueError('OPD GPU memory entry requires --master-weights gpu')
        result = original_protocol(args, *rest, **kwargs)
        result['optimizer']['where'] = 'gpu'
        result['optimizer_where'] = 'gpu'
        result['local_gpu_memory_entry'] = 'local_h100/train_video_opd_gpu_memory.py'
        result['hashes']['local_gpu_memory_entry'] = wrapper_sha
        result['gpu_memory_policy'] = dict(
            selected_evidence_mode=mode, checkpoint_storage='unchanged_full_training_state',
            optimizer_state_saved=True, resume_supported=False,
            objective_optimizer_zero_grad_and_rng='unchanged',
            metrics='process allocator peaks and actual master/Adam moment devices at checkpoint',
            counters_reset=False, cpu_gpu_bitwise_equivalence_claimed=False)
        result.pop('protocol_sha256', None)
        result['protocol_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        # Inner evidence_runtime enriches this same dictionary and rehashes it.
        captured['protocol'] = result
        return result

    def save(destination, head, state, *, metadata=None):
        opt, protocol = captured['optimizer'], captured['protocol']
        if opt is None or protocol is None:
            raise RuntimeError('OPD memory checkpoint lacks captured optimizer/protocol')
        expected = protocol['protocol_sha256']
        if state.get('protocol_sha256') != expected or (metadata or {}).get('protocol_sha256') != expected:
            raise ValueError('OPD memory checkpoint protocol identity mismatch')
        identity = dict(metadata, optimizer_where='gpu',
                        gpu_memory_entry_sha256=wrapper_sha,
                        gpu_memory_protocol_sha256=expected,
                        gpu_memory_diagnostics=optimizer_snapshot(opt))
        return original_save(destination, head, state, metadata=identity)

    trainer.MasterWeightAdamW, trainer.make_protocol = optimizer, protocol
    training_resume.save_atomic_checkpoint = save
    try:
        yield captured
    finally:
        trainer.MasterWeightAdamW, trainer.make_protocol = original_optimizer, original_protocol
        training_resume.save_atomic_checkpoint = original_save


def main(argv=None):
    from video_caption import train_video_9k as trainer
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = trainer.parser()
    parser._option_string_actions['--updates'].required = True
    parser.add_argument('--evidence-mode', choices=('auto', 'plain', 'text', 'visual'), default='auto',
                        help='auto reads the checkpoint; evidence always requires explicit time encoding off')
    args = parser.parse_args(argv)
    if args.master_weights != 'gpu':
        parser.error('this diagnostic entry requires --master-weights gpu')
    local.validate_execution(args)
    mode = select_evidence(args.init_from, args.evidence_mode)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--evidence-mode')
    _, inherited_argv = selector.parse_known_args(argv)
    with local.local_runtime(trainer), memory_runtime(trainer, mode), \
            (evidence.evidence_runtime(trainer) if mode != 'none' else nullcontext()):
        return trainer.main(inherited_argv)


if __name__ == '__main__':
    main()
