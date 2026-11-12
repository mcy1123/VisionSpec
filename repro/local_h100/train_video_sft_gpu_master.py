#!/usr/bin/env python3
"""Finite pilot SFT with fp32 master weights and Adam state on the head GPU.

CLI and coverage-CE training are inherited unchanged from train_video_sft.
Only optimizer placement and its provenance differ from the CPU-master pilot.
Checkpoints remain weights-only and cannot resume optimizer state. CPU and GPU
AdamW executions are not claimed to be numerically or bitwise equivalent.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
import torch

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

from local_h100 import train_video_sft as base
from local_h100 import train_video_sft_pilot as pilot
from shared import training_resume


def head_cuda_device(head):
    """Fail instead of silently selecting CPU for MasterWeightAdamW(gpu)."""
    devices = {p.device for p in head.parameters()}
    if len(devices) != 1 or next(iter(devices)).type != 'cuda':
        raise ValueError('GPU master requires all head parameters on one CUDA device')
    return next(iter(devices))


def memory_diagnostics(optimizer):
    """Read process allocator peaks at save time without resetting any counters."""
    devices = sorted({str(value.device) for value in optimizer.master.values()})
    if not devices or any(torch.device(device).type != 'cuda' for device in devices):
        raise ValueError('GPU master checkpoint contains missing or non-CUDA master tensors')
    return dict(optimizer_where=optimizer.where,
                optimizer_device=str(optimizer._dev), master_devices=devices,
                master_tensor_count=len(optimizer.master),
                peak_scope='this process since CUDA allocator statistics began; not reset by wrapper',
                cuda={device: dict(max_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                   max_memory_reserved_bytes=torch.cuda.max_memory_reserved(device))
                      for device in devices})


@contextmanager
def gpu_master_runtime():
    """Compose inside pilot_runtime and outside an optional evidence_runtime.

    Example: with pilot.pilot_runtime(), gpu_master_runtime(), evidence_runtime():
    base.main(argv). The evidence optimizer factory must capture this factory,
    so its freeze-after-unfreeze policy executes before GPU master creation.
    Every substitution is process-local, synchronous and restored on failure.
    """
    originals = dict(optimizer=base.MasterWeightAdamW, protocol=base.build_protocol,
                     checkpoint=base.checkpoint)
    wrapper_sha = training_resume.file_hash(__file__)

    def optimizer(head, **kwargs):
        device = head_cuda_device(head)
        if kwargs.get('where', 'cpu') not in ('cpu', 'gpu'):
            raise ValueError('unexpected optimizer placement requested')
        kwargs['where'] = 'gpu'
        result = originals['optimizer'](head, **kwargs)
        if result.where != 'gpu' or result._dev != device:
            raise RuntimeError('optimizer did not retain master weights on the head GPU')
        return result

    # Base build_protocol hashes the defining optimizer module. Preserve that
    # implementation identity and record this placement wrapper separately.
    optimizer.__module__ = originals['optimizer'].__module__

    def protocol(*args, **kwargs):
        result = originals['protocol'](*args, **kwargs)
        if result.get('checkpoint_storage') != 'weights_only_pilot':
            raise ValueError('gpu_master_runtime must be composed inside pilot_runtime')
        if result.get('resume_supported') is not False or result.get('optimizer_state_saved') is not False:
            raise ValueError('GPU master pilot requires explicit non-resumable weights-only storage')
        result['gpu_master_base_entry'] = result['entry']
        result['entry'] = 'local_h100/train_video_sft_gpu_master.py'
        result['optimizer_where'] = 'gpu'
        result['config']['optimizer_where'] = 'gpu'
        result['optimizer']['where'] = 'gpu'
        result['gpu_master_policy'] = dict(
            master_dtype='float32', adam_moments_dtype='float32',
            placement='same single CUDA device as all head parameters',
            objective_and_update_formula='unchanged shared MasterWeightAdamW and coverage CE',
            cpu_gpu_bitwise_equivalence_claimed=False,
            cpu_gpu_numerical_equivalence_tested=False,
            adoption='requires longest-input memory probe; no CPU fallback')
        result['hashes']['gpu_master_wrapper'] = wrapper_sha
        result.pop('protocol_sha256', None)
        result['protocol_sha256'] = base.digest(result)
        return result

    def checkpoint(out, step, head, opt, protocol):
        if protocol.get('optimizer_where') != 'gpu' or protocol['optimizer']['where'] != 'gpu':
            raise ValueError('GPU master checkpoint requires its matching protocol')
        if opt.where != 'gpu':
            raise ValueError('GPU master checkpoint received a non-GPU optimizer')
        diagnostics = memory_diagnostics(opt)
        original_write = base.write_json_exclusive

        def write(path, value):
            if Path(path).name == 'CHECKPOINT_IDENTITY.json':
                value = dict(value, optimizer_where='gpu',
                             gpu_master_wrapper_sha256=wrapper_sha,
                             gpu_master_diagnostics=diagnostics,
                             cpu_gpu_bitwise_equivalence_claimed=False)
            return original_write(path, value)

        # Enrich identity before the original atomic pilot saver hashes and
        # publishes it. Never mutate an already-published checkpoint or SHA.
        base.write_json_exclusive = write
        try:
            return originals['checkpoint'](out, step, head, opt, protocol)
        finally:
            base.write_json_exclusive = original_write

    base.MasterWeightAdamW, base.build_protocol, base.checkpoint = optimizer, protocol, checkpoint
    try:
        yield
    finally:
        base.MasterWeightAdamW = originals['optimizer']
        base.build_protocol, base.checkpoint = originals['protocol'], originals['checkpoint']


def main(argv=None):
    with pilot.pilot_runtime(), gpu_master_runtime():
        return base.main(argv)


if __name__ == '__main__':
    main()
