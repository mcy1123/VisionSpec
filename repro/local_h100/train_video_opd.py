#!/usr/bin/env python3
"""Single-GPU local storage/execution adapter for the existing video OPD trainer.

Shared runtime, decoder and loss are unchanged. Single-GPU loading needs no
cross-device output wrapper, so the target keeps its original forward and model
objects (including direct rope_deltas writes). Multi-GPU use is refused here.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
from shared import training_resume

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
ORIGINAL_VALIDATE = training_resume.validate_output


def validate_output(path):
    resolved = Path(path).resolve()
    root = ROOT.resolve()
    if root.exists() and resolved != root and resolved.is_relative_to(root):
        return
    ORIGINAL_VALIDATE(path)


def validate_execution(args, visible=None):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '') if visible is None else visible
    uuids = visible.split(',')
    if len(uuids) != 1 or not uuids[0].startswith('GPU-') or uuids[0].strip() != uuids[0]:
        raise ValueError('local OPD requires exactly one explicit GPU UUID; multi-GPU wrapping is not validated')
    if args.device_map != 'balanced':
        raise ValueError('local OPD requires --device-map balanced with exactly one GPU UUID')
    return uuids


def preserve_single_gpu_target(target, device):
    """No wrapping or mutation. Check accelerate did not silently offload."""
    if device != 'cuda:0':
        raise ValueError('local OPD must execute on the sole visible cuda:0')
    mapping = getattr(target, 'hf_device_map', None)
    if mapping and any(str(value) not in ('0', 'cuda:0') for value in mapping.values()):
        raise ValueError(f'single-GPU target was unexpectedly split/offloaded: {mapping}')
    return target


@contextmanager
def local_runtime(trainer):
    """Process-local substitutions, restored even after validation/training fails."""
    old_wrap, old_protocol = trainer.wrap_target, trainer.make_protocol
    old_validate = training_resume.validate_output
    def protocol(*args, **kwargs):
        result = old_protocol(*args, **kwargs)
        result['local_execution'] = dict(
            source_sha256=training_resume.file_hash(__file__),
            visible_gpu_uuids=os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','),
            target_wrapping='none_original_forward_and_model_preserved',
            device_map='balanced_single_visible_GPU',
            loss='unchanged_generic_OPD_and_optional_top8_overlap_prefix_proxy',
            visual_weighting_enabled=False)
        result.pop('protocol_sha256', None)
        result['protocol_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result
    trainer.wrap_target = preserve_single_gpu_target
    trainer.make_protocol = protocol
    training_resume.validate_output = validate_output
    try:
        yield
    finally:
        trainer.wrap_target, trainer.make_protocol = old_wrap, old_protocol
        training_resume.validate_output = old_validate


def main(argv=None):
    from video_caption import train_video_9k as trainer
    args = trainer.parser().parse_args(argv)
    validate_execution(args)
    with local_runtime(trainer):
        return trainer.main(argv)


if __name__ == '__main__':
    main()
