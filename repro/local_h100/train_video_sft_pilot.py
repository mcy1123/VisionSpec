#!/usr/bin/env python3
"""Unchanged local SFT training with explicitly non-resumable pilot checkpoints.

Only checkpoint I/O and target-hash acquisition differ from train_video_sft.
The original full-state entry remains available. No optimizer state is saved.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import uuid

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()
from local_h100 import train_video_sft as base
from local_h100.generate_teacher_cache import file_identity, target_weight_hashes
from shared import training_resume


def weights_only_checkpoint(out, step, head, optimizer, protocol):
    """Atomic model+identity+protocol; optimizer argument is deliberately unused."""
    dest = base.safe_output(Path(out) / f'step{step}')
    if dest.exists():
        raise FileExistsError(dest)
    tmp = dest.with_name('.' + dest.name + '.tmp-' + uuid.uuid4().hex)
    tmp.mkdir()
    head.save_pretrained(tmp)
    identity = dict(schema='local_video_sft_checkpoint_v1', optimizer_step=step,
                    qa_exposures=step * protocol['config']['qa_per_update'],
                    protocol_sha256=protocol['protocol_sha256'],
                    initialization_weights_sha256=protocol['hashes']['initial_head_weights'],
                    pool_sha256=protocol['hashes']['pool'], schedule_sha256=protocol['hashes']['schedule'],
                    target_weights_sha256=protocol['hashes']['target_weights'],
                    target_dtype='float32', head_dtype='bfloat16', master_dtype='float32',
                    checkpoint_storage='weights_only_pilot', pilot_only=True,
                    optimizer_state_saved=False, resume_supported=False,
                    training_state_file=None)
    base.write_json_exclusive(tmp / 'CHECKPOINT_IDENTITY.json', identity)
    base.write_json_exclusive(tmp / 'TRAINING_PROTOCOL.json', protocol)
    base.write_json_exclusive(tmp / 'SHA256.json', {
        path.name: training_resume.file_hash(path) for path in sorted(tmp.iterdir()) if path.is_file()})
    os.rename(tmp, dest)
    return str(dest)


def pilot_protocol(original, args, inputs, dimensions, gpus):
    target = Path(args.target)
    paths = sorted(target.glob('*.safetensors')) + sorted(target.glob('pytorch_model*.bin'))
    if not paths:
        raise ValueError('target has no local weights')
    original_hash = training_resume.file_hash
    if args.dry_run:
        # Preserve the base entry's no-output-write dry run contract.
        hashes = {path.name: original_hash(path) for path in paths}
        before = {path.name: file_identity(path) for path in paths}
        verification = dict(method='dry_run_full_SHA256_no_cache_writes', file_identity=before)
    else:
        cache_root = base.safe_output(args.teacher_cache or Path(args.out) / 'teacher_cache')
        hashes, verification = target_weight_hashes(target, cache_root / '_target_identity')
        before = verification['file_identity']
    by_path = {path.resolve(): (hashes[path.name], before[path.name]) for path in paths}
    def selected_hash(path):
        resolved = Path(path).resolve()
        if resolved in by_path:
            cached_hash, fingerprint = by_path[resolved]
            if file_identity(resolved) != fingerprint:
                raise RuntimeError('target file identity changed during SFT protocol construction')
            return cached_hash
        return original_hash(path)
    # Scope is only this synchronous protocol call. All non-target files retain
    # the original complete SHA256 path, and the shared function is restored.
    training_resume.file_hash = selected_hash
    try:
        protocol = original(args, inputs, dimensions, gpus)
    finally:
        training_resume.file_hash = original_hash
    if protocol['hashes']['target_weights'] != hashes:
        raise RuntimeError('cached target SHA dictionary differs from the base protocol')
    protocol['base_entry'] = protocol['entry']
    protocol['entry'] = 'local_h100/train_video_sft_pilot.py'
    protocol['checkpoint_storage'] = 'weights_only_pilot'
    protocol['optimizer_state_saved'] = False
    protocol['resume_supported'] = False
    protocol['pilot_only'] = True
    protocol['training_math'] = 'unchanged train_video_sft.main / hidden_loss / MasterWeightAdamW'
    protocol['hashes']['pilot_wrapper'] = original_hash(__file__)
    protocol['target_weight_verification'] = verification
    protocol.pop('protocol_sha256', None)
    protocol['protocol_sha256'] = base.digest(protocol)
    return protocol


@contextmanager
def pilot_runtime():
    original_protocol, original_checkpoint = base.build_protocol, base.checkpoint
    base.build_protocol = lambda *args, **kwargs: pilot_protocol(original_protocol, *args, **kwargs)
    base.checkpoint = weights_only_checkpoint
    try:
        yield
    finally:
        base.build_protocol, base.checkpoint = original_protocol, original_checkpoint


def main(argv=None):
    with pilot_runtime():
        return base.main(argv)


if __name__ == '__main__':
    main()
