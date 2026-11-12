#!/usr/bin/env python3
"""Matched text/visual readouts with explicitly disabled temporal encoding.

This is a local controlled architecture experiment built from the repository's
EvidenceReadout, not an official ViSpec implementation or reproduction.
"""
from __future__ import annotations

import argparse
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

import torch
from video_caption import init_evidence_heads as base
from shared import training_resume

LOCAL_ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')


def destination(path):
    root, out = LOCAL_ROOT.resolve(), Path(path).expanduser().resolve()
    if not root.is_dir() or root not in out.parents:
        raise ValueError('output must be a new child of the existing local_h100_20261112 run root')
    if out.exists():
        raise FileExistsError(f'refusing to overwrite initialization directory: {out}')
    return out


def freeze_time_off(head):
    """Enforce the explicit contract after any blanket requires_grad_(True).

    requires_grad is not stored by safetensors, so training must call this after
    loading and unfreezing the head. Never repair a nonzero projection silently.
    """
    from local_h100.evidence_sft import freeze_time_after_unfreeze
    freeze_time_after_unfreeze(head)
    return {'evidence_time_encoding': 'off', 'time_projection_frozen': True,
            'time_projection_parameter_count': head.evidence_readout.time.weight.numel()}


def parameter_counts(head):
    evidence = head.evidence_readout
    return dict(total_parameter_count=sum(p.numel() for p in head.parameters()),
                trainable_parameter_count=sum(p.numel() for p in head.parameters() if p.requires_grad),
                evidence_total_parameter_count=sum(p.numel() for p in evidence.parameters()),
                evidence_trainable_parameter_count=sum(p.numel() for p in evidence.parameters() if p.requires_grad),
                frozen_parameter_names=[n for n, p in head.named_parameters() if not p.requires_grad])


@contextmanager
def local_initializer(metadata):
    original_make, original_roots = base.make_matched_head, base.ALLOWED_ROOTS
    def make(*args, **kwargs):
        head, new_keys = original_make(*args, **kwargs)
        head.config.dflash_config['evidence_time_encoding'] = 'off'
        with torch.no_grad():
            head.evidence_readout.time.weight.zero_()
        metadata.update(freeze_time_off(head))
        metadata.update(parameter_counts(head))
        metadata['output_projection_zero'] = not bool(torch.count_nonzero(head.evidence_readout.output.weight))
        return head, new_keys
    base.make_matched_head = make
    base.ALLOWED_ROOTS = (LOCAL_ROOT,)
    try:
        yield
    finally:
        base.make_matched_head, base.ALLOWED_ROOTS = original_make, original_roots


def initialize(source, out, *, seed=20261112, dim=256, max_tokens=128,
               spatial_bins=2, dry_run=False):
    out = destination(out)
    metadata = dict(
        evidence_time_encoding='off', time_projection_frozen=True,
        requires_grad_reload_contract='call freeze_time_off after head.requires_grad_(True); safetensors does not save requires_grad',
        architecture='existing_EvidenceReadout_matched_text_visual_no_time',
        official_vispec_reproduction=False,
        provenance_note='Uses repository EvidenceReadout; not an official ViSpec implementation or replication.',
        initializer_wrapper_sha256=training_resume.file_hash(__file__),
        original_initializer_sha256=training_resume.file_hash(base.__file__),
        time_contract_helper_sha256=training_resume.file_hash(REPRO / 'local_h100/evidence_sft.py'))
    with local_initializer(metadata):
        if dry_run:
            result = base.initialize(source, out, seed=seed, dim=dim, max_tokens=max_tokens,
                                     spatial_bins=spatial_bins, dry_run=True)
            result.update(metadata)
            result['parameter_counts_verified'] = False
            return result
        staging = out.with_name('.' + out.name + '.staging-' + uuid.uuid4().hex)
        # The original initializer computes added_tensor_sha256 AFTER the
        # patched factory has zeroed time.weight, and checks B/C equality.
        result = base.initialize(source, staging, seed=seed, dim=dim, max_tokens=max_tokens,
                                 spatial_bins=spatial_bins, dry_run=False)
    result.update(metadata)
    result.update(output=str(out), parameter_counts_verified=True,
                  arms={arm: str(out / arm) for arm in ('B_text', 'C_visual')})
    for arm in ('B_text', 'C_visual'):
        directory = staging / arm
        config = json.loads((directory / 'config.json').read_text())
        if config['dflash_config'].get('evidence_time_encoding') != 'off':
            raise ValueError('serialized config lost the explicit time-off contract')
        state = base.read_weights(base.weight_files(directory))
        added = {name: value for name, value in state.items() if name.startswith(base.PREFIX)}
        if base.state_digest(added) != result['added_tensor_sha256']:
            raise ValueError('manifest added hash does not identify the saved time-zero tensors')
        for key in ('evidence_readout.time.weight', 'evidence_readout.output.weight'):
            if bool(torch.count_nonzero(state[key])):
                raise ValueError(f'new initialization must have zero {key}')
        manifest_path = directory / 'INIT_MANIFEST.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(metadata)
        manifest.update(output=str(out), parameter_counts_verified=True,
                        initializer_summary=str(out / 'INIT_MANIFEST.json'))
        # Only rewrite manifests in this unpublished, newly created staging
        # directory. No historical checkpoint or result is ever replaced.
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        del state, added
    (staging / 'INIT_MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
    if out.exists():
        raise FileExistsError(f'destination appeared while initializing: {out}')
    os.rename(staging, out)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=20261112)
    p.add_argument('--evidence-dim', type=int, default=256)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--spatial-bins', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    result = initialize(a.source, a.out, seed=a.seed, dim=a.evidence_dim,
                        max_tokens=a.max_tokens, spatial_bins=a.spatial_bins, dry_run=a.dry_run)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
