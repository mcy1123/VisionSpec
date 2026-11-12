#!/usr/bin/env python3
"""Finite local evidence SFT with the baseline coverage-CE objective.

CLI matches train_video_sft_pilot. This entry explicitly admits only the
minimal text/visual readout with time encoding off. The plain entry's guard
is retained unchanged. All substitutions exist only in this Python process.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

from local_h100 import evidence_sft as evidence
from local_h100 import train_video_sft as base
from local_h100 import train_video_sft_pilot as pilot
from shared import training_resume
from shared.evidence import build_evidence_request, evidence_settings


def validate_head(path):
    evidence.minimal_config(path)
    evidence_settings(path)  # Enforces retained layer 33 and positive settings.
    config = json.loads((Path(path) / 'config.json').read_text())
    draft = config.get('dflash_config', {})
    block = int(draft.get('block_size', config.get('block_size', 16)))
    mask = int(draft.get('mask_token_id', config.get('mask_token_id', base.common.MASK_ID)))
    if block < 2 or not 0 <= mask < config['vocab_size']:
        raise ValueError('invalid evidence head block size or mask id')
    return block, mask


@contextmanager
def evidence_runtime():
    """Explicitly adapt input binding, forward and optimizer setup; restore all."""
    from qa_opd_fair import videoencode
    originals = dict(validate=base.validate_head, loss=base.hidden_loss,
                     optimizer=base.MasterWeightAdamW, protocol=base.build_protocol,
                     checkpoint=base.checkpoint, encode=videoencode.encode_inputs)
    pending = {'request': None}

    def encode(processor, row, *args, **kwargs):
        if pending['request'] is not None:
            raise RuntimeError('previous encoded QA was not consumed by the SFT loss')
        inputs = originals['encode'](processor, row, *args, **kwargs)
        pending['request'] = build_evidence_request(inputs, row, processor)
        return inputs

    def loss(head, target, ctx, full, anchors, reveals, mask_id):
        request = pending['request']
        pending['request'] = None
        if request is None:
            raise RuntimeError('SFT loss has no encoded-input-bound evidence request')
        return evidence.hidden_loss(originals['loss'], head, target, ctx, full,
                                    anchors, reveals, mask_id, request)

    def optimizer(head, **kwargs):
        # base.main has already called head.requires_grad_(True).
        evidence.freeze_time_after_unfreeze(head)
        return originals['optimizer'](head, **kwargs)
    # Base protocol records the actual optimizer implementation source.
    optimizer.__module__ = originals['optimizer'].__module__

    def protocol(args, inputs, dimensions, gpus):
        result = originals['protocol'](args, inputs, dimensions, gpus)
        result['base_entry'] = result['entry']
        result['entry'] = 'local_h100/train_video_evidence_sft.py'
        result['evidence_architecture'] = dict(
            **evidence.minimal_config(args.init_from),
            implementation='existing EvidenceReadout cross-attention and zero-init residual',
            official_Vispec_reproduction=False,
            memory='last retained target layer 33, detached; text excludes visual tokens',
            packed_forward='one actual head.forward; original anchor mask; separate prefix memory per anchor',
            time_policy='time.weight exactly zero and frozen after all-head unfreeze',
            request_policy='exact encoded input identity, bind and clear per QA',
            parameter_dtype='float32',
            supervision='unchanged baseline variable-reveal labels, weights, target LM head and CE')
        result['training_math'] = 'baseline coverage CE with actual evidence-enabled head.forward'
        result['hashes']['evidence_sft'] = training_resume.file_hash(evidence.__file__)
        result['hashes']['evidence_sft_entry'] = training_resume.file_hash(__file__)
        result['hashes']['evidence_readout'] = training_resume.file_hash(
            REPRO.parent / 'scripts/dflash_pkg/dflash-0.1.0/dflash/evidence.py')
        result.pop('protocol_sha256', None)
        result['protocol_sha256'] = base.digest(result)
        return result

    def checkpoint(out, step, head, opt, protocol):
        evidence.assert_time_off(head)
        if head.evidence_readout.request is not None:
            raise RuntimeError('checkpoint cannot retain a previous QA request')
        return originals['checkpoint'](out, step, head, opt, protocol)

    base.validate_head, base.hidden_loss, base.MasterWeightAdamW = validate_head, loss, optimizer
    base.build_protocol, base.checkpoint = protocol, checkpoint
    videoencode.encode_inputs = encode
    try:
        yield
    finally:
        pending['request'] = None
        base.validate_head, base.hidden_loss = originals['validate'], originals['loss']
        base.MasterWeightAdamW = originals['optimizer']
        base.build_protocol, base.checkpoint = originals['protocol'], originals['checkpoint']
        videoencode.encode_inputs = originals['encode']


def main(argv=None):
    # Pilot I/O is the outer layer so the evidence wrapper enriches its protocol
    # and validates time-off before its weights-only checkpoint publication.
    with pilot.pilot_runtime(), evidence_runtime():
        return base.main(argv)


if __name__ == '__main__':
    main()
