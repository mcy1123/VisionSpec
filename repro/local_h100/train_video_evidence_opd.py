#!/usr/bin/env python3
"""Existing local video OPD with the same explicit time-off evidence policy.

No rollout, scoring, loss or checkpoint-storage math changes. In particular,
this entry retains the existing full-state checkpoint storage. New local SFT
and this OPD entry share evidence_sft.freeze_time_after_unfreeze.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
from local_h100 import evidence_sft as evidence
from local_h100 import train_video_opd as base
from shared import training_resume


def clear_request(head):
    head.evidence_readout.request = None
    head.evidence_readout.generation += 1


@contextmanager
def evidence_runtime(trainer):
    originals = dict(validate=trainer.validate_training_head, optimizer=trainer.MasterWeightAdamW,
                     protocol=trainer.make_protocol, collect=trainer.adapters.collect,
                     score=trainer.score_training_trace, gradients=trainer.evidence_gradient_diagnostics,
                     checkpoint=training_resume.save_atomic_checkpoint)

    def validate(path, *args, **kwargs):
        evidence.minimal_config(path)
        return originals['validate'](path, *args, **kwargs)

    def optimizer(head, **kwargs):
        evidence.freeze_time_after_unfreeze(head)
        return originals['optimizer'](head, **kwargs)
    optimizer.__module__ = originals['optimizer'].__module__

    def protocol(args, *rest, **kwargs):
        result = originals['protocol'](args, *rest, **kwargs)
        result['evidence_architecture'] = dict(
            **evidence.minimal_config(args.init_from),
            implementation='existing EvidenceReadout cross-attention and zero-init residual',
            official_Vispec_reproduction=False,
            time_policy='time.weight exactly zero and frozen after all-head unfreeze',
            request_policy='existing identity-bound per-trace request; clear head binding after collect/score',
            parameter_dtype='float32',
            optimization='unchanged original OPD runtime and normalization')
        result['local_evidence_entry'] = 'local_h100/train_video_evidence_opd.py'
        result['hashes']['local_evidence_entry'] = training_resume.file_hash(__file__)
        result['hashes']['local_evidence_policy'] = training_resume.file_hash(evidence.__file__)
        result.pop('protocol_sha256', None)
        result['protocol_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result

    def collect(target, head, *args, **kwargs):
        evidence.assert_time_off(head)
        try:
            return originals['collect'](target, head, *args, **kwargs)
        finally:
            clear_request(head)

    def score(target, head, *args, **kwargs):
        evidence.assert_time_off(head)
        try:
            yield from originals['score'](target, head, *args, **kwargs)
        finally:
            clear_request(head)

    def gradients(head):
        evidence.assert_time_off(head)
        return originals['gradients'](head)

    def checkpoint(path, head, *args, **kwargs):
        evidence.assert_time_off(head)
        if head.evidence_readout.request is not None:
            raise RuntimeError('checkpoint cannot retain an evidence QA binding')
        return originals['checkpoint'](path, head, *args, **kwargs)

    trainer.validate_training_head, trainer.MasterWeightAdamW = validate, optimizer
    trainer.make_protocol, trainer.adapters.collect = protocol, collect
    trainer.score_training_trace, trainer.evidence_gradient_diagnostics = score, gradients
    training_resume.save_atomic_checkpoint = checkpoint
    try:
        yield
    finally:
        trainer.validate_training_head, trainer.MasterWeightAdamW = originals['validate'], originals['optimizer']
        trainer.make_protocol, trainer.adapters.collect = originals['protocol'], originals['collect']
        trainer.score_training_trace, trainer.evidence_gradient_diagnostics = originals['score'], originals['gradients']
        training_resume.save_atomic_checkpoint = originals['checkpoint']


def main(argv=None):
    from video_caption import train_video_9k as trainer
    args = trainer.parser().parse_args(argv)
    base.validate_execution(args)
    with base.local_runtime(trainer), evidence_runtime(trainer):
        return trainer.main(argv)


if __name__ == '__main__':
    main()
