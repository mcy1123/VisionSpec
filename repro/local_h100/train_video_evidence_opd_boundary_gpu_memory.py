#!/usr/bin/env python3
"""Time-off evidence OPD boundary adapter with existing GPU memory accounting."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
from local_h100 import train_video_opd as local
from local_h100 import train_video_opd_gpu_memory as memory
from local_h100.train_video_evidence_opd import evidence_runtime
from local_h100.evidence_opd_prompt_boundary import prompt_boundary_runtime
from shared import training_resume


@contextmanager
def combined_runtime(trainer, mode):
    if mode not in ('text', 'visual'):
        raise ValueError('this boundary entry requires a text or visual evidence head')
    with local.local_runtime(trainer), memory.memory_runtime(trainer, mode), \
            evidence_runtime(trainer), prompt_boundary_runtime(trainer):
        original_protocol = trainer.make_protocol
        def protocol(*args, **kwargs):
            result = original_protocol(*args, **kwargs)
            result['combined_evidence_opd_entry'] = 'local_h100/train_video_evidence_opd_boundary_gpu_memory.py'
            result['hashes']['combined_evidence_opd_entry'] = training_resume.file_hash(__file__)
            result.pop('protocol_sha256', None)
            result['protocol_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
            return result
        trainer.make_protocol = protocol
        try:
            yield
        finally:
            trainer.make_protocol = original_protocol


def main(argv=None):
    from video_caption import train_video_9k as trainer
    argv = list(sys.argv[1:] if argv is None else argv)
    # Same base parser, selector and flag-removal contract as memory.main;
    # contexts are composed here once, never by recursively calling that main.
    parser = trainer.parser()
    parser._option_string_actions['--updates'].required = True
    parser.add_argument('--evidence-mode', choices=('auto', 'plain', 'text', 'visual'), default='auto')
    args = parser.parse_args(argv)
    if args.master_weights != 'gpu':
        parser.error('this diagnostic entry requires --master-weights gpu')
    local.validate_execution(args)
    mode = memory.select_evidence(args.init_from, args.evidence_mode)
    if mode == 'none':
        parser.error('this entry requires an explicit time-off evidence head')
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--evidence-mode')
    _, inherited_argv = selector.parse_known_args(argv)
    with combined_runtime(trainer, mode):
        return trainer.main(inherited_argv)


if __name__ == '__main__':
    main()
