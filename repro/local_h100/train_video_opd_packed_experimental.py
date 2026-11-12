#!/usr/bin/env python3
"""Future opt-in packed OPD probe; CPU proof is not CUDA/throughput validation.

No existing experiment uses this entry. Requires explicit update budget, GPU
master, lambda=0, and the original global_pooled objective. Rollout, frozen
teacher and full-state checkpoint implementation remain unchanged.
Packed graphs may raise peak memory and change bf16 accumulation/Adam updates;
adoption needs a separate same-weight, same-data CUDA numerical/memory gate.
"""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import sys

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

from local_h100 import train_video_opd as local
from local_h100 import train_video_opd_gpu_memory as memory
from local_h100.train_video_evidence_opd import evidence_runtime
from local_h100.packed_opd_score import packed_runtime
from shared import training_resume


@contextmanager
def combined_runtime(trainer, mode, pack_size=4):
    with local.local_runtime(trainer), memory.memory_runtime(trainer, mode), \
            (evidence_runtime(trainer) if mode != 'none' else nullcontext()), \
            packed_runtime(trainer, pack_size):
        original_protocol = trainer.make_protocol
        def protocol(*args, **kwargs):
            value = original_protocol(*args, **kwargs)
            value['experimental_packed_entry'] = 'local_h100/train_video_opd_packed_experimental.py'
            value['hashes']['experimental_packed_entry'] = training_resume.file_hash(__file__)
            value.pop('protocol_sha256', None)
            value['protocol_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            return value
        trainer.make_protocol = protocol
        try:
            yield
        finally:
            trainer.make_protocol = original_protocol


def main(argv=None):
    from video_caption import train_video_9k as trainer
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = trainer.parser()
    parser.description = __doc__
    parser._option_string_actions['--updates'].required = True
    parser.add_argument('--evidence-mode', choices=('auto', 'plain', 'text', 'visual'), default='auto')
    parser.add_argument('--packed-anchors', type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args(argv)
    if args.master_weights != 'gpu' or args.e2e_tv_weight != 0 or args.loss_normalization != 'global_pooled':
        parser.error('experimental packing requires --master-weights gpu --e2e-tv-weight 0 --loss-normalization global_pooled')
    local.validate_execution(args)
    mode = memory.select_evidence(args.init_from, args.evidence_mode)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--evidence-mode')
    selector.add_argument('--packed-anchors')
    _, inherited_argv = selector.parse_known_args(argv)
    with combined_runtime(trainer, mode, args.packed_anchors):
        return trainer.main(inherited_argv)


if __name__ == '__main__':
    main()
