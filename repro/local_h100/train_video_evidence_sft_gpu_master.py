#!/usr/bin/env python3
"""Time-off evidence coverage-CE pilot with explicit GPU master placement.

All training and checkpoint behavior comes from the three existing wrappers.
CLI is unchanged; checkpoints are weights-only with no optimizer resume.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

from local_h100 import train_video_sft as base
from local_h100 import train_video_sft_pilot as pilot
from local_h100 import train_video_sft_gpu_master as gpu
from local_h100 import train_video_evidence_sft as evidence
from shared import training_resume


@contextmanager
def combined_runtime():
    # Evidence freezes time.weight after blanket unfreeze, then calls the GPU
    # optimizer factory. Reversing these contexts would bypass that policy.
    with pilot.pilot_runtime(), gpu.gpu_master_runtime(), evidence.evidence_runtime():
        original_protocol = base.build_protocol

        def protocol(*args, **kwargs):
            result = original_protocol(*args, **kwargs)
            result['evidence_gpu_master_base_entry'] = result['entry']
            result['entry'] = 'local_h100/train_video_evidence_sft_gpu_master.py'
            result['hashes']['evidence_gpu_master_entry'] = training_resume.file_hash(__file__)
            result.pop('protocol_sha256', None)
            result['protocol_sha256'] = base.digest(result)
            return result

        base.build_protocol = protocol
        try:
            yield
        finally:
            base.build_protocol = original_protocol


def main(argv=None):
    with combined_runtime():
        return base.main(argv)


if __name__ == '__main__':
    main()
