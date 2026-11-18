"""Run the existing GPU master OPD trainer with one weights-only endpoint."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100.train_video_evidence_opd_boundary_gpu_memory import main

if __name__ == "__main__":
    main([*sys.argv[1:], "--weights-only-final"])
