"""Build the GPU6 paper-module initializations from one fixed DFlash head."""
import argparse
import json
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()

import torch
from safetensors.torch import load_file, save_file
from dflash.paper_transfer import METHODS, FEATURE_SOURCE, paper_readout
from shared.training_resume import file_hash


def build(source, parent):
    cfg = json.loads((source / "config.json").read_text())
    weights_path = source / "model.safetensors"
    state = load_file(weights_path, device="cpu")
    source_hash = file_hash(weights_path)
    prefix = "evidence_readout."
    backbone = {k: v for k, v in state.items() if not k.startswith(prefix)}
    for method in METHODS:
        dest = parent / method / "init"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(dest)
        draft = dict(cfg["dflash_config"], paper_method=method, video_layout="grid",
                     evidence_feature_source=FEATURE_SOURCE, evidence_position="output" if method=="fovea" else "input")
        head_cfg = dict(cfg, dflash_config=draft)
        torch.manual_seed(20261118)
        readout = paper_readout(method, hidden_size=cfg["hidden_size"], dim=draft["evidence_dim"],
                                max_tokens=draft["evidence_max_tokens"], spatial_bins=draft["evidence_spatial_bins"], mode="visual")
        static = method in ("projector_grid", "llava_pool", "tome")
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            staged = Path(tmp)/"init"
            staged.mkdir()
            if static:
                readout.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
                os.link(weights_path, staged/"model.safetensors")
            else:
                save_file(dict(backbone, **{prefix+k:v for k,v in readout.state_dict().items()}), staged/"model.safetensors", metadata={"format":"pt"})
            (staged/"config.json").write_text(json.dumps(head_cfg,indent=2))
            (staged/"INIT_MANIFEST.json").write_text(json.dumps(dict(method=method,source=str(source),
                source_weights_sha256=source_hash, source_config_sha256=file_hash(source/"config.json"),
                initialization_seed=20261118, feature_source=FEATURE_SOURCE,
                final_weights_sha256=source_hash if static else file_hash(staged/"model.safetensors"),
                unchanged_backbone_tensors=len(backbone), module_parameters=sum(p.numel() for p in readout.parameters()),
                position=draft["evidence_position"], weight_identity="common trained backbone + method-specific visual interface"),indent=2))
            staged.rename(dest)
        print(method, dest, flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",type=Path)
    ap.add_argument("parent",type=Path)
    args=ap.parse_args()
    build(args.source,args.parent)
