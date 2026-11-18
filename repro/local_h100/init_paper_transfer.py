"""Build or re-publish GPU6 paper-module initializations with verified base provenance."""
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


WEIGHT_FILE = "model.safetensors"
EVIDENCE_PREFIX = "evidence_readout."


def check_backbone(weights, base):
    names = {name for name in weights if not name.startswith(EVIDENCE_PREFIX)}
    if not names or names != base.keys():
        raise ValueError("paper-module backbone tensor names differ from plain source")
    for name in sorted(names):
        obtained, expected = weights[name], base[name]
        if (obtained.dtype != expected.dtype or obtained.shape != expected.shape
                or not torch.equal(obtained.view(torch.uint8), expected.view(torch.uint8))):
            raise ValueError(f"paper-module backbone tensor differs from plain source: {name}")
    return len(names)


def build(source, parent, reuse=None):
    cfg = json.loads((source / "config.json").read_text())
    weights_path = source / WEIGHT_FILE
    reference = json.loads((source / "BASE_INIT_MANIFEST.json").read_text())
    base_dir = Path(reference["source"])
    base_hashes = {WEIGHT_FILE: file_hash(base_dir / WEIGHT_FILE)}
    source_hash = file_hash(weights_path)
    if (reference["source_weights_sha256"] != base_hashes
            or reference["output_weights_sha256"] != {WEIGHT_FILE: source_hash}
            or file_hash(base_dir / "config.json") != reference["source_config_sha256"]):
        raise ValueError("source weights or config fail scaffold identity")
    base = load_file(base_dir / WEIGHT_FILE, device="cpu")
    state = load_file(weights_path, device="cpu")
    check_backbone(state, base)
    backbone = {k:state[k] for k in base}
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
            if reuse is not None:
                old = reuse/method/"init"
                original = json.loads((old/"INIT_MANIFEST.json").read_text())
                if (json.loads((old/"config.json").read_text()) != head_cfg
                        or original["source"] != str(source) or original["method"] != method
                        or original["source_config_sha256"] != file_hash(source/"config.json")):
                    raise ValueError(f"{method} does not match the original initialization")
                os.link(old/WEIGHT_FILE, staged/WEIGHT_FILE)
                os.link(old/"config.json", staged/"config.json")
            elif static:
                os.link(weights_path, staged/WEIGHT_FILE)
            else:
                save_file(dict(backbone, **{EVIDENCE_PREFIX+k:v for k,v in readout.state_dict().items()}),
                          staged/WEIGHT_FILE, metadata={"format":"pt"})
            published = load_file(staged/WEIGHT_FILE, device="cpu")
            count = check_backbone(published, base)
            readout.load_state_dict({k[len(EVIDENCE_PREFIX):]:v for k,v in published.items()
                                     if k.startswith(EVIDENCE_PREFIX)}, strict=True)
            result_hash = file_hash(staged/WEIGHT_FILE)
            if reuse is not None and original["final_weights_sha256"] != result_hash:
                raise ValueError(f"{method} does not match its frozen original head SHA")
            if not (staged/"config.json").exists():
                (staged/"config.json").write_text(json.dumps(head_cfg,indent=2))
            manifest = dict(method=method, source=str(base_dir), source_weights_sha256=base_hashes,
                            source_config_sha256=reference["source_config_sha256"],
                            base_tensors_verified_equal=True, unchanged_backbone_tensors=count,
                            scaffold_path=str(source), scaffold_weights_sha256={WEIGHT_FILE:source_hash},
                            scaffold_config_sha256=file_hash(source/"config.json"),
                            initialization_seed=20261118, feature_source=FEATURE_SOURCE,
                            final_weights_sha256=result_hash,
                            module_parameters=sum(p.numel() for p in readout.parameters()),
                            position=draft["evidence_position"], weight_identity="common trained backbone + method-specific visual interface")
            if reuse is not None:
                manifest.update(previous_initialization=str(old),
                                previous_manifest_sha256=file_hash(old/"INIT_MANIFEST.json"))
            (staged/"INIT_MANIFEST.json").write_text(json.dumps(manifest,indent=2))
            staged.rename(dest)
        print(method, dest, flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",type=Path)
    ap.add_argument("parent",type=Path)
    ap.add_argument("--reuse",type=Path,help="publish verified copies of existing initial weights in the new parent")
    args=ap.parse_args()
    build(args.source,args.parent,args.reuse)
