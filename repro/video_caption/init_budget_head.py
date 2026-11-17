#!/usr/bin/env python3
"""Create an audited budget-conditioned copy of a trained video head.

Only a new nn.Embedding(2, hidden) is added; all source tensors must load
byte-for-byte unchanged. The new module is deliberately nonzero (std=0.02) so
the budget condition is active from update zero instead of a zero-init gate.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
sys.path.insert(0, str(REPRO / ".." / "scripts" / "dflash_pkg" / "dflash-0.1.0"))
from dflash.model import DFlashDraftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    src, out = Path(a.source).resolve(), Path(a.out).resolve()
    if not (src / "model.safetensors").is_file():
        raise SystemExit("source must contain model.safetensors")
    if out.exists():
        raise SystemExit("refusing to overwrite")
    torch.manual_seed(a.seed)
    config = AutoConfig.from_pretrained(src, trust_remote_code=True)
    if not isinstance(getattr(config, "dflash_config", None), dict):
        raise SystemExit("source has no dflash_config")
    config.dflash_config["budget_conditioning"] = True
    model = DFlashDraftModel(config)
    source = load_file(str(src / "model.safetensors"))
    result = model.load_state_dict(source, strict=False)
    expected_missing = {"budget_embedding.weight"}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise SystemExit(f"unexpected load: missing={result.missing_keys} unexpected={result.unexpected_keys}")
    after = model.state_dict()
    for name, expected in source.items():
        if not torch.equal(after[name], expected):
            raise SystemExit(f"source tensor changed: {name}")
    model.to(torch.bfloat16).eval()
    out.mkdir(parents=True)
    model.save_pretrained(out)
    summary = {
        "schema": 1,
        "source": str(src),
        "out": str(out),
        "source_weights_sha256": __import__("hashlib").sha256((src / "model.safetensors").read_bytes()).hexdigest(),
        "budget_conditioning": True,
        "budget_embedding_shape": list(model.budget_embedding.weight.shape),
        "budget_embedding_std": float(model.budget_embedding.weight.std()),
        "source_tensors_verified_equal": True,
        "missing_keys": sorted(result.missing_keys),
        "seed": a.seed,
    }
    (out / "INIT_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
