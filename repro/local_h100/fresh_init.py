#!/usr/bin/env python3
"""Acquire and audit the immutable public DFlash head; never trains or uses CUDA.

This is a text-pretrained initialization, not a random initialization or a video
checkpoint. Keep its config and weights unchanged for reproducible adaptation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = "z-lab/Qwen3-8B-DFlash-b16"
REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"
MIRROR_REVISION = "783d39237915ffe36469b2437954d0bc7478135a"
WEIGHTS_SHA256 = "c702878094f38ad5843e6ac40b327720b3147ffbf669a5d9ca1974865ca6c080"
WEIGHTS_BYTES = 2097259104
CONFIG_SHA256 = "9834d608c9ca53d5548b415471ae9e8ebc9aab6cedfc2a7af95b6bd097373102"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str, size: int | None = None) -> None:
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Size mismatch: {path}")
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def write_exclusive(path: Path, payload: dict) -> None:
    # Reports cannot silently replace a previous completed audit.
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_contract(source: dict, target: dict) -> dict:
    text = target.get("text_config", {})
    if target.get("model_type") != "qwen3_vl":
        raise ValueError("Target must be qwen3_vl")
    for name, value in (("hidden_size", 4096), ("vocab_size", 151936),
                        ("head_dim", 128), ("num_attention_heads", 32),
                        ("num_key_value_heads", 8)):
        if source.get(name) != value or text.get(name) != value:
            raise ValueError(f"Public initialization/target mismatch: {name}")
    taps = source.get("dflash_config", {}).get("target_layer_ids")
    if taps != [1, 9, 17, 25, 33]:
        raise ValueError("Public head must use target taps [1,9,17,25,33]")
    if text.get("num_hidden_layers") != 36 or source.get("num_target_layers") != 36:
        raise ValueError("Target layer count mismatch")
    if source.get("num_hidden_layers") != 5 or source.get("block_size") != 16:
        raise ValueError("Unexpected public draft depth/block size")
    if source["dflash_config"].get("mask_token_id") != 151669:
        raise ValueError("Unexpected mask token")
    return {"hidden_size": 4096, "vocab_size": 151936, "target_layers": 36,
            "draft_layers": 5, "target_layer_ids": taps, "block_size": 16,
            "mask_token_id": 151669, "source_rope_theta": source.get("rope_theta"),
            "target_rope_theta": text.get("rope_theta"),
            "target_rope_scaling": text.get("rope_scaling"),
            "source_config_modified": False,
            "note": "Shape compatibility only. Text-to-video adaptation and GPU lossless gate remain required."}


def download(root: Path) -> None:
    import requests

    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.json"
    if config.exists():
        verify_file(config, CONFIG_SHA256, 1099)
    else:
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/config.json"
        response = requests.get(url, timeout=(20, 90))
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != CONFIG_SHA256:
            raise ValueError("Pinned public config SHA256 mismatch")
        with config.open("xb") as stream:
            stream.write(response.content)
    weights = root / "model.safetensors"
    if weights.exists():
        verify_file(weights, WEIGHTS_SHA256, WEIGHTS_BYTES)
        return
    part = root / "model.safetensors.incomplete"
    url = f"https://modelscope.cn/models/{REPO}/resolve/{MIRROR_REVISION}/model.safetensors"
    for attempt in range(4):
        start = part.stat().st_size if part.exists() else 0
        if start == WEIGHTS_BYTES:
            break
        if start > WEIGHTS_BYTES:
            raise ValueError("Incomplete download is larger than expected")
        try:
            with requests.get(url + f"?resume_offset={start}",
                              headers={"Range": f"bytes={start}-"},
                              stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not content_range.startswith(f"bytes {start}-"):
                    raise ValueError("Mirror did not honor exact Range")
                if int(content_range.split("/")[-1]) != WEIGHTS_BYTES:
                    raise ValueError("Mirror total size mismatch")
                print(f"DOWNLOAD offset={start} total={WEIGHTS_BYTES}", flush=True)
                last = time.monotonic()
                with part.open("ab") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        if time.monotonic() - last > 15:
                            print(f"PROGRESS bytes={stream.tell()} total={WEIGHTS_BYTES}", flush=True)
                            last = time.monotonic()
            if part.stat().st_size != WEIGHTS_BYTES:
                raise OSError("Incomplete response")
            break
        except requests.RequestException as exc:
            print(f"DOWNLOAD_RETRY {attempt + 1} {type(exc).__name__}: {exc}", flush=True)
            if attempt == 3:
                raise
            time.sleep(3)
    verify_file(part, WEIGHTS_SHA256, WEIGHTS_BYTES)
    # Never overwrite an existing historical checkpoint, even in a race.
    os.link(part, weights)
    part.unlink()
    print("PUBLIC_WEIGHTS_SHA256_VERIFIED", flush=True)


def audit(root: Path, target: Path) -> dict:
    # Explicitly disable CUDA before importing torch. The real Target is not loaded.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from safetensors import safe_open
    from transformers import Qwen3Config

    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "scripts/dflash_pkg/dflash-0.1.0"))
    from dflash.model import DFlashDraftModel

    verify_file(root / "config.json", CONFIG_SHA256, 1099)
    verify_file(root / "model.safetensors", WEIGHTS_SHA256, WEIGHTS_BYTES)
    source_config = json.loads((root / "config.json").read_text())
    target_config = json.loads((target / "config.json").read_text())
    contract = validate_contract(source_config, target_config)
    with torch.device("meta"):
        model = DFlashDraftModel(Qwen3Config.from_dict(source_config))
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as handle:
        actual = {name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        shapes = {k: [expected[k], actual[k]] for k in expected.keys() & actual.keys()
                  if expected[k] != actual[k]}
        raise ValueError(f"Strict local model schema mismatch: missing={missing} extra={extra} shapes={shapes}")
    del model
    model, info = DFlashDraftModel.from_pretrained(
        str(root), dtype=torch.bfloat16, local_files_only=True, output_loading_info=True)
    if any(info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Strict CPU model loading failed: {info}")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Fresh initializer must stay on CPU")
    return {"schema": "visionspec_public_head_v1", "status": "verified",
            "initialization": "public_text_pretrained_dflash_no_local_checkpoint",
            "repo": REPO, "revision": REVISION, "mirror_revision": MIRROR_REVISION,
            "weights_sha256": WEIGHTS_SHA256, "weights_bytes": WEIGHTS_BYTES,
            "config_sha256": CONFIG_SHA256, "root": str(root.resolve()),
            "target_config_sha256": sha256(target / "config.json"), "contract": contract,
            "strict_cpu_load": True, "tensor_count": len(actual),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "torch_version": torch.__version__, "gpu_test_performed": False,
            "source_model_sha256": sha256(project / "scripts/dflash_pkg/dflash-0.1.0/dflash/model.py")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/czsun/mcy/models/Qwen3-8B-DFlash-b16-public"))
    parser.add_argument("--target", type=Path, default=Path("/czsun/mcy/models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        download(args.out)
    report = audit(args.out, args.target)
    write_exclusive(args.out / "PUBLIC_INIT_VERIFIED.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
