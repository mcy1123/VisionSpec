"""Versioned full training-state contract shared by active SFT adapters."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import hashlib
import json
import random
import os
import uuid
from pathlib import Path
import numpy as np
import torch


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_rng():
    return dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint requires CUDA")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def identity(args, schedule):
    """Hash inputs and semantics; output location/device are not data identity."""
    fields = [
        "seed",
        "qa_per_update",
        "anchors_per_qa",
        "lr",
        "clip",
        "master_weights",
        "target_dtype",
        "time_mode",
        "max_pixels",
        "max_seq_len",
    ]
    config = {k: getattr(args, k, None) for k in fields}
    if getattr(args, "allow_partial_final", False):
        config["allow_partial_final"] = True
    config["model_config_sha256"] = file_hash(Path(args.init_from) / "config.json")
    return dict(
        schedule_sha256=file_hash(schedule),
        data_sha256={k: file_hash(getattr(args, k)) for k in ["pool", "records"]},
        config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
    )


def extras(step, metadata):
    return dict(
        resume_schema=1,
        update_step=step,
        data_cursor=step,
        resume_identity=metadata,
        rng=capture_rng(),
    )


def restore(state, head, optimizer, metadata, requested_step=None):
    if state.get("resume_schema") != 1:
        raise ValueError(
            "legacy checkpoint lacks full resume identity; use historical replay explicitly"
        )
    if state["resume_identity"] != metadata:
        raise ValueError("resume schedule/data/model configuration hash mismatch")
    step = state["update_step"]
    if step != state["data_cursor"] or step != state.get("update", step):
        raise ValueError("checkpoint cursor/update mismatch")
    if requested_step is not None and requested_step != step:
        raise ValueError("CLI resume step differs from checkpoint")
    head.load_state_dict(state["head"])
    optimizer.load_state_dict(state["master_optimizer"])
    restore_rng(state["rng"])
    return step


def validate_output(path):
    resolved = str(Path(path).resolve())
    if not any(
        resolved.startswith(x + "/")
        for x in ["/mnt/local", "/mnt/local2", "/mnt/local3"]
    ):
        raise ValueError("checkpoint output must resolve under /mnt/local{,2,3}")


def save_atomic_checkpoint(destination, head, state):
    """Publish a complete, hashed training checkpoint without replacing a prior one."""
    dest = Path(destination)
    validate_output(dest)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{uuid.uuid4().hex}")
    tmp.mkdir()
    head.save_pretrained(tmp)
    torch.save(state, tmp / "training_state.pt")
    hashes = {p.name: file_hash(p) for p in tmp.iterdir() if p.is_file()}
    (tmp / "SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    os.rename(tmp, dest)
    return hashes
