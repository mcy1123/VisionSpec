"""CPU preflight for the fixed target/head interface of the OPD diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

from shared import common


def validate_model_contract(target_path, head_path):
    """Reject incompatible local model configs before allocating target weights.

    Matching dimensions is necessary, not proof of checkpoint provenance. The
    caller must also record the exact target and head identities in its protocol.
    These entry points currently use common.KEPT_LAYERS for training features.
    """
    target = json.loads((Path(target_path) / "config.json").read_text())
    head = json.loads((Path(head_path) / "config.json").read_text())
    text = target.get("text_config", target)
    for field in ("hidden_size", "vocab_size"):
        if field not in text or field not in head or text[field] != head[field]:
            raise ValueError(f"target/head {field} mismatch: {text.get(field)} vs {head.get(field)}")
    layers = int(text["num_hidden_layers"])
    if int(head.get("num_target_layers", -1)) != layers:
        raise ValueError("target/head num_target_layers mismatch")
    taps = head.get("dflash_config", {}).get("target_layer_ids", head.get("target_layer_ids"))
    if taps != list(common.KEPT_LAYERS):
        raise ValueError(f"head target_layer_ids must match this trainer's taps {common.KEPT_LAYERS}")
    if any(not isinstance(layer, int) or isinstance(layer, bool) or layer >= layers or layer < 0
           for layer in taps):
        raise ValueError("head target_layer_ids must be valid integer layer indices")
    return dict(hidden_size=text["hidden_size"], vocab_size=text["vocab_size"],
                num_target_layers=layers, target_layer_ids=taps)
