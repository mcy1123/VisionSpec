#!/usr/bin/env python3
"""Validate the EAGLE-3 video run config and the offline feature contract.

Everything here is static or CPU-only, so it runs before a GPU is available and
fails on the mistakes that would otherwise surface hundreds of steps into a
training run:

  * a run-config field that does not exist (`StrictConfigModel` is
    `extra="forbid"`, so a typo is a hard error, but only when someone loads it);
  * `attention_backend` / `strategy` / modality combinations that the algorithm
    registry rejects;
  * a draft config whose `rope_scaling` would silently select 1D RoPE, or whose
    hidden size disagrees with the target;
  * an embedding/lm_head key that is absent from the target's shard index -- the
    Qwen3-VL trap, since its text tower is nested under `model.language_model`;
  * captured `.ckpt` files missing `position_ids`, or whose shapes disagree with
    what `normalize_offline_sample` and the collator will demand;
  * a `max_length` that would truncate the longest stored row (which would make
    `loss_mask[0, -1] = 0` eat a supervised token instead of the template tail).

    python validate_offline_ingest.py --config run_eagle3_video.yaml \
        --target-config /ckpt/eagle3_video/qwen3vl8b_config.json \
        --features /data3/eagle3_video/train
"""
from __future__ import annotations

import argparse
import json
import os
import sys

FAILURES = []
NOTES = []


def fail(message):
    FAILURES.append(message)


def note(message):
    NOTES.append(message)


# ---------------------------------------------------------------------------
# 1. run config vs SpecForge's schema
# ---------------------------------------------------------------------------

def check_config(path):
    import yaml
    from specforge.config.schema import Config

    raw = yaml.safe_load(open(path))
    try:
        config = Config.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError carries the field names
        fail(f"run config rejected by SpecForge's schema: {exc}")
        return None, raw
    print(f"  schema: OK ({len(raw)} top-level keys)")
    return config, raw


def check_registry(config):
    """Ask the composition root whether this combination is legal at all."""
    from specforge.application.composition import resolve_run

    try:
        resolved = resolve_run(config)
    except Exception as exc:
        fail(f"resolve_run rejected the config: {exc}")
        return False
    print(f"  registry: OK (mode={config.mode}, "
          f"strategy={config.training.strategy}, "
          f"attention_backend={config.training.attention_backend})")
    return True


# ---------------------------------------------------------------------------
# 2. draft config
# ---------------------------------------------------------------------------

def check_draft(draft_path, target_config_path):
    draft = json.load(open(draft_path))
    target = json.load(open(target_config_path))
    text = target["text_config"]

    rope = draft.get("rope_scaling") or {}
    kind = rope.get("rope_type", rope.get("type"))
    if kind != "mrope":
        fail(f"draft rope_scaling dispatches to {kind!r}, not 'mrope'; "
             f"_init_rope would install 1D RoPE")
    if rope.get("mrope_interleaved") is not True:
        fail("draft rope_scaling lacks mrope_interleaved: true; Qwen3-VL is "
             "interleaved and the chunked layout silently mis-rotates")
    expected_section = text.get("rope_scaling", {}).get("mrope_section")
    if list(rope.get("mrope_section", [])) != list(expected_section or []):
        fail(f"draft mrope_section {rope.get('mrope_section')} != target's "
             f"{expected_section}")

    for key in ("hidden_size", "intermediate_size", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps"):
        if draft.get(key) != text.get(key):
            fail(f"draft {key}={draft.get(key)} != target text_config "
                 f"{text.get(key)}")
    if draft.get("rope_theta") != text.get("rope_theta"):
        fail(f"draft rope_theta {draft.get('rope_theta')} != target "
             f"{text.get('rope_theta')}")
    if draft.get("num_hidden_layers") != 1:
        fail(f"EAGLE-3 forces exactly 1 draft layer, config says "
             f"{draft.get('num_hidden_layers')}")
    if draft.get("tie_word_embeddings"):
        fail("draft must not tie word embeddings (it has its own draft lm_head)")

    # Parameter budget, so the '~0.40B' claim in the report is checked, not assumed.
    h = draft["hidden_size"]
    i = draft["intermediate_size"]
    vocab = draft.get("draft_vocab_size") or draft["vocab_size"]
    heads, kv = draft["num_attention_heads"], draft["num_key_value_heads"]
    hd = draft["head_dim"]
    fc = 3 * h * h + h
    attn = h * heads * hd + 2 * (h * kv * hd) + heads * hd * h
    mlp = 3 * h * i
    head = h * vocab
    total = fc + attn + mlp + head
    print(f"  draft params: fc {fc / 1e6:.1f}M + layer {attn / 1e6:.1f}M attn + "
          f"{mlp / 1e6:.1f}M mlp + {head / 1e6:.1f}M head = {total / 1e9:.3f}B "
          f"(draft_vocab {vocab})")
    if not 0.30e9 <= total <= 0.50e9:
        note(f"trainable draft is {total / 1e9:.3f}B, outside the ~0.40B the "
             f"comparison table claims -- update the table or the config")
    return draft


# ---------------------------------------------------------------------------
# 3. target checkpoint keys
# ---------------------------------------------------------------------------

def check_target_keys(target_dir, embedding_key, lm_head_key):
    index_path = os.path.join(target_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        fail(f"no {index_path}; TargetHead.load_weights hard-requires a sharded "
             f"checkpoint (it globs *.index.json and raises otherwise)")
        return
    weight_map = json.load(open(index_path))["weight_map"]
    for label, key in (("embedding_key", embedding_key), ("lm_head_key", lm_head_key)):
        if key not in weight_map:
            near = [k for k in weight_map
                    if k.endswith(key.split(".")[-1]) or
                    key.split(".")[-1] in k][:4]
            fail(f"{label}={key!r} is not in the target shard index; "
                 f"candidates: {near}")
        else:
            print(f"  {label}: {key} -> {weight_map[key]}")


# ---------------------------------------------------------------------------
# 4. captured feature contract
# ---------------------------------------------------------------------------

def check_features(features_dir, max_length, hidden_size, aux_layers):
    import torch

    if not os.path.isdir(features_dir):
        note(f"features dir {features_dir} does not exist yet "
             f"(expected before capture runs)")
        return
    files = []
    for root, _dirs, names in os.walk(features_dir):
        files += [os.path.join(root, n) for n in names if n.endswith(".ckpt")]
    if not files:
        note(f"no .ckpt under {features_dir} yet")
        return
    files.sort()
    required = ("input_ids", "loss_mask", "hidden_state", "aux_hidden_state",
                "position_ids")
    checked = longest = 0
    worst = None
    for path in files[:8]:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        for key in required:
            if key not in raw:
                fail(f"{os.path.basename(path)}: missing {key!r} "
                     f"(present: {sorted(raw)})")
                return
        ids = raw["input_ids"]
        if ids.ndim != 1:
            fail(f"{os.path.basename(path)}: input_ids ndim {ids.ndim}, expected 1")
        if raw["loss_mask"].ndim != 1:
            fail(f"{os.path.basename(path)}: loss_mask must be 1-D")
        if int(raw["loss_mask"][-1]) != 0:
            fail(f"{os.path.basename(path)}: loss_mask[-1] is 1; "
                 f"normalize_offline_sample would zero a supervised token")
        if int(raw["loss_mask"].sum()) == 0:
            fail(f"{os.path.basename(path)}: loss_mask is all zero")
        hid = raw["hidden_state"]
        aux = raw["aux_hidden_state"]
        if hid.shape != (1, ids.shape[0], hidden_size):
            fail(f"{os.path.basename(path)}: hidden_state {tuple(hid.shape)}, "
                 f"expected (1, {ids.shape[0]}, {hidden_size})")
        if aux.shape != (1, ids.shape[0], 3 * hidden_size):
            fail(f"{os.path.basename(path)}: aux_hidden_state {tuple(aux.shape)}, "
                 f"expected (1, {ids.shape[0]}, {3 * hidden_size})")
        pos = raw["position_ids"]
        if pos.ndim != 3 or pos.shape[0] != 3 or pos.shape[1] != 1 \
                or pos.shape[2] != ids.shape[0]:
            fail(f"{os.path.basename(path)}: position_ids {tuple(pos.shape)}, "
                 f"expected (3, 1, {ids.shape[0]}) for the 3-D collator branch")
        checked += 1
        if ids.shape[0] > longest:
            longest, worst = ids.shape[0], os.path.basename(path)
    print(f"  features: checked {checked} of {len(files)} .ckpt; "
          f"longest {longest} ({worst})")
    if longest > max_length:
        fail(f"max_length {max_length} < longest stored row {longest}; the "
             f"normalizer would truncate and then zero a supervised token")


# ---------------------------------------------------------------------------
# 5. path reachability, as this process sees it
# ---------------------------------------------------------------------------

def check_paths(raw):
    for label, path in (("target_model_path",
                         raw.get("model", {}).get("target_model_path")),
                        ("draft_model_config",
                         raw.get("model", {}).get("draft_model_config")),
                        ("hidden_states_path",
                         raw.get("data", {}).get("hidden_states_path"))):
        if not path:
            continue
        exists = os.path.exists(path)
        print(f"  path {label}: {path} {'OK' if exists else 'MISSING'}")
        if not exists:
            note(f"{label} not visible from this process: {path} "
                 f"(expected when validating outside the training container)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target-config", required=True,
                    help="the target's own config.json")
    ap.add_argument("--target-dir", default=None,
                    help="target checkpoint dir; defaults to the config's dirname")
    ap.add_argument("--features", default=None)
    a = ap.parse_args()

    raw = json.load(open(a.target_config))
    target_dir = a.target_dir or os.path.dirname(os.path.abspath(a.target_config))

    print("[1] run config vs SpecForge schema")
    config, raw_cfg = check_config(a.config)
    if config is not None:
        check_registry(config)
    check_paths(raw_cfg)

    print("\n[2] draft config vs target")
    draft_path = raw_cfg["model"]["draft_model_config"]
    if os.path.exists(draft_path):
        draft = check_draft(draft_path, a.target_config)
    else:
        fail(f"draft config not found: {draft_path}")
        draft = None

    print("\n[3] target checkpoint keys")
    check_target_keys(target_dir, raw_cfg["model"]["embedding_key"],
                      raw_cfg["model"]["lm_head_key"])

    print("\n[4] captured feature contract")
    features = a.features or raw_cfg["data"]["hidden_states_path"]
    if draft is not None:
        check_features(features, int(raw_cfg["data"]["max_length"]),
                       int(draft["hidden_size"]),
                       raw_cfg["model"].get("aux_hidden_state_layer_ids"))

    if NOTES:
        print("\nnotes")
        for item in NOTES:
            print(f"  - {item}")
    if FAILURES:
        print(f"\nFAIL ({len(FAILURES)})")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("\nPASS: config, draft, target keys and feature contract are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
