#!/usr/bin/env python3
"""Build deeper draft heads (H-8 / H-12) from the warmed 5-layer head.

The capacity scan needs heads of 8 and 12 layers with everything else identical:
`target_layer_ids` stays [1,9,17,25,33], so `fc` / `norm` / `hidden_norm` keep
their shapes and only the head's own depth grows. Parameter counts then scale as
5:8:12 -- 209.7M -> 335M -> 503M -- which is exactly the H-5/H-8/H-12 ladder in
the task.

Warm start: the trained layers 0..4 are copied verbatim; the extra layers are
initialised by CYCLING the trained ones (layer i takes layer i % 5). Cycling
rather than duplicating the last layer keeps the 5 trained layer "types" spread
over the new depth instead of collapsing the tail to one of them. The choice is
recorded here because it is an init decision, not a recipe: the SFT run then
trains all layers from this starting point.

    python build_deep_head.py --src <5-layer ckpt> --layers 8 --out <dir>
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import argparse
import json
import os
import sys

import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/local/mcy/GLANCE/checkpoints/head_video439/step1000")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    base = json.load(open(os.path.join(a.src, "config.json")))
    n_src = int(base["num_hidden_layers"])
    assert a.layers > n_src, f"target depth {a.layers} must exceed source {n_src}"
    cfg = dict(base)
    cfg["num_hidden_layers"] = a.layers
    cfg["layer_types"] = ["full_attention"] * a.layers
    cfg["max_window_layers"] = a.layers

    from dflash.model import DFlashDraftModel
    # The repo loads the head with AutoConfig.from_pretrained(dir) + the class in
    # the config's auto_map; build the same way: write the config first, then
    # instantiate from that directory so no code path diverges from load_head().
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    from transformers import AutoConfig
    built = AutoConfig.from_pretrained(a.out)
    model = DFlashDraftModel(built)

    src_sd = DFlashDraftModel.from_pretrained(a.src, dtype=torch.bfloat16).state_dict()
    new_sd = model.state_dict()
    copied, cycled = 0, 0
    for k, v in new_sd.items():
        if k in src_sd and src_sd[k].shape == v.shape:
            new_sd[k] = src_sd[k].clone()
            copied += 1
        elif k.startswith("layers."):
            i = int(k.split(".")[1])
            srck = "layers.%d.%s" % (i % n_src, ".".join(k.split(".")[2:]))
            assert srck in src_sd, k
            assert src_sd[srck].shape == v.shape, (k, src_sd[srck].shape, v.shape)
            new_sd[k] = src_sd[srck].clone()
            cycled += 1
        else:
            raise SystemExit(f"unmatched parameter {k} {tuple(v.shape)}")
    model.load_state_dict(new_sd)
    n_param = sum(p.numel() for p in model.parameters())
    n_base = sum(p.numel() for p in DFlashDraftModel.from_pretrained(
        a.src, dtype=torch.bfloat16).parameters())
    print(f"H-{a.layers}: {n_param/1e6:.1f}M params (base H-{n_src} = {n_base/1e6:.1f}M; "
          f"copied {copied} tensors, cycled {cycled})", flush=True)
    # The ladder is exact because only the head's own layers are added: params grow
    # as (non-layer) + layers x (per-layer). The task's 209.7/340/510M figures are
    # the IMAGE-side 2B head; on the 8B video head the same ladder is
    # 1048.6M -> 1628M -> 2400M (measured, asserted here).
    per_layer = sum(v.numel() for k, v in src_sd.items() if k.startswith("layers.0."))
    expect = n_base + (a.layers - n_src) * per_layer
    assert abs(n_param - expect) / expect < 0.01, (
        f"param count {n_param/1e6:.1f}M does not follow the ladder "
        f"(expected ~{expect/1e6:.1f}M)")
    model.save_pretrained(a.out)
    print(f"saved {a.out}", flush=True)

    # Re-load through the project's own loader to prove the artifact is usable.
    back = common.load_head(a.out, device=f"cuda:{a.gpu}")
    print(f"reload ok: {sum(p.numel() for p in back.parameters())/1e6:.1f}M, "
          f"block {back.block_size}, target_layer_ids {back.target_layer_ids}", flush=True)


if __name__ == "__main__":
    main()
