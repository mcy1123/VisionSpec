#!/usr/bin/env python3
"""Prove SpecForge ingests our features, including 3-axis position_ids.

`specforge_patches.py --check` is static: it confirms the source text contains
the patches. This is the behavioural half. It drives the real reader, the real
normalizer and the real collator over real `.ckpt` files and asserts the tensors
that come out the far end.

The two `position_ids` patches both fail silently if absent, which is why they
are worth a runtime test:

  * without `"position_ids"` in `_OFFLINE_EAGLE3_KEYS`, the feature store fetches
    only the keys a SampleRef asks for, so the tensor on disk is simply never
    read -- no error, and the draft trains with plain 1D RoPE;
  * without the `normalize_offline_sample` branch, even a fetched tensor has no
    route into the model's forward signature.

So the decisive assertion is that `position_ids` survives to the collated batch
with shape `[axes, batch, seq]`. `SPECFORGE_VALIDATE_OFFLINE_FEATURES=1` is set
so a missing or malformed key fails at assembly rather than lazily.

    python test_offline_ingest.py --features /data3/eagle3_video/train_stub
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--ttt-length", type=int, default=3)
    ap.add_argument("--limit", type=int, default=4)
    a = ap.parse_args()

    # Fail at assembly, not lazily at first materialize.
    os.environ["SPECFORGE_VALIDATE_OFFLINE_FEATURES"] = "1"

    import torch

    from specforge.algorithms.eagle3.data import (
        build_offline_collator,
        build_offline_normalizer,
        build_offline_reader,
    )
    from specforge.runtime.data_plane.offline_reader import (
        _OFFLINE_EAGLE3_KEYS,
    )

    print(f"feature dir      : {a.features}")
    print(f"_OFFLINE_EAGLE3_KEYS = {_OFFLINE_EAGLE3_KEYS}")
    if "position_ids" not in _OFFLINE_EAGLE3_KEYS:
        print("\nFAIL: the reader does not list position_ids; the 3-axis mRoPE "
              "tensor would be dropped silently")
        return 1

    reader = build_offline_reader(a.features, run_id="eagle3-video",
                                  ttt_length=a.ttt_length, max_len=a.max_len)
    refs = reader.read(limit=a.limit)
    print(f"reader returned  : {len(refs)} sample refs")
    if not refs:
        print("\nFAIL: reader found no .ckpt files")
        return 1
    for ref in refs[:2]:
        print(f"  {ref.sample_id}  keys={sorted(ref.feature_keys)}")

    from specforge.runtime.data_plane.feature_store import LocalFeatureStore

    # LocalFeatureStore is the concrete store OfflineManifestReader's file://
    # refs are designed for; FeatureStore itself is abstract.
    store = LocalFeatureStore()
    normalize = build_offline_normalizer(a.max_len, ttt_length=a.ttt_length)
    collate = build_offline_collator()

    samples = []
    for ref in refs:
        tensors, handle = store.get(ref)
        try:
            samples.append(normalize(tensors))
        finally:
            store.release(handle)

    print(f"\nnormalized {len(samples)} samples")
    first = samples[0]
    print(f"  keys: {sorted(first)}")
    for key in ("input_ids", "loss_mask", "target", "hidden_state",
                "attention_mask", "position_ids"):
        if key not in first:
            print(f"\nFAIL: normalizer output is missing {key!r}")
            return 1
        print(f"  {key:16s} {tuple(first[key].shape)} {first[key].dtype}")

    positions = first["position_ids"]
    if positions.ndim != 3:
        print(f"\nFAIL: position_ids ndim {positions.ndim}; the collator only "
              f"pads the 3-D branch, so a 2-D tensor would be misread as a batch")
        return 1
    if positions.shape[0] != 3:
        print(f"\nFAIL: position_ids axes {positions.shape[0]}, expected 3")
        return 1
    if first["loss_mask"][0, -1] != 0:
        print("\nFAIL: loss_mask[-1] is not zero; normalize_offline_sample "
              "should have zeroed the target-less template tail")
        return 1

    batch = collate(samples)
    print("\ncollated batch")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:16s} {tuple(value.shape)} {value.dtype}")
        else:
            print(f"  {key:16s} {value!r}")

    ok = True
    if "position_ids" not in batch or batch["position_ids"] is None:
        print("\nFAIL: position_ids was dropped by the collator")
        return 1
    bp = batch["position_ids"]
    if bp.ndim != 3 or bp.shape[0] != 3 or bp.shape[1] != len(samples):
        print(f"\nFAIL: collated position_ids {tuple(bp.shape)}, expected "
              f"(3, {len(samples)}, seq)")
        return 1
    seq = batch["hidden_state"].shape[1]
    if bp.shape[-1] != seq:
        print(f"\nFAIL: position_ids last dim {bp.shape[-1]} != hidden_state "
              f"seq {seq}")
        return 1

    # The aux width the draft asserts on: 3 * target hidden.
    aux_width = batch["hidden_state"].shape[-1]
    target_width = batch["target"].shape[-1]
    if aux_width != 3 * target_width:
        print(f"\nFAIL: aux width {aux_width} != 3 x target width {target_width}; "
              f"LlamaForCausalLMEagle3.forward asserts this")
        return 1

    # A real axis difference, so we know the tensor is not a degenerate copy.
    if int(bp[0].max()) == int(bp[0].min()):
        print("\nnote: stub position_ids are all zero (expected for --stub); "
              "re-run against real captured features to check the axis spread")

    print(f"\nPASS: reader -> normalizer -> collator carries position_ids as "
          f"{tuple(bp.shape)} alongside hidden_state {aux_width} = 3 x "
          f"{target_width}")
    print("Both position_ids patches are functionally in place.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
