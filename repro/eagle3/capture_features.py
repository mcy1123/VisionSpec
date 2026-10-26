#!/usr/bin/env python3
"""Capture EAGLE-3 offline features for the video domain from Qwen3-VL-8B.

SpecForge cannot do this itself. Its capture runtime is text-only, SGLang-only,
and forces `ForwardMode.DECODE`, which excludes SGLang's own multimodal
embedding branch -- so even injecting media would silently produce text-only
hidden states for the video placeholder tokens. `target_model_type` is a dead
field (no Python consumer anywhere), and `model.input_modality` is gated to
`"text"` for EAGLE3 offline.

What SpecForge *can* do is train from hand-written features: `load_feature_file`
is a plain `torch.load`, the reader only walks for `*.ckpt`, and no manifest or
version stamp is checked. So this script produces exactly the binary contract
SpecForge's offline EAGLE-3 trainer consumes.

Why not re-tokenise the JSONL here. The training sequence must be the one the
labels belong to. `build_eagle3_data.py` proved that `encode(decode(ids)) == ids`
for 496/499 rows but not all (Qwen's BPE admits non-canonical segmentations), and
that Qwen's chat template appends a turn-closing tail the rollout does not
contain. So this script renders the conversation through the SAME processor
GLANCE used, runs one teacher-forced forward over it, and derives every tensor
from that single pass -- ids, hidden states and mask are then self-consistent by
construction.

The loss mask follows SpecForge's own parser semantics rather than a
reconstruction of it: `parse.py` supervises every token from the first assistant
content token through the turn-closing `<|im_end|>` inclusive. Because the
template's trailing `\\n` then sits at the last index, `normalize_offline_sample`
zeroing `loss_mask[0, -1]` lands on that target-less newline instead of eating a
real supervised token. That is why the stored sequence is the FULL rendered
length and not truncated to the rollout's `seq_len`:
`loss_mask[prompt_len : R - 1] = 1`, where `R - 1` is the `<|im_end|>`'s index
when the rollout was truncated, or the trailing `\\n` when it was not.

Output contract, one `.ckpt` per row:

    input_ids         int64 [L]              full rendered sequence
    loss_mask         int64 [L]              supervised positions
    hidden_state      bf16  [1, L, 4096]     target final hidden -> trainer `target`
    aux_hidden_state  bf16  [1, L, 12288]    3 aux layers concatenated -> `hidden_state`
    position_ids      int64 [3, 1, L]        mRoPE positions (SpecForge extension)

Standard EAGLE-3 uses 3 aux layers at low/mid/high depth. SpecForge's default for
a 36-layer target is `[1, 17, 32]` (= `[1, n//2-1, n-4]`); that is a deliberate
choice to use EAGLE-3's own recipe rather than GLANCE's 5 kept layers, so the
comparison is not tuned toward either method.

    # real capture, one GPU
    python capture_features.py --split train --out /data3/eagle3_video/train

    # zero-GPU plumbing check: synthetic tensors with the real shapes
    python capture_features.py --split train --stub --limit 8 --out /tmp/stub
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import common  # noqa: E402

DEFAULT_AUX_LAYERS = (1, 17, 32)  # [1, n//2-1, n-4] for n=36, SpecForge default


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_split(manifest_path, jsonl_path, split):
    """JSONL rows plus the manifest entry that pins the token boundaries."""
    manifest = json.load(open(manifest_path))
    by_id = {m["id"]: m for m in manifest["splits"][split]}
    rows = read_jsonl(jsonl_path)
    for row in rows:
        if row["id"] not in by_id:
            raise AssertionError(f"{row['id']} present in {jsonl_path} but not "
                                 f"in manifest split {split!r}")
    return rows, by_id


def build_loss_mask(prompt_len, render_len, seq_len, ended_with_eos):
    """SpecForge's `parse.py` span, expressed on the rendered sequence.

    Supervised = first assistant content token .. turn-closing `<|im_end|>`.
    The template's trailing newline stays unsupervised; `normalize_offline_sample`
    then re-zeros that same last index, which is a no-op by design.
    """
    mask = torch.zeros(render_len, dtype=torch.long)
    # `seq_len` is where the rollout stopped. If it ended with EOS that last
    # token IS `<|im_end|>`; if it was truncated, the template supplied one.
    goodbye_index = seq_len - 1 if ended_with_eos else seq_len
    if not (prompt_len <= goodbye_index < render_len):
        raise AssertionError(
            f"inconsistent boundaries: prompt_len={prompt_len} "
            f"goodbye={goodbye_index} render_len={render_len}")
    mask[prompt_len : goodbye_index + 1] = 1
    # The final position must be the template tail, never a supervised token.
    if int(mask[-1]) != 0:
        raise AssertionError("last position is supervised; the template tail is "
                             "missing, so normalize_offline_sample would zero a "
                             "real label")
    return mask, goodbye_index


@torch.no_grad()
def capture_one(target, processor, row, man, aux_layers, device):
    """One teacher-forced forward; every tensor comes out of this single pass."""
    convo = row["conversations"]
    rendered = processor.apply_chat_template(
        convo, add_generation_prompt=False, tokenize=False)
    ins = processor(text=[rendered], videos=[man["video"]], return_tensors="pt",
                    fps=man["fps"])
    input_ids = ins["input_ids"].to(device)
    pv = ins["pixel_values_videos"].to(device)
    vg = ins["video_grid_thw"].to(device)

    render_len = int(input_ids.shape[1])
    expected = man["seq_len"] + (1 if man["ended_with_eos"] else 2)
    if render_len != expected:
        raise AssertionError(
            f"{man['id']}: rendered {render_len} tokens, manifest says {expected} "
            f"(seq_len {man['seq_len']}, eos {man['ended_with_eos']})")

    # Same call GLANCE makes (common.target_context): video-conditional mRoPE.
    positions, _deltas = target.model.get_rope_index(input_ids, None, vg, None)

    out = target(input_ids=input_ids, pixel_values_videos=pv, video_grid_thw=vg,
                 position_ids=positions, use_cache=False,
                 output_hidden_states=True, logits_to_keep=1)
    hs = out.hidden_states  # [embeddings, layer1, ..., layer36]
    n_layers = len(hs) - 1
    for layer in aux_layers:
        if not 1 <= layer <= n_layers:
            raise AssertionError(f"aux layer {layer} outside 1..{n_layers}")
    if len(set(aux_layers)) != 3:
        raise AssertionError(f"EAGLE-3 expects exactly 3 distinct aux layers, "
                             f"got {aux_layers}")

    aux = torch.cat([hs[l] for l in aux_layers], dim=-1)  # [1, L, 3*hidden]
    final = hs[-1]                                        # [1, L, hidden]
    if aux.shape[-1] != final.shape[-1] * 3:
        raise AssertionError(f"aux width {aux.shape[-1]} != 3 x hidden "
                             f"{final.shape[-1]}")

    mask, goodbye = build_loss_mask(
        man["prompt_len"], render_len, man["seq_len"], man["ended_with_eos"])

    return {
        "input_ids": input_ids[0].to(torch.int64).cpu(),
        "loss_mask": mask,
        "hidden_state": final.to(torch.bfloat16).cpu(),
        "aux_hidden_state": aux.to(torch.bfloat16).cpu(),
        "position_ids": positions.to(torch.int64).cpu(),
    }, {"id": man["id"], "render_len": render_len, "goodbye_index": goodbye,
        "supervised": int(mask.sum()), "aux_layers": list(aux_layers)}


def make_stub(row, man, width, aux_layers):
    """Synthetic tensors with the real shapes, for zero-GPU plumbing checks."""
    render_len = man["seq_len"] + (1 if man["ended_with_eos"] else 2)
    mask, goodbye = build_loss_mask(
        man["prompt_len"], render_len, man["seq_len"], man["ended_with_eos"])
    return {
        "input_ids": torch.randint(0, 1000, (render_len,), dtype=torch.int64),
        "loss_mask": mask,
        "hidden_state": torch.zeros(1, render_len, width, dtype=torch.bfloat16),
        "aux_hidden_state": torch.zeros(1, render_len, width * 3,
                                        dtype=torch.bfloat16),
        "position_ids": torch.zeros(3, 1, render_len, dtype=torch.int64),
    }, {"id": man["id"], "render_len": render_len, "goodbye_index": goodbye,
        "supervised": int(mask.sum()), "aux_layers": list(aux_layers)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(common.DATA, "eagle3_manifest.json"))
    ap.add_argument("--split", required=True, choices=["train", "select", "holdout30"])
    ap.add_argument("--jsonl", default=None,
                    help="defaults to <DATA>/eagle3_<split>.jsonl")
    ap.add_argument("--out", required=True, help="output dir for .ckpt shards")
    ap.add_argument("--aux-layers", default=",".join(map(str, DEFAULT_AUX_LAYERS)))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows")
    ap.add_argument("--group-size", type=int, default=2000,
                    help="rows per rows_A-B/ subdirectory")
    ap.add_argument("--stub", action="store_true",
                    help="zero-GPU: write synthetic tensors of the real shape")
    ap.add_argument("--resume", action="store_true",
                    help="skip rows whose .ckpt already exists")
    a = ap.parse_args()

    aux_layers = tuple(int(x) for x in a.aux_layers.split(","))
    jsonl = a.jsonl or os.path.join(common.DATA, f"eagle3_{a.split}.jsonl")
    rows, by_id = load_split(a.manifest, jsonl, a.split)
    if a.limit:
        rows = rows[: a.limit]
    print(f"split {a.split}: {len(rows)} rows from {jsonl}", flush=True)
    print(f"aux layers {aux_layers} (EAGLE-3 default [1, n//2-1, n-4])", flush=True)

    target = processor = None
    device = f"cuda:{a.gpu}"
    width = 4096
    if a.stub:
        # The real width is read from the target config so the stub exercises the
        # same shapes SpecForge will assert on (aux == 3 * hidden).
        cfg = json.load(open(os.path.join(common.TARGET_PATH, "config.json")))
        width = int(cfg["text_config"]["hidden_size"])
        print(f"[stub] no model load; width {width} from target config", flush=True)
    else:
        target, processor = common.load_target(device=device)
        width = int(target.config.text_config.hidden_size)
        print(f"target loaded on {device}; hidden {width}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    started = time.time()
    written, skipped, report = 0, 0, []
    for index, row in enumerate(rows):
        man = by_id[row["id"]]
        group = index // a.group_size
        lo = group * a.group_size
        hi = min(lo + a.group_size, len(rows))
        sub = os.path.join(a.out, f"rows_{lo}-{hi}")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, f"data_{index}.ckpt")
        if a.resume and os.path.exists(path):
            skipped += 1
            continue
        if a.stub:
            tensors, info = make_stub(row, man, width, aux_layers)
        else:
            tensors, info = capture_one(target, processor, row, man,
                                        aux_layers, device)
        torch.save(tensors, path)
        written += 1
        report.append(info)
        if written % 20 == 0 or written == 1:
            el = time.time() - started
            print(f"  [{index + 1}/{len(rows)}] {info['id'][:18]:18s} "
                  f"L={info['render_len']:5d} sup={info['supervised']:5d} "
                  f"{el / max(written, 1):.1f}s/row", flush=True)

    meta = {"split": a.split, "rows": written, "skipped": skipped,
            "aux_layers": list(aux_layers), "hidden_size": width,
            "target": common.TARGET_PATH, "stub": bool(a.stub),
            "jsonl": jsonl, "manifest": a.manifest,
            "per_row": report}
    with open(os.path.join(a.out, "_capture_report.json"), "w") as handle:
        json.dump(meta, handle, indent=2)

    total = time.time() - started
    print(f"\nwrote {written} .ckpt (skipped {skipped}) -> {a.out} "
          f"in {total / 60:.1f} min", flush=True)
    if written:
        est = sum(r["render_len"] for r in report) * (width * 3 + width) * 2
        print(f"approx stored bytes: {est / 2**30:.2f} GiB", flush=True)
    print(f"report -> {os.path.join(a.out, '_capture_report.json')}", flush=True)


if __name__ == "__main__":
    main()
