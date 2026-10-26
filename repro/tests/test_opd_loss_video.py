#!/usr/bin/env python3
"""Gate for the video OPD loss: it must be the image recipe AND not a re-derivation.

Four checks on one real video row, before any training step is allowed to run:

  A. compat   -- with `reject_target="draft"` (k3 reverse KL), `alpha_rev=1`,
                 `drop_inexact_teacher=True`, the new function must reproduce
                 `opd_loss.opd_replay_loss(objective="bernoulli",
                 reject_mode="first")` to fp noise. This is the off-by-one
                 (`opd_loss.teacher_rows`) and the weight schedule
                 (`decay**(off-1)`) check: if the two disagree, the new code has
                 drifted and the run is void.
  B. no-op    -- with `reject_target="topk"` the reverse stream must be empty, so
                 `alpha_rev` 0.0 vs 1e3 must give the identical loss.
  C. active   -- the recipe run must supervise both streams (n_fwd > 0,
                 n_topk_rej > 0), otherwise "OPD" is silently the wrong loss.
  D. no-drop  -- `drop_inexact_teacher=True` must supervise no more positions than
                 the recipe's flag-off run, and the row must actually contain
                 positions whose proposed token is outside the stored top-k
                 (n_inexact > 0), or the flag being off is untested here.
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
import os
import sys

import torch

sys.path.insert(0, str(_REPRO_ROOT))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/GLANCE_repo")

from shared import common
from historical import opd_loss
from historical import opd_loss_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="/mnt/local/mcy/GLANCE/data/video_greedy_499.pt")
    ap.add_argument("--head", default="/mnt/local/mcy/GLANCE/checkpoints/head_video439/step1000")
    ap.add_argument("--seq", type=int, default=131072,
                    help="truncate the row to this many tokens. WARNING: a video "
                         "row cannot be cut inside its video-token block -- "
                         "get_rope_index then rejects the grid (measured: 'shape "
                         "mismatch: value tensor of shape [3, 350]'). The default "
                         "is therefore the whole row.")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"
    target, _ = common.load_target(device=dev)
    head = common.load_head(a.head, device=dev).eval()
    embed = target.get_input_embeddings().weight
    # The gate checks VALUES, never gradients, and the full-vocabulary logits of a
    # 1982-token video row (177 anchors x 15 offsets) are gigabytes: keeping three
    # autograd graphs alive at once OOMs a 48 GiB card. Measured, hence no_grad.
    torch.set_grad_enabled(False)

    blob = torch.load(a.records, weights_only=False)
    rec = max(blob["records"], key=lambda r: len(r["anchors"]))
    ids = rec["input_ids"][: a.seq].to(dev).long()
    import pandas as pd
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    pool = pd.read_parquet("/mnt/local/mcy/GLANCE/data/pool_499.parquet").set_index("video_name")
    p = pool.loc[rec["task"]]
    vtext = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": p["question"]}]}],
        add_generation_prompt=True, tokenize=False)
    ins = proc(text=[vtext], videos=[p["cache_path"]], return_tensors="pt", fps=float(p["fps"]))
    full_ids = rec["input_ids"][: a.seq].to(dev).long()
    ctx = common.target_context(target, full_ids[None],
                                pixel_values_videos=ins["pixel_values_videos"].to(dev),
                                video_grid_thw=ins["video_grid_thw"].to(dev))
    anchors = [x for x in rec["anchors"] if x["pos"] + head.block_size < a.seq]
    print(f"row {rec['task']}: S={full_ids.shape[0]}, anchors kept {len(anchors)}", flush=True)

    tk_ids, tk_val = rec["topk_ids"][: a.seq - 1].to(dev), rec["topk_val"][: a.seq - 1].to(dev)
    args = ([x["pos"] for x in anchors], [x["prop"] for x in anchors],
            [x["n_acc"] for x in anchors], tk_ids, tk_val, embed, head.block_size,
            common.MASK_ID, dev)

    L1 = opd_loss.opd_replay_loss(head, target, ctx, full_ids, *args,
                                  alpha_rev=1.0, beta_fwd=1.0, decay=0.9,
                                  objective="bernoulli", reject_mode="first",
                                  drop_inexact_teacher=True)
    L2 = opd_loss_video.opd_replay_loss_video(head, target, ctx, full_ids, *args,
                                              alpha_rev=1.0, beta_fwd=1.0, decay=0.9,
                                              reject_target="draft",
                                              drop_inexact_teacher=True)
    d = abs(L1[0].item() - L2[0].item())
    print(f"A compat:   ref loss {L1[0].item():.8f}  new loss {L2[0].item():.8f}  "
          f"|d| {d:.3e}  pos {L1[1]}/{L2[1]}  fwd {L1[2]:.6f}/{L2[2]:.6f}  "
          f"rev {L1[3]:.6f}/{L2[3]:.6f}  inexact {L1[4]}/{L2[4]}", flush=True)
    assert d < 1e-6, f"compat check failed: |d|={d:.3e}"
    assert (L1[1], L1[4]) == (L2[1], L2[4]), "position/inexact accounting drifted"

    R0 = opd_loss_video.opd_replay_loss_video(head, target, ctx, full_ids, *args,
                                              alpha_rev=0.0, beta_fwd=1.0, decay=0.9,
                                              reject_target="topk",
                                              drop_inexact_teacher=False,
                                              return_parts=True)
    R1 = opd_loss_video.opd_replay_loss_video(head, target, ctx, full_ids, *args,
                                              alpha_rev=1e3, beta_fwd=1.0, decay=0.9,
                                              reject_target="topk",
                                              drop_inexact_teacher=False,
                                              return_parts=True)
    print(f"B no-op:    alpha_rev 0 -> {R0[0].item():.8f}  1e3 -> {R1[0].item():.8f}  "
          f"|d| {abs(R0[0].item()-R1[0].item()):.3e}  n_k3_rej {R0[5]['n_k3_rej']}", flush=True)
    assert abs(R0[0].item() - R1[0].item()) < 1e-9, "alpha_rev is not a no-op"
    assert R0[5]["n_k3_rej"] == 0, "reverse stream is not empty under reject_target=topk"

    D1 = opd_loss_video.opd_replay_loss_video(head, target, ctx, full_ids, *args,
                                              alpha_rev=0.0, beta_fwd=1.0, decay=0.9,
                                              reject_target="topk",
                                              drop_inexact_teacher=True,
                                              return_parts=True)
    print(f"C active:   n_fwd {R0[5]['n_fwd']}  n_topk_rej {R0[5]['n_topk_rej']}  "
          f"l_fwd {R0[2]:.6f}  l_rev {R0[3]:.6f}", flush=True)
    assert R0[5]["n_fwd"] > 0 and R0[5]["n_topk_rej"] > 0, "a stream is empty"
    assert R0[3] == 0.0, "l_rev should be exactly 0 under the recipe"
    print(f"D no-drop:  positions kept {R0[1]} (flag off) vs {D1[1]} (flag on); "
          f"positions outside stored top-k on this row: {R0[4]}", flush=True)
    assert D1[1] <= R0[1], "dropping inexact teacher should not add positions"
    assert R0[4] > 0, "this row does not exercise the no-drop path; pick another row"

    print("ALL OPD-LOSS GATES PASSED", flush=True)


if __name__ == "__main__":
    main()
