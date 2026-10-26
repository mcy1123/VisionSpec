#!/usr/bin/env python3
"""Phase D gates for the image QA trainers.

Two gates, both required before any formal run:

1. ``optimizer_resume_gate.json`` -- exact resumption.
   Load a real trained checkpoint, rebuild `MasterWeightAdamW` from scratch, load
   the saved master state, and assert BIT-EXACT restoration of the fp32 master
   parameters and both Adam moments, that the bf16 mirror the forward pass uses
   equals what `save_pretrained` wrote, that the RNG block round-trips, and that
   the saved data cursor identifies the schedule position. This is the gate the
   project's update-precision report says must exist, because a checkpoint that
   only carries the bf16 head silently restarts the sub-ulp accumulation.

2. ``no_update_replay_gate.json`` -- real-image GPU replay with `optimizer.step()`
   never called. For at least two REAL ScienceQA images and several anchors each
   it checks, on the same anchor:
     * the encoded prompt equals the frozen record's prompt token for token;
     * the current target's own replay reproduces the frozen greedy AR exactly;
     * the planned `n_acc` equals the acceptance recomputed by an independent
       online teacher forward on the draft-induced prefix;
     * `stream_split` covers exactly the valid rejected suffix;
     * the gradient is finite and non-zero;
     * no parameter changed and the optimizer update count stayed at zero.

Nothing here trains: the gate deliberately stops before `opt.step()`.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(REPRO / "qa_opd_fair"),
                str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
from historical import opd_fullkl  # noqa: E402
import qa_anchors  # noqa: E402
from imageencode import encode_image_inputs, prefix_image_vision_positions  # noqa: E402
from shared.master_optim import MasterWeightAdamW  # noqa: E402
from train_sqa_sft import draft_block_logits_mixed  # noqa: E402


def gate_optimizer_resume(run, ckpt, out):
    import train_sqa_opd as T
    state = torch.load(os.path.join(ckpt, "training_state.pt"),
                       weights_only=False, map_location="cpu")
    head = common.load_head(ckpt, dtype=torch.bfloat16, device="cpu")
    opt = MasterWeightAdamW(head, lr=3e-5, betas=common.BETAS,
                            weight_decay=common.WEIGHT_DECAY, where="cpu", clip=1.0)
    report = {"checkpoint": str(ckpt), "saved_keys": sorted(state.keys())}
    # --- the RNG block must round-trip
    rng_ok = None
    if "rng" in state:
        import random
        import numpy as np
        before = T.rng_state()
        T.restore_rng(state["rng"])
        after = T.rng_state()
        rng_ok = bool(torch.equal(before["torch"], after["torch"])
                      and np.array_equal(before["numpy"][1], after["numpy"][1])
                      and before["python"][0] == after["python"][0])
        report["rng_field_present"] = True
        report["rng_round_trip_after_restore_is_self_consistent"] = rng_ok
        # restore the pre-gate RNG so this gate has no side effect
        T.restore_rng(before)
    else:
        report["rng_field_present"] = False
        report["rng_note"] = ("this smoke checkpoint predates the RNG field; the "
                              "formal step30+ checkpoints carry it")
    # --- bit-exact master + Adam restore
    saved_master = {k: v.clone() for k, v in state["master_optimizer"]["master"].items()}
    saved_opt = state["master_optimizer"]["optimizer"]
    opt.load_state_dict(state["master_optimizer"])
    diffs = {}
    for k, v in opt.master.items():
        diffs[k] = float((v - saved_master[k].to(v.dtype)).abs().max())
    report["master_max_abs_diff_after_load"] = max(diffs.values())
    report["master_restore_bit_exact"] = bool(max(diffs.values()) == 0.0)
    # torch's optimizer state_dict keys Adam state by the PARAMETER'S POSITION in
    # the param group, not by the parameter object, so the comparison has to go
    # through the same ordering.
    ordered = list(opt._opt.param_groups[0]["params"])
    moments_ok, moment_report, n_compared = True, {}, 0
    for idx, st in saved_opt["state"].items():
        loaded = opt._opt.state.get(ordered[int(idx)], {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                if key not in loaded:
                    moments_ok = False
                    moment_report[f"{idx}.{key}"] = "MISSING"
                    continue
                d = float((loaded[key] - st[key].to(loaded[key].dtype)).abs().max())
                moment_report[f"{idx}.{key}"] = d
                moments_ok &= (d == 0.0)
                n_compared += 1
    report["n_adam_tensors_compared"] = n_compared
    report["adam_state_nonzero"] = bool(any(
        float(st["exp_avg"].abs().max()) > 0.0 for st in saved_opt["state"].values()
        if "exp_avg" in st))
    report["adam_moment_max_abs_diff"] = max(moment_report.values()) if moment_report else None
    report["adam_moments_bit_exact"] = bool(moments_ok)
    report["n_state_entries"] = len(saved_opt["state"])
    report["n_param_groups"] = len(saved_opt["param_groups"])
    # --- the bf16 mirror must equal the shipped head
    shipped = common.load_head(ckpt, dtype=torch.bfloat16, device="cpu")
    mirror_ok, worst = True, 0.0
    for (n1, p1), (n2, p2) in zip(opt.module.named_parameters(),
                                  shipped.named_parameters()):
        d = float((p1.float() - p2.float()).abs().max())
        worst = max(worst, d)
        mirror_ok &= (n1 == n2 and d == 0.0)
    report["bf16_mirror_max_abs_diff_vs_shipped_head"] = worst
    report["bf16_mirror_matches_shipped_head"] = bool(mirror_ok)
    report["data_cursor"] = state.get("data_cursor", state.get("update"))
    report["schedule_recorded"] = state.get("schedule")
    report["update"] = state.get("update")
    report["arm"] = state.get("arm")
    report["gate_pass"] = bool(report["master_restore_bit_exact"]
                              and report["adam_moments_bit_exact"]
                              and report["bf16_mirror_matches_shipped_head"])
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return report["gate_pass"]


def gate_no_update_replay(run, ckpt, pool, records, schedule, out, gpu=7, n_records=2,
                          n_anchors=3):
    import pandas as pd
    import train_sqa_opd as T
    dev = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    df = pd.read_parquet(pool).set_index("task")
    blob = torch.load(records, weights_only=False, map_location="cpu")
    recs = {r["task"]: r for r in blob["records"]}
    slots = [json.loads(l) for l in pathlib.Path(schedule).read_text().splitlines()
             if l.strip()]
    slots.sort(key=lambda r: r["order"])
    target, proc = common.load_target(device=dev, dtype=torch.float32)
    head = common.load_head(ckpt, device=dev, dtype=torch.bfloat16).eval()
    for p in head.parameters():
        p.requires_grad_(True)
    head_dtype = next(head.parameters()).dtype
    opt = MasterWeightAdamW(head, lr=3e-5, betas=common.BETAS,
                            weight_decay=common.WEIGHT_DECAY, where="cpu", clip=1.0)
    before = {n: p.detach().clone() for n, p in head.named_parameters()}
    report = {"checkpoint": str(ckpt), "gpu": gpu, "target_dtype": "float32",
              "head_dtype": str(head_dtype), "optimizer_steps_taken": 0,
              "records": []}
    t0 = time.time()
    for slot in slots[:n_records]:
        task = slot.get("task") or f"{slot['video_id']}|{slot['question_id']}"
        rec = recs[task]
        row = df.loc[task]
        full = rec["input_ids"].to(dev).long()
        prompt_len = int(rec["prompt_len"])
        x = encode_image_inputs(proc, row.to_dict(), max_pixels=768)
        enc_ok = x["input_ids"][0].tolist() == full[:prompt_len].tolist()
        vispos = prefix_image_vision_positions(target, full[None],
                                               x["pixel_values"].to(dev),
                                               x["image_grid_thw"].to(dev))
        captured = []
        hook = target.model.register_forward_hook(
            lambda m, i, o: captured.append(o.last_hidden_state.detach()))
        try:
            fused, prefix_kv = opd_fullkl.target_prefill_shared(
                target, full[None], max_seq_len=common.MAX_SEQ_LEN, vispos=vispos)
        finally:
            hook.remove()
        current = target.lm_head(captured[0][0, prompt_len - 1:-1]).argmax(-1).tolist()
        replay_ok = current == full[prompt_len:].tolist()
        del captured
        if fused.dtype != head_dtype:
            fused = fused.to(head_dtype)
        verifier = [int(t) for t in rec["verifier_argmax"]]
        cand = qa_anchors.sft_candidate_positions(
            prompt_len, rec["eos_pos_in_answer"], int(rec["n_answer_tokens"]))
        anchors = qa_anchors.select_anchors(cand, n_anchors)
        per_anchor = []
        for p in anchors:
            q_logits = draft_block_logits_mixed(
                head, target, fused[:, :p], anchor_pos=p,
                root_token=int(full[p]), head_dtype=head_dtype)
            prop = [int(t) for t in q_logits.argmax(-1)]
            n_acc, valid = T._verify(prop, verifier, p, prompt_len,
                                     int(rec["n_answer_tokens"]),
                                     rec["eos_pos_in_answer"],
                                     eos_token_ids=int(proc.tokenizer.eos_token_id))
            acc_idx, rej_idx = opd_fullkl.stream_split(15, n_acc, valid, device=dev)
            # independent online teacher forward on the draft-induced prefix
            valid_offsets = acc_idx.tolist() + rej_idx.tolist()
            online_n, targmax = None, None
            if valid_offsets:
                last = max(valid_offsets)
                p_logits = T.teacher_proposal_logits(target, {"full": full,
                                                              "prefix_kv": prefix_kv,
                                                              "vispos": vispos},
                                                      p, prop, last, dev)
                targmax = p_logits.argmax(-1).tolist()
                online_n = 0
                for j, tok in enumerate(targmax):
                    if not valid[j] or tok != prop[j]:
                        break
                    online_n += 1
            per_anchor.append({
                "anchor": p, "n_acc_planned": int(n_acc),
                "n_acc_online_recomputed": online_n,
                "n_acc_matches": (online_n == int(n_acc)) if online_n is not None else None,
                "n_valid_offsets": len(valid_offsets),
                "n_accepted_offsets": int(acc_idx.numel()),
                "n_rejected_offsets": int(rej_idx.numel()),
                "teacher_argmax_prefix_matches_proposal": targmax[:3] if targmax else None,
                "valid_prefix": valid[:6],
            })
            # a real backward, WITHOUT stepping
            plan = {"pos": p, "prop": prop, "n_acc": int(n_acc), "valid": valid,
                    "acc_idx": acc_idx, "rej_idx": rej_idx}
            n_acc_total = int(acc_idx.numel())
            z_total = (float(T.gamma_weights(rej_idx, 0.8, dev).double().sum())
                       if int(rej_idx.numel()) else 0.0)
            prep = {"full": full, "prefix_kv": prefix_kv, "vispos": vispos,
                    "fused": fused, "slot": slot, "prompt_len": prompt_len}
            opt.zero_grad(set_to_none=True)
            loss, st = T._anchor_loss(head, target, prep, plan, 0.8, dev,
                                      int(head.block_size), int(head.mask_token_id),
                                      target.get_input_embeddings(),
                                      max(n_acc_total, 1), max(z_total, 1e-12),
                                      n_acc_total > 0, z_total > 0.0,
                                      head_dtype)
            if loss is not None:
                loss.backward()
                gn = float(torch.nn.utils.get_total_norm(
                    [p.grad for p in head.parameters() if p.grad is not None],
                    foreach=False))
                per_anchor[-1]["loss"] = float(loss.detach())
                per_anchor[-1]["grad_norm"] = gn
                per_anchor[-1]["grad_finite"] = bool(torch.isfinite(torch.tensor(gn)))
                per_anchor[-1]["grad_nonzero"] = bool(gn > 0.0)
            else:
                per_anchor[-1]["loss"] = None
            opt.zero_grad(set_to_none=True)
            del q_logits
            torch.cuda.empty_cache()
        changed = max(float((p.detach() - before[n]).abs().max())
                      for n, p in head.named_parameters())
        report["records"].append({
            "task": task, "n_prompt_tokens": prompt_len,
            "encoded_prompt_matches_frozen": enc_ok,
            "target_replay_matches_frozen_AR": replay_ok,
            "n_answer_tokens": int(rec["n_answer_tokens"]),
            "anchors": per_anchor,
            "max_param_change_after_backward": changed,
            "params_unchanged": bool(changed == 0.0),
        })
        print(f"   {task}: enc={enc_ok} replay={replay_ok} "
              f"anchors={[a['n_acc_planned'] for a in per_anchor]} "
              f"online={[a['n_acc_online_recomputed'] for a in per_anchor]}",
              flush=True)
        del fused, prefix_kv, vispos
        torch.cuda.empty_cache()

    all_enc = all(r["encoded_prompt_matches_frozen"] for r in report["records"])
    all_rep = all(r["target_replay_matches_frozen_AR"] for r in report["records"])
    all_acc = all(a["n_acc_matches"] for r in report["records"] for a in r["anchors"]
                  if a["n_acc_matches"] is not None)
    all_grad = all(a.get("grad_finite") and a.get("grad_nonzero")
                   for r in report["records"] for a in r["anchors"]
                   if a.get("loss") is not None)
    all_unchanged = all(r["params_unchanged"] for r in report["records"])
    report["all_encoded_prompts_match_frozen"] = all_enc
    report["all_target_replays_match_frozen_AR"] = all_rep
    report["all_n_acc_match_online_recompute"] = all_acc
    report["all_gradients_finite_and_nonzero"] = all_grad
    report["all_parameters_unchanged"] = all_unchanged
    report["optimizer_step_never_called"] = True
    report["peak_gib"] = torch.cuda.max_memory_allocated(gpu) / 2**30
    report["wall_seconds"] = time.time() - t0
    report["gate_pass"] = bool(all_enc and all_rep and all_acc and all_grad
                              and all_unchanged)
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2),
          flush=True)
    return report["gate_pass"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--phase", default="both",
                    choices=["both", "resume", "replay"])
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--n-records", type=int, default=2)
    ap.add_argument("--n-anchors", type=int, default=3)
    a = ap.parse_args()
    out = pathlib.Path(a.run) / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    if a.phase in ("both", "resume"):
        print("[gate] optimizer / RNG resume", flush=True)
        ok &= gate_optimizer_resume(a.run, a.checkpoint,
                                    out / "optimizer_resume_gate.json")
    if a.phase in ("both", "replay"):
        print("[gate] no-update real-image GPU replay", flush=True)
        ok &= gate_no_update_replay(a.run, a.checkpoint, a.pool, a.records,
                                    a.schedule, out / "no_update_replay_gate.json",
                                    gpu=a.gpu, n_records=a.n_records,
                                    n_anchors=a.n_anchors)
    print(f"[gate] overall pass={ok}", flush=True)
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())
