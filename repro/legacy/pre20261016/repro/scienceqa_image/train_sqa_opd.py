#!/usr/bin/env python3
"""SQA fixed-anchor OPD with explicit position contracts.

GENERATED FROM `repro/qa_opd_fair/train_qa_opd.py` by
`repro/scienceqa_image/make_image_trainers.py`. The only edited lines are the
image-input adaptation; the accepted one-hot CE, the rejected full-vocabulary
reverse KL with 0.8**(k-1), the update-global two-stream denominators, the
anchor rule, the `n_acc`/teacher-shift/suffix-mask contract and the optimizer are
byte-for-byte the frozen ones. The original file's docstring follows.

QA fixed-anchor OPD with explicit position contracts.

Current proposals are scored at shared fixed anchors on a verified target AR
trajectory; this is not a policy-visited online rollout. Accepted loss remains
one-hot CE (not soft forward KL), rejected loss remains full-vocabulary reverse
KL with gamma**(k-1), and update-global two-stream normalization is unchanged.
Offset k predicts absolute position p+k from full[:p+1] + proposal[:k-1].
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(REPRO / "qa_opd_fair"),
                str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

import common  # noqa: E402
import opd_fullkl  # noqa: E402
import qa_anchors  # noqa: E402
from master_optim import MasterWeightAdamW  # noqa: E402
from train_sqa_sft import LmHeadShim, draft_block_logits_mixed  # noqa: E402
from imageencode import encode_image_inputs, prefix_image_vision_positions  # noqa: E402

K = common.BLOCK_SIZE - 1


def gamma_weights(offsets, gamma, device):
    idx = offsets.detach().to(device=device)
    w = torch.pow(torch.as_tensor(float(gamma), dtype=torch.float64, device=device),
                  idx.double() - 1.0)
    return w.float()


def rng_state():
    """torch/numpy/python RNG state, so a checkpoint can resume exactly."""
    import random
    import numpy as _np
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else []),
        "numpy": _np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state):
    """Counterpart of `rng_state`; used by the resume gate."""
    import random
    import numpy as _np
    if not state:
        return
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    _np.random.set_state(state["numpy"])
    random.setstate(state["python"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schedule", default=None,
                    help="override the frozen schedule path; default reads "
                         "train_schedule_seed_<seed>.jsonl from --run")
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--qa-per-update", type=int, default=4)
    ap.add_argument("--anchors-per-qa", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--clip", type=float, default=common.GRAD_CLIP)
    ap.add_argument("--master-weights", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--target-dtype", default="float32",
                    choices=["float32", "bfloat16"],
                    help="the FROZEN target's dtype; the formal runs use float32, "
                         "bfloat16 exists only for a mechanics pilot on a busy card")
    ap.add_argument("--time-mode", default="legacy", choices=["legacy", "real"],
                    help="IGNORED for still images; kept so this CLI matches the "
                         "frozen video script. An image has no timeline.")
    ap.add_argument("--max-pixels", type=int, default=768,
                    help="longest-side cap; 768 is common.IMAGE_LONG_SIDE, used "
                         "for TRAINING AND EVALUATION here so the head sees the "
                         "same pixels in both")
    ap.add_argument("--save-at", default=None,
                    help="comma list of update numbers to checkpoint, replacing "
                         "the every-N rule; the final update is always saved")
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=common.MAX_SEQ_LEN)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--resume-update", type=int, default=0)
    ap.add_argument("--stop-after", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)
    os.makedirs(a.out, exist_ok=True)

    pool = pd.read_parquet(a.pool).set_index("task")
    blob = torch.load(a.records, weights_only=False, map_location="cpu")
    recs = {r["task"]: r for r in blob["records"]}

    from transformers import AutoProcessor, DynamicCache
    target, proc = common.load_target(
        device=dev,
        dtype=torch.float32 if a.target_dtype == "float32" else torch.bfloat16)
    head = common.load_head(a.init_from, device=dev, dtype=torch.bfloat16)
    head.eval()                      # the rollout and the loss use one policy
    for p in head.parameters():
        p.requires_grad_(True)
    head_dtype = next(head.parameters()).dtype
    embed = target.get_input_embeddings()
    mask_id = int(head.mask_token_id)
    block = int(head.block_size)

    opt = MasterWeightAdamW(head, lr=a.lr, betas=common.BETAS,
                            weight_decay=common.WEIGHT_DECAY,
                            where=a.master_weights, clip=a.clip)
    start_update = 0
    if a.resume_from:
        state = torch.load(os.path.join(a.resume_from, "training_state.pt"),
                           weights_only=False, map_location="cpu")
        head.load_state_dict(state["head"])
        opt.load_state_dict(state["master_optimizer"])
        start_update = int(a.resume_update)
        print(f"[resume] from {a.resume_from} at update {start_update}", flush=True)

    if not a.resume_from:
        head.save_pretrained(os.path.join(a.out, "step0"))
        torch.save({"head": head.state_dict(), "master_optimizer": opt.state_dict(),
                    "update": 0, "seed": a.seed, "arm": "SQA-OPD",
                    "rng": rng_state(), "data_cursor": 0,
                    "schedule": (a.schedule or str(
                        pathlib.Path(a.run) / "dataset" /
                        f"train_schedule_seed_{a.seed}.jsonl"))},
                   os.path.join(a.out, "step0", "training_state.pt"))
        print("[ckpt] step0 saved (the unmodified shared init-from weights)", flush=True)

    sched_path = (pathlib.Path(a.schedule) if a.schedule else
                  pathlib.Path(a.run) / "dataset" /
                  f"train_schedule_seed_{a.seed}.jsonl")
    order = [json.loads(l) for l in sched_path.read_text().splitlines()
             if l.strip()]
    order.sort(key=lambda r: r["order"])
    slots = {int(r["update"]): [] for r in order}
    for r in order:
        slots[int(r["update"])].append(r)
    for k in slots:
        slots[k].sort(key=lambda r: r["slot"])

    config = {
        "arm": "SQA-OPD", "entry_point": "train_qa_opd.py",
        "run": a.run, "seed": a.seed, "pool": a.pool, "records": a.records,
        "init_from": a.init_from, "out": a.out, "updates": a.updates,
        "qa_per_update": a.qa_per_update, "anchors_per_qa": a.anchors_per_qa,
        "lr": a.lr, "gamma": a.gamma, "master_weights": a.master_weights,
        "objective": "accepted one-hot CE, rejected full-vocabulary reverse KL "
                     "with 0.8**(k-1) weights, update-global denominators, one "
                     "stream dropped and renormalised when empty",
        "online_semantics": "fixed anchors; base trajectory = replay-verified target greedy AR on the "
                            "exact prompt; proposal = current head from a fresh "
                            "cache at each anchor; acceptance = current target "
                            "argmax match, recomputed, never read from a flag",
        "head_dtype": str(head_dtype), "target_dtype": a.target_dtype,
        "pilot_only": a.target_dtype != "float32",
        "time_mode": "n/a-still-image", "eos_stop": "immediate",
        "max_pixels": a.max_pixels, "modality": "image",
        "betas": list(common.BETAS), "weight_decay": common.WEIGHT_DECAY,
        "clip": a.clip, "torch": torch.__version__,
    }
    (pathlib.Path(a.out) / "config.json").write_text(json.dumps(config, indent=2))
    log = open(pathlib.Path(a.out) / "log.jsonl", "a")

    t0 = time.time()
    for upd in range(start_update, a.updates):
        torch.cuda.reset_peak_memory_stats(a.gpu)
        t_upd = time.time()
        # ---- pass 1: plan the update's anchors (no grad)
        plans = []
        prepared = []
        n_teacher_forwards = 0
        with torch.no_grad():
            for slot in slots[upd]:
                task = slot.get("task") or f"{slot['video_id']}|{slot['question_id']}"
                rec = recs[task]
                row = pool.loc[task]
                full = rec["input_ids"].to(dev).long()
                prompt_len = int(rec["prompt_len"])
                n_ans = int(rec["n_answer_tokens"])
                eos_pos = rec["eos_pos_in_answer"]
                if int(full.shape[0]) > a.max_seq_len:
                    raise SystemExit(f"{task}: trajectory too long")
                root = int(full[prompt_len])
                x = encode_image_inputs(proc, row.to_dict(), max_pixels=a.max_pixels)
                if x["input_ids"][0].tolist() != full[:prompt_len].tolist():
                    raise RuntimeError(f"{task}: encoded prompt differs from frozen record")
                vispos = prefix_image_vision_positions(
                    target, full[None],
                    x["pixel_values"].to(dev), x["image_grid_thw"].to(dev))
                captured = []
                hook = target.model.register_forward_hook(
                    lambda module, inputs, output: captured.append(output.last_hidden_state.detach()))
                try:
                    fused_all, prefix_kv = opd_fullkl.target_prefill_shared(
                        target, full[None],
                        max_seq_len=a.max_seq_len, vispos=vispos)
                finally:
                    hook.remove()
                current = target.lm_head(captured[0][0, prompt_len-1:-1]).argmax(-1).tolist()
                if current != full[prompt_len:].tolist():
                    raise RuntimeError(f"{task}: current target replay differs from frozen AR; rebuild records")
                del captured, current
                # The target is fp32 and the head is bf16: the context rows are a
                # head INPUT, so they are cast once here instead of once per
                # anchor. `opd_fullkl.draft_block_logits` casts the noise
                # embeddings and the hidden rows back to the target's lm_head
                # dtype itself.
                if fused_all.dtype != head_dtype:
                    fused_all = fused_all.to(head_dtype)
                raw_verifier = [int(t) for t in rec["verifier_argmax"]]
                if len(raw_verifier) != n_ans or raw_verifier != full[prompt_len:].tolist():
                    raise RuntimeError(f"{task}: invalid target AR record")
                verifier = raw_verifier  # answer index i predicts full[prompt_len+i]
                # Shared fixed anchors, NOT the rounds visited by an online policy.
                cand = qa_anchors.sft_candidate_positions(prompt_len, eos_pos, n_ans)
                anchors = []
                for p in qa_anchors.select_anchors(cand, a.anchors_per_qa):
                    if p + 1 >= int(full.shape[0]):
                        continue
                    q_logits = draft_block_logits_mixed(
                        head, target, fused_all[:, :p], anchor_pos=p,
                        root_token=int(full[p]), head_dtype=head_dtype)
                    prop = [int(t) for t in q_logits.argmax(-1)]
                    n_acc, valid = _verify(prop, verifier, p, prompt_len, n_ans,
                                          eos_pos, eos_token_ids=int(proc.tokenizer.eos_token_id))
                    anchors.append({"pos": p, "prop": prop, "n_acc": n_acc,
                                    "valid": valid})
                    del q_logits
                if not anchors:
                    del fused_all, prefix_kv, verifier
                    continue
                selected = set(qa_anchors.select_anchors(
                    [z["pos"] for z in anchors], a.anchors_per_qa))
                kept = [z for z in anchors if z["pos"] in selected]
                for z in kept:
                    acc_idx, rej_idx = opd_fullkl.stream_split(
                        K, z["n_acc"], z["valid"], device=dev)
                    if int(acc_idx.numel()) + int(rej_idx.numel()) == 0:
                        continue
                    plans.append({"task": task, "pos": z["pos"], "prop": z["prop"],
                                  "n_acc": z["n_acc"], "valid": z["valid"],
                                  "acc_idx": acc_idx, "rej_idx": rej_idx})
                prepared.append({"task": task, "full": full,
                                 "prompt_len": prompt_len, "n_ans": n_ans,
                                 "eos_pos": eos_pos, "vispos": vispos,
                                 "fused": fused_all, "prefix_kv": prefix_kv,
                                 "slot": slot, "pixel": x["pixel_values"],
                                 "grid": x["image_grid_thw"]})
                del x
        if not plans:
            raise SystemExit(f"update {upd + 1}: no usable anchors")
        n_acc_total = int(sum(int(p["acc_idx"].numel()) for p in plans))
        z_total = float(sum(float(gamma_weights(p["rej_idx"], a.gamma, dev).double().sum())
                            if int(p["rej_idx"].numel()) else 0.0 for p in plans))
        acc_present = n_acc_total > 0
        rej_present = z_total > 0.0

        # ---- pass 2: one microbatch per anchor, backward immediately
        torch.cuda.reset_peak_memory_stats(a.gpu)
        by_task = {}
        for prep in prepared:
            by_task[prep["task"]] = prep
        per_anchor_log = []
        loss_sum = 0.0
        n_backward = 0
        for plan in plans:
            prep = by_task[plan["task"]]
            p = int(plan["pos"])
            loss_mb, st = _anchor_loss(
                head, target, prep, plan, a.gamma, dev, block, mask_id, embed,
                n_acc_total, z_total, acc_present, rej_present, head_dtype)
            if loss_mb is None:
                continue
            loss_mb.backward()
            n_backward += 1
            loss_sum += float(loss_mb.detach())
            per_anchor_log.append(st)
            n_teacher_forwards += st["teacher_forwards"]
            del loss_mb
        if n_backward == 0:
            raise SystemExit(f"update {upd + 1}: every anchor contributed zero")
        opt.sync_grads()
        grad_norm = float(opt.grad_norm())
        applied = min(1.0, a.clip / (grad_norm + 1e-12)) if a.clip else 1.0
        if not np.isfinite(grad_norm):
            raise RuntimeError(f"update {upd + 1}: non-finite grad norm")
        opt.step()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated(a.gpu) / 2**30
        rec_log = {
            "arm": "QA-OPD", "seed": a.seed, "update": upd + 1, "lr": a.lr,
            "grad_norm": grad_norm, "clip_scale": applied,
            "loss_sum_microbatches": loss_sum,
            "N_acc_total": n_acc_total, "Z_total": z_total,
            "acc_present": acc_present, "rej_present": rej_present,
            "anchors": len(plans), "microbatches": n_backward,
            "teacher_forwards": n_teacher_forwards,
            "context_forwards": len(prepared),
            "context_tokens": sum(int(z["full"].numel()) for z in prepared),
            "teacher_input_tokens": sum(z.get("teacher_input_tokens", 0) for z in per_anchor_log),
            "supervised_positions": sum(z["n_acc_sup"]+z["n_rej_sup"] for z in per_anchor_log),
            "video_ids": sorted({pl["task"].split("|")[0] for pl in plans}),
            "per_anchor": per_anchor_log,
            "seconds": time.time() - t_upd, "peak_gib": peak,
            "elapsed_min": (time.time() - t0) / 60,
        }
        log.write(json.dumps(rec_log, ensure_ascii=False) + "\n")
        log.flush()
        print(f"[QA-OPD s{a.seed}] upd {upd+1}/{a.updates} sumloss {loss_sum:.4f} "
              f"anchors {len(plans)} Nacc {n_acc_total} Z {z_total:.2f} "
              f"tfwd {n_teacher_forwards} gnorm {grad_norm:.3f} "
              f"peak {peak:.1f}GiB {rec_log['seconds']:.1f}s", flush=True)
        # `--save-at` names the frozen checkpoint grid exactly. The plain
        # every-N rule would write step 90 as well, which is 16 GiB of disk the
        # amended grid does not budget for and does not need: 30/60/120 is the
        # registered grid, and the final update is always written.
        _save_at = ({int(x) for x in str(a.save_at).split(",") if x.strip()}
                    if a.save_at else None)
        _save_now = ((upd + 1) in _save_at if _save_at is not None
                     else bool(a.save_every and (upd + 1) % a.save_every == 0))
        _save_now = _save_now or (upd + 1) == a.updates
        if _save_now:
            ck = pathlib.Path(a.out) / f"step{upd + 1}"
            head.save_pretrained(ck)
            torch.save({"head": head.state_dict(),
                        "master_optimizer": opt.state_dict(),
                        "update": upd + 1, "seed": a.seed, "arm": "SQA-OPD",
                        "rng": rng_state(), "data_cursor": upd + 1,
                        "schedule": str(sched_path)},
                       ck / "training_state.pt")
            print(f"[ckpt] {ck}", flush=True)
        del prepared, by_task, plans
        torch.cuda.empty_cache()
        if a.stop_after and (upd + 1 - start_update) >= a.stop_after:
            print(f"[debug] stop-after reached at update {upd + 1}", flush=True)
            break
    print(f"[done] QA-OPD seed {a.seed} -> {a.out}", flush=True)


def _verifier_argmax(target, full, vispos, prefix_kv, prompt_len, n_ans, max_seq_len):
    """The frozen target's own next-token argmax over the answer positions.

    `prefix_kv` carries the prefill's per-position last hidden state under
    `opd_fullkl.PREFIX_LAST_HIDDEN` only for the LAST position, so this runs one
    extra cached forward over the whole trajectory with hidden states exposed.
    It is the only way to read the target's argmax at *every* answer position,
    and it is what acceptance is compared against.
    """
    hidden = _all_last_hidden(target, full, vispos)
    # Row `prompt_len - 1 + i` predicts the token at `prompt_len + i`; index 0 of
    # the returned list is therefore "the target's argmax predicted BY the last
    # prompt token", i.e. the root, and index i is the argmax for answer token i.
    rows = hidden[:, prompt_len - 1: prompt_len + n_ans - 1, :]
    return target.lm_head(rows)[0].argmax(-1).tolist()


def _all_last_hidden(target, full, vispos):
    from transformers import DynamicCache
    out = target.model(full[None], past_key_values=DynamicCache(),
                       use_cache=True, **vispos.prefill_inputs(full[None], None))
    return out.last_hidden_state


def _verify(prop, verifier, p, prompt_len, n_ans, eos_pos, *, eos_token_ids=None):
    """verifier[i] predicts answer i, i.e. absolute position prompt_len+i.

    Acceptance stops at the first mismatch. Validity does NOT stop at rejection:
    the entire rejected suffix is supervised, bounded by the answer horizon,
    reference EOS, and (when supplied) the first proposal EOS, inclusive.
    """
    if len(verifier) != n_ans:
        raise ValueError("verifier must have exactly one row per answer token")
    eos = set([] if eos_token_ids is None else
              [eos_token_ids] if isinstance(eos_token_ids, int) else eos_token_ids)
    valid, n_acc, accepting, ended = [], 0, True, False
    for k, token in enumerate(prop, 1):
        idx = p + k - prompt_len
        can = not ended and 0 <= idx < n_ans and (eos_pos is None or idx <= eos_pos)
        valid.append(bool(can))
        if accepting and can and token == verifier[idx]:
            n_acc += 1
        else:
            accepting = False
        if can and token in eos:
            ended = True
    return n_acc, valid


def teacher_proposal_logits(target, prep, p, prop, last, dev):
    """Row k-1 predicts p+k; consume root then proposal[:last-1]."""
    tokens = [int(prep["full"][p])] + list(prop[:last-1])
    positions = torch.arange(p, p+last, device=dev)[None]
    with torch.no_grad():
        out = target.model(torch.tensor([tokens], device=dev),
            past_key_values=_cache_from_prefix(prep["prefix_kv"], p),
            use_cache=True, position_ids=prep["vispos"](positions),
            cache_position=positions[0])
        logits = target.lm_head(out.last_hidden_state[0].to(target.lm_head.weight.dtype)).float()
    if logits.shape[0] != last:
        raise RuntimeError("teacher row count differs from prediction horizon")
    return logits


def _anchor_loss(head, target, prep, plan, gamma, dev, block, mask_id, embed,
                 n_acc_total, z_total, acc_present, rej_present, head_dtype):
    """One anchor's scaled contribution and its log record."""
    p = int(plan["pos"])
    prop = [int(t) for t in plan["prop"]]
    acc_idx, rej_idx = plan["acc_idx"], plan["rej_idx"]
    full, prompt_len = prep["full"], prep["prompt_len"]
    fused = prep["fused"]
    q_logits = draft_block_logits_mixed(
        head, target, fused[:, :p], anchor_pos=p,
        root_token=int(full[p]), head_dtype=head_dtype)
    q_ids = [int(t) for t in q_logits.argmax(-1)]
    if q_ids != prop:
        raise RuntimeError(f"anchor {p}: the gradient pass's proposal differs "
                           f"from the planning pass's; refusing to score an "
                           f"anchor whose labels moved between the two passes")
    terms = []
    stats = {"pos": p, "video": prep["slot"]["video_id"], "n_acc": int(plan["n_acc"]),
             "n_acc_sup": int(acc_idx.numel()), "n_rej_sup": int(rej_idx.numel()),
             "teacher_forwards": 0, "acc_kl_sum": 0.0, "rej_kl_sum": 0.0}
    if acc_present and int(acc_idx.numel()):
        # `KL(onehot(t) || q)` is exactly `-log q[t]`; no (V,) one-hot is built.
        tgt_tok = torch.tensor([int(full[p + int(k)]) for k in acc_idx.tolist()],
                               device=dev)
        logq = torch.log_softmax(q_logits[acc_idx - 1].float(), dim=-1)
        acc_kl = -logq.gather(1, tgt_tok[:, None])[:, 0]
        terms.append(acc_kl.sum() / float(n_acc_total))
        stats["acc_kl_sum"] = float(acc_kl.detach().sum())
    valid_offsets = acc_idx.tolist() + rej_idx.tolist()
    if valid_offsets:
        last = max(valid_offsets)
        p_logits_full = teacher_proposal_logits(target, prep, p, prop, last, dev)
        online_n = 0
        for j, token in enumerate(p_logits_full.argmax(-1).tolist()):
            if not plan["valid"][j] or token != prop[j]:
                break
            online_n += 1
        if online_n != int(plan["n_acc"]):
            raise RuntimeError(f"anchor {p}: online target acceptance {online_n} != planned {plan['n_acc']}")
        stats.update(teacher_forwards=1, teacher_input_tokens=last,
                     teacher_argmax=p_logits_full.argmax(-1).tolist(),
                     teacher_prediction_positions=list(range(p+1,p+last+1)),
                     proposal=prop, valid=list(plan["valid"]), online_n_acc=online_n)
    if rej_present and int(rej_idx.numel()):
        n_rej_sup = int(rej_idx.numel())
        stats["teacher_forwards"] = 1
        rel = (rej_idx - 1).tolist()
        p_rej = p_logits_full[rel]
        q_rej = q_logits[rej_idx - 1]
        rej_kl = opd_fullkl.full_kl_reverse(q_rej, p_rej)
        w = gamma_weights(rej_idx, gamma, dev)
        terms.append((w * rej_kl).sum() / z_total)
        stats["rej_kl_sum"] = float(rej_kl.detach().sum())
        stats["teacher_logit_gap"] = float(
            (p_logits_full[rel].argmax(-1) == torch.tensor(
                [prop[int(k) - 1] for k in rej_idx.tolist()], device=dev)
             ).float().mean()) if n_rej_sup else 0.0
    if not terms:
        return None, stats
    loss = sum(terms) / (2.0 if (acc_present and rej_present) else 1.0)
    return loss, stats


def _cache_from_prefix(prefix_kv, n):
    """A `DynamicCache` holding the first `n` rows of a shared prefix cache."""
    from transformers import DynamicCache
    data = []
    for layer in prefix_kv.layers:
        data.append((layer.keys[..., :n, :], layer.values[..., :n, :]))
    return DynamicCache(ddp_cache_data=data)


if __name__ == "__main__":
    main()
