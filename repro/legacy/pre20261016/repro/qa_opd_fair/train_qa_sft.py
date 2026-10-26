#!/usr/bin/env python3
"""QA-SFT: the fair control arm on the frozen QA schedule.

WHAT IT IS. The project's existing continue-SFT objective (`train_head.py`'s
variable-reveal block loss) applied to the frozen MSVD-QA schedule: for each of
the 4 scheduled QAs of an update, the target's own greedy AR continuation is the
label sequence, `prefill_len - 1 + answer_index` candidate anchors are enumerated,
`select_anchors` keeps at most 6, each anchor draws a reveal length from the
historical truncated-geometric prior, and the loss is the `0.8**(k - reveal)`
weighted cross-entropy over the supervised offsets.

WHAT IS SHARED WITH QA-OPD, deliberately and exactly:
  * the same frozen schedule slot file for that seed (same 4 QA per update,
    same order), so both arms see the same videos and the same questions;
  * the same init-from weights, reloaded independently;
  * the same target (fp32), the same cached/aligned context path, the same
    timestamp mode, the same `eos_stop="immediate"` deployment semantics;
  * the same bf16 head + fp32 CPU master-weight AdamW, lr 3e-5, grad clip 1.0,
    constant schedule, 120 updates;
  * the same checkpoint updates (0/30/60/90/120).

WHAT DIFFERS, and must be reported rather than hidden: the supervised positions
are this arm's reveal-sampled offsets, while QA-OPD's are its accepted/rejected
offsets, and QA-OPD additionally pays for teacher forwards. Teaching budget is
logged per update for both arms so the difference is auditable.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

import common  # noqa: E402
import qa_anchors  # noqa: E402
import train_head as th  # noqa: E402
from master_optim import MasterWeightAdamW  # noqa: E402
from videoencode import build_video_text, encode_inputs  # noqa: E402


class LmHeadShim:
    """`head.compute_logits` wants `target.lm_head`; cast the hidden rows first."""

    def __init__(self, target, head_dtype):
        self._w = target.lm_head
        self._w_dtype = target.lm_head.weight.dtype
        self._head_dtype = head_dtype

    def lm_head(self, hidden):
        if hidden.dtype != self._w_dtype:
            hidden = hidden.to(self._w_dtype)
        return self._w(hidden)


def qa_blocks(full_ids, anchors, reveals, block, mask_id, valid_end, device):
    """`train_head.build_blocks` plus the EOS/trajectory guard.

    Builds label/target matrices with `label_start=0` (offset `k` labels the token
    at `anchor + k`) and then removes every supervised position whose label lies
    past the first EOS, i.e. every token after the answer ended. Without
    this the block would supervise tokens the model would never have produced,
    which is exactly the label-noise failure the EOS audit warned about.
    """
    targets, super_mask, weights = th.build_blocks(
        full_ids, anchors, reveals, block, mask_id, device)[1:]
    n = len(anchors)
    offsets = torch.arange(block, device=device)
    label_pos = (torch.tensor(anchors, device=device).unsqueeze(1)
                 + offsets.unsqueeze(0))
    eos_ok = (label_pos <= int(valid_end))
    super_mask = super_mask * eos_ok.to(torch.float32)
    weights = weights * eos_ok.to(torch.float32)
    blocks = th.build_blocks(full_ids, anchors, reveals, block, mask_id, device)[0]
    return blocks, targets, super_mask, weights


def draft_block_logits_mixed(head, target, fused, *, anchor_pos, root_token,
                             head_dtype, block_size=None):
    """`opd_fullkl.draft_block_logits` for the mixed-precision configuration.

    `opd_fullkl.draft_block_logits` builds its noise embeddings from the TARGET's
    embedding table and hands them to the HEAD's Linears, so it only works when
    the target and the head share a dtype. The formal QA runs are
    `fp32 target + bf16 head`, so the embedding rows and the fused context are
    cast into the head's dtype here, while the target's `lm_head` receives rows
    cast back to ITS dtype (the KL is computed in fp32 downstream either way).
    This is the same contract `opd_train_video_fullkl.draft_block_logits`
    documents for the historical C2 arm.
    """
    import opd_fullkl

    fused = fused if fused.dtype == head_dtype else fused.to(head_dtype)
    embed = target.get_input_embeddings()

    class _HeadDtypeTarget:
        def __init__(self, real, embedding):
            self._lm = real.lm_head
            self._lm_dtype = real.lm_head.weight.dtype
            self._embedding = embedding

        def lm_head(self, hidden):
            if hidden.dtype != self._lm_dtype:
                hidden = hidden.to(self._lm_dtype)
            return self._lm(hidden)

        def get_input_embeddings(self):
            return self._embedding

    def _embed(ids):
        out = embed(ids)
        return out if out.dtype == head_dtype else out.to(head_dtype)

    return opd_fullkl.draft_block_logits(
        head, _HeadDtypeTarget(target, _embed), fused, anchor_pos=anchor_pos,
        root_token=int(root_token), block_size=block_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--pool", required=True, help="qa_pool_df<seed>.parquet")
    ap.add_argument("--records", required=True, help="target_ar_df<seed>.pt")
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--qa-per-update", type=int, default=4)
    ap.add_argument("--anchors-per-qa", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--clip", type=float, default=common.GRAD_CLIP)
    ap.add_argument("--master-weights", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--target-dtype", default="float32",
                    choices=["float32", "bfloat16"],
                    help="the FROZEN target's dtype. The formal runs use float32; "
                         "bfloat16 exists only so a small mechanics pilot can fit "
                         "on a busier card, and any pilot run with it is marked "
                         "pilot_only in its config")
    ap.add_argument("--time-mode", default="legacy", choices=["legacy", "real"])
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=common.MAX_SEQ_LEN)
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint dir to resume the module+master state from")
    ap.add_argument("--resume-update", type=int, default=0,
                    help="optimizer update the resumed checkpoint corresponds to")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="debug: stop after this many updates from the resume point")
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

    from transformers import AutoProcessor
    target, proc = common.load_target(
        device=dev,
        dtype=torch.float32 if a.target_dtype == "float32" else torch.bfloat16)
    head = common.load_head(a.init_from, device=dev, dtype=torch.bfloat16)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    head_dtype = next(head.parameters()).dtype
    embed = target.get_input_embeddings().weight
    if embed.dtype != head_dtype:
        embed = embed.to(head_dtype)

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

    if start_update == 0 and not a.resume_from:
        head.save_pretrained(os.path.join(a.out, "step0"))
        torch.save({"head": head.state_dict(), "master_optimizer": opt.state_dict(),
                    "update": 0, "seed": a.seed, "arm": "QA-SFT"},
                   os.path.join(a.out, "step0", "training_state.pt"))
        print("[ckpt] step0 saved (the unmodified shared init-from weights)", flush=True)

    order = [json.loads(l) for l in
             (pathlib.Path(a.run) / "dataset" /
              f"train_schedule_seed_{a.seed}.jsonl").read_text().splitlines()
             if l.strip()]
    order.sort(key=lambda r: r["order"])
    slots = {int(r["update"]): [] for r in order}
    for r in order:
        slots[int(r["update"])].append(r)
    for k in slots:
        slots[k].sort(key=lambda r: r["slot"])
        if len(slots[k]) != a.qa_per_update:
            raise SystemExit(f"update {k} has {len(slots[k])} QA slots, expected "
                             f"{a.qa_per_update}")

    config = {
        "arm": "QA-SFT", "entry_point": "train_qa_sft.py",
        "run": a.run, "seed": a.seed, "pool": a.pool, "records": a.records,
        "init_from": a.init_from, "out": a.out, "updates": a.updates,
        "qa_per_update": a.qa_per_update, "anchors_per_qa": a.anchors_per_qa,
        "lr": a.lr, "master_weights": a.master_weights,
        "objective": "variable-reveal block CE, 0.8**(k-reveal) weighted, "
                     "labels = frozen target greedy AR; EOS guard zeroes every "
                     "label past the first EOS",
        "head_dtype": str(head_dtype), "target_dtype": a.target_dtype,
        "pilot_only": a.target_dtype != "float32",
        "time_mode": a.time_mode, "eos_stop": "immediate",
        "betas": list(common.BETAS), "weight_decay": common.WEIGHT_DECAY,
        "clip": a.clip, "torch": torch.__version__,
    }
    (pathlib.Path(a.out) / "config.json").write_text(
        json.dumps(config, indent=2))
    log = open(pathlib.Path(a.out) / "log.jsonl", "a")

    t0 = time.time()
    for upd in range(start_update, a.updates):
        torch.cuda.reset_peak_memory_stats(a.gpu)
        t_upd = time.time()
        qa_stats, total_positions, total_loss_num, total_w = [], 0, 0.0, 0.0
        opt.zero_grad(set_to_none=True)
        n_total_anchors = 0
        for slot in slots[upd]:
            task = f"{slot['video_id']}|{slot['question_id']}"
            if task not in recs:
                raise KeyError(f"{task} has no frozen target record")
            rec = recs[task]
            row = pool.loc[task]
            full = rec["input_ids"].to(dev).long()
            prompt_len = int(rec["prompt_len"])
            eos_pos = rec["eos_pos_in_answer"]
            n_ans = int(rec["n_answer_tokens"])
            if int(full.shape[0]) > a.max_seq_len:
                raise SystemExit(f"{task}: trajectory {int(full.shape[0])} exceeds "
                                 f"--max-seq-len {a.max_seq_len}")
            text = build_video_text(proc, row["prompt"])
            x = encode_inputs(proc, row.to_dict(), a.time_mode, text=text)
            if x["input_ids"][0].tolist() != full[:prompt_len].tolist():
                raise RuntimeError(f"{task}: encoded prompt differs from frozen record")
            captured = []
            hook = target.model.register_forward_hook(
                lambda module, inputs, output: captured.append(output.last_hidden_state.detach()))
            try:
                ctx = common.target_context_cached(
                    target, full[None], pixel_values_videos=x["pixel_values_videos"].to(dev),
                    video_grid_thw=x["video_grid_thw"].to(dev))
            finally:
                hook.remove()
            with torch.no_grad():
                current = target.lm_head(captured[0][0, prompt_len-1:-1]).argmax(-1).tolist()
            if current != full[prompt_len:].tolist():
                raise RuntimeError(f"{task}: current target replay differs from frozen AR")
            del captured, current
            if ctx.dtype != head_dtype:
                ctx = ctx.to(head_dtype)
            cand = qa_anchors.sft_candidate_positions(prompt_len, eos_pos, n_ans)
            # Candidate p predicts p+1: include the EOS label itself.
            valid_end = int(cand[-1]) + 1 if cand else prompt_len - 2
            anchors = qa_anchors.select_anchors(cand, a.anchors_per_qa)
            if not anchors:
                continue
            gen = torch.Generator().manual_seed(a.seed * 7919 + upd * 131 + len(qa_stats))
            reveals = th.sample_reveals(len(anchors), head.block_size, gen, "cpu")
            anchors, reveals = th.live_anchors(anchors, reveals,
                                               int(full.shape[0]), head.block_size)
            if not anchors:
                continue
            anchors_t = torch.tensor(anchors, device=dev)
            reveals = reveals.to(dev)
            blocks, targets, super_mask, weights = qa_blocks(
                full, anchors, reveals, head.block_size, common.MASK_ID,
                valid_end, dev)
            flat_super = super_mask.reshape(-1).bool()
            if int(flat_super.sum()) == 0:
                continue
            block_embed = F.embedding(blocks, embed).reshape(
                1, len(anchors) * head.block_size, -1)
            ctx_positions = torch.arange(ctx.shape[1], device=dev).unsqueeze(0)
            q_positions = (anchors_t.unsqueeze(1)
                           + torch.arange(head.block_size, device=dev).unsqueeze(0)
                           ).reshape(1, -1)
            attn_mask = th.build_anchor_mask(anchors, head.block_size,
                                             ctx.shape[1], dev, block_embed.dtype)
            hidden = th.forward_anchors(head, ctx, block_embed, ctx_positions,
                                        q_positions, attn_mask)
            hidden = hidden.reshape(1, len(anchors) * head.block_size, -1)[0]
            sel = hidden[flat_super]
            # `compute_logits(hidden, output_head)` calls `output_head(hidden)`
            # and then applies the head's own `output_multiplier` /
            # `final_logit_softcapping`. The DEPLOYMENT path in
            # `glance/decode.py` (and `opd_fullkl.draft_block_logits`) uses
            # `target.lm_head(hidden)` directly, so training must do the same or
            # the two objectives would be defined on different logits.
            logits = LmHeadShim(target, head_dtype).lm_head(sel[None])[0]
            labels = targets.reshape(-1)[flat_super]
            flat_w = weights.reshape(-1)[flat_super]
            nll = F.cross_entropy(logits.float(), labels, reduction="none")
            loss = (nll * flat_w).sum() / flat_w.sum().clamp_min(1.0)
            loss.backward()
            with torch.no_grad():
                correct = (logits.argmax(-1) == labels).float()
                total_loss_num += float((nll.detach() * flat_w).sum())
                total_w += float(flat_w.sum())
                total_positions += int(flat_super.sum())
                n_total_anchors += len(anchors)
                qa_stats.append({
                    "task": task, "video_id": slot["video_id"],
                    "question_id": slot["question_id"],
                    "prompt_len": prompt_len, "answer_tokens": n_ans,
                    "hit_eos": eos_pos is not None,
                    "candidates": len(cand), "anchors": len(anchors),
                    "positions": int(flat_super.sum()),
                    "loss": float(loss.detach()),
                    "top1": float(correct.mean()),
                    "reveal_mean": float(reveals.float().mean()),
                })
            del ctx, x, blocks, targets, super_mask, weights, hidden, logits, nll, loss
        if not qa_stats:
            raise SystemExit(f"update {upd + 1} produced no supervision at all")
        # Clipping happens on the fp32 master gradients inside `step()`, exactly
        # as in the verified C4 combination gate; the pre-clip norm is read here
        # for the log and must be finite before any weight is touched.
        opt.sync_grads()
        grad_norm = float(opt.grad_norm())
        applied = min(1.0, a.clip / (grad_norm + 1e-12)) if a.clip else 1.0
        if not np.isfinite(grad_norm):
            raise RuntimeError(f"update {upd + 1}: non-finite grad norm "
                               f"({grad_norm}); refusing to step")
        opt.step()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated(a.gpu) / 2**30
        rec_log = {
            "arm": "QA-SFT", "seed": a.seed, "update": upd + 1,
            "lr": a.lr, "grad_norm": float(grad_norm), "clip_scale": float(applied),
            "loss_weighted_mean": total_loss_num / max(total_w, 1e-9),
            "supervised_positions": total_positions,
            "anchors": n_total_anchors,
            "supervision_weight_sum": total_w,
            "context_forwards": len(slots[upd]),
            "context_tokens": sum(int(recs[f"{z['video_id']}|{z['question_id']}"]["input_ids"].numel()) for z in slots[upd]),
            "teacher_forwards": 0,
            "qa": qa_stats,
            "video_ids": [s["video_id"] for s in qa_stats],
            "seconds": time.time() - t_upd, "peak_gib": peak,
            "elapsed_min": (time.time() - t0) / 60,
        }
        log.write(json.dumps(rec_log, ensure_ascii=False) + "\n")
        log.flush()
        if upd % a.log_every == 0 or upd + 1 == a.updates:
            print(f"[QA-SFT s{a.seed}] upd {upd+1}/{a.updates} "
                  f"loss {rec_log['loss_weighted_mean']:.4f} "
                  f"pos {total_positions} anchors {n_total_anchors} "
                  f"gnorm {grad_norm:.3f} peak {peak:.1f}GiB "
                  f"{rec_log['seconds']:.1f}s", flush=True)
        if a.save_every and (upd + 1) % a.save_every == 0:
            ck = pathlib.Path(a.out) / f"step{upd + 1}"
            head.save_pretrained(ck)
            torch.save({"head": head.state_dict(),
                        "master_optimizer": opt.state_dict(),
                        "update": upd + 1, "seed": a.seed, "arm": "QA-SFT"},
                       ck / "training_state.pt")
            print(f"[ckpt] {ck}", flush=True)
        if a.stop_after and (upd + 1 - start_update) >= a.stop_after:
            print(f"[debug] stop-after reached at update {upd + 1}", flush=True)
            break
    if start_update == 0 and not (pathlib.Path(a.out) / f"step{a.updates}").exists() \
            and a.updates % a.save_every != 0:
        head.save_pretrained(pathlib.Path(a.out) / f"step{a.updates}")
    print(f"[done] QA-SFT seed {a.seed} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
