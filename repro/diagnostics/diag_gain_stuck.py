#!/usr/bin/env python3
"""WHY: the gain-stuck diagnostic -- WHERE the phaseC4 gain is getting stuck.

phaseC4 measured, on `video_fresh30` (n=29, fp32, tree b63k8), C0 3.7377 /
C1 3.7225 / C2 3.7403, i.e. `C2 - C0 = +0.0027` and `C2 - C1 = +0.0179`, both
positive in sign and both indistinguishable from zero. The next question is not
"train more" but WHERE the effect stops: in the drafter's per-offset distribution,
in the decode topology, or in the data the update actually saw.

This file is a READ-ONLY diagnostic. It never builds an optimizer, never calls
`backward`, never writes a checkpoint and never touches the frozen test-set
candidates. It re-uses the verified phaseC4 machinery by IMPORTING it:
`opd_train_video_fullkl.build_video_trace` (pass 1: one cached fp32 target prefill
+ the chain rollout), `select_anchors`, `build_anchor_plan`,
`attach_cropped_prefix` and `draft_block_logits`, plus `opd_fullkl`'s
`target_prefill_shared`, `teacher_block_logits` and the two full-vocabulary KLs.
Nothing in those files is modified; a second copy of any of them would be exactly
the drifting duplicate this project keeps re-learning to avoid.

THE PROTOCOL (D1: fixed-context prediction change). Anchors are generated ONCE,
by the C0 policy (`--student C0=<path>` in the `anchors` phase), with the same
`max_rounds` / per-video budget phaseC4 used. Every student is then scored on
those IDENTICAL anchors, IDENTICAL draft-induced proposals and IDENTICAL valid
masks, against ONE fp32 teacher, so the only thing that varies between students is
the student. This is deliberately an OFFLINE diagnostic: formal online decoding
still generates each arm's anchors from its own policy and is not changed here.

Per offset `k = 1..K` of every anchor we record, for each student:
  * `fkl = KL(p_k || q_k)` and `rkl = KL(q_k || p_k)`, full vocabulary, fp32;
  * the rank of the teacher's greedy token in the student's distribution, and
    whether the student's argmax IS that token (`hit`);
  * `n_acc_own` -- the student's own leading agreement with the teacher's greedy
    chain, i.e. the first-rejection point the same proposal WOULD have had under
    this student's policy.

Offsets are then split three ways and reported separately, because a gain that
only shows up in the accepted prefix and not at the first rejection cannot move
acceptance: (1) `acc` = `k <= n_acc` (2) `first_rej` = `k == n_acc + 1`
(3) `suffix` = `k > n_acc + 1`. The primary split uses the ANCHOR's `n_acc` (the
C0-rollout verifier's count), which is the same offset set for all three students
and therefore the only version that is honestly paired; a second, per-student
split by `n_acc_own` is reported alongside because that is the quantity deployment
actually experiences.

FRAGMENTATION NOTE (why the anchor pass is separate from the metric passes). An
fp32 teacher is 32.7 GiB and one fp32 head is 3.9 GiB, so a target + ONE head fits
comfortably (36.6 GiB) but a target + THREE heads is 44.4 GiB (measured) and
leaves 2.75 GiB, which is not enough for a 1500-token fp32 prefill. The metric
passes therefore run one student each, each recomputing the teacher. That makes
the teacher a re-computation rather than a shared object, so every pass records an
exact bit-pattern checksum of the teacher logits per anchor
(`teacher_bits`, an int64 sum of the fp32 bit patterns, plus the argmax row) and
`report` refuses to pair two students whose teacher checksums differ -- the
pairing claim is verified, not assumed.

Two levels of honesty about what was not checked: `--limit` exists so a smoke run
can score a prefix of each group, and the emitted JSON records `limit` so a
truncated run can never be mistaken for a full one.
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
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(_REPRO_ROOT))

from shared import common  # noqa: E402
from historical import opd_fullkl  # noqa: E402
from historical import opd_train_video_fullkl as trainer  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

#: The three students of the phaseC4 comparison, in report order. C0 is the start
#: checkpoint (`head_video439_aligned/step1000`), C1 continue-SFT u10 and C2
#: paper-faithful full-vocab online OPD u10.
STUDENTS = ("C0", "C1", "C2")

#: The parts of an anchor block the diagnostic reports separately. `acc` is the
#: matched prefix the verifier commits; `first_rej` is the single offset that
#: decides whether the block could have been one token longer; `suffix` is
#: everything the rejected suffix objective (reverse KL, 0.8**(k-1)) sees.
PARTS = ("acc", "first_rej", "suffix")

#: Group 2's selection rule is a fixed, score-free hash order. The salt is part of
#: the pre-registration: changing it changes the group, so it is written into
#: `split.json` and must not be edited after results exist.
GROUP2_SALT = "phaseC5-g2"

#: Bootstrap settings for every paired delta. Resampling unit = video (a video's
#: offsets are not independent draws, so the video is the exchangeable unit).
BOOT_N = 10_000
BOOT_SEED = 12345


def _json_line(handle, obj) -> None:
    handle.write(json.dumps(obj) + "\n")


def load_pool(records_path: str, pool_path: str):
    """The 439 `split == 'train'` records, in record order, plus the pool frame.

    The order matters: both arms' video sequences are permutations OF THIS LIST,
    so a re-ordering here would silently change which videos were trained on.
    """
    import pandas as pd

    blob = torch.load(records_path, weights_only=False, map_location="cpu")
    recs_all = blob["records"]
    pool = pd.read_parquet(pool_path)
    split_of = dict(zip(pool["video_name"].astype(str), pool["split"].astype(str)))
    recs = [r for r in recs_all if split_of.get(str(r["task"])) == "train"]
    return recs, recs_all, pool.set_index("video_name")


# ==========================================================================
# Pre-registration: reconstruct the trained videos, fix groups 2 and 3
# ==========================================================================

def phasec4_trained_videos(records_path: str, pool_path: str):
    """Rebuild both arms' 10 trained videos FROM THE CODE, not from the logs.

    C2 (`opd_train_video_fullkl.main`) walks `video_order(len(recs), seed, epoch)`
    -- `random.Random(seed * 1_000_003 + epoch).shuffle(range(n))` -- and takes
    up to `--anchors-per-video` anchors per video; with 24 anchors/video and
    `anchors_per_update = 24` the phaseC4 log shows `videos: 1` for all ten
    updates, so the ten updates consumed exactly `order[0:10]` with no skipped
    videos (`anchorless_videos` and `no_anchor_videos` are 0 in every update).

    C1 (`train_head_video.main`) draws ONE video per step as
    `recs[order[step % len(order)]]` with `order = np.random.default_rng(seed)
    .permutation(len(recs))`, so its ten steps consumed `order[0:10]`.

    Returns `(recs, tasks, c1, c2)` -- the record list, its task order, and the two
    arms' video lists in the order they were consumed.
    """
    recs, _recs_all, _pool = load_pool(records_path, pool_path)
    tasks = [str(r["task"]) for r in recs]

    order_c2 = trainer.video_order(len(recs), 0, 0)
    c2 = [tasks[i] for i in order_c2[:10]]

    rng = np.random.default_rng(0)
    order_c1 = rng.permutation(len(recs))
    c1 = [tasks[int(order_c1[i])] for i in range(10)]
    return recs, tasks, c1, c2


def c1_length_verification(recs, c1, c1_jsonl: str):
    """Cross-check the C1 reconstruction against its own log's trajectory lengths.

    `train_head_video.py` logs `L` = the encoded trajectory length of the row it
    trained on. The reconstruction is a claim about WHICH row each step took, and
    `L` is an independent observable of that row; a permutation that is off by one
    index has no reason to reproduce ten consecutive lengths.
    """
    by_task = {str(r["task"]): int(r["input_ids"].shape[0]) for r in recs}
    logged = [json.loads(line) for line in open(c1_jsonl) if line.strip()]
    logged_L = [int(row["L"]) for row in logged[:len(c1)]]
    rebuilt_L = [by_task[t] for t in c1]
    return {"logged_L": logged_L, "rebuilt_L": rebuilt_L,
            "exact_match": logged_L == rebuilt_L,
            "n_compared": len(logged_L)}


def group2_rule(tasks, exclude, n: int, salt: str = GROUP2_SALT):
    """`n` training videos phaseC4 did NOT sample, by a fixed, score-free rule.

    The rule is a SHA-256 of `"<salt>:<video>"`, sorted ascending: deterministic,
    reproducible from the salt alone, and independent of any model score, anchor
    count, τ or loss. IMPORTANT LABEL: these rows are still part of the historical
    training set (`pool_499` `split == 'train'`, the same 439 records A4/phaseC4
    trained from). They are "not sampled by phaseC4", NOT held out.
    """
    remaining = [t for t in tasks if t not in exclude]
    keyed = sorted(remaining,
                   key=lambda t: hashlib.sha256(f"{salt}:{t}".encode()).hexdigest())
    return keyed[:n]


def cmd_split(a) -> int:
    """Write `$R/results/phaseC5/split.json` BEFORE any result is looked at."""
    recs, tasks, c1, c2 = phasec4_trained_videos(a.records, a.pool)
    union = sorted(set(c1) | set(c2))
    g2 = group2_rule(tasks, set(union), len(union))

    import pandas as pd

    fresh = pd.read_parquet(a.fresh30)
    g3 = [str(x) for x in fresh["video_name"]]
    all_rec_tasks = {str(r["task"]) for r in torch.load(
        a.records, weights_only=False, map_location="cpu")["records"]}
    missing = [t for t in g3 if t not in all_rec_tasks]
    if missing:
        raise KeyError(f"{len(missing)} fresh30 rows have no pooled trajectory, "
                       f"e.g. {missing[:3]}; their prompts/anchors are unknown")

    ver = {}
    if a.c1_jsonl:
        ver["c1_trajectory_length_match"] = c1_length_verification(
            recs, c1, a.c1_jsonl)
    blob = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": ("phaseC5 gain-stuck diagnostic (D1/D3). Pre-registered BEFORE "
                    "any diagnostic metric was computed."),
        "records": a.records, "pool": a.pool, "fresh30": a.fresh30,
        "n_train_records": len(recs),
        "seed": 0,
        "rules": {
            "g1": ("union of the two phaseC4 arms' 10-video samples, rebuilt from "
                   "the trainer code: C2 = random.Random(0).shuffle(range(439))"
                   "[0:10] (no video skipped: every update logged videos=1), "
                   "C1 = np.random.default_rng(0).permutation(439)[0:10]"),
            "g2": (f"sha256('{GROUP2_SALT}:<video>') ascending over the 439 train "
                   f"records minus g1, take the first |g1|. Score-free. These rows "
                   "are STILL IN THE HISTORICAL TRAINING SET (pool split=='train'); "
                   "they are 'not sampled by phaseC4', NOT held out."),
            "g3": ("the existing validation $R/data/video_fresh30.parquet "
                   "(pool split=='heldout'; already used for A4 selection, so it is "
                   "validation, not an untouched test)"),
        },
        "groups": {
            "g1_phaseC4_trained": {"n": len(union), "videos": union,
                                   "C1_order": c1, "C2_order": c2,
                                   "arms_of": {t: [n for n, s in (("C1", c1), ("C2", c2))
                                                   if t in s] for t in union}},
            "g2_train_not_sampled_by_phaseC4": {"n": len(g2), "videos": g2},
            "g3_validation_fresh30": {"n": len(g3), "videos": g3},
        },
        "verification": ver,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as handle:
        json.dump(blob, handle, indent=1)

    # The D2 evaluator takes an EXPLICIT row list (its docstring: a split filter
    # cannot silently change the denominator), so the g1+g2 rows are materialised
    # as their own parquet in the pre-registration step, in split order. g3 is
    # scored from the existing `video_fresh30.parquet` rather than a copy, so
    # there is exactly one definition of the validation rows.
    pool = pd.read_parquet(a.pool).set_index("video_name")
    g12 = pool.loc[union + g2].reset_index()
    g12_path = os.path.join(os.path.dirname(os.path.abspath(a.out)),
                            "group_g1g2_rows.parquet")
    g12.to_parquet(g12_path, index=False)
    blob["eval_rows_parquet"] = {"g1g2": g12_path, "g3": a.fresh30}
    with open(a.out, "w") as handle:
        json.dump(blob, handle, indent=1)

    print(f"[split] g1 {len(union)} (C1 {len(c1)} + C2 {len(c2)}, "
          f"intersection {len(set(c1) & set(c2))})  g2 {len(g2)}  g3 {len(g3)}")
    print(f"[split] g1+g2 eval rows -> {g12_path} ({len(g12)} rows)")
    if ver:
        print(f"[split] C1 reconstruction vs C1.jsonl L: "
              f"{ver['c1_trajectory_length_match']}")
    print(f"[split] -> {a.out}")
    return 0


# ==========================================================================
# Model loading (fp32 only; this diagnostic never differentiates anything)
# ==========================================================================

def load_student(path: str, device: str):
    """One fp32 head, in eval mode, with gradients disabled."""
    head = common.load_head(path, device=device, dtype=torch.float32).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def records_by_task(records_path: str):
    blob = torch.load(records_path, weights_only=False, map_location="cpu")
    return {str(r["task"]): r for r in blob["records"]}


def read_split(path: str):
    blob = json.load(open(path))
    order = []
    for name in ("g1_phaseC4_trained", "g2_train_not_sampled_by_phaseC4",
                 "g3_validation_fresh30"):
        for task in blob["groups"][name]["videos"]:
            order.append((name.split("_")[0], task))
    return blob, order


# ==========================================================================
# Phase 1: C0 anchors (the fixed prefixes every student is scored on)
# ==========================================================================

def cmd_anchors(a) -> int:
    if a.student.split("=")[0] != "C0":
        raise ValueError("the anchor pass must run the C0 policy; a different "
                         "anchor source would make D1 a different experiment")
    name, path = a.student.split("=", 1)
    split, order = read_split(a.split)
    if a.limit:
        order = order[:a.limit]
    recs = records_by_task(a.records)

    dev = f"cuda:{a.gpu}"
    target, _ = common.load_target(device=dev, dtype=torch.float32)
    head = load_student(path, dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    with open(a.out, "w") as handle:
        for i, (group, task) in enumerate(order):
            rec = recs[task]
            row = split_rows(a.pool).loc[task]
            t_v = time.time()
            trace, rollout_anchors = trainer.build_video_trace(
                target, head, proc, rec, row, device=dev,
                max_seq_len=a.max_seq_len, max_rounds=a.max_rounds,
                verify="chain", head_dtype=torch.float32, rollout_mode="legacy-reference")
            kept, info = trainer.select_anchors(
                rollout_anchors, budget=a.anchors_per_video,
                max_seq_len=a.max_seq_len, prompt_len=trace.prompt_len,
                valid_end=trace.valid_end, per_video=a.anchors_per_video)
            plans = [trainer.build_anchor_plan(
                task, anchor, total=int(trace.full_ids.shape[0]),
                valid_end=trace.valid_end, prompt_len=trace.prompt_len,
                gamma=a.gamma, device=dev) for anchor in kept]
            total = int(trace.full_ids.shape[0])
            _json_line(handle, {
                "group": group, "task": task, "L": total,
                "prompt_len": trace.prompt_len, "valid_end": trace.valid_end,
                "n_rollout_anchors": len(rollout_anchors), "n_kept": len(kept),
                "select_info": info,
                "anchors": [{"pos": int(p.pos), "prop": [int(x) for x in p.prop],
                             "n_acc": int(p.n_acc), "valid": [bool(v) for v in p.valid]}
                            for p in plans]})
            handle.flush()
            del trace
            print(f"  [{i+1}/{len(order)}] {group} {task[:16]:16s} L={total:4d} "
                  f"anchors {len(kept)}/{len(rollout_anchors)} "
                  f"{time.time()-t_v:.1f}s peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    print(f"[anchors] {len(order)} videos in {(time.time()-t0)/60:.1f} min -> {a.out}")

    if a.verify_c2_u1:
        verify_c2_update1(a, order, recs, target, head, dev)
    return 0


def verify_c2_update1(a, order, recs, target, head, dev):
    """Reproduce C2's update-1 supervision counts from C0-generated anchors.

    `opd_train_video_fullkl` logs, for update 1 (which runs on the C0 policy, i.e.
    the checkpoint C2 started from), `N_acc_total`, `Z_total` and
    `n_rej_offsets`. If the reconstructed video order and the anchor selection are
    right, replaying them reproduces those three numbers exactly. A mismatch means
    the anchor pass and the phaseC4 trainer disagree, and every D1 number is then
    about a different set of blocks.
    """
    split, _ = read_split(a.split)
    c2_order = split["groups"]["g1_phaseC4_trained"]["C2_order"]
    first = c2_order[0]
    logged = [json.loads(line) for line in open(a.verify_c2_u1) if line.strip()]
    u1 = next(r for r in logged if int(r["update"]) == 1)

    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    row = split_rows(a.pool).loc[first]
    trace, rollout_anchors = trainer.build_video_trace(
        target, head, proc, recs[first], row, device=dev,
        max_seq_len=a.max_seq_len, max_rounds=a.max_rounds, verify="chain",
        head_dtype=torch.float32, rollout_mode="legacy-reference")
    kept, _info = trainer.select_anchors(
        rollout_anchors, budget=a.anchors_per_video, max_seq_len=a.max_seq_len,
        prompt_len=trace.prompt_len, valid_end=trace.valid_end,
        per_video=a.anchors_per_video)
    plans = [trainer.build_anchor_plan(
        first, anchor, total=int(trace.full_ids.shape[0]),
        valid_end=trace.valid_end, prompt_len=trace.prompt_len, gamma=a.gamma,
        device=dev) for anchor in kept]
    n_acc, z_total, _acc_present, _rej_present = trainer.plan_update_denominators(
        plans)
    n_rej = sum(int(p.offsets_rej.numel()) for p in plans)
    got = {"N_acc_total": int(n_acc), "Z_total": float(z_total),
           "n_rej_offsets": int(n_rej), "anchors": len(plans)}
    print(f"[verify] C2 update1 video {first}: got {got}")
    print(f"[verify] logged: N_acc_total {u1['N_acc_total']} Z_total "
          f"{u1['Z_total']:.6f} n_rej_offsets {u1['n_rej_offsets']} anchors "
          f"{u1['anchors']}")
    ok = (got["N_acc_total"] == int(u1["N_acc_total"])
          and got["n_rej_offsets"] == int(u1["n_rej_offsets"])
          and abs(got["Z_total"] - float(u1["Z_total"])) < 1e-6
          and got["anchors"] == int(u1["anchors"]))
    print(f"[verify] C2 UPDATE-1 REPLAY {'PASS' if ok else 'FAIL'}")
    del trace


_SPLIT_ROWS = {}


def split_rows(pool_path: str):
    """Pool rows keyed by video_name, loaded once per process."""
    if pool_path not in _SPLIT_ROWS:
        import pandas as pd

        _SPLIT_ROWS[pool_path] = pd.read_parquet(pool_path).set_index("video_name")
    return _SPLIT_ROWS[pool_path]


# ==========================================================================
# Phase 2: per-offset metrics for one student on the fixed anchors
# ==========================================================================

@torch.no_grad()
def student_metrics(p_logits: torch.Tensor, q_logits: torch.Tensor):
    """Per-offset KL (both directions), target-token rank and top-1 hit.

    `p_logits`/`q_logits` are `(K, V)`. The KLs are `opd_fullkl`'s full-vocabulary
    fp32 functions -- no top-k, no renormalisation, so "the teacher" is the whole
    distribution and a rank is a genuine rank. The rank is 1-based and counts
    strictly-greater student logits, i.e. ties would be counted as shared first
    place; in fp32 on a 151936-way support the tie probability is negligible and
    the convention is recorded here rather than hidden.
    """
    p = p_logits.float()
    q = q_logits.float()
    p_arg = p.argmax(dim=-1)
    q_arg = q.argmax(dim=-1)
    q_at_target = q.gather(1, p_arg[:, None]).squeeze(1)
    rank = (q > q_at_target[:, None]).sum(dim=-1) + 1
    return {
        "fkl": [float(x) for x in opd_fullkl.full_kl_forward(p, q)],
        "rkl": [float(x) for x in opd_fullkl.full_kl_reverse(q, p)],
        "rank": [int(x) for x in rank],
        "hit": [bool(x) for x in (q_arg == p_arg)],
        "p_arg": [int(x) for x in p_arg],
        "q_arg": [int(x) for x in q_arg],
        "teacher_bits": int(p.contiguous().view(torch.int32).to(torch.int64).sum()),
    }


def leading_agreement(hit):
    """`n_acc_own`: how many leading offsets the student's argmax got right.

    This is the same rule `opd_online_rollout.anchor_from_logits(verify='chain')`
    applies to a rollout block, evaluated on the FIXED proposal: the first
    rejection point this student's policy would have produced for this block.
    """
    n = 0
    for h in hit:
        if not h:
            break
        n += 1
    return n


def cmd_metrics(a) -> int:
    name, path = a.student.split("=", 1)
    split, order = read_split(a.split)
    if a.limit:
        order = order[:a.limit]
    recs = records_by_task(a.records)
    anchors = {json.loads(line)["task"]: json.loads(line)
               for line in open(a.anchors) if line.strip()}
    dev = f"cuda:{a.gpu}"
    target, _ = common.load_target(device=dev, dtype=torch.float32)
    head = load_student(path, dev)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    with open(a.out, "w") as handle:
        for i, (group, task) in enumerate(order):
            if task not in anchors:
                raise KeyError(f"{task!r} has no pre-registered anchor record; "
                               "the anchor pass and the split disagree")
            arec = anchors[task]
            t_v = time.time()
            trace, _rollout = trainer.build_video_trace(
                target, head, proc, recs[task], split_rows(a.pool).loc[task],
                device=dev, max_seq_len=a.max_seq_len,
                max_rounds=a.rollout_rounds, verify="chain",
                head_dtype=torch.float32, rollout_mode="legacy-reference")
            if int(trace.prompt_len) != int(arec["prompt_len"]):
                raise ValueError(f"{task}: prompt_len {int(trace.prompt_len)} != "
                                 f"anchor pass {int(arec['prompt_len'])}")
            out_anchors = []
            for anchor in arec["anchors"]:
                pos = int(anchor["pos"])
                prop = [int(x) for x in anchor["prop"]]
                if int(trace.full_ids.shape[0]) != int(arec["L"]):
                    raise ValueError(f"{task}: trajectory length changed between "
                                     "passes; the stored anchors are invalid")
                anchor_vispos = opd_fullkl.prefix_vision_positions(
                    target, trace.full_ids[:pos + 1][None],
                    pixel_values_videos=trace.pixel_values_videos,
                    video_grid_thw=trace.video_grid_thw)
                before = int(trace.prefix_kv.get_seq_length())
                anchor_kv = trainer.attach_cropped_prefix(trace, pos)
                p_logits = opd_fullkl.teacher_block_logits(
                    target, anchor_vispos, anchor_kv, anchor_pos=pos, prop=prop,
                    chunk=a.chunk)
                if int(trace.prefix_kv.get_seq_length()) != before:
                    raise RuntimeError("the shared prefix cache was mutated while "
                                       "scoring an anchor")
                q_logits = trainer.draft_block_logits(
                    head, target, trace.fused[:, :pos], anchor_pos=pos,
                    root_token=int(trace.full_ids[pos]),
                    prompt_len=trace.prompt_len, head_dtype=torch.float32)
                m = student_metrics(p_logits, q_logits)
                m["pos"] = pos
                m["n_acc_anchor"] = int(anchor["n_acc"])
                m["valid"] = [bool(v) for v in anchor["valid"]]
                m["n_acc_own"] = leading_agreement(m["hit"])
                out_anchors.append(m)
                del p_logits, q_logits, anchor_kv
            _json_line(handle, {"group": group, "task": task,
                                "L": int(trace.full_ids.shape[0]),
                                "prompt_len": int(trace.prompt_len),
                                "valid_end": int(trace.valid_end),
                                "anchors": out_anchors})
            handle.flush()
            del trace
            print(f"  [{i+1}/{len(order)}] {name} {group} {task[:16]:16s} "
                  f"anchors {len(out_anchors)} {time.time()-t_v:.1f}s peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    print(f"[metrics {name}] {len(order)} videos in "
          f"{(time.time()-t0)/60:.1f} min -> {a.out}")
    return 0


# ==========================================================================
# Phase 3: aggregation and the paired tables (pure CPU)
# ==========================================================================

def part_of(k_index: int, n_acc: int, valid: bool):
    """Which of the three parts a 0-based offset index falls in (1-based `k`)."""
    if not valid:
        return None
    k = k_index + 1
    if k <= n_acc:
        return "acc"
    if k == n_acc + 1:
        return "first_rej"
    return "suffix"


def proposal_lookup(anchors_path):
    """`(task, pos) -> prop` from the anchor-pass file.

    Needed for the teacher-consistent re-split: the rollout's `n_acc` was computed
    against the STORED trajectory's tokens, which the ACL audit found diverges from
    the current fp32 target's greedy path on 69/70 videos. Whether the frozen
    proposal still matches the CURRENT teacher is `argmax(p_k) == prop[k]`, and it
    can only be evaluated where `prop` is known.
    """
    out = {}
    for line in open(anchors_path):
        rec = json.loads(line)
        for a in rec["anchors"]:
            out[(rec["task"], int(a["pos"]))] = [int(x) for x in a["prop"]]
    return out


def offset_aligned(align, task, prompt_len, pos, k_index):
    """Is offset `k = k_index+1` still on the stored/AR common prefix?

    Stream position `pos + k` holds a target token; if it lies at or beyond the
    first divergence, the stored token is NOT what the current fp32 target would
    produce, so the acceptance label at this offset is stale (ACL finding).
    """
    info = align.get(task)
    if info is None:
        return True
    return (int(pos) + k_index + 1 - int(prompt_len)) < int(info["common"])


def collect(metrics, split, by_own: bool, props=None, align=None):
    """Flatten one student's per-video records into (group, part) -> per-video rows.

    Returns `{(group, part): {task: {"fkl": [...], "rkl": [...], "rank": [...],
    "hit": [...], "n_anchor": int, "n_acc_own": [...], "n_acc_anchor": [...]}}}`.
    Aggregation is per VIDEO first (later paired deltas resample videos), so the
    flat per-offset rows are kept grouped by their video.
    """
    out: dict = {}
    for rec in metrics:
        group = rec["group"]
        task = rec["task"]
        for anchor in rec["anchors"]:
            n_acc_anchor = int(anchor["n_acc_anchor"])
            n_acc_own = int(anchor["n_acc_own"])
            if props is not None:
                # Teacher-consistent acceptance of the FROZEN proposal: the
                # leading offsets whose proposal token is still the current fp32
                # teacher's greedy token. `n_acc_anchor` is the same count against
                # the stored trajectory instead.
                prop = props[(task, int(anchor["pos"]))]
                n_acc_anchor = 0
                for k in range(len(anchor["p_arg"])):
                    if k >= len(prop) or anchor["p_arg"][k] != prop[k]:
                        break
                    n_acc_anchor += 1
            n_acc = n_acc_own if by_own else n_acc_anchor
            for idx, valid in enumerate(anchor["valid"]):
                if align is not None and not offset_aligned(
                        align, task, rec["prompt_len"], anchor["pos"], idx):
                    continue
                part = part_of(idx, n_acc, bool(valid))
                if part is None:
                    continue
                cell = out.setdefault((group, part), {}).setdefault(
                    task, {"fkl": [], "rkl": [], "rank": [], "hit": [],
                           "n_acc_own": [], "n_acc_anchor": []})
                cell["fkl"].append(float(anchor["fkl"][idx]))
                cell["rkl"].append(float(anchor["rkl"][idx]))
                cell["rank"].append(int(anchor["rank"][idx]))
                cell["hit"].append(bool(anchor["hit"][idx]))
            for cell_name, value in (("n_acc_own", n_acc_own),
                                     ("n_acc_anchor", n_acc_anchor)):
                cell = out.setdefault((group, "acctab"), {}).setdefault(
                    task, {"n_acc_own": [], "n_acc_anchor": []})
                cell[cell_name].append(value)
    return out


def cell_stats(cell_map):
    """Means over one video's offsets in one part (the unit of the pairing)."""
    fkl = cell_map["fkl"]
    rkl = cell_map["rkl"]
    rank = cell_map["rank"]
    hit = cell_map["hit"]
    if not fkl:
        return None
    return {"n_off": len(fkl), "fkl": float(np.mean(fkl)),
            "rkl": float(np.mean(rkl)), "rank": float(np.mean(rank)),
            "top1": float(np.mean([1.0 if h else 0.0 for h in hit]))}


def _tstat(mean, sd, n):
    """One-sample t of a paired mean; `nan` when the sd is exactly zero.

    A zero sd happens legitimately here (a metric identical on every video), and
    `nan` says "no evidence" instead of an infinity that would read as
    overwhelming evidence.
    """
    if n < 2 or sd == 0.0:
        return float("nan")
    return float(mean / (sd / math.sqrt(n)))


def paired_delta(a_map, b_map, n_boot=BOOT_N, seed=BOOT_SEED):
    """Mean of (b - a) over the videos present in BOTH, with a bootstrap CI.

    The resampling unit is the video, because a video contributes many correlated
    offsets; resampling offsets would shrink the interval without justification.
    """
    tasks = sorted(set(a_map) & set(b_map))
    diffs = []
    for t in tasks:
        sa = cell_stats(a_map[t])
        sb = cell_stats(b_map[t])
        if sa is None or sb is None:
            continue
        diffs.append([sb["fkl"] - sa["fkl"], sb["rkl"] - sa["rkl"],
                      sb["rank"] - sa["rank"], sb["top1"] - sa["top1"]])
    if not diffs:
        return None
    arr = np.asarray(diffs, dtype=float)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(4)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.shape[0], size=(n_boot, arr.shape[0]))
    boots = arr[idx].mean(axis=1)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return {"n_videos": int(arr.shape[0]), "mean": mean.tolist(),
            "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
            "t": [_tstat(float(mean[i]), float(sd[i]), arr.shape[0])
                  for i in range(4)],
            "n_pos_fkl": int((arr[:, 0] > 0).sum()),
            "n_pos_rank_neg": int((arr[:, 2] < 0).sum())}


def first_reject_stats(metrics):
    """Per video: mean `n_acc_own`, all-accepted anchor fraction, n_acc histogram."""
    out: dict = {}
    for rec in metrics:
        group, task = rec["group"], rec["task"]
        own = [int(x["n_acc_own"]) for x in rec["anchors"]]
        hist = [0] * (trainer.K_OFFSETS + 1)
        for v in own:
            hist[v] += 1
        out.setdefault(group, {})[task] = {
            "mean_n_acc_own": float(np.mean(own)) if own else float("nan"),
            "all_acc_frac": (float(np.mean([1.0 if v == trainer.K_OFFSETS else 0.0
                                            for v in own])) if own else float("nan")),
            "hist": hist, "n_anchors": len(own)}
    return out


def _subset(cell, tasks):
    """One (group, part) cell restricted to a video subset (`None` = all)."""
    if cell is None:
        return None
    if tasks is None:
        return cell
    return {t: v for t, v in cell.items() if t in tasks}


def emit_group_tables(lines, summary, tag, gname, fixed, own, frej, students,
                      tasks=None):
    """Append one group's (or subgroup's) part-split + first-reject tables.

    `tag` is the key the numbers are filed under in `summary` (a subgroup gets its
    own tag so it cannot overwrite the group's), `gname` the group the cells were
    collected under, and `tasks` an optional video subset -- used for the exposure
    subgroups, where the videos one arm was trained on are the interesting unit.
    Table entries are pooled per-offset means; the paired deltas resample VIDEOS,
    which is why both views are emitted.
    """
    for label, tables in (("fixed", fixed), ("own", own)):
        title = ("fixed (anchor n_acc, identical offset sets)" if label == "fixed"
                 else "own (per-student n_acc_own)")
        lines.append(f"\n### part split: {title}\n")
        lines.append("| student | part | n_off | fKL(p\\|\\|q) | rKL(q\\|\\|p) "
                     "| target rank | top-1 hit |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for name in students:
            for part in PARTS:
                cell = _subset(tables[name].get((gname, part)), tasks)
                if not cell:
                    continue
                fkl = [x for v in cell.values() for x in v["fkl"]]
                rkl = [x for v in cell.values() for x in v["rkl"]]
                rank = [x for v in cell.values() for x in v["rank"]]
                hit = [x for v in cell.values() for x in v["hit"]]
                lines.append(
                    f"| {name} | {part} | {len(fkl)} | {np.mean(fkl):.4f} | "
                    f"{np.mean(rkl):.4f} | {np.mean(rank):.2f} | "
                    f"{np.mean([1.0 if h else 0.0 for h in hit]):.4f} |")
                summary["fixed_split" if label == "fixed" else "own_split"][
                    f"{tag}|{name}|{part}"] = {
                        "n_off": len(fkl), "fkl": float(np.mean(fkl)),
                        "rkl": float(np.mean(rkl)), "rank": float(np.mean(rank)),
                        "top1": float(np.mean([1.0 if h else 0.0 for h in hit]))}
        lines.append("")
        for other in [s for s in students if s != "C0"]:
            for part in PARTS:
                am = _subset(tables["C0"].get((gname, part)), tasks)
                bm = _subset(tables[other].get((gname, part)), tasks)
                if not am or not bm:
                    continue
                d = paired_delta(am, bm)
                if d is None:
                    continue
                lines.append(
                    f"* **{other}-C0 {part}** (n={d['n_videos']} v): "
                    f"\u0394fKL {d['mean'][0]:+.4f} "
                    f"[{d['ci_lo'][0]:+.4f},{d['ci_hi'][0]:+.4f}] "
                    f"(t {d['t'][0]:+.2f}, {d['n_pos_fkl']}/{d['n_videos']} +) | "
                    f"\u0394rKL {d['mean'][1]:+.4f} "
                    f"[{d['ci_lo'][1]:+.4f},{d['ci_hi'][1]:+.4f}] | "
                    f"\u0394rank {d['mean'][2]:+.2f} "
                    f"[{d['ci_lo'][2]:+.2f},{d['ci_hi'][2]:+.2f}] "
                    f"({d['n_pos_rank_neg']}/{d['n_videos']} better) | "
                    f"\u0394top1 {d['mean'][3]:+.4f} "
                    f"[{d['ci_lo'][3]:+.4f},{d['ci_hi'][3]:+.4f}]")
                summary.setdefault("paired_" + label, {})[
                    f"{tag}|{other}-C0|{part}"] = d
        lines.append("")
    lines.append("\n### first-reject / anchor acceptance\n")
    lines.append("| student | mean n_acc_own | all-15-accepted anchors "
                 "| histogram n_acc_own (0..15) |")
    lines.append("|---|---:|---:|---|")
    for name in students:
        cell = _subset(frej[name].get(gname), tasks)
        if not cell:
            continue
        hist = np.zeros(trainer.K_OFFSETS + 1)
        for v in cell.values():
            hist += np.asarray(v["hist"], dtype=float)
        n = sum(v["n_anchors"] for v in cell.values())
        # Pooled: total accepted-leading offsets / total anchors, and total
        # all-accepted anchors / total anchors -- not a mean over videos.
        mean_acc = float((hist * np.arange(trainer.K_OFFSETS + 1)).sum()
                         / max(hist.sum(), 1.0))
        allacc = float(hist[trainer.K_OFFSETS] / max(hist.sum(), 1.0))
        lines.append(f"| {name} | {mean_acc:.3f} | {allacc:.3f} | "
                     f"{[int(x) for x in hist]} |")
        summary["first_reject"][f"{tag}|{name}"] = {
            "mean_n_acc_own": mean_acc, "all_acc_frac": allacc,
            "hist": [int(x) for x in hist], "n_anchors": int(n)}
    lines.append("")


def ar_alignment(records_path, eval_paths):
    """`video -> (prompt_len, stored_continuation, ar_continuation)`.

    WHY: the stored `input_ids` in the pool records are the trajectory the head was
    trained against, but the deployment (and the lossless AR gate) runs the CURRENT
    fp32 target. The ACL audit found the two diverge on 69/70 videos, so a stored
    offset is only a valid reference where the two still share a token prefix. The
    AR continuation is read from the eval JSONs (`ar_ids`, prompt stripped), which
    is exactly the sequence the gate compared against; the stored continuation is
    read from the record.
    """
    blob = torch.load(records_path, weights_only=False, map_location="cpu")
    stored = {}
    for r in blob["records"]:
        pl = int(r["prompt_len"])
        stored[str(r["task"])] = (pl, [int(x) for x in r["input_ids"][pl:].tolist()])
    ar = {}
    for path in eval_paths:
        for row in json.load(open(path))["rows"]:
            ids = row.get("ar_ids")
            if ids is None:
                raise KeyError(f"{path} carries no ar_ids; the alignment of the "
                               "stored trajectory to the deployed AR path cannot "
                               "be established without it")
            video = str(row["video"])
            if video in ar and ar[video] != [int(x) for x in ids]:
                raise ValueError(f"{video}: two different ar_ids in the eval files; "
                                 "the AR reference is not unique")
            ar[video] = [int(x) for x in ids]
    out = {}
    for video, (pl, cont) in stored.items():
        if video not in ar:
            continue
        common = 0
        ref = ar[video]
        while common < min(len(cont), len(ref)) and cont[common] == ref[common]:
            common += 1
        out[video] = {"prompt_len": pl, "common": common,
                      "stored_len": len(cont), "ar_len": len(ref),
                      "diverges": common < min(len(cont), len(ref))}
    return out


def cmd_report(a) -> int:
    split = json.load(open(a.split))
    group_names = {g.split("_")[0]: g for g in split["groups"]}
    metrics = {name: [json.loads(line) for line in open(path) if line.strip()]
               for name, path in (s.split("=", 1) for s in a.metrics)}

    # --- teacher checksums: the pairing claim has to be verified, not assumed
    teacher = {}
    for name, recs in metrics.items():
        for rec in recs:
            for anchor in rec["anchors"]:
                key = (rec["task"], int(anchor["pos"]))
                teacher.setdefault(key, {})[name] = int(anchor["teacher_bits"])
    disagree = {k: v for k, v in teacher.items() if len(set(v.values())) != 1}
    print(f"[report] teacher checksums: {len(teacher)} anchors x "
          f"{len(metrics)} students, {len(disagree)} disagreements")

    props = proposal_lookup(a.anchors) if a.anchors else None
    if props is not None:
        # The ACL check, re-derived from this run's own files: how many anchors
        # would move if the split used the CURRENT teacher instead of the stored
        # trajectory's tokens. `p_arg` and `n_acc_anchor` come from the metric
        # pass, `prop` from the anchor pass.
        moved = 0
        total = 0
        for name, recs in metrics.items():
            for rec in recs:
                for anchor in rec["anchors"]:
                    total += 1
                    prop = props[(rec["task"], int(anchor["pos"]))]
                    n_t = 0
                    for k in range(len(anchor["p_arg"])):
                        if k >= len(prop) or anchor["p_arg"][k] != prop[k]:
                            break
                        n_t += 1
                    if n_t != int(anchor["n_acc_anchor"]):
                        moved += 1
            break
        print(f"[report] teacher-consistent re-split: {moved}/{total} anchors "
              "change their acceptance label")
        summary_prefix = {"teacher_label_changed_anchors": moved,
                          "teacher_label_anchors_compared": total}
    else:
        summary_prefix = {}
    align = None
    if a.records and a.evals:
        align = ar_alignment(a.records,
                             [spec.split("=", 1)[1] for spec in a.evals])
        n_div = sum(1 for v in align.values() if v["diverges"])
        print(f"[report] stored vs deployed AR: {n_div}/{len(align)} videos diverge; "
              f"median common prefix "
              f"{int(np.median([v['common'] for v in align.values()]))} tokens")
    if a.aligned_only and align is None:
        raise ValueError("--aligned-only needs --records and --evals: the AR "
                         "reference lives in the eval JSONs")
    use_align = align if a.aligned_only else None
    fixed = {name: collect(recs, split, by_own=False, props=props, align=use_align)
             for name, recs in metrics.items()}
    own = {name: collect(recs, split, by_own=True, align=use_align)
           for name, recs in metrics.items()}
    frej = {name: first_reject_stats(recs) for name, recs in metrics.items()}

    summary = {"teacher_checksum_disagreements": len(disagree),
               "limit": a.limit, "groups": {},
               "fixed_split": {}, "own_split": {}, "first_reject": {}}
    summary.update(summary_prefix)

    lines = []
    lines.append("# phaseC5 gain-stuck diagnostic -- D1 tables\n")
    if a.aligned_only:
        n_off = sum(len(v["fkl"]) for tbl in fixed.values()
                    for v in tbl.values() if "fkl" in v)
        lines.append(f"**D1 RESTRICTED to offsets inside the stored/AR common "
                     f"prefix** (stale-reference offsets dropped; {n_off} offsets "
                     f"survive). Do not compare these n_off against the "
                     f"unrestricted run.\n")
    lines.append(f"students: {', '.join(sorted(metrics))}   "
                 f"teacher-checksum disagreements: {len(disagree)} / "
                 f"{len(teacher)} anchors\n")

    for gname, full in group_names.items():
        n_vid = split["groups"][full]["n"]
        lines.append(f"\n## group {gname} ({full}, n={n_vid} videos)\n")
        summary["groups"][gname] = {"n_declared": n_vid}
        emit_group_tables(lines, summary, gname, gname, fixed, own, frej,
                          sorted(metrics))

    # Exposure subgroups inside g1. Each arm trained on ITS OWN 10 videos and the
    # two samples are DISJOINT (verified at split time), so "the videos actually
    # trained on" decomposes into "trained on by C1" and "trained on by C2". A gain
    # that only shows up on the arm's own rows is memorisation of those rows, not
    # generalisation -- and g2 is the same-distribution, same-split control for it.
    arms_of = split["groups"]["g1_phaseC4_trained"].get("arms_of", {})
    for arm in ("C1", "C2"):
        subset = {t for t, arms in arms_of.items() if arm in arms}
        lines.append(f"\n## group g1 / videos trained on by {arm} "
                     f"(n={len(subset)} videos)\n")
        emit_group_tables(lines, summary, f"g1|{arm}-exposed", "g1", fixed, own,
                          frej, sorted(metrics), tasks=subset)

    # ---- the paired loss -> rank -> first-reject -> tau table, on group 3
    if a.evals:
        # ---- load every eval json. Several files may cover DIFFERENT row lists
        # for the same (topology, arm) -- fresh30 for D2, the g1+g2 parquet for
        # D3 -- so they are MERGED per (topology, arm) and a video that appears in
        # two files with two different taus is a hard error, not an overwrite.
        evals = {}
        for spec in a.evals:
            topo, rest = spec.split(":", 1)
            name, path = rest.split("=", 1)
            blob = json.load(open(path))
            evals.setdefault((topo, name, "rowsets"), set()).add(
                os.path.basename(str(blob["config"]["rows"])))
            evals[(topo, name, "W")] = int(blob["config"]["budget"])
            evals[(topo, name, "k")] = int(blob["config"]["branch"])
            evals[(topo, name, "lossless")] = (
                evals.get((topo, name, "lossless"), 0)
                + sum(1 for r in blob["rows"] if r.get("lossless")))
            evals[(topo, name, "n")] = (evals.get((topo, name, "n"), 0)
                                        + len(blob["rows"]))
            taus = evals.setdefault((topo, name), {})
            accepts = evals.setdefault((topo, name, "acc"), {})
            # `--warmup-policy keep-jobs`: `--warmup 1` is a JOB flag, not a
            # property of a video. The formal validation keeps its original
            # warmup row excluded (that is the row every historical number
            # dropped), but the g1/g2 exposure files' warmup row is an arbitrary
            # first row of a 40-row job, so dropping it would silently delete one
            # video from a pre-registered group. Dropping it for the g1/g2 files
            # is what the `formal` policy does and what it is named for.
            drop_warmup = (a.warmup_policy == "formal"
                           or os.path.basename(str(blob["config"]["rows"]))
                           == os.path.basename(str(split["fresh30"])))
            for r in blob["rows"]:
                if r["warmup"] and drop_warmup:
                    continue
                prev = taus.get(r["video"])
                if prev is not None and prev != float(r["tau"]):
                    raise ValueError(
                        f"{r['video']} carries two different taus in {topo}/{name} "
                        "from different eval files; the merge would be wrong")
                taus[r["video"]] = float(r["tau"])
                accepts[r["video"]] = [int(x) for x in r["accepted"]]
        topologies = sorted({k[0] for k in evals if len(k) == 2})
        eval_students = sorted({k[1] for k in evals if len(k) == 2})

        # The exposure subsets are the pre-registered ones, so the tau table lines
        # up row-for-row with the D1 part tables. `arms_of` is written by `split`.
        arms_of = split["groups"]["g1_phaseC4_trained"].get("arms_of", {})
        subsets = [("g1_all_phaseC4_trained",
                    set(split["groups"]["g1_phaseC4_trained"]["videos"]))]
        for arm in ("C1", "C2"):
            subsets.append((f"g1|{arm}-exposed",
                            {t for t, a_ in arms_of.items() if arm in a_}))
        subsets.append(("g2_train_not_sampled",
                        set(split["groups"]["g2_train_not_sampled_by_phaseC4"]["videos"])))
        subsets.append(("g3_validation",
                        set(split["groups"]["g3_validation_fresh30"]["videos"])))
        g3 = set(split["groups"]["g3_validation_fresh30"]["videos"])

        def tau_mean(topo, name, subset):
            m = evals.get((topo, name), {})
            vals = [v for t, v in m.items() if t in subset]
            return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)

        lines.append("\n### D2: tau by topology on the validation rows "
                     f"(n={len(g3)} rows declared, warmup dropped per file)\n")
        lines.append("| topology | budget/branch | student | n | mean tau "
                     "| lossless | eval rows |")
        lines.append("|---|---|---|---:|---:|---|---|")
        summary["d2"] = {}
        for topo in topologies:
            for name in eval_students:
                if (topo, name) not in evals:
                    continue
                mean, n = tau_mean(topo, name, g3)
                rowsets = ",".join(sorted(evals[(topo, name, "rowsets")]))
                lines.append(f"| {topo} | {evals[(topo,name,'W')]}/"
                             f"{evals[(topo,name,'k')]} | {name} | {n} | "
                             f"{mean:.4f} | {evals[(topo,name,'lossless')]}/"
                             f"{evals[(topo,name,'n')]} | {rowsets} |")
                summary["d2"][f"{topo}|{name}"] = {
                    "mean_tau_g3": mean, "n_g3": n,
                    "budget": evals[(topo, name, "W")],
                    "branch": evals[(topo, name, "k")],
                    "lossless": evals[(topo, name, "lossless")],
                    "n_rows": evals[(topo, name, "n")]}

        lines.append("\n### D2 paired tau deltas (validation rows)\n")
        lines.append("| topology | comparison | mean dtau | 95% CI | t | positive |")
        lines.append("|---|---|---:|---|---:|---:|")
        summary["d2_paired"] = {}
        for topo in topologies:
            for other in [x for x in eval_students if x != "C0"]:
                a_map = evals.get((topo, "C0"), {})
                b_map = evals.get((topo, other), {})
                tasks = sorted((set(a_map) & set(b_map)) & g3)
                if not tasks:
                    continue
                d = np.asarray([b_map[t] - a_map[t] for t in tasks])
                rng = np.random.default_rng(BOOT_SEED)
                boots = d[rng.integers(0, len(d), size=(BOOT_N, len(d)))].mean(axis=1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                tstat = _tstat(float(d.mean()), float(d.std(ddof=1)), len(d))
                lines.append(f"| {topo} | {other}-C0 | {d.mean():+.4f} | "
                             f"[{lo:+.4f},{hi:+.4f}] | {tstat:+.2f} | "
                             f"{int((d > 0).sum())}/{len(d)} |")
                summary["d2_paired"][f"{topo}|{other}-C0"] = {
                    "mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                    "t": float(tstat), "n_pos": int((d > 0).sum()), "n": len(d)}

        # ---- D3: tau by exposure group, the same subsets D1 was split into
        lines.append("\n### D3: tau by exposure group (paired per video)\n")
        lines.append("| group | topology | n | tau C0 | tau C1 | tau C2 "
                     "| dtau C1-C0 [95% CI] (t) | dtau C2-C0 [95% CI] (t) |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        summary["d3_tau"] = {}
        for label, subset in subsets:
            for topo in topologies:
                means = {name: tau_mean(topo, name, subset) for name in eval_students}
                n = max(v[1] for v in means.values())
                if n == 0:
                    continue
                cells = []
                for other in [x for x in eval_students if x != "C0"]:
                    am = evals.get((topo, "C0"), {})
                    bm = evals.get((topo, other), {})
                    tasks = sorted((set(am) & set(bm)) & subset)
                    if not tasks:
                        cells.append("n/a")
                        continue
                    d = np.asarray([bm[t] - am[t] for t in tasks])
                    rng = np.random.default_rng(BOOT_SEED)
                    boots = d[rng.integers(0, len(d),
                                           size=(BOOT_N, len(d)))].mean(axis=1)
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    tstat = _tstat(float(d.mean()), float(d.std(ddof=1)), len(d))
                    cells.append(f"{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
                                 f"({tstat:+.2f})")
                    # A subgroup delta is only interesting if its interval is not
                    # the whole story: the CI is reported next to it so a +0.08 on
                    # 19 videos cannot be read as a +0.08 effect.
                    summary["d3_tau"][f"{label}|{topo}|{other}-C0"] = {
                        "d_tau": float(d.mean()), "ci_lo": float(lo),
                        "ci_hi": float(hi), "t": float(tstat), "n": len(tasks),
                        "n_pos": int((d > 0).sum())}
                while len(cells) < 2:
                    cells.append("n/a")
                lines.append(
                    f"| {label} | {topo} | {n} | "
                    f"{means.get('C0', (float('nan'), 0))[0]:.4f} | "
                    f"{means.get('C1', (float('nan'), 0))[0]:.4f} | "
                    f"{means.get('C2', (float('nan'), 0))[0]:.4f} | "
                    f"{cells[0]} | {cells[1]} |")
        lines.append("")

        # ---- D1 <-> D2 bridge: the first-token acceptance rate, computed twice
        # from two independent objects. D1 side: the fraction of offset-1 blocks
        # whose teacher greedy token the student ranks <= R (R=1 is what the b1k1
        # probe proposes; R=8 is the tree's per-node branching factor, so it is
        # what the tree can accept). Decode side: the fraction of rounds that
        # committed >=1 draft token. If these two agree, the offline fixed-context
        # probe IS measuring the same acceptance event the decoder experiences --
        # which is the only reason the loss->rank->first-reject->tau chain can be
        # read end to end at all.
        lines.append("\n### D1<->D2 bridge: first-token acceptance, offline vs decode\n")
        lines.append("| group | topology | arm | D1 P(rank<=1) @offset1 | "
                     "D1 P(rank<=8) @offset1 | decode P(round commits >=1) | "
                     "decode mean accepted |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        summary["bridge"] = {}
        for label, subset in subsets:
            for topo in topologies:
                for name in eval_students:
                    recs_s = metrics.get(name)
                    if not recs_s:
                        continue
                    per_v = {}
                    for rec in recs_s:
                        if rec["task"] not in subset:
                            continue
                        ranks = [int(a["rank"][0]) for a in rec["anchors"]
                                 if a["valid"] and a["valid"][0]
                                 and (use_align is None or offset_aligned(
                                     use_align, rec["task"], rec["prompt_len"],
                                     a["pos"], 0))]
                        if ranks:
                            per_v[rec["task"]] = (float(np.mean([r <= 1 for r in ranks])),
                                                  float(np.mean([r <= 8 for r in ranks])))
                    dec = evals.get((topo, name, "acc"), {})
                    dv = {t: (float(np.mean([1.0 if x >= 1 else 0.0 for x in a])),
                              float(np.mean(a)))
                          for t, a in dec.items() if t in subset and a}
                    if not per_v or not dv:
                        continue
                    p1 = float(np.mean([v[0] for v in per_v.values()]))
                    p8 = float(np.mean([v[1] for v in per_v.values()]))
                    d1 = float(np.mean([v[0] for v in dv.values()]))
                    d2 = float(np.mean([v[1] for v in dv.values()]))
                    lines.append(f"| {label} | {topo} | {name} | {p1:.4f} | {p8:.4f} "
                                 f"| {d1:.4f} | {d2:.4f} |")
                    summary["bridge"][f"{label}|{topo}|{name}"] = {
                        "d1_p_rank_le_1": p1, "d1_p_rank_le_8": p8,
                        "decode_p_accept_ge_1": d1, "decode_mean_accepted": d2,
                        "n_d1_videos": len(per_v), "n_decode_videos": len(dv)}
        lines.append("")

        # ---- D3b: tau restricted to the FIRST `--window` rounds, i.e. exactly the
        # horizon D1 scores (24 C0-anchor blocks per video). `tau = 1 +
        # mean(accepted)` holds bit-exactly on every row of every file here, so a
        # windowed tau is the same estimator over a prefix of the same rounds.
        # WHY it matters: if the whole tau gain lived inside this window, D1 would
        # have had to see it; if the gain lives after it, D1's horizon is simply
        # too short and the two probes are not in conflict.
        lines.append(f"\n### D3b: tau over the first {a.window} rounds (the D1 "
                     "horizon) vs the full response\n")
        lines.append("| group | topology | n rows >= W | dtau(W) C1-C0 | dtau(W) C2-C0 "
                     "| dtau(full) C1-C0 | dtau(full) C2-C0 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        summary["d3b_window"] = {}

        def tau_window(topo, name, tasks):
            """1 + mean(accepted[:W]) per row; `tau = 1 + mean(accepted)` is exact
            on every row of every file here, so this is the same estimator over a
            prefix of the same rounds."""
            out = {}
            for t in tasks:
                acc = evals.get((topo, name, "acc"), {}).get(t)
                if acc is None or len(acc) < a.window:
                    continue
                out[t] = 1.0 + float(np.mean(acc[:a.window]))
            return out

        def paired_ci(d):
            rng = np.random.default_rng(BOOT_SEED)
            boots = d[rng.integers(0, len(d), size=(BOOT_N, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            return (float(d.mean()), float(lo), float(hi),
                    _tstat(float(d.mean()), float(d.std(ddof=1)), len(d)),
                    int((d > 0).sum()))

        for label, subset in subsets:
            for topo in topologies:
                others = [x for x in eval_students if x != "C0"]
                if not others:
                    continue
                wc = {x: tau_window(topo, x, sorted(subset)) for x in eval_students}
                n_rows = len(wc.get("C0", {}))
                if n_rows == 0:
                    continue
                cells_w, cells_f = [], []
                for other in others:
                    common_w = sorted(set(wc["C0"]) & set(wc.get(other, {})))
                    dw = np.asarray([wc[other][t] - wc["C0"][t] for t in common_w])
                    m, lo, hi, tstat, npos = paired_ci(dw)
                    cells_w.append(f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] ({tstat:+.2f})")
                    summary["d3b_window"][f"{label}|{topo}|{other}-C0"] = {
                        "window": a.window, "d_tau_window": m, "ci_lo": lo,
                        "ci_hi": hi, "t": tstat, "n": len(common_w),
                        "n_pos": npos}
                    full = evals.get((topo, other), {})
                    common_f = sorted(set(evals.get((topo, "C0"), {})) & set(full) & subset)
                    df = np.asarray([full[t] - evals[(topo, "C0")][t] for t in common_f])
                    mf = paired_ci(df)
                    cells_f.append(f"{mf[0]:+.4f} [{mf[1]:+.4f},{mf[2]:+.4f}]")
                    summary["d3b_window"][f"{label}|{topo}|full|{other}-C0"] = {
                        "d_tau": mf[0], "ci_lo": mf[1], "ci_hi": mf[2],
                        "t": mf[3], "n": len(common_f), "n_pos": mf[4]}
                while len(cells_w) < 2:
                    cells_w.append("n/a")
                while len(cells_f) < 2:
                    cells_f.append("n/a")
                lines.append(f"| {label} | {topo} | {n_rows} | {cells_w[0]} | "
                             f"{cells_w[1]} | {cells_f[0]} | {cells_f[1]} |")
        lines.append("")

        # ---- the required single paired table on the validation rows, because
        # that is where the already-measured tau lives.
        lines.append("\n### paired table (validation = g3): loss change -> rank "
                     "change -> first-reject change -> actual tau change\n")
        lines.append("| comparison | part | d fKL | d rKL | d rank | d top1 "
                     "| d mean n_acc_own | d all-accepted | d tau tree "
                     "| d tau chain |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        summary["paired_table_g3"] = {}
        for other in [x for x in sorted(metrics) if x != "C0"]:
            tau_cells = {}
            for topo in topologies:
                am = evals.get((topo, "C0"), {})
                bm = evals.get((topo, other), {})
                tasks = sorted((set(am) & set(bm)) & g3)
                if tasks:
                    tau_cells[topo] = float(np.mean([bm[t] - am[t] for t in tasks]))
            am = frej["C0"].get("g3", {})
            bm = frej[other].get("g3", {})
            tasks = sorted(set(am) & set(bm))
            d_mean = float(np.mean([bm[t]["mean_n_acc_own"] - am[t]["mean_n_acc_own"]
                                    for t in tasks]))
            d_all = float(np.mean([bm[t]["all_acc_frac"] - am[t]["all_acc_frac"]
                                   for t in tasks]))
            for part in PARTS:
                amap = fixed["C0"].get(("g3", part))
                bmap = fixed[other].get(("g3", part))
                if not amap or not bmap:
                    continue
                d = paired_delta(amap, bmap)
                lines.append(
                    f"| {other}-C0 | {part} | {d['mean'][0]:+.4f} | "
                    f"{d['mean'][1]:+.4f} | {d['mean'][2]:+.2f} | "
                    f"{d['mean'][3]:+.4f} | {d_mean:+.3f} | {d_all:+.3f} | "
                    f"{tau_cells.get('tree', float('nan')):+.4f} | "
                    f"{tau_cells.get('chain', float('nan')):+.4f} |")
                summary["paired_table_g3"][f"{other}-C0|{part}"] = {
                    "d_fkl": d["mean"][0], "d_rkl": d["mean"][1],
                    "d_rank": d["mean"][2], "d_top1": d["mean"][3],
                    "d_mean_n_acc_own": d_mean, "d_all_acc_frac": d_all,
                    "d_tau_tree": tau_cells.get("tree"),
                    "d_tau_chain": tau_cells.get("chain")}
        lines.append("")

    text = "\n".join(lines)
    with open(a.out, "w") as handle:
        handle.write(text + "\n")
    if a.out_json:
        with open(a.out_json, "w") as handle:
            json.dump(summary, handle, indent=1)
    print(text)
    print(f"[report] -> {a.out}")
    return 0


# ==========================================================================
# CLI
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split", help="pre-register split.json (no GPU needed)")
    p.add_argument("--records", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--fresh30", required=True)
    p.add_argument("--c1-jsonl", default=None,
                   help="phaseC4 C1.jsonl, used only to verify the reconstruction")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("anchors", help="C0 anchor pass (fp32 target + C0 head)")
    p.add_argument("--split", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--student", required=True, help="C0=<head path>")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = every video")
    p.add_argument("--max-rounds", type=int, default=24)
    p.add_argument("--anchors-per-video", type=int, default=24)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--max-seq-len", type=int, default=common.MAX_SEQ_LEN)
    p.add_argument("--verify-c2-u1", default=None,
                   help="phaseC4 C2.jsonl; replays update 1 and compares counts")
    p.set_defaults(func=cmd_anchors)

    p = sub.add_parser("metrics", help="per-offset metrics for ONE student")
    p.add_argument("--split", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--student", required=True, help="C1=<head path>")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--rollout-rounds", type=int, default=1,
                   help="the metric pass scores STORED anchors, so the rollout "
                        "build_video_trace would run is pure overhead; one round "
                        "keeps the validated pass-1 builder (prefill, per-position "
                        "hidden, valid_end, prompt_len check) as the single code "
                        "path, at the cost of one extra head forward per video")
    p.add_argument("--max-seq-len", type=int, default=common.MAX_SEQ_LEN)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("report", help="aggregate D1 + D2 into tables (CPU only)")
    p.add_argument("--split", required=True)
    p.add_argument("--metrics", action="append", required=True,
                   help="C0=<d1_C0.jsonl> (repeat per student)")
    p.add_argument("--records", default=None,
                   help="pool .pt with the stored trajectories; with --evals it "
                        "enables the stored-vs-deployed-AR alignment report")
    p.add_argument("--aligned-only", action="store_true",
                   help="restrict D1 to offsets still inside the stored/AR common "
                        "prefix, i.e. drop the stale-reference offsets the ACL "
                        "audit identified (needs --records and --evals)")
    p.add_argument("--anchors", default=None,
                   help="the anchor-pass jsonl; with it present the 'fixed' split "
                        "is re-derived from the CURRENT teacher (argmax(p)==prop) "
                        "instead of the stored trajectory's tokens, and the number "
                        "of anchors whose acceptance label changes is reported")
    p.add_argument("--evals", action="append", default=[],
                   help="topology:student=<eval json>, e.g. tree:C0=...")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--warmup-policy", default="formal",
                   choices=["formal", "keep-jobs"],
                   help="formal = drop the first row of every eval file (n=29 on "
                        "a 30-row file); keep-jobs = drop it only for the "
                        "validation row list, so a 40-row exposure job does not "
                        "silently lose one pre-registered video")
    p.add_argument("--window", type=int, default=24,
                   help="round window for the D3b 'is the gain inside the D1 "
                        "horizon' check; 24 = the anchor budget D1 and phaseC4 used")
    p.add_argument("--out", required=True)
    p.add_argument("--out-json", default=None)
    p.set_defaults(func=cmd_report)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
