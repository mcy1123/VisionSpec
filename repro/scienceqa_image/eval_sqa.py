#!/usr/bin/env python3
"""Lossless b63k8 online evaluation of one head on the frozen ScienceQA split.

Per record it runs the deployment `glance.generate` (budget 63, branch_k 8,
temperature 0, decode-only timing) and compares the committed token ids against
the independent greedy AR oracle **token for token**, including EOS, bonus EOS,
root EOS and the ``max_new_tokens`` cap. Comparing decoded text is not a
lossless check and is not done here.

The AR oracle is loaded from a ``build_eval_ar.py`` cache when given
(``--ar-cache``); otherwise it is derived in-process. The oracle does not depend
on the draft head, so every arm of a split shares it -- which is also why the
"SPEC == AR" claim can be checked against a single, frozen oracle per record.

Outputs:
  ``<out>.json``   full summary + per-record rows
  ``--per-question`` CSV (default next to --out)
  ``--spec-equals-ar`` JSON gate artifact
  ``--load-manifest`` JSON with the GPU occupancy sampled at start and end, so a
     shared-host timing run can be marked contaminated honestly.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPRO = HERE.parent
ROOT = REPRO.parent
sys.path[:0] = [str(HERE), str(REPRO), str(ROOT / "GLANCE_repo"),
                str(ROOT / "scripts" / "dflash_pkg" / "dflash-0.1.0")]

from shared import common  # noqa: E402
import glance  # noqa: E402
from glance.vl import VisionPositions  # noqa: E402

ANSWER_RE = re.compile(r"Answer:\s*\(?\s*([A-Ea-e])\s*\)?")


def _ar_key(pid):
    return pid


def load_split(run, split, limit=None):
    rows = []
    with open(pathlib.Path(run) / "dataset" / "records_manifest.csv") as fh:
        for r in csv.DictReader(fh):
            if r["canonical_split"] == split:
                rows.append(r)
    rows.sort(key=lambda r: r["official_row_index"])
    return rows[:limit] if limit else rows


def gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        return [l for l in out.strip().splitlines() if l]
    except Exception as e:      # noqa: BLE001
        return [f"nvidia-smi failed: {e}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=768)
    ap.add_argument("--budget", type=int, default=63)
    ap.add_argument("--branch-k", type=int, default=8)
    ap.add_argument("--head-dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--ar-cache", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split-label", default=None,
                    help="label recorded in the artifact; use it when --limit "
                         "selects a frozen SUBSET so the subset's numbers can never "
                         "be confused with the full split's")
    ap.add_argument("--repeat-spec", type=int, default=1,
                    help="extra SPEC repeats on the first --repeat-n records for "
                         "the timing repeatability requirement; the first pass of "
                         "every record is the reported one")
    ap.add_argument("--repeat-n", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-question", default=None)
    ap.add_argument("--spec-equals-ar", default=None)
    ap.add_argument("--load-manifest", default=None)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(a.gpu)
    torch.set_grad_enabled(False)
    # Resolved once, before anything prints or writes: a frozen dev SUBSET must be
    # labelled differently from the full dev split or the two could be confused.
    split_label = a.split_label or a.split

    head = common.load_head(a.head, dtype=getattr(torch, a.head_dtype),
                            device=dev).eval()
    target, proc = common.load_target(device=dev, dtype=torch.float32)
    eos = int(proc.tokenizer.eos_token_id)
    rows = load_split(a.run, a.split, a.limit or None)

    ar = {}
    if a.ar_cache and pathlib.Path(a.ar_cache).exists():
        blob = torch.load(a.ar_cache, weights_only=False, map_location="cpu")
        if blob["max_pixels"] != a.max_pixels or blob["max_new"] != a.max_new:
            raise SystemExit(f"AR cache {a.ar_cache} was built with "
                             f"max_pixels={blob['max_pixels']}/max_new={blob['max_new']} "
                             f"but this run uses {a.max_pixels}/{a.max_new}; refusing "
                             f"to compare against a different oracle")
        ar = {r["pid"]: r for r in blob["records"]}
        print(f"[eval] AR oracle from {a.ar_cache} ({len(ar)} records, fp32)",
              flush=True)
    load_start = gpu_snapshot()
    print(f"[eval] {a.tag} on {split_label}: {len(rows)} records, budget {a.budget} "
          f"branch {a.branch_k}, head {a.head} ({a.head_dtype})", flush=True)

    out_rows, t_all = [], time.time()
    repeats = []
    for i, r in enumerate(rows):
        from PIL import Image
        img = Image.open(r["image_path"]).convert("RGB")
        if a.max_pixels and max(img.size) > a.max_pixels:
            w, h = img.size
            s = a.max_pixels / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        ids, pixels, grid = common.encode(proc, r["prompt"], img)
        ids = ids.to(dev)
        pix = pixels.to(dev) if pixels is not None else None
        g = grid.to(dev) if grid is not None else None
        pos = VisionPositions(pix, g)

        if r["pid"] in ar:
            ar_tok = list(ar[r["pid"]]["tokens"])
            ar_ms_per_token = ar[r["pid"]]["ms_per_token"]
            ar_rounds = ar[r["pid"]]["n_rounds"]
            ar_prefill_ms = float(ar[r["pid"]].get("prefill_ms", 0.0))
            ar_cached = True
        else:
            ar_final, ar_st = glance.greedy_generate(target, ids, a.max_new, [eos],
                                                    positions=pos, return_stats=True)
            ar_tok = ar_final[0, ids.shape[1]:].tolist()
            ar_ms_per_token, ar_rounds, ar_cached = ar_st.ms_per_token, ar_st.rounds, False
            ar_prefill_ms = ar_st.prefill_ms

        final, st = glance.generate(head, target, ids, a.max_new, [eos],
                                   budget=a.budget, branch_k=a.branch_k,
                                   temperature=0.0, positions=pos,
                                   return_stats=True, timing="decode_only",
                                   eos_stop="immediate")
        sp_tok = final[0, ids.shape[1]:].tolist()
        exact = sp_tok == ar_tok
        text = proc.tokenizer.decode(sp_tok, skip_special_tokens=True)
        m = ANSWER_RE.findall(text)
        pred = m[-1].upper() if m else None
        rec = {
            "tag": a.tag, "split": split_label, "pid": r["pid"],
            "image_group_id": r["image_group_id"], "subject": r["subject"],
            "grade": r["grade"], "n_choices": r["n_choices"],
            "gold_letter": r["gold_letter"], "pred_letter": pred,
            "answer_correct": (pred == r["gold_letter"]) if pred else False,
            "answer_parse_ok": pred is not None,
            "prompt_len": int(ids.shape[1]),
            "n_tokens": st.n_tokens, "ar_n_tokens": len(ar_tok),
            "spec_rounds": st.rounds, "ar_rounds": ar_rounds,
            "tau": st.tau, "ar_tau": 1.0,
            "tokens_per_round": st.n_tokens / max(st.rounds, 1),
            "tree_size": st.tree_size,
            "ms_per_token_decode_only": st.ms_per_token,
            "ms_per_token_wall": st.ms_per_token_wall,
            "prefill_ms": st.prefill_ms,
            "ar_ms_per_token": ar_ms_per_token,
            "spec_total_ms": st.ms_per_token_wall * st.n_tokens,
            "ar_total_ms": ar_ms_per_token * len(ar_tok),
            "ar_cached": ar_cached,
            "ends_with_eos": bool(sp_tok and sp_tok[-1] == eos),
            "hit_cap": len(sp_tok) >= a.max_new,
            "ar_hit_cap": len(ar_tok) >= a.max_new,
            "token_exact": exact,
            "first_mismatch": next((k for k, (x, y) in enumerate(zip(ar_tok, sp_tok))
                                    if x != y), None),
        }
        # decode-only totals and the prefill-inclusive end-to-end total, kept
        # separate so a decode-only win is never reported as an end-to-end win.
        rec["spec_end_to_end_ms"] = rec["prefill_ms"] + rec["spec_total_ms"]
        rec["ar_end_to_end_ms"] = ar_prefill_ms + rec["ar_total_ms"]
        rec["decode_speedup"] = (rec["ar_total_ms"] / rec["spec_total_ms"]
                                 if st.n_tokens else float("nan"))
        rec["decode_speedup_wall"] = (
            ar_ms_per_token * len(ar_tok) / max(st.ms_per_token_wall * st.n_tokens, 1e-9))
        out_rows.append(rec)

        if a.repeat_spec > 1 and i < a.repeat_n:
            for _ in range(a.repeat_spec - 1):
                _f2, st2 = glance.generate(head, target, ids, a.max_new, [eos],
                                          budget=a.budget, branch_k=a.branch_k,
                                          temperature=0.0, positions=pos,
                                          return_stats=True, timing="decode_only",
                                          eos_stop="immediate")
                if _f2[0, ids.shape[1]:].tolist() != sp_tok:
                    raise RuntimeError(f"{r['pid']}: the SPEC repeat is not "
                                       f"deterministic; timing repeats are void")
                repeats.append({"pid": r["pid"], "ms_per_token": st2.ms_per_token,
                                "ms_per_token_wall": st2.ms_per_token_wall,
                                "n_tokens": st2.n_tokens, "rounds": st2.rounds})
        if (i + 1) % 10 == 0 or i == 0:
            avg = statistics.fmean(x["spec_total_ms"] for x in out_rows)
            print(f"[eval] {i+1}/{len(rows)} {r['pid']} exact={exact} "
                  f"n={st.n_tokens} tau={st.tau:.2f} "
                  f"{st.ms_per_token:.1f}ms/tok avg {avg:.0f}ms "
                  f"eta {(len(rows)-i-1)*avg/1000/60:.1f}min", flush=True)
        del ids, pix, g, pos, final
        torch.cuda.empty_cache()

    load_end = gpu_snapshot()
    taus = [x["tau"] for x in out_rows]
    n_exact = sum(x["token_exact"] for x in out_rows)
    groups = {}
    for x in out_rows:
        groups.setdefault(x["image_group_id"], []).append(x["tau"])
    group_tau = {k: float(np.mean(v)) for k, v in groups.items()}
    summary = {
        "tag": a.tag, "split": split_label, "ar_source_split": a.split,
        "head": a.head, "head_dtype": a.head_dtype,
        "target_dtype": "float32", "budget": a.budget, "branch_k": a.branch_k,
        "max_new": a.max_new, "max_pixels": a.max_pixels, "eos_stop": "immediate",
        "n_records": len(out_rows), "n_image_groups": len(groups),
        "n_token_exact": n_exact,
        "lossless": bool(n_exact == len(out_rows)),
        "mean_tau_pooled_rounds": float(sum(x["tau"] * x["spec_rounds"] for x in out_rows) / max(sum(x["spec_rounds"] for x in out_rows), 1)),
        "tau_definition": "historical accepted+1 before termination clipping; effective metric is tokens_per_round",
        "mean_tau_per_record": float(np.mean(taus)),
        "mean_tau_per_image_group": float(np.mean(list(group_tau.values()))),
        "median_tau": float(np.median(taus)),
        "tau_min": float(np.min(taus)), "tau_max": float(np.max(taus)),
        "total_rounds": int(sum(x["spec_rounds"] for x in out_rows)),
        "total_tokens": int(sum(x["n_tokens"] for x in out_rows)),
        "total_tokens_per_round": float(
            sum(x["n_tokens"] for x in out_rows) / max(sum(x["spec_rounds"] for x in out_rows), 1)),
        "mean_tree_size": float(np.mean([x["tree_size"] for x in out_rows])),
        "ms_per_token_decode_only": float(np.mean([x["ms_per_token_decode_only"] for x in out_rows])),
        "ms_per_token_wall": float(np.mean([x["ms_per_token_wall"] for x in out_rows])),
        "prefill_ms": float(np.mean([x["prefill_ms"] for x in out_rows])),
        "ar_ms_per_token": float(np.mean([x["ar_ms_per_token"] for x in out_rows])),
        "mean_decode_speedup": float(np.mean([x["decode_speedup"] for x in out_rows])),
        "mean_decode_speedup_wall": float(np.mean([x["decode_speedup_wall"] for x in out_rows])),
        "mean_spec_end_to_end_ms": float(np.mean([x["spec_end_to_end_ms"] for x in out_rows])),
        "mean_ar_end_to_end_ms": float(np.mean([x["ar_end_to_end_ms"] for x in out_rows])),
        "mean_end_to_end_speedup": float(np.mean(
            [x["ar_end_to_end_ms"] / max(x["spec_end_to_end_ms"], 1e-9) for x in out_rows])),
        "n_natural_eos": sum(x["ends_with_eos"] for x in out_rows),
        "n_cap_hits": sum(x["hit_cap"] for x in out_rows),
        "answer_parsed": sum(x["answer_parse_ok"] for x in out_rows),
        "answer_correct": sum(x["answer_correct"] for x in out_rows),
        "answer_accuracy_all_questions": sum(x["answer_correct"] for x in out_rows) / max(len(out_rows), 1),
        "answer_accuracy_denominator": "parsed_only (legacy field); use answer_accuracy_all_questions for full dataset",
        "answer_accuracy": (sum(x["answer_correct"] for x in out_rows)
                            / max(sum(x["answer_parse_ok"] for x in out_rows), 1)),
        "wall_minutes": (time.time() - t_all) / 60,
        "timing_contaminated": None,
        "gpu_at_start": load_start, "gpu_at_end": load_end,
        "rows": out_rows,
    }
    # Contamination is judged from the OTHER cards too: a foreign job that starts
    # mid-run changes the clocks this run reports, and the honest response is to
    # label the timing rather than to keep a clean-looking number.
    summary["timing_contaminated"] = bool(load_start != load_end)
    summary["timing_repeats"] = repeats
    if repeats:
        a_vals = [x["ms_per_token"] for x in repeats]
        summary["timing_repeat_spread_frac"] = float(
            (max(a_vals) - min(a_vals)) / max(min(a_vals), 1e-9))

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=2)

    pq_path = a.per_question or (str(a.out).rsplit(".", 1)[0] + "_per_question.csv")
    cols = [k for k in out_rows[0]]  # every recorded column, `tag` included
    with open(pq_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    sea = a.spec_equals_ar or (str(a.out).rsplit(".", 1)[0] + "_spec_equals_ar.json")
    mismatches = [x for x in out_rows if not x["token_exact"]]
    json.dump({
        "tag": a.tag, "split": split_label, "budget": a.budget,
        "branch_k": a.branch_k, "max_new": a.max_new, "eos_stop": "immediate",
        "n_records": len(out_rows), "n_token_exact": n_exact,
        "all_token_exact": n_exact == len(out_rows),
        "n_mismatch": len(mismatches),
        "mismatch_detail": [{"pid": x["pid"], "first_mismatch": x["first_mismatch"],
                             "n_tokens": x["n_tokens"], "ar_n_tokens": x["ar_n_tokens"]}
                            for x in mismatches[:20]],
        "note": ("compares committed token ids, not decoded text; the oracle is the "
                 "independent greedy AR on the same frozen prompt/image and includes "
                 "EOS, bonus EOS, root EOS and the max_new_tokens cap"),
    }, open(sea, "w"), indent=2)

    lm = a.load_manifest or (str(a.out).rsplit(".", 1)[0] + "_latency_load.json")
    json.dump({"tag": a.tag, "split": split_label, "gpu_requested": a.gpu,
               "gpu_at_start": load_start, "gpu_at_end": load_end,
               "timing_contaminated": summary["timing_contaminated"],
               "same_gpu_precision_and_cache": True,
               "repeat_spec": a.repeat_spec, "repeat_n": a.repeat_n},
              open(lm, "w"), indent=2)

    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("rows", "gpu_at_start", "gpu_at_end",
                                   "timing_repeats")}, indent=2), flush=True)
    print("wrote", a.out, pq_path, sea, lm, flush=True)
    return 0 if n_exact == len(out_rows) else 5


if __name__ == "__main__":
    sys.exit(main())
