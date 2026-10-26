#!/usr/bin/env python3
"""Recover per-question rows for an evaluation stage from its stdout log.

WHY this exists. A stage's `per_question.csv` is written only after every head
finishes, and the retry loops delete a stage directory whose attempt failed.
When a run is killed or pre-empted near the end, the only surviving record is the
one line per decode the evaluator printed. Those lines carry the fields the
report and the paired statistics need (arm, video, question, format, token count,
tau, rounds, lossless), so a stage whose CSV was lost can still be reconstructed.

PROVENANCE IS MARKED: the recovered CSV is written to `eval/recovered_<stage>/`
with a `provenance.json` that records the source log and the parse counts. It is
NOT written over any evaluator directory, so a recovered table can never be
confused with a freshly evaluated one.

Usage:
  python recover_eval_from_log.py --run <run> --log <stage log> --stage <name> \
      [--time-mode legacy] [--formats B]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aggregate_eval import paired  # noqa: E402

LINE = re.compile(
    r"^(?P<stage>dev|main|confirm)/(?P<time_mode>\w+) (?P<arm>\S+) "
    r"(?P<dataset>\S+) (?P<video_id>\S+) (?P<question_id>\S+) (?P<mode>[AB]) "
    r"n (?P<n_tokens>\d+) tau (?P<tau>[\d.]+) rounds (?P<rounds>\d+) "
    r"lossless (?P<lossless>\w+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--time-mode", default=None)
    ap.add_argument("--formats", default="B")
    a = ap.parse_args()
    R = pathlib.Path(a.run)
    rows = []
    by_key = {}
    conflicts = []
    seen = collections.Counter()
    for raw in pathlib.Path(a.log).read_text().splitlines():
        m = LINE.match(raw.strip())
        if not m:
            continue
        d = m.groupdict()
        if d["stage"] != a.stage:
            continue
        if a.time_mode and d["time_mode"] != a.time_mode:
            continue
        if d["mode"] not in a.formats.split(","):
            continue
        key = (d["arm"], d["video_id"], d["question_id"], d["mode"])
        # A retried attempt re-prints the same decode. The inputs, checkpoint and
        # decode parameters are frozen, so the repeated line must carry the SAME
        # tau; `conflicts` counts the cases where it does not, and the FIRST
        # occurrence is kept so the recovery is deterministic either way.
        seen[key] += 1
        if key in by_key:
            prev = by_key[key]
            if abs(prev["tau"] - float(d["tau"])) > 1e-9 or \
               prev["n_tokens"] != int(d["n_tokens"]) or \
               prev["rounds"] != int(d["rounds"]):
                conflicts.append({"key": list(key), "first": prev["tau"],
                                  "later": float(d["tau"])})
            continue
        by_key[key] = {"dataset": d["dataset"], "video_id": d["video_id"],
                     "question_id": d["question_id"], "mode": d["mode"],
                     "time_mode": d["time_mode"], "arm": d["arm"],
                     "n_tokens": int(d["n_tokens"]), "tau": float(d["tau"]),
                     "rounds": int(d["rounds"]),
                     "lossless_spec_eq_ar": d["lossless"],
                     "decode_ms_token": "", "model_wall_ms": "",
                     "effective_tokens_per_round": "",
                     "ar_n_tokens": "", "tree_size": "", "hit_eos": "",
                     "eos_stop_round": "", "eos_stop_kind": "",
                     "timing_contaminated": "True", "head_path": ""}
    rows = list(by_key.values())
    if not rows:
        raise SystemExit(f"no parsable {a.stage!r} lines in {a.log}")
    out = R / "eval" / f"recovered_{a.stage}"
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "video_id", "question_id", "mode", "time_mode",
                  "arm", "n_tokens", "tau", "rounds", "lossless_spec_eq_ar",
                  "decode_ms_token", "model_wall_ms",
                  "effective_tokens_per_round", "ar_n_tokens", "tree_size",
                  "hit_eos", "eos_stop_round", "eos_stop_kind",
                  "timing_contaminated", "head_path"]
    with (out / "per_question.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # per-video + paired tau, so the recovered stage still yields a usable number
    by_video = collections.defaultdict(list)
    for r in rows:
        by_video[(r["video_id"], r["arm"])].append(r)
    per_video = collections.defaultdict(dict)
    for (vid, arm), rs in by_video.items():
        per_video[arm][vid] = sum(x["tau"] for x in rs) / len(rs)
    arms = sorted({r["arm"] for r in rows})
    with (out / "per_video.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "video_id", "mode", "arm", "n_qa", "tau"])
        for (vid, arm), rs in sorted(by_video.items()):
            w.writerow([rs[0]["dataset"], vid, rs[0]["mode"], arm, len(rs),
                        sum(x["tau"] for x in rs) / len(rs)])
    paired_rows = []
    for opd in [x for x in arms if x.startswith("QA-OPD")]:
        sft = "QA-SFT-" + opd.split("-")[-1]
        for other, label in ((sft, "QA-OPD-QA-SFT"), ("C0", "QA-OPD-C0")):
            if other in per_video:
                p = paired(per_video[opd], per_video[other])
                if p:
                    paired_rows.append({"stage": a.stage, "metric": "tau",
                                        "comparison": label, "arm_a": opd,
                                        "arm_b": other, **p})
    if "QA-SFT" in " ".join(arms):
        for sft in [x for x in arms if x.startswith("QA-SFT")]:
            if "C0" in per_video:
                p = paired(per_video[sft], per_video["C0"])
                if p:
                    paired_rows.append({"stage": a.stage, "metric": "tau",
                                        "comparison": "QA-SFT-C0", "arm_a": sft,
                                        "arm_b": "C0", **p})
    with (out / "paired_tau.csv").open("w", newline="") as fh:
        cols = ["stage", "metric", "comparison", "arm_a", "arm_b", "mean",
                "ci_low", "ci_high", "positive", "negative", "unchanged",
                "n_videos"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in paired_rows:
            w.writerow(r)
    provenance = {
        "recovered_from_log": str(a.log),
        "stage": a.stage, "time_mode": a.time_mode, "formats": a.formats,
        "rows": len(rows), "arms": {arm: sum(1 for r in rows if r["arm"] == arm)
                                    for arm in arms},
        "repeated_lines": sum(1 for v in seen.values() if v > 1),
        "n_repeat_conflicts": len(conflicts),
        "repeat_conflicts": conflicts[:20],
        "fields_lost": ["decode_ms_token", "model_wall_ms", "ar_n_tokens",
                        "tree_size", "hit_eos", "eos_stop_kind"],
        "note": "tau/rounds/n_tokens/lossless come verbatim from the evaluator's "
                "own per-decode print; timing fields are NOT recoverable from the "
                "log and are left blank rather than estimated.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2,
                                                    ensure_ascii=False))
    print(json.dumps(provenance, indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
