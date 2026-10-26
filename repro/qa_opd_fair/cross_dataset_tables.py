#!/usr/bin/env python3
"""Build the cross-dataset tables from whatever the evaluator has written.

WHY this is separate from `aggregate_eval.py`: that tool consumes the
`per_question.csv` tables, which the evaluator writes only after every head of a
stage finishes. A long matrix on a shared card can be split across invocations or
interrupted between heads, and the per-decode `per_question.jsonl` is the record
that always exists. This script reads the jsonl (deduplicating by
`(arm, video, question)`, keeping the first occurrence so repeated attempts
cannot double count), computes per-arm and paired per-video statistics, and
writes both a CSV per stage and a markdown block ready for the report.

Usage:
  python cross_dataset_tables.py --run <run> --stages main_msvd_b,main_msrvtt_b,main_next_b
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

METRICS = ["tau", "rounds", "n_tokens", "decode_ms_token", "model_wall_ms"]
BOOT = 2000
SEED = 20261013


def load_stage(R, name):
    p = R / "eval" / name / "per_question.jsonl"
    if not p.exists():
        return None
    seen = {}
    attempted = collections.Counter()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["arm"], r["video_id"], str(r["question_id"]))
        attempted[key] += 1
        seen.setdefault(key, r)
    return {"rows": list(seen.values()), "unique": len(seen),
            "lines": sum(attempted.values()),
            "repeats": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in attempted.items() if v > 1}}


def per_video(rows, metric):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["arm"], r["video_id"])].append(float(r[metric]))
    out = collections.defaultdict(dict)
    for (arm, vid), vals in g.items():
        out[arm][vid] = float(np.mean(vals))
    return out


def paired(a, b, n=BOOT, seed=SEED):
    vids = sorted(set(a) & set(b))
    if not vids:
        return None
    d = np.array([a[v] - b[v] for v in vids])
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(len(d), size=(n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "positive": int((d > 0).sum()),
            "negative": int((d < 0).sum()), "unchanged": int((d == 0).sum()),
            "n_videos": len(vids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stages", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = pathlib.Path(a.run)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    summary, md = {}, []
    md.append("## 跨数据集主表（每数据集 B 格式，fp32，legacy 时间轴，`eos_stop=immediate`）")
    md.append("")
    for name in stages:
        st = load_stage(R, name)
        if st is None or not st["rows"]:
            md += [f"### {name}", "", "- not run", ""]
            continue
        rows = st["rows"]
        arms = sorted({r["arm"] for r in rows})
        pv = {m: per_video(rows, m) for m in METRICS}
        md += [f"### {name}", "",
               f"（{len(rows)} 条 QA 记录，去重后 {st['unique']} 条；"
               f"重复打印 {len(st['repeats'])} 个键，全部为同一冻结输入的多次尝试）", "",
               "| arm | QA | 视频 | mean τ | 无损 | 轮数 | decode-only ms/token | model wall ms/request |",
               "|---|---:|---:|---:|---|---:|---:|---:|"]
        for arm in sorted(arms, key=lambda x: -np.mean(list(pv["tau"][x].values()))):
            rs = [r for r in rows if r["arm"] == arm]
            ll = sum(1 for r in rs if r["lossless_spec_eq_ar"] in (True, "True"))
            md.append(f"| {arm} | {len(rs)} | {len(pv['tau'][arm])} | "
                      f"{np.mean(list(pv['tau'][arm].values())):.4f} | {ll}/{len(rs)} | "
                      f"{np.mean([float(r['rounds']) for r in rs]):.2f} | "
                      f"{np.mean([float(r['decode_ms_token']) for r in rs]):.2f} | "
                      f"{np.mean([float(r['model_wall_ms']) for r in rs]):.0f} |")
        md.append("")
        md += ["| 比较 | Δτ | 95% CI（视频级 bootstrap 2000） | 正/负/不变 | n |",
               "|---|---:|---|---|---:|"]
        pairs = []
        opds = [x for x in arms if x.startswith("QA-OPD")]
        sfts = [x for x in arms if x.startswith("QA-SFT")]
        for opd in opds:
            tag = opd.split("-")[-1]
            for other, lab in ((f"QA-SFT-{tag}", f"{opd} − QA-SFT-{tag}"),
                               ("C0", f"{opd} − C0")):
                if other in pv["tau"]:
                    p = paired(pv["tau"][opd], pv["tau"][other])
                    if p:
                        lab = lab.replace("QA-SFT-s1-120", "QA-SFT-s1").replace("QA-SFT-s2-120", "QA-SFT-s2")
                        md.append(f"| {lab} | {p['mean']:+.4f} | "
                                  f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] | "
                                  f"{p['positive']}/{p['negative']}/{p['unchanged']} | {p['n_videos']} |")
                        pairs.append({"stage": name, "comparison": lab, **p})
        for sft in sfts:
            if "C0" in pv["tau"]:
                p = paired(pv["tau"][sft], pv["tau"]["C0"])
                if p:
                    lab = f"{sft} − C0".replace("QA-SFT-s1-120", "QA-SFT-s1").replace("QA-SFT-s2-120", "QA-SFT-s2")
                    md.append(f"| {lab} | {p['mean']:+.4f} | "
                              f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] | "
                              f"{p['positive']}/{p['negative']}/{p['unchanged']} | {p['n_videos']} |")
                    pairs.append({"stage": name, "comparison": lab, **p})
        md.append("")
        summary[name] = {
            "n_rows": len(rows), "n_unique": st["unique"],
            "arms": {arm: {"n_qa": sum(1 for r in rows if r["arm"] == arm),
                           "n_videos": len(pv["tau"][arm]),
                           "mean_tau": float(np.mean(list(pv["tau"][arm].values()))),
                           "lossless": sum(1 for r in rows if r["arm"] == arm
                                           and r["lossless_spec_eq_ar"] in (True, "True"))}
                     for arm in arms},
            "paired_tau": pairs,
        }
        outdir = R / "eval" / name
        with (outdir / "cross_dataset_summary.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "n_qa", "n_videos", "mean_tau", "lossless"])
            for arm in arms:
                d = summary[name]["arms"][arm]
                w.writerow([arm, d["n_qa"], d["n_videos"], round(d["mean_tau"], 4),
                            d["lossless"]])
    out = pathlib.Path(a.out) if a.out else R / "reports/cross_dataset_tables.md"
    out.write_text("\n".join(md) + "\n")
    (R / "eval/cross_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n".join(md))


if __name__ == "__main__":
    main()
