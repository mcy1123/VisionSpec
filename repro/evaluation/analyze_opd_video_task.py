#!/usr/bin/env python3
"""Phase 6: the detailed analysis behind the video-OPD report.

Reads the per-row eval JSONs produced by `eval_video_opd.py` for the three arms
(zero-shot image head / video SFT / OPD-best) and prints sections 6.1-6.6 plus the
paired statistics they need. Everything is per-row paired on the SAME 30 held-out
videos; nothing is pooled across rows without saying so.

Definitions used (stated here because they are not universal):
  * tau          mean accepted length per round over the whole row (decode only).
  * tau60        the same mean restricted to the FIRST 60 rounds -- the "60-round"
                 criterion the image-side reports flip-checked against.
  * Delta        arm - SFT, per row, paired.
  * coverage     fraction of the generated response reached after 60 draft
                 rounds (image-side definition, `多模态OPD诊断2_2026-09-24.md` 3.1).
  * anchor frac  share of draft rounds whose block start falls in the first /
                 middle / last third of the generated response, and the share in
                 the first 32 generated positions.
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
import json
import math
import os
import statistics as st


# ---------------------------------------------------------------- statistics

def paired(d):
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    m = st.fmean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    return m, sd, t, n, sum(1 for x in d if x > 0)


def welch(x, y):
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mx, my = st.fmean(x), st.fmean(y)
    vx, vy = st.variance(x), st.variance(y)
    se = math.sqrt(vx / len(x) + vy / len(y))
    return (mx - my) / se if se > 0 else float("nan")


def _rank(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def quantiles(x, qs=(0.25, 0.5, 0.75)):
    s = sorted(x)
    out = []
    for q in qs:
        if len(s) == 1:
            out.append(s[0])
        else:
            pos = q * (len(s) - 1)
            lo = int(math.floor(pos))
            hi = min(lo + 1, len(s) - 1)
            out.append(s[lo] + (s[hi] - s[lo]) * (pos - lo))
    return out


def tau60(row, k=60):
    acc = row.get("accepted") or []
    if not acc:
        return float("nan")
    use = acc[:k]
    return st.fmean(a + 1 for a in use)


def load(path):
    d = json.load(open(path))
    # All 30 rows are kept for every tau/delta statistic: the protocol's warm-up
    # drop is a TIMING convention (first-sample GPU state), not a reason to throw
    # a row out of a paired tau comparison. `warmup` is read where it matters.
    return {r["video"]: r for r in d["rows"]}, d["config"]


def tertiles(vals):
    q1, q2 = quantiles(vals, (1 / 3, 2 / 3))
    return q1, q2


def bucket_report(name, groups, sft, opd, out):
    out.append(f"\n**{name}**\n")
    out.append("| bucket | n | Δ mean | Δ median | t | +/n | Welch t vs rest |")
    out.append("|---|---|---|---|---|---|---|")
    allkeys = sorted(set().union(*[set(g[1]) for g in groups])) if groups else []
    for label, keys in groups:
        if not keys:
            out.append(f"| {label} | 0 | – | – | – | – | – |")
            continue
        d = [opd[k]["tau"] - sft[k]["tau"] for k in keys]
        m, sd, t, n, pos = paired(d)
        rest = [k for k in allkeys if k not in set(keys)]
        wt = welch(d, [opd[k]["tau"] - sft[k]["tau"] for k in rest]) if rest else float("nan")
        out.append(f"| {label} | {n} | {m:+.4f} | {st.median(d):+.4f} | {t:+.2f} "
                   f"| {pos}/{n} | {wt:+.2f} |")
    return out


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--opd", required=True)
    ap.add_argument("--heldout-parquet", required=True)
    ap.add_argument("--opd-pool", default=None, help=".pt used for OPD training")
    ap.add_argument("--old-pool", default=None, help="zero-shot-rolled pool, for anchor drift")
    ap.add_argument("--train-parquet", default=None)
    ap.add_argument("--video-root", default="/mnt/local2/mcy/datasets/VideoDetailCaption/Test_Videos")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    zero, zcfg = load(a.zero)
    sft, fcfg = load(a.sft)
    opd, ocfg = load(a.opd)
    keys = sorted(set(zero) & set(sft) & set(opd))
    assert len(keys) == 30, f"expected 30 held-out rows, got {len(keys)}"
    print(f"arms: zero={zcfg['head']}\n      sft={fcfg['head']}\n      opd={ocfg['head']}")
    print(f"paired rows: {len(keys)}")

    out = []
    out.append("# Phase 6 analysis (auto-generated)\n")
    out.append(f"arms: zero `{zcfg['head']}` | sft `{fcfg['head']}` | opd `{ocfg['head']}`\n")

    # ---------------------------------------------------------------- 6.1
    d = [opd[k]["tau"] - sft[k]["tau"] for k in keys]
    m, sd, t, n, pos = paired(d)
    q25, med, q75 = quantiles(d)
    out.append("## 6.1 per-row Δ = OPD − SFT\n")
    out.append(f"- n={n}, positive {pos}, negative {n-pos-sum(1 for x in d if x==0)}, "
               f"zero {sum(1 for x in d if x == 0)}")
    out.append(f"- mean {m:+.4f}, median {med:+.4f}, p25 {q25:+.4f}, p75 {q75:+.4f}, "
               f"min {min(d):+.4f}, max {max(d):+.4f}, sd {sd:.4f}")
    out.append(f"- paired t {t:+.2f} (df {n-1})")
    out.append("\n| video | zero τ | SFT τ | OPD τ | Δ | τ60 zero/SFT/OPD | cov60 SFT/OPD |")
    out.append("|---|---|---|---|---|---|---|")
    for k in sorted(keys, key=lambda k: opd[k]["tau"] - sft[k]["tau"]):
        out.append(f"| {k} | {zero[k]['tau']:.3f} | {sft[k]['tau']:.3f} | "
                   f"{opd[k]['tau']:.3f} | {opd[k]['tau']-sft[k]['tau']:+.3f} | "
                   f"{tau60(zero[k]):.2f}/{tau60(sft[k]):.2f}/{tau60(opd[k]):.2f} | "
                   f"{sft[k].get('cov60', float('nan')):.2f}/{opd[k].get('cov60', float('nan')):.2f} |")

    # ---------------------------------------------------------------- 6.2
    import pandas as pd
    ho = pd.read_parquet(a.heldout_parquet).set_index("video_name")

    # Source duration is only used for a bucket, so its absence is handled
    # explicitly (and reported) rather than by swallowing a decode error.
    durations, srcres, dur_src = {}, {}, "n/a"
    if not os.path.isdir(a.video_root):
        dur_src = f"source root absent ({a.video_root}); bucket skipped"
    else:
        import av
        miss = []
        for k in keys:
            p = None
            for cand in (k, k + ".mp4", k + ".mkv", k + ".webm"):
                cand_p = os.path.join(a.video_root, cand)
                if os.path.exists(cand_p):
                    p = cand_p
                    break
            if p is None:
                miss.append(k)
                continue
            with av.open(p) as c:
                durations[k] = float(c.duration) / av.time_base
                vs = c.streams.video[0]
                srcres[k] = (int(vs.codec_context.width),
                             int(vs.codec_context.height))
        dur_src = f"source duration via PyAV ({len(durations)}/{len(keys)} rows" + \
                  (f", MISSING {miss})" if miss else ")")

    out.append("\n## 6.2 buckets\n")
    if durations:
        q1, q2 = tertiles([durations[k] for k in durations])
        groups = [("short (<%.1fs)" % q1, [k for k in keys if k in durations and durations[k] < q1]),
                  ("mid", [k for k in keys if k in durations and q1 <= durations[k] < q2]),
                  ("long (≥%.1fs)" % q2, [k for k in keys if k in durations and durations[k] >= q2])]
        out.append(f"\nsource duration: {dur_src}\n")
        bucket_report("by source video duration", groups, sft, opd, out)
    else:
        out.append(f"\nsource duration: {dur_src} -- NOT AVAILABLE, bucket skipped\n")

    gl = [sft[k]["gen_tokens"] for k in keys]
    q1, q2 = tertiles(gl)
    groups = [("short EOS (<%d tok)" % q1, [k for k in keys if sft[k]["gen_tokens"] < q1]),
              ("mid EOS", [k for k in keys if q1 <= sft[k]["gen_tokens"] < q2]),
              ("long EOS (≥%d tok)" % q2, [k for k in keys if sft[k]["gen_tokens"] >= q2])]
    bucket_report("by generated length (SFT arm)", groups, sft, opd, out)

    out.append("\nprompt type: the video pool carries ONE prompt for all 499 rows "
               f"({ho['question'].astype(str).unique()[0][:70]!r}...), so a prompt-type "
               "bucket is not defined. Multi-prompt expansion was not run "
               "(see the report's data-preparation section).\n")

    # Source resolution, NOT the pool's width/height: the cache normalises every
    # clip to long side 384, so the pool columns describe the cached geometry
    # (they only carry aspect ratio, e.g. 384x216 vs 216x384).
    if srcres:
        res_groups = {}
        for k in keys:
            if k in srcres:
                res_groups.setdefault(srcres[k], []).append(k)
        groups = [(f"{w}x{h}", ks) for (w, h), ks
                  in sorted(res_groups.items(), key=lambda kv: -len(kv[1]))]
        out.append(f"\nsource resolution from the coded stream "
                   f"({len(srcres)}/{len(keys)} rows); the pool's own width/height "
                   f"is the normalised 384-long-side cache geometry\n")
        bucket_report("by source resolution", groups, sft, opd, out)
    else:
        out.append("\nsource resolution: not available, bucket skipped\n")

    # ---------------------------------------------------------------- 6.3
    out.append("\n## 6.3 criterion flip check (full length vs first 60 rounds)\n")
    out.append("| arm | τ med full | τ mean full | τ60 mean | Δfull vs SFT | t | Δ60 vs SFT | t | sign |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for tag, arm in (("zero", zero), ("sft", sft), ("opd", opd)):
        full = [arm[k]["tau"] for k in keys]
        t60 = [tau60(arm[k]) for k in keys]
        if tag == "sft":
            out.append(f"| {tag} | {st.median(full):.4f} | {st.fmean(full):.4f} "
                       f"| {st.fmean(t60):.4f} | – | – | – | – | – |")
            continue
        df = [arm[k]["tau"] - sft[k]["tau"] for k in keys]
        d6 = [tau60(arm[k]) - tau60(sft[k]) for k in keys]
        mf, _, tf, _, _ = paired(df)
        m6, _, t6, _, _ = paired(d6)
        sign = "SAME" if (mf >= 0) == (m6 >= 0) else "**FLIPPED**"
        out.append(f"| {tag} | {st.median(full):.4f} | {st.fmean(full):.4f} "
                   f"| {st.fmean(t60):.4f} | {mf:+.4f} | {tf:+.2f} | {m6:+.4f} "
                   f"| {t6:+.2f} | {sign} |")
    if durations:
        out.append("\nΔ amplitude in each criterion is the row above; a flip is only "
                   "called when the two signs differ.\n")

    # ---------------------------------------------------------------- 6.4
    out.append("\n## 6.4 anchor position distribution (eval-time draft rounds)\n")
    out.append("| arm | front third | mid third | back third | first-32 share | rounds med |")
    out.append("|---|---|---|---|---|---|")
    for tag, arm in (("zero", zero), ("sft", sft), ("opd", opd)):
        fr = [arm[k]["third_frac"] for k in keys if arm[k].get("third_frac")]
        f32 = [arm[k].get("front32_frac", float("nan")) for k in keys]
        rounds = [arm[k]["rounds"] for k in keys]
        out.append(f"| {tag} | {st.fmean(x[0] for x in fr):.3f} | "
                   f"{st.fmean(x[1] for x in fr):.3f} | {st.fmean(x[2] for x in fr):.3f} "
                   f"| {st.fmean(f32):.3f} | {st.median(rounds):.0f} |")

    for label, path, tr in (("OPD training pool (warm-start rollout)", a.opd_pool, a.train_parquet),
                            ("old pool (zero-shot rollout)", a.old_pool, None)):
        if not path or not os.path.exists(path):
            out.append(f"\n{label}: not available\n")
            continue
        import torch
        blob = torch.load(path, weights_only=False)
        recs = blob["records"]
        train_names = None
        if tr:
            train_names = set(pd.read_parquet(tr)["video_name"].astype(str))
        subset = [r for r in recs if not train_names or r["task"] in train_names]
        fr, f32 = [], []
        for r in subset:
            gen = r["seq_len"] - r["prompt_len"]
            if gen <= 0 or not r["anchors"]:
                continue
            rel = [x["pos"] - r["prompt_len"] for x in r["anchors"]]
            fr.append([sum(1 for p in rel if p < gen / 3) / len(rel),
                       sum(1 for p in rel if gen / 3 <= p < 2 * gen / 3) / len(rel),
                       sum(1 for p in rel if p >= 2 * gen / 3) / len(rel)])
            f32.append(sum(1 for p in rel if p < 32) / len(rel))
        if fr:
            out.append(f"\n{label}: n={len(fr)} rows  front {st.fmean(x[0] for x in fr):.3f} "
                       f"mid {st.fmean(x[1] for x in fr):.3f} back {st.fmean(x[2] for x in fr):.3f} "
                       f"first-32 {st.fmean(f32):.3f}  anchors/row med "
                       f"{st.median(len(r['anchors']) for r in subset):.0f}")

    # ---------------------------------------------------------------- 6.5
    out.append("\n## 6.5 coverage vs Δ\n")
    out.append("| pair | Spearman ρ | n |")
    out.append("|---|---|---|")
    for tag, arm in (("opd", opd), ("sft", sft)):
        cov = [arm[k].get("cov60", float("nan")) for k in keys]
        dd = [opd[k]["tau"] - sft[k]["tau"] for k in keys]
        out.append(f"| ρ(cov60 {tag}, Δ) | {spearman(cov, dd):+.3f} | {len(keys)} |")
    cov = [sft[k].get("cov60", float("nan")) for k in keys]
    out.append(f"| ρ(cov60 SFT, cov60 OPD) | {spearman(cov, [opd[k].get('cov60', float('nan')) for k in keys]):+.3f} | {len(keys)} |")

    Ls = [sft[k]["prompt_len"] for k in keys]
    medL = st.median(Ls)
    short = [k for k in keys if sft[k]["prompt_len"] <= medL]
    longk = [k for k in keys if sft[k]["prompt_len"] > medL]
    ds = [opd[k]["tau"] - sft[k]["tau"] for k in short]
    dl = [opd[k]["tau"] - sft[k]["tau"] for k in longk]
    ms, _, ts, _, _ = paired(ds)
    ml, _, tl, _, _ = paired(dl)
    out.append(f"\ncontext split at prompt_len median {medL:.0f} tokens: "
               f"short n={len(short)} Δ {ms:+.4f} (t {ts:+.2f}); "
               f"long n={len(longk)} Δ {ml:+.4f} (t {tl:+.2f}); "
               f"Welch t(short vs long) {welch(ds, dl):+.2f}")

    # ---------------------------------------------------------------- 6.6
    out.append("\n## 6.6 vs the image side\n")
    zt = st.median(zero[k]["tau"] for k in keys)
    ft = st.median(sft[k]["tau"] for k in keys)
    ot = st.median(opd[k]["tau"] for k in keys)
    out.append("| | image SQA | image ShareGPT4V | video (this task) |")
    out.append("|---|---|---|---|")
    out.append(f"| SFT τ | 2.43 | 1.04 | {ft:.2f} |")
    out.append(f"| zero-shot τ | – | – | {zt:.2f} |")
    out.append(f"| OPD Δ vs SFT | +0.24 | +0.10 | {m:+.4f} |")
    out.append("| end-to-end speedup | 1.96× | ? | see the report's speedup table |")
    out.append("\n(image-side numbers are quoted from `多模态OPD实验_2026-09-24.md` / "
               "`SPEEDUP_实测_2026-09-23.md`; the video column is measured here.)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out[:60]))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
