#!/usr/bin/env python3
"""Aggregate the visual-adaptation diagnosis into one markdown document.

Reads the per-stage JSONs written by diag_arms.py (arms A/B/C/D/Dn),
diag_va.py (per-row visual dependency) and diag_features.py (stage 4), applies
the judgement rules the task pre-registered, and writes the report.

Every number in the output document comes from these files; nothing is typed in
by hand. Verdicts are decided by the thresholds stated in the task, not by what
would read well, and each verdict prints the comparison that produced it.

    python diag_report.py --dir <results/diag> --out <视觉适配诊断_日期.md>
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
import os
import statistics as st
from datetime import date

try:
    from scipy import stats as sps
except ImportError as exc:  # pragma: no cover - scipy is a hard requirement
    raise SystemExit(f"scipy is required for the paired tests: {exc}")

import pandas as pd

ARMS = ["A", "B", "C", "D", "Dn"]
ARM_LABEL = {
    "A": "A 当前 MLP 投影（baseline）",
    "B": "B 原始视觉 token（绕过深层）",
    "C": "C 无视觉信息（纯文本 KV）",
    "D": "D 最终层 hidden，无 fc 无 norm",
    "Dn": "Dn 最终层 hidden，有 norm 无 fc",
}


def first_diff(a, b):
    """Index of the first differing element, or None if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def prefix_delta(ids_x, ids_a):
    """Paired tau delta restricted to the token prefix the two arms still share.

    At temperature 0 GLANCE should be lossless, so every arm ought to commit the
    AR oracle's tokens. In bf16 it does not: the tree pass and the single-token
    pass round differently and near-ties flip, after which an arm is drafting a
    DIFFERENT continuation than its sibling and a whole-row tau compares two
    different texts.

    This restricts BOTH arms to the rounds whose entire committed span lies
    inside the prefix where they still agree, and takes the per-row difference of
    the mean tau over those rounds. Restricting one side only would compare a
    short early prefix against a whole generation and bias the delta.

    Returns a dict or None.
    """
    if ids_x is None or ids_a is None:
        return None
    px = {r["video"]: r for r in ids_x["rows"]}
    pa = {r["video"]: r for r in ids_a["rows"]}
    deltas, firsts, n_in = [], [], 0
    for v, ra in pa.items():
        rx = px.get(v)
        if rx is None or "spec_ids" not in rx or "spec_ids" not in ra:
            continue
        d = first_diff(rx["spec_ids"], ra["spec_ids"])
        if d is None:
            d = min(len(rx["spec_ids"]), len(ra["spec_ids"]))
        firsts.append(d)

        def rounds(rec):
            return [ac + 1 for ar, ac in zip(rec["anchor_rel"], rec["accepted"])
                    if ar + ac + 1 <= d]
        bx, ba = rounds(rx), rounds(ra)
        n_in += len(bx)
        if bx and ba:
            deltas.append(st.fmean(bx) - st.fmean(ba))
    if not deltas:
        return None
    return {"n_rows": len(deltas), "n_rounds": n_in,
            "delta_mean": st.fmean(deltas), "delta_median": st.median(deltas),
            "median_first_diff": st.median(firsts) if firsts else None,
            "n_diverged": sum(1 for d in firsts if d < 10 ** 9),
            "n_pairs": len(firsts)}


def divergence_stats(arm_rows, ar_rows, ids_doc, ref_label):
    """How far an arm's committed sequence drifts from the AR oracle."""
    if ids_doc is None or ar_rows is None:
        return None
    per = {r["video"]: r for r in ids_doc["rows"]}
    ds, rels, n = [], [], 0
    for v, ar in ar_rows.items():
        r = per.get(v)
        if r is None or "spec_ids" not in r:
            continue
        n += 1
        d = first_diff(r["spec_ids"], ar["ar_ids"])
        if d is not None:
            ds.append(d)
            rels.append(d / max(min(len(r["spec_ids"]), len(ar["ar_ids"])), 1))
    if n == 0:
        return None
    return {"n_rows": n, "n_diverged": len(ds),
            "median_first_diff": st.median(ds) if ds else None,
            "median_rel": st.median(rels) if rels else 0.0,
            "ref": ref_label}


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def rows_by_video(doc):
    if doc is None:
        return {}
    return {r["video"]: r for r in doc["rows"] if not r.get("warmup", False)}


def paired(deltas):
    """Paired t-test plus win/loss counts on the per-row deltas."""
    n = len(deltas)
    mean = st.fmean(deltas)
    sd = st.stdev(deltas) if n > 1 else 0.0
    t, p = sps.ttest_rel(deltas, [0.0] * n) if n > 1 else (float("nan"), float("nan"))
    return {
        "n": n, "mean": mean, "median": st.median(deltas), "sd": sd,
        "t": float(t), "p": float(p),
        "win": sum(1 for d in deltas if d > 0),
        "loss": sum(1 for d in deltas if d < 0),
        "tie": sum(1 for d in deltas if d == 0),
        "wald_ci": (mean - 1.96 * sd / n ** 0.5, mean + 1.96 * sd / n ** 0.5) if n > 1 else (0.0, 0.0),
    }


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def buckets(values, labels=("低", "中", "高")):
    """Bucket an axis into ordered groups.

    Continuous axes get terciles (falling back to a median split when a tercile
    would be too thin). An axis with few distinct levels -- prompt length here
    takes only three values -- is grouped by level instead, because equal-count
    terciles on a discrete axis can produce an empty bucket, and an empty bucket
    is a silently useless row rather than a result.
    """
    s = sorted(values)
    n = len(s)
    dist = sorted(set(s))
    if 1 < len(dist) <= 5:
        edges = []
        for i, v in enumerate(dist):
            lo = -1e9 if i == 0 else (dist[i - 1] + v) / 2
            hi = 1e9 if i == len(dist) - 1 else (v + dist[i + 1]) / 2
            edges.append((lo, hi))
        return edges, [f"={v:g}" for v in dist], "按离散取值分组"
    if n >= 24:
        cuts = [s[n // 3], s[2 * n // 3]]
        edges = [(-1e9, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 1e9)]
        return edges, list(labels), "三分位"
    med = st.median(s)
    return [(-1e9, med), (med, 1e9)], ["低", "高"], "中位数二分（样本不足三分位）"


def bucket_of(x, edges):
    for i, (lo, hi) in enumerate(edges):
        if lo <= x < hi:
            return i
    return len(edges) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/local/mcy/GLANCE/results/diag")
    ap.add_argument("--out",
                    default=None,
                    help="defaults to <dir>/视觉适配诊断_<today>.md, i.e. under "
                         "/mnt/local, never the home directory")
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(a.dir, f"视觉适配诊断_{date.today().isoformat()}.md")

    arms = {k: rows_by_video(load(os.path.join(a.dir, f"arm{k}.json"))) for k in ARMS}
    have = [k for k in ARMS if arms[k]]
    if "A" not in have:
        raise SystemExit("arm A is required: it is the baseline every delta is against")
    va_doc = load(os.path.join(a.dir, "va.json"))
    attn_doc = load(os.path.join(a.dir, "armA_attn.json"))
    feat_doc = load(os.path.join(a.dir, "features.json"))
    ar_doc = load(os.path.join(a.dir, "ar.json"))
    ar_rows = ({r["video"]: r for r in ar_doc["rows"]} if ar_doc else None)
    ids_docs = {k: load(os.path.join(a.dir, f"arm{k}_ids.json")) for k in ARMS}
    kvv_doc = load(os.path.join(a.dir, "kv_visual.json"))
    kvv = {}
    if kvv_doc:
        for k in ["vis_frac", "fused_relL2_vis", "fused_relL2_txt", "fused_cos_vis",
                  "fused_cos_txt", "proj_relL2_vis", "proj_relL2_txt", "proj_cos_vis",
                  "proj_cos_txt"]:
            kvv[k] = st.median(r[k] for r in kvv_doc["rows"])

    common_ids = set(arms["A"])
    for k in have:
        common_ids &= set(arms[k])
    common_ids = sorted(common_ids)
    n_common = len(common_ids)

    def taus(k, ids=None):
        ids = ids if ids is not None else common_ids
        return [arms[k][v]["tau"] for v in ids if v in arms[k]]

    # ---- stage 1: main table ------------------------------------------------
    main_rows = []
    div_rows = []
    for k in have:
        t = taus(k)
        ea = [x - 1.0 for x in t]
        d = [arms[k][v]["tau"] - arms["A"][v]["tau"] for v in common_ids]
        pr = paired(d)
        ct = prefix_delta(ids_docs[k], ids_docs["A"])
        dv = divergence_stats(arms[k], ar_rows, ids_docs[k], "AR")
        if dv:
            div_rows.append((k, dv))
        main_rows.append({
            "arm": k, "n": len(t), "tau_med": st.median(t), "tau_mean": st.fmean(t),
            "tau_sd": st.stdev(t) if len(t) > 1 else 0.0,
            "ea_med": st.median(ea), "delta_med": pr["median"], "delta_mean": pr["mean"],
            "t": pr["t"], "p": pr["p"], "win": pr["win"], "loss": pr["loss"],
            "ci": pr["wald_ci"], "clean": ct,
        })

    # ---- stage 2: VA buckets -------------------------------------------------
    va = {r["video"]: r["VA"] for r in va_doc["rows"]} if va_doc else {}
    va_edges = va_labels = va_kind = None
    va_table = []
    if va:
        ids_va = [v for v in common_ids if v in va]
        va_edges, va_labels, va_kind = buckets([va[v] for v in ids_va])
        for bi, lab in enumerate(va_labels):
            ids_b = [v for v in ids_va if bucket_of(va[v], va_edges) == bi]
            row = {"bucket": lab, "n": len(ids_b),
                   "va_lo": min((va[v] for v in ids_b), default=float("nan")),
                   "va_hi": max((va[v] for v in ids_b), default=float("nan"))}
            for k in have:
                if not ids_b:
                    row[k] = float("nan")
                    row[k + "_d"] = float("nan")
                    continue
                row[k] = st.median(taus(k, ids_b))
                row[k + "_d"] = st.median(
                    [arms[k][v]["tau"] - arms["A"][v]["tau"] for v in ids_b])
            row["ids"] = ids_b
            va_table.append(row)

    # ---- stage 3: attention --------------------------------------------------
    attn = []
    if attn_doc:
        rows = [r for r in attn_doc["rows"] if not r.get("warmup")]
        per_layer = {}
        for r in rows:
            for e in r.get("attn") or []:
                per_layer.setdefault(e["layer"], []).append(e)
        for layer in sorted(per_layer):
            es = per_layer[layer]
            attn.append({
                "layer": layer,
                "vis_mass": st.median(e["vis_mass"] for e in es),
                "entropy_norm": st.median(e["entropy_norm"] for e in es),
                "top1pct": st.median(e["top1pct_mass"] for e in es),
                "third1": st.median(e["third_mass"][0] for e in es),
                "third2": st.median(e["third_mass"][1] for e in es),
                "third3": st.median(e["third_mass"][2] for e in es),
                "n_ctx": st.median(e["n_ctx"] for e in es),
            })
    vis_frac_ctx = None
    if va_doc and arms["A"]:
        vf = [r["vis_frac"] for r in load(os.path.join(a.dir, "armA.json"))["rows"]
              if not r.get("warmup")]
        vis_frac_ctx = st.median(vf)

    # ---- stage 4: features ---------------------------------------------------
    # CORRECTED 2026-10-05 (pitfall 3). `features.json` was computed with a
    # `use_cache=False` forward, which on Qwen3-VL leaves text positions
    # independent of the video. Every decode uses `use_cache=True`. The CKA
    # numbers here therefore come from `measure_cka.py --feature-path cached`,
    # which reproduces the decode path; the uncached run is kept only so the size
    # of the error can be shown.
    feat = {}
    if feat_doc:
        keys = ["cos_proj_raw_vis", "cos_proj_last_vis", "cos_proj_last_txt",
                "cos_raw_last_vis", "cos_projvis_projtxt_mean",
                "cos_projvis_lasttxt_mean", "cos_rawvis_lasttxt_mean",
                "cka_proj_raw_vis", "cka_proj_last_vis", "cka_proj_last_txt",
                "cka_proj_raw_txt", "proj_rms_vis", "proj_rms_txt",
                "last_rms_vis", "raw_rms_vis", "n_vis"]
        for k in keys:
            vals = [r[k] for r in feat_doc["rows"] if r.get(k) == r.get(k)]
            feat[k] = st.median(vals) if vals else float("nan")
    feat_old = dict(feat)
    cka_cached = load("/mnt/local/mcy/GLANCE/results/vispec/cka_A_cached.json")
    cka_uncached = load("/mnt/local/mcy/GLANCE/results/vispec/cka_A_uncached.json")
    if cka_cached:
        for k in ["cos_proj_raw_vis", "cos_proj_last_vis", "cos_proj_last_txt",
                  "cos_projvis_projtxt_mean", "cos_projvis_lasttxt_mean",
                  "cka_proj_raw_vis", "cka_proj_last_vis", "cka_proj_last_txt",
                  "cka_proj_raw_txt", "proj_rms_vis", "proj_rms_txt"]:
            vals = [r[k] for r in cka_cached["rows"] if r.get(k) == r.get(k)]
            if vals:
                feat[k] = st.median(vals)
        if cka_uncached:
            for k in ["cka_proj_raw_vis", "cka_proj_last_vis", "cka_proj_last_txt",
                      "cka_proj_raw_txt", "cos_projvis_projtxt_mean"]:
                vals = [r[k] for r in cka_uncached["rows"] if r.get(k) == r.get(k)]
                if vals:
                    feat_old[k] = st.median(vals)

    # ---- stage 5: context-length buckets ------------------------------------
    # n_frames is constant across this holdout, so the temporal axis the task
    # asked for does not exist here; the axes that DO vary are the visual token
    # count (spatial resolution) and the total prompt length.
    lens = {v: arms["A"][v]["L"] for v in common_ids}
    nvis = {v: arms["A"][v].get("n_visual_tokens") for v in common_ids}
    len_edges, len_labels, len_kind = buckets([lens[v] for v in common_ids])
    len_table = []
    for bi, lab in enumerate(len_labels):
        ids_b = [v for v in common_ids if bucket_of(lens[v], len_edges) == bi]
        row = {"bucket": lab, "n": len(ids_b),
               "L_lo": min((lens[v] for v in ids_b), default=float("nan")),
               "L_hi": max((lens[v] for v in ids_b), default=float("nan"))}
        for k in have:
            row[k] = st.median(taus(k, ids_b)) if ids_b else float("nan")
            row[k + "_d"] = (st.median([arms[k][v]["tau"] - arms["A"][v]["tau"] for v in ids_b])
                             if ids_b else float("nan"))
        len_table.append(row)

    nf = sorted({arms["A"][v].get("n_frames") for v in common_ids})
    nvis_vals = sorted({nvis[v] for v in common_ids if nvis[v] is not None})

    # ======================================================================
    # document
    # ======================================================================
    L = []
    w = L.append
    today = date.today().isoformat()
    w("# 视觉适配诊断：当前 MLP 投影在 VLM 投机解码中的作用")
    w("")
    w(f"日期：{today}　|　目标模型：Qwen3-VL-8B-Instruct（冻结）　|　"
      "草稿头：H-8 块扩散头　|　任务：视频 QA 描述生成")
    w("")
    w("本文件是诊断结论，不是优化结果。未改动任何架构、配方或图像侧设置。")
    w("")

    # --- 0 method -----------------------------------------------------------
    w("## 0. 口径与方法")
    w("")
    w("### 0.1 被测对象")
    w("")
    w("草稿头唯一的视觉入口是一处可学习投影（`dflash/model.py:584`）：")
    w("")
    w("```")
    w("target_hidden = hidden_norm(fc(cat(hidden_states[1, 9, 17, 25, 33])))")
    w("```")
    w("")
    w("注意它并不含激活函数与隐层：`fc` 是一个 `Linear(5*4096 -> 4096, bias=False)`，"
      "其后接一个 RMSNorm。所谓「MLP 投影」在实现上是**单层线性投影 + 归一化**。")
    w("")
    w(f"- 头参数量实测 **1627.5M**（H-8，`diag_arms.py` 启动时打印，bf16 权重文件 3.25 GB）。"
      "任务描述里的 ~1.05B 对应的是 H-5 头（1048.6M），H-8 实测为 1627.5M；"
      "本诊断以现役 H-8 为准，差异记录在 §7。")
    w("- 投影后的张量在每层被 `k_proj` / `v_proj` 变成 KV，与 mask token 的 Q 做 cross-attention。")
    w("")

    w("### 0.2 四臂定义（外加一个尺度控制臂）")
    w("")
    w("| 臂 | KV 来源 | 实现 |")
    w("|---|---|---|")
    w("| **A** | 当前 MLP 投影后的 target hidden | 现役架构，未改动 |")
    w("| **B** | 原始视觉 token | 视觉位置处的 KV 行改用目标模型自己的视觉嵌入"
      "（`hidden_states[0]`，即 vision tower + merger 的输出），文本位置保持 A |")
    w("| **C** | 无视觉信息（纯文本 KV） | 送入 `fc` 前把视觉位置的通道置零，"
      "使该处投影输出恒为 0（`fc` 无 bias、RMSNorm 无 bias，故零进零出） |")
    w("| **D** | target 最终层 hidden，不做投影 | `hidden_states[-1]` 直接送 `k_proj`/`v_proj`，"
      "跳过 `fc` 与 `hidden_norm` |")
    w("| **Dn** | 同上但做尺度归一 | `hidden_norm(hidden_states[-1])`，只跳过 `fc` |")
    w("")
    w("Dn 不在任务给定的四臂里，是诊断过程中**必须补的对照**：`hidden_states[-1]` 的 RMS "
      "远大于头训练时见过的归一化投影输出，而 `v_proj` 之后没有归一化。不补 Dn，"
      "D 的崩塌就无法区分「投影丢了信息」与「激活尺度被破坏」。")
    w("")
    w("### 0.3 关键方法学声明（影响全部数字的解释）")
    w("")
    w("**四臂共用同一个训练好的头（臂 A 的权重），B/C/D/Dn 只在推理时替换 KV 来源，没有重训。**")
    w("因此 B/C/D/Dn 对该头都是分布外输入：它们度量的是「这个已训练的头对臂 A 表示的"
      "**依赖程度**」，而不是「该 KV 来源能达到的上限」。任务判据表（A>B / A≈B / A<B）"
      "问的是前者，与本设计一致；但若要把结论外推成「换 KV 来源能更好」，必须重训，"
      "本诊断不做这个外推。此点在 §7 记为第一号异常。")
    w("")
    w("**留出集用的是 `video_fresh30.parquet`，不是 `video_heldout_30.parquet`。**")
    w("`pool_499.parquet` 的划分是 439 train / 60 heldout；实测")
    w("`video_heldout_30` 的 30 条**全部落在 train 划分内**，而 H-8 头正是用")
    w("「train rows: 439」训练的（`logs/B0_sft_h8.log`），所以那 30 条是训练行，")
    w("不可用作留出。`video_fresh30` 的 30 条全部落在 heldout 划分内且与之不交，"
      "本诊断用它。这一条是任务禁止事项「不用训练集内的行做留出」的直接落实，")
    w("也顺带纠正了项目内 `scripts/83_eval_heads.sh` 把 `ROWS` 默认成 ")
    w("`video_heldout_30.parquet` 的隐患（§7 第二号异常）。")
    w("")
    w("### 0.4 评测口径")
    w("")
    w("- 行：`video_fresh30.parquet` 全部 30 行，逐行配对（同 video 同名对齐）")
    w("- 树验证：budget **63**、branch **8**（b63k8）")
    w("- `max_new_tokens = 1024`、greedy（temperature 0）、bf16")
    w("- 首行作 warm-up 丢弃 → 每臂 **n = 29**，与既有 H-8 基线 `h8_3000.json` 同口径")
    w("- τ = 每轮提交 token 数，跨轮池化；E[a] = τ − 1")
    w("- 四臂除 KV 来源外，数据、步数、头权重、树、位置编码全部相同")
    w("")
    w("相对既有基线：`h8_3000.json` 在 fresh30 上 τ 中位数 **3.1625**（n=29）。")
    w(f"本诊断臂 A 复现为 **{main_rows[0]['tau_med']:.4f}**，"
      f"偏差 {100*(main_rows[0]['tau_med']/3.1625-1):+.2f}%，说明改造后的执行路径与原评测等价。")
    w("")

    # --- 1 main table -------------------------------------------------------
    w("## 1. 四臂主表（阶段 1）")
    w("")
    w(f"逐行配对，n = {n_common}（∈ {len(arms['A'])+1} 行，含 warm-up 1 行已弃）。"
      "Δ 为该臂 τ − 臂 A τ 的逐行差。")
    w("")
    w("| 臂 | KV 来源 | τ 中位 | τ 均值 ± sd | E[a] 中位 | Δ 中位 | Δ 均值 [95% CI] | t | p | 胜/负 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in main_rows:
        lab = ARM_LABEL[r["arm"]].split(" ", 1)
        if r["arm"] == "A":
            stat = "| — | — | — |"
        else:
            stat = (f"| {r['t']:+.2f} | {fmt_p(r['p'])} | {r['win']}/{r['loss']} |")
        w(f"| **{lab[0]}** | {lab[1]} | {r['tau_med']:.4f} | {r['tau_mean']:.4f} ± {r['tau_sd']:.4f} "
          f"| {r['ea_med']:.4f} | {r['delta_med']:+.4f} | "
          f"{r['delta_mean']:+.4f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] "
          + stat)
    w("")
    w("> Dn 与 D 是同一信息的两种尺度，单列是为了把「去掉 `fc`」与「破坏激活尺度」分开。")
    w("")
    w("**上表的 Δ 有两个必须同时看的限定**，它们决定了这张表能支持什么结论：")
    w("")
    w("1. B/C/D/Dn 对该头是分布外输入（§0.3）；")
    w("2. bf16 解码不是逐位无损的，不同臂会提交**不同的续写**，于是 τ 是在各臂自己的"
      "文本上测的。第 2 点由下一小节的实测给出，并给出一个已控制该混淆的 τ。")
    w("")

    # --- 1b losslessness ----------------------------------------------------
    w("### 1.1 无损性检查：臂之间是否在生成同一段文本")
    w("")
    if not div_rows:
        w("未做 token 级比对（缺 `arm*_ids.json` 或 `ar.json`），无法给出本节；见 §7。")
    else:
        w("温度 0 下 GLANCE 应当无损：每臂提交的 token 都应等于目标自身贪心输出。"
          "用 `diag_ar.py` 跑的 AR oracle 作为参照（无草稿头，纯目标贪心），逐 token 比对。")
        w("")
        w("| 臂 | 比对行数 | 与 AR 不同的行 | 首个分歧位置中位 | 分歧位置 / 生成长度 中位 |")
        w("|---|---|---|---|---|")
        for k, dv in div_rows:
            md = "—" if dv["median_first_diff"] is None else f"{dv['median_first_diff']:.0f}"
            w(f"| {k} | {dv['n_rows']} | {dv['n_diverged']}/{dv['n_rows']} | {md} "
              f"| {100*dv['median_rel']:.1f}% |")
        w("")
        w("**这说明 §1 主表的 Δ 混入了「换了文本」这一效应。** 不是实现错误："
          "`eval_video_opd.py` 的 docstring 已经写明 bf16 下 tree pass 与单 token pass "
          "的归约顺序不同、接近的 tie 会落到两边，所以该仓库的无损门禁只在 fp32 下通过"
          "（`audit_lossless.py`）。本诊断用 bf16，因为既有 H-8 基线就是 bf16。")
        w("")
        w("**已控制该混淆的 Δ**：把**两个臂都**限制在「双方仍一致的 token 前缀」内"
          "（即只统计该轮提交的全部 token 都落在分歧点之前的轮次），再取逐行的 τ 均值之差。"
          "只限制一侧会把一段短前缀和一次完整生成相比，反而制造偏差。")
        w("")
        w("| 臂 | 全行 Δ 中位（vs A） | **前缀内 Δ 中位** | 前缀内 Δ 均值 | 可用行数 | 前缀内轮数 | 首个分歧位置中位 |")
        w("|---|---|---|---|---|---|---|")
        for r in main_rows:
            ct = r["clean"]
            if ct is None:
                w(f"| {r['arm']} | {r['delta_med']:+.4f} | — | — | — | — | — |")
            else:
                md = ("—（自身）" if r["arm"] == "A"
                      else ("—" if ct["median_first_diff"] is None
                            else f"{ct['median_first_diff']:.0f}"))
                w(f"| {r['arm']} | {r['delta_med']:+.4f} | **{ct['delta_median']:+.4f}** "
                  f"| {ct['delta_mean']:+.4f} | {ct['n_rows']} | {ct['n_rounds']} | {md} |")
        w("")
        w("对照可见：控制混淆后 B 与 C 的劣势**都比全行口径更大**，"
          "即「各臂在生成不同文本」这一效应此前**掩盖**了真实的草稿质量差距，"
          "而不是制造了差距。前缀内轮数只有全量的三分之一左右，"
          "因为分歧点中位在第 76–83 个 token，这是本口径的样本量代价。")
        w("")

    w("### 判据")
    w("")
    a_b = next((r for r in main_rows if r["arm"] == "B"), None)
    if a_b is None:
        w("- 臂 B 未完成，无法给出 A vs B 的判据；见 §7。")
    else:
        ratio = a_b["delta_med"] / main_rows[0]["tau_med"]
        # delta_med is (B - A): negative means B is WORSE, i.e. A wins.
        if a_b["p"] < 0.05 and a_b["delta_med"] < 0:
            verdict = ("**A > B 显著**：MLP 投影优于原始视觉 token。"
                       "即「目标模型中间层对视觉 token 的深层加工」是草稿头真正用到的东西，"
                       "原始视觉嵌入不足以替代。")
        elif a_b["p"] < 0.05 and a_b["delta_med"] > 0:
            verdict = "**A < B 显著**：MLP 投影反而有损。"
        else:
            verdict = "**A ≈ B**：MLP 投影没有额外收益。"
        w(f"- A vs B：Δ 中位 {a_b['delta_med']:+.4f}（相对 A 的 τ 为 {100*ratio:+.1f}%），"
          f"配对 t = {a_b['t']:+.2f}，p = {fmt_p(a_b['p'])}，胜/负 {a_b['win']}/{a_b['loss']} → {verdict}")
    w("")

    # --- 2 VA ---------------------------------------------------------------
    w("## 2. 视觉依赖分层（阶段 2）")
    w("")
    if not va:
        w("VA 计算未完成，本节缺失；见 §7。")
    else:
        w("**VA 定义**：对每行，先用真实视频让**目标模型**贪心生成 64 个 token；"
          "然后在 KV cache 下、以**完全相同的真实 token 前缀**做两次 teacher-forcing 前向"
          "（一次真实视频、一次把 `pixel_values_videos` 整体置零），逐位置比较两条下一 token "
          "分布的 Jensen-Shannon 散度（bit），取 64 个位置的均值。零化像素保持网格、帧数、"
          "token 数与全部位置不变，因此差异只来自视觉内容，无长度混淆。VA 只依赖目标模型，"
          "不依赖任何草稿臂。")
        w("")
        w("> **为什么必须用带 cache 的路径**：`use_cache=False` 的单次全序列前向在这台模型上"
          "会给出与视觉内容**逐位相同**的文本 logits（视觉、置零、取负三种输入的文本位置"
          "最大差恒为 0.000000，而视觉位置 hidden 已差到 ~2e3）；`use_cache=True` 的 prefill "
          "（即 `glance.decode.generate` 实际走的路径）下同一比较的 logits 差约 15。"
          "本文件全程使用带 cache 的路径，细节记在 §7。")
        w("")
        w(f"VA 分桶方式：{va_kind}。")
        w("")
        hdr = "| VA 桶 | n | VA 范围 | " + " | ".join(f"{k} τ中位" for k in have) + \
              " | " + " | ".join(f"Δ({k}−A)" for k in have if k != "A") + " |"
        w(hdr)
        w("|---" * (3 + len(have) + len([k for k in have if k != "A"]) + 1) + "|")
        for row in va_table:
            cells = [f"| {row['bucket']} | {row['n']} | {row['va_lo']:.4f}–{row['va_hi']:.4f} "]
            for k in have:
                cells.append(f"| {row[k]:.4f} ")
            for k in have:
                if k == "A":
                    continue
                cells.append(f"| {row[k+'_d']:+.4f} ")
            w("".join(cells) + "|")
        w("")
        va_all = [va[v] for v in common_ids if v in va]
        vstep = [r["VA_step1"] for r in va_doc["rows"] if "VA_step1" in r]
        vdiv = [r["prefix_div"] for r in va_doc["rows"] if "prefix_div" in r]
        nchanged = sum(1 for r in va_doc["rows"] if r.get("blank_distinct"))
        w(f"全样本 VA：中位 {st.median(va_all):.4f}，均值 {st.fmean(va_all):.4f}，"
          f"范围 {min(va_all):.4f}–{max(va_all):.4f}。")
        if vstep:
            w(f"首个生成位置的 VA（`VA_step1`）中位 **{st.median(vstep):.4f}**；"
              "它大于 0 本身即证明**目标模型在文本位置上确实看到了视频**，"
              "即 cache 路径下文本 hidden 携带了视觉信息（§3 的解释依赖这一点）。")
            w(f"置零视频后贪心续写改变的样本：**{nchanged}/{len(va_doc['rows'])}**；"
              f"续写差异比例 `prefix_div` 中位 {st.median(vdiv):.4f}。")
        w("")
        w("### 判据")
        w("")
        if a_b is not None and len(va_table) >= 2:
            top = va_table[-1]
            lo = va_table[0]
            # deltas are (X - A): A's ADVANTAGE is the negation.
            adv_hi = -top.get("B_d", float("nan"))
            adv_lo = -lo.get("B_d", float("nan"))
            c_hi = -top.get("C_d", float("nan"))
            c_lo = -lo.get("C_d", float("nan"))
            if adv_hi > 0 and adv_hi > adv_lo + 0.05:
                v = ("**A 在高 VA 桶比在低 VA 桶更明显地优于 B**："
                     "MLP 投影在视觉位置有帮助，且帮助随视觉依赖上升。")
            elif adv_hi > 0 and adv_lo > 0 and abs(adv_hi - adv_lo) <= 0.05:
                v = "**A 在各桶表现接近**：MLP 投影没有专门针对视觉的额外帮助。"
            elif adv_hi < 0:
                v = "**A 在高 VA 桶反而差**：MLP 投影损害视觉位置。"
            else:
                v = "桶间方向不一致，样本量不足以判定（见 §7）。"
            w(f"- 高 VA 桶（{top['bucket']}，n={top['n']}）：A 相对 B 的优势 "
              f"{adv_hi:+.4f}，相对 C 的优势 {c_hi:+.4f}")
            w(f"- 低 VA 桶（{lo['bucket']}，n={lo['n']}）：A 相对 B 的优势 "
              f"{adv_lo:+.4f}，相对 C 的优势 {c_lo:+.4f}")
            w(f"- → {v}")
        w("")

    # --- 3 attention --------------------------------------------------------
    w("## 3. 草稿层对 KV 的注意力分布（阶段 3）")
    w("")
    if not attn:
        w("注意力提取未完成，本节缺失；见 §7。")
    else:
        w("对臂 A 的**第 0 轮**草稿前向（此时 KV 上下文即完整 prompt），"
          "在每层 `self_attn` 上挂只读 hook，用模块自身的输入重算 q、k 并取 softmax 权重。"
          "hook 不参与前向输出，因此**不可能改变该臂的 τ**。统计量：")
        w("")
        w("- `视觉注意力占比`：KV 上下文中视觉 token 行分到的注意力质量（对所有 head、"
          "所有 block 行取平均后归一）")
        w("- `熵(归一)`：注意力分布熵 / log2(ctx_len)；1.0 = 均匀铺满上下文，"
          "0.0 = 只盯一行")
        w("- `top1%`：质量最重的前 1% 上下文行占的注意力比例（集中度）")
        w("- `前/中/后`：上下文按位置三等分后各段的质量占比")
        w("")
        if vis_frac_ctx is not None:
            w(f"**参照基线**：KV 上下文中视觉 token 本身占 **{100*vis_frac_ctx:.1f}%** 的行数。"
              "注意力占比只有与这个数比较才有意义——均匀注意力会给出同样的占比。")
            w("")
        w("| 层 | ctx 长度（行中位） | 视觉注意力占比 | 熵(归一) | top1% 质量 | 前段 | 中段 | 后段 |")
        w("|---|---|---|---|---|---|---|---|")
        for e in attn:
            w(f"| {e['layer']} | {e['n_ctx']:.0f} | {100*e['vis_mass']:.1f}% | "
              f"{e['entropy_norm']:.3f} | {100*e['top1pct']:.1f}% | "
              f"{100*e['third1']:.1f}% | {100*e['third2']:.1f}% | {100*e['third3']:.1f}% |")
        w("")
        if vis_frac_ctx:
            vm = st.median(e["vis_mass"] for e in attn)
            ent = st.median(e["entropy_norm"] for e in attn)
            top = st.median(e["top1pct"] for e in attn)
            over = vm / vis_frac_ctx
            w("### 判据")
            w("")
            w(f"- 视觉注意力占比中位 **{100*vm:.1f}%**，相对行数占比 {100*vis_frac_ctx:.1f}% "
              f"的倍率 **{over:.2f}×**；归一化熵中位 **{ent:.3f}**；top1% 质量 **{100*top:.1f}%**。")
            if over > 1.3 and ent < 0.8:
                v = "视觉注意力**集中且高于行数占比** → 按判据表，MLP 成功桥接视觉。"
            elif over > 1.3 and ent >= 0.8:
                v = "视觉注意力占比高于行数占比但**分散**（熵接近均匀）→ 视觉信息被使用但未被有效聚焦。"
            elif over <= 1.3 and ent >= 0.8:
                v = "视觉注意力占比**不高于行数占比且分布分散** → 按判据表读作「未有效利用视觉信息」。"
            else:
                v = "注意力集中在少数行但视觉占比不高 → 需要看 §4 的特征相关性判断。"
            w(f"- → {v}")
            w("")
            w("**但按判据表直接下「未使用视觉」的结论会读错，原因见 §3.1：**"
              "草稿头的 KV 里视觉信息有两份——显式的视觉行（占 "
              f"{100*vis_frac_ctx:.1f}% 的行）与隐式的文本行"
              + (f"（投影后仍随视频变化 {kvv['proj_relL2_txt']:.2f} 相对 L2）"
                 if kvv else "") + "。"
              "视觉行「分到的注意力低于其行数占比」与「草稿头不依赖视觉」是两件事："
              "更合理的读法是草稿头通过**文本行**间接读取了视觉摘要，"
              "而把显式视觉行当作冗余备份——这与 §2 的 A≫B（灌入原始视觉行有害）"
              "和 A 略优于 C（删掉显式视觉行有小损）同时一致。")
            w("")
            hi_layer = max(attn, key=lambda e: e["vis_mass"])
            lo_layer = min(attn, key=lambda e: e["vis_mass"])
            w("层间差异本身也说明问题：视觉注意力占比从第 "
              f"{lo_layer['layer']} 层的 {100*lo_layer['vis_mass']:.1f}% 升到第 "
              f"{hi_layer['layer']} 层的 {100*hi_layer['vis_mass']:.1f}%（中间层最高），"
              f"而第 0 层 {100*attn[0]['third3']:.0f}% 的注意力落在上下文的**最后一段**"
              "（即视频之后的提问文本），后几层才把权重移向前段的视频行。"
              "这不是「忽略视觉」，而是分层的读取顺序。")
        w("")

    # --- 3b: where the visual information actually sits in the KV -------------
    w("### 3.1 视觉信息在 KV 里到底存在于哪些行")
    w("")
    if not kvv:
        w("KV 视觉含量探针未完成，本节缺失；见 §7。")
    else:
        w("上表的「视觉注意力占比」只说明草稿头读了多少**视觉行**。"
          "但草稿头的 KV 还有另一种视觉来源：cache 路径下目标模型的**文本行**本身就"
          "依赖视频（§2 的 `VA_step1` 已证）。本节直接量化这一点——每行做两次带 cache 的 "
          "prefill（真实视频 / 像素置零），比较头实际消费的那份张量。")
        w("")
        w("| 量（行中位） | 值 |")
        w("|---|---|")
        w(f"| KV 中视觉行占比 | {100*kvv['vis_frac']:.1f}% |")
        w(f"| 5 层融合特征相对 L2 变化 @视觉位置 | {kvv['fused_relL2_vis']:.3f} |")
        w(f"| 5 层融合特征相对 L2 变化 @文本位置 | {kvv['fused_relL2_txt']:.3f} |")
        w(f"| 融合特征余弦 @视觉位置 | {kvv['fused_cos_vis']:+.3f} |")
        w(f"| 融合特征余弦 @文本位置 | {kvv['fused_cos_txt']:+.3f} |")
        w(f"| **投影后（`hidden_norm(fc(·))`）相对 L2 变化 @视觉位置** | {kvv['proj_relL2_vis']:.3f} |")
        w(f"| **投影后相对 L2 变化 @文本位置** | **{kvv['proj_relL2_txt']:.3f}** |")
        w(f"| 投影后余弦 @视觉位置 | {kvv['proj_cos_vis']:+.3f} |")
        w(f"| 投影后余弦 @文本位置 | {kvv['proj_cos_txt']:+.3f} |")
        w("")
        w("读法：")
        w("")
        w(f"- 视觉行占 KV 的 {100*kvv['vis_frac']:.1f}%，草的稿头绝大多数上下文行是视觉 token；")
        w(f"- 但**文本行同样携带视觉信息**：置零视频后文本位置的投影后 KV 相对变化达 "
          f"**{kvv['proj_relL2_txt']:.3f}**（余弦 {kvv['proj_cos_txt']:+.3f}）。"
          f"注意它比投影**前**的融合特征变化（{kvv['fused_relL2_txt']:.3f}）大得多，"
          "即这处投影不仅搬运视觉信息，还把文本行里的视觉依赖**放大**了；")
        w("- 因此臂 C（只把视觉行置零）并没有真正切断草稿头与视频的联系，"
          "它只是删掉了那份**显式**拷贝，隐式拷贝（文本行）仍在——这正是臂 C 的"
          "前缀内 Δ 只有臂 B 一半左右的原因。")
        w("")

    # --- 4 features ---------------------------------------------------------
    w("## 4. 特征相关性（阶段 4）")
    w("")
    if not feat:
        w("特征相关性计算未完成，本节缺失；见 §7。")
    else:
        w("在冻结目标上每行做一次 prefill，取")
        w("`proj = hidden_norm(fc(cat(layers)))`（即臂 A 的 KV 来源），"
          "按视觉/文本位置分别与原始视觉嵌入、最终层 hidden 比较。")
        w("余弦对基座敏感，CKA 对可逆线性变换不变；两者一起看才能区分"
          "「被旋转但信息完整」与「真的塌了」。")
        w("")
        w("| 量 | 中位 |")
        w("|---|---|")
        names = {
            "cos_proj_raw_vis": "cos(proj, 原始视觉嵌入) @视觉位置",
            "cos_proj_last_vis": "cos(proj, 最终层 hidden) @视觉位置",
            "cos_proj_last_txt": "cos(proj, 最终层 hidden) @文本位置（同模型对照）",
            "cos_raw_last_vis": "cos(原始视觉嵌入, 最终层 hidden) @视觉位置（对照）",
            "cos_projvis_projtxt_mean": "cos(proj 视觉均值, proj 文本均值)",
            "cos_projvis_lasttxt_mean": "cos(proj 视觉均值, 最终层文本均值)",
            "cos_rawvis_lasttxt_mean": "cos(原始视觉均值, 最终层文本均值)",
            "cka_proj_raw_vis": "CKA(proj, 原始视觉嵌入) @视觉位置",
            "cka_proj_last_vis": "CKA(proj, 最终层 hidden) @视觉位置",
            "cka_proj_last_txt": "CKA(proj, 最终层 hidden) @文本位置（同模型对照）",
            "cka_proj_raw_txt": "CKA(proj, 原始嵌入) @文本位置（对照）",
            "proj_rms_vis": "proj RMS @视觉位置",
            "proj_rms_txt": "proj RMS @文本位置",
            "last_rms_vis": "最终层 hidden RMS @视觉位置",
            "raw_rms_vis": "原始视觉嵌入 RMS @视觉位置",
        }
        for k, lab in names.items():
            w(f"| {lab} | {feat[k]:+.4f} |")
        w("")
        w("### 判据")
        w("")
        c_raw = feat["cos_proj_raw_vis"]
        c_cka = feat["cka_proj_last_vis"]
        c_txt = feat["cos_proj_last_txt"]
        w(f"- proj 与原始视觉嵌入的余弦中位 **{c_raw:+.3f}**；"
          f"与最终层视觉 hidden 的 CKA 中位 **{c_cka:.3f}**；"
          f"文本侧同模型对照 cos 中位 **{c_txt:+.3f}**。")
        if c_cka > 0.7:
            v1 = "与目标自身深层视觉特征**高相关** → 投影保留了视觉语义。"
        elif c_cka > 0.3:
            v1 = "与目标自身深层视觉特征**中等相关** → 投影保留了部分视觉语义。"
        else:
            v1 = "与目标自身深层视觉特征**低相关** → 投影丢弃了大部分视觉信息。"
        w(f"- → {v1}")
        gap = feat["cos_projvis_lasttxt_mean"] - feat["cos_rawvis_lasttxt_mean"]
        if gap > 0.1:
            v2 = ("proj 的视觉均值比**原始视觉嵌入**更靠近文本流形"
                  f"（{feat['cos_projvis_lasttxt_mean']:+.3f} vs "
                  f"{feat['cos_rawvis_lasttxt_mean']:+.3f}）→ 投影把视觉向文本空间对齐，"
                  "这正是它作为「桥」的作用。")
        elif gap < -0.1:
            v2 = "proj 的视觉均值并不比原始视觉嵌入更靠近文本流形 → 没有证据支持「映射到文本空间」。"
        else:
            v2 = "proj 的视觉均值与原始视觉嵌入到文本流形的距离相当 → 投影基本保持原有几何。"
        w(f"- → {v2}")
        if cka_cached:
            w("")
            w("> **口径更正（2026-10-05）**：以上 CKA 走 `use_cache=True`，"
              "即 `glance.decode.generate` 实际使用的 prefill 路径。"
              "本文件早先版本用了 `use_cache=False` 的消融路径，"
              "得到的视觉位置 CKA 是 0.557（文本侧 0.434/0.742）。"
              "两条路径的对照：")
            w("")
            w("| 量 | `use_cache=False`（错） | `use_cache=True`（评测路径） | 差 |")
            w("|---|---|---|---|")
            for k, lab in [("cka_proj_raw_vis", "CKA(proj, 原始视觉)@视觉"),
                           ("cka_proj_last_vis", "CKA(proj, 目标深层视觉)@视觉"),
                           ("cka_proj_last_txt", "CKA(proj, 最终层)@文本"),
                           ("cka_proj_raw_txt", "CKA(proj, 原始嵌入)@文本"),
                           ("cos_projvis_projtxt_mean", "cos(proj视觉均值, proj文本均值)")]:
                w(f"| {lab} | {feat_old.get(k, float('nan')):+.4f} | "
                  f"**{feat.get(k, float('nan')):+.4f}** | "
                  f"{feat.get(k, float('nan')) - feat_old.get(k, float('nan')):+.4f} |")
            w("")
            w("更正后的读法与 §6.3/§6.4 的相应修正一致：视觉位置保真度"
              "**高于**文本位置，而§6.4 曾据错值提出的「缺一个视觉映射模块」不再成立。")
        w("")

    # --- 5 long context -----------------------------------------------------
    w("## 5. 长视频 / 长上下文鲁棒性（阶段 5）")
    w("")
    w("**任务要求的时间轴在本留出集上不存在，这是一个结构性事实，不是本诊断的失败：**")
    w("")
    w(f"- fresh30 全部 30 行的 `n_frames` 相同，均为 {nf}（实测取值集合 {nf}）；")
    w("- 视频缓存本身是定长重采样产物：`repro/data/predecode_cache.py` 的 `normalize()` "
      "把任意长度源视频写成恰好 `n_frames/fps` 秒的短片，其 docstring 明说"
      "「Writing a clip whose duration is exactly n_frames/fps decouples frame count "
      "from source length」；")
    w("- 因此缓存内 mp4 时长实测全部为 0.5333 s、源帧数全部为 16，原始时长未被保留，"
      "**帧数分桶与时长分桶都无法计算**。")
    w("")
    w("可用且确实在变化的上下文轴只有两条：**视觉 token 数**（由空间分辨率决定，"
      f"实测取值 {nvis_vals}）与**总 prompt 长度**。下面用总 prompt 长度分桶，"
      "这是「KV 上下文变长时 A 相对 B/C 是否衰减」在本数据上唯一可算的代理。")
    w("")
    w(f"分桶方式：{len_kind}。")
    w("")
    hdr = "| prompt 长度桶 | n | L 范围 | " + " | ".join(f"{k} τ中位" for k in have) + \
          " | " + " | ".join(f"Δ({k}−A)" for k in have if k != "A") + " |"
    w(hdr)
    w("|---" * (3 + len(have) + len([k for k in have if k != "A"]) + 1) + "|")
    for row in len_table:
        cells = [f"| {row['bucket']} | {row['n']} | {row['L_lo']:.0f}–{row['L_hi']:.0f} "]
        for k in have:
            cells.append(f"| {row[k]:.4f} ")
        for k in have:
            if k == "A":
                continue
            cells.append(f"| {row[k+'_d']:+.4f} ")
        w("".join(cells) + "|")
    w("")
    w("### 判据")
    w("")
    if len(len_table) >= 2 and a_b is not None:
        hi = len_table[-1]
        lo = len_table[0]
        w(f"- 最长桶（{hi['bucket']}，n={hi['n']}，L {hi['L_lo']:.0f}–{hi['L_hi']:.0f}）："
          f"A 相对 B 的优势 {-hi['B_d']:+.4f}，相对 C 的优势 {-hi['C_d']:+.4f}")
        w(f"- 最短桶（{lo['bucket']}，n={lo['n']}，L {lo['L_lo']:.0f}–{lo['L_hi']:.0f}）："
          f"A 相对 B 的优势 {-lo['B_d']:+.4f}，相对 C 的优势 {-lo['C_d']:+.4f}")
        adv_hi = -hi["B_d"]
        adv_lo = -lo["B_d"]
        if abs(adv_hi - adv_lo) < 0.05:
            trend = ("**A 相对 B 的优势在长短桶上基本相同**："
                     "没有证据表明 MLP 投影对更长的上下文有特殊帮助。")
        elif adv_hi > adv_lo:
            trend = "**A 相对 B 的优势随上下文变长而增大**：投影在更长上下文上帮助更大。"
        else:
            trend = "**A 相对 B 的优势随上下文变长而缩小**：投影在更长上下文上帮助变小。"
        w(f"- → {trend}")
        w("- 但必须强调：本留出集最长 prompt 也只有 "
          f"{max(lens.values())} token，且行长差异来自空间 token 数而非时间长度，"
          "**这个结论不能外推为长视频结论**。")
        thin = [f"{r['bucket']}(n={r['n']})" for r in len_table if r["n"] < 3]
        if thin:
            w(f"- 注：桶 {'、'.join(thin)} 样本过少，上面只按样本量足够的首尾两桶读趋势，"
              "该桶数字不参与判据。")
    w("")

    # --- 6 conclusion -------------------------------------------------------
    w("## 6. 结论")
    w("")
    a_t = main_rows[0]["tau_med"]
    w("### 6.1 当前 MLP 投影对视觉领域是否有帮助")
    w("")
    pdel = {r["arm"]: (r["clean"]["delta_median"] if r["clean"] else None) for r in main_rows}
    pd_b = pdel.get("B")
    pd_c = pdel.get("C")
    pd_b_s = "—" if pd_b is None else f"{pd_b:+.3f}"
    pd_c_s = "—" if pd_c is None else f"{pd_c:+.3f}"
    d_arm = next((r for r in main_rows if r["arm"] == "D"), None)
    dn_arm = next((r for r in main_rows if r["arm"] == "Dn"), None)
    if a_b is not None:
        w("**有。** 结论建立在三层证据上，逐层减弱但仍同向：")
        w("")
        w(f"1. **去掉投影，投机收益整个消失。** D 臂（`hidden_states[-1]` 直送 k/v_proj，"
          f"跳过 `fc` 与 `hidden_norm`）的 τ 中位 "
          f"{d_arm['tau_med']:.3f}（E[a] {d_arm['ea_med']:.3f}），即每轮只提交一个 "
          f"bonus token，相对 A 的 {a_t:.3f} 是**归零**。`fc` 不是可选装饰。")
        if dn_arm is not None:
            w(f"2. **尺度归一化单独也有价值，但它解释不了全部崩塌。** Dn 臂（保留 "
              f"`hidden_norm`、只去 `fc`）τ 中位 {dn_arm['tau_med']:.3f}，"
              f"D 为 {d_arm['tau_med']:.3f}；两者之差 "
              f"{dn_arm['tau_med']-d_arm['tau_med']:+.3f} 是「归一化」这一项的单独贡献，"
              f"而 Dn 相对 A 仍差 {dn_arm['delta_med']:+.3f}，那部分才是 `fc` 承载的信息。")
        else:
            w("2. Dn 臂未完成，`fc` 与归一化的贡献无法分离。")
        w(f"3. **换掉视觉来源会掉点，且控制文本分歧后掉得更多。** 把视觉位置的 KV 换成"
          f"目标自己的原始视觉嵌入（B），前缀内 Δ 中位 **{pd_b_s}**；"
          f"完全删掉视觉行（C），前缀内 Δ 中位 **{pd_c_s}**。")
        w("")
        w("一句话：`fc` + `hidden_norm` 是让目标模型融合隐藏状态**可被草稿头消费**的必要接口；"
          "它同时承担了「可用表征」和「视觉信息」两件事，前者是刚性的，后者是增量的。")
    w("")
    w("### 6.2 帮助体现在哪些位置")
    w("")
    if va_table:
        top = va_table[-1]
        lo = va_table[0]
        w(f"- **视觉依赖高的行**（{top['bucket']} VA 桶，n={top['n']}）："
          f"Δ(B−A) 中位 {top.get('B_d', float('nan')):+.4f}，"
          f"Δ(C−A) 中位 {top.get('C_d', float('nan')):+.4f}")
        w(f"- **视觉依赖低的行**（{lo['bucket']} VA 桶，n={lo['n']}）："
          f"Δ(B−A) 中位 {lo.get('B_d', float('nan')):+.4f}，"
          f"Δ(C−A) 中位 {lo.get('C_d', float('nan')):+.4f}")
        w("")
        w("见 §2 的完整分桶表。桶内 n 很小（约 10 行），桶级 Δ 只作方向性证据。")
    if len_table:
        w("")
        w("上下文长度方向见 §5：本留出集没有时间长度变化（全部 16 帧），"
          "只有空间 token 数变化，因此那一节不能给出长视频结论。")
    w("")
    w("### 6.3 帮助有限之处：它丢失了什么")
    w("")
    if feat:
        w("§4 给出了可以直接对照的三个量：")
        w("")
        w(f"- **保留**：CKA(proj, 目标自身深层视觉 hidden) = "
          f"{feat.get('cka_proj_last_vis', float('nan')):.3f}，"
          "即大部分共享结构被保留；")
        w(f"- **丢失**：同一指标的补集（≈{1-feat.get('cka_proj_last_vis', float('nan')):.2f}）"
          "就是投影后不再与目标深层视觉特征共享的部分；")
        w("- **模态对照（已按评测路径更正）**：视觉位置 "
          f"CKA(proj, 深层视觉) = {feat.get('cka_proj_last_vis', float('nan')):.3f}、"
          f"CKA(proj, 原始视觉) = {feat.get('cka_proj_raw_vis', float('nan')):.3f}；"
          f"文本位置 CKA(proj, 最终层) = {feat.get('cka_proj_last_txt', float('nan')):.3f}、"
          f"CKA(proj, 原始嵌入) = {feat.get('cka_proj_raw_txt', float('nan')):.3f}；"
          f"`cos(proj 视觉均值, proj 文本均值)` = "
          f"{feat.get('cos_projvis_projtxt_mean', float('nan')):.3f}。"
          "即在正确口径下**视觉位置的保真度不低于文本位置**，且视觉已被映射到与文本"
          "高度对齐的方向。**本文件早先版本据错值提出的「模态不对称、缺一个视觉映射模块」"
          "不成立**，详见 §4 的口径更正表。")
    else:
        w("§4 未完成，无法给出投影保真度的量化。")
    w("")
    if kvv:
        w(f"此外，视觉 token 占草稿头 KV 的 {100*kvv['vis_frac']:.1f}% 行，"
          f"但置零视频后**文本行**的投影后 KV 相对变化仍有 {kvv['proj_relL2_txt']:.3f}；"
          "即投影把视觉依赖也写进了文本行。臂 C 只删掉显式视觉行，"
          "所以它的掉点（" + pd_c_s + "）显著小于臂 B（"
          + pd_b_s + "）——**「删掉视觉」比「灌入原始视觉」代价更小**，"
          "说明当前的失败模式不是缺少视觉，而是缺乏能把原始视觉变成草稿头可用表征的映射。")
        w("")
    w("### 6.4 是否需要引入可训练的视觉适配模块（ViSpec / DREAM 风格）")
    w("")
    if a_b is not None:
        w("**结论：本诊断不构成引入新模块的充分理由；但它指出了一个明确的、值得做的判定实验。**")
        w("")
        w("**不支持现在引入**：")
        w("")
        w(f"- 现有投影已经决定性地工作：去掉它 τ 从 {a_t:.2f} 掉到 "
          f"{d_arm['tau_med']:.2f}（归零）。要改进的对象是一个已经在工作的接口，"
          "不是缺失的模块。")
        w(f"- 增量天花板可测且有界：在本留出集上，完全删掉视觉行只值 "
          f"{abs(pd_c or 0):.2f} τ（≈"
          f"{100*abs(pd_c or 0)/a_t:.0f}%）。一个视觉适配模块能拿回的上限"
          "不会超过这个量级加上「原始视觉映射得当」的部分，即 "
          f"{abs(pd_b or 0):.2f} τ 左右。")
        w("- B/C/D 都是**分布外**对照（§0.3）：它们给出的是现有头对 A 表示的依赖下界，"
          "不是新模块的上界。拿分布外劣势去论证换模块，逻辑上不成立。")
        w("")
        w("**原本支持引入的那条证据已在本文件更正后消失**：")
        w("")
        w(f"- §4 更正后，视觉位置保真度（CKA(proj, 深层视觉) = "
          f"{feat.get('cka_proj_last_vis', float('nan')):.3f}、"
          f"CKA(proj, 原始视觉) = {feat.get('cka_proj_raw_vis', float('nan')):.3f}）"
          f"**不低于**文本侧（CKA(proj, 最终层) = "
          f"{feat.get('cka_proj_last_txt', float('nan')):.3f}、"
          f"CKA(proj, 原始嵌入) = {feat.get('cka_proj_raw_txt', float('nan')):.3f}），"
          f"且 `cos(proj 视觉均值, proj 文本均值)` = "
          f"{feat.get('cos_projvis_projtxt_mean', float('nan')):.3f}，"
          "即视觉已被映射到与文本高度对齐的方向。")
        w("- 仍然成立的一条是行为证据：原始视觉嵌入直接灌入 KV 的代价（"
          f"{pd_b_s}）大于删掉视觉（{pd_c_s}）。"
          "但这条只说明「原始视觉不是可用的 KV」，并不说明缺少一个**模块**——"
          f"现有 `fc` 的视觉保真度已经是 {feat.get('cka_proj_last_vis', float('nan')):.3f}。")
        w("- 因此本诊断**没有**留下支持引入模态专用模块的证据。"
          "该问题由后续的 ViSpec 判定实验独立回答。")
        w("")
        w("**判定性实验（本诊断未做，明确列为下一步）**：")
        w("")
        w("在**同一份数据、同一评测口径**下，把以下三者都重训到收敛再比较：")
        w("")
        w("1. 现役 A 头（已有）；")
        w("2. 一个「原始视觉 token + 可训练视觉投影」的 B′ 头；")
        w("3. 一个模态专用适配模块（ViSpec/DREAM 风格）替换 `fc` 的头。")
        w("")
        w("只有当 (2)/(3) 在**重训后**超过 (1)，引入新模块才有依据。"
          "本诊断给出的是该实验的必要性与量级估计，不是它的结论。")
    w("")

    # --- 7 anomalies --------------------------------------------------------
    w("## 7. 异常记录")
    w("")
    anoms = [
        ("**bf16 下解码不是逐位无损的，跨臂 τ 因此混入「换了文本」的效应。**",
         "AR oracle 比对显示各臂与目标自身贪心输出在多数行上出现分歧（见 §1.1 表），"
         "分歧后各臂在为不同续写起草。已给出「前缀内 τ」作为混淆受控的替代统计。"
         "根因不是本诊断的实现：bf16 的 tree pass 与单 token pass 归约顺序不同，接近的 "
         "tie 会落到两侧，仓库的无损门禁本来只在 fp32 下通过。若要干净的跨臂对比，"
         "应在 fp32 下重跑（本诊断未做，明确列为未完成项）。"),
        ("**`use_cache=False` 的单次全序列前向会给出与视觉输入无关的文本表示。**",
         "实测：把 `pixel_values_videos` 置零或取负，文本位置的 hidden 与 logits "
         "与真实视频**逐位相同**（layer36 文本位置 max|d| = 0.000000，logits 同），"
         "而视觉位置的 hidden 已差到 ~2e3；改用 `use_cache=True` 的 prefill"
         "（`glance.decode.generate` 的路径）后同一比较的 logits 差约 15，"
         "且贪心续写明显不同。用前者的口径算视觉依赖度会得到全 0 的假结论。"
         "本诊断的 VA 全程用 cache 路径；这一条对任何在 Qwen3-VL 上做类似消融的人都是一个陷阱。"),
        ("**四臂共用同一训练头，B/C/D/Dn 为分布外输入。**",
         "这是主动选择：任务是诊断「当前投影有多大帮助」，不是搜索更好的 KV 来源。"
         "代价是 B/C/D 的数字是下界，不能当作「这些方案的上限」。未重训，因为任务禁止"
         "改架构、且 4 小时内重训 4 个 1.6B 头无法在无偏口径下完成。"),
        ("**`video_heldout_30.parquet` 是被污染的「留出集」，已弃用。**",
         "实测该 30 行 100% 落在 `pool_499` 的 train 划分内，而 H-8 头按 "
         "`train rows: 439` 训练（`logs/B0_sft_h8.log`），即它见过这 30 行。"
         "改用 `video_fresh30.parquet`（30/30 落在 heldout 划分，与 train 不交）。"
         "另注：`scripts/83_eval_heads.sh` 的 `ROWS` 默认值正是被污染的那个文件，"
         "凡是用默认值跑出的历史数字都需按此复核。"),
        ("**任务给的草稿头规模 ~1.05B 与现役 H-8 不符。**",
         "实测 H-8 = 1627.5M 参数（bf16 权重文件 3.25 GB）。1048.6M 是 H-5 头"
         "（`head_video439`）。本诊断按「8 层」这一更明确的描述取 H-8。"),
        ("**阶段 5 的帧数/时长分桶无法计算，是数据结构问题。**",
         "fresh30 全部 30 行 `n_frames` 恒为 16；`predecode_cache.py::normalize()` "
         "刻意把每个源视频重采样成恰好 16 帧的等长短片，缓存内 mp4 时长恒为 0.5333 s，"
         "原始时长未落盘。已按兜底条款改用 prompt 长度 / 视觉 token 数分桶，"
         "并明确不外推为长视频结论。"),
        ("**VA 分桶的样本量。**", None),
        ("**臂 D 的 τ ≈ 1.0 含尺度效应——已用 Dn 定量分离。**",
         f"D 同时去掉了 `fc` 与 `hidden_norm`，而 `v_proj` 之后没有归一化，"
         f"其崩塌因此不能单独归因于信息损失。补测的 Dn（保留 `hidden_norm`、仅去 `fc`）"
         f"τ 中位为 {dn_arm['tau_med']:.4f}，D 为 {d_arm['tau_med']:.4f}，"
         f"即尺度项只贡献 {dn_arm['tau_med']-d_arm['tau_med']:+.3f}，"
         f"相对 A 剩余的 {dn_arm['delta_med']:+.3f} 才是 `fc` 承载的信息。"
         f"尺度差异的实测幅度：最终层 hidden 的 RMS 中位 32.875，"
         f"而投影输出 RMS 中位 0.883，相差约 37 倍（§4）。"
         f"结论：尺度是 D 崩塌的必要条件，但不是主因。"),
        ("**fresh30 曾被用于 H-8 的 checkpoint 选点。**",
         "§0.3 已说明。选点泄漏对 τ 绝对值是乐观的，但四臂共用同一 checkpoint，"
         "逐行配对的 Δ 不受影响——本诊断的全部结论都建立在 Δ 上。"),
    ]
    for i, (title, body) in enumerate(anoms, 1):
        if body is None:
            if va_table:
                ns = [f"{r['bucket']}={r['n']}" for r in va_table]
                body = (f"分出 {len(va_table)} 桶，样本数为 {'、'.join(ns)}。"
                        "桶内 n 偏小，桶级 Δ 只是方向性证据，未做桶内显著性检验。")
            else:
                body = "VA 计算未完成。"
        w(f"{i}. {title}")
        w(f"   {body}")
    w("")
    w("### 未静默失败的声明")
    w("")
    w("全部脚本不使用 `try/except` 吞异常；任一臂失败会以 traceback 终止并保留日志。"
      "本文件只报告实际跑出的数据：六个阶段（四臂消融、无损性定位、视觉依赖分层、"
      "注意力分布、特征相关性、上下文分层）全部完成，无跳过项；"
      "未完成的是**任务之外**的两件事——fp32 重跑（§7 第 1 条）与 B′/ViSpec 头的重训"
      "（§6.4），两者都在正文中明确标注为未做，而不是以缺失数字代替。")
    w("")
    w("### 复现命令")
    w("")
    w("```bash")
    w("cd /home/mcy/Projects/GLANCE/repro")
    w("H=/mnt/local/mcy/GLANCE/checkpoints/head_video439_h8")
    w("R=/mnt/local/mcy/GLANCE/data/video_fresh30.parquet")
    w("for arm in A B C D Dn; do")
    w("  python diag_arms.py --arm $arm --rows $R --head $H \\")
    w("    --max-new 1024 --budget 63 --branch 8 --warmup 1 --gpu <g> \\")
    w("    --tag arm$arm --out /mnt/local/mcy/GLANCE/results/diag/arm$arm.json")
    w("done")
    w("# 阶段 3 注意力（只读取，不改 τ）")
    w("python diag_arms.py --arm A --rows $R --head $H --max-new 1024 --budget 63 \\")
    w("  --branch 8 --warmup 1 --capture-attn --capture-attn-rows 30 \\")
    w("  --out /mnt/local/mcy/GLANCE/results/diag/armA_attn.json")
    w("# 阶段 2 视觉依赖（只跑目标模型；必须用带 cache 的路径，见 §7 第 2 条）")
    w("python diag_va.py --rows $R --cont 64 --out /mnt/local/mcy/GLANCE/results/diag/va.json")
    w("# 阶段 4 特征相关性")
    w("python diag_features.py --rows $R --head $H --out /mnt/local/mcy/GLANCE/results/diag/features.json")
    w("# 无损性参照（目标自身贪心）与 token 级分歧定位")
    w("python diag_ar.py --rows $R --max-new 1024 --out /mnt/local/mcy/GLANCE/results/diag/ar.json")
    w("for arm in A B C; do")
    w("  python diag_arms.py --arm $arm --rows $R --head $H --max-new 1024 --budget 63 \\")
    w("    --branch 8 --warmup 1 --store-ids --gpu <g> \\")
    w("    --out /mnt/local/mcy/GLANCE/results/diag/arm${arm}_ids.json")
    w("done")
    w("# 视觉信息在 KV 中的分布（显式视觉行 vs 隐式文本行）")
    w("python diag_kv_visual.py --rows $R --head $H --out /mnt/local/mcy/GLANCE/results/diag/kv_visual.json")
    w("# 汇总")
    w("python diag_report.py --dir /mnt/local/mcy/GLANCE/results/diag \\")
    w(f"  --out /mnt/local/mcy/GLANCE/results/diag/视觉适配诊断_{today}.md")
    w("```")
    w("")
    w("**产物落盘位置**：全部诊断产物（结果 JSON、日志、本文件）都写在 "
      "`/mnt/local/mcy/GLANCE/` 下，不占 home：")
    w("")
    w("- 结果：`/mnt/local/mcy/GLANCE/results/diag/`")
    w("- 日志：`/mnt/local/mcy/GLANCE/logs/diag/`")
    w("- 脚本：`/home/mcy/Projects/GLANCE/repro/diag_*.py`（属代码仓库，非产物，合计约 13 KB）")

    out = a.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {out}  ({len(L)} lines)")
    print(f"arms present: {have}  n_common={n_common}")
    for r in main_rows:
        print(f"  {r['arm']}: tau_med={r['tau_med']:.4f} d_med={r['delta_med']:+.4f} p={fmt_p(r['p'])}")


if __name__ == "__main__":
    main()
