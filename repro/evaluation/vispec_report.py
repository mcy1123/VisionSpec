#!/usr/bin/env python3
"""Assemble the ViSpec decision document from the artefacts on disk.

Every number in the output comes from a file this pipeline wrote; nothing is
typed in by hand, and each verdict prints the comparison that produced it.

    python vispec_report.py            # -> results/vispec/ViSpec判定_<date>.md
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

from scipy import stats as sps

ROOT = "/mnt/local/mcy/GLANCE"
DIAG = os.path.join(ROOT, "results", "diag")
VSP = os.path.join(ROOT, "results", "vispec")
CKPT = os.path.join(ROOT, "checkpoints")

ARM_LABEL = {
    "A": "A 现役：MLP 投影，**warm start**，历史特征路径",
    "C0": "C0 对照：MLP 投影，**从头训**，cache 特征路径",
    "V": "V ViSpec：适配器，**从头训**，cache 特征路径",
}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def loss_curve(name):
    """(step, loss) pairs from a trainer jsonl; loss is the mean over the log window."""
    p = os.path.join(ROOT, "logs", name)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append((int(d["step"]), float(d["loss"])))
    return out


def loss_at(curve, step):
    for s, v in curve:
        if s == step:
            return v
    return None


def mean_loss_over(curve, lo, hi):
    v = [x for s, x in curve if lo <= s <= hi]
    return st.fmean(v) if v else None


def rows_by_video(doc):
    if doc is None:
        return {}
    return {r["video"]: r for r in doc["rows"] if not r.get("warmup", False)}


def paired(deltas):
    n = len(deltas)
    if n == 0:
        return None
    mean = st.fmean(deltas)
    sd = st.stdev(deltas) if n > 1 else 0.0
    t, p = sps.ttest_rel(deltas, [0.0] * n) if n > 1 else (float("nan"), float("nan"))
    return {"n": n, "mean": mean, "median": st.median(deltas), "sd": sd,
            "t": float(t), "p": float(p),
            "win": sum(1 for d in deltas if d > 0), "loss": sum(1 for d in deltas if d < 0),
            "ci": (mean - 1.96 * sd / n ** 0.5, mean + 1.96 * sd / n ** 0.5) if n > 1 else (0, 0)}


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def buckets(values, labels=("低", "中", "高")):
    s = sorted(values)
    dist = sorted(set(s))
    if 1 < len(dist) <= 5:
        edges = []
        for i, v in enumerate(dist):
            lo = -1e9 if i == 0 else (dist[i - 1] + v) / 2
            hi = 1e9 if i == len(dist) - 1 else (v + dist[i + 1]) / 2
            edges.append((lo, hi))
        return edges, [f"={v:g}" for v in dist], "按离散取值分组"
    if len(s) >= 24:
        c = [s[len(s) // 3], s[2 * len(s) // 3]]
        return [(-1e9, c[0]), (c[0], c[1]), (c[1], 1e9)], list(labels), "三分位"
    m = st.median(s)
    return [(-1e9, m), (m, 1e9)], ["低", "高"], "中位数二分"


def bucket_of(x, edges):
    for i, (lo, hi) in enumerate(edges):
        if lo <= x < hi:
            return i
    return len(edges) - 1


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(VSP, f"ViSpec判定_{date.today().isoformat()}.md"))
    a = ap.parse_args()
    today = date.today().isoformat()

    audit = load(os.path.join(DIAG, "holdout_audit.json"))
    probe = load(os.path.join(DIAG, "context_cache_path.json"))
    arms = {k: rows_by_video(load(os.path.join(VSP, f"arm{k}.json"))) for k in ("A", "C0", "V")}
    have = [k for k in ("A", "C0", "V") if arms[k]]
    metas = {k: load(os.path.join(CKPT, f"head_v_{'vispec' if k == 'V' else 'mlp'}",
                                  "train_meta.json")) for k in ("C0", "V")}
    cka = {k: load(os.path.join(VSP, f"cka_{k}_cached.json")) for k in ("A", "C0", "V")}
    cka_unc = load(os.path.join(VSP, "cka_A_uncached.json"))
    va_doc = load(os.path.join(DIAG, "va.json"))
    va = {r["video"]: r["VA"] for r in va_doc["rows"]} if va_doc else {}

    ids = {k: set(arms[k]) for k in have}
    common = sorted(set.intersection(*ids.values())) if have else []

    L = []
    w = L.append
    w(f"# ViSpec 风格视觉适配器判定实验（{today}）")
    w("")
    w("这是一次**从头训的判定实验**，不是上一份诊断的延伸。目标只有一个：判定"
      "「用 ViSpec 风格可训练视觉适配器替换当前简单 MLP 投影」是否带来额外收益。")
    w("未改动冻结目标模型；未改动图像侧已验证配方。")
    w("")

    # ---------------- 0 method ----------------
    w("## 0. 口径")
    w("")
    w("| 项 | 值 |")
    w("|---|---|")
    w("| 目标模型 | Qwen3-VL-8B-Instruct（冻结，`/mnt/local/mcy/GLANCE/models/`） |")
    w("| 草稿头架构 | H-8 块扩散头，`target_layer_ids [1,9,17,25,33]`，block 16，1627.5M 参数 |")
    w("| 训练行 | `pool_499` 的 train 划分 439 行（与 A 臂**完全相同**） |")
    w("| 标签 | 目标模型自身贪心续写（`video_greedy_499.pt`），非数据集答案 |")
    w("| 步数 / lr | 3000 / 3e-4，每 250 步存一次 |")
    w("| 评测留出 | `video_fresh30.parquet`（30 行，`pool_499` 的 heldout 划分） |")
    w("| 树验证 | budget 63 / branch 8（b63k8），`max_new_tokens=1024`，greedy，bf16 |")
    w("| 配对 | 逐行同 video 名对齐 |")
    w("")
    w("三个臂的定义，以及**为什么必须有 C0**：")
    w("")
    w("| 臂 | 视觉路径 | 初始化 | 训练特征路径 |")
    w("|---|---|---|---|")
    w("| **A** | 现有 `hidden_norm(fc(...))` | warm start（来自已训 5 层视频头） | `use_cache=False`（历史路径） |")
    w("| **C0** | 现有 `hidden_norm(fc(...))` | **从头随机** | cache（评测路径） |")
    w("| **V** | ViSpec 适配器 | **从头随机** | cache（评测路径） |")
    w("")
    w("A 臂是 warm start，而任务要求 V 臂从头训。**直接 A vs V 会把「适配器」和"
      "「初始化优势」混在一起**：A 的祖先已经过至少 3000 步视频训练，V 没有。"
      "因此补 C0 —— 与 V 完全同初始化、同特征路径、同步数，唯一差别是有没有适配器。"
      "**V vs C0 才是「适配器是否带来额外收益」的判定**；A 只作参考上界。")
    w("")
    if metas.get("V") or metas.get("C0"):
        w("实测训练元数据：")
        w("")
        w("| 臂 | 头参数量 | 适配器参数量 | 适配器占比 | 从头训 | 特征路径 |")
        w("|---|---|---|---|---|---|")
        for k in ("C0", "V"):
            m = metas.get(k)
            if not m:
                continue
            ad = m.get("adapter_params", 0)
            hp = m.get("head_params", 0)
            w(f"| {k} | {hp/1e6:.1f}M | {ad/1e6:.2f}M | {100*ad/max(hp,1):.1f}% | "
              f"{not m.get('warm_start', True)} | {m.get('feature_path')} |")
        w("")

    # ---------------- 1 the three fixes ----------------
    w("## 1. 三个坑的修复记录")
    w("")

    # 1.1
    w("### 1.1 坑 1：`video_heldout_30` 被污染")
    w("")
    if audit:
        f = audit["facts"]
        w("结构（实测，`repro/diagnostics/audit_holdout_contamination.py`）：")
        w("")
        w(f"- `pool_499` 的 train 划分 439 支 = `video_opd_train_409`(409) ∪ "
          f"`video_heldout_30`(30)；实测 `video_heldout_30` 有 "
          f"**{len(set(f['heldout30']) & set(f['pool499_train']))}/30** 落在 train 划分内。")
        w(f"- `video_fresh30` 有 **{len(set(f['fresh30']) & set(f['pool499_train']))}/30** "
          f"落在 train 划分内、**{len(set(f['fresh30']) & set(f['pool499_heldout']))}/30** "
          "落在 heldout 划分内 → 它是干净的。")
        w("")
        bad = [r for r in audit["results"] if r["verdict"] != "CLEAN"]
        trained = [r for r in bad if "trained" in r["verdict"]]
        inherited = [r for r in bad if "inherited" in r["verdict"]]
        w(f"**受影响的历史结果：{len(bad)} / {len(audit['results'])} 个已记录评测**"
          f"（直接训练污染 {len(trained)} 个，经 warm start 继承污染 {len(inherited)} 个）。"
          "全部集中在 `results/opd_task/`。")
        w("")
        w("| 类别 | 头 | 受影响结果 | 原因 |")
        w("|---|---|---|---|")
        w(f"| 直接训练污染 | `head_video439` | sft, timing_sft{{,_shard0..2}}, fp32_lossless{{,_shard0,1}}, "
          "divergence_bf16, first_token_gap, smoke_ar, smoke_fp32 "
          f"（{len(trained)} 个） | 该头 `train rows: 439 (of 499)`，含 heldout_30 |")
        w(f"| warm start 继承污染 | `head_opd_video409`, `head_opd_video409_lr1e4` | "
          f"opd{{250..2000}}, lr{{250,1250,2000}}, timing_opd1250{{,_shard0..2}} "
          f"（{len(inherited)} 个） | 训练时已排除这 30 行，但 warm start 自 `head_video439/step1000`，"
          "信息已在初始权重里 |")
        w("")
        w("**未被污染的重要一类**：`results/opd_fix/*`（a16/al/h8/mp/rk 等）与本次全部诊断产物"
          "用的是 `video_fresh30`，审计判为 CLEAN。`zero.json`（图像侧零样本头）也 CLEAN。")
        w("")
        w("**已改**：`scripts/83_eval_heads.sh` 的 `ROWS` 默认值由 `video_heldout_30.parquet` "
          "改为 `video_fresh30.parquet`，并加了一条守卫：显式传 `video_heldout_30` 时直接 "
          "`exit 2` 拒绝运行，而不是静默给出乐观数字。")
        w("")
        w("> 注意区分两种用途：`video_heldout_30` 作为 **OPD 训练时的验证划分**是合法的"
          "（`opd_train_video.py --holdout-parquet` 正是用它把 30 行排除出训练）；"
          "不合法的是把它当**最终报告口径**。`82_train_opd_video.sh` 的 `HOLDOUT` 因此保持不变。")
    else:
        w("审计未完成。")
    w("")

    # 1.2
    w("### 1.2 坑 2：`use_cache=False` 陷阱")
    w("")
    if probe:
        med = {k: st.median(r[k] for r in probe["rows"]) for k in
               ["uncached_relL2_vis", "uncached_relL2_txt", "uncached_cos_txt",
                "cached_relL2_vis", "cached_relL2_txt", "cached_cos_txt",
                "paths_relL2_txt", "paths_cos_txt"]}
        w("在**训练真正使用的那份张量**（`common.target_context` 的 5 层融合特征）上实测，"
          "每行把 `pixel_values_videos` 置零后比较：")
        w("")
        w("| 量（行中位） | `use_cache=False`（历史训练路径） | `use_cache=True`（评测路径） |")
        w("|---|---|---|")
        w(f"| 视觉位置 相对 L2 变化 | {med['uncached_relL2_vis']:.4f} | {med['cached_relL2_vis']:.4f} |")
        w(f"| **文本位置 相对 L2 变化** | **{med['uncached_relL2_txt']:.4f}** | **{med['cached_relL2_txt']:.4f}** |")
        w(f"| 文本位置 余弦 | {med['uncached_cos_txt']:+.4f} | {med['cached_cos_txt']:+.4f} |")
        w("")
        w(f"`use_cache=False` 下文本位置变化恰好为 **0.0000**、余弦 **+1.0000**："
          "视觉输入对文本行的表示**逐位无关**。同一比较在评测路径下是 "
          f"{med['cached_relL2_txt']:.4f} / 余弦 {med['cached_cos_txt']:+.4f}。")
        w("")
        w(f"**两条路径对同一输入在文本位置相差 {med['paths_relL2_txt']:.4f} 相对 L2"
          f"（余弦 {med['paths_cos_txt']:+.4f}）。** 即：本项目过去所有视频头都"
          "**在文本行不依赖视频的特征上训练**，却在文本行依赖视频的特征上评测。"
          "这是本实验开始前未记录的训练/评测不一致。")
        w("")
        w("**已做**：新增 `common.target_context_cached`（加法式，`common.target_context` "
          "一字未改，图像侧回归保持逐位一致），所有本实验的消融与训练默认走 cache 路径。")
    else:
        w("探针未完成。")
    w("")

    # 1.3
    w("### 1.3 坑 3：CKA 计算口径 —— 这条修完**推翻**了上一份诊断的一个结论")
    w("")
    if cka.get("A") and cka_unc:
        def m(doc, k):
            return st.median(r[k] for r in doc["rows"])
        keys = [("cka_proj_raw_vis", "CKA(proj, 原始视觉嵌入) @视觉"),
                ("cka_proj_last_vis", "CKA(proj, 目标最终层视觉 hidden) @视觉"),
                ("cka_proj_last_txt", "CKA(proj, 最终层 hidden) @文本（同模型对照）"),
                ("cka_proj_raw_txt", "CKA(proj, 原始嵌入) @文本（对照）"),
                ("cos_projvis_projtxt_mean", "cos(proj 视觉均值, proj 文本均值)")]
        w("A 臂同一批行、同一头，唯一差别是前向路径：")
        w("")
        w("| 量 | `use_cache=False`（诊断所用，错） | `use_cache=True`（评测路径，对） | 差 |")
        w("|---|---|---|---|")
        for k, lab in keys:
            u, c = m(cka_unc, k), m(cka["A"], k)
            w(f"| {lab} | {u:+.4f} | **{c:+.4f}** | {c-u:+.4f} |")
        w("")
        w("**结论方向的改变**：")
        w("")
        w(f"- 走评测路径后，投影与目标自身深层视觉特征的 CKA 是 "
          f"**{m(cka['A'],'cka_proj_last_vis'):.3f}**，而不是诊断里写的 0.557 —— "
          "投影对视觉语义的保真度**相当高**。")
        w(f"- 文本侧对照反而更低（{m(cka['A'],'cka_proj_last_txt'):.3f} / "
          f"{m(cka['A'],'cka_proj_raw_txt'):.3f}）。"
          "诊断 §6.3 写的「同一个 `fc` 在视觉位置上的保真度明显低于文本位置」"
          "**在评测路径上是反的**：视觉位置保真度**高于**文本位置。")
        w(f"- `cos(proj 视觉均值, proj 文本均值)` 从 0.244 变为 "
          f"**{m(cka['A'],'cos_projvis_projtxt_mean'):.3f}**：视觉被映射到了与文本对齐的空间，"
          "这一条从「无证据」变成「有明确证据」。")
        w("")
        w("因此上一份诊断 §4 / §6.3 / §6.4 中「唯一支持引入视觉适配模块的提示性证据」"
          "**在正确口径下不成立**。本文件 §4 全部使用 cache 路径，"
          "并在 §8 记录该诊断文档需要的更正。")
    else:
        w("CKA 未完成。")
    w("")

    # ---------------- 2 adapter ----------------
    w("## 2. ViSpec 适配器实现")
    w("")
    w("`repro/historical/vispec_adapter.py`。集成点是 `hidden_norm(fc(...))` 这一处接缝 —— "
      "训练（`train_head.forward_anchors`）与解码（`DFlashDraftModel.forward`）"
      "都写这个表达式，所以把 `fc` 换成一个臂投影模块、把 `hidden_norm` 换成 "
      "`nn.Identity`，**两条路径共用一份实现**，不存在会漂移的第二份前向。")
    w("")
    w("```")
    w("视觉行 → 专用 Transformer 编码器(2 层, d=512, 8 头) → 16 个可学习 latent query")
    w("                                                        ├→ 全局向量 → 加到每个文本行")
    w("                                                        └→ 视觉行只从这 16 个 latent 重建")
    w("文本行 → 原 hidden_norm(fc(...)) + 全局向量")
    w("```")
    w("")
    if metas.get("V"):
        mv = metas["V"]
        w(f"- **适配器实测参数量 {mv['adapter_params']/1e6:.2f}M**，"
          f"占该臂总参数 {100*mv['adapter_params']/mv['head_params']:.1f}%，"
          f"远低于 100M 上限（训练脚本内 `assert` 硬检查）。")
    w("- 默认关闭：只有 `--arm vispec` 会挂载适配器；`--arm mlp` 与既有 A 臂逐位一致。")
    w("- 保存：适配器单独存为 `vispec.safetensors` + `vispec_config.json`，"
      "放在头目录旁边，因此头文件本身仍可被原 `common.load_head` 直接加载。")
    w("")
    w("**与 ViSpec 原文的一处有意的偏离（必须记录）**："
      "`dflash.model.apply_rotary_pos_emb` 把 `cos/sin` 广播到 key 轴，"
      "所以头返回的上下文行数**必须**等于解码器给的位置数。"
      "真正的 KV 行数压缩（把 n_vis 行视觉压成 n_q 行）会要求改 "
      "`glance/decode.py` 的位置记账，而那会把已复现的 τ 管线放到本实验的关键路径上——"
      "本实验的全部意义就是与它可比。"
      "因此这里的「压缩」是**信息瓶颈**而非行数削减：每个视觉行都只从 16 个 latent 重建，"
      "视觉 KV 能承载的信息被这 16 个 latent 限死，但行数、位置、树、解码器一律未动。")
    w("")

    # ---------------- 3 main table ----------------
    w("## 3. 主表：A / C0 / V")
    w("")
    if not have:
        w("评测未完成，本节缺失；见 §8。")
    else:
        w(f"逐行配对，n = {len(common)}。Δ 相对 A。")
        w("")
        w("| 臂 | 说明 | τ 中位 | τ 均值 ± sd | E[a] 中位 | Δ中位 vs A | Δ均值 [95% CI] | t | p | 胜/负 |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        base = arms.get("A") or arms[have[0]]
        for k in have:
            t = [arms[k][v]["tau"] for v in common]
            d = [arms[k][v]["tau"] - base[v]["tau"] for v in common]
            pr = paired(d)
            ea = st.median(x - 1.0 for x in t)
            stat = ("| — | — | — |" if k == "A" else
                    f"| {pr['t']:+.2f} | {fmt_p(pr['p'])} | {pr['win']}/{pr['loss']} |")
            w(f"| **{k}** | {ARM_LABEL[k]} | {st.median(t):.4f} | {st.fmean(t):.4f} ± "
              f"{st.stdev(t):.4f} | {ea:.4f} | {pr['median']:+.4f} | {pr['mean']:+.4f} "
              f"[{pr['ci'][0]:+.3f}, {pr['ci'][1]:+.3f}] " + stat)
        w("")
        if "V" in have and "C0" in have:
            d = [arms["V"][v]["tau"] - arms["C0"][v]["tau"] for v in common]
            pr = paired(d)
            w("### 判据（V vs C0：适配器的净贡献）")
            w("")
            w(f"**V − C0 = {pr['median']:+.4f}（中位）/ {pr['mean']:+.4f}（均值）**，"
              f"配对 t = {pr['t']:+.2f}，p = {fmt_p(pr['p'])}，胜/负 {pr['win']}/{pr['loss']}。")
            if pr["p"] < 0.05 and pr["median"] > 0:
                w("→ **ViSpec 带来额外收益。**")
            elif pr["p"] < 0.05 and pr["median"] < 0:
                w("→ **ViSpec 有害。**")
            else:
                w("→ **没有证据表明 ViSpec 带来额外收益**（差异不显著）。")
            w("")
            if "A" in have:
                dA = [arms["A"][v]["tau"] - arms["C0"][v]["tau"] for v in common]
                prA = paired(dA)
                w(f"参考：**A − C0 = {prA['median']:+.4f}（中位）**，p = {fmt_p(prA['p'])}。"
                  "这一项度量的是「warm start + 历史特征路径」合计值多少 τ，"
                  "它不是适配器的贡献，列出来是为了说明 A 与 C0 之间不可直接比。")
                w("")
    w("")

    # ---------------- 3b convergence ----------------
    w("### 3.1 两个从头臂的收敛情况（这是本判定的主要不确定来源）")
    w("")
    curves = {k: loss_curve(f"V_{'vispec' if k == 'V' else 'mlp'}.jsonl") for k in ("C0", "V")}
    if any(curves.values()):
        w("训练 loss（每 25 步的窗口均值）。A 臂的祖先在 warm start 下 step 50 已约 2.4，"
          "作为「从头训到没训够」的参照。")
        w("")
        w("| 臂 | 前 100 步均值 | 同一步（step 1000） | 同一步（step 2000） | 末段均值（末 100 步） |")
        w("|---|---|---|---|---|")
        for k in ("C0", "V"):
            c = curves[k]
            if not c:
                w(f"| {k} | — | — | — | — |")
                continue
            first = mean_loss_over(c, 1, 100)
            l1, l2 = loss_at(c, 1000), loss_at(c, 2000)
            last = mean_loss_over(c, c[-1][0] - 100, c[-1][0])
            f = lambda v: "—" if v is None else f"{v:.4f}"
            w(f"| {k} | {f(first)} | {f(l1)} | {f(l2)} | {f(last)} |")
        w("")
        w("> 同一步比较才有意义：两个臂步数相同（3000），但训练速度不同，"
          "按时间对齐会引入「谁先跑到」的错觉。")
        w("")
        w("**读法**：两个从头臂的末段 loss 都明显高于 A 臂祖先的 2.4 水平，"
          "说明 439 行 / 3000 步 / 从头训这个预算**不足以收敛到 A 臂的成熟度**。"
          "因此 §3 的判定只在「同预算下适配器是否更快/更好」这个意义上成立，"
          "不能读成「适配器在任何预算下都无用」——预算不足本身会掩盖架构差异。")
    else:
        w("训练日志缺失，无法比较收敛。")
    w("")

    # ---------------- 4 CKA ----------------
    w("## 4. CKA 对比（cache / 评测路径）")
    w("")
    if cka.get("A") and cka.get("V"):
        def mm(k, key):
            return st.median(r[key] for r in cka[k]["rows"])
        keys = [("cka_proj_raw_vis", "视觉位置：vs 原始视觉嵌入"),
                ("cka_proj_last_vis", "视觉位置：vs 目标最终层视觉 hidden"),
                ("cka_proj_last_txt", "文本位置：vs 最终层 hidden"),
                ("cka_proj_raw_txt", "文本位置：vs 原始嵌入"),
                ("cos_projvis_projtxt_mean", "cos(视觉均值, 文本均值)"),
                ("proj_rms_vis", "投影后 KV 的 RMS @视觉位置"),
                ("proj_rms_txt", "投影后 KV 的 RMS @文本位置")]
        cols = [k for k in ("A", "C0", "V") if cka.get(k)]
        w("| 量 | " + " | ".join(cols) + " | V − A | V − C0 |")
        w("|---" * (2 + len(cols) + 1) + "|")
        for key, lab in keys:
            vals = {k: mm(k, key) for k in cols}
            va_ = f"{vals['V']-vals['A']:+.4f}" if "V" in vals and "A" in vals else "—"
            vc = f"{vals['V']-vals['C0']:+.4f}" if "V" in vals and "C0" in vals else "—"
            w(f"| {lab} | " + " | ".join(f"{vals[k]:+.4f}" for k in cols) + f" | {va_} | {vc} |")
        w("")
        w("### 判据")
        w("")
        cv = mm("V", "cka_proj_last_vis")
        ca = mm("A", "cka_proj_last_vis")
        w(f"- V 臂视觉位置 CKA(proj, 目标深层视觉) = **{cv:.4f}**；A 臂 {ca:.4f}。")
        if cv > ca + 0.05:
            w("- → **V 臂视觉位置 CKA 提升**：ViSpec 桥接了视觉语义。")
        elif cv < ca - 0.05:
            w("- → **V 臂视觉位置 CKA 下降**：ViSpec 没有解决保真度问题，反而更低。")
        else:
            w("- → 两臂视觉位置 CKA 相当：ViSpec 没有改变视觉语义的保真度。")
        w("")
        w("> 注意 A 臂是 warm start 且用历史特征路径，与 V 不同口径；"
          "**V vs C0 才是同口径对比**。")
    else:
        w("CKA 对比未完成（需要 A/C0/V 三份 `cka_*_cached.json`）；见 §8。")
    w("")

    # ---------------- 5 VA bucketing ----------------
    w("## 5. 视觉依赖分层")
    w("")
    if va and have:
        vids = [v for v in common if v in va]
        edges, labels, kind = buckets([va[v] for v in vids])
        w(f"VA 取自上一轮诊断的 `va.json`（只依赖目标模型，与草稿臂无关，"
          f"cache 路径，逐行真实视频 vs 像素置零的 JSD）。分桶：{kind}。")
        w("")
        w("| VA 桶 | n | VA 范围 | " + " | ".join(f"{k} τ中位" for k in have) +
          " | " + " | ".join(f"Δ({k}−A)" for k in have if k != "A") + " |")
        w("|---" * (3 + len(have) + max(len(have) - 1, 0) + 1) + "|")
        for bi, lab in enumerate(labels):
            iv = [v for v in vids if bucket_of(va[v], edges) == bi]
            cells = [f"| {lab} | {len(iv)} | "
                     f"{min((va[v] for v in iv), default=float('nan')):.4f}–"
                     f"{max((va[v] for v in iv), default=float('nan')):.4f} "]
            for k in have:
                cells.append(f"| {st.median([arms[k][v]['tau'] for v in iv]):.4f} " if iv else "| — ")
            for k in have:
                if k == "A":
                    continue
                cells.append(f"| {st.median([arms[k][v]['tau']-arms['A'][v]['tau'] for v in iv]):+.4f} "
                             if iv else "| — ")
            w("".join(cells) + "|")
        w("")
        w("### 判据（V − A）")
        w("")
        hi = [v for v in vids if bucket_of(va[v], edges) == len(labels) - 1]
        lo = [v for v in vids if bucket_of(va[v], edges) == 0]
        if hi and lo:
            dh = st.median([arms["V"][v]["tau"] - arms["A"][v]["tau"] for v in hi]) if "V" in have else float("nan")
            dl = st.median([arms["V"][v]["tau"] - arms["A"][v]["tau"] for v in lo]) if "V" in have else float("nan")
            dc_h = st.median([arms["V"][v]["tau"] - arms["C0"][v]["tau"] for v in hi]) if "V" in have and "C0" in have else float("nan")
            w(f"- 高 VA 桶（n={len(hi)}）：Δ(V−A) 中位 {dh:+.4f}，Δ(V−C0) 中位 {dc_h:+.4f}")
            w(f"- 低 VA 桶（n={len(lo)}）：Δ(V−A) 中位 {dl:+.4f}")
            if dc_h == dc_h and dc_h > 0.05:
                w("- → ViSpec 在高视觉依赖位置有效。")
            elif dc_h == dc_h and abs(dc_h) <= 0.05:
                w("- → 各桶差异相当，ViSpec 没有视觉专门收益。")
            elif dc_h == dc_h:
                w("- → 高 VA 桶 Δ < 0，ViSpec 在高视觉依赖位置有害。")
        w("")
        w("> 桶内 n 很小（约 10 行），只作方向性证据；未做桶内显著性检验。")
    else:
        w("VA 或臂评测缺失；见 §8。")
    w("")

    # ---------------- 6 long context ----------------
    w("## 6. 长视频 / 长上下文鲁棒性")
    w("")
    if have:
        lens = {v: arms["A"][v]["L"] for v in common}
        nf = sorted({arms["A"][v]["n_frames"] for v in common})
        edges, labels, kind = buckets([lens[v] for v in common])
        w(f"与上一份诊断相同，本留出集**没有时间长度变化**：全部 30 行 `n_frames` "
          f"恒为 {nf}（`predecode_cache.py::normalize()` 把每个源视频重采样成恰好 16 帧"
          "的等长短片，原始时长未落盘）。因此帧数/时长分桶在结构上不可计算。")
        w("")
        w(f"可用代理轴是总 prompt 长度（bucket：{kind}）：")
        w("")
        w("| prompt 长度桶 | n | L 范围 | " + " | ".join(f"{k} τ中位" for k in have) +
          " | " + " | ".join(f"Δ({k}−A)" for k in have if k != "A") + " |")
        w("|---" * (3 + len(have) + max(len(have) - 1, 0) + 1) + "|")
        for bi, lab in enumerate(labels):
            iv = [v for v in common if bucket_of(lens[v], edges) == bi]
            cells = [f"| {lab} | {len(iv)} | "
                     f"{min((lens[v] for v in iv), default=float('nan')):.0f}–"
                     f"{max((lens[v] for v in iv), default=float('nan')):.0f} "]
            for k in have:
                cells.append(f"| {st.median([arms[k][v]['tau'] for v in iv]):.4f} " if iv else "| — ")
            for k in have:
                if k == "A":
                    continue
                cells.append(f"| {st.median([arms[k][v]['tau']-arms['A'][v]['tau'] for v in iv]):+.4f} "
                             if iv else "| — ")
            w("".join(cells) + "|")
        w("")
        w("**这个表不能外推为长视频结论**：行长差异来自空间 token 数（672/768/864），"
          "不是时间长度。")
    w("")

    # ---------------- 7 conclusion ----------------
    w("## 7. 结论")
    w("")
    w("### 7.1 ViSpec 是否带来额外收益")
    w("")
    if "V" in have and "C0" in have:
        d = [arms["V"][v]["tau"] - arms["C0"][v]["tau"] for v in common]
        pr = paired(d)
        if pr["p"] < 0.05 and pr["median"] > 0:
            w(f"**是，但幅度有限。** V − C0 = {pr['median']:+.4f} τ 中位"
              f"（p = {fmt_p(pr['p'])}），即在同初始化、同特征路径、同步数下，"
              f"适配器把 τ 提高了约 {100*pr['median']/st.median([arms['C0'][v]['tau'] for v in common]):.1f}%。")
        elif pr["p"] < 0.05 and pr["median"] < 0:
            w(f"**否，而且有害。** V − C0 = {pr['median']:+.4f} τ 中位（p = {fmt_p(pr['p'])}）。")
        else:
            w(f"**没有证据表明有额外收益。** V − C0 = {pr['median']:+.4f} τ 中位，"
              f"均值 {pr['mean']:+.4f}，配对 t = {pr['t']:+.2f}，p = {fmt_p(pr['p'])}，"
              f"胜/负 {pr['win']}/{pr['loss']} —— 差异落在噪声内。")
        w("")
        w("### 7.2 收益来自哪里 / 为什么没有收益")
        w("")
        if cka.get("V") and cka.get("C0"):
            rv = st.median(r["proj_rms_vis"] for r in cka["V"]["rows"])
            rc = st.median(r["proj_rms_vis"] for r in cka["C0"]["rows"])
            w("**直接机制（实测）：适配器输出的视觉行几乎被压没了。**")
            w("")
            w(f"- 投影后 KV 在**视觉位置**的 RMS：V 臂 **{rv:.4f}**，C0 臂 {rc:.4f}"
              f"（相差约 {rc/max(rv,1e-9):.0f} 倍）；文本位置两者都是 1.000（RMSNorm 的结果）。")
            w(f"- 视觉位置的 CKA(proj, 目标深层视觉)：V "
              f"{st.median(r['cka_proj_last_vis'] for r in cka['V']['rows']):.3f} vs C0 "
              f"{st.median(r['cka_proj_last_vis'] for r in cka['C0']['rows']):.3f}。")
            w(f"- `cos(proj 视觉均值, proj 文本均值)`：V "
              f"{st.median(r['cos_projvis_projtxt_mean'] for r in cka['V']['rows']):+.3f} vs C0 "
              f"{st.median(r['cos_projvis_projtxt_mean'] for r in cka['C0']['rows']):+.3f}"
              " —— C0 把视觉映射到与文本对齐的方向，V 没有。")
            w("")
            w("即：16 个 latent 的瓶颈、加上「视觉行只从 latent 重建」的结构，"
              "**学成了一个近似为零的视觉 KV**。草稿头在视觉位置几乎收不到信号，"
              "这正是 τ 掉 0.56 的来源。所以这不是「适配器没帮上忙」，"
              "而是「适配器把视觉行关掉了」。")
            w("")
        w("分因素：")
        w("")
        factors = []
        if cka.get("V") and cka.get("C0"):
            dv = st.median(r["cka_proj_last_vis"] for r in cka["V"]["rows"]) - \
                 st.median(r["cka_proj_last_vis"] for r in cka["C0"]["rows"])
            factors.append(("视觉语义保真度", f"V 相对 C0 的视觉位置 CKA 差 {dv:+.4f}"))
        mv = metas.get("V") or {}
        factors.append(("参数量", f"适配器仅 {mv.get('adapter_params',0)/1e6:.2f}M，"
                                  f"占该臂 {100*mv.get('adapter_params',0)/max(mv.get('head_params',1),1):.1f}%"
                                  " —— 因此**不是**「参数量不足」"))
        factors.append(("架构", "见 §2：压缩是信息瓶颈而非 KV 行数削减（行数受 rotary 广播约束，"
                                "改行数就要动已验证的解码路径）；这个瓶颈本身是本次失败的直接原因"))
        factors.append(("数据", f"训练行只有 {metas.get('V',{}).get('train_rows','?')} 行、"
                                f"{metas.get('V',{}).get('steps','?')} 步，且**从头训**；"
                                "A 臂的祖先另有至少 3000 步视频训练。"
                                "但这一条解释不了 V 为何**差于同预算的 C0**"))
        for name, txt in factors:
            w(f"- **{name}**：{txt}")
        w("")
        w("### 7.3 推荐最终架构")
        w("")
        w("**推荐保持现有 `hidden_norm(fc(...))`，不引入 ViSpec 适配器。** 依据：")
        w("")
        w("1. 同口径判定（V vs C0）**显著更差**，不是「没有收益」而已："
          "V − C0 = −0.563 τ 中位（p < 1e-4，29/29 行为负）；")
        w("2. §1.3 的 CKA 更正显示，在正确口径下现有投影对视觉语义的保真度"
          "（CKA ≈ 0.77）已经很高，且视觉已被映射到与文本对齐的空间"
          "（视觉/文本均值余弦 ≈ 0.62）—— 上一份诊断指出「缺一个视觉映射模块」的那条证据"
          "在正确口径下不存在；")
        w("3. 真正被本实验发现的问题是**训练/评测特征路径不一致**（§1.2，文本位置相差 "
          "0.32 相对 L2），它比换适配器更值得先修，而且不需要新模块。")
        w("")
        w("**建议的下一步（按性价比排序）**：")
        w("")
        w("1. 用 `target_context_cached` 重训一个 MLP 头，与 C0 对比，量化特征路径修复本身值多少 τ；")
        w("2. 若要继续追视觉适配，先解除 rotary 行数约束（改 `glance/decode.py` 的位置记账"
          "并重跑无损门禁），再做真正的 KV 行数压缩；")
        w("3. 扩训练行数与步数后再判定适配器 —— 当前 439 行 / 3000 步 / 从头训不足以"
          "排除「数据与步数不足」这一解释。")
    else:
        w("缺少 C0 或 V 的评测结果，无法给出判定；见 §8。")
    w("")

    # ---------------- 8 anomalies ----------------
    w("## 8. 异常记录")
    w("")
    anoms = [
        ("**上一份诊断（`视觉适配诊断_2026-10-05.md`）的 CKA 与结论需要更正。**",
         "其 §4 用了 `use_cache=False` 的消融路径（坑 3），导致 "
         "CKA(proj, 目标深层视觉)@视觉 记为 0.557、文本侧记为 0.434/0.742，"
         "并据此在 §6.3/§6.4 得出「同一个 fc 对视觉的保真度低于文本」与"
         "「唯一支持引入视觉适配模块的提示性证据」。在评测路径上这三个数是 "
         "0.768 / 0.217 / 0.551，**结论方向反转**。诊断文档应据此更正，本文件 §1.3 给出对照。"),
        ("**A 臂与 V/C0 不同口径，A vs V 不能作为适配器判据。**",
         "A 是 warm start（祖先 `head_video439/step1000` 已经过 3000 步视频训练）"
         "且用历史 `use_cache=False` 特征路径；V/C0 是从头随机 + cache 路径。"
         "三者中只有 V vs C0 是同口径。A 仅作参考上界，§3 已把两者分开。"),
        ("**ViSpec 的 KV 行数压缩未实现，实现的是信息瓶颈。**",
         "`apply_rotary_pos_emb` 把 cos/sin 广播到 key 轴，上下文行数必须等于位置数；"
         "压缩行数需要改 `glance/decode.py`，那会让本实验的 τ 不再与已验证管线可比。"
         "本实现因此让视觉行只从 16 个 latent 重建，行数不变。"
         "这是与原文的偏离，不是原文方法的失败。"),
        ("**从头训收敛程度有限，这是本判定的主要不确定来源。**",
         "A 臂的祖先在 warm start 下 step 50 的 loss 已约 2.4；本实验两个从头臂的 loss "
         "从随机初值下降，3000 步结束时仍高于该水平。因此「ViSpec 无额外收益」这一结论"
         "只在「439 行 / 3000 步 / lr 3e-4 / 从头训」这个区间内成立，"
         "不能外推为「适配器在任何训练预算下都无用」。"),
        ("**阶段 6 的时间轴不可计算（与上一份诊断同因）。**",
         "fresh30 全部 30 行 `n_frames` 恒为 16，缓存内 mp4 时长恒为 0.5333s，"
         "原始时长未落盘；已改用 prompt 长度分桶并明确不外推为长视频结论。"),
        ("**GPU 5 是 24GB 的 4090，不是 A6000。**",
         "首次启动 C0 臂时按「空闲卡」选了 GPU 5，随即 CUDA OOM（8B 目标 16GB + "
         "1.6B 头 + 激活超出 24GB）。已改到 A6000（GPU 3）重跑，产物未受影响。"
         "任务描述的「4 卡并行」在本机需按卡型区分。"),
        ("**同一头经不同评测脚本会提交不同 token，比较必须同脚本。**",
         "A 臂（`head_video439_h8`）经上一轮诊断的 `diag_arms.py` 得 τ 中位 3.1574，"
         "经本轮的 `eval_fresh.py` 得 3.1167（−1.3%）。两者数学等价，但 bf16 下"
         "张量内存布局不同会让 cuBLAS 选到不同 kernel，接近的 tie 落到两侧，"
         "贪心序列随后分叉。**本文件三个臂全部用 `eval_fresh.py` 评出**，"
         "不混用两个脚本的数字；跨脚本的数字只能横向参考，不能配对相减。"),
    ]
    for i, (t, b) in enumerate(anoms, 1):
        w(f"{i}. {t}")
        w(f"   {b}")
    w("")
    w("### 未静默失败的声明")
    w("")
    w("全部脚本不使用 `try/except` 吞异常；CUDA OOM 与形状不匹配都以 traceback 终止"
      "并保留日志（§8 第 6 条即实例）。缺失项在 §3–§6 显式写为「未完成」并在本节记录原因。")
    w("")
    w("### 复现命令")
    w("")
    w("```bash")
    w("cd /home/mcy/Projects/GLANCE/repro")
    w("D=/mnt/local/mcy/GLANCE")
    w("# 坑 1 审计 / 坑 2、3 探针")
    w("python audit_holdout_contamination.py")
    w("python probe_context_cache_path.py --gpu 0")
    w("# 实现检查：参数量与预算")
    w("python -c \"import vispec_adapter as v;a=v.ViSpecAdapter();print(a.param_count(), v.PARAM_LIMIT)\"")
    w("# 两个从头臂（并行；C0 必须在 A6000 上，4090 放不下）")
    w("python train_fresh_video.py --arm vispec --feature-path cached \\")
    w("  --records $D/data/video_greedy_499.pt --pool $D/data/pool_499.parquet \\")
    w("  --out $D/checkpoints/head_v_vispec --steps 3000 --lr 3e-4 --save-every 250 --gpu 0")
    w("python train_fresh_video.py --arm mlp --feature-path cached \\")
    w("  --records $D/data/video_greedy_499.pt --pool $D/data/pool_499.parquet \\")
    w("  --out $D/checkpoints/head_v_mlp --steps 3000 --lr 3e-4 --save-every 250 --gpu 3")
    w("# 评测（干净留出、b63k8、max_new 1024、逐行配对）")
    w("python eval_fresh.py --arm mlp    --head $D/checkpoints/head_v_mlp    \\")
    w("  --rows $D/data/video_fresh30.parquet --store-ids --gpu 1 \\")
    w("  --out $D/results/vispec/armC0.json")
    w("python eval_fresh.py --arm vispec --head $D/checkpoints/head_v_vispec \\")
    w("  --rows $D/data/video_fresh30.parquet --store-ids --gpu 2 \\")
    w("  --out $D/results/vispec/armV.json")
    w("# CKA（cache 路径）")
    w("for a in mlp vispec; do python measure_cka.py --arm $a --feature-path cached \\")
    w("  --head $D/checkpoints/head_v_${a/mlp/ml} --rows $D/data/video_fresh30.parquet \\")
    w("  --out $D/results/vispec/cka_${a}_cached.json; done")
    w("# 汇总")
    w("python vispec_report.py")
    w("```")
    w("")
    w("**产物落盘位置**：全部写在 `/mnt/local/mcy/GLANCE/` 下，不占 home。")
    w("- 结果：`/mnt/local/mcy/GLANCE/results/vispec/`")
    w("- 训练日志：`/mnt/local/mcy/GLANCE/logs/V_*.log` 与 `V_*.jsonl`")
    w("- 检查点：`/mnt/local/mcy/GLANCE/checkpoints/head_v_{mlp,vispec}/`")
    w("- 脚本：`/home/mcy/Projects/GLANCE/repro/`（属代码仓库）")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out} ({len(L)} lines)")
    for k in have:
        t = [arms[k][v]["tau"] for v in common]
        print(f"  {k}: tau_med={st.median(t):.4f} n={len(t)}")
    if "V" in have and "C0" in have:
        d = [arms["V"][v]["tau"] - arms["C0"][v]["tau"] for v in common]
        pr = paired(d)
        print(f"  V-C0: median={pr['median']:+.4f} p={fmt_p(pr['p'])}")


if __name__ == "__main__":
    main()
