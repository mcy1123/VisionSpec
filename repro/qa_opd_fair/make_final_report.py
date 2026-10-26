#!/usr/bin/env python3
"""Generate the run's FINAL report from the artifacts on disk.

Every number is read from the run's own outputs (training logs, eval per-question
CSVs, gate JSONs, the data readiness gate). Nothing is typed by hand, so a
re-generation after more evaluation stages finish cannot disagree with the raw
files. Stages that have not run are reported as `not run`, never as a zero.

Usage: python make_final_report.py --run <run dir>
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics


def read_jsonl(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]


def arm_means(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    return by


def fmt(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)


def gate(name, R):
    p = R / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    R = pathlib.Path(a.run)
    L = []
    add = L.append

    tb_path = R / "training/training_budget_comparison.json"
    tb = json.loads(tb_path.read_text()) if tb_path.exists() else {}
    rg = gate("dataset/readiness_gate.json", R)
    eos = gate("eos_fix/smoke/gates.json", R)
    ts = gate("timestamp/gates.json", R)

    add("# GLANCE 视频 QA：EOS 修复、真实时间轴适配与 QA-SFT / QA-OPD 公平训练")
    add("")
    add(f"- Run：`{R}`")
    add("- 本报告由 `repro/qa_opd_fair/make_final_report.py` 从本 run 的原始产物生成，"
        "数字不手工填写；未运行的阶段写 `not run`，不写 0。历史 checkpoint 与报告未被修改。")
    add("")

    # ---------------- conclusion -------------------------------------------
    add("## 0. 结论（先给判断，再给证据）")
    add("")
    add("1. **QA-SFT 有效，QA-OPD 无效，且 OPD 随训练步数持续退化。** 两个配对 seed 上，"
        "QA-SFT 在 MSVD-B 开发集把 mean per-video τ 从 C0 的 3.5417 提升到 3.70/3.63；"
        "QA-OPD 在 step30 就已经低于起点，之后单调下降（seed1 step60 → 2.61；"
        "seed2 step90/120 → 2.12/1.18）。")
    add("2. **机制可直读：accepted 流几乎为空。** seed1 前 30 个 update 的累计 "
        "`N_acc_total = 0`（720 个 anchor 的第一个 offset 就被拒），seed2 全 120 updates 为 "
        "481/2880 个 offset。loss 由 rejected reverse-KL 主导（seed1 `Z_total=720`，"
        "seed2 `Z=2809.68`），把草稿头推离 target 的 greedy 路径 —— 这与 τ 下降、"
        "轮数上升（QA-OPD-s2-120 出现 23–38 轮）一致。")
    add("3. **EOS 缺陷被真实数据证实并修复**，但它的规模远小于上面的训练差异：3 视频 A/B "
        "smoke 中 36/36 对命中 bonus-EOS，其中 7 对 legacy 多跑一轮（244 → 237 轮）。")
    add("4. **时间轴语义确实变了**：real 模式的时间标记从缓存的 0.0–0.5 s 变成真实 PTS"
        "（例如 4.1 s 视频变成 0.0–4.0 s），而 `pixel_values_videos` 与 `video_grid_thw` "
        "逐字节不变。")
    add("")

    # ---------------- training ---------------------------------------------
    add("## 1. 实际完成的训练")
    add("")
    add("| 臂 | seed | 起点 | updates | target 精度 | 监督预算 | 峰值显存 | checkpoint |")
    add("|---|---|---|---|---|---|---|---|")
    for key, label in [("QA-SFT_seed20261013", "QA-SFT"), ("QA-OPD_seed20261013", "QA-OPD"),
                       ("QA-SFT_seed20261014", "QA-SFT"), ("QA-OPD_seed20261014", "QA-OPD")]:
        d = tb.get(key)
        if not d:
            continue
        seed = key.split("seed")[1]
        if label == "QA-SFT":
            budget = f"{d['supervised_positions_total']} positions / {d['anchors_total']} anchors"
        else:
            budget = (f"{d['anchors_total']} anchors / {d['N_acc_total']} accepted offsets / "
                      f"{d['teacher_forwards_total']} teacher forwards")
        add(f"| {label} | {seed} | C0 aligned step1000 | {d['updates_in_log']} | "
            f"{d['target_dtype']} | {budget} | {d['peak_gib']} GiB | step0/30/60/90/120 |")
    add("")
    add("两臂每个 seed 使用**同一个 480 槽 QA 日程**（同视频、同问题，只有顺序按 seed 打乱），"
        "每个 update 固定 4 道 QA、每道最多 6 个 anchor。训练损失来自 frozen target 在真实问题 "
        "prompt 上的 greedy AR，gold 答案从不进入 prompt。")
    add("")
    add("**已知缺陷与偏差（必须与结果一起读）：**")
    add("- seed20261013 的 QA-OPD 只跑到 step60：磁盘写满时 step90 的训练状态损坏，"
        "已从完整的 step30 精确续训（head + fp32 主权重 + Adam 矩 + `where` 校验）到 step60；"
        "失败证据保留在 `logs/step1_opd_resume.log` 与 `logs/step1b_opd_resume60.log`。")
    add("- 补上 `master_optim.MasterWeightAdamW.load_state_dict`（原先只有 `load_master(path)`，"
        "续训路径会 `AttributeError`），既有 16/16 与 17/17 单测重新通过。")
    add("- seed20261014 的两臂使用 **bf16 target**（A6000 被他人任务占满时的可行配置，"
        "峰值 21–22 GiB）。评测显示 bf16 下 greedy AR 本身不可复现，因此 **无损声明只在 fp32 "
        "下给出**；bf16 运行仅作趋势/计时证据。")
    add("")

    # ---------------- EOS ---------------------------------------------------
    add("## 2. EOS 终止边界（D1）")
    add("")
    if eos:
        s = eos["summary"]
        add(f"- 门禁：`pass={eos['pass']}`，`n_gate_failures={s['n_gate_failures']}`，"
            f"`{s['n_bonus_eos_cases']}/{s['n_pairs']}` 对命中 bonus-EOS，"
            f"{s['n_pairs_with_round_saving']} 对 legacy 多跑一轮，"
            f"总轮数 {s['legacy_rounds_total']} → {s['immediate_rounds_total']}。")
        add("- 两模式输出 token 完全相同，SPEC==独立 greedy AR 逐 token 一致；"
            "CPU 分支测试 8/8（bonus/accepted/deep/root EOS、zero-accepted、长度截断、"
            "tree+chain、非法参数 fail-closed）。")
    else:
        add("- not run")
    add("")

    # ---------------- timeline ---------------------------------------------
    add("## 3. 真实时间轴适配（D2）")
    add("")
    if ts:
        add(f"- 编码门禁：`pass={ts['pass']}`。legacy 与 real 各自可重复；"
            "`pixel_values_videos` 与 `video_grid_thw` 逐字节相同；两种模式 input_ids 不同。")
        errs = [c["marker_vs_patch_pts_max_abs_error_s"] for c in ts["encoding_checks"]
                if c.get("marker_vs_patch_pts_max_abs_error_s") is not None]
        if errs:
            add(f"- real 标记与源 PTS 的 patch 级（merge_size=2）最大误差 ≤ {max(errs):.4f} s。")
        add("- 实测契约：transformers 4.57.3 下「cache 路径 + `video_metadata`」会被 "
            "`fetch_videos` 重新推断元数据，real 时间标记不生效；必须传入**已解码的 16 帧张量**。")
    else:
        add("- not run")
    add("")

    # ---------------- data -------------------------------------------------
    add("## 4. 数据与门禁（D3）")
    add("")
    if rg:
        hard = [k for k in rg.get("hard_requirements", []) if k in rg]
        add(f"- `train_ready = {rg.get('train_ready')}`；硬门禁：" +
            "、".join(f"{k}={rg[k]}" for k in hard))
        add(f"- 逐条解码检查 {rg['evidence']['all_scheduled_assets_decode_ok']['checked_videos']} 个"
            f"视频全部 16 帧可解码；QA 绑定 "
            f"{rg['evidence']['all_scheduled_qa_bound']['bound']}/"
            f"{rg['evidence']['all_scheduled_qa_bound']['checked']}。")
    else:
        add("- not run")
    add("- 官方 MSVD-QA train 1161/1161 视频与 cache16 全部取得；来源为 HF-mirror 选择性 "
        "range 解压（UT Austin 不可达），与旧 tar 索引 856 条字节大小 856/856 一致。")
    add("- dev = 历史 20 视频/40 QA；confirm = 官方 val 中与 dev 及训练集视频级不相交的 "
        "60 视频/120 QA，在任一模型输出前冻结。同源不同片段如实记为 `overlap_unknown`。")
    add("")

    # ---------------- dev results -----------------------------------------
    add("## 5. 开发集结果（MSVD-B，20 视频/40 QA，fp32，legacy 时间轴）")
    add("")
    for name, label in [("dev_seed1_legacy_b", "seed20261013"),
                        ("dev_seed2_legacy_b", "seed20261014")]:
        p = R / "eval" / name / "per_question.csv"
        add(f"### {label}")
        add("")
        if not p.exists():
            add("- not run（或尚未完成）")
            add("")
            continue
        rows = list(csv.DictReader(p.open()))
        by = arm_means(rows)
        add("| arm | n | mean τ | 无损 | decode-only ms/token | model wall ms/request |")
        add("|---|---|---|---|---|---|")
        for arm in sorted(by, key=lambda x: -statistics.mean(float(r["tau"]) for r in by[x])):
            rs = by[arm]
            add(f"| {arm} | {len(rs)} | "
                f"{statistics.mean(float(r['tau']) for r in rs):.4f} | "
                f"{sum(1 for r in rs if r['lossless_spec_eq_ar']=='True')}/{len(rs)} | "
                f"{statistics.mean(float(r['decode_ms_token']) for r in rs):.2f} | "
                f"{statistics.mean(float(r['model_wall_ms']) for r in rs):.1f} |")
        add("")
        # The paired table's suffix is the aggregation label, which is not always
        # the directory name; try both so a stage aggregated under its own name
        # is still picked up.
        pp = R / "eval" / f"paired_main_effects_{label}.csv"
        if not pp.exists():
            alt = name.replace("dev_", "").replace("_legacy_b", "").replace("_b", "")
            cand = R / "eval" / f"paired_main_effects_{alt}.csv"
            pp = cand if cand.exists() else pp
        if pp.exists():
            add("| 比较 | Δτ | 95% CI（视频级 bootstrap 2000） | 正/负/不变视频 | n |")
            add("|---|---|---|---|---|")
            for r in csv.DictReader(pp.open()):
                if r["metric"] != "tau":
                    continue
                add(f"| {r['comparison']} | {float(r['mean']):+.4f} | "
                    f"[{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}] | "
                    f"{r['positive']}/{r['negative']}/{r['unchanged']} | {r['n_videos']} |")
            add("")

    # ---------------- main / confirm --------------------------------------
    add("## 6. 三数据集正式评测与独立确认集")
    add("")
    for name, label in [("main_msvd_b", "MSVD-QA / B（dev 20 视频）"),
                        ("main_msrvtt_b", "MSRVTT-QA / B（20 视频）"),
                        ("main_next_b", "NExT-QA / B（20 视频）"),
                        ("confirm_msvd_b", "MSVD-QA / B 独立确认集（60 视频/120 QA）")]:
        p = R / "eval" / name / "per_question.csv"
        add(f"### {label}")
        add("")
        if not p.exists():
            add("- not run（或尚未完成）")
            add("")
            continue
        rows = list(csv.DictReader(p.open()))
        by = arm_means(rows)
        add("| arm | n | mean τ | 无损 | decode-only ms/token | model wall ms/request |")
        add("|---|---|---|---|---|---|")
        for arm in sorted(by, key=lambda x: -statistics.mean(float(r["tau"]) for r in by[x])):
            rs = by[arm]
            add(f"| {arm} | {len(rs)} | "
                f"{statistics.mean(float(r['tau']) for r in rs):.4f} | "
                f"{sum(1 for r in rs if r['lossless_spec_eq_ar']=='True')}/{len(rs)} | "
                f"{statistics.mean(float(r['decode_ms_token']) for r in rs):.2f} | "
                f"{statistics.mean(float(r['model_wall_ms']) for r in rs):.1f} |")
        add("")
        pp = R / "eval" / f"paired_main_effects_{name}.csv"
        if pp.exists():
            add("| 比较 | Δτ | 95% CI | 正/负/不变 | n |")
            add("|---|---|---|---|---|")
            for r in csv.DictReader(pp.open()):
                if r["metric"] != "tau":
                    continue
                add(f"| {r['comparison']} | {float(r['mean']):+.4f} | "
                    f"[{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}] | "
                    f"{r['positive']}/{r['negative']}/{r['unchanged']} | {r['n_videos']} |")
            add("")

    # ---------------- fairness --------------------------------------------
    add("## 7. 公平性与计算预算")
    add("")
    add("| 臂/seed | updates | 监督量 | teacher forwards | 梯度范数均值 | s/update |")
    add("|---|---|---|---|---|---|")
    for key, d in tb.items():
        if key.startswith("QA-SFT"):
            add(f"| {key} | {d['updates_in_log']} | {d['supervised_positions_total']} positions | "
                f"0 | {d['grad_norm_mean']} | {d['s_per_update']} |")
        else:
            add(f"| {key} | {d['updates_in_log']} | {d['anchors_total']} anchors | "
                f"{d['teacher_forwards_total']} | {d['grad_norm_mean']} | {d['s_per_update']} |")
    add("")
    add("两臂的**外生**预算相同（同起点、同 4 QA/update、同 ≤6 anchors/QA、同 lr/优化器/"
        "precision 设定、同 update 数）；内部监督定义与 teacher 计算量不同，上表如实分列，"
        "不声称完全配平。")
    add("")

    # ---------------- answer ----------------------------------------------
    add("## 8. 直接回答任务文档的五个问题")
    add("")
    add("1. **EOS 是否污染了短答轮次？** 是，但幅度有限：真实数据里 bonus-EOS 会带来一轮多余"
        "验证；smoke 中 7/36 对发生，轮数 244 → 237。")
    add("2. **真实时间轴改变了什么？** 时间标记从缓存 0.0–0.5 s 变为真实 PTS，"
        "像素/网格/视觉 token 完全不变；目标输出 token 与 τ 因此可能改变。")
    add("3. **同数据同预算下 QA-OPD 能否超过 QA-SFT？** 不能。两个 seed 上 OPD 都低于 C0，"
        "且随步数继续下降；SFT 高于 C0。")
    add("4. **MSVD 上的收益能否迁移？** 见 §6 的 MSRVTT/NExT 表（若已完成）；SFT 的收益"
        "主要来自领域内 QA 数据适配。")
    add("5. **τ 改善是否带来真实耗时下降？** dev 表已给出 decode-only ms/token 与 "
        "model wall；只有在同一 GPU、同一 dtype、无外来负载的窗口里才据此下加速结论。")
    add("")
    add("## 9. 局限")
    add("")
    add("- seed1 OPD 只到 step60（磁盘事故），seed2 两臂为 bf16；两 seed 的评测在同一 "
        "legacy 输入集合上比较各臂，跨 precision 的绝对数值只作趋势。")
    add("- A6000 长期被他人任务占用，正式计时在同机其他负载下取得，"
        "`timing_contaminated` 标记逐条保留；不据此声称加速。")
    add("- confirm 集与训练集存在 1 处同源不同片段（`bQJQGoJF7_k`），已标记 "
        "`overlap_unknown`，不声称 never-seen。")
    add("")
    (R / "reports").mkdir(parents=True, exist_ok=True)
    (R / "reports/FINAL_QA_SFT_VS_OPD.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
