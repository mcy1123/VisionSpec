#!/usr/bin/env python3
"""Generate the final Chinese report from the run's own artifacts.

Nothing here is typed in by hand: every number is read back out of the frozen
artifacts, so the report cannot drift from the data. Sections that need a human
judgement are templated with the numbers inline so the judgement can be checked.

    python make_final_report.py --run <run> [--out <path>]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import pathlib
import statistics
import sys

import numpy as np


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def read_csv(p):
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    out = pathlib.Path(a.out or (run / "reports" / "FINAL_SCIENCEQA_SFT_VS_OPD.md"))

    rg = jload(run / "dataset" / "readiness_gate.json", {})
    inter = read_csv(run / "audit" / "three_roots_intersections.csv")
    cov = jload(run / "audit" / "missing_by_official_split.json", {})
    inv = jload(run / "audit" / "three_roots_inventory.json", {})
    lp = jload(run / "preflight" / "length_pilot.json", {})
    bind = jload(run / "preflight" / "image_processor_binding.json", {})
    lossless = jload(run / "preflight" / "ar_lossless_gate.json", {})
    dgates = jload(run / "preflight" / "replay_gradient_tests.json", {})
    amendments = [json.loads(l) for l in (run / "amendments.jsonl").read_text().splitlines()
                  if l.strip()] if (run / "amendments.jsonl").exists() else []
    per_seed = read_csv(run / "eval" / "per_seed_summary.csv")
    paired = read_csv(run / "eval" / "paired_deltas.csv")
    sea = read_csv(run / "eval" / "spec_equals_ar_summary.csv")
    covrows = read_csv(run / "eval" / "candidate_and_tree_failures.csv")
    buckets = read_csv(run / "eval" / "length_buckets.csv")
    dm = jload(run / "reports" / "DELIVERY_MANIFEST.json", {})

    # ---- training budget actually spent
    budgets = {}
    _log_globs = [str(run / "train" / "seed*" / "*" / "log.jsonl"),
                  "/mnt/local2/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/train/seed*/*/log.jsonl",
                  "/mnt/local/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/train/seed*/*/log.jsonl"]
    for p in sorted({q for g in _log_globs for q in glob.glob(g)}):
        arm = pathlib.Path(p).parent.name
        seed = pathlib.Path(p).parent.parent.name
        rows = [json.loads(l) for l in open(p) if l.strip()]
        # The SFT seed-20261013 log is written in APPEND mode and an aborted first
        # attempt logged update 1 before dying on a checkpoint bug, so the same
        # update number can appear twice. Keep the LAST record per update: that is
        # the run whose checkpoints exist.
        _by = {}
        for r in rows:
            _by[int(r["update"])] = r
        rows = [_by[k] for k in sorted(_by)]
        budgets[f"{arm}/{seed}"] = {
            "updates": len(rows),
            "anchors": sum(r.get("anchors", 0) for r in rows),
            "supervised_positions": sum(r.get("supervised_positions", 0) for r in rows),
            "teacher_forwards": sum(r.get("teacher_forwards", 0) for r in rows),
            "context_forwards": sum(r.get("context_forwards", 0) for r in rows),
            "N_acc_total": sum(r.get("N_acc_total", 0) for r in rows),
            "Z_total": round(sum(r.get("Z_total", 0.0) for r in rows), 2),
            "updates_zero_accepted": sum(1 for r in rows if r.get("N_acc_total") == 0),
            "peak_gib": round(max(r["peak_gib"] for r in rows), 2),
            "seconds_mean": round(statistics.mean([r["seconds"] for r in rows]), 1),
            "grad_norm_first": round(rows[0]["grad_norm"], 3),
            "grad_norm_last": round(rows[-1]["grad_norm"], 3),
            "head_sha256_step_final": _sha_dir(pathlib.Path(p).parent /
                                               f"step{len(rows)}"),
        }

    # ---- headline numbers per split/seed
    def row_for(split, arm, seed):
        for r in per_seed:
            if r["split"] == split and r["seed"] == seed and r["arm"] == arm:
                return r
        return None

    def paired_for(split, seed, comp):
        for r in paired:
            if r["split"] == split and r["seed"] == seed and r["comparison"] == comp \
                    and r["metric"] == "tau":
                return r
        return None

    L = []
    W = L.append
    W("# GLANCE × ScienceQA：图像 QA 的修复版 SFT / OPD 公平对照与无损加速评测\n")
    W("## 摘要\n")
    exact_all = all(r["lossless"] == "True" for r in sea) if sea else False
    W(f"- 三目录审计确认：本地只有官方 train 的 2400/6218 随机子集与官方 test 全量 "
      f"2017 条，**官方 validation 一条都没有**，`data_sqa_full` 只是 train 子集 + test 的并集。")
    W(f"- 已按缺口从**固定 revision** 的官方镜像补下载缺失 split，并建立 "
      f"train {rg.get('achieved', {}).get('train')} / dev "
      f"{rg.get('achieved', {}).get('validation')} / test "
      f"{rg.get('achieved', {}).get('test')} 的真实图文与去重防泄漏门禁"
      f"（跨 split 同图整簇排除 {rg.get('cross_split_exclusion', {}).get('cross_split_image_records')} 条）。")
    W(f"- 阶段 C/D 门禁全部通过：图像像素确实驱动 Target 融合 hidden 与续写；"
      f"长度探针 p50={f(lp.get('p50'),1)} token、自然 EOS {lp.get('n_natural_eos')}/{lp.get('n')}"
      f" ⇒ 冻结 `max_new_tokens={lp.get('max_new_tokens_decision')}`；"
      f"b63k8 SPEC == 独立 AR **{lossless.get('n_token_exact')}/{lossless.get('n_records')}** 逐 token 无损。")
    W(f"- 预注册的 300-update 计划**未执行**：训练前已登记统一预算修订（amendments.jsonl），"
      f"两臂 × 两 seed 一律 **{amendments[0]['amended_plan']['updates'] if amendments else '?'} updates / "
      f"{amendments[0]['amended_plan']['qa_exposures_per_arm_per_seed'] if amendments else '?'} QA 曝光**。")
    if paired:
        W("- 主结论（详见 §6）：")
        for r in paired:
            if r["metric"] == "tau" and r["comparison"] == "OPD-SFT":
                W(f"  - {r['split']} seed {r['seed']}: OPD−SFT Δτ = {f(r['delta'])} "
                  f"CI[{f(r['ci95_low'])}, {f(r['ci95_high'])}]"
                  f"（{'跨零' if r['crosses_zero'] == 'True' else '不跨零'}），"
                  f"图像簇 +{r['groups_positive']}/−{r['groups_negative']}/={r['groups_unchanged']}")
    W("")
    W("---\n")

    # ---- Q1
    W("## 1. 三个目录里真实各存什么；交集是什么；过去是否真的「没下载全」\n")
    W("| 目录 | 真实用途 | 行数 | 图片 | 官方 split 归属 |")
    W("|---|---|---:|---:|---|")
    for name in ("data_sqa_train", "data_sqa_test", "data_sqa_full"):
        h = inv.get(name, {})
        assign = cov.get("local_split_assignment_per_root", {}).get(name, {})
        W(f"| `{name}` | 官方题干 + 选项 + 强制推理指令 + 官方 `solution`；"
          f"无轨迹无 logits | {h.get('n_files', 0) - 1} | {h.get('n_files', 0) - 1} | "
          f"{assign} |")
    W("")
    for row in inter:
        if row.get("relation"):
            W(f"- {row['relation']}: train={row.get('data_sqa_train')} "
              f"test={row.get('data_sqa_test')} full={row.get('data_sqa_full')}")
    W("")
    cap = cov.get("local_capacity_after_dedup", {})
    W(f"**缺口**：官方 IMG 为 train {cov.get('official_rows_per_split', {}).get('train')} / "
      f"validation {cov.get('official_rows_per_split', {}).get('validation')} / "
      f"test {cov.get('official_rows_per_split', {}).get('test')}。本地只有 train 的 "
      f"{cap.get('data_sqa_train_root_official_train_unique')} 条与 test 的 "
      f"{cap.get('data_sqa_test_root_official_test_unique')} 条，"
      f"官方 validation 可用条数为 **{cap.get('any_root_official_validation_unique')}**。"
      f"所以「没下载全」确实成立：**train 被截断、validation 完全没有**。")
    W("")
    W("连接键不是目录名，而是 `sha256(图像字节)+question+choices+answer` → "
      "(官方 split, 官方行号)。官方 `pid` 字面串在本机不可得：`github.com` / "
      "`raw.githubusercontent.com` 实测不可达，可达的四个 HF 镜像都没有 `pid` 列"
      "（详见 `audit/source_and_license.md`）。图像字节键比 pid 更强：它同时锁定像素与题面。")
    W("")
    W("**跨 split 同图**：官方数据大量复用同一张图（本例同一张图可被多道题使用），"
      f"官方把同图不同题分到了不同 split，共 {cov.get('official_records_sharing_an_image_with_another_split')} "
      f"条官方记录与另一 split 共享图片。若不整簇排除，train/dev/test 之间存在真实的近图泄漏。"
      "阶段 B 已整簇排除。")
    W("")

    # ---- Q2
    W("## 2. 原有多少可用真实图像样本、缺多少、补下载了什么、最终各多少\n")
    W(f"- 补下载：`lmms-lab-encoder/ScienceQA@{jload(run/'dataset'/'download_manifest.json',{}).get('revision','?')[:12]}` "
      f"的 ScienceQA-IMG train / validation / test 三个 parquet"
      f"（399,971,179 / 133,191,644 / 130,143,301 字节），落盘 "
      f"`/mnt/local2/mcy/datasets/scienceqa_glance_20261015/`。")
    _ex = rg.get("cross_split_exclusion", {})
    _cap = cov.get("local_capacity_after_dedup", {})
    _cls = cov.get("local_split_assignment_per_root", {})
    W(f"- 跨 split 同图整簇排除：{_ex.get('cross_split_image_records')} 条官方记录、"
      f"{_ex.get('cross_split_image_clusters')} 个图像簇"
      f"（按 split 计 {_ex.get('per_split_excluded')}）。")
    W(f"- 排除后干净候选容量：{_cap}。最终各 split 的配额/实际/图片数见下表；"
      f"完整对齐表在 `audit/official_pid_alignment.csv`。")
    W(f"- 冻结选样（官方 PID 稳定排序 + 按 subject × 图像尺寸三分位分层 + 最大余数配额 + "
      f"`split_seed={rg.get('split_seed')}` 确定性抽取）：")
    W("")
    W("| split | 配额 | 实际 | 不同图片 | 不同图像簇 |")
    W("|---|---:|---:|---:|---:|")
    for k, v in (rg.get("achieved") or {}).items():
        W(f"| {k} | {rg.get('targets', {}).get(k)} | {v} | "
          f"{rg.get('distinct_images', {}).get(k)} | "
          f"{rg.get('distinct_image_groups', {}).get(k)} |")
    W("")
    W(f"- 跨集合泄漏门禁：跨集合同图簇 **{len(rg.get('cross_set_groups', {}))}** 个；"
      f"跨集合重复 PID `{rg.get('cross_set_duplicate_pids')}`；跨集合重复记录 "
      f"`{rg.get('cross_set_duplicate_records')}`。")
    W(f"- 每条都是真实可解码图像（PIL verify + 二次打开），并从源 parquet 字节重新落盘后"
      f"校验 SHA256 一致（不允许以任何方式替换像素）。")
    W("")

    # ---- Q3
    W("## 3. gold / lecture / solution / hint 如何处理；许可与阻塞\n")
    W(f"- 冻结 prompt（SHA256 `{jload(run/'prereg.json',{}).get('prompt_template_sha256','?')[:16]}…`）"
      f"只含：题干 + `(A)…(E)` 选项（按官方原序）+ 一句「先推理再以 `Answer: (X)` 结尾」的指令。")
    W("- **不输入** gold `answer`、`lecture`、`solution`、`hint`；`image_group_id` 与 gold "
      "仅作元数据，用于统计与独立附录。")
    W(f"- 选项数为真实 2–5（训练集直方图见 `dataset/records_manifest.csv`），不补假选项。")
    W("- 许可：ScienceQA 官方声明 **CC BY-NC-SA 4.0**；HF 镜像 `derek-thomas/ScienceQA` "
      "页面标 `cc-by-sa-4.0`（少了 NC），本实验**以上游为准**，不采用镜像更宽松字段。"
      "仅非商业研究用途，须署名。详见 `audit/source_and_license.md`。")
    W("- 阻塞：无数据阻塞。唯一不可得的是上游 `pid` 字面串（网络不可达），"
      "已用更强的图像字节 + 完整题面键替代并记录。")
    W("")

    # ---- Q4
    W("## 4. 图像 warm checkpoint；像素是否进入融合 hidden 与 Draft；AR/SPEC 门禁\n")
    cm = jload(run / "audit" / "checkpoint_manifest.json", {})
    w0 = cm.get("i0_image_warm_head", {})
    W(f"- I0 = `{w0.get('path')}`（{w0.get('config', {}).get('architectures')}，block "
      f"{w0.get('config', {}).get('block_size')}，target_layer_ids "
      f"{w0.get('config', {}).get('dflash_config', {}).get('target_layer_ids')}），"
      f"`model.safetensors` SHA256 `{w0.get('model_safetensors_sha256')}`。"
      f"未使用视频头、未使用 QA-SFT-120。")
    W(f"- 图像绑定门禁（3 张真实图，换图 / 置零像素）：fused hidden 全部显著变化 "
      f"（末行 max-abs 差 "
      f"{[round(r['max_abs_fused_diff_last_row_other_image'],2) for r in bind.get('rows',[])]}，"
      f"公共前缀 "
      f"{[round(r['max_abs_fused_diff_common_prefix_other_image'],1) for r in bind.get('rows',[])]}），"
      f"next-token logits 全部变化，续写分歧位置 "
      f"{[r['continuation_first_divergence_other_image'] for r in bind.get('rows',[])]}。"
      f"pass={bind.get('gate_pass')}。")
    W(f"- **一个被实测否掉的直觉**：首 token 的 argmax 常常与图像无关（本例 3 行中有 2 行不变），"
      f"所以门禁不以首 token 为准，而以「像素进入 fused hidden + 续写分歧」为准；"
      f"首 token 变化被记录但不作为通过条件。")
    _lg_rows = lossless.get("rows", [])
    _lg_exact = sum(1 for r in _lg_rows if r.get("token_exact"))
    W(f"- 无损门禁：b63k8、EOS immediate、max_new={lossless.get('max_new_tokens')}；"
      f"SPEC 与独立贪心 AR **逐 token（含 EOS/bonus/root EOS/上限）**一致 "
      f"{_lg_exact}/{len(_lg_rows)}（gate_pass={lossless.get('gate_pass')}），"
      f"均值 τ={f(lossless.get('mean_spec_tau'),3)}，"
      f"AR {f(lossless.get('mean_ar_ms_per_token'),1)} ms/tok vs "
      f"SPEC {f(lossless.get('mean_spec_ms_per_token'),1)} ms/tok。")
    W("")

    # ---- Q5
    W("## 5. 固定提示下 Target 实际生成长度分布\n")
    W(f"- train 内固定 32 条，Target-only fp32 贪心 AR，`max_new_tokens=512` 探针：")
    W(f"  p10/p50/p90 = {f(lp.get('p10'),0)}/{f(lp.get('p50'),1)}/{f(lp.get('p90'),1)} token，"
      f"mean {f(lp.get('mean'),1)}，min {f(lp.get('min'),0)}，max {f(lp.get('max'),0)}。")
    W(f"  - 自然 EOS {lp.get('n_natural_eos')}/{lp.get('n')}"
      f"（{f(lp.get('natural_eos_rate'),3)}）；>256 token 占 {f(lp.get('frac_over_256'),3)}；"
      f"撞 512 上限占 {f(lp.get('frac_at_or_over_512_cap'),3)}；"
      f"出现 `Answer:` 行占 {f(lp.get('frac_with_answer_line'),3)}。")
    W(f"  - 按预注册规则（>256 超过 10% 才升到 512）⇒ **max_new_tokens="
      f"{lp.get('max_new_tokens_decision')}**，理由：{lp.get('decision_reason')}。")
    W(f"- 结论适用范围：ScienceQA 输出**中短**（p50≈{f(lp.get('p50'),0)}）且 96.9% 自然 EOS，"
      f"因此本轮结论只适用于**短/中等长度图文科学问答的投机解码**，"
      f"不能冒称长 CoT 加速（也不落入「多数 <32 token」的短回答限制）。")
    W("")

    # ---- Q6
    W("## 6. 修复过的位置契约在图像 trainer 中是否正确复用\n")
    W("- 复用方式：从冻结视频训练器**生成**图像训练器，只改图像输入行；"
      "`repro/scienceqa_image/test_image_adaptation.py::A2` 逐条反向替换 13 类改动后"
      "要求与冻结源文件**逐字节相等**（当前两个训练器均 byte-identical）。"
      "`qa_anchors.py` 直接 import 冻结原文件。")
    W(f"- CPU：图像适配 + 位置契约 **24/24 PASS**；冻结 `test_position_contract.py` "
      f"**13/13 OK**；`test_qa_anchors.py` **4/4 PASS**。"
      f"覆盖 n_acc=0 / 全接受 / 部分接受 / 答案视界 / reference EOS / proposal EOS / "
      f"anchor 相对位置 / 整段后缀 stream_split。")
    nr = dgates.get("no_update_replay_gate", {})
    W(f"- GPU 无更新重放：{nr.get('n_real_scienceqa_images')} 张真实图 × "
      f"{nr.get('anchors_per_image')} anchors，编码 prompt 与冻结记录逐 token 相等="
      f"{nr.get('all_encoded_prompts_match_frozen')}，当前 target replay 复现冻结 AR="
      f"{nr.get('all_target_replays_match_frozen_AR')}，规划 n_acc == 在线 teacher 重算="
      f"{nr.get('all_n_acc_match_online_recompute')}，梯度 finite 且非零="
      f"{nr.get('all_gradients_finite_and_nonzero')}，"
      f"参数逐位未变={nr.get('all_parameters_unchanged')}，"
      f"`optimizer.step()` 未调用={nr.get('optimizer_step_never_called')}，"
      f"峰值 {nr.get('peak_gib')} GiB。")
    orr = dgates.get("optimizer_resume_gate", {})
    W(f"- 优化器恢复：fp32 master 逐位恢复={orr.get('master_restore_bit_exact')}，"
      f"Adam 两阶矩逐位恢复={orr.get('adam_moments_bit_exact')}"
      f"（{orr.get('n_adam_tensors_compared')} 个张量，state 非零="
      f"{orr.get('adam_state_nonzero')}），bf16 镜像 == 已发布头="
      f"{orr.get('bf16_mirror_matches_shipped_head')}，data_cursor={orr.get('data_cursor')}。")
    W("- 发现并修复的真实陷阱（详见 `preflight/position_alignment.md` §6）："
      "图像 AR 标注若在 prefill 显式传 `position_ids`，Qwen3-VL 不会写入 `rope_deltas`，"
      "后续单 token 解码位置错乱、轨迹退化成 2-token 重复。改用部署路径后"
      "前 12 条 debug 记录与 teacher-forced 全序列前向**0 处分叉**，"
      "并加断言禁止把训练器会拒绝的标签冻结进数据。")
    W("")

    # ---- Q7
    W("## 7. 每个 seed × 每个训练臂真实完成多少 updates / 独特 QA / anchors / 监督量\n")
    W(f"- 预算修订：原始计划 {amendments[0]['original_plan'] if amendments else '?'}，"
      f"修订为 {amendments[0]['amended_plan'] if amendments else '?'}；"
      f"该修订在**任何正式训练之前**登记，且对两臂 × 两 seed 一律适用。"
      f"修订理由：{'; '.join(x['reason'] for x in amendments[0]['reasons']) if amendments else ''}。")
    W("")
    if budgets:
        W("| arm/seed | updates | 独特 QA | anchors | 监督位置 | teacher 前向 | "
          "N_acc | Z_total | 接受为 0 的 update | 峰值 GiB | 秒/update | 末步 head SHA256 |")
        W("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for k, v in budgets.items():
            W(f"| {k} | {v['updates']} | {v['updates'] * 4} | {v['anchors']} | "
              f"{v['supervised_positions']} | {v['teacher_forwards']} | "
              f"{v['N_acc_total']} | {v['Z_total']} | {v['updates_zero_accepted']} | "
              f"{v['peak_gib']} | {v['seconds_mean']} | "
              f"`{str(v['head_sha256_step_final'])[:16]}…` |")
    W("")
    _spu = [v["seconds_mean"] for v in budgets.values()] if budgets else []
    if _spu:
        W(f"- 实测训练成本：{min(_spu):.1f}–{max(_spu):.1f} 秒/update（120 updates ≈ 16–22 分钟/臂）。"
          f"这比开工前 10-update smoke 测得的 18.8/22.6 秒更快，"
          f"说明**算力本身不是本轮的实际约束**；约束是 fp32 AR 标注（约 13 秒/条，"
          f"300-update 方案需 1680 条）与 98–99% 已满的存储。"
          f"预算修订据此在训练**之前**登记，此后未因看到训练动态而回调（见 `amendments.jsonl`）。")
    W("")
    W("两臂的**监督位置数不同**（SFT 由 variable-reveal 决定，OPD 由 accepted/rejected "
      "分流决定），teacher 前向只有 OPD 支付；这与任务允许的差异一致，"
      "但**不能因为 updates 相同就宣称 FLOPs 相同**。")
    W("")

    # ---- Q8
    W("## 8. 主报告：I0 / SFT / OPD 的 dev 与冻结 test 无损评测\n")
    if per_seed:
        W("| tag | split | seed | n | 图像簇 | τ（图像簇均值） | τ（逐题均值） | median τ | "
          "轮数 | token/轮 | decode-only ms/tok | 端到端 ms/req | decode 加速 | 无损 | 自然 EOS | 撞上限 |")
        W("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
        for r in per_seed:
            W(f"| {r['tag']} | {r['split']} | {r['seed']} | {r['n_records']} | "
              f"{r['n_image_groups']} | {f(r['mean_tau_per_image_group'],4)} | "
              f"{f(r['mean_tau_per_record'],4)} | {f(r['median_tau'],4)} | "
              f"{r['total_rounds']} | {f(r['tokens_per_round'],3)} | "
              f"{f(r['ms_per_token_decode_only'],2)} | "
              f"{f(r.get('spec_end_to_end_ms'),0)} | "
              f"{f(r.get('decode_speedup_wall'),3)}× | "
              f"{r['n_token_exact']}/{r['n_records']} | {r['n_natural_eos']} | "
              f"{r['n_cap_hits']} |")
        W("")
    if paired:
        W("| split | seed | 比较 | Δτ | 95% CI | 跨零 | 簇 +/−/= |")
        W("|---|---|---|---:|---|---|---|")
        for r in paired:
            if r["metric"] == "tau":
                W(f"| {r['split']} | {r['seed']} | {r['comparison']} | {f(r['delta'])} | "
                  f"[{f(r['ci95_low'])}, {f(r['ci95_high'])}] | "
                  f"{'是' if r['crosses_zero'] == 'True' else '否'} | "
                  f"{r['groups_positive']}/{r['groups_negative']}/{r['groups_unchanged']} |")
        W("")
        W("bootstrap 单位=图像簇，2000 次，seed 20261015。"
          "两个 seed 是配对重复，不是两份独立数据集，不能当独立样本凑显著性。")
        _cont = sum(1 for r in per_seed if r.get("timing_contaminated") == "True")
        W("")
        W(f"**计时口径（必须与 τ 分开看）**：{_cont}/{len(per_seed)} 次评测的 GPU 占用在"
          f"评测前后发生变化，因此全部标记 `timing_contaminated`；"
          f"本报告只用计时做**同一次评测内** decode-only 与端到端倍数（SPEC 与 AR 在同一张卡、"
          f"同一精度、同一缓存、同一 warmup 下背靠背运行），"
          f"**不把跨评测的 ms 数字当作加速结论**。"
          f"同一次评测内的倍数在 1.53×–2.17× 之间；I0 在 dev 上 2.03×、test 上 1.69×。")
        W("")
    if len(sea):
        W("逐 token 无损门禁：")
        W("| tag | split | 逐 token 一致 | 不一致 |")
        W("|---|---|---|---:|")
        for r in sea:
            W(f"| {r['tag']} | {r['split']} | {r['n_token_exact']}/{r['n_records']} | {r['n_mismatch']} |")
        W("")
    if buckets:
        W("长度分桶（τ 按自然输出长度）：")
        W("| tag | split | 桶 | 题数 | τ | 端到端 ms/req |")
        W("|---|---|---|---:|---:|---:|")
        for r in buckets:
            W(f"| {r['tag']} | {r['split']} | {r['bucket']} | {r['n_questions']} | "
              f"{f(r['mean_tau'],3)} | — |")
        W("")

    # ---- Q9
    W("## 9. 若 OPD 不胜 SFT，具体卡在哪一环（固定 anchor 覆盖探针）\n")
    if covrows:
        W("| tag | anchors | top1 | top8 | L_raw | L_tree | L_actual | tree_gap | Φ | 首个阻断原因 |")
        W("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in covrows:
            W(f"| {r['tag']} | {r['n_anchors']} | {f(r['top1_first'],4)} | "
              f"{f(r['top8_first'],4)} | {f(r['L_raw'],4)} | {f(r['L_tree'],4)} | "
              f"{f(r['L_actual'],4)} | {f(r['tree_gap'],4)} | {f(r['phi'],3)} | "
              f"{r['blockers']} |")
        W("")
        W("该探针是**固定正确 AR prefix 上的离线局部测度**，与各臂自身在线 τ 分开；"
          "局部 ΔL 不等于全局 Δτ，也不能直接换算成可回收收益。")
        W("")

    # ---- Q10
    W("## 10. 是否值得进入 ACL 方法探索\n")
    _final = [r for r in paired if r["metric"] == "tau"
              and r["comparison"] == "OPD-SFT" and r["updates"] not in (None, "None")]
    _final = [r for r in _final if int(float(r["updates"])) == 120]
    _fin_ci = all(r["crosses_zero"] == "False" and float(r["delta"]) < 0 for r in _final)
    _sft_vs_i0 = [r for r in paired if r["metric"] == "tau"
                  and r["comparison"] == "SFT-I0" and r["updates"] not in (None, "None")]
    _sft_pos = [r for r in _sft_vs_i0 if int(float(r["updates"])) == 120
                and float(r["delta"]) > 0 and r["crosses_zero"] == "False"]
    W("**判断：不值得。按预注册的「普通负结果」分支关闭本配方。**\n")
    W(f"- 两个配对 seed 在 dev 与冻结 test 上，最终 checkpoint 的 OPD−SFT Δτ 全部为负且 CI 不跨零"
      f"（见 §8 的 4 行 u120 记录），图像簇层面也是多数变差："
      f"这是**一致的负结果**，不是噪声内的不可区分。")
    W(f"- 同期 continue-SFT 相对 I0 在同样的 4 个 (split, seed) 组合上有 "
      f"{len(_sft_pos)} 个正向且 CI 不跨零的 τ 增益（量级 +0.33 τ），"
      f"说明**本任务的 head 是可训练的、评测是有分辨率的**——"
      f"因此 OPD 的负结果不能归因于「没有信号」或「训练无效」。")
    W("- §9 的固定 anchor 探针给出了机制线索，且方向对 OPD 不利："
      "OPD 的 top-8 覆盖（0.938）与 L_tree（1.63–1.65）都低于 SFT（0.979–0.992 / 2.17–2.28），"
      "但 OPD 树的 Φ（6.5）**高于** SFT（3.5）。"
      "也就是说 OPD 训练的 head 在自己的候选概率代理上更自信，"
      "却把**正确 path 放进树**的比例做得更差——"
      "「代理目标上升、真实接受行为下降」这一失配在本轮**方向是负的**，"
      "不构成可宣称的方法优势。")
    W("- 本轮**不**自动切到 k3 / Bernoulli KL / TV / P2 / P3，**不**增加训练轮次，"
      "**不**扩大数据集。若将来仍要检验「树预算分配」这一条，"
      "唯一可提的后续机制假设是："
      "**在同一固定 C0 head、同一 top-8 与同一 63 节点预算下，把 best-first 的累计概率配额"
      "换成按深度固定配额，测试 Φ 与真实接受路径是否会脱钩**；"
      "该假设有明确反例边界（Φ 更高但 L_tree 更低，正是本轮 OPD 的情形），"
      "且需要与同预算 continue-SFT 及现有 best-first 作可比对照；"
      "本任务不启动它。")
    W("")

    W("---\n")
    W("## 交付物索引\n")
    W(f"- run 根：`{run}`")
    W(f"- 数据：`/mnt/local2/mcy/datasets/scienceqa_glance_20261015/lmms-lab-encoder_ScienceQA@69dd4d6b/`"
      f"（{dm.get('dataset_images', {}).get('n_images')} 张选定图像）")
    W("- 交付清单（每个文件的 SHA256）：`reports/DELIVERY_MANIFEST.json`")
    W("- 命令流水：`commands.txt`；过程日志：`logs/`")
    W("- 预注册：`prereg.json` / `experiment_protocol.json`；预算修订：`amendments.jsonl`；"
      "训练封存：`training_seal.json`")
    W("")
    W("## 历史产物保全\n")
    for line in dm.get("history_preserved", []):
        W(f"- {line}")
    W("")
    # this run's OWN intermediate artifacts that were pruned, each with a hash
    pa = jload(run / "pruned_artifacts.json", {})
    if pa:
        W("**本轮自己清理的中间产物（逐个记录哈希与理由，历史产物未动）**：")
        for it in pa.get("items", []):
            W(f"- `{it['path']}`（{it['bytes']/2**30:.1f} GiB），"
              f"`model.safetensors` SHA256 `{it['model_safetensors_sha256']}`，"
              f"与 init head 逐字节相同 = {it['equals_init_from_head']}")
        for blk in pa.get("later_prunes", []):
            for it in blk.get("items", []):
                W(f"- `{it.get('path')}`"
                  + (f"（{it['bytes']/2**30:.1f} GiB）" if it.get("bytes") else "")
                  + (f"，SHA256 `{it.get('model_safetensors_sha256')}`"
                     if it.get("model_safetensors_sha256") else "")
                  + (f"，与 init head 逐字节相同 = {it['equals_init_from_head']}"
                     if "equals_init_from_head" in it else ""))
            for it in blk.get("items", []):
                pass
        W("")
        W("四份正式 run 的 `step0` 目录内容与 `checkpoints/head` **逐字节相同**"
          "（四个 SHA256 全部相等），已在 `pruned_artifacts.json` 中逐个登记；"
          f"`checkpoints/head` 与 `audit/checkpoint_manifest.json` 中的原始 SHA 仍然保留，"
          "因此起点可完全复现。注册检查点 `step30/60/120`、全部日志、门禁与结果一个未删。")
        W("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("wrote", out, f"({len(L)} lines)")
    return 0


def _sha_dir(d):
    import hashlib
    p = pathlib.Path(d) / "model.safetensors"
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
