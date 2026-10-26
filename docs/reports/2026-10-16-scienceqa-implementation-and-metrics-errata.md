# ScienceQA 实现与统计纠错 2026 年 10 月 16 日

本次只做代码审计和既存记录离线复算，没有训练、没有重新访问冻结 test 模型评测。旧报告和原始评测保留原件。

## 仍成立的结果

2026-10-15 ScienceQA 运行实际为 fixed-AR-anchor CE + full-reverse-KL。相对同数据、同起点继续 SFT 的负差异成立；“完整官方 online KL 已复现”的命名撤回。Task A 仅改 rejected k3，未完成 accepted Bernoulli 或官方池化。

原始 τ 是每轮 accepted+1（终止裁剪前），下表另用真实输出 tokens/rounds，按图像簇配对 bootstrap 2000 次、seed=20261015：

| split | seed | OPD−SFT | 95% CI |
|---|---|---:|---|
| dev | 20261013 | -0.401522 | [-0.443004, -0.361779] |
| dev | 20261014 | -0.392335 | [-0.432881, -0.352133] |
| test | 20261013 | -0.362774 | [-0.398183, -0.325154] |
| test | 20261014 | -0.375168 | [-0.416132, -0.333089] |

## 撤回的机制判断与离线修正

首阻断原实现先扫描候选缺失再扫描树，掩盖更早树遗漏。真实反例 L_raw=9/L_tree=1，应在 depth=2 判树遗漏，原表却记录 depth=10 候选缺失。

L_actual 原来未裁剪终止后的树路径；现在从已存的 L_correct_in_actual_path 恢复、排除 root/bonus 并尊重有效 horizon。全部 1200 行满足 0≤L_actual≤L_tree≤L_raw≤horizon。

| head | 树遗漏 | 候选缺失 | horizon 结束 | 修正 L_actual |
|---|---:|---:|---:|---:|
| i0 | 1 | 121 | 118 | 0.750000 |
| opds20261013u120 | 40 | 83 | 117 | 1.654167 |
| opds20261014u120 | 43 | 80 | 117 | 1.633333 |
| sfts20261013u120 | 69 | 52 | 119 | 2.275000 |
| sfts20261014u120 | 51 | 70 | 119 | 2.170833 |

“树遗漏只有 0–2 次，因此不是树预算瓶颈”撤回。不能由此反向断言树选择已被证明是退化主因。五个 head 各有 8 条原 L_actual 超出 horizon=1。每个 head 的 240 anchors 均为 120 题×首尾两位置，120 行 horizon=1、120 行 horizon=15；正常策略访问位置分布 unavailable，不能生成全在线机制结论。

## 准确率与训练审计

- dev parsed-only=186/203=91.6256%，all-questions=186/240=77.5000%。
- test parsed-only=183/198=92.4242%，all-questions=183/240=76.2500%。
- 无损使三臂输出相同；未解析题不能从全题准确率分母中悄悄删除。
- 两 seed 同一数据日程，OPD 固定 anchors/greedy/eval，无显式不同数据顺序；不能作为独立数据抽样的复现。
- 原恢复只载入 head/master/Adam，不调用 RNG restore，step 来自 CLI；“完整端到端 bitwise resume”旧表述撤回。新 SFT helper 的 CPU 4 vs 2+2 已通过，不倒推历史运行已通过。
- SFT seed13 日志含重复 update1，121 行不等于执行 121 个有效顺序更新。历史训练预算复算须按更新身份去重，不能直接加总日志。
- 新基础脚本审计发现 k3/k3 默认与交接中 Bernoulli/k3 不符；详见唯一契约。此前基准描述也须按有效覆盖链理解。

## 证据与复算

- [新统计与来源 SHA](/mnt/local3/mcy/GLANCE/results/official_kl_cleanup_20261016_131301/audit/corrected_scienceqa/corrected_summary.json)
- [官方来源 manifest](/mnt/local3/mcy/GLANCE/results/official_kl_cleanup_20261016_131301/audit/official_source_manifest.json)
- [当前唯一契约](../methods/draft_opd_official_kl_contract.md)
- [原 ScienceQA 报告，保持原件](2026-10-15-scienceqa-image-sft-vs-opd.md)
- 离线复算器：`repro/scienceqa_image/correct_historical_metrics.py`；不写回旧 run。

## 历史 manifest 另行核验

本轮只读散列268个外部现存产物。旧交付manifest有240项匹配、4项不符、24项原核验路径无对应值（含迁移到其他挂载点的文件，另按相对路径核验）。4项为 package_delivery_final.log、pruned_artifacts.json、DELIVERY_MANIFEST.json 自身、FINAL_SCIENCEQA_SFT_VS_OPD.md。保留旧记录和当前SHA，不回填旧manifest；不把已有差异隐去。原FINAL报告当前SHA：`e3bfa537ac6355aeeb46ea7f7bcb0a9fa5d5041034b7e4004c556d684110a851`。23份旧仓库报告与本轮改动前快照逐SHA相同。新代码不会倒推历史训练身份。

用户随后批准基础脚本的显式方向覆盖；代码等级为L2 CPU-only，GPU门禁未跑，不升级为正式训练验证。
