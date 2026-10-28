# 实验报告索引

报告统一采用 `YYYY-MM-DD-主题.md`，按日期从早到晚排列；同一天按主题排序。日期沿用原报告日期，不使用文件修改时间。复现历史按正文标注的截至日期归档。

当前项目状态仍以 [AGENTS.md](../../AGENTS.md) 为准。

| 日期 | 报告 |
|---|---|
| 2026-09-25 | [GLANCE 复现：TextVQA+COCO 训练 → ChartQA 接受率与加速比](2026-09-25-reproduction-history.md) |
| 2026-09-28 | [EAGLE-3 对比：阶段 0 交接（2026-09-28）](2026-09-28-eagle3-stage0.md) |
| 2026-10-05 | [视觉适配诊断：当前 MLP 投影在 VLM 投机解码中的作用](2026-10-05-视觉适配诊断.md)（产物盘软链接） |
| 2026-10-08 | [Video OPD：online freshness × verifier topology 诊断](2026-10-08-video-opd-2x2.md) |
| 2026-10-09 | [B1 公式核实记录（§10.3）](2026-10-09-b1-formula.md) |
| 2026-10-09 | [B1 trainer 真实验证、teacher 精度判定与 Phase C seed-0 配置冻结](2026-10-09-b1-precision-and-phasec-config.md) |
| 2026-10-09 | [阶段 A 执行报告：clean SFT plateau 曲线](2026-10-09-phase-a.md) |
| 2026-10-09 | [阶段 A4 执行报告：aligned SFT 重新定峰](2026-10-09-phase-a4.md) |
| 2026-10-10 | [Phase C 报告补正 + fp32/bf16 混合精度门禁 + continue-SFT 小步校准](2026-10-10-phase-c-corrections.md) |
| 2026-10-10 | [Phase C seed-0 三臂报告：aligned 起点下 paper-faithful OPD 未改善投机解码效率](2026-10-10-phase-c-seed0.md) |
| 2026-10-10 | [Step 3/4 预注册：continue-SFT 小步校准的候选选择规则](2026-10-10-phase-c2-prereg.md) |
| 2026-10-10 | [参数更新精度诊断：低学习率下更新被 bf16 舍掉了](2026-10-10-update-precision.md) |
| 2026-10-12 | [当前 AR 固定 anchor 诊断闭环（2026-10-12）](2026-10-12-current-ar-fixed-anchor.md) |
| 2026-10-12 | [收益瓶颈诊断：2026-10-12 ACL 审计与 CPU 补证](2026-10-12-gain-stuck.md) |
| 2026-10-12 | [任务 A：可部署树选择与接受路径失配（2026-10-12 并行实验）](2026-10-12-parallel-A-tree.md) |
| 2026-10-12 | [并行任务 B：GLANCE 视频时序条件响应与候选覆盖诊断（2026-10-12）](2026-10-12-parallel-B-temporal.md) |
| 2026-10-12 | [phaseC4：C2 主权重接入 + 组合门禁 + 冻结 10-update 正式对照](2026-10-12-phase-c4-frozen-10update.md) |
| 2026-10-13 | [当前 target 在线 OPD 闭环修复（2026-10-13）](2026-10-13-opd-online-fix.md) |
| 2026-10-13 | [当前 OPD 实现审计：proposal 在线，但验证闭环未达到严格 Draft-OPD](2026-10-13-opd-online-implementation-audit.md) |
| 2026-10-15 | [GLANCE × ScienceQA：修复版图像 SFT / OPD 公平对照（普通负结果）](2026-10-15-scienceqa-image-sft-vs-opd.md) |
| 2026-10-15 | [QA位置契约修复与10-update公平验证](2026-10-15-qa-contract-fix-10update.md) |
| 2026-10-15 | [QA 单路径接受、候选树与梯度方向诊断](2026-10-15-qa-tree-semantics-diagnosis.md) |
| 2026-10-15 | [任务 A（P1）：被拒段 KL 换成官方最新 KL 版 k3 单样本估计量](2026-10-15-taskA-kl-official-p1.md) |
| 2026-10-15 | [任务 B：公平重做视频 + 图像 QA 两组对照](2026-10-15-taskB-fair-qa-redo.md) |
| 2026-10-16 | [ScienceQA 实现与统计纠错](2026-10-16-scienceqa-implementation-and-metrics-errata.md) |
| 2026-10-18 | [OPD 效率优化：无更新门禁与实现记录](2026-10-18-opd-efficiency-gates.md) |
| 2026-10-18 | [OPD backward 非确定性、独立门禁与 isolated update](2026-10-18-opd-backward-noise.md) |
| 2026-10-18 | [OPD 四卡rollout与rank0 canonical训练](2026-10-18-opd-rollout-only-parallel.md) | 并行rollout唯一候选；24次有界隔离更新、确定性门禁、连续/恢复与3×3计时，状态见报告 |
| 2026-10-18 | [OPD 确定性分布式累积协议实测](2026-10-18-opd-exact-distributed-protocol.md) |

- 2026-10-16：[官方 KL 独立验收与 ScienceQA 训练放行审查](/mnt/local3/mcy/GLANCE/results/official_kl_final_acceptance_20261016_165617/reports/FINAL_ACCEPTANCE_REVIEW.md)（L2_cpu；HOLD，未训练）。

- 2026-10-16：[真实 GPU 官方 KL 无更新门禁审阅](/mnt/local3/mcy/GLANCE/results/official_kl_gpu_watch_20261016/review/GPU_GATE_REVIEW.md)（L2_gpu；建议下一轮授权小规模正确性探针，未训练）。

- 2026-10-16：[官方 KL ScienceQA 夜间训练容量审计与收口](/mnt/local3/mcy/GLANCE/results/official_kl_scienceqa_overnight_20261016_222121/reports/FINAL_OVERNIGHT_SCIENCEQA.md)（BLOCKED，固定档位均超9小时；0 updates）。

- 2026-10-16：[官方KL块验证加速独立验收：HOLD / rejected proposal分叉](/mnt/local3/mcy/GLANCE/results/official_kl_block_verify_20261016_223740/FINAL_BLOCK_VERIFICATION_OPTIMIZATION.md)，24项CPU通过，真实GPU硬门禁失败，未进入测速/训练。

- 2026-10-16：[块验证精确分叉定位与提交前缀重放补丁：HOLD](/mnt/local3/mcy/GLANCE/results/official_kl_divergence_20261016_225216/FINAL_DIVERGENCE_REVIEW.md)，原proposal分叉已修复，真实Head梯度阈值仍未全过，原Head删除后核查36份候选未找到同SHA副本。

- 2026-10-17：[ScienceQA 新图像 Warm 重建与官方 KL 成对训练最终报告](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/reports/FINAL_SCRATCH_WARM_AND_OFFICIAL_KL.md)：Warm 2000 updates，SFT/OPD 各 120 updates，M2 同一 dev8 逐 token 无损；[19 项独立验收](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/audit/INDEPENDENT_FINAL_ACCEPTANCE.json) PASS。结果仅为小型开发集描述，非独立泛化结论。

- 2026-10-18：ScienceQA 新 validation 200 图像确认集：[最终报告](/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/reports/FINAL_CONFIRM200.md)、[预冻结名单及指标](/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/freeze/cohort_frozen.json)、[独立复核](/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/audit/INDEPENDENT_RESULT_RECHECK.json)。三臂各 200/200 逐 token 无损；OPD−SFT 逐图像有效 token/轮 +0.0749，95% 配对 CI [+0.0444,+0.1063]。原交付清单仅 `runner.log` SHA 过早记录，见[纠正清单](/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/reports/DELIVERY_MANIFEST_CORRECTED.json)。

- 2026-10-19：[四卡 rollout-only 的 opt-in 生产接入与小门禁](2026-10-19-opd-production-optin.md)（9次预算已执行；实际结论见报告，默认serial未切换）。

- 2026-10-19：[pure_block_native 显式实验入口](2026-10-19-pure-block-native-entry.md)（34项CPU、四卡零更新成功；120步已获授权并启动，实时状态见报告与账本；部署无损要求不变）。

- 2026-10-19：[ScienceQA默认pure block四卡与一次smoke](2026-10-19-scienceqa-pure-default-smoke.md)（GO_SMOKE；仅证明能跑，不证明稳定/等价/恢复）。

- 2026-10-19：[ScienceQA 全量配对训练脚本交付（CPU准备，待用户启动）](2026-10-19-scienceqa-full-epoch-launcher.md)。

- 2026-10-19：[新 Warm-2000 训练曲线检查](2026-10-19-warm-training-curve-audit.md)（历史日志CPU绘图；后期loss缓降，未证明开发集已饱和）。

- 2026-10-19：[训练阶段严格重放断言清理](2026-10-19-training-assertion-cleanup.md)（SFT Target重放与OPD proposal重放不再要求argmax相等；未重启训练）。

- 2026-10-19：[ScienceQA 全量训练重新启动](2026-10-19-scienceqa-full-epoch-restart.md)（用户明确授权；复用标签，从Warm重开，旧101次消耗保留；无持续监控）。

- 2026-10-19：[ScienceQA 共享GPU续训](2026-10-19-scienceqa-shared-gpu.md)（显存足够即可共享；从两臂step400续训）。
| 2026-10-20 | [ScienceQA 最终 SFT / OPD 实测推理速度](2026-10-20-scienceqa-inference-speed.md) |

- 2026-10-20：[ScienceQA 三卡续训两个 epoch](2026-10-20-scienceqa-three-card-continuation.md)（从两臂 step1182 完整状态续训；每臂新增2364 updates；独立三卡/四slot入口，固定两轮）。

- 2026-10-26/27：[视频长生成 OPD：3 epoch 小幅提升、4 epoch 回落、SFT 崩溃](2026-10-26-video-caption-opd.md)（用户授权夜间测试；VideoDetailCaption 439 train，fresh30 n=30，平均生成 270.9 token；OPD 3e τ +0.0575，SFT 崩溃到 1.92）。
