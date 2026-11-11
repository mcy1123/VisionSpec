# 实验报告索引

本仓库报告统一命名 `YYYY-MM-DD-主题.md`，按报告日期（不是文件修改时间）排列。项目当前状态以 [AGENTS.md](../../AGENTS.md) 为准；方法说明见 [methods/](../methods/)；组会材料见 [talks/](../talks/)。

结论栏只是导航用的摘要，细节和反例以报告正文为准。负结果不删不改。

## 本仓库报告

| 日期 | 报告 | 结论 / 状态 |
|---|---|---|
| 2026-09-25 | [GLANCE 复现：TextVQA+COCO 训练 → ChartQA 接受率与加速比](2026-09-25-reproduction-history.md) | 原 repro README；含已被后续审计替代的历史表述 |
| 2026-09-28 | [EAGLE-3 对比：阶段 0 交接](2026-09-28-eagle3-stage0.md) | M-RoPE 静默失效等实现结论的来源 |
| 2026-10-05 | [视觉适配诊断：MLP 投影在 VLM 投机解码中的作用](2026-10-05-视觉适配诊断.md) | 软链到产物盘，依赖本机挂载 |
| 2026-10-08 | [Video OPD：online freshness × verifier topology](2026-10-08-video-opd-2x2.md) | 四臂诊断 |
| 2026-10-09 | [B1 公式核实记录（§10.3）](2026-10-09-b1-formula.md) | 公式与官方实现出处 |
| 2026-10-09 | [B1 trainer 验证、teacher 精度与 Phase C seed-0 配置冻结](2026-10-09-b1-precision-and-phasec-config.md) | 配置冻结依据 |
| 2026-10-09 | [阶段 A：clean SFT plateau 曲线](2026-10-09-phase-a.md) | legacy SFT |
| 2026-10-09 | [阶段 A4：aligned SFT 重新定峰](2026-10-09-phase-a4.md) | |
| 2026-10-10 | [Phase C 补正 + fp32/bf16 混合精度门禁](2026-10-10-phase-c-corrections.md) | 精度门禁 |
| 2026-10-10 | [Phase C seed-0 三臂：paper-faithful OPD 未改善](2026-10-10-phase-c-seed0.md) | 负结果 |
| 2026-10-10 | [Step 3/4 预注册：continue-SFT 小步校准](2026-10-10-phase-c2-prereg.md) | 预注册规则 |
| 2026-10-10 | [参数更新精度：低学习率下更新被 bf16 舍掉](2026-10-10-update-precision.md) | 精度诊断 |
| 2026-10-12 | [当前 AR 固定 anchor 诊断闭环](2026-10-12-current-ar-fixed-anchor.md) | |
| 2026-10-12 | [收益瓶颈诊断：ACL 审计与 CPU 补证](2026-10-12-gain-stuck.md) | 长报告 |
| 2026-10-12 | [任务 A：可部署树选择与接受路径失配](2026-10-12-parallel-A-tree.md) | 并行实验 |
| 2026-10-12 | [任务 B：视频时序条件响应与候选覆盖](2026-10-12-parallel-B-temporal.md) | 并行实验 |
| 2026-10-12 | [phaseC4：C2 主权重接入 + 冻结 10-update 对照](2026-10-12-phase-c4-frozen-10update.md) | |
| 2026-10-13 | [当前 target 在线 OPD 闭环修复](2026-10-13-opd-online-fix.md) | |
| 2026-10-13 | [OPD 实现审计：验证闭环未达严格 Draft-OPD](2026-10-13-opd-online-implementation-audit.md) | 审计发现 |
| 2026-10-15 | [修复版图像 SFT / OPD 公平对照](2026-10-15-scienceqa-image-sft-vs-opd.md) | 普通负结果 |
| 2026-10-15 | [QA 位置契约修复与 10-update 公平验证](2026-10-15-qa-contract-fix-10update.md) | |
| 2026-10-15 | [QA 单路径接受、候选树与梯度方向诊断](2026-10-15-qa-tree-semantics-diagnosis.md) | |
| 2026-10-15 | [任务 A（P1）：被拒段换成官方 k3 单样本估计量](2026-10-15-taskA-kl-official-p1.md) | |
| 2026-10-15 | [任务 B：公平重做视频 + 图像 QA 对照](2026-10-15-taskB-fair-qa-redo.md) | |
| 2026-10-16 | [ScienceQA 实现与统计纠错](2026-10-16-scienceqa-implementation-and-metrics-errata.md) | 负结果保留，错误机制解释撤回 |
| 2026-10-18 | [OPD 效率优化：无更新门禁与实现记录](2026-10-18-opd-efficiency-gates.md) | |
| 2026-10-18 | [OPD backward 非确定性、独立门禁与 isolated update](2026-10-18-opd-backward-noise.md) | |
| 2026-10-18 | [OPD 四卡 rollout 与 rank0 canonical 训练](2026-10-18-opd-rollout-only-parallel.md) | 旧 serial 方法；2.7647× 不作 native 验收 |
| 2026-10-18 | [OPD 确定性分布式累积协议实测](2026-10-18-opd-exact-distributed-protocol.md) | |
| 2026-10-19 | [四卡 rollout-only 的 opt-in 生产接入](2026-10-19-opd-production-optin.md) | 9 次预算；默认 serial 未切换 |
| 2026-10-19 | [pure_block_native 显式实验入口](2026-10-19-pure-block-native-entry.md) | 34 项 CPU、四卡零更新；120 步实验 |
| 2026-10-19 | [ScienceQA 默认 pure block 四卡与一次 smoke](2026-10-19-scienceqa-pure-default-smoke.md) | GO_SMOKE：只证明能跑 |
| 2026-10-19 | [ScienceQA 全量配对训练脚本交付（CPU 准备）](2026-10-19-scienceqa-full-epoch-launcher.md) | 交付记录 |
| 2026-10-19 | [新 Warm-2000 训练曲线检查](2026-10-19-warm-training-curve-audit.md) | 未证明开发集饱和 |
| 2026-10-19 | [训练阶段严格重放断言清理](2026-10-19-training-assertion-cleanup.md) | 未重启训练 |
| 2026-10-19 | [ScienceQA 全量训练重新启动](2026-10-19-scienceqa-full-epoch-restart.md) | 复用标签，从 Warm 重开 |
| 2026-10-19 | [ScienceQA 共享 GPU 续训](2026-10-19-scienceqa-shared-gpu.md) | 从两臂 step400 续训 |
| 2026-10-20 | [ScienceQA 最终 SFT / OPD 实测推理速度](2026-10-20-scienceqa-inference-speed.md) | 空卡测速协议 |
| 2026-10-20 | [ScienceQA 三卡续训两个 epoch（最终报告）](2026-10-20-scienceqa-three-card-continuation.md) | 两臂到 step3546；结果与协议正文 |
| 2026-10-26 | [视频长生成 OPD：3 epoch 小幅提升、4 epoch 回落、SFT 崩溃](2026-10-26-video-caption-opd.md) | n=30，单 seed，不外推 |
| 2026-11-02 | [图像 OPD 视觉依赖诊断与六种视觉侧干预](2026-11-02-visual-opd.md) | 六种干预均无统计支持的 high-g 增益，判阴性 |
| 2026-11-04 | [视频 OPD 视觉依赖复现：blank 增益更大、视觉依赖下降](2026-11-04-video-visual-replication.md) | visual reliance 0.652→0.502，DiD −0.1504 |
| 2026-11-04 | [投机解码 baseline 在同一测试集上的复现](2026-11-04-speculative-baselines.md) | 5 条 baseline；CLI 见 `repro/baselines/` |
| 2026-11-05 | [多模态特征/OPD 训练优化夜跑](2026-11-05-multimodal-architecture.md) | 架构与 cross-modal 阴性，无 visual-specific 优化 |
| 2026-11-09 | [视频 target 输入 5–10k 视觉 token](2026-11-09-video-target-9k-tokens.md) | 9k fp32 3 卡 30/30 无损；τ **3.4098** vs 1k 3.7841；token 数本身不是收益 |
| 2026-11-12 | [真实视频长生成 OPD 夜间执行](2026-11-12-real-video-opd-overnight.md) | Video-MME 80/40；gate 4/4；λ0 35/λ10 21步；dev20 128f 6.578→7.070→7.171（λ10−λ0 +0.101），64f 8.341→8.970（128-trained head） |
| 2026-11-11 | **[视频高视觉 token 多模态完整整理](2026-11-11-consolidated-video-token-multimodal.md)** | 汇总 image E2E120 正结果、1k/3k/9k32f2x、128f1x 两 seed、TV20、token 减半、layer gate/xattn/zero-visual 与存储清理 |
| 2026-11-11 | [真实视频长生成：诊断修复与跨服务器执行方案](2026-11-11-real-video-opd-execution-plan.md) | CPU 修复与回归；真实帧/时间、归一化、视觉掩码、xattn 缓存；含零更新及截止时间驱动的夜间 prompt，支持更新边界保存停止，联合选择器训练尚待实现 |
| 2026-11-11 | [多模态融合与视觉 token 选择](2026-11-11-multimodal-fusion-token-selection.md) | 128f1x 9k **3.8796**；32/64f1x 3.75/3.68；9k head 减半 3.70；layer gate/xattn 中性；zero-visual 掉 0.248（blank 1.04）；**TV20 3.9046（+0.025，CI 刚跨零）** |
| 2026-11-10 | [视频 5–10k 视觉 token 的多模态特性](2026-11-10-high-token-video-multimodal.md) | 128f1x 9k **3.8796**（blank 2.8428，reliance 1.04）；32f2x 9k 仅 3.533，上采样稀释 |
| 2026-11-06 | [长训练 / 多数据 / 高视觉 token：E2E-TV 120 vs base120](2026-11-06-long-horizon-token-scaling.md) | image E2E-TV high-g **+0.0768 [+0.0102,+0.1463]**、top +0.0849；video 3456 token FOVEA 无增益 |
| 2026-11-05/06 | [多模态架构适配探索：raw visual / FOVEA / FiLM / Perceiver / video](2026-11-05-multimodal-architecture-exploration.md) | 架构变体均阴性/中性；image E2E-TV 仍是唯一 visual-specific 正结果 |
| 2026-11-05 | [SpecOPD E2E-TV 在图像 OPD 上的首次实验](2026-11-05-specopd-e2e-tv.md) | 三 seed 合并 overall +0.0301 [+0.0087,+0.0515]，高 g +0.0465 [+0.0136,+0.0793]，首个 visual-specific 正结果 |

## 结果盘报告（链接到 `/mnt/local3`，不在本仓库）

| 日期 | 报告 | 结论 / 状态 |
|---|---|---|
| 2026-10-16 | [官方 KL 独立验收与 ScienceQA 放行审查](/mnt/local3/mcy/GLANCE/results/official_kl_final_acceptance_20261016_165617/reports/FINAL_ACCEPTANCE_REVIEW.md) | L2_cpu；HOLD，未训练 |
| 2026-10-16 | [真实 GPU 官方 KL 无更新门禁审阅](/mnt/local3/mcy/GLANCE/results/official_kl_gpu_watch_20261016/review/GPU_GATE_REVIEW.md) | L2_gpu；建议小规模探针，未训练 |
| 2026-10-16 | [官方 KL 夜间训练容量审计与收口](/mnt/local3/mcy/GLANCE/results/official_kl_scienceqa_overnight_20261016_222121/reports/FINAL_OVERNIGHT_SCIENCEQA.md) | BLOCKED；0 updates |
| 2026-10-16 | [块验证加速独立验收](/mnt/local3/mcy/GLANCE/results/official_kl_block_verify_20261016_223740/FINAL_BLOCK_VERIFICATION_OPTIMIZATION.md) | HOLD；24 项 CPU 通过，GPU 门禁失败 |
| 2026-10-16 | [块验证精确分叉定位与提交前缀重放补丁](/mnt/local3/mcy/GLANCE/results/official_kl_divergence_20261016_225216/FINAL_DIVERGENCE_REVIEW.md) | HOLD；原 Head 未找回同 SHA 副本 |
| 2026-10-17 | [新图像 Warm 重建与官方 KL 成对训练最终报告](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/reports/FINAL_SCRATCH_WARM_AND_OFFICIAL_KL.md) | Warm 2000 + 各 120 updates；小型开发集描述 |
| 2026-10-18 | [ScienceQA 新 validation 200 确认集](/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/reports/FINAL_CONFIRM200.md) | 三臂 200/200 无损；OPD−SFT +0.0749 [+0.0444,+0.1063] |
