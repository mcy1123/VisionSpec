# 对 GLANCE 当前实验最有用的借鉴

依据：2026-10-09 的 [AGENTS.md](../../../AGENTS.md)；aligned step1000 mean τ=3.7377，旧视频 OPD 四臂均未胜过旧起点；B0.5/B1 完整验证仍待完成。下文是研究建议，不是已跑出的实验，也不替代 `docs/plans/video-opd.md` 的预注册配置。

## 先读这几条，再决定加什么实验

**首要工作仍是把当前忠实版基线跑清楚。** 最值得补充的视角不是“更多 OPD 总会变好”，而是把状态分布、监督哪些位置、损失目标、验证策略及实际耗时拆开。新论文给出了互相竞争的方案：Draft-OPD 原文学习整段 rejected 后缀；Spec-AUF 只保留到首次失败位置的 CE；VAT 改变拒绝位置附近的权重；EAL/WTV直接面向接受长度。这些是不同的实验假说，不能拼成一套再把变化归因于 OPD。[Draft-OPD](https://arxiv.org/abs/2605.29343)、[Spec-AUF](https://arxiv.org/abs/2607.01893)、[VAT](https://arxiv.org/abs/2608.30135)、[EAL/WTV](https://arxiv.org/abs/2609.24150)

| 阅读顺序 | 论文/实现 | 为什么适合现在 | 建议行动 |
|---|---|---|---|
| 1 | [Draft-OPD + 官方 TV 扩展](notes/2605.29343.md) | 与当前 B1 直接对应；原论文和最新代码目标有分叉 | 固定版本，分别记录 paper-full-KL 与官方工程目标 |
| 2 | [VAT](notes/2608.30135.md)、[Spec-AUF](notes/2607.01893.md) | 解释“平均 loss 降，但 τ 不升”的位置监督问题 | 先看首拒分布及逐 offset 梯度；后续独立消融监督范围 |
| 3 | [EAL/WTV](notes/2609.24150.md) | 对 greedy 与 sampling 用不同的接受目标 | 先评估 surrogate 与本地树 τ 是否相关，再投入训练 |
| 4 | [ALR-IRA](notes/2609.38795.md) | 标签/anchor 有效监督不足可能与优化器问题混淆 | 核查视频标签与部署轨迹、有效位置覆盖 |
| 5 | [AdaFlash](notes/2607.19223.md) | OPD diffusion drafter 与动态长度，架构接近 | 把损失和长度调度拆成两个变量 |
| 6 | [Sparrow](notes/2602.15318.md)、[HiViS](notes/2509.23928.md) | 已融合 hidden、长视觉上下文、训练推理偏差 | 检查特征随视觉/步数变化，再考虑新结构 |
| 7 | [MMSpec](notes/2603.14989.md) | 多模态评测与视觉依赖诊断 | 不只看总 τ，按视频/视觉内容/长度分析失败 |
| 8 | [SharpDraft](notes/2610.05106.md)、[AdaSpark](notes/2610.05774.md) | 10月新作，可降低长上下文或验证成本 | 独立冻结head做推理侧对照，不能同时改OPD |
| 9 | [D-Loop](notes/2610.06011.md) | 新的块内自条件化路线，对本地Markov负结果有参考 | 先查重复token与首拒关系；训练与额外前向成本都计入 |
| 10 | [TIGER](notes/2607.11131.md) | 多模态 accepted-prefix reward 的直接相关探索 | 放到有可靠full-KL基线后的方法分支 |

## 必须先处理的两个版本问题

### GLANCE 已有 v3，本地论文是 v1

本地 `glance.txt` 首页写 `2609.00355v1`；线上 v3 更新于 2026-10-06，摘要最高加速已变为 3.05×。这只说明参考版本发生变化，不能据此把本地 3.36× decode-only 结果重写，也不能将新版生产引擎指标与本地脚本直接相减。建议先制作 v1/v3 配置与实现差异表，确认新增内容是否影响复现结论。[GLANCE 版本历史](https://arxiv.org/abs/2609.00355)

### Draft-OPD 原论文 full-KL 与官方 TV 分支分开

官方 README 的 2026 年 8 月更新宣布了 E2E TV loss 与 first-reject-only 扩展；原论文 v3 与本项目 B1 所选择的两流 KL 不是同一目标。该分支快照已记录为 `e81a8bc488ef762f178f8708ffbc49bd92c3a93d`。工程更新值得优先读，但作者“更强”的声明尚不能替代同配置本地比较。[官方仓库](https://github.com/Simplified-Reasoning/Draft-OPD)、[固定提交](https://github.com/Simplified-Reasoning/Draft-OPD/commit/e81a8bc488ef762f178f8708ffbc49bd92c3a93d)

尤其不能从“first-reject-only 后来有效”推断本项目旧 top-k forward-KL 配方其实正确：两者还可能同时不同于 teacher scoring、TV 目标、状态分布和归一化。最便宜的下一步是代码审计与固定轨迹 loss/梯度比较，而不是直接重训。

## 按成本安排的候选实验

以下成本为工程判断，不是论文测量；只有当前 B0.5/B1/Phase C 属于已有计划，新增实验须另写配置与对照。

### E0：完成现有门禁，并增加不改变目标的诊断

保留 fp32 teacher、两流定义、整 update 归一及原三臂预算。可在冻结训练子集上记录：accepted/rejected 数量、首拒 offset、每 offset loss、两流梯度范数、梯度夹角、clipping 比例、teacher/student 熵与概率重叠、draft/teacher/backward 分项耗时。抽样诊断也有计算成本，须单独记时。

这些观测分别检验数值误差、信号失衡、监督错位和资源瓶颈，不改变优化目标。一般 OPD 研究提示 rollout policy、KL 方向和步长有交互，并不支持“on-policy天然更优”；这一观点用于提出竞争解释，不能直接当作本地失败原因。[Distillation Dynamics](https://arxiv.org/abs/2609.35259)、[Unmasking OPD](https://arxiv.org/abs/2605.10889)、[Rethinking OPD](https://arxiv.org/abs/2604.13016)

### E1：拒绝后缀是否在消耗无效梯度？低至中成本

先看当前 C2 在已固定轨迹上的“逐位置 KL 改善—首拒概率变化”，观察远端后缀改善是否带来更长连续接受。再选一个后续监督支持集实验：忠实 full-KL 与官方 TV 扩展对照，或 continue-SFT 与 Spec-AUF 对照。两种对照不能混为“只改一个mask”，因为基础目标不同。

Spec-AUF 的边界是**包含首次失败位置**、截断其后的监督；VAT 的思路是拒绝相关的位置加权，两者也不是同一种方法。若观察到 τ 改善但训练/推理总成本更高，仍不能判定工程成功。[Spec-AUF](https://arxiv.org/abs/2607.01893)、[VAT](https://arxiv.org/abs/2608.30135)

### E2：接受目标是否优于分布匹配？中成本

EAL 面向 greedy，WTV面向sampling。论文已包含 tree 设置，不能误写成“只测了链”；但它的概率 surrogate 并不是本地确定性 Top-k tree 的真实期望 τ。其 greedy 实验增益也远小于某些摘要式推荐可能让人想象的幅度。

建议先冻结head，在不同 checkpoint/anchor 上检验 surrogate 与实际接受长度排序的一致性；再决定是否新增训练臂。若一致性差，先检查候选树分支覆盖、位置依赖与argmax margin，不立即上 GRPO。此建议是本地迁移假说。[EAL/WTV](https://arxiv.org/abs/2609.24150)

### E3：数据数量不足，还是有效监督不足？低至中成本

ALR-IRA 从标签和anchor配置恢复监督；可先检查旧视频标签是否由相同 prompt、target路径、精度和停止条件产生，以及当前anchor是否总在相似状态。若重标注，应同时给 continue-SFT 与 OPD 对照相同的数据条件，否则不能归因于loss。[ALR-IRA](https://arxiv.org/abs/2609.38795)

一般 OPD 的少样本研究提示“query数”和“访问状态覆盖”不同，但不证明几个视频足够。后续多prompt实验需按视频去重拆分，不能让同视频不同prompt跨train/test泄漏。[One Training Example](https://arxiv.org/abs/2609.04172)

### E4：固定 b63 是否浪费验证时间？低至中成本

先在冻结head和固定输入上记录每轮 context长度、树节点数、draft/verify耗时、提交token数。使用粗略成本指标 `总 decode 时间 / 总提交 token 数` 判定收益；它是本项目分析量，不是跨论文speedup换算。

有证据显示长后缀低存活时，再单独测试动态深度/宽度。AdaFlash 的长度头、DSpark 的置信度调度、AdaSpark 的在线时间定价是三条不同路径；后两者的online不等于OPD。动态预算必须保留原target验证并再次做fp32 token审计，固定b63仍是训练方法比较的共同基线。[AdaFlash](https://arxiv.org/abs/2607.19223)、[DSpark](https://arxiv.org/abs/2607.05147)、[AdaSpark](https://arxiv.org/abs/2610.05774)

### E5：长视频是否导致 draft attention 稀释？低至中成本

按实际可见视觉keys、文本长度和生成位置分桶，避免把原始帧数当作attention有效长度。可先在训练/监控数据上判断错误率是否随这些变量系统变化。SharpDraft 的query scaling值得作为独立推理侧候选，但只改draft，不改target；视觉mRoPE和重复帧会让长文本假设不直接成立。[SharpDraft](https://arxiv.org/abs/2610.05106)

Sparrow/HiViS 提示：draft不显式读视觉token，仍可通过target text hidden获得视觉信息。这与旧uncached路径“文本hidden对视频无依赖”完全不同；不能用这些论文为旧路径辩护。先做blank-video、frame-shuffle、hidden依赖等诊断，再考虑窗口或层级融合；视频prefill计入端到端速度。[Sparrow](https://arxiv.org/abs/2602.15318)、[HiViS](https://arxiv.org/abs/2509.23928)

### E6：重新研究块内条件化？中至高成本

本地Markov负结果已表明增加τ可能仍降低速度。D-Loop提出共享backbone的第二次条件前向，适合检验“相邻位置重复导致首拒”的假说，但需要匹配的训练目标，不能拿未经训练的第二次推理当论文复现。建议先报告重复模式与实际首拒的关系，再决定是否值得增加一次draft pass。[D-Loop](https://arxiv.org/abs/2610.06011)

### E7：结构或强化学习支线，放在最后

TIGER的accepted-prefix reward与KL anchoring、DFlare逐层target feature融合、IPD teacher/student插值都值得跟进，但分别改变优化目标、结构或训练状态分布，调试成本较高。IPD精确采样的是插值策略，不是原target策略；只能考虑训练rollout用途。[TIGER](https://arxiv.org/abs/2607.11131)、[DFlare](https://arxiv.org/abs/2606.02091)、[IPD](https://arxiv.org/abs/2609.37170)

## 暂不进入无损主线的方案

- LVSpec/loosely speculative acceptance：可作质量—速度曲线对照，不能放入当前逐token一致主表。[LVSpec](https://arxiv.org/abs/2604.05650)
- target侧视觉剪枝/近似KV：除非完整验证恢复了原target条件并通过审计，否则改变目标模型。draft侧压缩与target侧近似必须分开。
- HB-SJD：其“visual OPD”是图像生成，不能直接当视频理解rollout方案。[HB-SJD](https://arxiv.org/abs/2608.18183)
- 在B1中用高概率top-k重归一代替全词表：即使某些general OPD任务可行，也违反本轮目标定义。
- 反复在fresh30选新结构/超参后宣称test收益：当前数据只能用于validation探索，最终需新冻结测试集。

## 阅读后的决策顺序

先完成 E0 与原 Phase C。如果KL降低而τ不升，优先E1/E2；若监督有效位置太少或标签错位，优先E3；若τ改善但walltime不改善，优先E4；若错误集中在长视觉上下文，优先E5。E6/E7需要新结构/训练预算，排在诊断之后。这是基于项目状态的工程判断，不是文献已经证明的最优路径。
