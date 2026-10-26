# DeepSeek 长程任务 B：GLANCE 视频时序条件响应与 top-8 候选缺失诊断

**请执行真实干预和模型前向实验，不是只写计划。** 同时遵循 `00_parallel_execution_rules.md`。用户今晚离线，独立推进，不等待任务A的结果。目标是取得可复现的正面/负面机制证据；不能为了“跑出收益”编造或过度解释结果。

## 核心研究问题

C6的42个非终止首阻断中有24个 `not_in_topk`，仅说明候选覆盖缺口，并**未证明是视频时序原因**。本任务独立检验：当视频事件顺序或实体状态变化影响冻结target在某个共享文本prefix上的token偏好时，draft对同样视频条件变化的分布响应是否系统性更弱？与一般视觉扰动、上下文错位、target高熵位置如何区分？本轮不训练、不改head/verifier、不用未打开的新测试集。

## 1. 审计与预注册

完整读根及适用AGENTS.md、`docs/plans/video-opd.md`、C4/C5/C6报告和原始trace/manifest、`repro/common.py`、`repro/aligned_forward.py`、真实VL视觉处理、视频帧提取、M-RoPE/VisionPositions、feature-path cached与target/draft前向代码。核对C6的真实C0固定anchor、`not_in_topk`定义和正确AR来源，明确C5 B-reference与C6抽样及旧轨迹差异。写独立 `prereg.json` 固定数据筛选、干预、控制、统计、排除与停止规则。使用C0及冻结target；不替换checkpoint。

首先对真实数据检查：有没有具有可核验事件先后/状态变化的样本及原始帧/时间信息；对每条给出视频证据、原时间戳、采样帧时刻、事件标注（若没有现成标注则明确为人工/视觉保守判断）、候选缺失位置。缓存16帧封装顺序不是原视频时间轴。**不得拿无事实证据的随机反序列声称“真实语义事件顺序干预”**。

## 2. 冻结小样本与三条件干预

从既有训练/validation中预先选择最多6条**合格**视频，每条最多4个经核验的有效共享prefix/offset（最多24对比单位）；优先选择C6真实 `not_in_topk` 且事件可解释的样本。事先固定排序规则（例如按稳定ID并按原始round先后选），不能挑看到效果好的样本。记录选中与排除的所有理由；如无足够合格视频则报告覆盖率，不凑数。不得使用未来target生成答案来挑有利干预。

每样本至多三个条件：
- O：真实原视频；
- T：经核验的时序干预（例如对有明确A→B事件的片段调换事件顺序，严格保留必要实体并标记干预会改变的问题语义；若无法合理实现，标注不可用并排除该样本的时序因果结论）；
- K：不改变关键事件先后关系的对照扰动（同处理预算的无关片段交换/微扰；如无合法对照，报告缺失，不强行制造）。

禁止创造不存在的事实标签。将具体干预脚本、参数、随机seed、原视频与产物哈希、帧位置信息保存。对O/T/K每种条件重新调用真实processor及target视觉融合、cached特征，不能复用O的视觉缓存。比较token之前核对实际输入帧与时间/位置元数据映射、visual token数、M-RoPE/offset、文本prompt一致性。

## 3. 共享prefix与模型前向正确性

只选在O/T/K语义都成立的文本prefix（包括question及既有答案前缀）；如果T改变了已经描述过的事实导致prefix矛盾，剔除该prefix或单列，不作为因果对比。记录每个条件文本token IDs、prefix hash、offset、mask、VisionPositions、M-RoPE及视觉特征绑定情况。

对每个合法配对，用冻结target与C0 draft在相同**文本prefix / offset / mask**上取真实logits。各条件分别重建各自视觉上下文，不共享视觉KV。先核实O复现已有C6对应首个可预测位置的top-8/rank与target greedy；允许声明数值容差，但若与真部署路径不一致，先定位dtype、cached路径、position与prefix，失败数据隔离。完整独立AR轨迹不应错套到T/K变体。

主分析只使用**首个真正由draft预测的offset**及固定共享prefix；额外offset单独作为探索分析，不把teacher-forced suffix与在线真实轮次混为一谈。目标模型对T/K的greedy token可能变化，分别比较它们对应条件下自身target的正确token是否在draft top-8；不要强迫使用原O条件的答案。

## 4. 统计与负对照

逐视频、条件、prefix保存：target/draft top-8 IDs及rank、目标token logprob、target greedy是否变化、每模型entropy/margin、`Δ log p`或分布散度、`Δtarget`与`Δdraft`方向及幅度、rank变化、覆盖损失；只做离线分布诊断，不声称直接等于τ或速度。

定义并计算一个可审计的响应指标：对共同候选token集合（如O/T/K target top-k并集，固定k=8）比较O→T 的target/draft log probability变化向量，并给出余弦相似度、L1幅度比（分母小于预先阈值则记NA）、top-8覆盖变化；同样计算O→K作控制。不得仅依据target top-1变化判定target理解正确。预先在prereg固定小幅变化阈值（如target L1<0.05时响应比NA）和有效样本判据，之后不得调门槛。

以**视频**聚合报告O→T与O→K的差异，给出正/负/混合案例，以及视频级bootstrap探索区间（样本太少则报告描述性指标而不夸大推断）。补充target entropy、prefix长度、offset、视频来源/难度的分层表；只把“视频扰动导致分布条件响应差异”解释为模型响应，不把它等同于人类事实正确性或真实时序理解缺陷。

## 5. 自主执行门禁和资源选择

先运行纯CPU数据选择、可用视频/帧审计和干预脚本测试，再在GPU互斥锁下执行小批量真实前向（例如先2条视频作数值/缓存门禁，成功后扩展至预冻结最多6条）。确认CUDA可用/显存后选择恰当dtype与批次；OOM不改变checkpoint/模型语义，可降低批量/串行或采用已有证实的运行配置。不得后台等待不结束；有限资源重试后输出充分CPU与阻塞证据。任务A GPU占用期间可做CPU数据审计、干预与测试，不写它的目录。

必须实际运行可行的实验；即便得到“target对干预基本不敏感”或“draft与target响应相近”，也作为负/不支持证据报告。若原数据不足以构造真实语义时序干预，至少输出可验证的样本可行性统计和对照试验，并明确**不支持时序机制结论**，不要悄悄换成不相关的图像扰动来满足产出。

## 6. 交付与停止范围

独立产物 `results/parallel_B_temporal/<timestamp>/` 保存 `prereg.json`、选样全清单、原视频元数据/哈希、干预前后帧索引/时间戳、测试、逐样本表、按视频汇总、图表可选、真实logit结果（注意存储体积）、stdout/err、源/配置/模型hash、运行命令、数值对齐/排除记录、`run_manifest.json`。报告 `docs/reports/2026-10-12-parallel-B-temporal.md` 写明结果属于：`支持可重复的条件响应差异` / `未观察到稳定差异` / `数据或实现无法判定`，并提出有机制证据支持的下一步训练设计或明确不建议训练。

**本任务禁止启动训练、改loss、多seed或打开新测试集**。不要自动把诊断扩展成新的蒸馏方法。不要并发编辑根 `AGENTS.md`：提供可粘贴的状态更新块，由后续统一汇总。

现在立刻开展源码/数据审计、构造合法干预、运行CPU检查与可用GPU上的真实target/draft前向，完成报告，不要只写研究计划。
