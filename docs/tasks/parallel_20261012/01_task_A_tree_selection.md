# DeepSeek 长程任务 A：GLANCE 可部署树选择与接受路径失配

**请执行实验而非仅写计划。** 请同时遵循同目录 `00_parallel_execution_rules.md`。用户今晚离线，无需阶段性确认；按门禁自行继续、定位问题和收口。成功定义是真实、可复现的结果（正或负），绝非强制优化成功。

## 任务目标

在冻结 C0、原始 top-8 候选和 b63k8 预算下，检验唯一预先固定的深度配额构树能否相对当前 best-first 改善正确 AR 前缀覆盖，若通过门禁再测试真实 verifier/在线无损 τ、时延。进一步解释概率代理目标与真实接受路径的关系，但不从单一试验冒充视频领域创新。

## 1. 审计阶段（必须真正读代码）

完整阅读适用 AGENTS.md、`docs/plans/video-opd.md`、`docs/reports/2026-10-12-gain-stuck.md` §12–13、`docs/reports/2026-10-12-current-ar-fixed-anchor.md`、Phase C5 B-reference原始记录、C6 manifest与原始 GPU 候选、`repro/diag_bref_align.py`，及 `GLANCE_repo/glance/` 构树、verifier、commit 的真实调用链。确认真实路径和模型 hash，不依赖本提示中的路径简写。

查明并记录：budget63的root/bonus计数、offset最大深度、每位置候选来源及分数、路径累计分数与best-first展开/停止/去重/tie-break、parent/index、祖先mask、position/M-RoPE与VisionPositions、EOS和剩余长度裁剪。验证 best-first 是否已是其原有概率代理目标下的最优预算分配；如果是，不能声称新方法更好地求解同一最优目标。

重要：C5 B-reference的48轮前8轮和 C6等距抽样48轮不相同。**本任务使用C6的6视频/48轮；可使用C5核查实现，但不得混用标签或聚合。** 核对C6当前正确AR reference与原GPU top-8分数，历史失败pilot不能进入正式数据。

## 2. 冻结的唯一实验策略

A0 基线：生产best-first，以真实draft score/累计路径概率构建前缀闭合树，b63k8预算，最多15个draft offset（以代码核实）。

A1 固定深度配额：depth1–3各最多5个保留节点；depth4–15各最多4个保留节点；按depth由浅入深，仅从上层已选节点扩展合法候选，在该深度按**与基线相同的累计路径分数**排序，确定性tie-break后选前q。总上限 `3*5+12*4=63`（不含root）；若EOS、剩余生成长度或父节点不足则少选，不用未来深度填补空缺。实现前要核实项目中depth的定义与已使用的GPU候选。

严格不使用target logits、正确token、oracle路径或样本收益构树；仅使用原draft候选、分数、depth和prefix；不新调用head。**不得搜索第二套配额、调整quota或事后改阈值**。仅使用独立wrapper/实验模块实现；不改变默认生产树。

在计算A1收益之前，先将参数、样本ID、代码/config/input哈希、tie-break、口径、门禁和停止条件写入本任务独立 `prereg.json` 与实验报告的预注册节。承认在已反复查看的validation上开发，不宣称独立验证。

## 3. 可复现与正确性门禁

先写测试并执行：预算、深度、候选来源、无重复、前缀闭合、parent/index、mask、tie-break确定性、EOS、末轮截断、root和bonus；小规模例子穷举校验逐深度配额贪心规则；基线调用保持原树字节/字段一致。

Stage 0 CPU：只重放C6真实保存的GPU候选ID及分数，不允许重新CPU log_softmax后冒充生产得分。重建48个A0 baseline节点/parent/depth、正确AR路径；要求精确复现`mean L_tree=2.895833`、`Σ(L_raw−L_tree)=40`、commit gap=0，并核对每条anchor原始tree。若不一致，定位浮点、tie-break或样本ID；修复正确性并保留失败证据。不能重建则停止收益宣称，交付审计及阻塞报告。

Stage 1 CPU反事实：48个相同anchor上构建A1，输出每anchor `L_raw,L_tree_A0,L_tree_A1,ΔL_tree,Φ_A0,Φ_A1,first_missing_depth, recovered/lost path,depth counts,actual nodes`；按视频分别聚合，再等权平均；视频级bootstrap CI只用于探索，不能将48 anchors当独立数据。`Φ(T)=sum_{node in T} exp(sum_path log p_d)`。

必须在看结果前将以下具体规则写进prereg：1) 净 `ΣΔL_tree>0`；2) 至少2条视频的平均`ΔL_tree>0`（防止全由1视频推动）；3) 早期覆盖的严重损失定义为任一视频在depth<=4上相对基线净损失超过该视频8个anchors中的2个draft tokens（统计定义与短轨迹处理在prereg精确固化）。**三项全满足才进入Stage 2**。阈值是资源筛选，非显著性。未通过则作为负结果收口，禁止调参。

Stage 2 真实GPU局部验证：串行持GPU锁，先检查显存。用同一正确prefix、target/verifier与独立缓存，核对树mask、位置与视觉绑定、M-RoPE、accept/bonus/EOS；A0重放与已有对应结果一致；A1每条提交必须是独立AR continuation的前缀，且`L_actual_A1=L_tree_A1`逐anchor。失败先定位修复（最多合理修复），失败记录隔离。

Stage 3 在线：仅前门禁通过，在C6的6条视频上分别完整解码A0/A1直到EOS或预设长度上限；两者各6条、全部逐token SPEC=AR。每种策略自然产生各自在线anchors；分别记录mean per-video τ、median、pooled τ、video paired delta、rounds、generated tokens、node counts、树构造耗时、target verify与decode-only、可比端到端时间。计时同卡、同精度、相同warmup、交替A/B顺序，至少3次重复（资源足够时）；昂贵trace不混入正式计时，外部争用导致不可比则明确`speed=inconclusive`。不在本任务扩展到29视频或C1/C2，也不打开独立测试。

## 4. 科学解释和交付

判定属于：`未发现净局部改善` / `局部有改善但在线未转化` / `在线τ改善但速度未知或无收益` / `在线τ与可靠速度均改善` / `正确性或资源阻塞`。区分Oracle路径、局部draft长度、历史τ与实际速度。不因单一A1失败推断所有树策略均无效。

输出独立目录 `results/parallel_A_tree/<timestamp>/`，保存源码快照/哈希、命令、完整stdout/err、原始逐anchor表、视频表、bootstrap、测试、门禁、CPU/GPU运行、失败pilot、`run_manifest.json`；报告 `docs/reports/2026-10-12-parallel-A-tree.md`。报告需提供明确的“下一轮方法研究是否值得开展及为什么”的结论，并提供可复制到AGENTS.md的简短状态更新文本，不直接并发修改根AGENTS.md。

**立刻开始核查代码、执行Stage0和Stage1；满足门禁则自主继续，不要只输出研究计划。**
