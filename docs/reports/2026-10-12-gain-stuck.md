# 收益瓶颈诊断：2026-10-12 ACL 审计与 CPU 补证

本轮完整读取 `GLANCE_ACL_diagnostic_prompt.md`、`AGENTS.md` 及相关实现后执行。
**已完成现场审计、旧产物再分析、输入重放和 CPU 测试；正确 AR reference、真实 tree-anchor
oracle 和 D4 未完成，不能把本轮标为 D1–D4 全部完成。**

审计启动时本报告尚不存在，虽然交接入口已经引用它；本文件首次建立，不覆盖历史报告。
产物目录：`/mnt/local/mcy/GLANCE/results/phaseC5/diagnostic_acl_20261012T213006/`。
初始 CPU 审计的证据快照固定在该目录 `snapshots/`；后续补证单独保存。
**收口时新增：原有训练组 tree 评测已完成并纳入；最终范围与结论以 §10 补注为准。**

## 1. 已有产物 → 有效性 → 可复用内容 → 缺失证据 → 最小补跑

| 已有产物 | 有效性与可复用范围 | 缺失证据 | 最小补跑/本轮处理 |
|---|---|---|---|
| phaseA4 C0、phaseC4 C1/C2 validation 原始 JSON | 配置一致；各 30/30 保存的 SPEC/AR token 完全相等；去同一 warmup 后 n=29 | 原始 top-8、完整树及独立 anchor continuation | 复用 τ、token、原始计时；局部 oracle 保留 N/A |
| `d1_anchors.jsonl`、三份 `d1_C*.jsonl` | 70 视频 ×24 anchors=1680；同一冻结 proposal、位置、mask；teacher argmax 与 checksum 跨三臂一致 | 正确 AR reference；概率、entropy、top-k IDs；实际可达性 | CPU 重算两流目标、三段 rank/top-k 代理；不冒称正确 AR 诊断 |
| 三份 `phaseC5_chain_*_fresh30.json` | 各 30/30 AR 一致；实际 b1k1 单候选探针 | 完整 block16 top-1 链 | 不重跑已有探针；完整链需 b15k1，当前未运行 |
| `phaseC5_tree_*_g1g2` | 本轮开始时正在运行，初始快照尚无最终 JSON | 最终 JSON/全组门禁 | 不重复启动、不改运行脚本、不终止进程；初始总表相关 τ 为 N/A |
| phaseC4 训练日志、启动脚本、C2 配置 | 支持恢复两臂训练顺序和监督量；日志未直接保存视频 ID | 原始逐更新 ID 审计记录 | CPU 按真实随机顺序重建、对齐 10 个 C1 长度与 C2 无跳过计数；注明重建证据级别 |
| D4 | 无产物 | 一致性门禁、合格视频对、有效共享前缀 | 因已确认参考不一致而 blocked；未做视频条件模型前向 |

没有训练、optimizer update、新超参搜索、新测试集评测、下载、依赖修改。
仅增加独立诊断代码；既有模型、原始结果、视频缓存及运行脚本均未覆盖。

## 2. 环境、模型身份与计算边界

- `GLANCE_ROOT` 未设置；`common.py` 实际解析为 `/mnt/local/mcy/GLANCE`。
  项目根目录没有 `results/`、`checkpoints/` 软链接，不能直接按相对路径访问。
- shell 默认 Python 为 `/usr/local/anaconda3/bin/python`；本轮显式使用
  `/home/mcy/.conda/envs/fin_research/bin/python`（Python 3.11.14、torch 2.9.1+cu128、
  transformers 4.57.3、pandas 2.3.3）。
- C0：`checkpoints/head_video439_aligned/step1000`；C1：`checkpoints/phaseC4_C1/step10`；
  C2：`checkpoints/phaseC4_C2/step10`。从正式评测 JSON、phaseC4 启动脚本、训练日志及 C2
  `opd_fullkl_config.json` 交叉追溯，未使用 phaseC3。
- 各 head 的 `model.safetensors` SHA-256 在 CPU 审计前后相同；哈希、config、来源在
  `manifest.json/head_checkpoints`。所有新脚本无模型参数更新路径。
- 历史 D1 与部署评测为 fp32 target/fp32 head，cached context，SDPA；训练为 fp32
  teacher/context、bf16 head。D1 的 full-KL 是**部署精度下的冻结目标 replay**，
  不是 bf16 训练前向的精确数值重现，也不是 C1 的原生 variable-reveal CE。
- 启动检查没有符合既有 fp32 门槛的空闲 GPU；后续 GPU 6 虽接近/超过显存门槛，仍有
  活跃任务。未抢占共享 GPU，未无限排队或重试。资源快照在 `audit.json`、`supplement.json`。
- 另发现 phaseC4 报告称“两臂 checkpoint 各含 optimizer_master.pt”不准确：
  C1 step10 实际只有 config/model，C2 才有 optimizer state。这不影响推理复用，
  但 C1 不能据此宣称精确恢复 Adam 状态。未恢复或修改任何优化器。

## 3. 关键实现审计：当前策略 anchors 不等于当前 target AR 轨迹

`opd_train_video_fullkl.build_video_trace` 从 `record['input_ids']` 取整条旧轨迹；
`opd_online_rollout.rollout` 用当前 head 重算 proposal 和 anchor 跳步，但 `want` 直接取
`seq_ids[pos+1:pos+block]`。接受长度是对**旧序列 token**的匹配，不是调用当前 fp32 target
进行增量验证。随后 full-KL teacher 又是在 fp32 下对该旧 prefix / draft proposal 前向。

因此应保留“proposal/anchor 位置随当前 head 重算”这一事实，同时撤回将其等同于
“与当前 fp32 部署 AR 完全一致的在线 rollout”的强解释。未证明这是负结果的主要原因；
本轮没有训练消融，也没有分离旧轨迹生成精度、历史数值路径等来源。

直接证据：

- validation 旧 continuation 与已保存 fp32 增量 AR **30/30 不相同**；720 anchors 中
  **381** 个的 prefix（包含 root）仍位于共同 token 前缀内，其余 **339** 个不在。
  该 30 条包含原 warmup；正式 τ 统计仍为另外 29 条。
- 例：`v_-nl4G-00PtA` 首分叉在生成 offset 14；`v_0UTrTg3DF5g` 在 29；
  `v_1d_U_2HIPmY` 在 35。逐视频两侧 token ID、长度和共同前缀在
  `trajectory_ar_alignment.json`。旧视频生成长度与新 AR 也可能不同。
- 全 70 条/1680 anchors 中，**58 anchors** 的旧 `n_acc` 与当前 A-teacher 对冻结 proposal
  的 leading greedy agreement 不同；有 **44 个旧 accepted 位置**的 teacher token
  不匹配，以及 **16 个旧 first-rejected 位置**的 teacher token 实际匹配。
  详见 `d1_reference_mismatch.json`。这些是位置计数，不能当独立统计样本。
- **70/70 输入 prompt 逐 token 重放一致**，不仅是长度相同。首条重复处理的 input IDs、
  pixel tensor、video grid 完全相同。见 `input_replay_v2.jsonl`；这排除了本次 processor
  prompt 编码变化，但不证明所有模型中间特征/增量数值路径一致。
- 输入重放还保存了实际帧索引与视频元数据。索引是缓存的 16 帧视频内部的索引；
  `fps=30`、`duration≈0.533s` 是缓存封装信息，**不是原始长视频时间轴**。
  原视频采样时间映射未恢复，不能据此做时序机制结论。

部署 AR 门禁通过与上述发现不矛盾：前者验证正式 decode 的输出无损；后者指出训练/诊断
所用旧轨迹不等于该部署输出。不能据此宣称 tree verification/commit 已发现错误。

## 4. D1：可以确认的变化与不能确认的转换

复用全部既有 anchors（24/视频，不重新采样）。统计单位为视频；配对 bootstrap 2000 次，
seed=20261012。先对一个视频的所有有效 accepted 求 forward-KL 均值、所有 rejected
按 `0.8^(k-1)` 求 reverse-KL 加权均值，两流等权；不平均三段或 microbatch 均值。
三段保持 C0 旧 `n_acc` 分组，且明确其局限。

**A reference** 为旧轨迹 prefix + 冻结 C0 proposal；suffix 中的 teacher token
绝不改名为正确 AR token。`teacher_bits` 是位模式求和而非无碰撞哈希；三臂 argmax 也一致，
支持重复 teacher 配对，但不是完整 logits 的逐位持久化证据。

保存的 rank 是严格大于目标 logit 的 token 数 +1，因此 `rank<=8/32` 只能作为 top-k
覆盖代理。旧产物没有实际 top-k IDs，无法复核 cutoff ties；新 metrics helper 和测试已
明确区分 torch.topk 实际成员与共享排名。

下面均为 **C2−C0**，百分点以 pp 表示：

| 数据组 | n | A full-KL replay Δ | A 首拒 rank Δ | A 首拒 top-1 Δ | A 首拒 rank≤8 Δ | 真实 tree Δτ |
|---|---:|---:|---:|---:|---:|---|
| C2 本轮曝光 | 10 | −0.724883 | −0.850000 | +18.333 pp | +2.917 pp | 初始快照 N/A |
| 等量本轮未曝光训练视频 | 10 | −0.127754 | −0.087500 | +4.583 pp | −0.833 pp | 初始快照 N/A |
| 全部预冻结未曝光对照 | 20 | −0.127123 | −0.089583 | +3.958 pp | −0.625 pp | 初始快照 N/A |
| validation | 29 | −0.129751 | −0.104885 | +3.736 pp | −0.144 pp | +0.002658 |

validation 的 A 首拒 `rank≤8` 差异 CI 为 [−0.575,+0.287] pp；rank 差异 CI
[−0.333333,+0.145223]。loss 下降进入了旧首拒位置的概率目标与 top-1，但没有显示 top-8
候选边界的可靠改善，更没有据此建立正确 AR 连续覆盖或实际 τ 的因果链。

同一 validation 上，accepted forward-KL Δ=−0.010069、accepted top-1 Δ=−1.797 pp；
首拒 reverse-KL Δ=−0.305124；strict suffix reverse-KL Δ=−0.158654。
**KL 下降并不保证贪心 token 或候选边界变好**。这也不能推导“suffix 梯度无用”。

**B reference 未完成**：没有独立正确 continuation 上的完整 target 分布、候选概率、
entropy/margin、top-k IDs 或固定 anchor 连续覆盖。共同 prefix 内的 381 anchors 也
不能自动把所有后续 offsets 的 A-teacher 分布改标 B，首拒后的 teacher 条件仍可能分叉。
C1 原生 CE replay 同样 N/A。完整三臂、三段、组别和区间见 `summary.json`、
`d1_per_video_A.json`、`d1_tokens_A.jsonl`、`diagnosis_table.csv`。

## 5. D2：真实 decode 可复用，局部 oracle 尚缺

`build_budget_tree` 的 budget 数的是候选节点，**不含 root**；packed target 输入另加 root。
b63k8 每轮至多验证 64 节点；现有“chain”b1k1 每轮只有 1 个 draft 候选、2 个验证位置，
但仍做一次完整 block16 head 前向。它同时改变宽度与深度，不能作为完整 chain/tree
消融。使用同一 `generate` 路径的 b15k1 才能生成 15 offsets 的 top-1 链；与 b63k8
仍非等验证预算。库中的独立 `chain_generate` 还使用另一 target 验证路径，不能未经门禁替代。

| 拓扑 / 数据 | n | C0 mean τ | C1 mean τ | C2 mean τ | C0/C1/C2 decode 秒合计 | AR 门禁 |
|---|---:|---:|---:|---:|---|---|
| b63k8 / validation | 29 | 3.737664 | 3.722464 | 3.740323 | 471.625 / 223.616 / 223.179 | 各 30/30 |
| b1k1 单候选 / validation | 29 | 1.686480 | 1.675060 | 1.689073 | 373.433 / 377.365 / 375.272 | 各 30/30 |
| b15k1 完整 top-1 链 | N/A | N/A | N/A | N/A | N/A | 未运行 |
| 同一 tree anchor 四个局部 oracle | N/A | N/A | N/A | N/A | N/A | 缺少原始候选/树 trace |

计时是各原始文件 `ms_per_token × gen_tokens` 的合计，warmup 已按 validation 固定身份去除。
C0 与 C1/C2 来自不同时间/负载的历史运行；明显计时差异**不能用来声称训练加速**。
沿用既有 decode-only 窗口（首次 drafting 在窗口外），不是端到端时间。

视频级精确配对重算：

| tree 对比 | mean Δτ | 95% CI（本次 seed、2000 次） | 正向 / 变化 |
|---|---:|---|---|
| C1−C0 | −0.015200 | [−0.045255,+0.013325] | 9/29；20/29 |
| C2−C0 | +0.002658 | [−0.020952,+0.027658] | 7/29；16/29 |
| C2−C1 | +0.017858 | [−0.015702,+0.055659] | 11/29；22/29 |

区间与旧报告略不同来自 bootstrap seed/次数不同，没有从 rounded means 相减。
另 b1k1 C2−C0=+0.002593，CI[−0.001989,+0.007477]；它只描述单候选探针。

代码在 EOS/输出长度裁剪之前记录 `accepted` 与 `tau=accepted+1`。validation 中
C0/C2 各有 **1 个** raw accepted draft token 最终被裁掉，C1 为 0。
`d2_per_video.json` 同时保存 raw 与 terminal-clipped 数量；原 τ 保留历史语义，
不能直接当输出有效 draft-token 长度，`gen_tokens/rounds` 也不是同一个指标。

新增 `local_oracles` 已测试：offset 从 root 后的 1 开始；budget 不含 root；所有长度按
EOS/remaining clipping；只接收经独立 AR 验证的 reference；检查前缀闭合、候选成员、
预算与 `L_actual≤L_tree≤L_candidate≤L_raw`。对当前 independent-offset 架构，
理想正确链只需每个深度一个候选节点，所以 `L_candidate=min(L_raw,budget)`。
**这里只是已测算子，不是模型实测 oracle 结果。** 没有 trace 就不填数，也不能把局部
空间换算全局 τ。首拒的预测缺失/树选择/提交分类当前 N/A。

## 6. D3：曝光集合、真实预算和关联证据

两臂各 10 条，交集 **0**；不能合并为“同一批训练视频”。C1 顺序按实际 NumPy permutation
重建，与日志十个 L 逐一相符；C2 按 Python shuffle 和配置重建，十次日志均一视频、
24 anchors、零跳过，旧 update-1 replay 计数亦通过。日志不直接记录 IDs，故身份为
多项证据约束下的重建，不伪称直接从日志读出了 ID。

- C1：240 anchors，**2404 positions**，权重和 **1008.353666939**。CPU 使用实际
  `sample_reveals/live_anchors/build_blocks` 和同一 RNG 消费顺序复算。
- C2：240 anchors，**3600 offsets=485 accepted+3115 rejected**；accepted 权重和485，
  rejected 权重和 **786.725809440**。两流分别归一，不能把两者合成与 C1 相同的监督预算。
- 监督位置比=1.49750，约1.50×。视频数、updates 一致不等于视频身份或监督预算完全配平。
- 对照沿用诊断前冻结的 `phaseC5-g2` hash 排序 20 条；其前10条构成按臂等量对照，
  同时保留20条完整结果，未按收益挑选。未匹配输入长度/难度是限制。
- C1-only、C2-only、未曝光20、validation30 的视频身份不重叠；对照前10与20的嵌套关系
  在 manifest 明示。缓存内容哈希和 source-ID 检查见 `data_manifest.json`、
  `d3_exposure_manifest.json`；没有原视频 URL/片段谱系，不能声称已排除所有同源关系。
- “本轮未曝光”组属于历史训练池，绝不是 never-trained 或 untouched test。

A-reference 上，C2 相对 C0 的变化在 C2 曝光10条比未曝光10条更大：
loss 差中差 −0.597129，CI[−0.836326,−0.386142]；旧首拒 top-1 差中差
+13.750 pp，CI[+7.917,+20.000]；rank≤8 差中差 +3.750 pp，CI[+0.417,+7.083]。
使用组内视频配对差，再对两组独立做视频 bootstrap，见 `supplement.json`。

**支持旧参考下的曝光关联；真实 τ 收益集中和泛化仍证据不足。** 原因包括参考路径不一致、
样本每组10条、曝光非随机分配、初始快照训练组 tree 尚未收齐。不能把关联改写成记忆机制。

## 7. D4 门禁、测试与失败记录

D4 状态 **blocked / not_run**。已确认 D1/rollout 参考路径不一致，按用户指示先交付此阻塞。
没有构造时序干预、没有视频条件下 target/draft 分布前向，也没有时序理解优劣结论。
缓存 key helper 只测试了内容、帧序、时间、processor 和 intervention ID 隔离；
**不能把 helper 单元测试当作真实干预缓存门禁通过**。

`test_diagnostic_acl.py` **16/16** 通过，覆盖排名/实际 top-k 与 ties、三段/valid mask、
全局两流权重及空流、视频配对 bootstrap、b1 与 b15 拓扑、oracle 预算与前缀闭合、
root/bonus/EOS/长度截断、未验证 reference 拒绝、重复算子与缓存隔离。
模型前向重复性未运行；processor 重复性实测1条，不能泛化为模型数值门禁。

两项开发失败保留解释：

1. JS(p,p) 的 fp32 计算为 1.76e-8；初始测试使用精确零失败，改为明确 1e-7 容差后通过，
   没有放宽部署 token 门禁。
2. 输入审计首次显式 `return_metadata=True` 时，当前 processor 尝试把 VideoMetadata
   转成 tensor，首条前失败。原脚本和日志保留为 `audit_acl_inputs.failed.py`、
   `input_replay.log`；改为仅在本进程观察内部 metadata，保持 evaluator 原始调用不变，
   `input_replay_v2.log` 70/70 完成，未改共享依赖。

## 8. 结论与唯一下一步

| 问题 | 判定 | 范围与限制 |
|---|---|---|
| 有无实现或口径不一致？ | **支持** | 30/30 旧 continuation 分叉；58/1680 anchors 接受标签与 teacher 不同；b1 非完整链；末轮计数先于裁剪。不等于部署 commit 有 bug |
| loss 是否进入有效候选并延长接受长度？ | **证据不足** | A-reference loss 下降有证据；validation 首拒 rank≤8 与 tree Δτ 区间均跨零；B-reference 尚缺 |
| 瓶颈主要在预测还是树选择？ | **证据不足** | 原始 top-8/树结构没有保存，四个局部 oracle 无实测数；不能用 b1/tree 差替代 |
| 有无曝光相关收益集中？ | **支持（限旧参考预测变化）**；真实收益**证据不足** | C2-only vs 未曝光10条 A-loss/rank边界改善更强；不能外推真实 τ、泛化或记忆因果 |
| 有无视频时序条件响应差异？ | **证据不足** | D4 未运行；没有事实核验/有效干预，不将 target 当真值 |

**唯一建议：优先做“当前 fp32 AR 轨迹与旧轨迹的冻结 anchor 一致性小实验”，不训练。**

假设：旧轨迹/接受标签不一致影响了当前 loss→rank→首拒诊断，必须先把参考与部署对齐，
才有依据选择数据、预测目标或候选选择方向。固定6条非 warmup validation，按 ID/输入长度
分层、seed20261012，不看收益；先2条门禁，通过才扩展至6条。每视频最多8个实际 C0
tree anchors：复用现有正确 AR token，在相同 processor/context 精度下重建并验证
incremental prefix；保存原始候选、树、实际提交，分别计算 A/B reference 和四个局部 oracle。
C0/C1/C2 共至多144个固定 anchor head评分；额外 target 计算为6条增量 AR/一致性重放和
至多48个 anchor 的 A/B continuation；串行，仅使用合格空闲 GPU，不新增训练样本或测试集。

推翻标准：在输入/特征/精度门禁通过后，旧与对齐参考得到的关键位置候选边界及首拒结论
基本相同，且可用固定的局部 oracle 解释真实提交；则“该不一致实质影响诊断”的假设不获
支持，不应继续把它当作主要收益瓶颈。相反若差异改变结论，先修复诊断/rollout参考再讨论
训练机制。这个实验仍不证明任何新方法有效；本轮未擅自启动。

## 9. 复现与交付入口

代码：`repro/audit_diagnostic_acl.py`、`repro/diagnostic_acl_metrics.py`、
`repro/audit_acl_inputs.py`、`repro/finalize_acl_audit.py`、`repro/test_diagnostic_acl.py`。
总表 `diagnosis_table.csv` 覆盖比较对象×组别×reference×loss/rank/top-k×首拒/oracle×τ×
耗时×n×门禁；N/A 附原因。脚本不会将缺少的诊断填零；真实观测的零保留为零。

```bash
cd /home/mcy/Projects/GLANCE
PY=/home/mcy/.conda/envs/fin_research/bin/python
# 必须用一个尚不存在的独立目录；不覆盖本轮或历史原始结果。
RUN=/mnt/local/mcy/GLANCE/results/phaseC5/diagnostic_acl_NEW_ID
"$PY" -u repro/audit_diagnostic_acl.py --out "$RUN" > "${RUN}.command.log" 2>&1
"$PY" -u repro/audit_acl_inputs.py --split "$RUN/snapshots/phaseC5/split.json" \
  --out "$RUN/input_replay_v2.jsonl" > "$RUN/input_replay_v2.log" 2>&1
"$PY" -m unittest discover -s repro -p test_diagnostic_acl.py -v > "$RUN/tests.log" 2>&1
"$PY" repro/finalize_acl_audit.py --run "$RUN" > "$RUN/supplement.log" 2>&1
```

原始 inputs 的小文件快照、head 前后哈希、脚本哈希及实际命令在 manifest；
processor/tokenizer 配置哈希、额外输入审计和曝光对比在 supplement。完整日志包括独立
目录旁的 `.command.log`。若复现时原有后台评测已完成，脚本会记录新增输入及其快照；
相应结果范围会扩大，应使用新 run ID，不能修改此初始审计快照。

## 10. 同日收口补注：原有训练组 tree 已完成，已复用全部证据

在撰写初始审计报告期间，用户环境中原已运行的三个 g1/g2 tree 任务自然结束。
本轮没有启动这些任务，也未等待/重试队列；完成后立即复用最终 JSON，而不重复模型推理。
新输入的独立快照和完整重分析保存于本 run 下 **`completed_tree_audit/`**，初始结果保留。

**最终主诊断总表**：`diagnosis_table_final.csv`（计时已排除作业warmup，并独立列timed n）；配对原始值与所有区间：
`completed_tree_audit/summary.json`；组间 τ 对比、跨臂 AR 和计时 warmup 细节：
`completed_tree_audit/completed_eval_checks.json`。上文“初始快照 N/A/正在运行”只保留审计时点，
不代表当前状态。D1-B、完整链、局部 oracle 和 D4 仍 N/A。

追加40条后，三臂 tree **各70/70 SPEC==AR**，且 C1/C2 的 AR 输出均与 C0 逐 token 相同。
旧 continuation 与当前部署 AR **69/70分叉**；**883/1680** anchors 的 prefix 含root仍在
共同前缀内，797个不在。唯一未分叉样本不改变validation 30/30分叉的结论。
输入重放仍是相同70条、70/70 prompt相等，未新增任何测试集。

| 数据组 | n | C0 τ | C1 τ | C2 τ | C2−C0（95% CI） | C2−C1（95% CI） |
|---|---:|---:|---:|---:|---|---|
| C1-only | 10 | 3.955118 | 3.941897 | 4.010780 | +0.055662 [+.009434,+.095217] | +0.068883 [+.022518,+.121099] |
| C2-only | 10 | 3.843711 | 3.810357 | 3.938734 | +0.095023 [+.022485,+.177488] | +0.128377 [+.040566,+.229204] |
| 本轮未曝光，hash前10 | 10 | 3.802099 | 3.815971 | 3.813388 | +0.011290 [−.015298,+.039024] | −0.002583 [−.061650,+.057167] |
| 本轮未曝光，完整20 | 20 | 4.069697 | 4.048893 | 4.136148 | +0.066451 [+.026797,+.110563] | +0.087255 [+.015263,+.176557] |
| validation | 29 | 3.737664 | 3.722464 | 3.740323 | +0.002658 [−.020952,+.027658] | +0.017858 [−.015702,+.055659] |

这些是既有训练池上的诊断关联，不能推翻phaseC4 validation收口规则，也不是untouched test
效果。C2在没有本轮曝光的完整20条训练池视频上也有τ变化，不能说收益只发生在曝光视频。

**曝光集中结论对对照范围敏感**：

- C2-only10 相对未曝光前10的配对 Δτ 差中差 +0.083734，CI[+0.005097,+0.167164]。
- 相对预冻结的**完整未曝光20条**，差中差仅 +0.028572，CI[−0.058555,+0.117143]。

不能筛选前10条的较有利区间作为主结论。A-reference 的loss/排名变化集中仍有证据；
**真实τ有无稳健曝光集中：证据不足**。也不能把完整20条的结果解释为已证明跨域或新视频泛化。

原 g1/g2 入口 `--warmup 1` 把 `v_-cAcA8dO7kA` 标为该作业warmup。本次τ比较固定组别身份，
保留这条样本，不将 C1-only 任意减为9条；正式 validation 只去原来的 `v_-nl4G-00PtA`。
**计时另外排除该作业warmup**，不能把timed n与τ n混用。C1-only计时n=9；其余组n相同：

| 组别 | timed n | C0/C1/C2 decode 秒合计（不作速度结论） |
|---|---:|---|
| C1-only | 9 | 55.396 / 56.746 / 55.805 |
| C2-only | 10 | 76.028 / 78.464 / 76.738 |
| 未曝光前10 | 10 | 79.524 / 81.263 / 81.351 |
| 未曝光完整20 | 20 | 154.381 / 158.308 / 156.236 |

`completed_tree_audit/diagnosis_table.csv` 中原 `elapsed_s` 是组内原始计时合计，包括作业warmup；
交付版 `diagnosis_table_final.csv` 已采用 `completed_eval_checks.json/timing` 的排除后值，
并保留timed n与τ n的差异。不进行硬件/负载未控制的速度归因。

任务结束时GPU0–2已释放，**资源不足不再是最终阻塞理由**。最终收口依据是用户
“发现明确实现不一致先交付该阻塞”的门禁：本轮先交付已经证实的旧轨迹/teacher/部署
参考不一致，不在未验证的reference上继续补模型机制结论或扩展D4。
§8的唯一建议保持为小规模当前AR/旧轨迹一致性对照，无新训练。

补注复现命令（仍只读，无GPU模型推理）：

```bash
"$PY" repro/audit_diagnostic_acl.py --out "$RUN/completed_tree_audit" \
  > "$RUN/completed_tree_audit.command.log" 2>&1
"$PY" repro/check_acl_completed_eval.py --run "$RUN/completed_tree_audit" \
  > "$RUN/completed_tree_audit/completed_eval_checks.log" 2>&1
"$PY" repro/package_acl_delivery.py --run "$RUN" > "$RUN/delivery.log" 2>&1
```

`package_acl_delivery.py` 为本次交付封装入口，读取保留的 `AGENTS.before.md` 生成变更diff；
换新run复现时先把维护入口的当前版本复制到该文件。`delivery_manifest.json` 连接最终表、
初始/完成后两套审计、输入重放、测试、文档/脚本哈希与各阶段未运行状态。

## 11. phaseC5 D1/D2/D3 主交付：三段预测变化 → 首拒 → chain/tree → 真实 τ（并发子任务追加）

**追加说明**：本节由与 §1–§10 并发的 phaseC5 执行子任务追加，不改动上文的任何结论与产物；
重叠处按"独立复算"处理（§11.2 的 58/1680 与 §3 的计数独立相符，§11.6 的 τ 与 §10 独立相符）。
全部只读：没有训练、没有 optimizer、没有 `save_pretrained`、没有评测任何新测试集候选。
**D1 的 reference 是 §3/§4 的 A reference（旧轨迹 + 冻结 C0 proposal），τ 是真实部署 decode
（lossless 门禁通过）**；两者不是同一个对象，§11.4 专门度量它们的一致性。

### 11.1 产物与命令

| 类型 | 路径 |
|---|---|
| 诊断脚本（新增，只读） | `repro/diag_gain_stuck.py` |
| D2 启动器（新增，只读） | `repro/run_phaseC5_d2.sh` |
| 预注册划分（先于任何结果写出） | `results/phaseC5/split.json`、`results/phaseC5/group_g1g2_rows.parquet` |
| C0 anchors（固定 prefix/proposal/mask） | `results/phaseC5/d1_anchors.jsonl`（70 视频 × 24 anchors，1680） |
| D1 逐视频指标 | `results/phaseC5/d1_{C0,C1,C2}.jsonl` |
| D1 主表 / 稳健性表 | `results/phaseC5/d1_tables.md`、`d1_summary.json`；`d1_tables_aligned.md`、`d1_summary_aligned.json` |
| D2/D3 评测 | `results/phaseC5/phaseC5_{tree,chain}_{C0,C1,C2}_{fresh30,g1g2}.json`+`.log` |

```bash
cd /home/mcy/Projects/GLANCE/repro
PY=/home/mcy/.conda/envs/fin_research/bin/python ; R=/mnt/local/mcy/GLANCE
# 预注册（先跑，早于任何诊断指标）
$PY diag_gain_stuck.py split --records $R/data/video_greedy_499.pt \
  --pool $R/data/pool_499.parquet --fresh30 $R/data/video_fresh30.parquet \
  --c1-jsonl $R/results/phaseC4/C1.jsonl --out $R/results/phaseC5/split.json
# D1：C0 生成 anchors（并重放 C2 update-1 计数）→ 三臂各一次 fp32 metrics pass
$PY diag_gain_stuck.py anchors --split .../split.json --records ... --pool ... \
  --student C0=$R/checkpoints/head_video439_aligned/step1000 \
  --out .../d1_anchors.jsonl --gpu 0 --verify-c2-u1 $R/results/phaseC4/C2.jsonl
$PY diag_gain_stuck.py metrics --split ... --records ... --pool ... --anchors ... --student C1=... 
# D2/D3：tree=b63k8、chain=b1k1，均带 --ar 门禁、fp32
./run_phaseC5_d2.sh tree  C0 $R/data/video_fresh30.parquet  phaseC5_tree_C0_fresh30  0
./run_phaseC5_d2.sh chain C0 $R/results/phaseC5/group_g1g2_rows.parquet phaseC5_chain_C0_g1g2 0
# 汇总（CPU）→ 主表 + teacher-consistent 重新划分 + 公共前缀稳健性
$PY diag_gain_stuck.py report --split ... --anchors ... --records ... \
  --metrics C0=... --metrics C1=... --metrics C2=... <12 个 --evals> \
  --warmup-policy keep-jobs --out .../d1_tables.md --out-json .../d1_summary.json
```

**D1 机制身份检查**（不是假设）：C0 anchors 与 phaseC4 完全一致 —— 用 C0 策略重放 C2 update-1，
得到 `N_acc_total=47`、`Z_total=80.40114618837833`、`n_rej_offsets=313`、`anchors=24`，
与 `C2.jsonl` 第 1 条**逐位相同**。三臂 teacher logits 的位模式 checksum 在 **1680/1680
anchors 上完全一致**（1680 × 3 次独立进程/GPU 重算）。

### 11.2 D1 三段拆分（teacher-consistent 划分）

三段定义：`acc = k <= n_acc`、`first_rej = k == n_acc + 1`、`suffix = k > n_acc + 1`，
`K = 15`。主划分用**当前 fp32 teacher** 对冻结 proposal 的 leading agreement
（`argmax(p_k) == prop_k`），这直接回应 §3 的旧标签问题：**58/1680 anchors 的接受标签会改变**
（与 §3 独立相符）；三段合计偏移数不变，只有 4 个偏移从 acc 移到 suffix（1129/720/8951 → 1125/720/8955），下面所有 Δ 变化 < 0.02（最大出现在小样本的 first_rej 段），
**即旧轨迹标签问题不驱动 D1 结论**。`own` 划分用各学生自己的 `n_acc_own`；C0 的
`own` 与 `fixed` 完全相等，说明 fresh-cache 前向在这些 anchor 上逐偏移复现了 rollout 的 argmax。

固定划分（pooled，非"先按视频平均再平均"），C2−C0 为逐视频配对 bootstrap（10k，seed 12345）：

| 组 | 段 | n_off | fKL C0→C2 | rKL C0→C2 | rank C0→C2 | top1 C0→C2 | ΔfKL (C2−C0) | ΔrKL | Δrank | Δtop1 |
|---|---|---:|---|---|---|---|---|---|---|---|
| g1 本轮训练20条 | acc | 1057 | 0.6098→0.5692 | 3.1053→2.6289 | 1.00→1.02 | 1.0000→0.9811 | **−0.0480** [−0.0749,−0.0228] | −0.5274 | +0.02 | −0.0181 |
| g1 | first_rej | 480 | 2.2750→2.1715 | 10.2919→8.9801 | 8.34→7.81 | 0.0000→0.1187 | **−0.1035** [−0.1652,−0.0476] | −1.3118 | −0.53 | **+0.1187** |
| g1 | suffix | 5663 | 2.7359→2.6898 | 6.2370→5.8573 | 96.79→96.45 | 0.2259→0.2638 | −0.0447 [−0.0741,−0.0179] | −0.3728 | −0.35 | +0.0372 |
| g2 未曝光训练20条 | acc | 840 | 0.5324→0.5227 | 2.8055→2.6430 | 1.00→1.03 | 1.0000→0.9738 | −0.0093 [−0.0145,−0.0042] | −0.1683 | +0.03 | −0.0288 |
| g2 | first_rej | 480 | 2.3875→2.3864 | 8.9541→8.5969 | 9.71→9.69 | 0.0000→0.0396 | −0.0011 [−0.0323,+0.0337] | −0.3572 | −0.02 | +0.0396 |
| g2 | suffix | 5880 | 2.7826→2.7818 | 6.2990→6.1516 | 141.40→143.36 | 0.2259→0.2422 | −0.0009 [−0.0091,+0.0075] | −0.1464 | +1.91 | +0.0165 |
| g3 validation30条 | acc | 1125 | 0.4058→0.3966 | 2.1340→2.0068 | 1.00→1.02 | 1.0000→0.9822 | −0.0087 [−0.0132,−0.0038] | −0.1241 | +0.02 | −0.0183 |
| g3 | first_rej | 720 | 2.4516→2.4561 | 8.3585→8.0470 | 12.77→12.68 | 0.0000→0.0403 | +0.0045 [−0.0200,+0.0324] | −0.3115 | −0.09 | +0.0403 |
| g3 | suffix | 8955 | 2.8041→2.8033 | 6.2703→6.1127 | 134.45→135.07 | 0.2433→0.2590 | −0.0010 [−0.0090,+0.0083] | −0.1577 | +0.59 | +0.0157 |

（Δ 为逐视频配对 bootstrap 95% CI，10k 次、seed 12345。D1 用 30 条 validation 评分；
τ 用 29 条（去原 warmup 一条），因此 §11.7 的 τ 列与这里的 n 差 1 条视频。
C0 在 `fixed` 下的 acc/first_rej top-1 恰为 1.0000/0.0000，是 teacher-consistent 划分下
C0 的 `fixed` 与 `own` 相等的必然结果，不是独立发现。）

C2 的画像很清楚：**三段 reverse KL 全部下降**（acc/first_rej/suffix，CI 都不跨零）；
按 `own` 划分，三段 forward-KL 基本不动而**首拒 reverse-KL 大幅下降**（−0.42，CI 排除 0）；
按 `fixed` 划分，**first_rej 与 suffix 的 top-1 命中上升**（+4.0…+11.9pp 与 +1.6…+3.7pp），
**只有 acc 段下降**（−1.8…−2.9pp）。注意 `acc` 段的 C0 top-1 = 1.0000 是定义上的必然
（见上一条注：C0 的 `fixed` 与 `own` 相等），因此"acc 段下降"只能与 C1 的同类数字相比，
不能当作独立证据；`own` 划分里 acc/first_rej 的 top-1 为 1.0000/0.0000 同样是定义使然。
C1（continue-SFT）在**所有组、所有段**都更差：如 g3 acc ΔfKL +0.0236 (t+5.13)、
first_rej +0.0745 (t+4.78)、suffix +0.0636 (t+8.20, 30/30 正向)，
即这个配置的 continue-SFT 让草稿头相对 C0 的链式一致性**变差**（它是"更差"的对照臂，不是中性臂）。

### 11.3 首拒位置 / anchor 接受分布

`n_acc_own` = 学生在冻结 proposal 上按 teacher 贪心链的前导一致长度（= 该块在 chain 语义下的接受长度）：

| 组 | C0 | C1 | C2 | C2−C0 |
|---|---:|---:|---:|---:|
| g1 本轮训练20条 | 2.202 | 2.108 | 2.269 | **+0.067** |
| g1 中 C2 实际曝光的10条 | 1.925 | 1.871 | 2.129 | **+0.204** |
| g1 中 C1 实际曝光的10条（C2 未见过） | 2.479 | 2.346 | 2.408 | −0.071 |
| g2 未曝光训练20条 | 1.750 | 1.708 | 1.733 | −0.017 |
| g3 validation30条 | 1.562 | 1.540 | 1.579 | +0.017 |

没有任何 anchor 达到 15/15（`all-accepted` 全为 0.000），分布集中在 0–5，最大值 10，
所以"整块全接受率"这一列在本数据上不可用，**用直方图替代**（`d1_tables.md` 各组给出 0..15 计数）。

### 11.4 D1↔D2 桥接：离线首 token 接受率 vs 实测接受

把 D1 的 offset-1 rank 与真实 decode 的"每轮是否提交 ≥1 个 draft token"对齐：

| 组 | 拓扑 | D1 P(rank==1)@k=1 | D1 P(rank≤8)@k=1 | decode P(提交≥1) | decode 平均提交 |
|---|---|---:|---:|---:|---:|
| g3 validation | chain b1k1 | 0.6944 / 0.6903 / 0.6972 | 0.9597/.9597/.9611 | 0.6865 / 0.6751 / **0.6891** | =P(≥1) |
| g3 validation | tree b63k8 | 同上 | 同上 | 0.9324 / 0.9308 / **0.9304** | 2.7377 / 2.7225 / **2.7403** |
| g2 未曝光20条 | chain | 0.7805/0.7718/**0.7548** | .9771/.9771/**.9775** | 0.7149/0.7103/**0.7148** | =P(≥1) |
| g2 未曝光20条 | tree | 同上 | 同上 | 0.9356/0.9386/**0.9429** | 3.0697/3.0489/**3.1361** |
| g1 训练20条 | chain | 0.8201/0.8148/**0.7731** | .9764/.9788/**.9833** | 0.7036/0.6985/**0.7077** | =P(≥1) |
| g1 训练20条 | tree | 同上 | 同上 | 0.9258/0.9299/**0.9340** | 2.8994/2.8761/**2.9748** |

- **validation 上只有 chain 侧对上了**：D1 的 offset-1 `rank==1` 增量 **+0.0028**
  vs chain 实测 **+0.0026**（绝对水平 0.694 vs 0.687，差 0.008）。**这一条成立**
  （b1k1 的每个 round 只提交 1 个候选，所以"首个 offset 的 rank==1"与"提交 ≥1"是同一个事件）。
- **~~validation 上两种拓扑都对上了~~ —— 更正（2026-10-12，用户核对）**：
  D1 的 `rank≤8` 增量是 **+0.0014**（0.9597 → 0.9611），而 tree 实测 P(提交≥1) 是
  **0.9324 → 0.9304 = −0.0020**。**两者的符号相反、量级也不同，不能声称一致**；
  原文把 D1 的 +0.0014 当成了 tree 实测值，属**引用错误**。
  因此**"离线指标与部署已经校准"目前不成立**——只在 b1k1 单候选探针上成立，
  在 tree（多候选、可提交多个 token）上不成立。这也与下一条 g2 的口径缺口一致。
- **增量碰巧接近也不构成校准**：即便某一组上两个增量数值接近，它们是在**不同的 anchor
  分布**上算出来的（D1 用 C0 策略 anchors，decode 用真实提交轨迹），不能据此断言同分布一致。
- **g2 上出现唯一不一致**：tree 的 P(≥1) 提高 **+0.0072**（全响应；前 24 轮 **+0.0125**, t+2.85），
  而 D1 的 `rank≤8` 增量为 **0.0000**、`rank==1` 为 **−0.0042**。**chain 侧则与 D1 一致**
  （实测 −0.0001 vs D1 −0.0042）。按 §11.2 的公共前缀限制重算（`d1_tables_aligned.md`）
  仍为 0.9705 vs 0.9705，**说明这不是旧轨迹标签造成的**，而是 tree 的多 token 提交
  与"单偏移 top-k 覆盖"之间在本数据上确有约 1pp 量级的口径缺口。**不把它解释成任一方的错误。**

### 11.5 D2：chain vs tree（同一批视频，全部带 `--ar` 门禁）

| 拓扑 | budget/branch | 数据 | C0 | C1 | C2 | C2−C0（95% CI, t） | C1−C0 | AR 门禁 |
|---|---|---|---:|---:|---:|---|---|---|
| tree b63k8 | 63/8 | validation n=29 | 3.7377 | 3.7225 | 3.7403 | +0.0027 [−0.0205,+0.0273] (+0.21) | −0.0152 [−0.0458,+0.0131] | 各 30/30 |
| chain b1k1 | 1/1 | validation n=29 | 1.6865 | 1.6751 | 1.6891 | +0.0026 [−0.0021,+0.0072] (+1.08) | **−0.0114 [−0.0196,−0.0038]** | 各 30/30 |
| tree b63k8 | 63/8 | g1+g2 n=39（作业 warmup 去掉） | 3.9956 | 3.9721 | 4.0674 | +0.0718 | −0.0235 | 各 40/40 |
| chain b1k1 | 1/1 | g1+g2 n=39 | 1.7099 | 1.7049 | 1.7119 | +0.0020 | −0.0050 | 各 40/40 |

**"component-wise chain" 身份声明**：`--budget 1 --branch 1` 每轮只允许 1 个 draft 候选、
2 个验证位置，但**仍然跑完整的 block-16 head 前向**，因此它是"每轮单候选"探针，
**不是**训练里用的 block-16 chain rollout（每轮最多 15 个 offsets），也**不是**完整链消融。
§5 的同一保留意见适用。本节新跑 9 次（chain×2 行集 + tree×g1g2，各 3 臂），另 3 次 tree/validation 复用 §4 的冻结产物；12 次的配置逐项一致（fp32、`--ar`、`--warmup 1`），**全部通过 AR 逐 token 门禁**（30/30 或 40/40）。

结论：**两种拓扑给出同一排序**（C2 名义最好、C1 最差），C2−C0 在两个拓扑下几乎相同
（+0.0027 vs +0.0026）。唯一差别是 b1k1 的方差小得多，使 C1 的退化在 chain 下**可判**
（CI 不含 0），而在 tree 下不可判。**不支持"chain 改善、tree 不改善"的拓扑不匹配分支。**

### 11.6 D3：曝光组与真实 τ

| 组 | n | 拓扑 | C0 | C1 | C2 | Δτ C2−C0（95% CI, t） |
|---|---:|---|---:|---:|---:|---|
| g1 全部 phaseC4 训练20条 | 20 | tree | 3.8994 | 3.8761 | 3.9748 | **+0.0753 [+0.0323,+0.1226] (+3.16)** |
| g1 中 C1 曝光10条（C2 未见过） | 10 | tree | 3.9551 | 3.9419 | 4.0108 | **+0.0557 [+0.0091,+0.0962] (+2.36)** |
| g1 中 C2 曝光10条 | 10 | tree | 3.8437 | 3.8104 | 3.9387 | **+0.0950 [+0.0213,+0.1785] (+2.27)** |
| g2 未曝光训练20条 | 20 | tree | 4.0697 | 4.0489 | 4.1361 | **+0.0665 [+0.0275,+0.1109] (+3.06)** |
| g3 validation | 29 | tree | 3.7377 | 3.7225 | 3.7403 | +0.0027 [−0.0205,+0.0273] (+0.21) |
| g1 全部 | 20 | chain | 1.7036 | 1.6985 | 1.7077 | +0.0041 [−0.0021,+0.0108] (+1.22) |
| g1 中 C1 曝光10条 | 10 | chain | 1.7212 | 1.7162 | 1.7234 | +0.0021 [−0.0064,+0.0120] (+0.44) |
| g1 中 C2 曝光10条 | 10 | chain | 1.6861 | 1.6808 | 1.6921 | +0.0060 [−0.0026,+0.0155] (+1.26) |
| g2 未曝光训练20条 | 20 | chain | 1.7149 | 1.7103 | 1.7148 | **−0.0001 [−0.0061,+0.0067] (−0.03)** |
| g3 validation | 29 | chain | 1.6865 | 1.6751 | 1.6891 | +0.0026 [−0.0021,+0.0072] (+1.08) |

（组身份固定，保留 g1/g2 作业自己的 warmup 行；validation 仍只去原来那一条。
本表与 §10 独立复算一致到小数第 4 位。）

两个必须一起读的事实：

1. **tree 下训练池收益是真的**（g1/g2 三组 CI 都排除 0，+0.056…+0.095），而且**不集中在
   曝光视频**：C2 曝光10条 +0.0950、C1 曝光10条（C2 从未见过）+0.0557、未曝光20条 +0.0665；
   差异中差异按 §10 为 +0.0286、CI 跨零。**所以不能说"只记住了它训练的那10条"。**
2. **同一批视频换到 b1k1 单候选探针，收益全部消失**（g1 +0.0041、g2 −0.0001、均跨零），
   而 validation 在两种拓扑下都 ~+0.003。→ 训练池上的收益是**多 token / 深偏移提交**的收益，
   不是首 token 接受率收益。

前 24 轮窗口（D1 的 horizon）检查（`--window 24`）：

| 组 | 拓扑 | Δτ(前24轮) C2−C0 | Δτ(全响应) C2−C0 |
|---|---|---:|---:|
| g1 训练20条 | tree | +0.0521 [−0.0167,+0.1271] | +0.0753 [+0.0323,+0.1226] |
| g2 未曝光20条 | tree | +0.0729 [+0.0146,+0.1355] | +0.0665 [+0.0275,+0.1109] |
| g3 validation | tree | −0.0115 [−0.0575,+0.0330] | +0.0027 [−0.0205,+0.0273] |
| g3 validation | chain | +0.0086 [−0.0057,+0.0230] | +0.0026 [−0.0021,+0.0072] |

g2 的收益**在 D1 的前 24 轮 horizon 之内就已经存在**，所以"horizon 太短"不能解释
§11.4 的 g2 缺口；能解释它的是"首拓接受"与"多 token 提交"的差别（§11.6 第 2 点）。

### 11.7 配对表：loss 变化 → rank 变化 → 首拒变化 → 真实 τ 变化（validation, g3）

| 对比 | 段 | ΔfKL | ΔrKL | Δrank | Δtop1 | Δ mean n_acc_own | Δτ tree | Δτ chain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C1−C0 | acc | +0.0236 | +0.1197 | +0.05 | −0.0360 | −0.022 | −0.0152 | −0.0114 |
| C1−C0 | first_rej | +0.0745 | +0.0349 | +0.33 | +0.0431 | −0.022 | −0.0152 | −0.0114 |
| C1−C0 | suffix | +0.0636 | +0.0541 | −1.94 | −0.0127 | −0.022 | −0.0152 | −0.0114 |
| C2−C0 | acc | −0.0087 | −0.1241 | +0.02 | −0.0183 | +0.017 | +0.0027 | +0.0026 |
| C2−C0 | first_rej | +0.0045 | −0.3115 | −0.09 | +0.0403 | +0.017 | +0.0027 | +0.0026 |
| C2−C0 | suffix | −0.0010 | −0.1577 | +0.59 | +0.0157 | +0.017 | +0.0027 | +0.0026 |

读法：C2 的 reverse-KL 在三段都显著下降、first_rej 与 suffix 的 top-1 命中上升
（acc 段下降但该段对 C0 是定义性对照，见 §11.2）、`n_acc_own` 上升 +0.017 个 offset（+1.1%），
而 τ 上升 +0.0027；**四者同向、量级自洽（约 1% 的接受长度）**。
C1 则在三段都变差、`n_acc_own` 下降、τ 下降。**这条链在 validation 上是自洽的**；
它的问题是**绝对量级只有 ~0.003 τ**，不是方向错了。

### 11.8 分支判定（按证据，不预先认定）

| 候选分支 | 判定 | 依据 |
|---|---|---|
| 训练样本与接受指标都几乎不变 → 本轮学习量太小 | **部分成立（仅 validation）** | validation 三段 ΔfKL ≤0.009、Δn_acc_own +0.017、Δτ +0.0027；但训练池组不是"几乎不变"（+0.06…+0.10），所以不能只归因于"更新太小" |
| KL 明显下降但首拒 rank/接受不改善 → 目标/接受错配 | **不支持** | 在 KL 明显下降的地方（g1 C2 曝光10条：first_rej ΔfKL −0.198, t−4.50），first_rej top-1 同时 **+18.3pp**、`n_acc_own` **+0.204** —— 目标与首拒指标同向而非脱节 |
| 训练样本改善、validation 不改善 → 数据覆盖/泛化，不是 KL 调参 | **支持（主结论）** | tree 下 g1/g2（+0.056…+0.095，CI 排除 0）vs validation（+0.0027，CI 跨零）；且收益不限于曝光视频，说明是本轮更新在**训练池分布**上有效、**不迁移到 heldout 视频** |
| chain 改善、tree 不改善 → 训练/部署拓扑错配 | **不支持（且方向相反）** | validation 两拓扑都是 ~+0.003；训练池上 **tree 明显改善而 b1k1 chain 不动**。方向是"tree 有效、单候选无效"，不是"chain 有效、tree 无效" |
| 所有指标都只有微不足道的变化 → 按"该短配置没有产生有意义的收益"收口 | **对 validation 成立，但不是全貌** | validation 口径确实可以这样收口；但训练池上存在统计可辨的 +0.06…+0.10，因此不能宣称"机制已被证明无效" |

**一句话：证据支持"训练池上有效、heldout validation 上不迁移"（数据覆盖/泛化），
不支持"目标与接受错配"，也不支持"chain/tree 拓扑错配"。** 训练池收益是**多 token 提交**型，
不是首 token 接受型。

**关于 suffix（明确按要求不预先否定）**：validation 上 suffix 的 reverse-KL 下降
（−0.1577，CI 排除 0）**没有**带来 τ 提升；同一段 fKL 基本不动（−0.0010），top-1 只有小幅上升（+1.6pp），远小于它的 reverse-KL 变化。
因此本轮**能说的**是"这个 10-update 短配置里，suffix 的 KL 改善未转化为可测的接受收益"，
**不能**说"suffix 梯度无用"——它在更长 rollout / 未来轮次里的作用本轮没有测到。

### 11.9 限制与未能验证项

1. **reference**：D1 是 A reference（旧轨迹）。§11.2 已量化它对结论的影响（58/1680 标签、
   三段集合几乎不动、各 Δ 变化 ≤0.02），§11.4 的 g2 缺口按公共前缀限制后**依然存在**，
   所以缺口的来源仍未确定为"参考不一致"还是"top-k 覆盖 vs 树提交"的口径差异。B reference
   仍未做（§4/§8）。
2. **行集不可交换**：g1/g2 的 C0 基线 τ≈3.90–4.07，validation≈3.74；两组不是同一分布，
   "training 改善 / validation 不改善"的证据来自 split 身份，**不是**匹配难度后的比较。
3. **g2 不是 held-out**：它与 g1 一样属于 A4（C0 的生产者）训练过的 439 条，也与本轮
   训练池同源；只有 g3（fresh30）两级都没见过。因此 g2 的读数只能叫"phaseC4 未采样"。
4. **D1 horizon**：anchors 是每条视频 C0 rollout 的前 24 个 anchor block（按各族 `n_acc_own` 均值折算≈前 60–80 个生成 token），
   与 phaseC4 监督的是同一批块；它不覆盖整条回答。
5. **b1k1 不是完整 chain**：见 §11.5 身份声明与 §5。
6. **n=10 的子组**：曝光子组只有 10 条，CI 很宽（如 C2 曝光10条 [+0.021,+0.179]）；
   §10 已给出差中差不稳健的正确边界，本节的子组数字应作为描述而非确认。
7. **计时未作结论**：本轮不报加速；tree/chain 的 SPEC/AR 秒数为共享机器上的 decode-only 合计。
8. **`all-accepted anchors` 全零**：本数据的 anchor 接受长度最大为 10，故该列不可用。
9. 未评测任何新测试集候选；未加 seed、未加训练步数、未改共享依赖。


---

## 12. CPU 口径收口（2026-10-12，用户第二轮核对后）

本节只做 CPU 口径核对，不训练、不改参数、不动新测试集。**保留 §1–§11 原始证据**，本节只做修正与补充。

### 12.1 g1 的 19/20：只是 warmup 排除，不是连接错误

`v_-cAcA8dO7kA` 属于 g1，且正是 g1g2 评测的**第 0 行（warmup=True）**。因此：

| 组 | 成员数 | 进入指标的条数 | warmup 被排除 |
|---|---:|---:|---|
| g1（phaseC4 训练过） | 20 | **19** | `v_-cAcA8dO7kA` |
| g2（训练池未曝光） | 20 | **20** | 无 |
| 合计 | 40 | **39** | 1 |

§11 表格里写的 "g1 trained(20)" 指的是**成员数**，而所有指标实际用 **19** 条。
**我此前把 19 vs 20 说成"视频名连接错误"是错的**，不存在连接错误；两者是同一件事的两种计数。

### 12.2 §11.4 的引用错误（已在正文更正）

原文写 "D1 的 `rank≤8` 增量 **+0.0014** vs tree 实测 P(≥1) **+0.0014**"。实际：

| 量 | C0 | C2 | 增量 |
|---|---:|---:|---:|
| D1 `P(rank≤8)@k=1`（validation） | 0.9597 | 0.9611 | **+0.0014** |
| tree 实测 `P(提交≥1)`（validation） | 0.9324 | 0.9304 | **−0.0020** |

**符号相反、量级不同，原文把 D1 的值当成了 tree 实测值。** 因此"离线与部署已经校准"**只在 b1k1 单候选探针上成立**（`rank==1` +0.0028 vs chain 实测 +0.0026），**在 tree 上不成立**。这与 §11.4 记录 g2 上约 1pp 的口径缺口方向一致。

### 12.3 训练池与 validation 的收益差异区间（用户要求直接计算）

用户指出"一组显著、另一组不显著 ≠ 两组差异显著"。直接算差中差（bootstrap 10k，按视频重采样，两组独立）：

| 组 | n | C2−C0 mean Δτ | 95% CI |
|---|---:|---:|---|
| 训练池（g1+g2） | 39 | **+0.0718** | [+0.0411, +0.1040] |
| validation | 29 | +0.0027 | [−0.0203, +0.0274] |
| **差中差** | — | **+0.0691** | **[+0.0301, +0.1089]** |

差中差 CI **不含 0**（99.97% 的 bootstrap 抽样 > 0；Welch t = +3.35）。**所以组间差异本身是显著的**，
比"一组显著另一组不显著"更强。

**但仍然不能据此归因为"数据覆盖"**：两组难度不同（C0 基线 τ 3.996 vs 3.738），
且 g2 属于历史训练池（A4 在全部 439 条上训练过），所以两组的**历史曝光也不同**。
差中差显著只说明"两组收益不同"，不说明**为什么**不同。

### 12.4 本节未解决

- §11.4 的 g2 口径缺口（tree 首 token 接受率上升，而 D1 的 top-8 覆盖恰好为 0）**仍未解释**。
- 重算 teacher 标签修正了"旧标签错误"，但**没有让旧 prefix 变成当前部署实际访问的 prefix**
  （另见 §3：旧 trajectory 与当前 fp32 部署 AR 在 69/70 条上分叉）。
- `b1k1` 是单候选探针，**不能排除完整 block16 chain 与 tree 的差异**。
- 因此 B-reference 对齐实验仍是唯一能推进的一步（见 §13）。

---

## 13. B-reference 对齐实验：真实部署 anchor 上的三臂同上下文重放（2026-10-12，独立子任务追加）

**本节由与 §1–§12 及 phaseC6 并发的独立子任务追加，不修改上面任何内容。** 全程只读：
没有训练、没有 optimizer/backward、没有 `save_pretrained`、没有评测任何新测试集候选、
没有新增训练样本。脚本 `repro/diag_bref_align.py`（新增）；产物
`/mnt/local/mcy/GLANCE/results/phaseC5_brief/`。

本节针对 §3/§8/§12.4 留下的三条阻塞：

| 阻塞 | 本节是否解决 | 结论 |
|---|---|---|
| §3：旧轨迹 prefix 不是部署实际访问的 prefix | **是** | anchors 直接取当前 fp32 C0 部署解码自己的轮次，不再用磁盘旧 token |
| §11.4/§12.2：tree 首 token 接受率上升而离线 `rank≤8` 持平 | **是（在固定对齐上下文上）** | 同一 anchor 同一边际分布下，两者是**同一个事件**；历史缺口是"两个不同 anchor 分布相比"造成的 |
| §5：`b1k1` 不是完整 block16 chain | **是** | 用同一条 `glance.generate` 路径的 `--budget 15 --branch 1` 做真 block16 chain，并在同一 anchor 上做配对 chain 重放 |

与并发 phaseC6 的关系：phaseC6（`docs/reports/2026-10-12-current-ar-fixed-anchor.md`）用可移除 hook 捕获真实
decode，锚点取 C0 轮次序列的 8 个**等距**索引；本节用独立 instrumentation 复制 round loop（由门禁验证），
锚点取**前 8 轮**，预注册 6 条 validation，并额外给出 (i) C0 round 复现门禁、(ii) 真实 block16 chain 的
配对与整轨迹对照、(iii) **离线首拒位置 vs 实际首拒位置/提交长度**。两条独立实现有 1 条视频重叠
（`v_0x6ZV9Y8g_c`），第 13.7 节给出逐 offset 交叉验证。

### 13.1 预注册（先写盘，后跑分）

`results/phaseC5_brief/prereg.json`，创建于任何模型运行之前。只用 ID 与输入 token 长度，不看任何收益。

- 候选：`video_fresh30.parquet` 去掉第 0 行（eval 约定 `--warmup 1`，该行是 `v_-nl4G-00PtA`）后的 **29** 条。
- 规则：按 `input_ids_len` 排序切 3 个等量层，每层按 `sha256("phaseC5-bref-align-2026-10-12:<video_name>")`
  升序取 2 条；gate = 最短层的第一个 + 最长层的第一个。
- anchors：每条视频真实 fp32 C0 tree 解码（b63k8）的**前 8 轮**，不做任何按分数选点。
- gate 判据（预注册）：C0 重放必须在 gate 的**每一个** anchor 上复现真实 C0 的提交长度**与**提交 token；
  否则**停止**并把不一致本身当作发现。

| 层 | 长度范围 | 选中 |
|---|---|---|
| 0 | 766 | `v_4oi6dXfpJS4`（gate）、`v_0k0xNs8Zjb4` |
| 1 | 766–958 | `v_1dDAcUliXrQ`、`v_2nDbYhgdw2A` |
| 2 | 958 | `v_-MbZ-W0AbN0`（gate）、`v_0x6ZV9Y8g_c` |

`v_0x6ZV9Y8g_c` 与 phaseC6 的 6 条独立抽样重叠，构成一次跨实现对照。

### 13.2 方法：真实部署状态 + 三臂重放

1. 未修改的 `glance.generate`（fp32、C0、b63k8、greedy、max_new 1024）跑一遍，记录每轮 `(pos, accepted,
   path_tokens)` —— 这是 reference。
2. 同一个 round loop 的 instrumented 副本（新增脚本内，`instrumented_decode`）再跑一遍；它在每轮
   **verify pass 之前**冻结状态：覆盖 `[0, start)` 的 target KV、覆盖 `[0, start)` 的 drafter 上下文特征、
   root token、block mask。副本与原 `generate` 的 token 流与逐轮 accepted **必须逐位相等**，否则失败。
3. 每个 anchor 上把 C0/C1/C2 三个 head 在**同一份冻结状态**上各重放一次：一次 draft pass（top-8 候选）、
   一次前缀闭合树、一次 fp32 target verify、同一条 walk。三臂之间只有 head 不同。
4. 同一 draft logits 再单独构建 head 的 top-1 脊（budget 15 / branch 1）并用同一次 verify 语义提交 ——
   这是同一上下文上的**真 block16 chain**（配对版）；另用未修改的 `glance.generate` 跑整轨迹
   `--budget 15 --branch 1` 与 `--budget 63 --branch 8`（各 3 臂 × 6 视频，均带 AR 逐 token 门禁）。

**实现缺陷（已修复，失败产物隔离）**：第一版重放把 `VisionPositions` 当作普通对象传入，没有
`bind()`，于是 M-RoPE `delta` 默认为 0；实测 delta 为 **−576 / −672 / −768**，重放的 target verify
位置全错，C0 重放只有 4/16 commit 长度一致。修复后重跑。失败版原始文件保留在
`results/phaseC5_brief/failed/`（`anchors_gate_delta0bug.jsonl`、`run_gate_delta0bug.log`），未混入正式指标。
`verify_topology` 现已加显式守卫：未 bind 的 adapter 直接报错。

### 13.3 门禁一：instrumented decode == 未修改的 `generate`

6/6 视频 token 流逐位相同、逐轮 accepted 相同；6/6 AR 逐 token 无损（`--ar` 等价门禁）。
C0 的 tree τ 在整轨迹重跑中与 anchor 阶段完全一致（如 `v_-MbZ-W0AbN0` 3.6618、
`v_2nDbYhgdw2A` 4.1714），说明解码本身可复现。

### 13.4 门禁二：C0 重放复现真实 C0 提交

| 量 | gate 2 条视频 16 anchors | 全 6 条视频 48 anchors |
|---|---:|---:|
| commit 长度完全一致 | **16/16** | **48/48** |
| commit token 完全一致 | **16/16** | **48/48** |
| bonus token 一致 | **16/16** | **48/48** |
| draft top-8 集合一致 | **16/16** | **48/48** |
| 候选树 nodes 完全一致 | **16/16** | **48/48** |
| draft logits **逐位**一致 | 2/16 | 6/48 |

**gate 通过 → 继续其余 4 条视频。** 唯一未逐位一致的是 draft logits 的 fp32 位模式：只有每条第 0 轮
逐位相同，机制清楚 —— 第 0 轮重放的一次性全前缀 draft pass 与部署的分块 cached pass 行数/形状恰好
相同，之后 `start+16` 与 `(start−prev_start)+16` 的 `q_len` 不同，SDPA 归约分块不同。它**没有**
改变任何一个 anchor 的 top-8 或树结构（48/48 一致），因此没有改变提交。这一条要按"减少顺序不同、
分块不同"如实记录，不能当作逐位等价。

### 13.5 离线指标与实际首拒在同一 anchor 上是否同一对象

`rank==1 ⟺ 该 offset 的 head argmax 就是 target 的 AR token`：**506/506** 一致；
`rank==r≤8 ⟺ 该 token 恰好落在已保存 top-8 的第 r−1 位`：**1343/1343** 一致。
（报告里的 `offline_alignment_gate`，pass=True。）这是本节所有离线/实际比较成立的前提。

### 13.6 三臂同上下文结果

全部 48 anchors：每个 anchor 上 C0/C1/C2 的 context hash 完全相同（`ctx_hash_identical_across_arms=True`）。

（"首拒"列只对有检出首拒的 anchor 求均值；`rank1 首拒` 与 `chain commit` 逐 anchor 相等所以均值必然相同。）

| arm | n | 离线 rank1 首拒 | 离线 top8 首拒 | tree 首拒 = tree commit | chain commit | 离线 rank1 一致长度 | 离线 top8 一致长度 | rank1 首拒 == tree 首拒 | chain commit == tree commit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 48 | 2.8333 | 6.2766 | **4.3125** | **2.8333** | 1.8333 | 5.4792 | 9/48 | 9/48 |
| C1 | 48 | 2.7500 | 5.8936 | 4.2083 | 2.7500 | 1.7500 | 5.1042 | 11/48 | 11/48 |
| C2 | 48 | 2.9167 | 6.0417 | 4.2500 | 2.9167 | 1.9167 | 5.0417 | 9/48 | 9/48 |

注意：**离线 rank1 首拒只在 9–11/48 个 anchor 上等于 tree 的实际首拒** —— 也就是说离线链式预测
对"这一轮树实际提交多少"只有约 20% 的逐 anchor 命中率，即使它的均值与 chain commit 完全对齐。

**chain commit == 离线 rank1 首拒：三臂各 48/48 完全相等。** 即真实 block16 chain 的提交长度被离线
`rank==1` 指标**精确预测**。§11.5 用 b1k1 得到的"离线 rank==1 与部署一致"在完整链上同样成立。

**首拒原因分类**（在第一个被拒 offset 上，正确 token 是否在 head 的 top-8 / 是否在预算树里）：

| arm | 不在 top-8（候选覆盖） | 在 top-8 但不在树（树选择/预算） | 在树里（矛盾） | 无（整块全接受） |
|---|---:|---:|---:|---:|
| C0 | 19 | **29** | 0 | 0 |
| C1 | 20 | **28** | 0 | 0 |
| C2 | 19 | **29** | 0 | 0 |

**离线 top-8 覆盖率与 tree commit 的差**（`top8 一致长度 + 1 − commit`，事后 oracle 上界）：
C0 +2.17（29/48 为正，最大 13）、C1 +1.90（28/48，最大 13）、C2 +1.79（29/48，最大 9）。

两个必须一起读的事实：

1. **离线 top-8 覆盖解释不了 tree 的提交长度。** 60% 的首拒（29/48）发生在正确 token
   **就在 head 的 top-8 里**、但没有进入 63 节点预算树。把 AR token 沿实际树节点走一遍得到的
   "路径 oracle"提交长度与 tree commit **144/144 完全相同**，所以 commit 完全由"预算树里有没有
   这条路径"决定。
2. **树的浅层宽度很小。** 144 个 arm-anchor 中，树在第一层只有 **1** 个候选的有 42 个、只有 2 个的
   14 个；offset-1 的正确 token 出现在第一层的有 135/144。树的最大深度分布为 3–11，**从未达到 12–15**。
   两个可复现实例：
   - `v_4oi6dXfpJS4` round 2：head 的 offset-1 top-8 = `[389,264,279,…]`，target 的 offset-1 token
     是 264（rank 2），但 63 节点树里**第一层只有 389 一个节点**，264 根本没有进树 → commit=1。
   - `v_4oi6dXfpJS4` round 5：**全部 15 个 offset 的正确 token 都在 head 的 top-8 内**，树却只提交 3；
     树深度分布 {1:2, 2:4, 3:13, 4:16, 5:15, 6:13}，即预算全花在少数几条深路径上。
   注意：这是**事后 oracle** 与**事前最优分配**的差别，不等于生产树"有 bug"；它说明的是
   best-first 累计概率分配不保证浅层/宽覆盖，与 phaseC6 独立发现的 `tree_gap`（44 轮合计 40 个 token）同源。

### 13.7 §11.4 缺口在真实对齐上下文上的答案

`提交 ≥1 个 draft token` ≡ `tree commit ≥ 2`；离线同一事件 ≡ `offset-1 的 rank ≤ 8`。在**同一 anchor、
同一份边际分布**上：

| arm | 集合 | n | P(tree 提交≥1) | P(离线 top8@1) | 不一致 |
|---|---|---:|---:|---:|---:|
| C0/C1/C2 各自 | 全部 48 | 48 | 45/48 = 0.9375 | 46/48 = 0.9583 | 1 |
| C0/C1/C2 各自 | 共同前缀 12 | 12 | 12/12 = 1.0000 | 12/12 = 1.0000 | 0 |

唯一不一致是 `v_4oi6dXfpJS4` round 2（§13.6 的树浅层宽度实例）：正确 token 在 top-8 里但掉出了预算树。
**所以 §11.4 观察到的"tree 首 token 接受率与离线 rank≤8 符号相反"在本节的固定对齐上下文里不出现** ——
两个量是同一个事件，唯一致因是树预算。§11.4 自己也写明两条数来自**不同的 anchor 分布**（离线用 C0 策略
anchors、decode 用真实提交轨迹），本节的测量与该保留意见一致：**缺口来自口径，不是来自"离线指标与部署
本质不校准"。** 但这只推翻"在 tree 上也不校准"这一读法的普遍性，不推翻它的历史数值。

**跨实现交叉验证**：重叠视频 `v_0x6ZV9Y8g_c` round 0（C0），本节独立捕获的 15 个 offset rank
`[1,2,1,3,58,20,58,5,162,1,182,137,2689,718,3]` 与 phaseC6 的 `token_metrics` **15/15 逐位相同**；
本节 tree commit=5 与 phaseC6 的 `tree_covered`（offset 1–4）/`next_uncovered`（offset 5）一致。

### 13.8 真实 block16 chain vs tree

**配对同上下文（前 8 轮，全部 48 anchors）**：三臂 chain commit 均值 2.75–2.92，tree commit 均值
4.21–4.31 → 同一上下文上 tree 每轮多提交约 **1.4–1.5** 个 token。

**整轨迹（未修改 `glance.generate`，6 视频，各臂两种拓扑都 6/6 AR 无损）**：

| 拓扑 | C0 τ | C1 τ | C2 τ |
|---|---:|---:|---:|
| tree b63k8 | 3.7920 | 3.7991 | 3.7868 |
| chain b15k1 | 2.5255 | 2.5214 | 2.5432 |
| Δτ(chain − tree) | **−1.2665 [−1.3918,−1.1357]** | **−1.2777 [−1.3631,−1.1877]** | **−1.2436 [−1.3474,−1.1399]** |

**结论：真实 block16 chain 与 tree 差别很大**（约 −1.25 τ，CI 远不含 0）。tree 的多 token 提交增益
是真实的，不是 b1k1 探针造成的假象。同时注意：这 6 条视频上 tree 的 τ 名义排序变成
**C1 > C0 > C2**（phaseC4 的 29 条 validation 是 C2 > C0 > C1），chain 上 C2 名义最高 ——
两种排序的差距都 ≤0.012 τ，**n=6 完全无法分辨 phaseC4 那 0.003 τ 量级的臂差**，不能用它给臂重新排序。

### 13.9 common-prefix 记账

- **48 个 anchor 中只有 12 个**处在三臂**真正共同的前缀**上（该 anchor 之前所有轮三臂提交 token 与 C0 相同）；
  **36 个已分叉**。逐视频首次 commit 差异轮：

| 视频 | C1 首次 commit 差异轮 | C2 首次 commit 差异轮 | 与 C0 提交不同的轮数 C1 / C2 |
|---|---:|---:|---|
| v_-MbZ-W0AbN0 | 0 | 0 | 3 / 2 |
| v_0k0xNs8Zjb4 | 0 | 未在前 8 轮 | 5 / 0 |
| v_0x6ZV9Y8g_c | 6 | 0 | 1 / 2 |
| v_1dDAcUliXrQ | 2 | 0 | 2 / 3 |
| v_2nDbYhgdw2A | 6 | 未在前 8 轮 | 1 / 0 |
| v_4oi6dXfpJS4 | 0 | 未在前 8 轮 | 3 / 0 |

**三臂各自在线解码在绝大多数视频的第 0–1 轮就分开**，所以后续 anchor 虽然仍在我们冻结的**同一份
C0 上下文**上评分（这正是本节的设计），但对该臂**自己的**轨迹而言它根本不会走到这个前缀。
这类行标为 diverged，**不与共同前缀行合并**；上表与 §13.6 的"共同前缀 12"是同一份记账。

**共同前缀 12 个 anchor 的逐 anchor 表**（每行一个 anchor，三臂并列）：

| video | round | start | arm | 离线 rank1 首拒 | 离线 top8 首拒 | tree 首拒 | tree commit | chain commit | 首拒原因 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| v_-MbZ-W0AbN0 | 0 | 958 | C0 | 2 | 9 | 7 | 7 | 2 | in_topk_not_in_tree |
| v_-MbZ-W0AbN0 | 0 | 958 | C1 | 5 | 9 | 6 | 6 | 5 | in_topk_not_in_tree |
| v_-MbZ-W0AbN0 | 0 | 958 | C2 | 6 | 9 | 8 | 8 | 6 | in_topk_not_in_tree |
| v_0k0xNs8Zjb4 | 0 | 766 | C0 | 5 | 12 | 6 | 6 | 5 | in_topk_not_in_tree |
| v_0k0xNs8Zjb4 | 0 | 766 | C1 | 5 | 7 | 7 | 7 | 5 | not_in_topk |
| v_0k0xNs8Zjb4 | 0 | 766 | C2 | 5 | 12 | 6 | 6 | 5 | in_topk_not_in_tree |
| v_0x6ZV9Y8g_c | 0 | 958 | C0 | 2 | 5 | 5 | 5 | 2 | not_in_topk |
| v_0x6ZV9Y8g_c | 0 | 958 | C1 | 2 | 5 | 5 | 5 | 2 | not_in_topk |
| v_0x6ZV9Y8g_c | 0 | 958 | C2 | 1 | 4 | 2 | 2 | 1 | in_topk_not_in_tree |
| v_1dDAcUliXrQ | 0 | 862 | C0 | 3 | 5 | 4 | 4 | 3 | in_topk_not_in_tree |
| v_1dDAcUliXrQ | 0 | 862 | C1 | 3 | 5 | 4 | 4 | 3 | in_topk_not_in_tree |
| v_1dDAcUliXrQ | 0 | 862 | C2 | 4 | 5 | 5 | 5 | 4 | not_in_topk |
| v_2nDbYhgdw2A | 0 | 958 | C0 | 5 | 11 | 8 | 8 | 5 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 0 | 958 | C1 | 5 | 10 | 8 | 8 | 5 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 0 | 958 | C2 | 5 | 11 | 8 | 8 | 5 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 1 | 966 | C0 | 2 | 4 | 4 | 4 | 2 | not_in_topk |
| v_2nDbYhgdw2A | 1 | 966 | C1 | 2 | 4 | 4 | 4 | 2 | not_in_topk |
| v_2nDbYhgdw2A | 1 | 966 | C2 | 2 | 4 | 4 | 4 | 2 | not_in_topk |
| v_2nDbYhgdw2A | 2 | 970 | C0 | 2 | 2 | 2 | 2 | 2 | not_in_topk |
| v_2nDbYhgdw2A | 2 | 970 | C1 | 2 | 2 | 2 | 2 | 2 | not_in_topk |
| v_2nDbYhgdw2A | 2 | 970 | C2 | 2 | 2 | 2 | 2 | 2 | not_in_topk |
| v_2nDbYhgdw2A | 3 | 972 | C0 | 4 | 8 | 5 | 5 | 4 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 3 | 972 | C1 | 3 | 8 | 5 | 5 | 3 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 3 | 972 | C2 | 4 | 5 | 5 | 5 | 4 | not_in_topk |
| v_2nDbYhgdw2A | 4 | 977 | C0 | 3 | 5 | 4 | 4 | 3 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 4 | 977 | C1 | 3 | 5 | 4 | 4 | 3 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 4 | 977 | C2 | 3 | 5 | 4 | 4 | 3 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 5 | 981 | C0 | 2 | 2 | 2 | 2 | 2 | not_in_topk |
| v_2nDbYhgdw2A | 5 | 981 | C1 | 2 | 2 | 2 | 2 | 2 | not_in_topk |
| v_2nDbYhgdw2A | 5 | 981 | C2 | 2 | 2 | 2 | 2 | 2 | not_in_topk |
| v_2nDbYhgdw2A | 6 | 983 | C0 | 4 | 11 | 5 | 5 | 4 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 6 | 983 | C1 | 5 | 11 | 7 | 7 | 5 | in_topk_not_in_tree |
| v_2nDbYhgdw2A | 6 | 983 | C2 | 4 | 11 | 5 | 5 | 4 | in_topk_not_in_tree |
| v_4oi6dXfpJS4 | 0 | 766 | C0 | 2 | 10 | 4 | 4 | 2 | in_topk_not_in_tree |
| v_4oi6dXfpJS4 | 0 | 766 | C1 | 2 | 10 | 3 | 3 | 2 | in_topk_not_in_tree |
| v_4oi6dXfpJS4 | 0 | 766 | C2 | 3 | 10 | 4 | 4 | 3 | in_topk_not_in_tree |

共同前缀上的配对差（视频 bootstrap 10k，seed 20261012，单位=视频）：

| 对比 | tree commit | chain commit | 离线 rank1 一致长度 | 离线 top8 一致长度 |
|---|---:|---:|---:|---:|
| C1−C0 | +0.0833 [−0.6667,+0.3333] | +0.2500 [+0.0000,+1.5000] | +0.2500 [+0.0000,+1.5000] | −0.5000 [−2.5000,+0.0000] |
| C2−C0 | **−0.0833** [−1.1667,+0.6667] | +0.4167 [−0.1250,+2.0000] | +0.4167 [−0.1250,+2.0000] | **−0.3333** [−0.5000,+0.0000] |

注意方向：在共同前缀上 C2 的**离线 top-8 覆盖下降**、而**离线 rank1/chain 一致长度上升**，
tree commit 基本不动。这与 §11.4 的缺口是同一枚硬币的两面（top-8 覆盖与多 token 提交不同步），
但 n 只有 12 个 anchor / 6 条视频，只能作描述，不能作确证。

### 13.10 分支判定（按证据，不强行选一支）

| 候选分支 | 判定 | 依据 |
|---|---|---|
| 对齐 reference 会改变关键结论 | **不支持（强形式）** | 对齐后 phaseC4 的臂差仍是噪声量级：共同前缀 C2−C0 tree commit −0.0833（CI 跨零）、chain +0.4167（CI 跨零）；6 条视频的整轨迹 tree τ 名义排序甚至变成 C1>C0>C2。历史 τ 数字无需改写。**部分成立（弱形式）**：§11.4/§12.2 的"离线未与部署校准"被收窄为"跨 anchor 分布口径差异"，在固定对齐上下文上两者是同一事件 |
| 结论稳定，局部候选变化能解释实际提交 | **不支持** | 离线 rank1 只解释 **chain**（48/48 精确）；tree 的多 token 提交**不能**由候选覆盖解释：29/48 首拒时正确 token 已在 top-8 内却被预算树排除，top-8 上界与 tree commit 平均差 +1.8~+2.2 token |
| 提交仍解释不了 → 继续查树构造/候选选择/统计口径 | **支持（主结论）** | 144/144 的提交长度完全由"预算树里有没有这条路径"决定；树第一层常常只有 1–2 个候选；`in_tree` 矛盾 0/48（walk 与掩码实现自洽） |
| 训练样本/接受指标几乎不变 | **不适用** | 本节不训练；只说明 6 条视频无法分辨 phaseC4 的 0.003 τ |
| 目标/接受错配、chain/tree 拓扑错配 | **不支持** | 完整 chain 在共同上下文上与离线 rank1 精确一致（无目标错配）；tree 与 chain 差别很大（−1.25 τ）但方向是 tree 有效、chain 弱，不是拓扑错配 |

**一句话：对齐 reference 之后，离线指标对"链"是精确的、对"树的首 token"是同一事件，
真正没被解释的是树在中深层能提交多少 token —— 这由候选树分配（预算 vs 浅层覆盖）决定，
不由 head 的候选覆盖决定。** 因此下一步应做**固定 head / 候选集 / 验证预算的树选择对照**
（与 phaseC6 §6 的独立建议一致，本轮未启动），**不启动任何新训练**。

### 13.11 未能验证 / 限制

1. **n=6 视频、48 个 anchor**：这是预注册探针，不是 τ 检验。整轨迹 tree τ 在本样本上把臂排序
   名义反转，而 phaseC4 的 29 条结论是臂差 0.003 τ 级；**不能用本节给臂重新排序**。
2. **36/48 anchors 已在分叉之后**：这些行是在 C0 上下文上的**单轮反事实重放**，不是该臂自己
   轨迹上的状态，不能当 same-state 比较（已分别标注）。
3. **"top-8 上界"是事后 oracle**（用真实 AR token 才知道路径），生产树在建树时不知道它；
   因此 gap ≠ 实现 bug。本节的数字不能当作"换个树就能拿到的收益"。
4. **只测了现有 best-first 树**（b63k8）；没有测任何替代分配方案，也没有测其它 budget/branch。
5. **draft logits 不是逐位一致**（48/48 top-8 一致，但只有 6/48 位模式一致）。近 tie 在别的 anchor
   上可能翻候选，本轮未越出这 48 个 anchor 检验。
6. **只覆盖前 8 轮**（这些视频整条回答为 35–97 轮）；deep-round 行为未测。
7. 未做计时/加速结论；未评测新测试集候选；未训练、未更新参数。
8. 原始 jsonl 早期字段名的说明：度量所用的 AR 窗口由 `stage_a.*.ar_ids` 与 anchor 的 `start`/`n_in`
   在报告阶段重建 (`_ar_window`)，并由 §13.5 的 506/506 + 1343/1343 对齐门禁证明它就是离线 rank 所对的那个窗口；
   脚本已改为在 anchor 记录里直接写 `ar_window`（含 `ar_first_pos`、`ar_continuation_sha`）。

### 13.12 复现入口与产物

```bash
cd /home/mcy/Projects/GLANCE/repro
PY=/home/mcy/.conda/envs/fin_research/bin/python ; R=/mnt/local/mcy/GLANCE/results/phaseC5_brief
$PY -u diag_bref_align.py prereg --out $R/prereg.json
# gate：先 2 条，必须逐 anchor 复现 C0 才继续
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  $PY -u diag_bref_align.py run --videos v_4oi6dXfpJS4 v_-MbZ-W0AbN0 --tag gate --gpu 0
# 其余 4 条
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  $PY -u diag_bref_align.py run --videos v_0k0xNs8Zjb4 v_1dDAcUliXrQ v_2nDbYhgdw2A \
      v_0x6ZV9Y8g_c --tag rest --gpu 0
# 整轨迹两种拓扑（未修改的解码路径，均带 AR 门禁）
$PY -u diag_bref_align.py chain16 --gpu 0 --out $R/chain16_all6.json
$PY -u diag_bref_align.py chain16 --budget 63 --branch 8 --gpu 0 --out $R/tree63k8_all6.json
# 汇总（CPU）
$PY -u diag_bref_align.py report --prereg $R/prereg.json \
  --stage-a $R/stage_a_gate.json $R/stage_a_rest.json \
  --anchors $R/anchors_gate.jsonl $R/anchors_rest.jsonl \
  --decode $R/chain16_all6.json $R/tree63k8_all6.json \
  --out-md $R/bref_report.md --out-json $R/bref_report.json
```

| 类型 | 路径 |
|---|---|
| 预注册 | `results/phaseC5_brief/prereg.json` |
| gate 判定 | `results/phaseC5_brief/gate_decision.json` |
| 冻结 anchor 与三臂重放原始记录 | `results/phaseC5_brief/anchors_gate.jsonl`、`anchors_rest.jsonl`（144 条 = 48 anchors × 3 臂） |
| 轨迹/门禁元数据 | `results/phaseC5_brief/stage_a_gate.json`、`stage_a_rest.json` |
| 整轨迹 chain/tree | `results/phaseC5_brief/chain16_all6.json`、`tree63k8_all6.json` |
| 主表 | `results/phaseC5_brief/bref_report.md`、`bref_report.json` |
| 日志 | `results/phaseC5_brief/logs/` |
| 已隔离的失败运行（M-RoPE delta 未绑定） | `results/phaseC5_brief/failed/` |

本轮未修改 `opd_fullkl.py`、`opd_train_video_fullkl.py`、`train_head_video.py`、`common.py`、
`master_optim.py`、`diag_gain_stuck.py`、`GLANCE_repo/`、`AGENTS.md` 或任何既有结果文件。
