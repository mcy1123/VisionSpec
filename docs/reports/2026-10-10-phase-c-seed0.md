# Phase C seed-0 三臂报告：aligned 起点下 paper-faithful OPD 未改善投机解码效率

> **补注（2026-10-10，审计后）**：本报告以下三处解释**作废**，数字不变、无需重跑。
> 完整审计见 [`2026-10-10-phase-c-corrections.md`](2026-10-10-phase-c-corrections.md)。
>
> 1. §6「C1 只监督 1200 个位置 / C2 是其 15 倍」**错误**。`train_head.build_blocks`
>    把每个 anchor 展开成整个 block，按 `super_mask.sum()` 实测：C1（24 anchors×50）
>    监督 **11648** 个位置、权重和 **4928**；C2/C1 = **1.55×**，不是 15×。
> 2. §5「本轮 anchors 统一由 fp32 rollout 冻结后复用」**错误**。
>    `opd_train_video_fullkl.py` 每个 update 都调用 `opd_online_rollout.rollout`，
>    anchors 由**当前策略在线**产生；冻结 anchors 是 **B0.5** 的校准设置。
>    因此「bf16 会重选 anchors」在 C2 上**完全生效**，精度与轨迹本轮**未分离**。
> 3. §8.3「50 updates 的预算本身在退化」是**过度解读**。C1 相对 A4 同时改了 lr 之外的
>    anchor 上限（96→24，监督密度降到 1/4）、Adam 重初始化与采样顺序。后续校准
>    （lr 3e-5、96 anchors、10 updates）在完整 validation 上拿到 **+0.0061**，
>    说明 C1 的退化**主要来自学习率 3e-4**，而非「继续训练必然退化」。
> 4. §7 的计时数字**同样不可跨臂比较**（与 §7 原文结论一致，此处再次确认）。


日期：2026-10-10（运行窗口 2026-10-09 23:20 → 2026-10-10 01:55）
任务依据：`docs/plans/video-opd.md` §10.5（冻结配置）、§10.6（首轮训练与评测、停止/延长规则）
前置：`docs/reports/2026-10-09-b1-formula.md`（公式核实）、`docs/reports/2026-10-09-b1-precision-and-phasec-config.md`（精度判定与冻结配置）

## 0. 一句话结论

**在 aligned `step1000` 起点上、seed 0、每条训练臂 50 updates、fp32/b63k8/AR 门禁下，
paper-faithful 全词表 OPD（C2）没有改善投机解码效率：它同时低于不训练的 C0 和同预算
continue-SFT 的 C1，且三个配对 95% CI 全部落在零以下。** 按 §10.6 的预注册停止规则，
**暂停延长、不再扩到 100 updates、不展开多种子**。

**边界**：这是所测模型（Qwen3-VL-8B + 1.05B 5 层块头）、所测数据（439 行 VideoDetailCaption
train 划分）、所测预算（50 updates、24 anchors/update）、所测实现（bf16 teacher、见 §5）
下的结论，**不推广为「视频领域 OPD 无效」**。单 seed，且 C1/C2 都低于 C0 说明 **50 updates
这个预算本身太短、继续训练即在退化**，这是解释本结果不可忽略的背景。

## 1. 冻结的配置（见前置报告全文）

| 项 | 值 |
|---|---|
| 起点 | `checkpoints/head_video439_aligned/step1000`（model.safetensors sha256 `e2526a249a8ef7e1db3a9990589b7275…`） |
| 三臂 | C0 不训练；C1 `train_head_video.py --feature-path cached`；C2 `opd_train_video_fullkl.py` |
| 数据 | `pool_499` `split=="train"` 439 行；heldout 60 行不参与 |
| optimizer | AdamW lr 3e-4, betas (0.9,0.95), wd 0, clip 1.0, **常数 lr**（偏离论文 cosine，已记录） |
| anchors/update | 24（两臂同设；**定义不可比**，见 §4） |
| dtype | C2 teacher **bfloat16**（精度近似，见 §5）、head bfloat16；C1 head bfloat16 |
| γ | 0.8，只作用于 rejected 流 |
| 评测 | fp32、tree b63/k8、max_new 1024、greedy 跑到 EOS、`--ar` 逐 token 门禁 |
| C0 复用 | 直接取 A4 同配置评测，不重跑 |

## 2. 主结果：完整 `video_fresh30`（n=29）

30 条输入、首条 warmup 不参与聚合（与 A4 完全同协议）。**三臂各 29/29 与 30/30 无损**。

| arm | n | mean τ | median τ | EOS | lossless |
|---|---:|---:|---:|---:|---:|
| **C0** 不训练（aligned step1000） | 29 | **3.7377** | 3.7778 | 29/29 | 29/29 |
| **C1** aligned continue-SFT 50 updates | 29 | 3.6579 | 3.6471 | 29/29 | 29/29 |
| **C2** paper-faithful OPD 50 updates | 29 | 3.5095 | 3.5135 | 29/29 | 29/29 |

### 2.1 逐视频配对 Δτ（bootstrap 单位 = 视频，10k 次，never token）

| 对比 | mean Δτ | median Δτ | 95% CI | t | 正向视频 |
|---|---:|---:|---|---:|---:|
| **C2 − C0** | **−0.2281** | −0.2096 | **[−0.2830, −0.1738]** | −8.05 | 1/29 |
| **C2 − C1** | **−0.1483** | −0.1485 | **[−0.2037, −0.0953]** | −5.22 | 5/29 |
| C1 − C0 | −0.0798 | −0.1021 | [−0.1269, −0.0311] | −3.21 | 5/29 |

**三个 CI 全部排除零，方向一致为负**，不是「趋势但不确定」。按 §10.9：C2 **既不优于 C0
也不优于 C1**，因此连「缓解继续训练退化」这一较弱表述都不成立。

## 3. 过程监控子集（8 条，选点规则与模型得分无关）

| 节点 | arm | n | mean τ | median τ | lossless |
|---|---:|---:|---:|---:|---:|
| 0 | C0（取自 A4） | 8 | 3.8385 | 3.9037 | 8/8 |
| 10 | C1 | 8 | 3.8061 | 3.8006 | 8/8 |
| 10 | C2 | 8 | 3.6643 | 3.5996 | 8/8 |
| 25 | C1 | 8 | 3.6957 | 3.7321 | 8/8 |
| 25 | C2 | 8 | 3.6855 | 3.6774 | 8/8 |

配对（Δ 为正表示该臂更好）：

| 对比 | mean Δτ | 95% CI | 正向 |
|---|---:|---|---:|
| C0 − C1@10 | +0.0324 | [−0.0960, +0.1475] | 5/8 |
| C0 − C2@10 | **+0.1742** | [+0.0513, +0.3074] | 6/8 |
| C0 − C1@25 | **+0.1428** | [+0.0508, +0.2577] | 6/8 |
| C0 − C2@25 | +0.1530 | [−0.0097, +0.3109] | 4/8 |
| C2@10 − C1@10 | **−0.1418** | [−0.2563, −0.0319] | 2/8 |
| C2@25 − C1@25 | −0.0103 | [−0.1094, +0.0802] | 5/8 |

**两条训练臂在监控子集上就已经一起低于 C0**，且 C1 在 25 节点也掉到 C0 之下 0.14。
这把「OPD 特有地有害」与「50 updates 的继续训练本身在退化」区分开：**两者都在退化，
OPD 退化更多**。

## 4. 触发的是哪条预注册规则

§10.6 停止/延长规则 2：「小子集上 C2 相对 C0 连续两个节点 mean τ 下降超过 0.05：
暂停延长，在完整 validation 上复核最新节点后决定收口。」

- 监控子集 C2−C0 = **−0.1742（节点 10）**、**−0.1530（节点 25）**，连续两节点均 >0.05 → **条件成立**。
- 规则 6 的延长条件（update50 的 C2 同时高于 C0/C1）**不成立**。

> **判定：PAUSE——暂停延长，在完整 `video_fresh30` 复核 update50（§2 已完成），决定收口。
> 不扩到 100 updates，不展开 3 seeds，不搜索超参数。**

## 5. 必须随结论一起报告的精度近似（§10.4 / §10.2）

本轮 C2 的 teacher 是 **bfloat16，不是 fp32**：实测 fp32 teacher + AdamW 在 48 GiB A6000 上
**放不下**（24 anchors 与 1 anchor 两种设置都在 `optimizer.step()` 初始化 Adam 时
`torch.OutOfMemoryError`；1-anchor 对照证明减少 anchors 不是杠杆）。详见
`docs/reports/2026-10-09-b1-precision-and-phasec-config.md`。

B0.5 在同轨迹、同 prefix、同 mask 上的量化：

- 总 loss 相对差 **1.2e-3**（各自 student）/ **7.1e-4**（固定同一 student）；
- **逐 anchor 相对差 mean 1.67e-2、max 9.5e-2**（聚合靠正负抵消）；
- teacher argmax 一致率 97.7%；head 梯度 cosine（纯 teacher 效应）0.99986；
- **但 bf16 会重新选 anchors：8/8 条视频的 proposal 列表都不同，2/8 条 `pos`/`n_acc` 直接不同。**

因此**不得把 C2−C1 的差异整体归因于 KL 语义**：bf16 臂同时是「不同精度」和「不同轨迹」。
本轮把 anchors 统一由 fp32 rollout 冻结后复用，部分控制了这一点，但不能消除。

## 6. 训练与计算预算（两臂不可配平之处，如实记录）

| 量 | C1 continue-SFT | C2 OPD |
|---|---:|---:|
| optimizer updates | 50 | 50 |
| anchors/update | 24 | 24 |
| **anchor 的定义** | 1 个被监督**位置** | 1 个 draft **block**（15 offsets） |
| 总 anchors | 1200 | 1200 |
| **总有效监督位置** | **1200** | **18000**（accepted 2043 + rejected 15957） |
| 视频曝光数 | 50 行/步（每步 1 行） | 300（每 update 6 条） |
| 峰值显存 | 26.2 GiB | **26.42 GiB** |
| 训练墙钟 | 约 1.5 min | 约 7 min |
| 梯度范数（clip 前） | — | 1.53 – 5.50，平均 clip 比例 0.626 |
| 非有限值 / 空 update / EOS 排除 | 无 | **0 / 0 / 0** |

**同一个「24 anchors/update」在两条臂下不是同一份计算预算**：C2 的监督位置数是 C1 的 **15 倍**，
且 88.6 % 落在 rejected 流上（15957/18000）。这与 §10.2 在新起点上测到的形态一致
（B0.5：accepted mean KL 0.515 vs rejected mean KL 6.516）。

## 7. 计时：本轮数字**不可跨臂比较**，不构成速度结论

| arm | spec ms/tok（中位） | AR ms/tok | AR/spec |
|---|---:|---:|---:|
| C0 | 59.29 | 117.34 | 1.979× |
| C1 | 63.30 | 119.61 | 1.890× |
| C2 | 31.20 | 56.13 | 1.799× |

**这三个绝对时间不能用来比较速度**：C2 的 τ 最低却显示最快的 spec 时间（31.2 vs C0 的 59.3 ms/tok），
这只可能是三次评测落在不同 GPU 争用条件下（有他人任务、且评测串行跨越约 2 小时）。
τ 与 ms/tok 的方向不一致，本身就是该组计时被混淆的证据。
按 §10.6 第 5 条，**本轮只报无损门禁通过（三臂各 30/30 逐 token 一致），不宣称任何端到端加速**。
要得到可用计时需要在同一时间窗内交错测量并加重复数。

## 8. 结论与边界

1. **C2 未改善投机解码效率**：相对 C0 −0.2281（CI [−0.2830,−0.1738]），相对 C1 −0.1483
   （CI [−0.2037,−0.0953]），三个配对 CI 均排除零。
2. **不能表述为「OPD 缓解了继续训练退化」**：C2 低于 C1，§10.9 允许的那条较弱表述也不成立。
3. **50 updates 的预算本身在退化**：C1 相对 C0 也显著为负（−0.0798，CI [−0.1269,−0.0311]）。
   因此本实验真正比较的是「两种继续训练谁退化得少」，而不是「谁在收敛方向上更好」。
   这是本结果最重要的限定：**它没有测试论文的 8-epoch 量级预算**。
4. **paper-faithful 语义本身按论文实现并经核实**：全词表 forward/reverse KL、
   `w_k=0.8^(k-1)` 只作用于 rejected、两流各自归一后等权（Eq 13）、整段被拒后缀
   （不是 first-reject-only）、draft-induced prefix（首个 rejected 位置前缀按定义相同、
   分歧从 `k≥n_acc+2` 出现）。与官方代码的两处分歧（采样估计量、池化归一）已在
   `docs/reports/2026-10-09-b1-formula.md` 记录并显式选择论文语义。
5. **本结果不推广**到其他模型/数据/预算/实现；bf16 teacher 与 50 updates 是两个明确的限制。

## 9. 产物与复现

| 类型 | 路径 |
|---|---|
| 训练入口 C2 | `repro/opd_train_video_fullkl.py`（CPU 测试 `repro/test_opd_train_fullkl.py`，17/17 通过） |
| full-KL 模块 | `repro/opd_fullkl.py`（测试 `repro/test_opd_fullkl.py`，16/16 通过） |
| 三臂启动 | `repro/run_phaseC_seed0.sh` |
| 评测启动 | `repro/run_phaseC_eval.sh` |
| 分析（含 bootstrap） | `repro/analyze_phaseC.py` |
| 子集冻结 | `repro/freeze_phase_subsets.py` → `results/phaseB05/subsets.json` |
| C1 检查点 | `checkpoints/phaseC_C1_continue_sft/step{5..50}` |
| C2 检查点 | `checkpoints/phaseC_C2_opd_fullkl/step{5..50}` |
| 训练日志 | `results/phaseC/C1_continue_sft.{log,jsonl}`、`results/phaseC/C2_opd_fullkl.{log,jsonl}` |
| 评测结果 | `results/phaseC/{C1,C2}_update{10,25}_monitor8.json`、`results/phaseC/{C1,C2}_update50_fresh30.json` |
| 汇总 | `results/phaseC/summary_phaseC.json` |

```bash
cd /home/mcy/Projects/GLANCE/repro
R=/mnt/local/mcy/GLANCE
PY=/home/mcy/.conda/envs/fin_research/bin/python

# 子集冻结（已跑，重跑会覆盖同一路径；本行仅供追溯）
$PY freeze_phase_subsets.py

# 训练（C1 然后 C2，串行）
GPU=0 ./run_phaseC_seed0.sh

# 评测（监控 8 条 + 完整 30 条）
./run_phaseC_eval.sh

# 分析
$PY analyze_phaseC.py
```

## 10. 下一步（不自动执行）

按 §10.6 规则 7「否则不自动扩展长程训练或多种子，完成原因分析并收口」，本轮到此收口。
若之后要继续，优先级建议为：

1. **先解决预算问题**：C1 自身在 50 updates 就退化，任何「OPD vs continue-SFT」的比较
   在退化区做都难以解释。应先找到 continue-SFT 不退化的预算（更小 lr、更少 updates 或
   混入原始 SFT 数据），再在该预算上重做对照。
2. **fp32 teacher**：需要 >48 GiB 单卡或模型并行；当前证据表明单卡不可行。
3. **冻结新测试集**：本轮全部结论都是 validation 口径（见下）。
