# GLANCE × ScienceQA：原 warm 丢失后，重建图像 Draft warm，再执行官方 KL / SFT 成对训练

**给可访问用户服务器的 Codex 的完整执行 Prompt｜2026-10-16｜取代此前 `GLANCE_ScienceQA_serial_overnight_Codex_20261016.md` 的“warm 缺失即 BLOCKED”规则。**

## 用户授权和核心目标

用户已明确说明：**原图像 warm checkpoint 被用户删除**。因此这次不是“从已有 warm 继续训练”，而是允许从**原始模型结构与新的初始化权重**开始，先训练一个新的有效 `warm_v2`，再把**同一个** `warm_v2` 用作 ScienceQA continue-SFT 与官方 Draft-OPD KL 两臂的共同起点。所有阶段都应在服务器上真实运行；但绝不能把没有训练过的新 Head 直接称为恢复的旧 warm，也不能把结果与历史 `head` 数值当作同基线公平比较。

本任务中“从零”的定义是：**只重新初始化并训练 DFlash Draft Head**，**不是从零预训练 Qwen3-VL-8B-Instruct Target**。Target 始终使用经核验的现有预训练权重、固定 processor、冻结参数；Draft 架构、所取融合隐状态层、Head 初始化方式以 GLANCE 真实可审计代码为准，禁止随意换成其他大小的模型。

**本轮固定使用此前已通过 L2_gpu 门禁的串行（serial）Target 验证路径，禁用仍处于 HOLD 的 block verifier。** 训练目标使用当前统一的官方 KL：accepted Bernoulli forward KL、rejected token-level k3、官方池化/位置衰减/stream weights、真正由当前策略访问的 on-policy rollout。不得回退历史 ScienceQA `CE + full reverse KL + fixed AR anchors`，不得只为提速减少 anchors，亦不擅自切换 TV loss。

用户希望夜间自动运行与可恢复结果，但不要求伪造一个“早晨必须完成全部三阶段”的承诺；时间、磁盘或 GPU 不够时，保存**真实已完成的 warm 或成对 checkpoint**，晨间汇报实际状态。不要不设停止时间无限运行，也不抢占其他人的 GPU。

## 0. 首先确认旧 warm 确实不可恢复（只做有限、定点审计）

1. 读最新 `/home/mcy/Projects/GLANCE/AGENTS.md`，已有 GPU 门禁、官方 KL 最终验收、ScienceQA 数据、serial 夜间任务报告。核实代码/数据/模型根目录，所有 `/home/mcy/local*` 软链的 `realpath`，目标挂载容量与 GPU 状态。
2. 旧 warm 记录的 `checkpoints/head`、曾报告的部分哈希 `fc0ad5a6…` 只能作为历史查找线索；优先使用旧报告 manifest 取得完整 SHA 与准确架构。检查**现存且已索引**的备份、归档、历史 checkpoint 或软链接目标；不要大范围扫描整台共享服务器，不得修改历史结果。
3. 如果找到**相同 SHA、相同架构、状态完整、可独立加载**的旧 warm，可以直接使用恢复副本，明确标为 `RECOVERED_EXACT_WARM`，然后跳过 scratch warm 阶段；禁止把不同 SHA 的旧 SFT/OPD 权重冒充旧 warm。
4. 如果找不到，**不要因为 `BLOCKED_MISSING_BASE_CHECKPOINT` 而停止**，转到下一阶段执行新 Head 初始化与 warm 训练，标记 `NEW_SCRATCH_WARM`。只有架构无法安全初始化、训练数据缺失、正确性/许可/资源条件不满足时，才以具体原因阻塞。

## 1. 从零初始化新的 DFlash Draft Head（不动 Target）

1. 用项目中原 GLANCE/DFlash Head 的**已验证构造器和配置**：约 1.05B、5 层、block16、Target 融合隐藏层 `{1,9,17,25,33}`（逐字段以实际 `config.json`/源码确认）。确认 embedding、lm_head、参数 dtype、初始化函数与参考训练脚本匹配；不得随意更换 architecture 或加载来源不明的随机权重。
2. 冻结 `scratch_seed`、初始化源码/架构 hash、Target revision、processor revision、环境版本；保存 `warm_v2_init_manifest.json`。新 Head 初始化使用可重复 RNG，并保存随机状态，但**不能因为随机种子相同便声称与已经删除的旧 Head 参数一致**。
3. 对新 Head 做图像/文本→Target fused features→Draft 的小型无更新结构门禁，确保视觉 patch/Grid、position、tokenizer/vocab、cache、融合状态维度无误。Target 不应收到任何梯度。
4. 完全禁止从历史 ScienceQA 官方 test、dev/confirm、gold solution 或过去评测时的 Target 答案偷取 warm 训练样本。所有 new warm 数据都要有 PID/图片哈希和 split manifest。

## 2. 训练新的图像 warm：先恢复可对照的强起点

**优先复用原 GLANCE 图像 warm 的已核实数据与训练配方。** 历史记录显示它与 COCO/TextVQA 图像任务相关（曾报告约 16K 行），但该数字与脚本必须以实际源记录、构建器与数据权限为准，**不能直接把旧项目里某份 16K 的文件当成已验明的完整 warm 配方**。先审计：原 warm 训练数据、真实 Head 初始化方法、variable-reveal/CE 目标、实际 optimizer steps、学习率、梯度裁剪、图像处理方式和预期 loss 形态。

如果原 warm 配方所需图像/问答数据全部可用、来源与许可核验通过，则优先严格执行**可复现的原配方**，但输出名称仍叫 `warm_v2`，因为它不是旧权重的逐位恢复。

如果原 warm 原始数据/配方无法重建：

- 允许使用**仅官方 ScienceQA train 的独立 warm 子集**做一个新的、明确标注为 `scienceqa_scratch_warm` 的图像 CE/variable-reveal warm；先冻结 **warm 集 / 后续对照训练集 / dev / test** 的题目与图像簇，不得跨集合重复图像或同题。
- 建议在 train 中划分可用且互异的 `warm_pool` 和 `compare_pool`，规模以真实图像数为准；**任何一个样本不得同时进入 warm_pool 与 compare_pool**。后续 compare 的 SFT/OPD 使用同一 `compare_pool` 和训练日程。这样不等于原 GLANCE warm 复现，须单独报告新协议。
- 不以固定“随便训 10 步”替代 warm：应先记录一个能让 Head 具备合理候选覆盖和实际无损树解码能力的 warm 训练计划，保存完整 updates/监督位置和训练耗时。无损是解码正确性的必要条件，不证明 Head 已充分训练；若 warm 基线仍接近随机，报告并阻止“OPD vs SFT 已在强 warm 上比较”的结论。
- 若既无原 warm 数据也无法构建干净可用的 train 子集，写 `BLOCKED_WARM_DATA`，不要自造训练数据。

Warm 训练使用已验证 SFT 训练入口（不是官方 OPD）。完整 checkpoint 必须保存 Head/bf16 映射与 fp32 master、Adam 两阶矩、RNG、schedule/data cursor、Target/processor 与源码 hash。正式 warm 阶段允许真实 GPU optimizer updates；前若干步内执行恢复与无损门禁，失败应停止并保存证据。

**warm 阶段的唯一验收**：训练确实成功、起点初始化可追溯、更新真实、模型可加载、保存可准确恢复、输入与 AR/SPEC 一致；进一步报告候选覆盖/τ 作为实际模型质量，不以达到旧 `τ≈5.2` 作为本次重建 warm 的硬要求，因为数据和初始化可能不同。

## 3. 新 warm 完成后，从同一份权重启动 ScienceQA 两臂

在 `warm_v2` 完整、SHA 核验和必要门禁通过后，冻结成对协议：

- I0：**新 warm_v2（zero update 基线）**。
- SFT：从 `warm_v2` 加载**相同模型参数、同一 data schedule**做 continue-SFT。
- OPD：从**同一个** `warm_v2` 加载，用官方 Bernoulli＋k3＋加权池化＋策略访问的 serial rollout 做官方 KL-OPD。
- 两臂独立、不得顺序串接（不能先训练 SFT，再把该 SFT 结果作为 OPD 的共同起点却还称为同起点试验）；真实 Target 权重始终冻结。
- 沿用项目现有经过审计的 ScienceQA 真实图像 train/val，保持相同 QA/图像 IDs、Prompt、batch、LR、精度、EOS、block16/b63k8。原 ScienceQA test 之前已用于机制开发，不用于当前选点/调参；未来正式泛化需新确认集。
- 建议继续采用此前夜间任务的两个成对里程碑：`M1=60 updates/arm`，`M2=120 updates/arm`，每 update 4 个不同 QA，准确记录真实 anchors/有效监督位置及 teacher 前向；两臂 compute 不可能只凭 updates 相同就相等。
- OPD 路径必须为 **serial**，不得因为 block 更快就开启 HOLD 的 block 前向，也不准使用 detached past cache 改变 scoring 梯度。

**特别重要：warm 重建的训练时间不得混计到 SFT/OPD 两臂的继续训练步数里。** warm 是双方共同的前置阶段，所有计算成本单列。

## 4. 夜间持久执行与完成度规则

1. 用真实空闲 GPU、合适的共享锁和独立 `tmux` 会话运行；使用 `/mnt/local`、`/mnt/local2`、`/mnt/local3` 的实体存储，home 仅存代码。现有卷曾接近 100% 使用率；训练前确认至少有完整 checkpoint＋原子暂存＋日志的空间。禁止删除历史 checkpoint、他人的缓存和旧报告。
2. 用一个固定且有限的工作窗口（与用户前次夜间授权一致的最多约 8.5 小时，不按受限两条样本吞吐捏造完成时长），在窗口结束时安全保存并退出。用户如果随后明确授权更长时间，再续训；不能自己无限延长。
3. 自动选择工作阶段：`AUDIT → SCRATCH_INIT → WARM_TRAINING → WARM_READY → SFT_M1 → OPD_M1 → MATCHED_EVAL → (有余量时 SFT_M2/OPD_M2)`。**只有 warm 完成才进入下一段**。若到凌晨仍只完成 warm，就给用户一个真实可恢复的 warm checkpoint，而不是用随机未热身 Head 继续 OPD 来制造结果。
4. `RUN_STATUS.json` 原子写入 `stage/status`, seed, warm checkpoint hash, SFT/OPD 各自 updates, GPU、显存与磁盘状态、PID/tmux、最近心跳、当前有效样本、最新可恢复 checkpoint、下一启动命令。`READ_ME_FIRST.md` 用通俗中文说明“今天重建到哪一阶段”“能否比较 SFT/OPD”“有没有真实更新”。
5. 分阶段完整保存，使用临时目录→校验 SHA→原子改名；不得覆盖既有 checkpoint。重复计算用于恢复检查的 updates 不能计入正式训练进度。
6. 正式成对训练若只完成一臂或步数不同，写 `PAIR_INCOMPLETE`，不比较不等训练步数的模型；只有两臂到达 M1/M2 同步里程碑，才报告在线 τ、有效 `generated_tokens/verify_rounds`、候选覆盖、正确树路径和逐 token `SPEC=AR`。
7. 不因任何阶段 τ 不好就随意切换模型结构、改 KL、临时缩减 anchors、启用 block、在测试集上选参数，或重新选择更有利的 warm checkpoint。

## 5. 必须通过的正确性与性能约束

- Target 权重前后 SHA 不变、无梯度；Head 参数更新真实、非 NaN；优化器 fp32 master、梯度裁剪合理。
- `n_acc`、accepted/rejected 位置、teacher token 和 serial rollout 实时一致。不能恢复旧 fixed AR anchors 或 one-hot CE＋full reverse KL，却把结果标为官方 OPD。
- 真 GPU 的独立 AR 与 SPEC **逐 token 完全一致**（EOS/cap/bonus/root 均计入）。任何不一致即停止后续评测与训练解释。
- 真实 head 的连续训练 4 updates 与 2+resume+2 在相同数据/RNG/优化器参数下通过等价门禁；确有不可避免的数值差异须留证，不得直接宣称逐位可恢复。
- 计时只使用可复核的真实调用与 GPU 同步结果，共享负载污染的数据不作可靠加速结论。统计以独立图像/题目簇为单位，不把同图 token 当独立样本。
- 如果 scratch 重建导致 warm 与旧模型表现差别很大，**不能将新训练结果与旧 ScienceQA 的 I0/SFT/OPD 数值直接当作固定同起点消融**。

## 6. 交付和最终中文报告

建立新的独立 run，不覆盖之前的 BLOCKED/验收/HOLD/ScienceQA 负结果：

```text
/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_<run_id>/
  audit/old_warm_search.json
  audit/source_and_license.md
  audit/data_split_and_leakage.json
  protocol.json
  amendments.jsonl
  warm/initialization.json
  warm/train_logs/
  warm/checkpoints/
  warm/final_sha256.json
  train/sft/
  train/opd/
  eval/paired_dev_metrics.csv
  eval/spec_equals_ar.json
  RUN_STATUS.json
  READ_ME_FIRST.md
  resume_command.sh
  reports/FINAL_SCRATCH_WARM_AND_OFFICIAL_KL.md
  reports/DELIVERY_MANIFEST.json
```

最终报告必须明确：旧 warm 是否真的缺失？用哪套架构/初始化从零构建新 Head？新 warm 使用的是**原配方复刻**还是 **ScienceQA train 子集自定义 warm**？warm 实际训练了多少 updates、用了多少数据和时间、有没有完成？共同 warm SHA 是什么？SFT/OPD 各完成多少匹配 updates、损失配置、真实 online anchors 与 Target 调用量如何？有无真正的可比接受率/加速结果？剩余工作与恢复命令是什么？

**执行要求：先核验来源，找不到原 warm 就启动新 Head 初始化与 warm 训练，不要停在旧的 `BLOCKED_MISSING_BASE_CHECKPOINT`。但没有可验证 warm 训练数据、GPU、磁盘或训练正确性时，必须如实 BLOCKED，不得伪造已恢复的旧 checkpoint 或已完成的成对实验。**
