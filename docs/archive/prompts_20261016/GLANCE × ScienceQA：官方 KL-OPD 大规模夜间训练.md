# GLANCE × ScienceQA：官方 KL-OPD 大规模夜间训练（Codex 直接执行版）

> **用户明确授权**：跳过单独的成对 10-update 探针，尽可能在一次夜间作业中直接启动并运行较大规模的 ScienceQA 正式训练；第二天查看 SFT 与官方 KL-OPD 的对照结果。
>
> **执行者身份**：能够在用户服务器上实际访问 `/home/mcy/Projects/GLANCE`、GPU 和 `/mnt/local*` 存储的 Codex/DeepSeek 执行代理。不能只写计划，必须执行至实际资源/时间约束允许的阶段；但是**不能声称本聊天助手已经登录或启动了服务器训练**。
>
> **原则**：已经完成 L2_gpu（2 图+1 视频真实模型的无更新 forward/backward、Target 无梯度、树/在线 rollout/独立 AR token 对齐、0 参数更新，峰值约39.90 GiB）。这些门禁无需重复为独立阶段，但**真实 GPU optimizer update、有效参数移动、断点恢复**尚未验证。将它们直接集成到正式训练的前几个 updates 的强制停止条件，而不是另跑一份独立 10-update pilot。

## 0. 目标与不可逾越的边界

**主要问题**：在冻结 Qwen3-VL-8B-Instruct Target、完全一致的 ScienceQA 图文输入和 decoder 上，**官方 DFlash KL-OPD** 是否在足够多的训练图像与更多 updates 下，相比同一初始化的 continue-SFT，提高真实 b63k8 在线接受长度并降低可比推理延迟？

使用**真实现行官方 KL**，不是上次负结果的 `accepted CE + rejected full-vocab reverse KL + fixed AR anchors`；后者的旧结果保留、只做历史背景，不作为官方 OPD 失败的证据。

官方基准依最新验收所冻结的 local upstream revision，优先核对对应 DFlash KL 版 `run_qwen_gsm8k.sh`（此前冻结 commit `d1476f5`）与 `losses.py/core_algos.py`，**实际文件/哈希为准**：
- Accepted：**Bernoulli forward KL**，该 token vs 非该 token；
- Rejected：**token 级 k3**，采用官方 clamping/单 token logprob 语义；
- Rejected decay：`0.9`（若验收配置锁定值不同，先列出冲突并按冻结官方基准处理，不得偷偷选有利设置）；
- 两流：**官方按有效权重/位置池化**，`rejected_draft_stream_weight=2.0`，accepted stream weight 1.0；
- **真正 on-policy rollout**：由当前草稿在当下的访问状态推进 anchor/teacher 验证，不是旧 AR 固定 anchors；`FIRST_TOKEN_ONLY=False` 表示被拒合法后缀；
- 严格区分完整官方 KL 配方、此前 Task A 仅 rejected=k3 的部分实验及 ScienceQA 历史 CE+full reverse 实验。

**禁止**：为了得到正结果混入 k3/TV/Bernoulli 的不同变体、换 baseline、调树预算、主动搜索超参数；使用 gold answer/lecture/solution 做冻结 target 的下一 token 教师轨迹；打开历史视频冻结测试集；覆盖历史产物；重写旧负结果；抢占 GPU、杀人进程；在共享磁盘已满的情况下无界生成模型文件；自动终止或清理其他人的任务。

**本次从已过验收的“公共官方 KL 训练入口”直接出发**，不要又复制一个新 ScienceQA trainer 产生代码分叉；只允许新的实验 launcher/manifest 和修复实际可复现的硬阻塞。如果必须修改计算语义，则新建 run、重新确定共同起点并对称重跑两臂，不能不同版本实验混进同一公平对照。

## 1. 开工前快速审计（只检验必需项，不重新做几小时的全面审计）

读取：

```text
/home/mcy/Projects/GLANCE/AGENTS.md
/mnt/local3/mcy/GLANCE/results/official_kl_final_acceptance_20261016_165617/reports/FINAL_ACCEPTANCE_REVIEW.md
/mnt/local3/mcy/GLANCE/results/official_kl_final_acceptance_20261016_165617/release_gate.csv
/mnt/local3/mcy/GLANCE/results/official_kl_gpu_watch_20261016/review/GPU_GATE_REVIEW.md
/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/reports/FINAL_SCIENCEQA_SFT_VS_OPD.md
```

现场确认文件路径是否存在、checkpoint 身份 SHA、代码版本、当前 CLI 与 `--resume-from` 能力。特别验证 ScienceQA CLI 最终调用公共官方 KL 入口，不能只是独立测试函数使用了公共 KL。

ScienceQA 数据使用经认证的以下 HF revision 及现有图片/源文件，不重新下载 240/240 dev/test：

```text
lmms-lab-encoder/ScienceQA@69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50
/mnt/local2/mcy/datasets/scienceqa_glance_20261015/
```

**切忌目录/用户提供的旧路径混同**：更早三个 `Draft-OPD/opd_multimodal/data_sqa_{train,test,full}` 是旧子集来源，可只读查重，但主入口采用上次审计并冻结的官方 img split、图像 SHA 与样本身份。既有数据 IMG train 可用量曾测为6218条，跨 split 共图簇较多；新的 eligible train 数必须重新扣除跨 split 重合以及开发/确认图像簇，不能把 6218 全视为合法训练量。

资源：在真实挂载点检查 `/mnt/local`、`/mnt/local2`、`/mnt/local3` 的 free bytes/inodes，峰值显存，已占用的其他进程与锁（原 L2_gpu 无更新 peak 39.90 GiB，**训练峰值可能更高**）。不许因为“长程训练”而抢占别人的任务。空间不足时允许**只清理本次新 run 的可再生临时缓存**，绝不清理历史 checkpoints/原始报告/官方数据；若必须回收历史冗余，先停机列明候选，不在本次自动删除。

## 2. 先冻结“尽量大且能成对完成”的训练规模

用户希望“大数据、大规模、一晚上”，但之前 ScienceQA 的 target AR 标注曾约 13s/QA，且共享 `/mnt` 接近满载。**不要直接假定 3,000 条实时 teacher 轨迹一定能在一个夜间完成**。优先只读取既有可靠且与本次 checkpoint/prompt/processor/数值路径一致的 Target AR 缓存。缓存必须逐条校验 input IDs、pixel sha、image grid、prompt hash、EOS、model/processor SHA、尤其 Qwen3-VL `rope_deltas` 的生成路径；不合格者重新生成，不能把旧 `position_ids` 错误导致的 2-token 循环当真标签。

**固定候选计划（由执行前的资源与吞吐计算决定，不根据训练/评测收益选择）：**

| 档位 | 互异、官方 train 且有真实图像的 QA | 每臂真实 updates | 两臂各自 QA slots | 适用条件 |
|---|---:|---:|---:|---|
| A：优先大规模 | **3,000** | **750** | 3,000 | 安全 eligible≥3,000，且标注+两臂训练+dev评测可以在≤9小时总 wall 预算完成 |
| B：中大型 | **2,400** | **600** | 2,400 | A 超预算，但 B 可在≤9小时成对完成 |
| C：中等规模 | **2,000** | **500** | 2,000 | B 超预算，但 C 可完成 |
| D：最低正式长跑 | **1,200** | **300** | 1,200 | C 超预算，但较原480条/120步有规模增加 |

- 全部为 **4 QA/update**，每个 seed 同一条 QA 顺序和相同数据曝光，固定起点训练到选定终点；所有 4 档只供**训练前按容量/吞吐选择一次**，不是根据效果做 hyperparameter search。
- 使用**已经通过 source/split 去重核验的图像簇**，冻结官方 train 身份；优先在数据池和 GPU 预算允许下选择 A，依次降档。不同官方 split 共用像素的图像簇必须隔离/排除；不能从 val/test 借题凑够3000。
- 从官方 train 的候选中按 seed 与主题/尺寸等既定规则预先固定样本优先级，保证不同档位是相同抽样计划的预先声明前缀（而非看到数据表现后选样）。训练、dev 和尚未用过的确认数据必须按图像簇隔离。
- **绝不对 SFT 和 OPD 使用不同档位或不同更新数。** 若第一臂因临时资源故障无法完成，另一臂的孤立结果标 `partial_unpaired`，不得冒称公平比较。
- 对数据标注/训练真实吞吐在启动前用**现有日志或不引入单独训练探针的极小只读 benchmark**估算，并加≥20%资源/时间余量。正式 `experiment_protocol.json` 与 `schedule.json` 必须在首次正式 update 和任何新模型 dev 评测之前冻结，并列明选择 A/B/C/D 哪档、估算依据、预计 target AR 缺口和存储占用。
- **9小时是最高总任务预算，不是必须烧满的目标，也不是对完成的保证。** 达到限制前若不能完整完成一对臂，则安全停止、原样保留各自精确续训状态与 run manifest；继续训练须用户再次授权。不得为了凑完成在半夜悄悄调整档位。

### Seed 与评测策略

- **首要目标：一对完整的 SFT/OPD 大规模训练（seed20261013）**，这比尝试两个只跑半截的 seed 更有解释价值。
- 如果一对完整、开发集评测与恢复状态全部完成后，**按照预先剩余时间预算**，再执行另一个配对 seed（`20261014`），从共同 I0 各自重新初始化、使用预先定义的第二 seed 数据顺序；不可仅因第一 seed 结果为正才执行第二 seed。为了防止选择偏差，第二 seed 是否启用的规则只依“两个完整臂的剩余预算”和系统资源，不依任何性能结果。
- **主结果**只用固定终点 checkpoint 的 `OPD−SFT`；中间 checkpoint 不准挑最优替代主终点。
- 之前的 ScienceQA 240 test 已被反复读取、用于负结果解释，**不能再次称为 untouched test**。本轮只用固定 240 dev 监测，最终可从剩余官方 val/test 中在看模型输出前冻结一批跨图像簇、与所有旧 dev/test/训练无交叉的 fresh confirm；如无法找到独立数据，明说仅是开发性对照，不得创造“新独立测试”称呼或为了凑样本重用测试数据。历史 video 冻结测试仍不得打开。

## 3. 直接启动正式 GPU 训练，最初 updates 内置安全检查

**用户明确说跳过独立10-update探针。不要另开 smoke 实验后停在那里。** 开始正式训练；第一组真实 optimizer.step **就是正式 run 的 step1...stepN**，同一数据日程，成功时继续累积而不清零。

以下检查为必须的硬门禁，且不会自动改 loss/学习率：

1. **首个真实 update 后**：检查模型 head 权重 SHA/version 变化、fp32 master step、Adam first/second moments 非零、SFT/OPD 梯度 finite、grad clipping 的真实次数、Target 全程 `requires_grad=False`，实际参数更新不是0。记录每 update 真实 QA IDs、on-policy visited anchors、accepted/rejected positions、`N_acc`、池化分母、k3 clamp 触发率与 loss 分流。OPD 的 teacher 不得落回固定 AR anchors；如发现语义不一致，立即 STOP，绝不掩盖后继续训满。
2. **前10 updates 内**：在同一个正式 run 上完成真实训练位置 / EOS / root / KV cache / fused images / teacher next-token 对齐，记录可回放 token 与模型 hash；若已保存运行时相同 QA 固定样本的生产 verifier，一并比对。不能因 top1 接受低就直接判 bug，先查真实语义。
3. **真实 resume 对照**：在不改变正式 run 状态的独立克隆副本上，用其实际前4次输入比较 `4 contiguous updates` 与 `2 updates + save + restore + 2 updates` 的参数、optimizer、RNG、data cursor 和 loss；要能证明四条路径中本次两个图像训练臂的真实混合精度恢复，而非仅 tiny CPU。测试可以小规模保存于本 run 临时目录，若状态保存造成 OOM/磁盘越界则止损并记录，不依赖 CPU18/21项测试替代；**不得将用于验证的克隆更新计入正式 optimizer updates**。
4. **训练中每50 updates**：保存有效 head+master+Adam+RNG+cursor 的 checkpoint 及其代码/数据 SHA，运行 `resume` 完整性快速测试；checkpoint 使用**新 run 独立路径**，不覆盖历史。checkpoint 体积压力大时采用保留 `latest + milestone + best-freeze` 的**本次 run 内**保留策略，但删前需验证完整 SHA 和可恢复性，且需在协议里事先约定，不删除历史或已发布的关键里程碑。
5. **每100–150 updates**：在**预先冻结的少量 dev 监控子集（如32/64题）**评估简洁 τ、有效输出 token/round、错误率、候选 top-8（可采样）与无损门禁。仅用于检测灾难性错误，不以指标选训练量/超参；输出必须标 `dev_monitor_only`，不能替代完整 dev 主结果。中间监控时如果 OPD 暂时落后，不自动停机换配方，除非达到预先定义的“模型数值/无损/正确性故障或严重退化”硬停止条件。
6. **公平性**：两臂相同模型起点、相同 QA ID/顺序、同 LR/AdamW/clip/bf16 master/Target dtype/prompt/EOS/block16/b63k8；一个 seed 必须成对完成。有效监督位置、online rollout/teacher FLOPs 可能不同，必须报 **实际 FLOPs/调用量**，不能称每 update 等计算成本。
7. **脚本启动形态**：若 Codex 会话无法保持9小时前台稳定，可在服务器侧创建自带日志、PID、超时和退出状态的单一受控 `tmux` / `systemd --user` 会话（仅当实际系统支持），通过 launcher 设置总 wall-clock 最长9小时、GPU独占锁、严格 `trap` 安全 checkpoint / cleanup。**必须确认作业进程实际存活、前若干 update 正常，并生成早晨可直接读取的 `RUN_STATUS.json` 和最终报告生成器。** 不应把仅创建了脚本视为实际启动。禁止无限重试或自动杀其他用户进程。

注意：真实 L2_gpu 的峰值显存39.90 GiB来自**无 optimizer update**；训练必须现场检查峰值和内存碎片。保留当前训练语义的同时可调**microbatch和gradient accumulation实现**以控制显存，但要证明有效 4 QA/update 与官方池化的全局分子/分母没有被错误地按 microbatch 平均；调整必须对两个臂相同并写在训练前协议中。如果实际训练必需的额外显存不可用，不要跑到半夜 OOM 死循环，标 `blocked_vram`。

## 4. 采集数据/缓存的具体要求

- Qwen3-VL 增量 AR 标注禁止在 prefill 显式传会丢失 `rope_deltas` 的 `position_ids`；必须按项目修复过的 `VisionPositions.ar_prefill_inputs` 语义执行，缓存输入 hash、model/processor/feature path/EOS 均一致。
- 缓存可靠的当前 Target 可见回答可供两臂复用，但 OPD on-policy proposal 和 teacher verification 必须在当前 head 实际访问状态下重新运行；**不能用预生成 AR 轨迹的固定 anchors 假扮 online rollout**。
- 若选较大档位需要下载更多 ScienceQA 训练图片，必须先按 frozen source revision 和 train PID/hash 补足**缺的图像**，且完全排除跨 split 共同像素簇；禁止把同一图像重复展开成多个训练条目凑数量。
- 不把 ScienceQA 的 gold `answer` / `lecture` / `solution` / `hint` 放进提问；Target 的答案/推理内容由当前模型生成，训练和评测使用同一个已冻结 Prompt。
- 若训练标注及 teacher 前向是绝对成本大头，在**正式训练前**决定可行档位，且先确认在目标档位能得到充足有效 AR 参考。不能仅因预期训练 updates 较快就忽略标注生成成本。

## 5. 正式评测与早晨必须交付的内容

训练完成一对臂后，在同一批固定 dev240 图片/QA 上，用**完全同一** Target/processor/feature path、fp32 Head/Target、EOS immediate、greedy、b63k8 做两臂和 I0 的**真实在线 SPEC decode**。保存每题的独立贪心 AR reference，执行逐 token、EOS与截断的 `SPEC==AR`。此门禁失败，不得报告真实加速收益。

**主结果**（每个 seed 固定最终 step）：

| arm | 已完成的真实 updates | 有效互异 train QA/曝光 | dev 图片簇均值 τ | 有效 output tokens / verify rounds | top-8 候选命中（仅有效 offsets） | GPU 耗时 |
|---|---:|---:|---:|---:|---:|---|
| I0 frozen | 0 | 0 | 待测 | 待测 | 待测 | 待测 |
| Continue-SFT | N | N | 待测 | 待测 | 待测 | 待测 |
| Official KL-OPD | N | N | 待测 | 待测 | 待测 | 待测 |

- `τ` 和实际 `gen_tokens/verify_rounds` 都要报；历史 τ 可能是裁剪前 `accepted+1`，本轮新增统一处理 EOS/terminal-clipping 的有效 token/round 作为关键参考。不要混合固定 anchor 的 `L_raw/L_tree/L_actual` 与真实在线 τ。
- 每张图像多个 QA 先聚合到簇级，独立图像簇配对 bootstrap（seed固定，2000 次）给出 `OPD−SFT` 的差值与95% CI；种子之间单列，不把一个 seed 的 QA/token 视为独立随机实验。
- 可从缓存/新独立诊断 trace 计算 top-8、`L_raw`, `L_tree`, `L_actual` 和首阻断；必须**先按正确前缀顺序判首阻断**，树遗漏可能早于后续 top-8 缺失。`L_actual` 必须在 EOS/horizon 停止，不能沿用 ScienceQA 旧探针 bug；正确情况下避免 `L_actual>L_tree` 的假象。选样不能只抽首尾 `horizon=1` 的位置。
- **时间**：如果 GPU 负载稳定，同卡交错评测 I0/SFT/OPD，关闭重型 trace hooks、同步计时，至少2次重复，区分 decode-only、prefill+decode 和 end-to-end，标精度/VRAM；若其他租户占用变化则 `timing_contaminated`、只报告 τ/轮次，不包装为 wall-clock 加速。
- **无需等到两 seed 都完成才交付可用结果。** 若只有一组成对臂跑完，明确只是一种子探索性正/负证据。若第二 seed 未完成，不用假造第二 seed 结果。
- 若9小时到时仍未完成一对臂，优先保存双方最近的**同一 milestone** 完整恢复点，标 `partial_unpaired` / `resume_ready`，保留恢复命令而不把不同 step SFT/OPD 当公平差异。
- **不再重新打开已用于历史研究的 ScienceQA test240** 来挑 checkpoint/方案；若训练前能冻结全新不相交确认子集，在正式开发完成后可做一次额外验证，但不因此强行延长总 wall 预算。

### 必须具备的早晨产物

将所有新结果写在一个全新且未存在的 `/mnt/local*/mcy/GLANCE/results/official_kl_scienceqa_overnight_<run_id>/` 下。至少提供：

```text
RUN_STATUS.json              # 当前状态：RUNNING / COMPLETE / PAUSED_BUDGET / BLOCKED / FAILED
READ_ME_FIRST.md             # 一屏摘要：当前训练档位、实际完成步数、三臂 dev、是否无损、结论
experiment_protocol.json     # 训练前冻结的数据/超参/训练档位/计时规则
schedule.json                # 所有 QA ID，按 seed 冻结；分割和图像簇 hash
manifest.json                # checkpoint/source/processor/data/official revision/GPU/环境 hash
logs/                         # 每 update 和每阶段 stdout/stderr，含系统资源与报错
checkpoints/                  # 可精确恢复的完整状态，不删除历史资产
train/                        # 分臂 accepted/rejected/loss/grad/updates/teacher 成本日志
validation/                   # 真实 dev 评测明细、AR门禁、τ、轮数、配对区间
reports/FINAL_OVERNIGHT_SCIENCEQA.md
reports/RESOURCE_AND_RESUME.md
scripts/                      # 实际运行的可复现 launcher 与报告生成器
```

**自动汇总状态**：launcher 每隔若干 updates 原子写入 `RUN_STATUS.json`（如写到临时 JSON 再 rename），标注有效 step、当前 arm、checkpoint、最新监测指标、GPU状态、错误与重启次数。最终通过报告生成脚本从实际产物读数生成表格，不手填假数。如果代理交互进程结束后训练仍被支持的远端受控作业管理器托管，`RUN_STATUS` 必须由远端作业实时更新；本聊天助手不能代为监控。

## 6. 明确的停止条件与最终判断

**只允许以下硬停止：**
1. 官方公共 KL 调用链不一致、Teacher 意外产生梯度、真实 optimizer 参数/状态不更新、on-policy 合约错位、数值 NaN/Inf、输入泄漏、`SPEC≠AR`；
2. 设备显存/存储不足、GPU不空闲且不能安全运行、超过预登记 9h 总时限、极端反复 OOM（有限次数重试且不改核心语义）；
3. 发生不可恢复的 checkpoint / RNG / dataset cursor 损坏；
4. 训练前协议中已明确量化的灾难性数值退化门禁（不得在看到 OPD 负结果后追加判据）。

**不允许**：因为某臂在 step50 暂时比另一臂差就改变配方、给 OPD 更多 updates、只评有利的 checkpoint、让第二 seed 变成有利结果才启动。

最终报告必须明确给出结论标签之一：`positive / negative / inconclusive / partial / blocked`，并回答：
- 官方 KL-OPD 在这次实测训练档位下，相比同数据/同 updates 继续 SFT 的真实 τ 与轮数变化如何？是否通过无损门禁？
- 真实训练完成多少独立 train QA 与 optimizer updates、几次 teacher 调用、多少 accepted/rejected 位置、各自多少监督 positions、实际计算时间/资源？
- 是否有干净的真实速度证据？如共享 GPU 干扰，明确没有。
- 如果 OPD 没超过 SFT，不能归因到尚未测过的某个 loss 组件，**也不能把 fixed-anchor CE+full reverse 的旧负结果重新解释成官方 OPD 失败**。
- 下一步最多建议一项新的可证伪机制实验，不自动换 loss 或新数据集继续训。

**执行指令**：现在开始。先快速核对最新 AGENTS 和 L2_gpu 验收，冻结样本与可完成档位，然后直接在服务器上启动 SFT/官方 KL-OPD 成对正式训练（不要另开10-update探针）；核对真实 GPU 参数更新和可恢复状态，夜间连续工作直到完整完成或达到硬停止。确保产出晨间 `READ_ME_FIRST.md`、`RUN_STATUS.json` 和真实结果，最后按实际完成情况更新 `AGENTS.md` 并提供报告/日志路径。不得声称未完成的实验已经完成。
