# GPU6 视频草稿的论文模块迁移

这次比较回答一个问题：在同一个 Qwen3-VL-8B + DFlash 并行草稿和 OPD 训练目标下，哪种已有视觉接口能提高长事件描述的每轮接受长度？2026-11-18 撤下的 48 项手工特征组合没有启动。旧队列及已完成的独立实验仍保存在原目录。

## 模块与迁移边界

所有新臂复用 target 正常 prefill 时的 Qwen3-VL vision merger 输出，以 request identity 绑定视频。位置留给 Qwen 实际的时间 patch 分组。普通 DFlash 主路继续读已提交上下文的五层 target hidden。论文模块按本机头训练，实验比较其在 DFlash 上的实现。

| 臂 | 论文操作 | 当前实现的选择 |
|---|---|---|
| ViSpec | 每组学习一个压缩 token，跨组聚合全局特征并注入每个文本位置。[论文附录 B.2](https://arxiv.org/html/2509.15235v5#A2.SS2) | 两个可学习查询分别产出组 token 与全局分量。压缩前缀在第一层草稿注意力中与原序列一起做 softmax，全局向量广播到块内位置。原论文使用单层树状草稿。 |
| FOVEA | 草稿状态与深度查询视觉记忆，按累计概率选有界子集，再以门控残差校正。[算法 1](https://arxiv.org/html/2608.22883v1) | 并行块的位置对应树深度，质量阈值 0.8、32–256 条，校正在最终 RMSNorm 前。 |
| Flamingo | Perceiver Resampler 与 gated cross-attention / FF 残差。[论文](https://arxiv.org/abs/2204.14198) | 一个窄的 resampler 层和 8 个 latent，在草稿入口注入，每个请求只有一个视频。 |
| ToMe | 二分匹配相似 token，按成员数加权合并。[论文](https://arxiv.org/abs/2210.09461) | 在每个时间组至多 4×4 的空间网格上按 projector 特征余弦相似度做匹配，再压至至多 128 个 token 接入共同读出。 |
| LLaVA-OneVision | 对每个时间组的视觉特征做步长 2 的双线性空间降采样。[附录 C.1](https://arxiv.org/html/2408.03326v2#A3.SS1) | 沿用 Qwen 时间组和 DFlash 读出，再施加与 projector_grid 相同的 128-token 网格容量。 |
| projector_grid | 同源对照 | 将相同 merger 输出直接平均池化，以相同草稿入口读取。与此前第 33 层网格、运动和文本读出分别记录。 |

旧版 ViSpec 风格移植曾用 16 个 latent 重建所有视觉 KV。历史视频报告记录 29 个配对样本全部下降；本次逐组学习压缩前缀与当时的重建路径不同。[历史评测记录](../../repro/evaluation/vispec_report.py)

## 排队与判读

teleai_gpu6 的 tmux 为 `visionspec-paper-transfer`。运行根目录是 `/czsun/mcy/video_opd_pilot/local_h100_20261112/paper_transfer_gpu6_20261119`。精确命令、协议、运行状态和配对结果分别在 SCREEN_COMMANDS.json、QUEUE_PROTOCOL.json、STATUS.json、RESULTS.json。截止北京时间 2026-11-19 09:00。

- 使用同一训练起点、seed 20261118、相同 80 条 QA 的日程。每臂 20 次更新，4 QA/update，64 帧，OPD λ=10。筛选训练共 120 次更新。
- 128 帧 Video-MME 衍生的事件描述 dev40，最多生成 1024 token，fp32 target，b63/k8。每臂分两个不重叠的片，每条都比对独立 AR 的完整输出。以逐视频平均 τ 筛选，同时保存有效 token/轮和配对区间。共享负载计时只作描述。
- 在完整无损且输入与 AR token 都匹配的臂中，最高 τ 须同时超过同源 projector_grid 和已有最强后端文本头，才将赢家和该文本头各追加 60 步。两头都从自己的 step20 加载权重、重置 Adam、使用同一后续日程。总新增上限 240 步。每个训练阶段只存终点，同一开发集用于选优。
- 接入点、参数量与实际读入的视觉 token 数可因模块而异；结果解释为整种实现的差异。前端接入的旧文本对照给出位置匹配的参考，后端文本对照给出最强已有成绩。若仅 τ 上升，需另外同卡实测才能主张端到端速度提升。

## 首次执行与终点保存修复

北京时间 11 月 18 日 23:14，六臂的训练日志都已记录 20/20 次 optimizer 更新。终点身份检查在写权重前拒绝了初始化清单。清单将直接父权重的 SHA 写成字符串，保存器需要公共 backbone 的文件名到 SHA256 的映射，同时要求记录基础张量已逐项核对。六臂只留下初始化和训练日志，12 个 dev40 分片均未执行。旧调度器的 FINISHED 表示队列已收口，不能解释为模块未超过对照。

修复使用单独运行目录 `/czsun/mcy/video_opd_pilot/local_h100_20261112/paper_transfer_recovery_20261119`，原失败目录保留。六个头都核对了相同的 58 个主干张量。直接初始化和公共主干的权重 SHA 分开记录，保存身份在启动训练前验证。原定可选追加训练的 120 次更新改用于同一 seed、同一批次顺序、同一起点的六臂各 20 步重跑，两次合计上限仍为 240 次，后续仅评测。实时状态读恢复目录的 STATUS.json。

同期完成的旧架构 128 帧 dev40 中，后端运动差分 τ=7.5330，后端多尺度 τ=7.5336，后端文本 τ=7.5350，均为 40/40 逐 token 对齐。前两者相对文本的配对差分别为 −0.0020、−0.0015，20,000 次逐视频 bootstrap 的 95% 区间分别为 [−0.0191,+0.0149]、[−0.0135,+0.0107]。

## 到北京时间 09:00 的后续比较

使用 GPU6 的八张 H100，调度在恢复队列释放设备后开始。主评测继续用相同的 128 帧事件时间线 dev40，以逐 token 无损为前提比较接受长度。所有训练保持同一个 seed 与 λ=10，模拟实际部署的 OPD。

| 阶段 | 比较 | 新增更新上限 |
|---|---|---:|
| A | 五个论文模块、视觉网格、前端和后端文本共八臂，从各自已核验的 step20 出发。每臂用 240 个尚未训练的视频做 60 步，输入 64 帧。 | 480 |
| B | 完整评测中排名前两名的论文模块、视觉网格、后端文本共四个头。每个从自己的 A 终点分出 64/128 帧两臂，按相同顺序训练同一批 320 个视频，共 80 步。 | 640 |
| C | 以相同 64/128 帧配对继续 80 步，衡量更长训练后的排名。剩余时间给已完成的模型安排 256 帧固定空间分辨率摘要迁移与视觉旁路对照。 | 640 |

每段训练从父头的已验证权重开始，使用新的 Adam 状态。每个完成的训练段只保存一个可推理的终点及 SHA256 清单。新的训练数据统一使用事件时间线提示词，640 份 64/128 帧编码均经过实际 Qwen3-VL processor 验证。256 帧迁移集使用有界摘要提示词，分开报告。以相同输入及独立 AR 序列做逐视频配对，报告两类对照与区间。后续阶段与分片按时间预算入场，截止时记录已完成的比较和缺失的对照。

线上协议及结果位于 `/czsun/mcy/video_opd_pilot/local_h100_20261112/paper_transfer_to0900_20261119`。

## 11 月 19 日主评测结果

A、B、C 共 24 个训练任务完成 1,760 次更新，每段保存一个仅含推理权重的终点。24 个头在同一 128 帧 dev40 上完成了 960 次 SPEC 与独立 AR 的逐 token 对比，全部一致。主评测每条输入 6,064–6,080 token，其中 5,376 个是视觉 token。40 条基准输出全部达到 1,024 token 上限，观察范围内没有自然 EOS。以下 τ 是逐视频平均的每轮提交 token 数。

阶段 A 的文本对照 τ 为 7.5240。五个迁移模块中 FOVEA 为 7.5160，Flamingo 为 7.5069，两者进入 B/C。LLaVA-OneVision 池化、ToMe 和 ViSpec 在同一阶段分别为 7.4917、7.4876、7.4859。最初 20 步筛选领先的 ViSpec 在追加相同训练后没有保持排名。

| 草稿接口 | B 训练 64 帧 | B 训练 128 帧 | C 训练 64 帧 | C 训练 128 帧 | C 的帧数效应及配对 95% 区间 |
|---|---:|---:|---:|---:|---:|
| 简单视觉网格 | 7.5753 | 7.7646 | 7.5744 | **7.7923** | +0.2178 [0.1791, 0.2572] |
| 后端文本读出 | 7.5886 | 7.7217 | 7.6271 | 7.7851 | +0.1580 [0.1127, 0.2053] |
| FOVEA | 7.5812 | 7.7246 | **7.6427** | 7.7739 | +0.1312 [0.0828, 0.1807] |
| Flamingo | 7.5637 | 7.7053 | 7.6096 | 7.7522 | +0.1425 [0.0968, 0.1899] |

B、C 的 128 帧训练组均由简单视觉网格取得最高点估计。它相对文本的 B 差值为 +0.0429，九项比较校正后的 95% 区间为 [−0.0114, +0.0973]。C 差值收窄至 +0.0071，逐视频配对区间为 [−0.0203, +0.0348]。C 的 64 帧 FOVEA 比视觉网格高 +0.0682 [0.0250, 0.1109]，比文本高 +0.0156 [−0.0092, 0.0396]。同一开发集参与阶段选择，单 seed 的视频 bootstrap 只刻画该集合上的逐视频变异。

若后续只保留一个视觉接口用于长输出，优先选择简单视觉网格。它复用目标模型正常 prefill 产生的 vision merger 输出，按时空网格池化，再在草稿入口读出。保留后端文本作为配对基线。此次各组的 OPD 都固定 λ=10，视觉接口与新多模态 OPD 目标的协同仍需单独比较。

核验使用运行目录内的 `STATUS.json`、`RESULTS.json` 与 `audit/PAIRED_ARCHITECTURE_EFFECTS.json`。加速计时跨卡和时段复用共享负载条件，主表依据 τ 排序。

## 256 帧长输入迁移

同一批 dev40 视频使用固定空间分辨率及有界摘要提示词，每条输入 12,034–12,065 token，其中 10,752 个是视觉 token。独立 AR 平均输出 232.4 token，40/40 条自然结束。阶段 C 的八个头合计 320/320 条 SPEC 逐 token 无损。

| 草稿接口 | C 训练 64 帧 | C 训练 128 帧 |
|---|---:|---:|
| 简单视觉网格 | 2.7504 | 2.8504 |
| 后端文本读出 | 2.7746 | 2.8674 |
| FOVEA | **2.7892** | **2.8835** |
| Flamingo | 2.7519 | 2.8348 |

128 帧训练的 FOVEA 相对后端文本多提交 +0.0161 token/轮，逐视频配对 95% 区间 [+0.0046, +0.0275]。它相对简单网格的差值为 +0.0331 [0.0115, 0.0544]。长输入摘要可保留 FOVEA 作为候选。256 帧的同提示词长输出只完成了输入编码，尚无长输入与长输出同时成立时的逐 token 评测。

## 08:36 按用户要求收尾

北京时间 2026-11-19 08:36:35，GPU6 实验队列转为 `STOPPED`，调度 tmux、实验子进程与 GPU 计算任务均已退出。主队列完成 24 个训练任务和 269 个评测分片，共 960 次主评测、865 次长输入摘要逐 token 一致。A 的 19 个摘要分片和 C 的 320 个视觉旁路分片在停止时未完成。视觉旁路没有产生可用于归因的结果。

运行根目录下保留以下终点，各目录有推理权重、配置、训练身份与 SHA256 清单。

| 用途 | checkpoint 相对路径 |
|---|---|
| 长输出视觉候选 | `stages/C/projector_grid/f128/run/step80` |
| 配对文本基线 | `stages/C/output_B_text/f128/run/step80` |
| 长输入摘要视觉候选 | `stages/C/fovea/f128/run/step80` |

全部 24 个训练终点均保留。最终 `DONE.json` 写入完成及停止明细，配对审计重新核验了 45 个完整评测组和 106 项配对比较。

## 另一台机器的 FOVEA × 视觉 OPD 交接方案

目标是判断加入 FOVEA 的草稿是否受益于视觉适配 OPD，并估计两者结合后比各自单独使用多带来多少收益。先固定一套另一台机器已有的视觉 OPD 实现，记录代码提交和完整参数。用本轮的通用 OPD（`e2e_tv_weight=10`）作为 O0。O1 在 O0 上接入选定的视觉适配，记录它修改的损失项、视觉信号来源和归一化方式。视觉权重或 teacher 信号须能在两种草稿上按同一规则计算。如果现有 O1 依赖 FOVEA 专属读出，须先为纯 DFlash 接入同源、同定义的 teacher 侧信号，才能解释下表的交互。

| 组 | 草稿结构 | 训练 |
|---|---|---|
| A0 | 纯 DFlash | O0 通用 OPD |
| A1 | 纯 DFlash | O1 视觉适配 OPD |
| F0 | DFlash + FOVEA | O0 通用 OPD |
| F1 | DFlash + FOVEA | O1 视觉适配 OPD |

**起点和迁移文件**

- 代码用本交接所在分支 `run/gpu6-paper-until0900-20261119`，FOVEA 实现在 `scripts/dflash_pkg/dflash-0.1.0/dflash/paper_modules.py` 的 `FoveaReadout`。在目标机把 O1 接到同一训练入口，核对 import 的 DFlash 包来自这个 checkout。可在目标机用 `git clone -b run/gpu6-paper-until0900-20261119 visionspec-fovea-opd-handoff.bundle VisionSpec-fovea-opd` 导入代码及本报告。
- 以下文件相对于 `/czsun/mcy/video_opd_pilot/local_h100_20261112`。A0/A1 共用 `jobs/night13_A_plain_lambda10_u80/run/step80` 的 `config.json`、`model.safetensors`。F0/F1 共用 `paper_transfer_recovery_20261119/fovea/init` 的 `config.json`、`model.safetensors`、`INIT_MANIFEST.json`。FOVEA 初始头已核验 58 个公共 DFlash 主干张量与纯头逐项相同，输出修正初始为零。只迁移这两个起点及身份文件，每个约 2 GiB。两份权重 SHA256 分别为 `79bb91f08301b8dd36ea3b4bb2484ec2914e65085b85dedc8345f6a53df95e76` 和 `3cd2966bed3826f6821e692d18350d0edd5aef1a71e7d023e7ce7383ad2f0354`。训练使用 fresh Adam。
- Target 用同版本 Qwen3-VL-8B-Instruct。训练用 `paper_transfer_to0900_20261119/data/train320_128.parquet` 和 `paper_transfer_to0900_20261119/data/schedule_B.jsonl`。评测用 `data/full/dev_128.{parquet,jsonl}` 和 `data/dev40_long_input_fixed_spatial/dev_256.{parquet,jsonl}`，256 帧同时带上该目录的 `PROCESSOR_PROFILE.json`。Parquet 只记录视频与帧缓存索引。同步 `cache_path` 对应的 NPZ 和必要的 128 帧父缓存。用其它挂载路径时迁移索引中的路径，保留原视频 ID、缓存哈希、prompt 和真实时间戳，先核对目标机的输入 token 与视觉 token 数。

**运行顺序**

1. 对四组跑 CPU dry-run，比对固定的训练池、日程、精度和训练配置。用 1–2 个 GPU update 做门禁，确认 O1 的视觉权重实际进入 loss、参数有更新、每组都能重新载入终点，随后从原始初始化正式开跑。OPD on-policy rollout 可以随模型不同而变化。
2. 正式四组各 80 更新、4 QA/update、同一 seed 20261118、同一顺序、真实 128 帧、`lr=1e-5`、GPU fp32 master、每组只保存最终一步。四张空闲 GPU 可并发，每组一张，实际并发按显存门禁决定。纯 DFlash 用 `repro/local_h100/train_video_opd_gpu_memory.py --evidence-mode plain --weights-only-final`，FOVEA 用 `repro/local_h100/train_video_evidence_opd_weights_only.py --evidence-mode visual`。两种 O0 命令均已用本机选定的模型与训练池通过 CPU dry-run。每个进程用 `CUDA_VISIBLE_DEVICES` 指定一张卡的 GPU UUID，输出写到本机已有训练入口支持的挂载盘路径。共同参数是 `--master-weights gpu --device-map balanced --updates 80 --save-every 80 --qa-per-update 4 --seed 20261118 --lr 1e-5 --max-new-tokens 1024 --max-seq-len 16384 --time-mode real --loss-normalization global_pooled --e2e-tv-weight 10`。另传本机的 `--pool`、`--schedule`、`--target`、`--init-from`、`--out` 与 O1 实际启用参数。
3. 两套评测分别汇报。128 帧时间线用 `repro/evaluation/eval_video_opd.py`，256 帧摘要用 `repro/local_h100/eval_long_input.py` 加冻结的 processor profile。两套统一 fp32 target、b63/k8、`--ar --require-exact-time --time-mode real --eos-stop immediate --max-new 1024`。256 帧摘要是 FOVEA 的主要验证场景，128 帧长输出用于检查迁移和回退。每组先验收 40/40 token 序列严格等于独立 AR，再按视频配对计算 τ。达到上限和自然结束分别记数。
4. 先看 **F1−A1**（同用视觉 OPD 时加 FOVEA）、**F1−F0**（FOVEA 上加视觉 OPD）、**(F1−F0)−(A1−A0)**（协同效应）。按视频做配对 bootstrap，并记录四组完整数值。若组合提高 τ 而交互区间跨零，可以报告组合表现，协同仍未确认。若组合相对两种单独做法都稳定提升且交互为正，再用未参与筛选的视频确认，并做同卡交替墙钟测速。获得同卡速度之前用每轮提交 token 衡量接受效率。

交接止于实验方案和已有代码。O1 的接口与参数以目标机已验证的实现为准，四组新训练尚未启动。
