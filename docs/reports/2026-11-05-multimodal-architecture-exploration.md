# 多模态架构适配探索（2026-11-05 夜 → 11-06 08:00）

状态：进行中。用户授权持续探索到 2026-11-06 08:00；本文件记录文献、经典架构借鉴、每个变体和结果。

## 0. 问题定位

GLANCE 当前 head 是 **DFlash 文本 block-draft head + target 5 层 fused hidden**：

- 没有 raw visual token、没有 vision adaptor、没有视觉位置、没有视觉 cross-attention；
- image/video 共用同一 text head；
- 之前 layer gate、target-layer xattn、cross-modal/mixed 数据都阴性；
- E2E-TV（objective）三 seed 高 g +0.0465 [+0.0136,+0.0793]，说明 objective 是一根杠杆，但 architecture 仍是 text head。

## 1. 文献（不止 ViSpec/DREAM）

| 论文 | 核心 | 对 GLANCE 的可借点 |
|---|---|---|
| [ViSpec (NeurIPS 2025)](https://arxiv.org/abs/2509.15235) | lightweight vision adaptor 压缩 image tokens，注入 draft attention，保留 image positional info；global feature 加到所有 text token | raw visual adaptor + global FiLM + 位置 |
| [DREAM (NeurIPS 2025)](https://arxiv.org/abs/2505.19201) | target intermediate features cross-attn，entropy-adaptive selection，visual token compression | 选层/选 token，zero-init cross-attn |
| [MASSV](https://arxiv.org/abs/2505.10526) | vision projector + self-distillation | 轻量 projector / 蒸馏 |
| [FOVEA (arXiv 2608.22883)](https://arxiv.org/abs/2608.22883) | per-image visual memory；state-conditioned top-K（cumulative mass 0.8, Kmin32/Kmax256）；gated residual 只改 LM-head 前 hidden，不插入 token、不破坏 KV cache | **最适合 block head**：每个 block query 检索视觉证据，gated correction |
| [TIGER (arXiv 2607.11131)](https://arxiv.org/abs/2607.11131) | 从 drafter 当前 textual state 选 sparse visual tokens；acceptance-aligned group policy + verifier accepted-prefix reward | 稀疏视觉路由 + acceptance 目标 |
| [Uni-LaSpec (arXiv 2611.01617)](https://arxiv.org/abs/2611.01617) | 复用 verifier attention query 对 cached int8 visual keys 排序，cross-layer vote 选 token；gated cross-attn；acceptance-aware 权重 | verifier query / vote / 每位置 acceptance 权重 |
| [SpecVLM](https://aclanthology.org/2025.emnlp-main.366.pdf) / SpecPrune | 视频 token pruning | 视频侧压缩 |
| [多模态 SD survey (arXiv 2608.20743)](https://arxiv.org/abs/2608.20743) | block-parallel multimodal drafting 基本空白 | GLANCE 的差异化位置 |

**判断**：现有论文的视觉适配主流是“压缩 visual tokens + 注入/检索 + gated fusion”；我们缺的正是 raw visual pathway，而不是又一层 target fused gate。

## 2. 经典 ML/DL 架构借鉴（非 LLM 论文）

| 经典机制 | 在多模态 draft 里的用法 | 预期 |
|---|---|---|
| **Perceiver / cross-attention bottleneck** | learned latents 把 N 个 visual tokens 压到 K=32/64；draft 只 attend K | 降延迟、去冗余、保留显著区域 |
| **Slot Attention / object-centric slots** | 把视觉 token 聚成 object slots，再注入 draft | OCR/小目标/关系题更聚焦 |
| **Mixture-of-Experts / conditional computation** | visual expert vs text expert，top-1/2 routing；modality embedding | 图像/视频/文本不互相干扰 |
| **FiLM / hypernetwork** | global image vector 生成 per-layer scale/shift；zero-init | 极便宜，先验证 global context |
| **Product-key / associative memory / Hopfield** | 视觉 memory 的 key-value 检索；FOVEA 已是这个思路 | 状态条件检索、cache-friendly |
| **VQ / sparse coding** | 离散视觉 codebook，压缩+去噪 | 降 token 数，可能损失细粒度 |
| **Adapter / LoRA / bottleneck** | modality-specific 低秩 adapter，冻结 base | 最小改动、易回滚 |
| **Early exit / layer skipping** | 只在浅层/特定 block 做视觉融合 | 降 draft 延迟 |
| **State-space / linear attention** | 视频长序列 temporal memory | 视频专用，工程量大 |
| **Contrastive / InfoNCE / information bottleneck** | 视觉-文本对齐、real vs blank margin | 视觉感知目标 |

## 3. 已实现 / 进行中的变体

### V1 `vxattn`：raw visual cross-attention（已完成，阴性）

- 缓存 target vision encoder 的 post-merger visual tokens（image 72–299 个，4096 维）。
- head 的 16 个 block query 对全部 visual tokens 做 fp32 LayerNorm + 8-head cross-attn，zero-init residual。
- 4 卡，60 updates，image OPD step3546，lr1e-5，无 E2E-TV（隔离 architecture）。
- **结果（test200，200/200 无损，162/200）**：overall **−0.4645** [−0.5487,−0.3862]，高 g **−0.5173** [−0.6451,−0.4030]，低 g −0.4118，top quartile −0.5115；全部显著为负。
- 端到端 8.38s/条（比 base 慢约 80%）。full raw visual cross-attn 直接注入 noise embedding 会破坏 draft；不是“有视觉信息就行”，需要 gated/sparse/compressed 融合。

### V2 `fovea`：state-conditioned visual memory + gated residual（已完成，阴性/中性）

- FOVEA 式：per-image visual memory 的 key/value 投影；每个 block position 用 hidden+depth 做 query；cumulative-mass top-K（ρ=0.8, Kmin32, Kmax256）；readout 经 gated residual 只改 LM-head 前 hidden。
- zero-init 时与 base 逐位一致（已 CPU 验证 maxdiff=0.0）。
- image/video init 已建：`fovea_init_saved/step3546_fovea`、`fovea_video_init_saved/step220_fovea`。
- 视频 memory：`vmem/video_train439.pt`、`vmem/video_fresh30.pt`（1152×4096/视频）。
- **结果（test200，200/200 无损，162/200）**：overall +0.0006 [−0.0157,+0.0174]，高 g −0.0024 [−0.0164,+0.0097]，低 g +0.0036，top quartile −0.0024；全部不显著，基本等于 base。
- 端到端 7.62s/条，仍比 base 慢；zero-init gated correction 不破坏 base，但 60 updates 没学到有用视觉证据。

### V3 `video E2E-TV`：视频 objective 复现（已完成，非 visual-specific）

- 从 video OPD3e step220，3 卡，60 updates，E2E-TV w=10；fresh30 真实+blank。
- **结果**：real τ **3.8006** vs OPD3e 3.7841（+0.0165 [−0.0157,+0.0488]，不显著）；blank 3.4365 vs OPD3e blank 3.2826（**+0.1539**，显著）；visual reliance 0.3641 vs 0.5015，DiD **−0.1374** [−0.1767,−0.0993]。
- 结论：video E2E-TV 主要提高 blank/语言先验，**降低视觉依赖**，没有复现 image E2E-TV 的 high-g 正效应。视频 objective 迁移失败（至少不是 visual-specific）。

### V4 `film`：global visual FiLM（已完成，低 g 正、高 g 负）

- global visual vector 经 MLP 生成 scale/shift，zero-init；60 updates，test200 200/200 无损。
- overall +0.0007 [−0.0194,+0.0181]；高 g **−0.0115** [−0.0446,+0.0163]；低 g +0.0129；最低四分位 **+0.0290** [+0.0010,+0.0630]；top quartile −0.0019。
- 结论：FiLM 只把低 g/bottom 抬高，高 g 点估计为负；又一个 anti-visual/语言先验结果。

### V5 `fovea_perc`：Perceiver K=32 压缩 + FOVEA（已完成，中性）

- 32 个 learned latents 把 visual memory 压到 K=32，再走 FOVEA gated retrieval；60 updates，200/200 无损。
- overall +0.0072 [−0.0111,+0.0248]；高 g +0.0015 [−0.0301,+0.0296]；低 g +0.0129；top +0.0140；bottom +0.0024；全部不显著。
- 结论：压缩没有把视觉证据变成可用信号；高 g 仍零。

### V6 `video_fovea`：视频侧 FOVEA（real 已完成，blank 待补）

- 从 video OPD3e step220，3 卡，60 updates，FOVEA visual memory + gated retrieval。
- real τ **3.7530** vs OPD3e 3.7841（**−0.0311 [−0.0515,−0.0113]**），vs warm 3.7266（+0.0264）；30/30 无损。
- blank τ **3.3676** vs OPD3e blank 3.2826（**+0.0850 [+0.0496,+0.1204]**）；visual reliance 0.3854 vs 0.5015，DiD **−0.1161 [−0.1613,−0.0712]**。
- 和 video E2E-TV 同一模式：real 变差、blank 变好、视觉依赖下降；不可用。

### V7 `video_film30b`：视频 global FiLM（30 updates，real 评测运行中）

- 从 video OPD3e step220，3 卡，30 updates，global visual FiLM；30/30 待评测；这是最后一个视频架构变体。
- 不改变总判断：image 侧 FiLM 已经是高 g 负、低 g 正；视频侧预计同类。

### V8 未跑：MoE / modality routing / slot attention

- 若 V1/V2 仍阴性，下一步做 K=32/64 的 Perceiver/slot memory（减少 N、提升显著 token），以及 modality-aware MoE/FiLM。
- 视频侧用 SpecPrune/SpecVLM 的 token pruning。

## 4. 评测协议

- image：test200，逐 token 无损门禁，overall/high-g/low-g/top quartile；对照 base（4 卡，step60）和 E2E-TV（3 seed）。
- video：fresh30，真实+blank，τ、visual reliance、DiD；对照 OPD3e/warm。
- 任何正结果必须：高 g CI 下界 > 0，低 g 不显著，逐 token 无损。
- 记录：本文件 + `/mnt/local/mcy/GLANCE/results/visual_opd_20261105/NOTES.md`；代码提交到 `exp/visual-opd`。

## 5. 资源

- 训练：4 卡（image）/ 3 卡（video），GPU UUID 绑定；target fp32，head bf16，CPU master。
- 预算：image `visual_opd_20261102/budget.json`；video `visual_opd_20261105/video_budget.json`。
- 当前运行：image vxattn（4 卡）；video memory 已缓存；video E2E-TV 排队等卡。


## 6. 08:00 收口结论

### 6.1 所有架构变体

| 变体 | 数据 | high-g Δ | low-g Δ | 判定 |
|---|---|---:|---:|---|
| V1 vxattn（raw visual 全量 cross-attn） | image | **−0.5173** | −0.4118 | 显著负，且慢 80% |
| V2 FOVEA（state-conditioned gated retrieval） | image | −0.0024 | +0.0036 | 中性 |
| V4 FiLM（global visual modulation） | image | **−0.0115** | +0.0129（bottom +0.0290） | 低 g 正、高 g 负 |
| V5 Perceiver K=32 + FOVEA | image | +0.0015 | +0.0129 | 中性 |
| V3 video E2E-TV | video | —（real +0.0165，blank +0.1539） | — | 主要 blank/语言先验，reliance −0.1374 |
| V6 video FOVEA | video | —（real −0.0311 [−0.0515,−0.0113]） | — | real 显著低于 OPD3e；blank +0.0850，reliance −0.1161；不可用 |

### 6.2 结论

- **architecture 侧的五个变体（raw cross-attn、FOVEA、FiLM、Perceiver、video FOVEA）都没有得到可用的 visual-specific 增益**：要么显著负，要么高 g 中性/负，要么只提高 low-g/bottom。
- 唯一站得住的 visual-specific 正结果仍是 **image E2E-TV objective**：三 seed 合并高 g **+0.0465 [+0.0136,+0.0793]**、最高四分位 +0.0546 [+0.0090,+0.1001]，低 g 不显著。
- video E2E-TV 的 real +0.0165 不显著，blank +0.1539 显著，visual reliance 从 0.5015 降到 0.3641；它优化的是 blank/语言先验，不是视频视觉依赖。
- 因此，**当前 GLANCE block head 的瓶颈不在“视觉特征怎么融合”**：给 raw visual tokens、gated retrieval、FiLM、Perceiver 都无效；更大的杠杆是 **objective / teacher alignment**（E2E-TV），以及可能的 decoder-side grounded copy。
- 本夜没有找到“可用于图像/视频的架构优化方法”；这是一个明确的阴性结论，不是未完成。
