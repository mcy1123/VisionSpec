# 多模态特征/OPD 训练优化夜跑（2026-11-05）

状态：已完成（2026-11-05 08:30 前）。A（xattn）工程失败；B（image layer gate）阴性；C（cross-modal）阴性；D（mixed image+video）overall/low-g 正但 high-g 负；E（video layer gate）显著降低 visual reliance。每一步的设计、失败与结果都记录在
`/mnt/local/mcy/GLANCE/results/visual_opd_20261105/NOTES.md`。

## 0. 背景与文献

图像侧六种输入干预（位置加权、cutout、mismatch、2×2/3×3 clue、target-attention 权重）和视频 blank 对照都表明：当前 OPD 的增益不是 visual-specific 的。这一轮换目标：**改架构/特征，而不是再改输入侧 loss 权重**。

检索到的可借鉴工作：

- [DREAM: Drafting with Refined Target Features and Entropy-Adaptive Cross-Attention Fusion for Multimodal Speculative Decoding](https://arxiv.org/abs/2505.19201)：target intermediate features 的 cross-attention 注入、attention entropy 选层、visual token compression。
- [MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of VLMs](https://arxiv.org/abs/2505.10526)：轻量 vision projector + target 自蒸馏视觉指令微调。
- [SPECVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning](https://aclanthology.org/2025.emnlp-main.366.pdf)：视频侧 verifier-guided token pruning（本轮未实现）。

本仓库 head 已经吃 target 的 5 层 fused hidden，所以不是“没有视觉输入”；这轮尝试在 head 内部做 **target 多层特征的可学习融合**（DREAM feature refinement 的最小版本），以及 **cross-modal OPD**（用视频数据优化 image head 的多模态特征）。

## 1. 实验 A：DREAM-style target-feature cross-attention（失败）

设计：把 fused target_hidden 拆成 5×4096，用 `nn.MultiheadAttention` 从 block hidden cross-attend 到 5×ctx target 层特征；`xattn_out` zero-init，初始行为应等于 base。

失败链：

1. 多 rank 各自随机初始化新模块 → `Head policy version/hash disagreement`；改用保存好的完整 init checkpoint。
2. `target_hidden` last dim=20480，误按 `[B,ctx,5,20480]` 拆；修正 `layer_dim=hidden//5`。
3. 修复后 `gradient must be finite`；加 LayerNorm/query LayerNorm/scale=0.1 仍 non-finite。
4. xattn 分支改 fp32 后出现 `mat1/mat2 dtype float vs bf16`。

结论：这条 cross-attention 在现有 bf16 head + 4 rank + CPU master 组合下集成成本过高；**不作为主实验**，代码默认关闭（`use_xattn` 默认 false），失败目录保留。

## 2. 实验 B：layer-wise target feature gate（阴性）

设计：

- 把 fused target_hidden 拆成 5×4096；
- `gates = 1 + MLP(mean_layers)`，MLP 4096→128→5，最后一层 zero-init；
- `target_hidden = layers * gates`，再走原 `fc/hidden_norm`；
- 初始 gates=1，初始 forward 与 base 逐位一致；只多约 0.5M 参数。

训练：image OPD step3546、60 updates、4 卡、同 schedule/seed/lr。update1 loss=2.6934439972459017，与 base update1 完全相同（zero-init 生效）。

test200：200/200 无损，162/200 正确。

| 分组 | base τ | lgate τ | Δ | 95% CI |
|---|---:|---:|---:|---:|
| overall | 6.2305 | 6.2330 | +0.0025 | [−0.0147,+0.0195] |
| 高 g | 6.1422 | 6.1382 | −0.0040 | [−0.0274,+0.0156] |
| 低 g | 6.3187 | 6.3278 | +0.0091 | [−0.0184,+0.0374] |
| 最高四分位 | 5.4706 | 5.4838 | +0.0132 | [0.0000,+0.0305] |
| 最低四分位 | 5.6253 | 5.6073 | −0.0180 | [−0.0608,+0.0125] |

结论：**可学习的 5 层 target feature gate 没有 high-g 增益**；整体也近零。DREAM-style feature refinement 在这个 head 上没有转化为 visual-specific 收益。

## 3. 实验 C：cross-modal OPD（image head × video data；运行中）

目的：既然输入侧和特征 gate 都不行，测试“用视频数据作为多模态正则”能否改善 image head 的视觉特征。

- init：image OPD step3546；
- 数据：VideoDetailCaption 前 60 updates（240 QA）；
- 3 卡/4 逻辑 slot；同 lr/seed；`--time-mode legacy`；
- 产物：`results/visual_opd_20261105/xmodal3/`；
- 计划：test200 + video fresh30 对照；若 image 高 g 有改善，再补 3 卡 image clean control 排除拓扑混杂；若阴性，转 mixed image+video。

结果：

- image test200：200/200 无损，162/200 正确；overall −0.0009 [−0.0371,+0.0354]，高 g **−0.0154** [−0.0648,+0.0330]，低 g +0.0136 [−0.0429,+0.0678]，最高四分位 +0.0105，最低四分位 +0.0206。
- video fresh30：τ **3.4517**（30/30 无损，30/30 hit EOS），低于 warm 3.7266 和 OPD3e 3.7841。
- 结论：视频数据 60 updates 既没有改善 image head，也把 video τ 拉低；cross-modal 作为正则不成立。

## 4. 实验 D：mixed image+video OPD（准备中）

- 合并 image/video pool（新增 `modality` 列），60 updates × 2 image + 2 video；
- `adapters.encode` 支持 per-row modality；
- init image step3546；4 卡；输出 `mixed/run`；
- 评测 image test200（+ 可选 video fresh30）。

结果（image test200，200/200 无损，162/200 正确）：

| 分组 | base τ | mixed3 τ | Δ | 95% CI |
|---|---:|---:|---:|---:|
| overall | 6.2305 | 6.2529 | +0.0224 | [−0.0065,+0.0531] |
| 高 g | 6.1422 | 6.1363 | **−0.0059** | [−0.0368,+0.0291] |
| 低 g | 6.3187 | 6.3695 | **+0.0507** | [+0.0024,+0.0995] |
| 最高四分位 | 5.4706 | 5.4405 | −0.0301 | [−0.0721,+0.0125] |
| 最低四分位 | 5.6253 | 5.6647 | +0.0393 | [−0.0021,+0.0819] |

- mixed 数据的收益集中在**低 g**（+0.0507，CI 不跨零）；高 g −0.0059，最高四分位 −0.0301；`Spearman(g,Δ)=−0.176`。方向与视觉 hypothesis 相反。
- 结论：image+video 混合训练能提高 overall/低 g，但不是 visual-specific；它强化的是语言先验/通用解码。

## 5. 实验 E：video layer gate（阴性，visual reliance 下降）

从 video OPD3e `run_stage2/step220` 加 zero-init layer gate，60 updates，3 卡，同视频数据/seed；fresh30 真实与 blank 各评 30 条（均 30/30 无损）。

| arm | 真实 τ | blank τ | 真实−blank |
|---|---:|---:|---:|
| warm | 3.7266 | 3.0747 | 0.6519 [0.5623,0.7398] |
| OPD3e | 3.7841 | 3.2826 | 0.5015 [0.3979,0.6056] |
| vlgate | 3.7469 | 3.3671 | **0.3798** [0.2767,0.4795] |

- vlgate−OPD3e 真实 τ **−0.0372** [−0.0618,−0.0132]（显著下降）；blank τ **+0.0845** [+0.0456,+0.1240]（显著上升）；DiD **−0.1217** [−0.1694,−0.0737]。
- 也就是说 video layer gate **进一步降低了视觉依赖**：真实变差、灰视频变好。和 image layer gate 一样，没有 visual-specific 收益，方向甚至相反。

## 6. 产物

- 运行记录：`/mnt/local/mcy/GLANCE/results/visual_opd_20261105/NOTES.md`
- lgate：`lgate/step60`、`eval/lgate_step60.json`、`compare_lgate_step60.json`
- xmodal3：`xmodal3/step60`、`eval/xmodal3_step60.json`、`eval/xmodal3_video.json`
- mixed：`mixed/run3/step60`、`mixed/pool.parquet`、`mixed/schedule_mixed_u60.jsonl`、`eval/mixed3_step60.json`、`eval/mixed3_video.json`
- vlgate：`vlgate/step60`、`eval/vlgate.json`、`eval/vlgate_blank.json`
- 代码：`scripts/dflash_pkg/dflash-0.1.0/dflash/model.py` 的 `use_layer_gate`（默认 false）、`repro/opd_official_kl/adapters.py` 的 per-row modality；分支 `exp/visual-opd`。

## 7. 边界与最终判断

- 所有实验单 seed；image test200 是复用集；video fresh30 n=30。
- xattn 失败是工程集成失败，不是“cross-attention 方法本身无效”的证明；但在这套 bf16 head/runtime 上不可用。
- layer gate 的零初始化保持 base 行为；image 上没有 high-g 收益，video 上显著降低 visual reliance。
- **最终判断**：这轮换目标（DREAM/MASSV-inspired 的 target feature 融合、cross-modal/mixed 数据、video layer gate）**没有找到 visual-specific 优化**。唯一可量化的正效应是 mixed image+video 的 overall +0.0224 和低 g +0.0507，但高 g −0.0059、video τ 降到 3.416，direction 与视觉 hypothesis 相反。
- 结论：当前 head 的瓶颈不在“target 多层特征如何融合”或“输入图像/视频如何混合”；继续在输入/特征融合侧做小改动，预期仍是低 g/语言先验收益。下一步应转向 **objective**（full/top-k KL、视觉对比 margin）或 **decoder-side grounded copy**，而不是继续改 feature gate。
