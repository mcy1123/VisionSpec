# 视频高视觉 token 多模态：完整实验整理（2026-11-06 → 2026-11-11）

本文件把最近几轮（image E2E-TV → 视频 1k/3k/9k → 128f1x 正结果 → token 选择/融合）的实验、结论和产物收拢到一处。子报告仍是细节来源：
[长训练/高 token](2026-11-06-long-horizon-token-scaling.md)、
[9k target 实测](2026-11-09-video-target-9k-tokens.md)、
[5–10k 多模态特性](2026-11-10-high-token-video-multimodal.md)、
[融合/token 选择](2026-11-11-multimodal-fusion-token-selection.md)。

## 0. TL;DR

| 结论 | 证据 |
|---|---|
| **image E2E-TV 120 是干净正结果** | 两 seed 合并 overall **+0.0368 [+0.0077,+0.0659]**、high-g **+0.0690 [+0.0229,+0.1150]**、200/200 无损 |
| **1k video OPD 仍是强基线** | OPD3e **3.7841**；E2E10 3.7595；30/30 无损 |
| **单纯把 16 帧上采样到 9k（32f2x）是负结果** | 零样本 **3.4098**；训练 5/15/20/30 = 3.4790/3.4725/**3.5334**/3.5171；仍低于 1k |
| **原生 128 帧（128f1x，9216 token）是正结果** | 零样本 3.7392；E2E10 **3.8796 / 3.8740（两 seed）**；blank 2.8428，reliance **1.037** |
| **TV20 是新的最高均值** | **3.9046**，较 TV10 +0.0250 [−0.0020,+0.0524]；blank 2.7970，reliance **1.108**；单 seed、CI 刚跨零 |
| **视觉 token 减半不是免费的** | 9k head 跑 64f1x/32f1x：3.7004/3.7358，比 128f1x 低 **0.179/0.144** |
| **零初始化层融合在 10-update 预算下中性** | layer gate 3.8751（−0.0045）；xattn 3.8725（−0.0070） |
| **视觉信息大部分已进入文本上下文** | zero-visual 只掉 **0.2481 [−0.373,−0.144]**，blank 掉 **1.0367**；约 76% 视觉依赖在上下文 hidden |

## 1. 协议与输入

- target：**Qwen3-VL-8B，fp32**；9k 用 `accelerate device_map="balanced"` 跨 **3×47GB**；单卡 9k fp32 OOM。
- fresh30：30 条视频问答；`b63/k8`、贪心、`max_new=1024`；每条同时跑 AR，要求 **逐 token SPEC=AR（lossless）**。
- 视频源 cache 只有 **16 帧，384×288，30fps**；所有“更多帧”都是复制，不是新增真实时间采样（除非特别说明）。
- 输入构造：
  - **1k/16f1x**：16 帧，384×288，1152 token；
  - **3k/16f2x**：16 帧，2× 分辨率，约 3.1k token；
  - **9k/32f2x**：32 帧（每帧复制 2×）+ 2× 分辨率，`video_grid_thw=[16,48,48]`，**9216 token**；
  - **32f1x/64f1x/128f1x**：16 帧分别复制 2×/4×/8×，**保持 384×288**，token 2304/4608/**9216**。
- head：OPD3e step220 或 E2E-TV 在对应 target 上下文上训练；9k 为单进程 target 3 卡 + head cuda:0。

## 2. Image 侧（视频实验的前置正结果）

从 image OPD step3546 出发，4 卡，同 schedule 的 base 对照：

| 分组 | base120 | E2E-TV 120 | Δ [95% CI] |
|---|---:|---:|---:|
| overall（两 seed 合并） | — | — | **+0.0368 [+0.0077,+0.0659]** |
| high-g | — | — | **+0.0690 [+0.0229,+0.1150]** |
| top quartile | — | — | **+0.0847 [+0.0293,+0.1401]** |
| low-g | — | — | 不显著 |

- 单 seed 均值：base120 **6.2325**、E2E120 **6.2745**、E2E120 seed2 **6.2651**；200/200 无损，162/200 准确。
- 180 updates 饱和：base180 6.2685、E2E180 6.2723；overall/high-g/top 全部不显著（high-g +0.0436）。
- 结论：**E2E-TV 在 image 120 是同预算、visual-specific 的正结果；180 时 base 追上，不能外推。**

## 3. 视频 token 数：先负后正

### 3.1 1k / 3k / 9k32f2x（零样本 + 训练）

| target 输入 | 视觉 token | 零样本 τ | E2E-TV 训练 τ |
|---|---:|---:|---:|
| 1k/16f1x | 1152 | **3.7841** | 3.7595（10 upd） |
| 3k/16f2x | ~3153 | 3.5714 | 3.5425（5 upd，反而降） |
| 9k/32f2x | 9216 | **3.4098** | 5→3.4790；15→3.4725；20→**3.5334**；30→3.5171 |

- 零样本：1k > 3k > 9k32f2x，**token 越多越差**。
- 9k32f2x 训练能恢复一部分（3.41→3.53），但 **20 updates 后仍比 1k 低 0.251**；30 回落。
- 9k32f2x 第二 seed 20 updates **3.4201**，seed 方差大；3.5334 不能当稳定值。
- 3k 上训练 5 updates **3.5425 < 零样本 3.5714**：短训练也会伤。

### 3.2 blank / mismatch / target 内容敏感性

| arm | real τ | blank τ | reliance |
|---|---:|---:|---:|
| 1k OPD3e | 3.7841 | 3.2826 | **0.5015** |
| 9k32f2x step15 | 3.4725 | 3.0750 | 0.3974 |
| 9k32f2x step20 | 3.5334 | 3.0129 | 0.5205 |

- **blank 显著下降**，但 9k32f2x 的 real/blank 绝对值都更低；reliance 可以更高，说明“看了视觉但融合质量更差”。
- mismatch：1k real=**3.7841**，mismatch=**3.7841**；9k32 step15 real=mismatch=**3.4725**（30/30 无损）。逐题 τ 不同，但**平均接受长度不奖励正确视频**。
- target first-token 敏感性（real vs mismatch）：

| target | KL | hidden cos | top1 一致 |
|---|---:|---:|---:|
| 1k | **0.6187** | 0.9697 | **3/5** |
| 9k32f2x | 0.3667 | 0.9741 | 5/5 |

- 9k32f2x 对“换视频”更不敏感 → **上采样/复制稀释了视觉特异性**。
- draft-side memory（FOVEA）9k：real **3.7485**，blank 3.3676，reliance 0.3809；比 OPD3e 3.7841 低 0.036，**加 memory 不救 9k**。

## 4. 128f1x：目前最稳的高 token 正结果

### 4.1 构造

- 16 个 unique 帧，每帧复制 8× → **128 帧**；分辨率保持 **384×288**；`video_grid_thw=[64,24,24]`，**9216 token**。
- 和 32f2x 同为 9216 token，但**没有空间上采样**。

### 4.2 结果

| arm | τ | lossless |
|---|---:|---:|
| 128f1x 零样本 OPD3e | 3.7392 | 30/30 |
| 128f1x E2E-TV 10 upd（seed 20261120） | **3.8796** | 30/30 |
| 128f1x E2E-TV 10 upd（seed 20261122） | **3.8740** | 30/30 |
| 128f1x blank（seed1） | 2.8428 | 30/30 |
| **128f1x TV20（seed 20261127）** | **3.9046** | 30/30 |
| TV20 blank | 2.7970 | 30/30 |

- 128f1x 零样本比 32f2x 高 **+0.3294**；训练 10 upd 比 32f2x 最好（3.5334）高 **+0.3462**。
- 两 seed 几乎相同（3.8796/3.8740）；对照 1k E2E10 3.7595：**+0.120/+0.115**；对照 1k OPD3e 3.7841：**+0.095/+0.090**。
- 但 30 条配对 bootstrap CI 宽（约 [−0.21,+0.55]），**不能说统计显著**；两 seed 复现的是方向和幅度。
- blank reliance：TV10 **1.0367**；TV20 **1.1075**（blank 2.7970），比 1k 的 0.5015 高一倍。
- TV20 相对 TV10 **+0.0250 [−0.0020,+0.0524]**，单 seed、CI 刚跨零；是目前最高均值，但不能当已确认。

## 5. 视觉 token 选择：减少 token 要付代价

### 5.1 帧复制曲线（同分辨率）

| 输入 | 视觉 token | 训练后 τ | 备注 |
|---|---:|---:|---|
| 16f1x | 1152 | 3.7595 | E2E10 |
| 32f1x | 2304 | 3.7475 | 单 seed |
| 64f1x | 4608 | 3.6757 | 单 seed |
| **128f1x** | **9216** | **3.8796 / 3.8740** | 两 seed |

- 32/64 单 seed，差异 0.07–0.10，不能排除 seed 噪声；但 **64 甚至低于 1k**，128 两 seed 稳定高。
- **不是“token 数单调收益”**；只有完整的 128 帧/9216 构造复现了正结果。

### 5.2 用 9k head 直接跑更少 token（不重训）

| 输入 | head | τ | 相对同输入训练 head | 相对 128f1x |
|---|---|---:|---:|---:|
| 64f1x | 64f1x | 3.6757 | — | −0.204 |
| 64f1x | **128f1x** | **3.7004** | +0.0246 [−0.0162,+0.0656] | **−0.179** |
| 32f1x | 32f1x | 3.7475 | — | −0.132 |
| 32f1x | **128f1x** | **3.7358** | −0.0116 [−0.0599,+0.0428] | **−0.144** |

- 9k head 对短输入有一点迁移，但**不能恢复 3.88**；4.6k 损失约 0.18，2.3k 损失约 0.14。
- 结论：**视觉 token 选择/截断不是免费加速**；128f1x 的收益依赖完整 9216-token 视觉上下文。

## 6. 融合架构与特征

### 6.1 层 gate / 跨层 xattn

| 128f1x + 模块（10 updates） | τ | 相对 base |
|---|---:|---:|
| base | 3.8796 | — |
| + layer gate（5 个 target 层） | 3.8751 | **−0.0045 [−0.0201,+0.0111]** |
| + 跨层 xattn | 3.8725 | **−0.0070 [−0.0417,+0.0249]** |

- layer gate 最后层权重范数仅 0.004–0.007，基本仍接近恒等；xattn `xattn_out` 范数 1.29，模块确实训练，但 τ 不变。
- **零初始化融合 + 10 updates 不足以偏离恒等映射**；不是“融合没用”，是预算/初始化不够。

### 6.2 静态层特征权重

head `fc` 各 target 层权重范数：

| layer | 1 | 9 | 17 | 25 | 33 |
|---|---:|---:|---:|---:|---:|
| norm | **1064.5** | 753.8 | 876.1 | 676.2 | **339.1** |

- 早层（1/17）占主导，layer33 最弱；但 layer gate 并没有因此学到可测增益。

### 6.3 zero-visual 因果探针

推理时只把 **head 看到的 visual-token target_hidden 置零**，文本和 target 计算不动：

| arm | τ | 相对 real |
|---|---:|---:|
| 128f1x real | 3.8796 | — |
| zero-visual | **3.6315** | **−0.2481 [−0.373,−0.144]** |
| blank（无视频） | 2.8428 | −1.0367 |

- 直接置零视觉 token 只掉 **0.248**，远小于 blank 的 **1.037**：约 **76% 的视觉依赖已经进入文本/上下文 hidden**，draft head 不是只从视觉 token 行读视频。
- 逐题异质：5 条掉 >0.4（最大 8.97→7.48），3 条略升，有的题几乎不变。
- 架构含义：**只在视觉 token 上做选择/融合会漏掉大部分视觉信号**；应做 target 上下文级/样本自适应融合。

## 7. 多模态特性总结

1. **token 数本身不是收益**：32f2x 9k 是负结果；1k > 3k > 9k32f2x 零样本。
2. **token 来源/分辨率是关键**：同为 9216，原生 128 帧（无空间上采样）3.8796，上采样 32 帧 3.4098/3.5334。
3. **视觉特异性会被稀释**：9k32f2x 对 real/mismatch 的 first-token KL 只有 0.367（1k 0.619），top1 更不易变。
4. **draft 接受长度不奖励视频配对正确**：1k/9k 的 mismatch 平均 τ 都等于 real；blank 才显著下降。
5. **视觉信息主要已在文本上下文**：zero-visual 只掉 0.248，blank 掉 1.04；未来融合应保留/利用上下文 hidden，而不是只挑视觉 token。
6. **视觉依赖是异质的**：zero-visual 逐题 3 升 26 降，最大 −1.49；固定全局融合/选择会伤一部分题。
7. **优化权重比零初始化架构更快见效**：TV20 3.9046（+0.025，边界），而 layer gate/xattn 中性。

## 8. 实验清单（可回溯）

### 8.1 Checkpoints

- `visual_long_20261106/video_e2e_9k_tv20/step10` — **TV20，3.9046**
- `visual_long_20261106/video_e2e_9k_128f1x/step10` — 128f1x seed1，3.8796
- `visual_long_20261106/video_e2e_9k_128f1x_s2/step10` — 128f1x seed2，3.8740
- `visual_long_20261106/video_e2e_9k_layer_gate/step10` — layer gate，3.8751
- `visual_long_20261106/video_e2e_9k_xattn/step10` — xattn，3.8725
- `visual_long_20261106/video_e2e_32f1x/step10`、`video_e2e_64f1x/step10` — 帧曲线
- `visual_long_20261106/video_e2e_1k_u10b/step5` — 1k 同预算对照
- `visual_long_20261106/video_e2e_9k_lr1e4_s2/step15`、`.../s2b/step5`、`.../s2b2/step10`、`video_e2e_9k_seed2/step20` — 32f2x 训练/seed 对照
- image：`image_e2etv120/step120`、`image_e2etv120_s2/step120`、`image_base120/step120`

### 8.2 Eval JSON

`eval/` 下关键文件：
`opd_9k_tv20.json`、`opd_9k_tv20_blank.json`、`opd_9k_128f1x_step10.json`、`opd_9k_128f1x_s2.json`、`opd_128f1x_blank.json`、`opd_1k_e2e10.json`、`opd_1k_mismatch.json`、`opd_3k_2gpu.json`、`opd_3k_step5.json`、`opd_9k_3gpu.json`、`opd_9k_lr1e4_step5/15.json`、`opd_9k_step20/30.json`、`opd_9k_seed2_step20.json`、`opd_9k_step20_blank.json`、`opd_9k_lr1e4_step15_blank2.json`、`opd_9k_lr1e4_step15_mismatch.json`、`opd_32f1x.json`、`opd_64f1x.json`、`opd_9khead_on32/64.json`、`opd_9k_layer_gate.json`、`opd_9k_xattn.json`、`opd_9k_zero_visual.json`、`visual_sensitivity_1k/9k.json`。

### 8.3 数据/cache

- `video_cache_32f_2x/`、`video_cache_32f_1x/`、`video_cache_64f_1x/`、`video_cache_128f_1x/`
- `video_cache_32f_2x_blank/`、`video_cache_128f_1x_blank/`
- `pool_video_32f_2x.parquet`、`pool_video_32f_1x.parquet`、`pool_video_64f_1x.parquet`、`pool_video_128f_1x.parquet`
- `video_schedule_u10.jsonl` 等

### 8.4 代码/提交

- `repro/video_caption/train_video_9k.py`：9k target 3 卡训练入口
- `repro/evaluation/eval_video_opd.py`：`--device-map`、`--max-memory`、`--zero-visual` 诊断
- `repro/diagnostics/expand_video_cache.py`：帧复制/上采样
- 提交：`5d7a7a9`（TV20 blank）、`a2a384f`（TV20）、`9fb6fd4`（融合/token 选择）、`52cdcfb`（seed2）、`9d22174`（128f1x）、`8e9e237`（9k 训练入口）

## 9. 存储清理

2026-11-10 按用户授权清理旧 checkpoint：删 45 个 `training_state.pt`（620.3 GiB）、旧负结果/中间权重 147.2 GiB、视频旧 stage 约 54 GiB、底座 training_state 95.7 GiB；合计释放约 **900 GiB**。保留上述关键 `model.safetensors`、eval/report/log。清单：
`/mnt/local/mcy/GLANCE/results/CLEANUP_20261110_summary.md`。

## 10. 限制与下一步

**限制**

- 128f1x 的 16 帧是**复制**成 128 帧，不是新增真实时间采样；收益含“长上下文/重复”成分，不能直接外推到真实 128 帧视频。
- fresh30 只有 30 条，逐题方差大；128f1x/TV20 的配对 CI 很宽，TV20 单 seed。
- 32/64f1x、TV20、layer gate、xattn 都是单 seed；32/64 的差异不能排除 seed 噪声。
- zero-visual 只置零 head 的视觉 token 行，文本 target_hidden 仍保留视觉信息，因此是**下界**因果，不是“完全去掉视觉”。
- 9k 训练/评测成本高（~3–7 分钟/update 或 row），架构臂只跑了 10 updates；零初始化模块需要更长训练才能判断。

**下一步**

1. **TV20 第二 seed**（最高优先级）；若复现，再做 TV20 的 mismatch/blank 分层。
2. **真实 128 帧源视频**（不是复制）：验证“原生时间采样”是否仍优于 1k。
3. **上下文级融合**：zero-visual 表明 76% 视觉信号在文本 hidden；试 context-FiLM/dual-path（视觉 token + 文本上下文），而不是只选视觉 token。
4. **更长融合训练**：layer gate/xattn 从 step10 续到 30–40 updates，看是否偏离恒等。
5. **样本自适应视觉门控**：zero-visual 有 3 条题反而变好，固定融合会伤这部分；可学 per-sample gate。
6. **真实帧选择**：用 128 帧中不同时间采样/帧重要性，而非复制 8×；验证 token 选择是否能真正做到“少 token、不降点”。
