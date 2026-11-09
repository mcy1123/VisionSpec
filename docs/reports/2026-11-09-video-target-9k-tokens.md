# 视频 target 输入 5–10k 视觉 token：实测与边界（2026-11-09）

## 1. 输入构造

现有视频 cache 只有 16 帧（384×288，30 fps）。为把 **target 输入** 提到 5–10k，把每帧复制 2×、分辨率 2×，得到 32 帧 / 768×576：

- Qwen3-VL processor 输出 `video_grid_thw=[16,48,48]`，`<|video_pad|>` = **9216 token/视频**；
- fresh30 实际 **4992–9216**，均值 **6118**；train439 实际 **3840–9216**，均值 **6025**。
- 注意：这是**上采样 + 帧复制**，不是新增真实帧；16 帧源信息不变。

## 2. target fp32：3 卡 model parallel 才能装下

- 单卡 47.5GB fp32：9k prefill 一次 attention 申请 **10.48 GiB**，OOM（卡上只剩 7.68 GiB）。
- 用 `accelerate` 的 `device_map="balanced"` 把 fp32 target 分到 **3×47GB**（layers 0–5 / 6–21 / 22–35），并补了两处设备处理：
  - target 输出 logits/hidden 统一搬回 `cuda:0`；
  - `DynamicCache` 的 gather 改成 **每层用自己的 device** 做 `index_select`（否则跨层 `cuda:0` index 对 `cuda:1` keys 报错）。
- 结果：9k fp32 **可以逐 token 无损跑**；单条 ~1–1.5 分钟（3 卡）。

## 3. 零样本结果：9k target 反而显著更差

fresh30，OPD3e step220，fp32，b63/k8，AR 逐 token 门禁：

| target 视觉 token | mean L | τ | lossless |
|---|---:|---:|---:|
| 1k（原 16f，单卡） | 862 | **3.7841** | 30/30 |
| **9k（32f2x，3 卡）** | **6276** | **3.4098** | **30/30** |

- 9k − 1k = **−0.3743**；虽然 30/30 无损，但接受长度大幅下降。
- 原因不是 token 本身无效，而是 **draft 只在 1k 上下文上训练过**；把 target 换成 6k 上下文后，draft 的视觉条件分布整体错位，树候选质量变差。
- bf16 9k 也在单卡跑过（τ 3.438，**lossless 1/30**），但 bf16 归约顺序导致非无损，不能作为正式结果。

## 4. 只加 draft-side 9k memory 也不行

用 3840–9216 token memory 训练 FOVEA 60 updates，target 仍 1k：

| arm | real τ | blank τ | reliance |
|---|---:|---:|---:|
| OPD3e | 3.7841 | 3.2826 | 0.5015 |
| FOVEA 1k memory | 3.7530 | 3.3676 | 0.3854 |
| FOVEA 3k memory | 3.7519 | 3.3676 | 0.3843 |
| FOVEA 9k memory | **3.7485** | 3.3676 | 0.3809 |

- 9k memory − OPD3e：real **−0.0357**；和 1k/3k memory 几乎一样；30/30 无损。
- 说明 FOVEA 的 topK 检索没有因为 token 多而变好。

## 5. 结论

1. **target 输入 5–10k 已经可以做了**：3 卡 fp32 model parallel + 设备本地 cache gather，9k 30/30 无损。
2. 但 **零样本 OPD3e 在 9k 上 τ 从 3.7841 掉到 3.4098**；token 数本身不是收益，draft 必须 **在 9k target 上下文上重新训练**，否则分布错位。
3. 只把 5–10k token 放进 draft memory（FOVEA）也没有收益。
4. 下一步要做的是 **9k target 上下文的 draft 训练**（target 3 卡 + head 1 卡，或 target 特征离线预计算后训练 head），而不是继续加 token 数。

## 6. 产物

- 9k cache：`video_cache_32f_2x/`、`video_fresh30_32f_2x.parquet`、`pool_video_32f_2x.parquet`
- 9k memory：`vmem/video_train439_9k.pt`（21G）、`vmem/video_fresh30_9k.pt`
- target 9k eval：`eval/opd_9k_3gpu.json`（30/30 lossless, τ 3.4098）
- 代码：`repro/shared/common.py`（`device_map`）、`repro/evaluation/eval_video_opd.py`（多卡输出/ lm_head 路由）、`GLANCE_repo/glance/decode.py`（device-local cache gather）


## 7. 9k target 训练（已安排并跑通）

新增单进程入口 `repro/video_caption/train_video_9k.py`：

- target：fp32，`device_map="balanced"` 跨 3×47GB；输出/lm_head 统一回 cuda:0；
- head/master/Adam：cuda:0；target 3 卡 + head 同卡 0；
- 9k 视频：`video_cache_32f_2x`，`max_seq_len=16384`；
- 1 update smoke：loss 4.99，grad 6.25，完成并保存 step1。

### 7.1 5-update 学习率对比（fresh30，9k fp32 target，30/30 无损）

| arm | τ |
|---|---:|
| OPD3e zero-shot @9k | 3.4098 |
| E2E-TV 9k，lr 1e-5，5 updates | 3.4189 |
| E2E-TV 9k，lr 1e-4，5 updates | **3.4790** |

- lr 1e-4 把 9k 上的 τ 从 3.4098 提到 3.4790（+0.069），30/30 无损；但仍比 1k 的 3.7841 低 **0.305**。
- 5 updates 只能恢复约 19% 的差距；训练确实在起作用，但预算远不够。
- 训练速度约 **3 分钟/update**（4 视频/update，9k target，3 卡），20 updates 约 1 小时；比 1k 训练慢很多。

### 7.2 状态

- `video_e2e_9k/step5`：lr 1e-5，5 updates（τ 3.4189）。
- `video_e2e_9k_lr1e4/step5`：lr 1e-4，5 updates（τ 3.4790）。
- 已从 `video_e2e_9k_lr1e4/step5` 续训 15 updates（lr 1e-4），产物在 `video_e2e_9k_lr1e4_s2`；完成后用同一 3 卡 9k 评测。
- 结论：**target 5–10k 的训练路径已经打通；5 updates 有正趋势但远未追平 1k；需要更长训练/更大 lr 预算才能判断 9k 是否能超过 1k。**
