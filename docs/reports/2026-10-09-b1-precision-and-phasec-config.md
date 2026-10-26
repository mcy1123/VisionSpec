# B1 trainer 真实验证、teacher 精度判定与 Phase C seed-0 配置冻结

> **补注（2026-10-10，审计后）**：§1 的结论范围需要收窄。
> 完整审计见 [`2026-10-10-phase-c-corrections.md`](2026-10-10-phase-c-corrections.md)。
>
> 本报告实测 OOM 的配置是 **fp32 teacher + fp32 head + Adam**（24 anchors 与 1 anchor
> 两种设置）。**fp32 teacher + bf16 head** 当时遇到的是 dtype 接口报错，**从未做显存测试**。
> 修好 `opd_online_rollout.rollout` 与 `draft_block_logits` 的 dtype 边界后实测：
> **3 次真实 optimizer updates 通过，峰值 43.84 GiB < 44 GiB**（N_acc/Z 两流均正常、
> 梯度有限非零）。因此「fp32 teacher 单卡不可行」**不成立**；正确的表述是
> 「**fp32 teacher + fp32 head** 不可行」。混合精度路径可用，且距 44 GiB 目标仅余
> 0.16 GiB，长跑需另行验证显存。


日期：2026-10-09
任务依据：`docs/plans/video-opd.md` §10.4（精度与真实训练可行性）、§10.5（冻结 seed-0 配置）
前置：`docs/reports/2026-10-09-b1-formula.md`（§10.3 公式核实）、`results/phaseB05/b05_calibration.json`（§10.2）

---

## 1. §10.4 真实 update 的显存门禁（实测，非估算）

命令（同一入口、同一数据、只改 dtype 与 anchors）：

```bash
cd /home/mcy/Projects/GLANCE/repro
R=/mnt/local/mcy/GLANCE
PYTORCH_ALLOC_CONF=expandable_segments:True \
/home/mcy/.conda/envs/fin_research/bin/python -u opd_train_video_fullkl.py \
  --records $R/data/video_greedy_499.pt --pool $R/data/pool_499.parquet \
  --init-from $R/checkpoints/head_video439_aligned/step1000 \
  --out <OUT> --updates 3 --anchors-per-update 24 --anchors-per-video 4 \
  --teacher-dtype <float32|bfloat16> --head-dtype <float32|bfloat16> \
  --max-rounds 4 --lr 3e-4 --save-every 0 --no-final-save --log-every 1 --gpu 0 \
  --log-file <OUT>/log.jsonl
```

| 配置 | 结果 | 证据 |
|---|---|---|
| **fp32 teacher + fp32 head, 24 anchors** | **OOM** | `torch.OutOfMemoryError` 在 `optimizer.step()` → Adam `state["exp_avg"] = torch.zeros_like(...)`；`47.51 GiB in use / 47.53 GiB` |
| **fp32 teacher + fp32 head, 1 anchor** | **OOM** | 同上，`46.73 GiB in use / 47.53 GiB` —— **证明减少 anchors 不是杠杆** |
| **bf16 teacher + bf16 head, 24 anchors** | **通过 3 次真实 update** | peak pass2 **26.33 / 26.42 / 26.37 GiB**，pass1 24.0/24.2/24.0 GiB，均 <44 GiB |
| fp32 teacher + bf16 head | 不可用 | 既有 `opd_online_rollout.rollout` 把 fp32 `fused` 直接喂给 bf16 head 的 `fc`：`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16` |

fp32 的定账（与实测一致）：target 权重 **32.8 GiB** + fp32 head 权重 3.91 + head 梯度 3.91
+ AdamW 两个 fp32 动量 7.81 = **48.4 GiB > 47.53 GiB 可用**，尚未计 activations 与 pass-1
持有的 trace。因此**在同一张 A6000 上，fp32 teacher + AdamW 的真实训练在权重/优化器层面
就已经放不下**，`--anchors-per-update` 降到 1 仍 OOM 直接证实了这一点。

### 1.1 已按 §10.4 依次尝试的缓解（均已内建或已排除）

| 缓解 | 状态 |
|---|---|
| 减小 microbatch anchors 并正确累积 | 已内建（microbatch = 1 anchor，按 update 全局分母缩放）；且 1-anchor 实测仍 OOM |
| 及时释放 teacher 中间结果、KV fork、无用图 | 已内建（每 anchor 释放；`attach_cropped_prefix` 每 anchor 复制后即弃） |
| 全词表分块 | 已内建（`teacher_block_logits(chunk=8)`） |
| 8-bit / offload optimizer | **未采用**：会与 C1 臂的优化器不一致，破坏 §10.5 要求的「两臂优化器配平」 |
| student activation checkpointing | 未采用：显存缺口在权重/优化器而非 activations |

### 1.2 判定：主实验改用 bf16 teacher，**标注为精度近似**

按 §10.4 的 fallback 条款执行。bf16 的误差由 §10.2（B0.5）在同一轨迹、同一 prefix、
同一 mask 上量化（`results/phaseB05/b05_calibration.json`）：

| 量 | bf16 vs fp32 teacher |
|---|---|
| teacher 分布 TV（同 480 行） | mean 0.0163 / P50 0.0112 / P90 0.0335 / P99 0.105 / max 0.232 |
| teacher max\|Δlogit\| | mean 0.392 / max 2.353（logit \|absmax\| ≈29.5） |
| teacher argmax 一致率 | **469/480 = 97.7 %** |
| 总 loss 相对差（各自 student） | **1.20e-3** |
| 总 loss 相对差（固定同一 student，纯 teacher 效应） | **7.11e-4** |
| **逐 anchor 相对差** | mean **1.67e-2**，max **9.5e-2**（32 anchor，17 正 15 负 → 聚合靠抵消） |
| head 梯度 cosine（两阶段整体） | 0.9887（范数比 0.983） |
| head 梯度 cosine（固定 bf16 student，纯 teacher 效应） | **0.99986**（范数比 0.998） |
| 显存 | fp32 42.02 GiB（含 backward，**无 Adam**）/ bf16 21.21 GiB |

**必须随结果一起报告的两条限制**：

1. **逐 anchor 的 loss 相对差可达 9.5 %**，聚合的 1.2e-3 是抵消的结果。任何「按 anchor/按视频」
   的细粒度结论都要按 1e-2 量级的不确定度读。
2. **bf16 会重新选择 anchors**：用 bf16 重跑 rollout，**8/8 条视频的 proposal 列表都不同**，
   其中 **2/8 条视频的 `pos`/`n_acc` 直接不同**。因此 bf16 臂不只是「同一个 loss 更噪」，
   而是**走了不同的轨迹、监督了不同的 accepted/rejected 划分**。本轮的 anchors 一律由
   fp32 rollout 冻结后复用（B0.5 的做法），但把一个 bf16 臂与 fp32 臂的差异全部归因于
   「KL 精度」是不成立的，报告里不得这样写。

---

## 2. §10.5 Phase C seed-0 配置（冻结）

### 2.1 三臂

| Arm | 设置 |
|---|---|
| **C0** | `head_video439_aligned/step1000`，不训练 |
| **C1** | 同起点 **aligned continue-SFT**：`train_head_video.py --feature-path cached` |
| **C2** | 同起点 **paper-faithful OPD**：`opd_train_video_fullkl.py`，在线 chain rollout |

### 2.2 冻结项

| 项 | 值 |
|---|---|
| 起点 | `$R/checkpoints/head_video439_aligned/step1000`（`model.safetensors` sha256 `e2526a249a8ef7e1db3a9990589b7275…`，`config.json` sha256 `9834d608c9ca53d5548b415471ae9e8e…`） |
| 数据 | `$R/data/video_greedy_499.pt` 的 `pool_499.parquet split=="train"` → **439 行**；heldout 60 行不参与训练 |
| seed | 0（C1 `--seed 0`；C2 `--seed 0`，视频顺序由 `video_order(..., seed, epoch)` 固定） |
| optimizer | AdamW，lr **3e-4**，betas `(0.9, 0.95)`，weight_decay **0.0**，clip **1.0** |
| LR schedule | **常数**（偏离论文的 cosine+warmup 0.05，理由见 §2.4） |
| anchors/update | **24**（两臂一致；定义不可比，见 §2.3） |
| C2 累积 | microbatch = 1 anchor；loss 用 **update 全局分母** `N_acc_total` / `Z_total` 缩放 |
| C2 运行 | `--anchors-per-video 4 --max-rounds 4 --max-seq-len 2048` |
| dtype | C2：teacher **bfloat16**（§1.2 的精度近似）、head bfloat16；C1：head bfloat16（`train_head_video.py` 默认） |
| γ | 0.8，只作用于 rejected 流 |
| checkpoint | **update 10 / 25 / 50** |
| 监控子集 | `$R/results/phaseB05/subsets.json` 的 `monitor_fresh30_videos`（8 条，sha256 `7ad692494eb7086a3c36b6ab29dc6cc6…`），**规则与模型得分无关** |
| 评测 | fp32、tree b63/k8、max_new 1024、跑到 EOS、`--ar` 逐 token 无损门禁 |
| C0 复用 | C0 的完整 `video_fresh30` 与监控子集数值直接取自 `results/phaseA4/sft439_step1000_fresh30_tree_float32.json`（完全同配置），**不重跑** |
| 上限 | 1 个正式 seed；每条训练臂最多 100 updates；不搜索超参数 |

### 2.3 必须如实记录的不可配平之处

- **anchor 定义不可比**：C1 的「anchor」是一个被监督的位置（24 位置/update）；
  C2 的「anchor」是一个 draft block（24 block/update，每 block 15 个 offset，
  实测 update 1 有 52 accepted + 308 rejected = **360 个被监督 offset**）。
  同一个「24 anchors/update」在两条臂下不是同一份计算预算。
- **恰好 1 道 prompt**：C1 每个被监督位置对应该位置的 token；C2 把整段**草稿诱导前缀**
  上的目标分布都纳入。两者监督的信号不同，这是方法差异本身，不是配置错误。
- 报告必须给出两臂的：视频曝光数、anchor 数、**有效监督位置数**、updates、GPU 时间、峰值显存。

### 2.4 显式偏离论文

| 偏离 | 理由 |
|---|---|
| 常数 lr（论文：cosine + warmup 0.05） | 论文的调度定义在 8 epochs × 16K prompts 上；本轮每臂 50–100 updates、439 行，没有有意义的 epoch 结构。且 C1/C2 必须用逐项相同的优化器才能把差异归因到 loss。项目既有基线也都是常数 lr 3e-4。 |
| bf16 teacher（论文未规定精度） | §1 的实测：fp32 不可行。按 §10.4 标注为精度近似。 |
| block 16 / 评测 tree b63k8 | 沿用 GLANCE 视频口径，论文主体是文本 EAGLE-3/DFlash。 |
| C1 用 SFT 目标而非 OPD 目标 | C1 的定义即同预算 continue-SFT。 |

---

## 3. 本轮 §10.4 交付的判定

- fp32 teacher **不可行**：有 OOM 实测 + 1-anchor 对照 + 定账，证据完整。
- bf16 teacher 工程替代 **可行**：3 次真实 update（含 rollout、teacher scoring、head backward、
  `optimizer.step()` 与 Adam 状态初始化）实测峰值 **26.33 GiB < 44 GiB**。
- **该替代标注为精度近似**：不得据 bf16 的负结果宣称「高精度 paper-faithful OPD 无效」，
  且不得把 bf16 臂的差异整体归因于 KL 精度（因为 anchors 会被重选，见 §1.2）。
