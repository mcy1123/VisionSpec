# 真实视频多模态 OPD recipe 扫描（2026-11-19）

## 0. 结论

**在当前真实视频任务上，多模态 OPD 的一阶变量是“训练预算/更新数”，不是 visual auxiliary 权重 λ。**

- 64 帧、20 updates：λ0 / λ10 / λ20 分别为 **9.1145 / 9.1230 / 9.1282**，三者在 0.014 token 内，**无法区分**。
- 同 20 updates 把 lr 从 1e-5 降到 5e-6：**9.0199**，更差，说明不是“越小越稳”。
- 64 帧、60 updates：λ0 **9.3190**、λ20 **9.3254**，仍几乎相同（差 +0.006 [−0.057,+0.072]）；λ10 的 D64 同 20 条为 **9.3536**，三者 CI 全部跨零。
- **60 updates 相对 20 updates 约 +0.20**（λ10：9.123→9.354；λ0：9.115→9.319），远大于 λ 的差异。
- 因此当前最优、最简单的 recipe 是：**64 真实帧 + lr1e-5 + global_pooled + 4 QA/update + 60 updates + pure OPD（λ0）**。  
  加 λ10/λ20 没有可测收益；这也让方案更简单，不依赖视觉辅助 loss。
- 按用户要求，本轮不再探索新多模态架构。

## 1. 运行身份

| 项 | 值 |
|---|---|
| 代码 | 最新 run/video-multimodal `73a18a3` |
| worktree | `/home/mcy/Projects/GLANCE-video-opd-recipe` |
| run 分支 | `run/video-opd-recipe-20261118` |
| 数据 | 真实 Video-MME-derived，80 train / 40 dev，64 帧 |
| base | `real_video_20261111/arms/seed20261111_lambda10/step20` |
| target | Qwen3-VL-8B-Instruct fp32 |
| 训练 | real time、global_pooled、4 QA/update、fp32 master、single A6000 per arm |
| 截止 | 2026-11-19 08:00 Asia/Shanghai |
| 原始结果 | `/mnt/local3/mcy/GLANCE/results/video_opd_recipe_20261118/` |

## 2. 20-update 扫描（dev20，64 帧）

所有臂 20/20 逐 token 无损：

| arm | λ | lr | τ |
|---|---:|---:|---:|
| L0 | 0 | 1e-5 | **9.1145** |
| L10 | 10 | 1e-5 | **9.1230** |
| L20 | 20 | 1e-5 | **9.1282** |
| L10-low-lr | 10 | 5e-6 | **9.0199** |

- λ0/10/20 最大差 **0.014**，远小于 seed/评测噪声。
- lr5e-6 比1e-5低 **0.103**，低 lr 在这个预算下没有帮助。
- 20 步时 e2e-TV 辅助项（top8-overlap proxy）对接受长度没有可测贡献。

## 3. 60-update 确认（dev20，64 帧）

| arm | λ | τ | 相对 20-step 同 λ |
|---|---:|---:|---:|
| L0_60 | 0 | **9.3190** | +0.2045 |
| L20_60 | 20 | **9.3254** | +0.1972 |
| D64 (λ10, 60) | 10 | **9.3536** | +0.2306 |

配对：

- L20−L0：**+0.0064 [−0.0566,+0.0715]**，10/9。
- D64−L0：**+0.0346 [−0.0213,+0.0928]**，11/9。
- D64−L20：**+0.0282 [−0.0018,+0.0573]**，11/4。
- 20→60 的增益：L0 **+0.2045**，λ10 **+0.2306**；远超 λ 的 0.006。

**λ0/10/20 在 60 步仍不可区分；更新数才是主变量。**

## 4. 推荐 recipe

```
target: Qwen3-VL-8B-Instruct fp32
frames: 64 real frames (L≈3073, visual≈2688)
objective: pure OPD, e2e_tv_weight=0
lr: 1e-5
loss normalization: global_pooled
QA/update: 4
updates: 60
time_mode: real
master weights: cpu fp32
```

- 若已有 λ10 的 D64，保留它也可以；差异只有 **+0.03** 且 CI 跨零。
- 不建议 λ20 或 lr5e-6；没有收益，λ20 还增加 loss 计算。
- 60 步单个 A6000 墙钟约 **2.26 h**；20 步约 **0.78 h**。

## 5. 限制

- 单 seed；20/60 步比较在 dev20；D64 的 9.3536 是同一 20 条，不是完整 dev40。
- 所有输出 1024 cap、`hit_eos=0`；τ 是接受效率，不是长报告质量。
- 20/60 臂单卡训练，D64 是 2×A6000 训练；接受对比可用，墙钟不完全可比。
- 没有 test；没有跨域视频。
- L0_60 的完整 dev40 评测在截止前启动，若未完成则只用 dev20 数字。

## 6. 产物

- 原始结果：`/mnt/local3/mcy/GLANCE/results/video_opd_recipe_20261118/`
  - `L0_lr1e5/step20`、`L10_lr1e5/step20`、`L20_lr1e5/step20`、`L10_lr5e6/step20`
  - `L0_60/step60`、`L20_60/step60`
  - `eval/{L0_lr1e5,L10_lr1e5,L20_lr1e5,L10_lr5e6,L0_60,L20_60}_64.json`
  - `schedule_64_u20.jsonl`、`schedule_64_u60.jsonl`
- 小型摘要：`docs/reports/2026-11-19-video-opd-recipe-summary.json`
- **未提交权重、cache、逐视频 JSON。**
