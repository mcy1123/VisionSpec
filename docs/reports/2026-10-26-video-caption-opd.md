# 视频长生成 OPD：3 epoch 小幅提升、4 epoch 回落、SFT 崩溃

日期：2026-10-26 夜 → 2026-10-27 晨
产物根：`/mnt/local3/mcy/GLANCE/results/video_caption_opd_20261026/`（完整报告 `REPORT.md`、协议 `PROTOCOL.json`、状态 `STATUS.md`）

## 结论

在 VideoDetailCaption（439 train 视频，独立 fresh30 评测，平均生成 **270.9 token/条**）上，
用当前 pure-block OPD 从 `head_video439_aligned/step1000` 继续训练：

| 臂 | τ mean | 对 warm | 95% CI | 备注 |
|---|---:|---:|---|---|
| warm | 3.7266 | — | — | 起点 |
| OPD 1 epoch（110 upd） | 3.7180 | +0.0086 | [−0.026, +0.043] | 跨零 |
| **OPD 3 epoch（330 upd）** | **3.7841** | **+0.0575** | **[+0.014, +0.102]** | 峰值；AR/投机 1.815→1.842 |
| OPD 4 epoch（440 upd） | 3.7104 | −0.016 | — | 比 3e 低 0.0737 |
| OPD 3e + 低 lr 110（1e-6） | 3.7846 | — | Δ vs 3e = −0.0005 | 不变 |
| SFT 1e（lr 1e-5） | 3.3336 | −0.393 | — | 退化 |
| SFT 1e（lr 3e-5） | 1.9164 | −1.810 | — | 崩溃 |
| SFT 3e（lr 1e-5） | 1.9168 | −1.810 | — | 同一平台 |

- 30/30 条视频逐 token 无损（所有臂）。
- 同 lr、同步数、同 fp32 target：OPD − SFT = **+0.3844**，CI [+0.301, +0.474]，30/30 OPD 胜。
- 固定 10 行子集曲线（warm=3.704）：update 20/40/60/80/100/110/165/220/275/330/385/440 = 3.706/3.642/3.610/3.615/3.668/3.688/3.767/3.816/**3.823**/3.794/3.764/3.739。
- 三档树鲁棒性：b31k4 3.4137→3.4754（+0.0616 [+0.0147,+0.1088]，18/30）；b63k8 3.7266→3.7841（+0.0575）；b127k8 3.9063→3.9521（+0.0458 [+0.0069,+0.0849]，16/30）。
- 机制：O 权重漂移始终 ≤0.3%；两个 SFT 臂以不同 lr/步数到达约 2.5% 漂移后都落到 τ≈1.917（≈只接受 root）。
  4-epoch 回落是步长而非数据耗尽（低 lr 续训完全不动）。

## 数据与口径

- VideoDetailCaption：500 段视频 / 499 行；train 439，pool heldout 60；fresh30 与 train 重叠 0/30。
- `video_heldout_30` 的 30 条**全部在 train 内**，不能作独立集。
- 16 帧、384 长边、prompt 862 token（视觉 672–1152 + 文本 ~94），AR prefill 只占墙钟 6.4%。

## 工程

视频入口原本显式报错；本次新增 `repro/video_caption/{build_pool,train_video_opd}.py`，
`--modality video`，三卡入口 `run(..., allow_fresh_start=False)`（ScienceQA 续训保护未动）。
首次 smoke 因 `CUDA_VISIBLE_DEVICES=3,6,7` 被 CUDA 枚举重映射（rank0 落到物理卡 1）而 OOM；改用 GPU UUID 绑卡。
CPU 回归改动前后同为 18 项 2 失败，无新增失败。

## 边界

单 seed、n=30 视频、单一独立集；stage2/3/5 各自重置 Adam（非严格连续训练）；
只验证接受长度与逐 token 无损，不验证生成质量；不外推为“视频领域 OPD 有效”。

## 加速比（同卡实测）

单卡 fp32、12 条 fresh30 ×2 次、两臂顺序交替、每条跑 AR oracle，48 次生成全部逐 token 无损：
warm AR/spec 端到端 1.825×（decode-only 1.949×），OPD 3e 1.844×（1.971×）；
**OPD 相对 warm：端到端 1.012×，95% CI [0.994,1.030]；decode-only 1.013×，CI [0.992,1.034]**。
即 τ 增益（n=30，+1.5%，CI 排除 0）在墙钟上只兑换约 1%，12 条上 CI 跨 1；视频 AR/spec 本身只有 1.82×（ScienceQA 2.3×），这是量化差距的主因。
