# 图像 OPD 的视觉依赖诊断与加权实验（2026-11-02）

状态：视觉依赖诊断完成；视觉加权 60-update 对照训练与 test200 评测进行中。本文件先记录诊断协议与结果，训练/评测完成后补最终数字。

## 1. 问题

当前 OPD 的输入已经是 target 的 fused vision-language hidden，但 loss 只在生成的文本 token 上计算 Bernoulli forward KL / rejected k3，没有任何项区分“这个 token 依赖图像”还是“语言先验就能猜出来”。因此需要先测量：OPD 相对 SFT 的增益，是否集中在视觉依赖的 token/问题上。

## 2. 诊断协议

- 固定 test200，target fp32，head 使用已完成的 SFT/OPD `step3546`。
- 对每个问题，把 target 自己的 greedy answer 在两种视觉输入下 teacher-force：
  - 正确图像；
  - 同尺寸纯灰图（blank），prompt/grid/answer token 位置保持不变。
- 每个 answer token 计算：

\[
g_i=\max\left(0,\ \log p_{\text{real}}(y_i)-\log p_{\text{blank}}(y_i)\right)
\]

- 问题级视觉依赖度取 answer token 的 `g_mean`。
- 用已有的 per-question `tokens_per_round` 连接 SFT/OPD 的 τ；按 `g_mean` 中位数、上下四分位分组，做 paired bootstrap 95% CI。
- 实现：`repro/diagnostics/visual_dependence_sqa.py`；产物 `results/visual_dependence_20261102/test200_blank.{json,csv}`。

## 3. 诊断结果

总体：`g_mean` 均值 0.607，中位数 0.580；SFT τ 5.3666，OPD τ 6.2206，差 +0.8541，95% CI [+0.7129, +1.0065]。

| 分组 | n | g_mean | SFT τ | OPD τ | OPD−SFT | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 高 g（≥中位数） | 100 | 0.776 | 5.2678 | 6.1211 | +0.8533 | [+0.6548, +1.0733] |
| 低 g（<中位数） | 100 | 0.437 | 5.4654 | 6.3202 | +0.8548 | [+0.6547, +1.0671] |
| 最高四分位 | 50 | 0.909 | 4.9303 | 5.4567 | +0.5264 | [+0.3755, +0.6905] |
| 最低四分位 | 50 | 0.350 | 5.0047 | 5.6311 | +0.6264 | [+0.3755, +0.9133] |

- 高 g 与低 g 的 OPD−SFT 差几乎相同（0.8533 vs 0.8548）；`g_mean` 与 `delta` 的 Spearman 相关为 **0.033**，`g_mean` 与 τ 的相关约 0。
- 最高四分位（最依赖图）的 OPD 增益反而略低（+0.526 vs +0.626），同时该组 τ 更低（问题更难）。
- 结论：**当前 OPD 的增益没有集中在视觉依赖的问题上，更像是在所有 token 上普遍改善；不能把现有 OPD 称为“多模态增益”。**

## 4. 视觉加权 OPD 实验

- 权重：`w_i = 1 + alpha * clip(g_i, 0, cap)`，`alpha=1.0`，`cap=2.0`；240 个训练 QA（60 updates × 4 slot）离线计算，权重均值 1.265，范围 [1,3]。
- 两臂：同一 OPD `step3546` 初始化、同一 60-update schedule、seed 20261102、lr 1e-5 constant、4 QA/update、4 卡；只差 `--visual-weights`。
- 训练产物：`results/visual_opd_20261102/{base,weighted}`；评测在 test200 上做，按 `g_mean` 中位数/四分位比较 grounded 与 ungrounded τ。
- 权重构建：`repro/diagnostics/build_visual_weights.py`；训练改动：`loss.py`、`runtime.py`、`parallel_trainer.py` 增加可选 `--visual-weights`，不影响无权重路径。
- 结果：待补。

## 5. 后续

如果加权臂在 test200 上提高高 g 组 τ 且不损总 τ，再延长预算或加入 clue 反事实/attention 蒸馏；如果只提高低 g 或整体下降，说明当前 token 级加权不够，下一杠杆应转向 grounded copy 候选（OCR/表格/视频字幕）或视觉 attention 蒸馏。
