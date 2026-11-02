# 图像 OPD 的视觉依赖诊断与加权实验（2026-11-02）

状态：视觉依赖诊断完成；位置加权、cutout、mismatch 三种视觉侧干预均完成，均为阴性；报告完成。

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
- 评测：test200，两臂均 **200/200 逐 token 无损**，准确数均 **162/200**。

| 分组 | n | base τ | weighted τ | weighted−base | 95% CI |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 6.2305 | 6.2421 | +0.0117 | [−0.0083, +0.0321] |
| 高 g（≥中位数） | 100 | 6.1422 | 6.1431 | +0.0009 | [−0.0239, +0.0251] |
| 低 g（<中位数） | 100 | 6.3187 | 6.3411 | +0.0224 | [−0.0085, +0.0565] |
| 最高四分位 | 50 | 5.4706 | 5.4931 | +0.0225 | [−0.0065, +0.0620] |
| 最低四分位 | 50 | 5.6253 | 5.6235 | −0.0018 | [−0.0138, +0.0083] |

- 60-update 继续训练本身相对原 OPD `step3546`（6.2206）约 +0.01；weighted−base 的整体 CI 跨零。
- 高 g 组几乎完全不变（+0.0009，CI 跨零）；低 g/最高四分位略正，但 CI 也跨零或接近零。**简单的 token 位置视觉加权没有把 OPD 增益移到 grounded token 上，结果阴性。**
- 训练日志的 rollout 轮数也没有实质差异（base/weighted 平均 38.4/38.4；最后 10 次 39.6/39.8）。

## 5. 视觉反事实 cutout

- 训练：同一 `step3546`、同一 60-update 日程/seed；每个 `(update, slot, task)` 以 0.5 概率做 28–42% 灰色矩形 cutout；实际覆盖 **129/240** 个 QA。无 visual weights，loss 不变。
- test200：两臂均 **200/200 逐 token 无损**，准确数 **162/200**。

| 分组 | n | base τ | cutout τ | cutout−base | 95% CI |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 6.2305 | 6.2430 | +0.0126 | [−0.0139, +0.0390] |
| 高 g | 100 | 6.1422 | 6.1339 | −0.0083 | [−0.0488, +0.0308] |
| 低 g | 100 | 6.3187 | 6.3521 | +0.0334 | [+0.0013, +0.0698] |
| 最高四分位 | 50 | 5.4706 | 5.4734 | +0.0029 | [−0.0251, +0.0368] |
| 最低四分位 | 50 | 5.6253 | 5.6548 | +0.0295 | [−0.0125, +0.0789] |

- 高 g 组为负（−0.0083），最高四分位几乎为零；唯一 CI 不跨零的是低 g 组（+0.0334）。**cutout 没有把收益移到视觉依赖位置，反而更像低 g 组的正则化；对“视觉 grounding”假设是阴性。**
- 结论：随机遮挡视觉输入训练不能解决“OPD 增益不集中在视觉问题”的问题；需要更 targeted 的 clue/attention 或解码侧 copy，而不是继续加视觉噪声。

## 6. 视觉反事实 mismatch

- 训练：同一 `step3546`、同一 60-update 日程/seed；每个 `(update, slot, task)` 以 0.5 概率把当前问题的图替换成另一条训练样本的图，prompt 不变；实际覆盖 **129/240** 个 QA。无 visual weights，loss 不变。
- test200：**200/200 逐 token 无损**，准确数 **162/200**。

| 分组 | n | base τ | mismatch τ | mismatch−base | 95% CI |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 6.2305 | 6.2277 | −0.0028 | [−0.0302, +0.0236] |
| 高 g | 100 | 6.1422 | 6.1217 | −0.0205 | [−0.0608, +0.0207] |
| 低 g | 100 | 6.3187 | 6.3337 | +0.0150 | [−0.0176, +0.0488] |
| 最高四分位 | 50 | 5.4706 | 5.4364 | −0.0342 | [−0.0862, +0.0140] |
| 最低四分位 | 50 | 5.6253 | 5.6400 | +0.0146 | [−0.0154, +0.0465] |

- mismatch 整体与高 g 都为负；最高四分位 −0.0342（CI 跨零），低 g 略正（CI 跨零）。**错图反事实也没有提高视觉依赖位置，方向甚至更差。**
- 三种视觉侧干预汇总（相对同一 base 臂）：

| arm | overall Δ | 高 g Δ | 低 g Δ |
|---|---:|---:|---:|
| weighted | +0.0117 | +0.0009 | +0.0224 |
| cutout | +0.0126 | −0.0083 | +0.0334 |
| mismatch | −0.0028 | −0.0205 | +0.0150 |

高 g 组没有一臂为正；低 g 组两臂略正。**“在现有 token-level OPD 上加视觉权重/视觉扰动”这条路线到此为止，不能把 OPD 的 +0.85 解释成多模态增益。**

## 7. 后续

这轮结果说明：**位置权重、cutout、mismatch 三种最便宜的视觉侧干预都没有把 head 推得更依赖视觉**；输入里已经有 fused vision-language hidden，瓶颈不在“给视觉位置更高权重”或“加视觉噪声”。下一杠杆按优先级：

1. **Clue 反事实 / 视觉 attention 蒸馏**：用 target 的视觉 attention 或归因选 clue patch/frame，去掉 clue 后要求 student 跟随 target 的 clue-removed 分布，或直接对齐 target 的 visual attention；这比位置权重更强。
2. **Grounded copy 候选**：OCR/表格/chart 文字、视频字幕作为候选前缀注入树，由 target 正常验证；这不改 OPD loss，但对 DocVQA/ChartQA/VideoDetailCaption 可能比 loss 调参更直接。
3. **更强/更细的权重**：当前 blank-image g 是粗代理；可换成 shuffled-image 或 clue-removal 的 g，或按 token 类型（OCR copy vs 推理）分层，而不是全位置统一加权。
4. **更长预算**：60 updates 只改变了约 +0.01 τ；如果前三条有信号再延长，不先用更长的同配方硬跑。
