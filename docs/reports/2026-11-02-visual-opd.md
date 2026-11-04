# 图像 OPD 的视觉依赖诊断与加权实验（2026-11-02）

状态：视觉依赖诊断完成；位置加权、cutout、mismatch、2×2 clue、3×3 clue、target-attention 权重六种输入侧干预均完成。最终结论为**阴性**：没有统计支持的 visual-specific 增益；3×3 高 g −0.0087，2×2 +0.0155 未复现，合并 +0.0039；target-attention 权重高 g −0.0041、整体 +0.0089。低 g 正效应反复出现但不是视觉特异性。三卡 clean base 因 GPU 占用未能执行，拓扑混杂已注明。

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
| clue (2×2 occlusion) | +0.0141 | +0.0155 | +0.0127 |
| clue3 (3×3, 3-card)* | +0.0121 | −0.0087 | +0.0328 |
| attnw (target attention weights) | +0.0089 | −0.0041 | +0.0220 |

高 g 组只有 2×2 clue 为正（+0.0155，CI 跨零）；3×3 clue 与 target-attention 权重都略负。低 g 组多臂略正。**视觉侧干预仍没有产生统计上站得住的 visual-specific 增益；不能把 OPD 的 +0.85 解释成多模态增益。**（*3×3 臂因当时只有三张 46GB 空闲卡，跑在三卡/四逻辑 slot 拓扑；三卡 clean base 对照未能执行，见 §8.2/§8.3。）

## 7. Clue-targeted cutout

### 7.1 Clue 定位

- 先试了 target fused hidden 的 answer-vs-visual cosine，以及 last-layer attention 的 image-token 权重；两者 top token 都落在图像左上角/attention sink，debug 图上明显不覆盖 Wisconsin 等答案区域，不能用作 clue。
- 改用 **target occlusion sensitivity**：对每道训练题把图分成 2×2，分别灰掉一个象限，teacher-force 原 greedy answer，取 answer mean log-prob 下降最大的象限作为 clue。240 题全部计算；drop 均值 0.235（中位 0.211，最大 0.817），129 题 drop>0.2，31 题 <0.05。选中的 cell 分布均衡（四象限 52–74 题）。
- debug 图上选中的象限覆盖了 map 的 Wisconsin 高亮区和 magnet 图的距离标签区域，定位合理。

### 7.2 训练与评测

- 同一 `step3546`、同一 60-update 日程/seed；`--visual-corrupt clue --visual-corrupt-prob 0.5`，实际覆盖 **129/240** 个 QA；被选中的 50% 象限替换为灰色，target/head 同时看到 clue-removed 图，loss 不变。
- test200：**200/200 逐 token 无损**，准确数 **162/200**。

| 分组 | n | base τ | clue τ | clue−base | 95% CI |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 6.2305 | 6.2445 | +0.0141 | [−0.0094, +0.0391] |
| 高 g | 100 | 6.1422 | 6.1577 | +0.0155 | [−0.0164, +0.0522] |
| 低 g | 100 | 6.3187 | 6.3314 | +0.0127 | [−0.0198, +0.0457] |
| 最高四分位 | 50 | 5.4706 | 5.4644 | −0.0062 | [−0.0369, +0.0288] |
| 最低四分位 | 50 | 5.6253 | 5.6657 | +0.0404 | [+0.0095, +0.0760] |

- 高 g 点估计 +0.0155 是四种干预里唯一为正，但 CI 跨零；最高四分位 −0.0062（CI 跨零）；唯一 CI 不跨零的是**最低四分位低 g +0.0404**。
- 结论：clue-targeted cutout 的方向比随机遮挡/错图好，但仍没有统计证据表明它把收益移到视觉依赖位置；当前证据更像低 g 组的正则化或噪声。**不能据此声称 clue 反事实有效。**
- 可能的下一步：把 2×2 的 50% clue box 缩小到 3×3（更 targeted）或做第二 seed；如果高 g 仍不显著，转 visual attention 蒸馏或 grounded copy。

## 8. 3×3 clue（预注册判定）

2×2 clue 的高 g 点估计 +0.0155、CI 跨零，不能给确定性结论。为区分“50% clue box 太大/太粗”与“clue 方向本身无效”，再跑一个更 targeted 的 3×3 遮挡 clue 臂（box 约 11% 图），其余协议完全相同（同一 `step3546`、60 updates、seed、4 QA/update；实际覆盖概率 0.5）。

**预注册判定规则（主终点：test200 高 g τ，clue3−base）：**

- **阳性**：高 g Δ > 0 且 95% CI 下界 > 0。
- **阴性**：高 g Δ ≤ +0.005，或 95% CI 上界 < +0.02（即排除了有实际意义的正效应）。
- **不确定**：介于两者之间；此时做第二 seed（同一 3×3 clue 协议，seed 20261103），用两 seed 的 inverse-variance 合并均值和 CI 再判；若合并后仍不满足阳性，则最终判阴性（无统计支持的 visual-specific 增益）。

这个规则在看 3×3 结果之前写定，避免事后挑口径。

### 8.1 3×3 clue 结果

- 同样 60 updates、同一 `step3546`/日程/seed；3×3 occlusion 的 clue cell 约 11% 图；240 题 drop 均值 0.220（中位 0.194，max 0.720）；129/240 QA 实际被改。
- test200：**200/200 无损，162/200 正确**。

| 分组 | n | base τ | clue3 τ | clue3−base | 95% CI |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 6.2305 | 6.2425 | +0.0121 | [−0.0127, +0.0366] |
| 高 g | 100 | 6.1422 | 6.1335 | **−0.0087** | [−0.0444, +0.0269] |
| 低 g | 100 | 6.3187 | 6.3516 | +0.0328 | [+0.0023, +0.0682] |
| 最高四分位 | 50 | 5.4706 | 5.4534 | −0.0171 | [−0.0480, +0.0094] |
| 最低四分位 | 50 | 5.6253 | 5.6647 | +0.0394 | [−0.0039, +0.0874] |

- 按预注册规则，高 g Δ = −0.0087 ≤ +0.005，**直接判阴性，不需要第二 seed**。2×2 clue 的 +0.0155 没有复现；更 targeted 的 3×3 反而略负；低 g 仍显著为正（+0.0328）。
- **但 3×3 臂跑在三卡拓扑**（当时只有三张 46GB 卡空闲），而 base 是四卡拓扑。三卡/四逻辑 slot 保留 slot 顺序和 rank0 scoring，但 per-slot RNG/拓扑与四卡不同；−0.0087 可能混入拓扑效应。
- 因此追加 **三卡 clean base**（`base3`，同三卡拓扑/同 60 updates/seed，无 clue），用 clue3−base3 隔离 clue 效应；最终结论在 §8.3 更新。

### 8.2 三卡 clean base 对照（未能执行）

- 目的：把 3×3 clue 与同拓扑的无 clue 三卡 run 比，排除“三卡拓扑本身让高 g 变差”的混杂。`base3` 的 budget case 和 launcher 已准备好。
- 2026-11-03 18:36 启动时 OOM：GPU3 被外部 vllm 进程占用 43.6GB，当时只有 GPU6/7 两张 46GB 卡空闲；三卡训练需要三张，启动失败，未消耗 base3 预算。之后 GPU1/2/3 的空闲显存都 <16GB，无法再起三卡（更无法四卡）。
- 因此没有 topology-controlled base3 结果；这是本节结论的明确限制，不把 3×3 的负值单独当作 clue 的因果证据。

### 8.3 最终结论：阴性

- 预注册主终点（3×3 高 g）：Δ = **−0.0087**，95% CI [−0.0444, +0.0269]。按预注册规则（Δ≤+0.005 → 阴性），**判阴性，不做第二 seed**。
- 2×2 clue（四卡）高 g 为 +0.0155，但 CI [−0.0164, +0.0522] 跨零；它不是显著正效应。把 2×2 与 3×3 的高 g 用 inverse-variance 合并：**+0.0039，95% CI [−0.0208, +0.0286]**——仍然没有正效应。
- 低 g 组反复为正：2×2 +0.0127，3×3 +0.0328（CI [+0.0023,+0.0682]）；底部四分位 +0.0404 / +0.0394。这个方向一致，但它是**低视觉依赖组的正则化/语言先验改善**，不是我们想要的 visual-specific 增益。
- 3×3 的三卡拓扑是一个混杂；但没有 topology-controlled base3 时，3×3 的负值不能单独证明 clue 有害。即便如此，**2×2 四卡臂的高 g 已经不显著，两臂合并高 g 约 +0.004，没有任何一臂给出 CI 下界 >0 的高 g 正效应**。因此对“clue-targeted cutout 能提高 high-g τ”的假设，结论是**否定/无统计支持**，而不是“还不确定”。
- **最终判定：阴性。** 位置加权、cutout、mismatch、2×2 clue、3×3 clue、target-attention 权重六种输入侧视觉干预都没有产生统计上站得住的 visual-specific 增益；可靠的只有低 g 组的小幅正效应。停止在输入侧继续加权重/噪声/clue cutout/target-attention 加权。

## 9. 后续

这轮结果说明：**五种输入侧视觉干预都没有给出统计上站得住的 visual-specific 增益**；可靠的只有低 g 组的小幅正效应。输入里已经有 fused vision-language hidden，瓶颈不在“给视觉位置更高权重”“加视觉噪声”或“去掉 clue 区域”。下一杠杆按优先级：

1. **Clue 反事实 / 视觉 attention 蒸馏**：用 target 的视觉 attention 或归因选 clue patch/frame，去掉 clue 后要求 student 跟随 target 的 clue-removed 分布，或直接对齐 target 的 visual attention；这比位置权重更强。
2. **Grounded copy 候选**：OCR/表格/chart 文字、视频字幕作为候选前缀注入树，由 target 正常验证；这不改 OPD loss，但对 DocVQA/ChartQA/VideoDetailCaption 可能比 loss 调参更直接。
3. **更强/更细的权重**：当前 blank-image g 是粗代理；可换成 shuffled-image 或 clue-removal 的 g，或按 token 类型（OCR copy vs 推理）分层，而不是全位置统一加权。
4. **更长预算**：60 updates 只改变了约 +0.01 τ；如果前三条有信号再延长，不先用更长的同配方硬跑。

## 10. Target visual-attention-guided OPD（进行中）

输入侧的位置权重之前用的是 blank-image 的 g；它测的是“去掉图后 target 的 token 概率变化”，不是 target 实际看哪里。这一轮换成 **target 自己的视觉 attention**：

- 用 eager target，对 240 个训练 QA teacher-force 原 greedy answer；取最后 4 层（32–35）的 attention，对 heads 取平均，对 image token 位置求和；去掉第一个 image token 的 attention sink。
- 每个 answer token 得到 `mass`；构建 `w = 1 + clip(mass/mean - 1, 0, 2)`（`alpha=1, cap=2`），即只把高于该题平均视觉 attention 的 token 加权。
- 240 题、38,691 个 answer token 全部算完；`w` 均值 1.228（min 1.0, max 3.0），target attention mass 均值 0.0389（min 0.0069, max 0.0872）。
- 与 blank-g 权重的 per-token 相关只有 **0.263**，不是同一个信号；因此这是一个独立的 attention-guided 权重实验。
- 训练：同一 `step3546`、同一 60-update 日程/seed、4 卡；只加 `--visual-weights attnw_u60_l32-35_alpha1_cap2.pt`。评测与 base 对照，重点看高 g τ。

### 10.1 结果

test200：**200/200 无损，162/200 正确**。

| 分组 | n | base τ | attnw τ | attnw−base | 95% CI |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 6.2305 | 6.2394 | +0.0089 | [−0.0131, +0.0312] |
| 高 g | 100 | 6.1422 | 6.1380 | **−0.0041** | [−0.0351, +0.0212] |
| 低 g | 100 | 6.3187 | 6.3407 | +0.0220 | [−0.0088, +0.0580] |
| 最高四分位 | 50 | 5.4706 | 5.4674 | −0.0032 | [−0.0320, +0.0285] |
| 最低四分位 | 50 | 5.6253 | 5.6189 | −0.0064 | [−0.0389, +0.0258] |

- 高 g −0.0041（CI 跨零），最高四分位 −0.0032；低 g +0.0220（CI 跨零）。`Spearman(g, delta)=0.006`，没有视觉特异性。
- **target attention 权重也失败**：它与 blank-g 权重的 per-token 相关只有 0.263，但训练结果同样是高 g 不升、低 g 略升。说明“用 target 实际看哪里来加权”这个信号本身不足以把 OPD 的增益移到视觉依赖位置。
- 这关闭的是 **attention-guided weighting**，不是“head attention 与 target attention 对齐”的完整蒸馏；但 target attention 质量本身与高 g 无关，完整蒸馏的优先级也应降低。

### 10.2 结论

**阴性。** 六种输入侧干预（位置权重、cutout、mismatch、2×2 clue、3×3 clue、target-attention 权重）都没有统计支持的 high-g 增益；唯一反复出现的是低 g 组的小幅正效应，方向与视觉 hypothesis 相反。继续在输入侧调权重/加噪声/去 clue/用 target attention 加权，预期收益很低。
