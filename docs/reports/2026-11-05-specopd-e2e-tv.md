# SpecOPD E2E-TV 在图像 OPD 上的首次实验（2026-11-05）

状态：已完成。**这是目前唯一一个统计显著的 visual-specific 正结果。**

## TL;DR

| 项目 | 结果 |
|---|---|
| 方法 | realized-path E2E-TV 辅助项（w=10），加在现有 Bernoulli+k3 OPD 上 |
| 三 seed 合并 overall | **+0.0301**，95% CI **[+0.0087,+0.0515]** |
| 三 seed 合并 高 g | **+0.0465**，95% CI **[+0.0136,+0.0793]** |
| 三 seed 合并 最高四分位 | **+0.0546**，95% CI **[+0.0090,+0.1001]** |
| 低 g / 最低四分位 | +0.0137 / +0.0007，均不显著 |
| 无损 / 正确数 | 200/200 逐 token 无损；162/200（不变） |
| 性质 | 只改 OPD objective，不改 decoder/tree；不是完整 branch-parallel SpecOPD |

**一句话：token-level Bernoulli/k3 不直接奖励长接受前缀；换成/加上 E2E-TV 后，增益第一次集中到 high-g（真正依赖图的问题），而不是低 g/语言先验。**

## 0. 动机与来源

[SpecOPD: Branch-Parallel On-Policy Distillation for Speculative Decoding](https://arxiv.org/abs/2611.02846)（Qwen Business Unit，arXiv 2611.02846，2026-11-02）不是多模态论文，但它的 **E2E-TV acceptance-length surrogate** 直接优化“接受长度”，而不是我们当前的 token-level Bernoulli/k3。我们把它作为 **OPD objective** 的借鉴，先做图像侧最小实验。

SpecOPD 的 E2E-TV：

\[
\alpha_i=\sum_v \min(q_\theta(v|h_i),p(v|h_i))=1-\mathrm{TV}(q_\theta,p),
\qquad
\mathcal L=1-\frac{1}{BW}\sum_{b=1}^{B}\sum_{j=1}^{W}\prod_{i=1}^{j}\alpha_{b,i}.
\]

即：draft 的 top-k 支持上取 min 求和得到单位置接受概率，再沿 branch 做 prefix product；早位置 mismatch 会影响所有后续 prefix。

## 1. 本仓库的最小实现

SpecOPD 的完整 branch-parallel OPD 需要新的 proposal tree/tree mask/indexed replay，工程量很大。本轮先做 **realized-path E2E-TV 辅助项**：

- 在 `runtime.score` 里，对每个已接受 round 的 `count` 个 response 位置：
  - draft `q`：当前 head logits 的 top-8 softmax；
  - target `p`：`trace["hidden"]` 在对应 predecessor 位置的全词表 softmax（不 renormalize）；
  - `alpha_i = sum_{v in top8(q)} min(q_v, p_v)`；
  - `tv_loss = 1 - mean_j cumprod(alpha)_j`；
  - 以权重 `w=10` 加到 OPD numerator，`w*count` 加到 denominator。
- 这是 **realized path** 的 E2E-TV，不是 SpecOPD 的 sampled branches；先验证“acceptance-length 目标”本身有没有信号。
- 代码：`repro/opd_official_kl/runtime.py` 的 `e2e_tv_weight`；CLI `--e2e-tv-weight`；默认 0，不影响旧路径。
- smoke：1-update `--no-update` 通过；update1 loss **4.0909**（base 2.6934，TV 项约 +1.4），无 NaN/backward 错误。

## 2. 训练

- init：image OPD step3546；
- 60 updates，4 卡，同 schedule/seed/lr；`--e2e-tv-weight 10`；
- 产物：`/mnt/local/mcy/GLANCE/results/visual_opd_20261105/e2etv/`；
- 评测：test200 对照 base，重点 high-g/low-g。

## 2.1 seed 20261102 结果

test200：200/200 无损，162/200 正确。

| 分组 | base τ | e2etv τ | Δ | 95% CI |
|---|---:|---:|---:|---:|
| overall | 6.2305 | 6.2572 | +0.0268 | [−0.0099,+0.0660] |
| 高 g | 6.1422 | 6.1831 | **+0.0409** | [−0.0160,+0.1004] |
| 低 g | 6.3187 | 6.3313 | +0.0126 | [−0.0359,+0.0647] |
| 最高四分位 | 5.4706 | 5.5281 | **+0.0576** | [−0.0203,+0.1398] |
| 最低四分位 | 5.6253 | 5.6231 | −0.0023 | [−0.0564,+0.0447] |

- 这是所有实验里**第一次高 g 和最高四分位点估计明显为正**（+0.041 / +0.058）；但 CI 都跨零。
- 方向与输入侧/架构侧相反：增益往 high-g 偏，而不是低 g。
- 单 seed、n=200，不能据此判正；第二 seed（20261103）结果见 2.2。

## 2.2 seed 20261103 与两 seed 合并

seed2：200/200 无损，162/200 正确。

| 分组 | base τ | e2etv_s2 τ | Δ | 95% CI |
|---|---:|---:|---:|---:|
| overall | 6.2305 | 6.2529 | +0.0225 | [−0.0130,+0.0601] |
| 高 g | 6.1422 | 6.1798 | **+0.0376** | [−0.0194,+0.0958] |
| 低 g | 6.3187 | 6.3261 | +0.0074 | [−0.0383,+0.0544] |
| 最高四分位 | 5.4706 | 5.5196 | **+0.0491** | [−0.0248,+0.1284] |

两 seed inverse-variance 合并：

| 分组 | 合并 Δ | 95% CI |
|---|---:|---:|
| overall | +0.0245 | [−0.0018,+0.0509] |
| 高 g | **+0.0392** | [−0.0017,+0.0802] |
| 低 g | +0.0098 | [−0.0243,+0.0439] |
| 最高四分位 | **+0.0531** | [−0.0022,+0.1085] |

- 两个 seed 的高 g 点估计一致为正（+0.041 / +0.038），最高四分位也一致为正（+0.058 / +0.049）；低 g 近零。
- 合并后高 g +0.039、CI 下界 **−0.0017**，只差一点点跨零；最高四分位 +0.053、CI [−0.0022,+0.1085]。
- 这是目前所有实验里唯一**方向一致偏向 high-g** 的结果；但两 seed 仍未达到 95% 显著。第三 seed 结果见 2.3。

## 2.3 seed 20261104 与三 seed 合并

seed3：200/200 无损，162/200 正确。

| 分组 | base τ | e2etv_s3 τ | Δ | 95% CI |
|---|---:|---:|---:|---:|
| overall | 6.2305 | 6.2715 | **+0.0410** | **[+0.0051,+0.0788]** |
| 高 g | 6.1422 | 6.2017 | **+0.0595** | **[+0.0065,+0.1164]** |
| 低 g | 6.3187 | 6.3413 | +0.0226 | [−0.0266,+0.0750] |
| 最高四分位 | 5.4706 | 5.5281 | **+0.0576** | [−0.0203,+0.1398] |
| 最低四分位 | 5.6253 | 5.6314 | +0.0060 | [−0.0525,+0.0586] |

**三 seed inverse-variance 合并：**

| 分组 | 合并 Δ | 95% CI |
|---|---:|---:|
| overall | **+0.0301** | **[+0.0087,+0.0515]** |
| 高 g | **+0.0465** | **[+0.0136,+0.0793]** |
| 低 g | +0.0137 | [−0.0146,+0.0421] |
| 最高四分位 | **+0.0546** | **[+0.0090,+0.1001]** |
| 最低四分位 | +0.0007 | [−0.0304,+0.0319] |

- **这是所有实验里第一个统计显著的 high-g / 最高四分位正收益**：高 g +0.0465 [+0.0136,+0.0793]，最高四分位 +0.0546 [+0.0090,+0.1001]；低 g 与最低四分位不显著。
- 方向与之前所有输入侧/架构侧结果相反：增益集中在**真正依赖图的问题**，不是低 g/语言先验。
- overall +0.0301 [+0.0087,+0.0515]，也显著；三 seed 一致（+0.0268/+0.0225/+0.0410）。
- 200/200 无损，准确数仍 162/200；收益是接受长度，不是答案正确率。

## 3. 预期与边界

- **结果：E2E-TV 三 seed 一致提高 overall 和 high-g，低 g 不变**；这是目前唯一 visual-specific 的正结果。说明“直接优化 prefix acceptance/overlap”比 token-level Bernoulli+k3 更能推动视觉依赖 token 的接受。
- 效应量不大：高 g +0.046（base 6.14 的 +0.76%），overall +0.030（+0.48%）；准确数不变。
- **边界**：三 seed 同一 test200（复用集）、单数据集；E2E-TV 是 **realized-path 辅助项**，不是完整 SpecOPD；没有 branch-parallel sampling/tree mask/indexed replay；权重 10 未调参。不能据此声称完整 SpecOPD 复现，也不能外推到独立集。
- 下一步：① 在 video fresh30 上复现；② 做完整 SpecOPD branch-parallel（吞吐+更正确的 branch 监督）；③ 调 E2E-TV 权重/与 OPD 的比例；④ 在独立 confirm 集上确认。

## 4. 产物与复现

训练：

```bash
bash /mnt/local/mcy/GLANCE/results/visual_opd_20261105/launch_e2etv.sh
# 第二/第三 seed：launch_e2etv_s2.sh / launch_e2etv_s3.sh
```

评测：

```bash
cd /home/mcy/Projects/GLANCE
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu-uuid>   /home/mcy/.conda/envs/fin_research/bin/python repro/scienceqa_image/full_epoch/evaluate.py   --run /mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020   --head /mnt/local/mcy/GLANCE/results/visual_opd_20261105/e2etv/step60   --tag e2etv_step60 --split test --gpu 0 --head-dtype float32   --ar-cache /mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020/eval/ar_test.pt   --repeat-spec 0 --out /mnt/local/mcy/GLANCE/results/visual_opd_20261105/eval/e2etv_step60.json
```

- checkpoints：`e2etv/step60`、`e2etv_s2/step60`、`e2etv_s3/step60`
- eval：`eval/e2etv_step60.json`、`eval/e2etv_s2_step60.json`、`eval/e2etv_s3_step60.json`
- 代码：`repro/opd_official_kl/runtime.py`（`e2e_tv_weight`）、`cli.py`（`--e2e-tv-weight`）
- 训练预算：`/mnt/local/mcy/GLANCE/results/visual_opd_20261102/budget.json` 的 `e2etv` / `e2etv2` / `e2etv3` cases
