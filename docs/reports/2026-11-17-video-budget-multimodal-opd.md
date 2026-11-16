# 视频帧预算作为多模态架构变量：64-frame OPD 与混合预算 pilot（2026-11-17）

## 0. 结论

**64 帧是这条真实长视频任务上更强的训练/推理预算；按 64 帧训练 60 updates 明显超过“128 帧训练 → 64 帧推理”的迁移基线。**

- D64（64 帧训练，60 updates）在 dev40 的 **64 帧输入**：τ **9.3888**，40/40 无损。
- 同一 D64 head 跑 **128 帧输入**：τ **7.3547**，40/40 无损；跨预算明显退化。
- 在共同 20 条上，D64 比“128-trained λ10 step20 → 64 输入”迁移基线 **+0.384 [+0.306,+0.462]**，**20/20 全胜**。
- 混合 64/128 训练 20 updates（M20）可以同时跑两种预算，但**没有超过专用 head**：
  - M20 64 帧 **9.1087**，低于 D64 的 **9.3536**（同 20 条，−0.245 [−0.330,−0.168]）；
  - M20 128 帧 **7.319**，低于 D64 的 **7.357**（−0.038，CI 跨零），也低于 128-trained A 的 **7.531**（−0.212）。
- 因此本轮支持：**预算专用 OPD 适配 > 混合预算单 head**。一个可用的多预算架构是 **budget-conditioned head registry**：64→D64，128→A；在 40 条上分层平均约 **8.468**，而任一单 head 都不能同时达到两端。

## 1. 运行身份与截止

| 项 | 值 |
|---|---|
| 代码 | 最新 `run/temporal-evidence-20261112` / `7a09095`（1aa387c + 报告） |
| worktree | `/home/mcy/Projects/GLANCE-video-multimodal` |
| run 分支 | `run/video-multimodal-20261116` |
| 截止 | **2026-11-17 08:00 Asia/Shanghai（00:00Z）** |
| 本轮新增 | D64 60 updates + M20 20 updates = **80** |
| GPU | 2×A6000：`GPU-8af87d4f…`（0）、`GPU-a474d3e4…`（3） |
| 数据 | 真实 Video-MME-derived 80 train / 40 dev；64/128 真实帧 |
| base | `real_video_20261111/arms/seed20261111_lambda10/step20` |
| 训练 | real time、global_pooled、lr 1e-5、4 QA/update、e2e-tv 10、fp32 master |

## 2. 门禁

- CPU 7 个视频测试文件：**60 passed, 1 skipped, 30 subtests passed**。
- 64 帧单卡 GPU 门禁（4 条）：**4/4 逐 token 无损**，τ [8.296, 9.179, 9.792, 10.300]。
- D64 训练/评测、M20 训练/评测全部 **AR/SPEC 逐 token 无损**。

## 3. 训练与墙钟

| arm | 预算 | updates | mean s/update | 墙钟 | loss 首/末 |
|---|---:|---:|---:|---:|---:|
| D64 λ10 | 64 帧 | 60/60 | 143.4 | **2.39 h** | 3.493 / 3.395 |
| M20 λ10 | 混合 64/128 | 20/20 | 195.1 | **1.08 h** | 3.787 / 3.828 |

- D64 在 2×A6000 上比 128 帧训练快约 40%：64 帧 prompt L≈3070，128 帧 L≈6070。
- M20 每 update 混合 2×64 + 2×128 QA；loss 略升，符合更难的多预算目标。
- 总训练 3.47 h；评测约 5.2 h（D64 dev40 两档 + M20 dev20 两档）。

## 4. D64：64 帧训练前沿

dev40，step60：

| 输入 | τ | 有效 tokens/round | 无损 | EOS | 截断 |
|---|---:|---:|---:|---:|---:|
| **64 帧** | **9.3888** | 9.327 | 40/40 | 0 | 40 |
| 128 帧 | 7.3547 | 7.301 | 40/40 | 0 | 40 |

共同 20 条与迁移基线（上一轮 128-trained λ10 step20 跑 64）：

| 对比 | τ | Δ | 95% CI | wins/losses |
|---|---:|---:|---:|---:|
| 128-trained transfer @64 | 8.9696 | — | — | — |
| **D64 @64** | **9.3536** | **+0.384** | **[+0.306,+0.462]** | **20/0** |

结论：**不是“64 推理更好”而是“64 训练也好”**。上一轮 10-step 64 训练只有 8.735（未超过迁移），这次 60 步到 9.354，说明此前是训练预算不足；同时 D64 跑 128 只有 7.355，**专用预算 head 不能直接跨预算复用**。

## 4.1 D64 的 zero-visual 机制探针（dev20）

D64 训练后，把 head 看到的 visual-token target_hidden 置零：

| 条件 | τ | 无损 | spec/AR 同文 |
|---|---:|---:|---:|
| D64 real @64 | 9.3536 | 20/20 | — |
| D64 zero-visual @64 | **9.0392** | 20/20 | 20/20 |
| Δ | **−0.3143 [−0.4157,−0.2109]** | — | 17/20 下降 |

- 这与上一轮 128-trained λ10 的 zero-visual **−0.093** 不同：**64 帧专用 head 对 visual rows 的依赖明显更强**。
- zero-visual 后的 **spec ids 和 AR ids 仍 20/20 完全一致**；下降来自接受路径变短，不是输出文本变化。
- 架构含义：64 帧预算下视觉证据确实进入 draft 接受路径；不应在该预算上继续削弱/绕过 visual token。**budget-conditioned adapter 应保留 64 的视觉通路，而在 128 上重点解决长上下文融合。**

## 5. M20：混合预算单 head

dev20（与 D64 相同 20 条）：

| head/输入 | 64 τ | 128 τ |
|---|---:|---:|
| D64 专用 | **9.3536** | 7.3573 |
| M20 混合 | 9.1087 | 7.3190 |
| 128-trained A | — | **7.5308** |

配对：

- M20−D64 @64：**−0.2448 [−0.3299,−0.1683]**，19/20 输。
- M20−D64 @128：−0.0384 [−0.1012,+0.0259]，CI 跨零。
- M20−A @128：**−0.2118 [−0.2902,−0.1376]**，17/20 输。

**混合预算训练让一个 head 能跑两档，但没有达到任何一档的专用 head。** 仅 20 updates 的 M20 是低预算 baseline；不能排除更长混合训练会追上，但当前证据不支持“混合单 head 自动适配”。

## 6. 可用的多预算架构

不新增模块、仅用已训练 head 的 **budget-conditioned registry**：

- 64 帧请求 → D64 head（9.389）
- 128 帧请求 → 128-trained A head（7.548）
- 两档分层平均：**8.468**（40 条，每档 40 条；这是分层均值，不是单次混合推理）

对照单 head：

| 单 head | 64 | 128 | 两档平均 |
|---|---:|---:|---:|
| D64 | **9.389** | 7.355 | 8.372 |
| A（128-trained） | 未测全40 | **7.548** | — |
| M20（混合） | 9.109 | 7.319 | 8.214 |

registry 的 8.468 高于 D64 单 head 平均 8.372（+0.096，描述性），说明**按预算路由**比强行一个 head 同时覆盖两档更合理。它不是新训练架构，但给出了下一阶段 **budget-conditioned adapter / mixture-of-heads** 的明确目标。

已实现可执行组件 `repro/evaluation/budget_head_router.py`，在 D64_64 + A_128 两个 eval 上生成 `eval/budget_router_D64_A.json`：两层平均 **8.46838**，两层均 40/40 无损。该 JSON 显式标记为 **known-tier registry / stratified mean，不是单模型混合推理**。

## 6.1 样本自适应 head 路由的 headroom

在相同 40 条上把 A（128-trained）也跑 64 输入，和 D64 做逐视频比较：

| 输入/head | τ | 无损 |
|---|---:|---:|
| 64/D64 | **9.3888** | 40/40 |
| 64/A（128-trained） | 9.3318 | 40/40 |
| 128/D64 | 7.3547 | 40/40 |
| 128/A（128-trained） | **7.5479** | 40/40 |

- 64 上 D64−A：**+0.057**（A−D CI [−0.111,+0.001]），D64 略好但接近。
- 128 上 A−D64：**+0.193 [+0.143,+0.245]**，35/40 胜。
- **逐视频 oracle 选择**（事后知道哪 head 更好）只把 64 从 9.389 提到 **9.435（+0.046）**，把 128 从 7.548 提到 **7.552（+0.004）**。
- 因此**样本自适应 head 路由的 headroom 很小**；主要收益来自“按预算选 head”，不是 per-sample 路由。下一阶段应做 **budget-conditioned adapter**，而不是再训练一个 per-sample gate。

## 7. 与上一轮 temporal readout 的关系

- 上一轮 C（temporal visual evidence readout）在 128 帧、A/B/C 各60下 **C−A +0.0088 [−0.0075,+0.0253]**，bypass/zero/time_shuffle 均无机制效应。
- 本轮没有重复 readout 模块，而是把**帧预算**作为多模态变量：结果说明“统一的视觉 token 读出”不是主要矛盾；**训练/推理的预算匹配**才是当前最大的一阶因素。
- 因此下一阶段不应再加零初始化视觉 readout；应做 **budget-conditioned adapter**，或者更直接的 **budget-specific head + registry**。

## 7.1 视觉依赖随预算变化

- 64 专用：zero-visual **−0.314**（17/20 降），视觉 rows 有实质因果贡献。
- 128-trained λ10：zero-visual **−0.093 [−0.200,+0.010]**（上一轮），视觉 rows 贡献小。
- 这解释了为什么“统一视觉 readout”在 128 上无效，却不能据此说 64 也一样：**视觉依赖是预算相关的**，架构应按预算保留不同通路。

## 8. 限制

- D64 单 seed、dev40；M20 只有 20 updates、dev20；没有第二 seed。
- 所有输出撞 **1024 cap**、`hit_eos=0`；τ 是接受效率，不是长报告质量。
- D64/M20 在 2×A6000；A 的 128 结果是上一轮 3×A6000/60-step，跨运行比较只作分层描述，不作严格同卡配对。
- registry 的“两档平均”是分层指标，不是同一条视频同时用两个预算；不能当作单模型推理结果。
- 没有 test；没有真实生产流量。

## 9. 产物

- 原始结果：`/mnt/local3/mcy/GLANCE/results/video_multimodal_20261116/`
  - `D64_lambda10/step{20,40,60}`、`M20_lambda10/step{10,20}`
  - `eval/D64_{64,128}.json`、`eval/M20_{64,128}.json`、`eval/gate64_base.json`
  - `schedule_64.jsonl`、`train_mixed.parquet`、`schedule_mixed20.jsonl`
- 小型摘要：`docs/reports/2026-11-17-video-budget-multimodal-summary.json`
- 路由组件：`repro/evaluation/budget_head_router.py`；组合结果 `eval/budget_router_D64_A.json`
- 代码：无新算法修改；使用现有 `train_video_9k.py` 与 `eval_video_opd.py`。
- **未提交权重、cache、逐视频 JSON。**
