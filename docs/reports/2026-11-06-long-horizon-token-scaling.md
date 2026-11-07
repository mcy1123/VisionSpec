# 长训练 / 多数据 / 高视觉 token：2026-11-06 夜跑收口

状态：已完成。**image E2E-TV 在 120 updates / 480 QA 上，对同预算 base120 仍是显著 visual-specific 正结果；视频侧把视觉 token 从 1152 提高到 3456（draft-side memory）没有带来收益。**

## 1. 视频视觉 token 到底是多少

当前视频 cache 只有 **16 帧**（384×288，30 fps）；Qwen3-VL video processor 实际采样 8 帧，`video_grid_thw=[8,24,24]`，`<|video_pad|>` = **1152 token/视频**（8×144）。

为了增加视觉 token，做了两种输入：

| 输入 | 分辨率 | grid | token/视频 |
|---|---|---:|---:|
| 原始 | 384×288 | [8,24,24] | **1152** |
| 1.5× | 576×432 | [8,28,36] | **2016** |
| 2× | 768×576 | [8,36,48] | **3456**（部分 2688） |

- 直接把 1.5×/2× 输入 target 做 eval：在 47GB A6000 上 **OOM**（target fp32 + 2–3.5k context + b63 验证；2× 在 prefill attention OOM）。
- 因此高 token 只能走 **draft-side visual memory**：target 仍看 1152 token，draft 的 visual memory 用 2× 的 2688–3456 token。

## 2. 视频 FOVEA + 2× memory（阴性）

- 从 video OPD3e step220，3 卡，60 updates，FOVEA zero-init；target 输入 1152，draft memory 2×（2688–3456 token）。
- fresh30，30/30 无损：

| arm | real τ | blank τ | visual reliance |
|---|---:|---:|---:|
| OPD3e | 3.7841 | 3.2826 | 0.5015 |
| FOVEA 1152 | 3.7530 | 3.3676 | 0.3854 |
| **FOVEA 2× memory (2688–3456)** | **3.7519** | 3.3676 | **0.3843** |

- FOVEA 2× − OPD3e：real **−0.0323 [−0.0528,−0.0125]**；blank +0.0850；DiD **−0.1173 [−0.1611,−0.0733]**。
- 与 1152-token FOVEA（real 3.7530）几乎完全相同：**token 从 1152 增到 3456（3×）没有带来任何增益**。
- 结论：FOVEA 这条路径不是“token 不够”，而是不会利用更多 token；视频侧 real 反而显著低于 OPD3e。

## 3. image E2E-TV 120 updates / 更多数据（显著正）

- 从 image OPD step3546，4 卡，120 updates（480 QA，full 4726-task pool 随机采样），同 schedule 的 base120 对照。
- test200，两臂均 **200/200 逐 token 无损，162/200 正确**。

| 分组 | base120 τ | E2E-TV 120 τ | Δ | 95% CI |
|---|---:|---:|---:|---:|
| overall | 6.2253 | 6.2672 | **+0.0419** | **[+0.0019,+0.0866]** |
| 高 g | 6.1321 | 6.2089 | **+0.0768** | **[+0.0102,+0.1463]** |
| 低 g | 6.3186 | 6.3255 | +0.0070 | [−0.0440,+0.0589] |
| 最高四分位 | 5.4754 | 5.5603 | **+0.0849** | **[+0.0098,+0.1663]** |
| 最低四分位 | 5.6379 | 5.6261 | −0.0119 | [−0.0815,+0.0422] |

- 同预算对比下，E2E-TV 120 的 overall、high-g、top quartile **CI 都不含 0**；low-g/bottom 不显著。
- 与 60-update E2E-TV（vs base60：overall +0.0301，high +0.0465）相比，**训练/数据拉长到 2× 后 high-g 增益从 +0.0465 增至 +0.0768**；方向一致且更强。
- 结论：**E2E-TV 的 visual-specific 收益随训练/数据预算增长**，这是目前唯一可用的优化。

## 4. 结论

- **image**：把训练从 60 拉到 120 updates、数据从 240 拉到 480 QA 后，E2E-TV 对同预算 base120 仍显著提高 high-g（+0.0768）和 top quartile（+0.0849），低 g 不变；200/200 无损。**这是可用的优化方向。**
- **video**：视觉 token 从 1152 提到 3456（仅 draft-side memory）没有改善 real τ；FOVEA 2× 与 1× 几乎相同，且 real 显著低于 OPD3e。**“更多视觉 token”本身不是杠杆。**
- 目标侧高 token（1.5×/2×）在 47GB 卡上 OOM；若要走 target 侧高 token，需要更大显存/模型并行/更小分辨率预算。
- 视频 cache 只有 16 帧，2× 是**上采样**而非新增真实帧；真正的更多帧需要重建 cache/源视频。

## 5. 产物

- 结果根：`/mnt/local3/mcy/GLANCE/results/visual_long_20261106/`
- image：`image_e2etv120/step120`、`image_base120/step120`
- eval：`eval/image_e2etv120_step120.json`、`eval/image_base120_step120.json`、`compare_e2etv120_vs_base120.json`
- video：`video_fovea_2x/step60`、`eval/video_fovea_2x.json`、`eval/video_fovea_2x_blank.json`
- 运行记录：`NOTES.md`
