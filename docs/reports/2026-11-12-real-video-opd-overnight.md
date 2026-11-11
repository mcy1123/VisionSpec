# 真实视频长生成 OPD：夜间执行报告（2026-11-12 08:00 截止）

## 0. 截止、分支与结论

- 用户截止：**2026-11-12 08:00:00 Asia/Shanghai（UTC+08:00）= 2026-11-12T00:00:00Z**。
- 训练停止阈值：**2026-11-12 03:30:00 +08:00 = 2026-11-11T19:30:00Z**；实际 λ10 在 **01:56 +08** 因保护评测的 `STOP_TRAINING` 在更新边界停止。
- 仓库/分支：`mcy1123/VisionSpec`、`fix/visual-opd-long-video-protocol`；已确认包含 **ab6e2fa**（`Add deadline-aware stopping and overnight OPD execution protocol`）。
- 隔离 worktree：`/home/mcy/Projects/GLANCE-real-video`（`run/real-video-20261111`）。

**一句话结论**：真实 80/40 Video-MME-derived 上，20 updates 的真实视频 OPD 训练把 128-frame fresh dev20 的 τ 从 **6.578 → 7.070（λ0）→ 7.171（λ10）**；λ10 相对 λ0 **+0.101 [0.040,0.170]**，20/20 无损。但同一 head 跑 **64-frame 输入更高：原 head 8.341，128-trained λ10 head 8.970**；所以这条任务上 **64 帧优于 128 帧**，128 训练的正增益不能掩盖预算选择错误。输出全部撞 1024 cap，λ10 只到21 updates，dev20 不是独立 test。

## 1. 执行门禁

- CPU 七组回归：**60 passed, 1 skipped, 30 subtests passed**（含 `test_video_training_stop.py`）。
- 数据：Video-MME 原视频 900 个；medium 且 240–600s 的 eligible **200**；固定 **80 train / 40 dev**，按源视频隔离。
- 历史重叠审计：旧 VideoDetailCaption `v_*` 与 Video-MME `videoID` 交集 **0**；空排除清单有审计依据。
- 真实帧：每源视频嵌套采样 **64/128 帧**，长边384，不上采样、不复制帧；128f 实际输入 **6070±5 token，其中 visual 5376**（`[64,14,24]` grid）。
- 生成协议：fp32 target、3×47GB balanced、b63/k8、max_new=1024、`eos_stop=immediate`、每条配对 AR、逐 token 无损门禁。
- 4 条 GPU 门禁：**4/4 lossless**，L=6064–6079，τ **5.863/6.860/6.150/6.095**。
- 训练 stop 接口：`--deadline-utc`/`--stop-file` 只在模型加载前和 update 边界检查；`RUN_STATUS.json` 区分 COMPLETED/STOPPED。

## 2. 固定子集任务检查（4 条，未按收益挑选）

固定 dev 子集 indices 0/10/20/30：

| 条件 | τ（4条） | lossless | gen | EOS |
|---|---:|---:|---:|---:|
| 64 真实帧（L≈3073） | **8.093** | 4/4 | 1024 | 0/4 |
| 128 真实帧（L≈6070） | 6.242 | 4/4 | 1024 | 0/4 |
| 128 + zero-visual | 6.272 | 4/4 | 1024 | 0/4 |

- 64 帧比 128 帧高；这是**换 target 输入的差**，不是 selector 因果，不能据此说少 token 更好。
- zero-visual 几乎没有变化；`VisualZeroProbe` 记录每次 zero 了 **5376** 个 visual rows（首轮），不是探针未生效。说明在这个真实长 prompt/该 head 上，draft 的视觉 token 行对 τ 不是主要因果路径，视觉信息更多在文本上下文 hidden 中（与旧合成 128f1x 的 −0.248 不同）。
- 4 条全部 output cap 1024、无 EOS；只是任务/成本检查，不是质量结论。

## 3. 训练：λ0 / λ10 配对

原计划 60 updates/臂；实测 **262–303 s/update**，按第8节“不得越过截止”的原则，改为先跑 **35 updates 计划的同 schedule 前35**：

| arm | planned | completed | loss 首/末 | mean s/upd | status |
|---|---:|---:|---:|---:|---|
| seed20261111 λ0 | 35 | **35** | 2.956 / 2.799 | 262.2 | COMPLETED |
| seed20261111 λ10 | 35 | **21** | 3.874 / 3.794 | 264.9 | STOPPED (`stop_file`) |

- 两臂同一 `VS_HEAD`（OPD3e step220）、同一 `train_128`、同一 schedule 前35、同 lr=1e-5、同 4 QA/update、同 fp32 target/master/optimizer；λ只控制现有 `e2e_tv_weight`（top8-overlap prefix proxy），两臂都保留 Bernoulli/k3 OPD。
- λ10 在 step20 后有 step21；λ0 有 step20/35。**公平比较只用两臂都有的 step20**；不把 λ0 step35 对 λ10 step21 当等预算。
- 真实 optimizer 更新：35 + 21 + 1（先前1-update试跑）= **57**，低于240上限；未做第二 seed，未追加 smoke。

## 4. 配对评测（fresh dev20，固定每2条取1条）

| arm/head | τ | 有效 tokens/round | lossless | gen | EOS |
|---|---:|---:|---:|---:|---:|
| 原 head（baseline128） | **6.578** | 6.542 | 20/20 | 1024 | 0/20 |
| λ0 step20 | **7.070** | 7.018 | 20/20 | 1024 | 0/20 |
| λ10 step20 | **7.171** | 7.131 | 20/20 | 1024 | 0/20 |

配对 bootstrap（20 视频，10k resample）：

| 对比 | Δ τ | 95% CI |
|---|---:|---:|
| λ0 − baseline | **+0.4916** | **[+0.3868,+0.6009]** |
| λ10 − baseline | **+0.5929** | **[+0.4692,+0.7290]** |
| **λ10 − λ0** | **+0.1012** | **[+0.0402,+0.1699]** |

- **训练本身是强正结果**：真实长视频 target 上 20 updates 即显著提高接受长度。
- **λ10 相对 λ0 也有可测增益**：+0.101 [0.040,0.170]；方向支持现有 top8-overlap 辅助项。
- 20/20 每条 AR/SPEC 逐 token 一致；但所有输出撞 1024 cap、无 EOS，τ 是长输出上的接受效率，不是完整长报告质量结论。
- 评测是开发集固定子集，不是独立最终 test；不能把 +0.59 当论文泛化数字。

## 4.1 64 vs 128 真实帧（同一原 head，dev20）

| 输入 | τ | lossless | 备注 |
|---|---:|---:|---|
| 64 真实帧（L≈3073） | **8.341** | 20/20 | 原 OPD3e head |
| 128 真实帧（L≈6070） | 6.578 | 20/20 | 原 OPD3e head |
| 128-trained λ10 head **跑64输入** | **8.970** | 20/20 | 跨 token 预算迁移 |
| 128-trained λ10 head 跑128输入 | 7.171 | 20/20 | 本训练目标 |

- 64−128（原 head）**+1.763 [+1.116,+2.392]**：**更多真实帧并没有自动更好**；这是换 target 输入，不是 selector 消融。
- 128-trained λ10 head 在64输入上 **8.970**，比原64 head **+0.629 [+0.472,+0.789]**，比同 head 的128输入 **+1.799 [+1.077,+2.537]**。
- 结论：**64 帧是这条任务上更有利的输入预算**；128 训练仍能改善128基线，但20 updates 后仍低于64基线。下一步优先做64帧同预算训练，而不是继续堆128。

## 4.2 zero-visual（λ10 step20，dev20）

| arm | τ | 相对 real |
|---|---:|---:|
| λ10 real | 7.171 | — |
| λ10 zero-visual | 7.078 | **−0.093 [−0.200,+0.010]** |

- 20/20 spec ids 与 AR ids **完全相同**；zero 只轻微改变接受路径，不改变输出文本。
- 视觉 token 行的因果贡献小且边界；不能把该任务上的 τ 增益归因于 head 直接读取 visual rows，更多来自 target 上下文。

## 4.3 64-frame 追加训练（截止前启动，评测可能未完成）

- 在已完成主配对后，用同一原 head 追加 **64-frame λ10、10 updates**（`arms/seed20261111_lambda10_64_u10`，lr1e-5、4 QA/update、real/global_pooled）。
- 目的：验证64输入不仅是推理预算更优，也能继续训练。该臂不计入主配对；总真实 optimizer updates 增至 **67**（仍≤240）。
- 截止时该臂的 dev20 64-frame 评测若未完整，不计入结论，只作为下一步实验。

## 5. 与旧结果的关系

- 旧合成 128f1x 9k（16 unique 帧复制8×）证明“原生构造 + 长上下文”可到 τ3.87；本任务的真实 128 帧是不同输入，τ 6–7 与旧 fresh30 不可直接比。
- 本任务首次把 **真实 PTS 采样、真实 80/40 隔离、real time 编码、global_pooled、deadline-aware stop** 串起来并跑出可用的正配对。
- 旧“76% 视觉信息在文本 hidden”不作为本报告结论；本任务 zero-visual 只说该 head 的 visual-row 因果很小，且只有4条。

## 6. 限制

- λ10 未完成35；与λ0只能比 step20。λ0 step35 的额外15 updates没有同预算对照。
- dev20 是固定子集，不是40条全 dev；未做第二 seed；未做独立 test。
- 所有输出 1024 cap、`hit_eos=0`；长报告质量、事件顺序、时间边界仍需人工审阅/更长 cap。
- 64 帧任务检查高于128，说明“更多真实帧”不自动更好；需要更长训练/或 64 对照，当前不能归因于某个 selector。
- zero-visual 探针只置零 head 的 visual rows，文本上下文仍含视觉信息；这是下界，不是完全去视觉。
- `training_state.pt` 仍不支持 resume；每臂一次进程，未把 fresh 重启伪装成续训。

## 7. 产物

- 数据：`/mnt/local/mcy/GLANCE/results/real_video_20261111/data_parallel/`（80 train/40 dev，64/128）
- 真实帧 cache：`/mnt/local3/mcy/GLANCE/results/real_video_20261111/frame_cache_parallel/`
- 训练：`arms/seed20261111_lambda0/step20,step35`、`arms/seed20261111_lambda10/step20,step21`
- 评测：`eval/dev20_baseline128.json`、`eval/dev20_lambda0_20.json`、`eval/dev20_lambda10_20.json`、`eval/gate_128.json`、`eval/baseline_64/128`、`eval/zero_visual_128.json`
- 日志/协议：`logs/`、`PROTOCOL.json`、`STATUS.md`、`RUN_STATUS.json`
- 代码：worktree `run/real-video-20261111`，基于 `ab6e2fa`；新增并行缓存构建与 PyAV ffprobe fallback（仅构建侧，未改 decoder/verifier）。

## 8. 建议

**值得继续**：真实长视频 target 的 OPD 训练方向有明确收益，λ10 也优于 λ0。

**下一步（按优先级）**：

1. 补齐 λ0/λ10 同预算 35–60 updates（当前 λ10 只有21；先跑满35再比）。
2. 40 条全 dev + 第二 seed；固定 20/40 子集不重复调参。
3. 做 64 vs128 的同预算训练对照，判断“真实帧数”而非输入差。
4. 加长 max_new / 自然 EOS 任务，人工审阅事件顺序，确认不是 cap 下的接受率假象。
5. 若继续视觉结构：先做 EvidenceReadout/时间组 memory 的 CPU 原型与单卡 gate，不直接把旧 B/C 搬进来。
