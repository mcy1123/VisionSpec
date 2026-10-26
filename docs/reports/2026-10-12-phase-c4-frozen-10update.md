# phaseC4：C2 主权重接入 + 组合门禁 + 冻结 10-update 正式对照

日期：2026-10-12
任务依据：用户 2026-10-12 第四轮指示（C2 主权重接入 → 组合门禁 → 冻结短程对照）
前置：`docs/reports/2026-10-10-update-precision.md`、`docs/reports/2026-10-10-phase-c-corrections.md`

---

## 0. 结论

**在配平预算、统一 context 数值路径、并修好优化器精度之后，paper-faithful OPD（C2）
相对不训练（C0）与同预算 continue-SFT（C1）都只是"名义为正、CI 跨零"。**
两项名义方向都为正，但效应量 ~0.003 / 0.018 τ、正向视频仅 7/29 与 11/29，
**判为噪声内，不进入复现**；按约定转入数据与目标诊断。

**与 Phase C 相比结果明显不同**：Phase C 里 C2 相对 C0 是 −0.2281
（CI[−0.2830,−0.1738]），现在同一比较是 +0.0027（CI[−0.0203,+0.0274]）。
但**不能据此断言"旧退化主要不是 OPD 语义"**：本轮同时改变了学习率（3e-4→3e-5）、
步数（50→10）、teacher/context 精度（bf16→fp32）、优化器精度（无主权重→fp32 主权重）
与视频曝光（6 条/update→1 条/update），这些因素**未分离**。能说的只是
"**该方法的结论对配置高度敏感**"。

---

## 1. 两处表述澄清（用户 2026-10-10 指正，已并入本项目口径）

1. **CI 跨零不等于不能谈非劣性。** phaseC3 的候选通过了预注册的 `Δτ ≥ −0.02` 操作门槛，
   且报告 CI 下界 `−0.0095` 高于该容忍界限，因此可以说"**通过预设保持性能门槛**"。
   但**候选是在同一 validation 上被筛出来的**，所以不能升级为独立确认的非劣性结论；
   "**未检出显著差异**"也**不等于证明两者相等**。
2. **fp32 主权重仍要写回 bf16 前向参数**，所以**单步写回的方向误差依然存在**。
   修复的关键是"**被舍弃的小更新保留在主权重中、跨步累积**"，而不是"每一步的 bf16 参数
   都能精确实现 fp32 更新"。另外，**逐张量余弦的中位数不是整个模型更新向量的夹角**——
   它是各张量各自展平后取中位数，只作方向退化的量级指示。

---

## 2. 任务 1：C2 主权重接入与集成验证

`repro/opd_train_video_fullkl.py` 新增 `--master-weights {none,cpu,gpu}`（默认 `none`
= 与历史 OPD 逐位一致），复用 `repro/master_optim.py`。四项不变量实测：

| 检查 | 结果 |
|---|---|
| **裁剪位置** | 改为在 **fp32 主权重梯度**上做（`master.sync_grads()` → `grad_norm()` → `master.step()` 内部按 `self.clip` 裁剪）。bf16 梯度裁剪与 fp32 梯度裁剪不是同一个数，必须单侧统一 |
| **主权重 ↔ 模块参数同步** | 写回后 58/58 张量一致，**worst \|d\| = 0.000e+00**（按 bf16 舍入比较） |
| **teacher 不更新** | optimizer 的 58 个张量与 head 的参数集合**完全相同**，target 从不进入优化器 |
| **checkpoint 保存/恢复** | 每个 checkpoint 目录含 `optimizer_master.pt`：fp32 参数 + `exp_avg`/`exp_avg_sq`/`step`（实测 `exp_avg` 非零、`step=3`）。`load_master` **逐位恢复**（0/58 不一致）且 bf16 前向参数不变；**对照：朴素续训（重建 master）的 optimizer state 为空 → Adam 从零重启** |

`test_opd_train_fullkl.py` 仍 **17/17 通过**。

---

## 3. 任务 2：组合门禁 `fp32 teacher + bf16 head + CPU 主权重`

命令要点：`--teacher-dtype float32 --head-dtype bfloat16 --master-weights cpu
--updates 3 --anchors-per-update 24 --anchors-per-video 4 --clip 1.0`（GPU 0，独立输出目录）。

| update | N_acc | Z | KL_acc | KL_rej | ‖g‖(fp32, clip 前) | clip 比例 | peak pass1 | **peak pass2** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 75.70 | 0.3237 | 5.9738 | 3.644 | 0.7255 | 38.03 | **39.69 GiB** |
| 2 | 59 | 73.53 | 0.4169 | 6.5949 | 3.767 | 0.7345 | 38.79 | **39.92 GiB** |
| 3 | 50 | 77.86 | 0.2164 | 5.9722 | 3.097 | 0.6771 | 38.18 | **39.61 GiB** |

- **峰值 39.92 GiB < 44 GiB**，且比"fp32 teacher + bf16 head 但**无**主权重"的
  43.84 GiB **多出约 4 GiB 余量**（主权重与 Adam 状态在 CPU，不占显存）。
- **host RSS 35.1 GiB**（含 11.72 GiB 的 fp32 主权重 + Adam 矩 + CPU 侧梯度）。
- 两流均非空、无空 update、无 EOS 排除；逐 update 计算约 21 s（pass1 15–17 s + pass2 4.5 s）。
- 门禁产物在 `/tmp/gate_combo`（一次性）；**正式实验从头重新开始，不复用该目录**。

---

## 4. 任务 3 的前置修正：C1 与 C2 的 context 路径原本不一致

用户指出的关键细节**成立**。实测：

- `train_head_video.py:77` 用 `common.load_target(device=dev)` → **bf16 target**；
- `opd_train_video_fullkl.py:1145` 用 `common.load_target(device=dev, dtype=teacher_dtype)`
  → 设 `--teacher-dtype float32` 时是 **fp32 target**。

而 head 的 context 正是由 target 前向产生的，所以 **phaseC3 的 C1 不能充当"只差 loss"的
严格对照**。修正：

1. `train_head_video.py` 新增 `--target-dtype {bfloat16,float32}`，并打印实际 dtype；
2. 修掉随之暴露的 dtype 边界：fp32 target 的 `ctx`、embedding 表要转成 head dtype，
   `lm_head` 通过一个只做入参转换的 shim 传入（不改共享的 `train_head.sequence_loss`）；
3. **两臂统一到 fp32**（C1 `--target-dtype float32` == C2 `--teacher-dtype float32`）。
   phaseC3 的原控制结果**保留为参考**，不覆盖。

---

## 5. 任务 3：冻结的 10-update 正式对照

### 5.1 冻结配置

| 项 | C1 | C2 |
|---|---|---|
| 起点 | `head_video439_aligned/step1000` | 同 |
| context 数值路径 | `--target-dtype float32` | `--teacher-dtype float32` |
| 优化器 | `--master-weights cpu`（fp32 主权重 + fp32 Adam） | 同 |
| lr / schedule | 3e-5 / 常数 | 同 |
| AdamW / clip | betas (0.9,0.95), wd 0, clip 1.0 | 同 |
| updates | **10** | **10** |
| video 曝光 | 1 条/update（共 10） | 1 条/update（共 10） |
| anchors/update | 24 | 24 |
| rollout | —（SFT 目标） | **当前策略在线 chain rollout** |
| checkpoint | step5, step10 | step5, step10 |
| seed | 0 | 0 |

**正式评测固定 update 10**（不按中途结果挑 checkpoint）。`phaseC3/.../step10` 是 C1 的
**结果**，未被用作任何初始化。

### 5.2 真实监督预算（用户要求记录）

| arm | anchors | **真实监督量** |
|---|---:|---|
| C1 | 240 | **2404 个监督位置**，权重和 1008.4 |
| C2 | 240 | **3600 个 offset**（accepted 485 + rejected 3115） |

**比 1.50×**（C2/C1）。两臂的视频曝光与 updates 已配平；"anchor" 的定义仍不可比
（C1 = 一个被监督位置，C2 = 一个 draft block），这是方法差异本身。

### 5.3 主结果：完整 `video_fresh30`（fp32, b63k8, 跑到 EOS, `--ar`，n=29）

| arm | n | mean τ | median τ | lossless |
|---|---:|---:|---:|---:|
| C0 起点（不训练） | 29 | 3.7377 | 3.7778 | 29/29 |
| C1 continue-SFT u10 | 29 | 3.7225 | 3.7681 | 29/29 |
| **C2 paper-faithful OPD u10** | 29 | **3.7403** | 3.7778 | 29/29 |

逐视频配对（bootstrap 单位 = 视频，10k）：

| 对比 | mean Δτ | median | 95% CI | t | 正向 |
|---|---:|---:|---|---:|---:|
| **C2 − C0** | **+0.0027** | +0.0000 | **[−0.0203, +0.0274]** | +0.21 | **7/29** |
| **C2 − C1** | **+0.0179** | +0.0000 | **[−0.0146, +0.0543]** | +0.99 | **11/29** |
| C1 − C0 | −0.0152 | +0.0000 | [−0.0454, +0.0128] | −1.00 | 9/29 |

其余参考：τ 与 C0 不同的视频数 C1 = 20/29、C2 = 16/29。

### 5.4 判定

**两项名义方向都为正**（C2 同时高于 C0 与 C1），按用户预定的分支进入"判断效应量与不确定性"：

- 效应量 **+0.0027** 与 **+0.0179 τ**，相对基线 3.74 分别为 0.07% 与 0.48%；
- **两个 CI 都包含 0**（t = +0.21 / +0.99），无法与"没有差异"区分；
- **正向视频数**只有 **7/29**（C2−C0）与 **11/29**（C2−C1）；
  与 C0 **发生 τ 变化**的视频数是 **16/29（C2）** 与 **20/29（C1）**——这两个数不同，
  前者是"变得更好"的条数，后者是"任何变化"的条数。

> **判定：噪声内。不足以宣称视频 OPD 有效，也不值得进入复现（多 seed / 加步数）。
> 按约定转入数据与目标诊断。**

---

## 6. 边界（本轮未做）

- **新测试集保持未评测**（`results/phaseC/testset_candidates/`）。
- 未加 seed、未扫超参、未加 OPD 步数。
- 全部结论仍是 **validation 口径**（`video_fresh30` 已用于 A4 选点）。
- 计时未作为速度结论：两次评测虽在同一天，仍受他人 GPU 任务影响，且 τ 与 ms/tok 的
  方向历史上曾不一致，本轮不报加速。

## 7. 产物与复现

| 类型 | 路径 |
|---|---|
| 训练入口（C2 主权重） | `repro/opd_train_video_fullkl.py`、`repro/master_optim.py` |
| 训练入口（C1 context dtype） | `repro/train_head_video.py` |
| 启动脚本 | `repro/run_phaseC4_10upd.sh`、`repro/run_phaseC4_eval.sh` |
| checkpoint | `checkpoints/phaseC4_C1/step{5,10}`、`checkpoints/phaseC4_C2/step{5,10}`（各含 `optimizer_master.pt`） |
| 训练日志 | `results/phaseC4/C1.{log,jsonl}`、`results/phaseC4/C2.{log,jsonl}` |
| 评测结果 | `results/phaseC4/phaseC4_{C1,C2}_update10_fresh30.{json,log}` |
| 组合门禁 | `/tmp/gate_combo`（一次性，未保留） |

```bash
cd /home/mcy/Projects/GLANCE/repro
R=/mnt/local/mcy/GLANCE
./run_phaseC4_10upd.sh      # C1 然后 C2（串行）
./run_phaseC4_eval.sh       # 固定 update10 的完整 fresh30 评测
```

## 8. 下一步

按用户约定：**转入数据与目标诊断**（改数据/改目标），而不是继续调 KL 权重或加 OPD 步数。
