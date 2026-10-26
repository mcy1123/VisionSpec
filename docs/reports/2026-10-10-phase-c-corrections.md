# Phase C 报告补正 + fp32/bf16 混合精度门禁 + continue-SFT 小步校准

日期：2026-10-10
起因：用户 2026-10-10 核对代码后指出 Phase C 报告中三处解释不符合代码/配置。本文件逐条
用代码、运行配置与实测给出结论，并对历史报告追加补注。**已测得的 τ 排序 C0 > C1 > C2 不变**，
改变的是原因分析与下一步选择。

---

## 第 1 步：三项表述审计

### 1.1 「C1 只监督 1200 个位置，C2 是它的 15 倍」——**成立，报告写错了**

**代码依据**：`train_head.build_blocks`（`repro/train_head.py:198-229`）把**每一个 anchor
展开成一个 `block_size` 宽的 block**：

```python
visible   = offsets.unsqueeze(0) < reveals.unsqueeze(1)   # (N, B)
in_bounds = index <= (seq_len - 1)
super_mask = (~visible & in_bounds).to(torch.float32)      # 每个 anchor 监督 B-reveal 个位置
```

`sequence_loss` 再以权重和归一：`loss = (nll * flat_w).sum() / flat_w.sum().clamp_min(1.0)`
（`train_head.py:323-324`），其中 `flat_w = weights[super_mask]`，
`w_k = DECAY ** (k - reveal)`。所以 1200 是**累计 anchor 数**，不是监督位置数。

**实测**（`repro/audit_supervision_counts.py`，逐位复现 `train_head_video.py` 的
`np.random.default_rng(seed)` 置换与子采样顺序 + 每步 `torch.Generator().manual_seed(seed+step)`，
只用纯 block 记账、不需要模型）：

| 配置 | anchors | **监督位置数** | **权重和（loss 分母）** | 位置/anchor |
|---|---:|---:|---:|---:|
| Phase C C1（max-anchors **24**）× 50 步 | 1200 | **11648** | **4928.0** | 9.71 |
| A4（max-anchors **96**）× 50 步 | 4800 | **46848** | 19767.4 | 9.76 |
| 本次校准（max-anchors 96）× 25 步 | 2400 | **23178** | 9840.8 | 9.66 |
| Phase C C2 × 50 updates | 1200 | 18000 offsets | n/a（OPD 归一不同） | — |

**结论**：`C2/C1 = 18000 / 11648 = **1.55×**`，不是 15×。「15 倍监督预算」不成立。
**另一条更该注意**：Phase C 把 anchor 上限从 A4 的 96 降到 24，使 C1 每步只看
**233 个位置**而 A4 看 **937 个**——C1 的监督密度只有 A4 的 **1/4**，这是 C1 退化的
一个未分离混杂（见 1.4）。

### 1.2 「fp32 teacher 单卡不可行」——**下得太早，实测可放**

Phase C 报告只测了 **fp32 teacher + fp32 head + Adam**（OOM）。**fp32 teacher + bf16 head**
当时遇到的是 dtype 接口报错，从未做显存测试。本轮修好 dtype 边界后实测（见第 2 步）：
**3 次真实 optimizer updates 通过，峰值 43.84 GiB < 44 GiB 目标**。
所以正确表述是「fp32 teacher + **fp32 head** 不可行」，不是「fp32 teacher 不可行」。

### 1.3 「本轮 anchors 全由 fp32 rollout 冻结复用」——**不成立，是 B0.5 的设置被误写成 Phase C 的**

**代码依据**：`repro/opd_train_video_fullkl.py:634-637`，每个 update 都调用

```python
with torch.no_grad():
    anchors = opd_online_rollout.rollout(
        head, target, fused, full_ids, prompt_len=stored, verify=verify,
        max_rounds=max_rounds, head_dtype=head_dtype)
```

即 **anchors 由当前策略在线 rollout 产生**，没有加载任何固定的 fp32 anchor 池。
运行配置也印证：`opd_fullkl_config.json` 里 `teacher_dtype_actual = torch.bfloat16`、
`verify = chain`，没有 anchor-pool 字段。

**推论（比原表述更严重）**：B0.5 测到的「**bf16 rollout 会重选 anchors，8/8 视频不同**」
在 Phase C 里是**完全生效**的，而不是「被部分控制」。C2 的 bf16 臂**同时**是
「不同精度」和「不同轨迹」，二者在本轮**没有分离**。

### 1.4 「C1 下降说明 50 步太短 / 预算本身有问题」——**过度解读**

C1 相对 A4 同时改了四件事：`lr 3e-4`（A4 也是 3e-4，但 A4 是 1000 步，C1 是 50 步）、
**重新初始化 Adam**、**anchor 上限 96→24**、**重新开始采样顺序**。这条下降至少可能来自：

- 1.1 的监督密度降到 1/4；
- 在 A4 的泛化峰值之后继续过拟合（A4 已知 step1000→1500 已经在降）；
- 低步数下 Adam 二阶矩尚未稳定；
- 以及学习率相对「只走 50 步」偏大。

**「50 步太短」与「延长到论文 8 epochs 就会恢复」都没有证据支持**，本轮不做该推断。

### 1.5 历史报告补注

`docs/reports/2026-10-10-phase-c-seed0.md` 与
`docs/reports/2026-10-09-b1-precision-and-phasec-config.md` 已追加补注段，指明上述
1.1–1.4 覆盖的原文表述作废范围。**τ 数字本身不变，无需重跑。**

---

## 第 2 步：修通 fp32 teacher + bf16 head（三次真实 update）

### 2.1 修了什么

dtype 边界有三处，缺一处就报 `RuntimeError: expected mat1 and mat2 to have the same dtype`：

| 位置 | 问题 | 修复 |
|---|---|---|
| `opd_online_rollout.rollout` 的 `target_hidden` | fp32 target 的 fused hidden 直接进 bf16 head 的 `fc` | 新增可选参数 `head_dtype`（默认 `None` = 旧行为逐位不变），进入前把 `ctx` 与 noise embedding 转到 head dtype |
| `rollout` 里的 `target.lm_head(hidden)` | bf16 hidden 撞 fp32 `lm_head` | 投影前转到 `lm_head.weight.dtype` |
| `opd_train_video_fullkl.draft_block_logits` 包装器 | 只转了 fused/embedding，`lm_head` 故意没转（原注释），fp32 target 下会炸 | 给 `lm_head` 包一层按权重 dtype 自动转换的调用 |

KL 仍在 fp32（`opd_fullkl.full_kl_forward/reverse` 内部 `.float()`），**精度契约未改**。
`opd_online_rollout.rollout` 的改动是**加法式**的（新 kwarg 默认 None），旧 2×2 路径行为不变。

### 2.2 实测（GPU 0，24 anchors/update，3 次真实 update）

```bash
cd /home/mcy/Projects/GLANCE/repro
PYTORCH_ALLOC_CONF=expandable_segments:True \
/home/mcy/.conda/envs/fin_research/bin/python -u opd_train_video_fullkl.py \
  --records $R/data/video_greedy_499.pt --pool $R/data/pool_499.parquet \
  --init-from $R/checkpoints/head_video439_aligned/step1000 \
  --out /tmp/gate_fp32_bf16 --updates 3 --anchors-per-update 24 --anchors-per-video 4 \
  --teacher-dtype float32 --head-dtype bfloat16 --max-rounds 4 \
  --lr 3e-4 --save-every 0 --no-final-save --log-every 1 --gpu 0 \
  --log-file /tmp/gate_fp32_bf16/log.jsonl
```

| update | N_acc | Z | KL_acc | KL_rej | ‖g‖ clip 前 | clip 比例 | peak pass1 | **peak pass2** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 75.70 | 0.3237 | 5.9738 | 3.656 | 0.727 | 38.03 | **42.85 GiB** |
| 2 | 48 | 77.48 | 0.2834 | 5.7926 | 3.391 | 0.705 | 42.70 | **43.84 GiB** |
| 3 | 56 | 73.41 | 0.2699 | 5.7080 | 2.953 | 0.661 | 42.09 | **43.51 GiB** |

**门禁**：3/3 update 完成（覆盖 rollout → teacher scoring → head backward → `optimizer.step()`
与 Adam 状态初始化）；dtype 实测 `teacher float32 / head bfloat16`；梯度有限非零；
`n_acc_offsets`/`n_rej_offsets` 两流均非空；无 EOS 排除；峰值 **43.84 GiB < 44 GiB**。

**必须记录的风险**：43.84 GiB 距 44 GiB 目标只剩 **0.16 GiB**、距 47.53 GiB 可用剩 3.7 GiB。
本次只跑了 3 个 update、且 `pass1_held` 只有 2.3–2.6 GiB。**长跑（50 updates）是否始终不 OOM
未被证明**；若要正式跑，需要降低 `--anchors-per-update`（减少同时持有的视频数与 prefill 峰值）
或接受该风险并留监控。

**结论**：混合精度路径可跑，因此**不必**为大显存卡才可以做 fp32 teacher。

---

## 第 3 步：continue-SFT 小步校准（lr 3e-5 vs 1e-5）

### 3.1 固定量（两臂逐项相同）

起点 `head_video439_aligned/step1000`；439 行 train；`--feature-path cached`；seed 0
（`np.random.default_rng(0)` 决定视频顺序**与** anchor 子采样 → **两臂看到完全相同的行与
anchor 集合**）；**anchor 上限回到 A4 的 96**（移除 1.1/1.4 的 96→24 混杂）；
block/reveal 先验未改；AdamW betas (0.9,0.95)、wd 0、clip 1.0、常数 lr；
25 updates，保存 5/10/25。**唯一变量：lr ∈ {3e-5, 1e-5}**。
预注册规则在跑之前写入 `docs/reports/2026-10-10-phase-c2-prereg.md`。

命令：`repro/run_phaseC2_calib.sh`（两臂串行），产物
`checkpoints/phaseC2_C1cal_lr{3e-5,1e-5}/step{5,10,25}`。
真实监督预算：2400 anchors → **23178 个监督位置，权重和 9840.8**（约 927 位置/步）。

### 3.2 监控子集筛选（8 条，与 Phase C 同协议，n=8，Δτ = 候选 − C0）

| 候选 | mean Δτ | median | 95% CI | 正向 | 权重相对变化 | 保留 |
|---|---:|---:|---|---:|---:|---|
| lr3e-5 u5 | +0.0000 | 0.0000 | [+0.0000,+0.0000] | 0/8 | 3.57e-05 | YES |
| lr3e-5 u10 | **+0.0058** | 0.0000 | [+0.0000,+0.0174] | 1/8 | 4.86e-05 | YES |
| lr3e-5 u25 | −0.0021 | 0.0000 | [−0.0237,+0.0174] | 1/8 | 6.70e-05 | YES |
| lr1e-5 u5 | +0.0000 | 0.0000 | [+0.0000,+0.0000] | 0/8 | 6.08e-06 | YES |
| lr1e-5 u10 | +0.0000 | 0.0000 | [+0.0000,+0.0000] | 0/8 | 7.47e-06 | YES |
| lr1e-5 u25 | +0.0000 | 0.0000 | [+0.0000,+0.0000] | 0/8 | 9.49e-06 | YES |

**6/6 通过阈值 −0.02**。按冻结的 tie-break（mean Δτ 最高 → 更小 lr → 更早 update）
**入选：lr 3e-5, update 10**。

### 3.3 一个必须点名的性质：它是「几乎不训练」而不是「稳定训练」

对照权重相对变化（`mean|Δ|/mean|w|`，对起点 checkpoint）：

| 臂 | 相对变化 | 与 lr 3e-4 之比 |
|---|---:|---:|
| **Phase C C1（lr 3e-4, u50）** | **8.69e-03** | 1× |
| calib lr3e-5 u25 | 6.70e-05 | 130× 小 |
| calib lr3e-5 u10 | 4.86e-05 | 179× 小 |
| calib lr1e-5 u25 | 9.49e-06 | 915× 小 |

且 58 个张量中只有 40 个发生变化（lr 3e-4 是 45 个）。**所以「保持起点」在很大程度上是
「权重几乎没动」**：lr 1e-5 / 25 updates 的 τ 在 8 条视频上**逐条与 C0 完全相同**。
这是有效的「不退化」设置，但**不能读成「找到了一个能稳定继续训练的配方」**——
它更接近「小于某阈值时什么也没发生」。

---

## 第 4 步：入选候选的完整 validation 复核（n=29）

| arm | n | mean τ | median τ | lossless |
|---|---:|---:|---:|---:|
| C0（aligned step1000，不训练） | 29 | 3.7377 | 3.7778 | 29/29 |
| **候选 lr3e-5 u10** | 29 | **3.7438** | 3.8039 | 29/29 |
| C1 lr3e-4 u50（Phase C） | 29 | 3.6579 | 3.6471 | 29/29 |

逐视频配对（bootstrap 单位 = 视频，10k）：

| 对比 | mean Δτ | median | 95% CI | t | 正向 |
|---|---:|---:|---|---:|---:|
| **候选 − C0** | **+0.0061** | +0.0000 | [+0.0000, +0.0137] | +1.76 | **3/29** |
| 候选 − C1(lr3e-4) | +0.0859 | +0.1021 | [+0.0373, +0.1329] | +3.46 | 21/29 |

**按冻结规则判定：`+0.0061 ≥ −0.02` → PASS。** 所以本轮**找到了一个在 validation 上
保持起点的 continue-SFT 设置**（lr 3e-5、10 updates、96 anchors/update）。

**读法上的三条限制**：
1. **29 条里只有 3 条 τ 变化**，其余 26 条与 C0 完全相同；CI 下界正好是 +0.0000。
   这是「基本不变」，不是「稳定提升」。
2. 它比 Phase C 的 C1 好 +0.0859，说明**Phase C 的 C1 退化主要是学习率（3e-4）造成的**，
   而不是「继续训练必然退化」——这修正了 Phase C 报告 §8.3 的推断。
3. **这是 validation 上的配方选择，不是最终效果证据**（§5.1 纪律不变，新测试集仍保持未评测）。

### 4.1 计时仍然不可跨臂比较

| arm | spec ms/tok（中位） | AR/spec |
|---|---:|---:|
| C0 | 59.29 | 1.979× |
| 候选 | 27.44 | 2.020× |
| C1 | 63.30 | 1.890× |

候选 τ 更高而绝对时间只有 C0 的 46%，与 Phase C 是同一个争用伪影（三臂评测相隔数小时）。
**不据此宣称任何加速。**

---

## 下一步建议（不自动执行）

按用户指示，**现在才具备讨论重启 C1/C2 对照的前提**，但仍需注意：

1. 候选控制臂是「基本不变」型。按用户预先给定的判据，**C2 仍须同时超过 C0 和 C1** 才有意义；
   而一个「什么都不做」的对照臂会让「C2 超过 C1」变得容易但含义单薄——报告必须同时给出
   C2−C0（真正的有效性）与 C2−C1（相对同预算继续训练）。
2. **fp32 teacher 已可用**（第 2 步），因此重启时应优先用 `--teacher-dtype float32
   --head-dtype bfloat16`，并同时解决 1.3 指出的「精度与 anchors 未分离」问题：
   要么固定 anchors，要么显式报告两者各自的贡献。
3. 若仍要跑较长预算，需先测 fp32 混合精度在更长 horizon 上的显存（43.84 GiB 只剩 0.16 GiB 余量）。
4. 若 C2 仍不达标，按用户指示转向**数据与目标诊断**，而不是继续加 OPD 步数。
