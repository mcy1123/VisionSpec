# 时间分组视觉读出架构 pilot：阴性结果（2026-11-13）

## 0. 结论

**在本次固定的 180-update、单 seed 真实长视频协议下，新的 temporal visual evidence readout 没有带来可测的接受长度增益。**

- A（无 readout）：τ **7.5479**
- B（text evidence）：τ **7.5490**，B−A **+0.0011 [−0.0176,+0.0189]**
- C（visual evidence）：τ **7.5567**，C−A **+0.0088 [−0.0075,+0.0253]**，C−B **+0.0077 [−0.0051,+0.0240]**
- 三臂 **40/40 逐 token 无损**；全部输出 1024 cap、`hit_eos=0`，τ 是 capped-output 接受效率，不是长报告质量。
- 机制探针同样阴性：C 与 bypass/zero 差 **−0.0036 [−0.0226,+0.0153]**，time_shuffle 逐条差 **恰好 0**，wrong-video 只 +0.0021。读出的 visual evidence 实际没有进入可测量的 draft 决策路径。
- 因此本轮**不支持**“时间分组视觉记忆/读出能改善真实长视频 OPD 接受率”，也不支持 C 与 OPD 的特有协同。

## 1. 运行身份与协议

| 项 | 值 |
|---|---|
| 仓库/分支 | `mcy1123/VisionSpec` `feat/temporal-evidence-readout` |
| 验证提交 | **1aa387c** |
| 独立 worktree | `/home/mcy/Projects/GLANCE-temporal-evidence` |
| run 分支 | `run/temporal-evidence-20261112` |
| 原协议 | A/B/C 各连续 60 updates，总预算 ≤180；单 seed |
| 实际 | A/B/C 各 **60/60 COMPLETED**，合计 **180** |
| seed | 20261112 |
| 数据 | 真实 Video-MME-derived，80 train / 40 dev，128 帧；L≈6070，visual 5376 token |
| target | Qwen3-VL-8B-Instruct，fp32 |
| 共同初始 head | 上一轮真实视频 `seed20261111_lambda10/step20`，SHA `d13635c7…` |
| 训练 | real time、global_pooled、lr 1e-5、4 QA/update、e2e-tv 10、fp32 master |
| 原始结果 | `/mnt/local3/mcy/GLANCE/results/temporal_evidence_20261112_094528/` |

**未提交权重、帧 cache、memory bank 或逐视频 eval JSON。**

## 2. CPU 与初始化门禁

- 13 个 CPU 测试文件：**136 passed, 1 skipped, 49 subtests passed**。
- B/C 初始化：新增 **4,212,225** 参数，两臂新增张量 SHA 完全一致；base tensor 逐项 `torch.equal` 未变。
- A/B/C 各 4 条 GPU 门禁：**全部 4/4 lossless**，三臂初始 τ 完全一致 `[6.364, 7.424, 7.111, 4.995]`。
- B/C 在 2×A6000 上的额外门禁也 **4/4 lossless**。

## 3. 训练与墙钟

| arm | readout | GPU | 60-update 墙钟 | mean s/update | loss 首/末 |
|---|---|---|---:|---:|---:|
| A | none | 3×A6000 | **4.03 h** | 242.1 | 3.793 / 2.942 |
| B | text | 2×A6000 | **4.13 h** | 247.9 | 3.793 / 2.916 |
| C | visual | 2×A6000 | **4.09 h** | 245.5 | 3.793 / 2.903 |

- 训练总墙钟约 **12.3 h**；之后 7 组 dev40/机制评测约 **11.8 h**。
- A 是3卡，B/C 是2卡（用户要求留两张 A6000 给他人）：**τ/无损可比，墙钟不可直接横向比较**。C 并没有因为 readout 变快或变慢；完整评测的 spec/end-to-end 也在 1.00× 附近。
- C 的 readout 在 60 步中 **始终有非零梯度**（`gradient_l2` 0.194–0.357，10 个参数张量全部有梯度），并非“梯度断掉”或“模块没接上”。
- 但 gate 的更新量约 **7.5e-6**，output 权重漂移约 **0.14**；读出输出幅度接近零，所以行为上等价于 bypass。

## 4. dev40 配对结果

固定 40 条 dev；每臂同 head step60；bootstrap 按 source video、10k 次。

| arm/condition | τ | 有效 tokens/round | lossless | gen | EOS | truncated |
|---|---:|---:|---:|---:|---:|---:|
| A | 7.5479 | 7.4945 | 40/40 | 1024 | 0 | 40 |
| B | 7.5490 | 7.4971 | 40/40 | 1024 | 0 | 40 |
| C | 7.5567 | 7.5018 | 40/40 | 1024 | 0 | 40 |
| C bypass | 7.5604 | 7.5076 | 40/40 | 1024 | 0 | 40 |
| C zero | 7.5604 | 7.5076 | 40/40 | 1024 | 0 | 40 |
| C time_shuffle | 7.5567 | 7.5018 | 40/40 | 1024 | 0 | 40 |
| C wrong_video | 7.5547 | 7.4998 | 40/40 | 1024 | 0 | 40 |

| 对比 | Δ τ | 95% CI | wins / ties |
|---|---:|---:|---:|
| B − A | +0.0011 | [−0.0176,+0.0189] | 14 / 13 |
| C − A | +0.0088 | [−0.0075,+0.0253] | 14 / 15 |
| C − B | +0.0077 | [−0.0051,+0.0240] | 9 / 24 |
| C − bypass | −0.0036 | [−0.0226,+0.0153] | 12 / 14 |
| C − zero | −0.0036 | [−0.0226,+0.0153] | 12 / 14 |
| C − time_shuffle | 0.0000 | [0,0] | 0 / 40 |
| C − wrong_video | +0.0021 | [−0.0012,+0.0064] | 2 / 37 |

所有 CI 都跨零；C−A/C−B 的点估计小于 0.01 token/round，远小于 protocol 可分辨的架构效应。**没有任何一条支持 visual readout 有正向机制的结论。**

## 5. 机制探针说明了什么

- **bypass**（完全不使用 readout）与 normal C 差异为负且 CI 跨零；zero 与 bypass 数值相同。说明 visual readout 的输出对 draft 行为没有可测贡献。
- **time_shuffle** 逐视频 τ 完全相同（0/40 改变）：时间分组读出没有把时间顺序带进决策。
- **wrong_video** 只 2/40 改变，且幅度 +0.0021；donor bank 没有造成预期的机制性破坏。
- readout 并非“没有梯度”：梯度存在且权重有小幅移动；**结论是它学到的输出幅度/门控太小，不能改变 block 接受路径**，而不是“未训练”。
- 由于 C 与 A/B 的行为几乎相同，本轮不能把 C−A 的正号解释为架构收益；它是采样/评测噪声量级。

## 6. 与计划/执行偏差

1. **C 首次启动在 patch 前失败，0 optimizer update**：训练首个 scoring block 位于 prompt 末尾 root 之前，`evidence.py` 要求“完整 known prompt”而拒绝。已改为“只要所有 visual positions 已覆盖即可”，补丁 `d42e228` 通过相关 53 项测试后 C 从头完成 60 步；失败那次不计入训练预算。
2. **卡数变化**：A 用3×A6000；按用户“留两张 A6000 给别人用”，B/C 改为2×A6000 顺序执行。训练数学/初始 head/数据/schedule 未变，接受与无损比较仍成立；墙钟和部署速度不作为架构结论。
3. **没有第二 seed**：授权为单 seed；不能估计训练随机性。
4. **dev40 已复用**：不是独立最终确认集；没有 test。
5. **全部输出 1024 cap、无 EOS**：不能把 τ 当作长报告质量或自然结束行为。

## 7. 结论与证据局限

**支持**：

- 真实 128 帧、80/40、逐 token 无损的 A/B/C 三段式架构 pilot 可以稳定执行；180 updates 完成。
- 共同初始化、同 schedule、同数据、同 target 的 A/B/C 对照是可执行的；本次没有实现 bug 导致的假阳性。

**不支持**：

- “时间分组 visual evidence readout 改善 OPD 接受率”；
- “C 优于 A/B”；
- “visual readout 与 OPD 有特有协同”；
- 任何把 +0.0088 当作真实收益的表述。

**证据不足/未覆盖**：

- 单 seed、dev40 复用、1024 cap；
- A/B/C 卡数不同；
- readout 结构/初始化幅度是否在更长训练、不同 target、不同帧预算下会有效，本轮不能回答；
- 没有 test，也没有第二数据域。

## 8. 产物

- 原始结果（不入 Git）：`/mnt/local3/mcy/GLANCE/results/temporal_evidence_20261112_094528/`
  - `ANALYSIS.json`、`eval/{A,B,C,C_bypass,C_zero,C_time_shuffle,C_wrong_video}.json`
  - `train_{A,B,C}/step60`、`initial_heads/`、`logs/`
- 代码修复：`scripts/dflash_pkg/dflash-0.1.0/dflash/evidence.py`（visual prefix 覆盖检查），commit `d42e228`
- 小型摘要：`docs/reports/2026-11-13-temporal-evidence-summary.json`
- run 分支：`run/temporal-evidence-20261112`
