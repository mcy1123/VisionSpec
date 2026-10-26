# OPD backward 非确定性、独立门禁与 isolated update

状态：核心诊断与一次 legacy isolated update 已完成；QA-parallel、Block 及默认 kernel 新噪声门禁均 HOLD。生产 serial 默认路径和旧门槛不变。

本轮产物：[opd_noise_20261018_141452](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/)。源码修改前快照为该目录的 `source_before.tar.gz`。本报告继承上一轮[效率门禁记录](2026-10-18-opd-efficiency-gates.md)，不改写其失败结果。

## 预注册

[PROTOCOL.json](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/PROTOCOL.json) 先于本轮 GPU 实验写入。

固定 step120 完整起点、原 update119 的 slot0–3 和上一轮保存的完整 trace。所有重复恢复同一 Head、RNG，使用原全局 denominator、逐 anchor backward。Target fp32、Head bf16、loss 配方均不改。

每模式 10 次重复：default、deterministic、math SDPA、deterministic + math SDPA、每 anchor fp32 累积诊断。最后一项仅将每次 backward 产生的 bf16 参数梯度转换并累加到 fp32 buffer，不声称整个 backward 是 fp32。

每张量记录相对 repeat0 的 max-diff 分布；逐元素 min/max 的最大差是 10 次间任意两次的最大 L∞ 差。独立噪声阈值为 `2 × max_noise + 1e-6`。整体 cosine≥0.9999、relative L2≤0.01、范数相对差≤0.002 不变。新门禁与旧门禁分别报告，候选执行前冻结校准结果，并先检查一次未参与校准的 legacy 重复。

## 默认 kernel 的初步结果

10 次 loss 均为 `3.514891132399215`。相对第 1 次，其余 9 次均不逐位一致，旧逐张量门禁均有失败；整体 cosine 范围 0.99996633–0.99996889，relative L2 为 0.00788758–0.00820571。

失败张量集合不固定：`layers.0.input_layernorm.weight` 6/9，`layers.1.input_layernorm.weight` 3/9，`layers.1.self_attn.k_norm.weight` 2/9，`hidden_norm.weight` 和 `layers.0.self_attn.k_norm.weight` 各 1/9。这里的计数仅相对本组 repeat0，不是所有 45 个配对的失败次数。

这证明固定输入和 loss 下存在反向噪声，不能据此宣称候选等价。逐层 tensor hooks 和同一前向图上的单参数 `autograd.grad` 另外执行；带 hooks 的结果不冒充未插桩基线。

## 执行边界

本轮用户明确授权一次 step120 isolated optimizer step，输出新目录中的诊断状态，不覆盖正式 checkpoint，也不作为下一轮训练的恢复点。Head broadcast 若无法在指定拓扑实测，报告缺测，不记作零开销。GPU2 有其他任务，禁止终止其进程。

多卡先两卡后四卡，但必须先通过兼容门禁；Block 本轮仅新 Warm/step120 回归，不冒充历史原 Head 复现，不放行。

## kernel 与累加诊断（已完成）

| 配置 | 完整 backward 次数 | 与 repeat0 逐位一致 | 旧逐张量门禁失败的重复次数 |
|---|---:|---:|---:|
| 默认 SDPA + bf16 累积 | 10 | 1/10（仅自身） | 9/9 |
| deterministic + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 10 | 10/10 | 0/9 |
| math SDPA | 0 | 不适用 | 前向 proposal replay 分叉，立即停止 |
| deterministic + math SDPA | 0 | 不适用 | 同样前向分叉，立即停止 |
| 默认 SDPA + 每 anchor fp32 累积（诊断） | 10 | 1/10（仅自身） | 2/9 |

fp32 accumulator 自重复的最大整体 relative L2 约 0.002826，仍有 `layers.0.input_layernorm.weight` 两次失败。它减少了本组观察到的累积误差，但没有消除 kernel 噪声，不作为生产精度修改。

[单 anchor 证据](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/localize/RESULT.json)：slot0 的原 anchor plan 最后一项 `(552, 0)`，保留原 rejected loss 和全局分母。10 次 hooks 中所有前向观察值一致；最后一层 SDPA 的上游输出梯度一致，dK/dV 一致，dQ 的 max diff 为 `3.725290298461914e-9`，随后差异向前层传播。该探针最先定位到 FlashAttention backward 的 dQ；这不是对所有 anchor 的全量逐算子证明。hooks 包含 CPU 拷贝，同未插桩基线分开解释。

[实测 operator/kernel](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/localize/ATTENTION_OPS.json) 包含 `aten::_scaled_dot_product_flash_attention_backward` 和 `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`。同一个前向图上重复单参数 `autograd.grad`，layer0 input RMSNorm、layer0 k_proj、layer1 k_norm 的最大差分别为 `2.8610e-6`、`1.9073e-6`、`3.0994e-6`，说明噪声不只来自跨 anchor `.grad` 累加。

[math 前向分叉](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/localize/MATH_FORWARD_DIVERGENCE.json)：按 rounds 原顺序扫描，首个分叉位于 slot0、anchor633、offset10。默认 top1/2 为 279/47803，logits 17.455585/17.432899；math 为 47803/279，logits 17.443838/17.443558。同一 block 的最大 logit 差 0.118932。未关闭 replay 检查，未用 math 结果替换原 trace 标签。

## 确定性兼容门禁（已完成）

新门槛由各自 baseline 校准，未使用候选数值调整。独立 held-out legacy 在 deterministic 配置下仍为 58/58 逐位一致。

`qa_fp32_reference` 与 deterministic legacy：cosine `0.9998853243`（要求≥0.9999），relative L2 `0.0151460578`（要求≤0.01），裁剪前范数相对差 `0.0001397725`。旧逐张量门槛 25/58 失败；确定性校准的噪声为零，独立绝对门槛为 1e-6，58/58 未通过。loss 完全相同，selected logp 最大差为 0。

这是**逐 QA fp32 聚合与旧 bf16 累积不兼容**，不是通信实现失败，也不能归为默认 kernel 的重复噪声。整体 cosine/L2 已独立失败，不依赖对逐张量噪声门槛的解释。因此不升级到两卡/四卡候选训练或真实候选更新。随后仍可完成独立 no-update serial/Block 诊断，以及用户另外明确授权的一次 legacy isolated step。

[分布式条件协议](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/MULTIGPU_PROTOCOL_NOT_EXECUTED.json) 已记录两卡优先、每 QA bf16 backward、QA 边界清梯度、CPU fp32 固定 slot 汇总、64 MiB 分桶点对点传输、全局 denominator 精确汇总、rank0 master/Adam、Head 广播和 per-rank RNG。沿用已批准的首版方案，不引入拓扑相关的浮点 all-reduce；本轮没有实现或执行分布式训练。

## Serial top1 与 Block：确定性对拍

合并 teacher top1：四 QA live rollout 的 ids、hidden、fused、round metadata、完整 teacher logp 与旧 trace 逐位一致；selected logp、loss 和 58 张梯度也逐位一致，旧门禁与独立噪声门禁均通过。这里未做无 profiler 的重复性能比较，不能据此报告速度收益，也未改生产默认开关。

Block（新 step120 回归，非历史原 Head）：proposal、teacher top1、accepted/committed、输出、boundaries、plan 等离散内容一致；teacher/selected logp 最大差 `0.00022315979`，loss `3.51489098405`，均在已设数值容差内。但梯度 cosine `0.9999471238`、relative L2 `0.0102859930`、范数相对差 `0.0001752168`，L2 超过 0.01，继续 HOLD。

本次旧逐张量门槛失败 4 张：layer0 input RMSNorm、layer0 q_proj、layer1 post-attention RMSNorm、layer1 k_norm。历史所指 layer2 input/post-attention RMSNorm 在新 Head 上的 max diff 为 `3.05176e-5` / `0.000366211`，本次通过旧 2% 相对门槛，但均超过 deterministic 校准的独立 1e-6 噪声门槛。不能把新 Head 上的结果当作历史原 Head 的修复或复现。

## 默认 kernel 的独立噪声门禁未通过

候选前的 held-out legacy 重复：cosine `0.9999675255`、relative L2 `0.0080590971`、范数相对差 `2.21438e-7`，整体门槛通过；loss 仍为 `3.514891132399215`。但 `layers.3.self_attn.k_proj.weight` 的 max diff 为 `4.57763671875e-5`，超过冻结阈值 `3.1517578125e-5`。因此 default 的噪声协议尚未通过独立基线验证，按预注册规则在候选之前停止。没有扩大 safety、修改 floor 或重新筛选校准样本。不能将本轮 deterministic 候选结果写成“默认 kernel 也已通过新门禁”。

**selected logp 统计口径：**candidate runner 的 held-out legacy 用于建立本组 selected-logp 参考，JSON 中该行的 0 是参考与自身的比较，不是跨 calibration 的独立检查。之后 deterministic 的 QA-fp32、top1、Block 行才与这一 held-out 参考逐元素比较。校准与 held-out 之间独立比较的是梯度和 loss；不将 selected-logp 自比较计入验收证据。

默认 kernel 若要继续噪声门禁研究，需要新的、事先冻结的基线采样及独立验证协议；本轮不在看过失败后追调阈值。生产默认不改，Block 和 QA-parallel 均 HOLD。

## 一次真实 isolated update

使用完整 step120 起点及原 update119 的四个 slot；先做完整 slot0 rollout warmup，再恢复完整状态，执行一次原 legacy update。CPU 8 threads，与历史 launch 脚本的 `OMP_NUM_THREADS=8` 一致。没有打开 serial 候选开关，没有关闭原逐参数 update-norm 统计。

| 分段 | 实测秒 |
|---|---:|
| rollout + trace offload | 182.8372 |
| scoring/backward | 10.2448 |
| bf16 grad → CPU fp32 master sync | 1.6911 |
| 裁剪前范数 | 0.3565 |
| clip | 0.4331 |
| CPU Adam | 2.0021 |
| write-back + 原逐参数 update-norm 统计 | 0.9163 |
| 完整 update（含少量验证/记录开销） | **198.5035** |

optimizer 相关分段合计 5.3990 秒；本次证据不支持“optimizer 约39秒”。write-back 与原 update-norm 统计是合并残差，不能把 0.9163 秒全部归因于未使用的 norm 统计。每 slot rollout 分别 49.9482 / 52.1436 / 28.1902 / 52.5549 秒。

[原始计时](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/isolated_step/TIMING.json)。这是**一次样本**，warmup 为一个完整 QA；未按最初 clean benchmark 的完整 update warmup + 三次重复执行，不据此宣布性能达标。加载、恢复、warmup、保存时间不在上述 update 中。

[隔离状态审计](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/isolated_step/STATE_AUDIT.json)：58 个 Adam 状态的 step 均为121，实际更新次数1；Head bf16 与 CPU fp32 master 的 bf16 写回全部逐位相同。完整 trace 与原基线一致，loss `3.514891132399215`，裁剪前范数 `0.3597863913`，全局 denominator `0x1.8c9132f8ef40bp+11`。原 Target、step120、数据和日程 SHA 不变。

产物 `isolated_step/isolated_state.pt` 含 Head、master/Adam、RNG 和父起点身份，明确 `diagnostic_only=True`；它重复原 update119，**不是正式训练第121步的可续训 checkpoint**。

## 多卡上限估算与限制

[条件估算](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/SPEEDUP_UPPER_ESTIMATES.json)：假设本次单卡每 QA 成本可原样搬到各 rank、scoring/backward 完美均摊，且暂不计梯度聚合、通信、Head broadcast 和其他分布式开销，按指定 slot 映射得到两卡约 **1.72×**、四卡约 **3.28×** 的乐观上限估算。若连 QA 工作量都理想均衡，Amdahl 上限约1.95×/3.70×。

这些不是实测 speedup，更不是可部署承诺。QA-fp32 与 legacy 的数值兼容门禁已失败；本轮没有两卡/四卡 QA-parallel update，也没有连续更新、恢复或 dev8/confirm200 放行评测。

## 交付与复现

新增诊断入口：

- `repro/diagnostics/opd_noise_diagnostic.py`：`--mode default|deterministic|math|deterministic_math|fp32_accumulator`，默认10次，无 optimizer。
- `repro/diagnostics/opd_backward_localize.py`：逐层 tensor hooks、同图单参数 autograd.grad、单 anchor profiler 和 math 前向分叉。
- `repro/diagnostics/opd_noise_candidates.py`：`--calibration` 指定冻结 baseline；先 held-out，再候选；`--deterministic` 需匹配环境。
- `repro/diagnostics/opd_isolated_step.py`：单次 legacy 更新，必须显式 `--authorized-single-step`；本轮授权的一次已执行，不能自动再次执行。
- `repro/diagnostics/opd_head_broadcast.py`：仅 Head 两卡通信探针，无 Target/梯度/optimizer，不是分布式训练。

基线输入固定为 `/mnt/local3/mcy/GLANCE/results/opd_efficiency_20261018_124902/runs/baseline_v2`。Python 为 `/home/mcy/.conda/envs/fin_research/bin/python`。单卡绑定 UUID `GPU-e1a07214-4092-280d-4e13-365bd4f4a439`，运行前重查占用；所有 `--out` 必须是 `/mnt/local{,2,3}` 下全新目录，存在即报错。deterministic 模式在进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，默认模式清除该变量。实际命令和环境保存在产物目录的 `*_command.json`；最初 default/deterministic 命令及完整复现说明见 `REPRODUCE.md`。

31 项 CPU 契约/边界/噪声门禁测试通过，见 `CPU_TESTS.log`。生产 runtime、trainer、master optimizer、AGENTS.md 均与本轮前快照一致。单卡 fallback 就是原 serial 默认路径；本轮未切换生产开关。

源码版本偏差已如实记录于 [SOURCE_PROVENANCE.md](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/SOURCE_PROVENANCE.md)：default 进程启动后曾短暂编辑未使用的 fp32 诊断分支并还原，不符合“不覆盖运行脚本”的规则；进程未重载函数。实际执行版本单独归档，不用最终源码 SHA 冒充早期执行源码 SHA。之后的源文件版本切换等待进程退出。

建议保留 deterministic 原 SDPA 作为独立验证环境，生产仍用默认 kernel。合并 top1 已取得确定性正确性证据，但默认噪声门禁及重复性能验收尚未完成，继续作为实验开关。不要靠放宽全局阈值推进 QA-parallel，也不放行 Block。

### Head broadcast 缺测项已补齐

单卡 isolated step 时未测广播；随后 GPU2 释放，重新检查占用后执行独立的 GPU1→GPU2 Head-only NCCL 探针。没有 Target、梯度或 optimizer，未进行 QA-parallel。2,097,252,864 bytes bf16 Head，固定参数名顺序，75 个不超过64 MiB的分块；warmup一次，正式三次，两 rank 每次耗时取最大值：`0.10680054 / 0.10689609 / 0.10680138` 秒，两端与隔离 Head 逐位一致。

[原始结果](/mnt/local3/mcy/GLANCE/results/opd_noise_20261018_141452/head_broadcast/RESULT.json)。它是独立通信成本，不含两 rank 上完整 Target/训练负载，不将其冒充完整多卡 update。原单卡 TIMING.json 的 broadcast=null 按原样保留，由此独立产物补充；四卡广播仍未测。
