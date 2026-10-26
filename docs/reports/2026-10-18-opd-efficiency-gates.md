# OPD 效率优化：基线重复性 HOLD

最终状态：**BASELINE_REPEATABILITY_HOLD**。已完成快照、实验契约、CPU测试、真实无更新计时、完整4 QA分段profiling及部分serial开关诊断。旧serial自身的相同trace重复backward也会违反原逐张量梯度门槛；不能将候选的超限直接认定为优化实现错误。原阈值未放宽，实验开关未默认启用，多卡与真实更新停止。

产物根：[本次运行](/mnt/local3/mcy/GLANCE/results/opd_efficiency_20261018_124902)。全过程真实模型optimizer steps=0；GPU诊断进程已结束。历史checkpoint和原始报告未覆盖。

## 已实现与边界

- `repro/opd_official_kl/efficiency.py`：trace身份与完整15个fp32 teacher logp校验；按原Python浮点顺序逐slot汇总分母并比较hex；逐QA CPU fp32梯度固定slot求和，保留None；原逐参数梯度门槛。
- `runtime.py` / `adapters.py`：默认关闭的`batch_teacher_ids`、`preallocate_context`；`score(..., deferred_checks=True)`在GPU累积检查标志，必须完全耗尽generator后才能更新。response teacher投影和rejected teacher逐行log-softmax未修改。
- `MasterWeightAdamW.set_external_master_grads()`：外部梯度就绪时禁止旧sync覆盖，保留全局None。`step(collect_update_norms=False)`可关闭更新范数同步；旧调用者默认行为保留。
- `opd_efficiency_gate.py`：固定checkpoint/RNG的无更新计时与QA聚合门禁。
- `opd_serial_candidate.py`：默认先做多次旧serial固定trace重复性检查，再逐项检查独立开关；GPU trace驻留有2 GiB默认预算和4 GiB余量。失败即停止，不组合未通过的开关。
- `opd_profile_sharded.py`与流式汇总：全部4 QA分段采集，支持接续已完成的slot0 raw trace，不构造庞大的PyTorch事件树。

**未完成/未放行：**两卡/四卡通信和分布式恢复入口依计划在兼容门禁通过后才实现，本轮未实现；未执行真实optimizer计时、连续更新、4 vs 2+resume+2、dev8/confirm200新评测。deferred checks、preallocation、resident和组合的真实GPU验收在前置失败后未执行，仅有CPU覆盖；不声称这些已满足正式门禁。Block维持原HOLD。

## 冻结输入与契约

完整OPD step120，原日程update=119的slot0–3：train:2131、5609、4321、1154；四个pool prompt与schedule prompt SHA另经核对一致。Target fp32、Head bf16、原Bernoulli+k3、rejected decay=.9、两流1:2。未引入SFT的.8或variable reveal。

GPU1 UUID为`GPU-e1a07214-4092-280d-4e13-365bd4f4a439`。本次环境torch 2.9.1+cu128，allow_tf32=False，deterministic_algorithms=False。真实GPU计算只使用该卡；其他GPU任务未被终止。

分母按原round/offset顺序计算每QA，再按slot0..3执行Python sum，结果精确为`0x1.8c9132f8ef40bp+11`。完整trace离线复核：206轮、210 anchors、3090个fp32 proposal teacher logp，均有限。CPU/GPU往返保持logp逐位相同。

梯度门槛：cosine≥.9999、relative L2≤.01；每张张量max_abs_diff≤1e-6+.02×reference_max_abs；范数相对差≤.002。loss atol=1e-4、rtol=1e-3。没有根据结果调整阈值。

## 数值结果

| 对照 | relative L2 | cosine | 超限张量 | 判定 |
|---|---:|---:|---:|---|
| 旧serial固定trace重复A | .00799575 | .99996804 | 0/58 | PASS，非逐位相同 |
| 旧serial固定trace重复B | .00806454 | .99996748 | 3/58 | HOLD |
| 逐QA fp32聚合 vs 旧serial | .01561581 | .99987811 | 22/58 | 观察到兼容门禁失败 |
| 每轮合并teacher top1传回 | .00807077 | .99996743 | 3/58 | HOLD |

以上loss均为3.514891132399215。合并top1开关的完整live trace，包括token、hidden/fused、teacher logp、plan、标签、分母，均与原trace逐位一致。其梯度差异与基线重复噪声相近，不能归因于该开关本身。旧serial重复B超限的是layer0 input_layernorm、layer0 k_proj、layer1 k_norm；这使“同实现重复性”这一前置条件失败。未继续重复C或后续候选，未用全局cosine覆盖逐张量失败。

初始记录`gradient_compatibility.json`仍保留ACCUMULATION_COMPATIBILITY_HOLD分类；最终解释以本报告及根RUN_STATUS为准：**尚未建立稳定的基线逐张量门禁，也不能据此声称已证明唯一的误差来源。** 后续应先定位/量化原实现backward非确定性；任何新的容差协议须单独决定，不能在本轮静默放宽。

## 计时与profiling

一次warmup后，三次独立恢复状态的无profiler计时：205.883、193.541、195.251秒，中位数195.251秒。峰值allocated=39,605,821,952 bytes，reserved=40,644,902,912 bytes。

这些是rollout+scoring+backward的**无更新**时间，不包括CPU master同步、Adam、clip和write-back；不能冒充完整update或与历史234秒直接比较。候选诊断含正确性检查，不用于speedup结论。

实际主干forward 3304次（4 prefill、3300 verification/final-root），Draft proposal 206次、replay/backward各210次，冻结共享lm_head共3926次。实测7,276,965个kernel事件、13,784次cudaStreamSynchronize；16次cudaDeviceSynchronize包含8个profiler片段的边界同步。

| 阶段 | GPU kernel秒 |
|---|---:|
| Target verification / final root | 152.469 |
| 视觉编码 | .312 |
| 文本prefill，不含视觉 | 1.230 |
| Draft Head proposal，不含共享投影 | 1.043 |
| Draft Head replay，不含共享投影 | 1.037 |
| 共享无梯度lm_head投影 | 13.137 |
| 有梯度lm_head投影 | .790 |
| backward，含autograd工作线程 | 4.534 |

Kernel总计175.108秒，约87%属于Target验证。CPU预处理区间约.790秒，trace D2H/H2D区间分别.102/.628秒；CPU区间嵌套且受profiler影响，不直接相加或冒充干净阶段时间。raw scope `lm_head_teacher`同时包括rollout Draft投影，不是纯Teacher scoring。

autograd跨线程kernel按launch时间落在主线程同步backward区间归属。初始汇总保留；修正版[audit/PROFILE_SUMMARY_CORRECTED.json](/mnt/local3/mcy/GLANCE/results/opd_efficiency_20261018_124902/audit/PROFILE_SUMMARY_CORRECTED.json)修正归属，没有改变原始kernel时间。约.289秒kernel仍未归属。

## 运行过程与完整性

第一次诊断启动因计时路径包含logp往返校验而主动停止，保留原脚本，改用baseline_v2并将校验移到计时外。全量shape/memory trace在CPU整理阶段占用约89 GiB，停止后改为分段。第一段raw trace成功导出后，PyTorch key_averages构造事件树仍过慢；保留第一段，从第二题接续并改用流式解析。两次停止均仅针对本次创建的诊断进程。

实际执行版本保存在audit/executed_sources。profiler raw trace共8段，第一段在runs/profile_sharded，其余在runs/profile_resume。原始数据和失败记录全部保留。

28项CPU测试通过，覆盖官方loss/恢复、原block限制、新分母/梯度presence/trace开关/跨线程profile汇总。快照已在新目录实际恢复，48个文件SHA逐项一致。profile_resume、serial_candidates、serial_repeatability的输入收尾SHA均与最初manifest一致，包括Target四分片、Head、完整training_state、pool和schedule。

## 复现与恢复

从仓库根运行，输出必须使用新的挂载目录；以下目录名是待替换的示例。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=repro /home/mcy/.conda/envs/fin_research/bin/python -m unittest tests.test_opd_efficiency tests.test_official_kl tests.test_official_cleanup tests.test_block_verification -q

CUDA_VISIBLE_DEVICES=GPU-e1a07214-4092-280d-4e13-365bd4f4a439 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=repro /home/mcy/.conda/envs/fin_research/bin/python repro/diagnostics/opd_efficiency_gate.py --out /mnt/local3/mcy/GLANCE/results/NEW_BASELINE --repeats 3

CUDA_VISIBLE_DEVICES=GPU-e1a07214-4092-280d-4e13-365bd4f4a439 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=repro /home/mcy/.conda/envs/fin_research/bin/python repro/diagnostics/opd_profile_sharded.py --baseline /mnt/local3/mcy/GLANCE/results/NEW_BASELINE --out /mnt/local3/mcy/GLANCE/results/NEW_PROFILE

python repro/diagnostics/summarize_opd_efficiency.py /mnt/local3/mcy/GLANCE/results/NEW_BASELINE
python repro/diagnostics/summarize_opd_profile_stream.py /mnt/local3/mcy/GLANCE/results/NEW_PROFILE/rollout_slot0/profile.json

CUDA_VISIBLE_DEVICES=GPU-e1a07214-4092-280d-4e13-365bd4f4a439 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=repro /home/mcy/.conda/envs/fin_research/bin/python repro/diagnostics/opd_serial_candidate.py --baseline /mnt/local3/mcy/GLANCE/results/NEW_BASELINE --out /mnt/local3/mcy/GLANCE/results/NEW_REPEATABILITY --variants baseline_repeat baseline_repeat_b baseline_repeat_c
```

不要用gate旧的`--profile`重型路径代替分段入口。快照为`source_before.tar.gz`，可解压到全新的目录审查/恢复；不直接覆盖当前工作目录。新增文件和补丁清单见交付manifest及changes.patch。
