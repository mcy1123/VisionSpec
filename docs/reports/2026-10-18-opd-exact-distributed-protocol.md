# OPD 确定性分布式累积协议实测

本轮产物：[/mnt/local3/mcy/GLANCE/results/opd_exact_protocol_20261018_152931/](/mnt/local3/mcy/GLANCE/results/opd_exact_protocol_20261018_152931/)。源码修改前已建立 `source_before.tar.gz`。两卡/四卡 live 检查与 Block 组件诊断已完成。用户授权的6次 isolated full update已按 deterministic 单/2/4卡、default 单/2/4卡顺序全部完成；serial top1 无更新配对性能测试也已完成，未达到预注册收益门槛，默认保持关闭。正式训练未启动。

## 验证协议

本轮 [PROTOCOL.json](/mnt/local3/mcy/GLANCE/results/opd_exact_protocol_20261018_152931/PROTOCOL.json) 先于新候选执行冻结。

- deterministic + `CUBLAS_WORKSPACE_CONFIG=:4096:8` 做梯度、语义验证，保持默认 SDPA，不强制 math。
- default kernel 做性能测量；本轮不再追逐 default per-tensor 噪声门禁。若将来需要，必须另行冻结50–100次校准及上分位数规则，不沿用10次 max。
- 原 step120 完整 checkpoint、原 update119 的 slot0–3；Target fp32、Head bf16、loss、teacher logp、clamp、decay0.9、流权重1:2、anchor顺序均保持 canonical 实现。
- 每 slot 保留原 denominator 运算次序，由 rank0 检查 slot覆盖与任务身份，再广播同一 binary64 `.hex()`；无 world-size 缩放。
- 2 rank先于4 rank。固定 GPU UUID，对应物理1/2及1/2/6/7；不终止其他任务。

## gradient-only 结果

先独立重新计算单卡 legacy：58张梯度与上一轮 deterministic reference 逐位一致。另保存四份逐 QA 梯度，以及单卡 qa_fp32 reference。

两种算法明确命名：

1. `qa_fp32_exact_transport`：原 MULTIGPU_PROTOCOL，QA内bf16 backward，QA结束转CPU fp32，按slot0–3固定顺序相加；Gloo点对点，每块≤64MiB，presence mask保留None。
2. `anchor_ordered_bf16`：本轮预注册的新诊断方案。每anchor清空叶子.grad并反向；rank0按原slot/anchor顺序，在GPU bf16累加；跨rank用NCCL点对点，每块≤64MiB。未切换目标函数或归一化，未修改容差。

| 实测项 | 2 rank | 4 rank |
|---|---|---|
| 各QA梯度 vs 单卡同QA | 全部逐位一致 | 全部逐位一致 |
| qa_fp32固定slot汇总 vs 单卡qa_fp32 reference | 58/58逐位一致 | 58/58逐位一致 |
| qa_fp32汇总 vs legacy | rel L2=0.0151461，不兼容 | 同样不兼容 |
| anchor_ordered_bf16 vs legacy | **58/58逐位一致** | **58/58逐位一致** |
| anchor_ordered计算与传输阶段 | 23.7697秒 | 29.8569秒 |

因此，之前 qa_fp32 的失败不能用来否决所有分布式累积协议；本轮已实测找到保留 legacy bf16运算次序的方案。但它需要传输每个anchor梯度，不能套用仅传四份QA梯度的成本预期。

以上是固定trace诊断：Target主干forward为0，无optimizer，无live rollout，计时不是完整update。梯度报告分别位于 `gradient2/`、`gradient4/`；`anchor_vs_legacy.json` 的 cosine=1、relative L2=0、norm差=0，`bitwise=true`。

## 四卡 Head broadcast

使用原step120完整状态中的Head，未将前轮isolated state当作正式step121。四端均与源Head逐位一致。2,097,252,864 bytes，75个≤64MiB分块，warmup一次，三次最大rank耗时为0.122799/0.121894/0.121877秒。仅通信探针，无Target/梯度/optimizer，不能当作完整多卡吞吐量。

## 实验入口与边界

- `repro/diagnostics/opd_exact_gradient.py`：独立单卡参考、两卡/四卡的两种梯度协议对拍；不构造optimizer。
- `repro/opd_official_kl/ordered_parallel.py`：独立的逐anchor bf16累积与Head广播组件，生产trainer未接入。
- `repro/diagnostics/opd_parallel_isolated.py`：单次live隔离实验入口，默认无update；真实step必须显式传入授权开关。输出固定为diagnostic_only，parent step120/reused schedule119，不提供正式续训入口。
- `repro/diagnostics/opd_block_components.py`：Block teacher/context/LM-head调用形状组件诊断，HOLD不变。
- `repro/diagnostics/opd_top1_performance.py`：default kernel，完整4QA warmup，3组交替顺序的baseline/top1配对，包含rollout和scoring/backward，不含optimizer；不做default梯度门禁。

新增真实optimizer更新仍按AGENTS的单独明确授权规则处理；前轮的一次授权已使用。本轮用户回复“授权6次隔离 full update”，范围见 `FULL_UPDATE_AUTHORIZATION.json` 和 `FULL_UPDATE_PROTOCOL.json`；每次重新加载原step120，输出diagnostic_only，不是正式step121。本轮没有把无更新计时写成full-update speedup。

## Live rollout 与组件诊断

两卡与四卡的live无更新检查均通过：四QA完整trace、teacher logp、hidden/fused与原参考逐位一致，最终58张梯度与deterministic legacy逐位一致。各rank Head SHA一致，全局denominator均为 `0x1.8c9132f8ef40bp+11`。live验证计时含trace比较、梯度参考加载与CPU比较，不是full-update性能结果。

### Block 因果干预

四QA committed hidden/fused均逐位一致，差异集中于rejected teacher logp。response student、response teacher、rejected student selected-logp均与serial一致。

| 干预 | relative L2 vs serial | 旧逐张量失败数 | 超过严格1e-6数 |
|---|---:|---:|---:|
| serial | 0 | 0 | 0 |
| 原Block | 0.010285993 | 4 | 58 |
| serial context + Block teacher logp | 0.010285993 | 4 | 58 |
| Block context + serial teacher logp | **0，58/58逐位一致** | 0 | 0 |
| Block context + 单行LM-head投影teacher | 0.009897806 | 2 | 57 |

Block rejected teacher logp最大差 `0.00022315979`；仅将LM-head投影改为逐行诊断调用，仍有 `0.00020599365`。这说明LM-head调用形状只解释部分差异，Block主干的隐藏状态数值差异仍存在。逐行投影诊断虽然使整体L2略低于0.01，仍未通过1e-6及旧逐张量门槛，不放行、不改生产代码或阈值。

原始 `block_traces.pt`、`row_projection_logp.pt`、五组梯度及逐参数报告均已保存于 `block_components/`。这些hybrid仅用于因果定位；未将正确commit轨迹的概率冒充rejected draft-induced prefix的teacher logp。

## 已完成的 deterministic full update

三次都从完整原step120出发，在新目录独立执行一次step；不是连续三步。两卡、四卡均通过原始字节比较（包括有符号零）：Head、fp32 master、完整Adam状态，以及每rank归一化后的Python/NumPy/CPU/CUDA RNG均与单卡一致。Adam step从120到121只是隔离状态内的诊断计数，未产生正式step121或日程推进。

| deterministic诊断阶段（秒） | 单卡 | 两卡 | 四卡 |
|---|---:|---:|---:|
| rollout/trace offload | 207.798 | 122.501 | 65.570 |
| scoring/backward及分布式anchor传输 | 9.662 | 23.734 | 34.156 |
| 梯度检查/主机传输 | 单列见原始TIMING | 5.161 | 5.392 |
| optimizer | step本体3.475 | 5.000 | 4.817 |
| Head broadcast | 不适用 | 0.107 | 0.215 |
| 完整计时窗口 | 222.313 | 156.502 | 110.150 |

这是正确性诊断时间，不用来替代default性能结果。单卡与多卡阶段划分不同，单卡total还含master sync、裁剪前范数、trace核验等；不把表内非同口径阶段简单相除。单卡deterministic原始TIMING沿用了旧broadcast备注，已保留原件并新增METADATA_CLARIFICATION.json纠正，见产物目录。其实际执行源码已按执行前SHA重建保存；该次未采集峰值显存和forward计数，不补造数据。

## 无更新复现与回退

先确认指定GPU空闲、UUID仍对应目标设备；全部新输出目录的realpath必须位于允许挂载点。以下不执行optimizer：

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PYTHONPATH="$PWD/repro"
PY=/home/mcy/.conda/envs/fin_research/bin/python
BASELINE=/mnt/local3/mcy/GLANCE/results/opd_efficiency_20261018_124902/runs/baseline_v2
REFERENCE=/mnt/local3/mcy/GLANCE/results/opd_exact_protocol_20261018_152931/reference
OUT=/mnt/local3/mcy/GLANCE/results/NEW_UNIQUE_DIAGNOSTIC_DIRECTORY
# 两卡；四卡在确认空闲后追加GPU6、7的UUID，并改nproc为4
export CUDA_VISIBLE_DEVICES=GPU-e1a07214-4092-280d-4e13-365bd4f4a439,GPU-be4999dd-0619-aeb4-c83d-97d249fd5476
CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  repro/diagnostics/opd_parallel_isolated.py --baseline "$BASELINE" \
  --reference "$REFERENCE" --out "$OUT" --kernel-mode deterministic
```

本轮真实step的精确命令与环境逐次保存在full-update根目录的`*_command.json`，不能据此自动再运行一次：6次授权只覆盖本轮队列。隔离入口不提供正式resume，不支持把这些状态接入训练；连续4步、2+恢复+2、dev8/confirm200未对候选执行，因此候选仍不能用于正式OPD。单卡fallback为原canonical trainer + serial_reference，默认关闭batch_teacher_ids；Block继续HOLD。

## 六次隔离更新：default完整耗时初测

6/6已完成，每次都重载原完整step120；本次有界授权已全部用完。三个default样本均通过完整trace/teacher logp及分母精确检查；只要求状态有限、master写回及Head广播一致，不把默认kernel的逐张量梯度噪声作为门禁。

| default实测 | 单卡 | 两卡 | 四卡 |
|---|---:|---:|---:|
| 完整计时（秒） | 201.409 | 141.817 | 95.463 |
| 相对本次单卡初测 | 1.000× | 1.420× | 2.110× |
| GPU-seconds/update（卡数×窗口） | 201.409 | 283.635 | 381.853 |
| rollout/trace/check及等待（秒） | 184.333 | 110.955 | 57.143 |
| scoring/backward/anchor通信（秒） | 10.986 | 25.817 | 31.268 |
| 梯度validation/transfer（秒） | 见下文分项 | 0.599 | 1.797 |
| optimizer（秒） | step本体4.091 | 4.338 | 5.087 |
| Head广播（秒） | 不适用 | 0.108 | 0.168 |

单卡另有master sync1.644秒、裁剪前范数0.322秒；step本体含clip0.444、Adam2.624、write-back及旧update norm统计1.023秒。单卡total还包含trace检查等间隔，不等于只加表内阶段；多卡optimizer阶段含外部master赋值、范数、step及barrier，不能按同口径逐项相除。

三种拓扑实际forward计数均为Target主干3304、LM-head3926、Draft Head416。峰值显存与各rank明细见`FULL_UPDATE_SUMMARY.json`和原始RESULT/TIMING。计数由forward hook实测，不是日志推算。四卡广播探针约0.122秒，实际四卡更新内0.168秒，二者分别报告。

计时窗口从状态恢复后开始，到optimizer及必要Head广播完成为止；不含模型加载、完整状态恢复或诊断checkpoint保存，本轮没有单独量化这些I/O时间。以上只有每拓扑**一次**完整更新，且单卡warmup一个QA、多卡无额外warmup，另有共享主机负载，**未满足三次配对、全快且中位>1.3×的正式性能门禁**。不承诺2卡1.7–1.9×或4卡3.0–3.3×。GPU-seconds增加，墙钟加速与资源效率须分开判断。逐anchor精确梯度通信的理论载荷，按完整Head字节数×远端anchor数推算约两卡253.8GB、四卡322.98GB/update；这是计算估计，不是链路实测流量。

## 测试与修改范围

CPU契约及回归测试33项全部通过（`cpu_tests_final.log`）：真实两进程Gloo传输、None/零梯度、固定slot和anchor累积、denominator/teacher契约、官方KL及Block现有边界测试。命令：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=repro /home/mcy/.conda/envs/fin_research/bin/python -m unittest \
 tests.test_opd_exact_transport tests.test_opd_efficiency tests.test_opd_noise_gate \
 tests.test_official_kl tests.test_official_cleanup tests.test_block_verification -q
```

新增源码：`opd_exact_gradient.py`、`opd_parallel_isolated.py`、`opd_block_components.py`、`opd_top1_performance.py`（均在`repro/diagnostics/`），`repro/opd_official_kl/ordered_parallel.py`及`repro/tests/test_opd_exact_transport.py`。修改现有诊断入口`opd_head_broadcast.py`、`opd_isolated_step.py`，以及AGENTS和报告索引。canonical runtime/trainer未改；保留旧生产默认路径。

本轮每个阶段先停止对应脚本，再修改下一阶段源码；没有覆盖运行脚本。已执行的旧版本通过执行前SHA绑定保存为`executed_gradient_source.py`、`executed_live_source.py`、`executed_det_serial_source.py`；后续版本逐次SHA见命令JSON，不能用最终源码冒充此前执行版本。

## 建议与未验收范围

- 保留精确anchor聚合实验入口：它解决的是legacy累积次序一致性，现阶段不是已验收的正式多卡trainer。
- qa_fp32与legacy不兼容、anchor方案逐字节通过是两个独立结论，不修改旧门槛或将前者误报通信错误。
- 优先考虑下一轮单独预注册“仅QA rollout并行，传回完整trace，rank0原顺序serial scoring/backward”的方案：现有4QA trace约182MB，而逐anchor梯度理论载荷为数百GB。它仍需重新对拍及完整计时；本轮**未实现、未测量，也未授权额外真实step**。
- 连续更新、同拓扑恢复与冻结候选的dev8/confirm200尚未执行，本轮没有新增确认集结论。此前正式step120的确认集报告保持原样，不借用为本候选验收。
- Block保持HOLD；不进入本轮多卡组合，不改变teacher前缀或目标函数迎合梯度门槛。

## 快照与回滚

修改前的可恢复快照为本run的`source_before.tar.gz`；交付同时保留执行版本、最终源码包、`SOURCE_CHANGES.patch`及逐文件SHA清单。回滚须先确认没有相应运行进程，再从快照只恢复本轮修改的既有文件（AGENTS、报告索引和两个旧诊断入口）；新增实验模块没有接入生产默认路径，可以保留作审计。不要恢复或删除历史权重、原始结果，不要将diagnostic state写回正式checkpoint目录。

### 分母与显存审计

固定slot0–3分别为train:2131、5609、4321、1154；完整response计数、按round/offset原顺序的rejected列表及每slot分母十六进制值保存在各多卡RESULT的records中。全局分母精确为`0x1.8c9132f8ef40bp+11`，三个default完整update loss均为3.514891132399215。三次deterministic的梯度及更新差异为零；default不做逐张量梯度等价判定。

| default完整update峰值 | 单卡 | 两卡最大rank | 四卡最大rank |
|---|---:|---:|---:|
| allocated（GiB） | 37.702 | 40.631 | 40.631 |
| reserved（GiB） | 38.953 | 42.809 | 42.795 |

这是PyTorch峰值，不等于nvidia-smi的总设备占用；其他rank的独立数值保存在原始RESULT及FULL_UPDATE_SUMMARY中。

## Serial top1 default性能对照：不放行默认开关

完整4QA warmup后，三组交替顺序配对结果：

| 配对 | baseline（秒） | 合并top1（秒） | speedup |
|---|---:|---:|---:|
| 1，baseline先 | 196.1407 | 190.5310 | 1.029443× |
| 2，候选先 | 194.5669 | 194.2842 | 1.001455× |
| 3，baseline先 | 195.6236 | 195.8044 | 0.999077× |

三组中位为**1.001455×**，且有一组更慢，未通过事先冻结的中位>1.05且全部更快门槛。所有样本的完整trace、teacher logp及denominator精确一致，loss均为3.514891132399215。此处是包含rollout与scoring/backward的无更新测量，不是完整update speedup，也没有用default梯度噪声作候选否决依据。`batch_teacher_ids=False`的生产默认保持不变。

前后完整checkpoint、数据、日程及Target四权重分片哈希核验通过；所有实验进程已退出，所选GPU已释放。未删除或终止其他GPU任务。最终状态为**诊断完成、候选未正式放行**：精确多卡协议正确性通过，重复性能/连续更新/恢复仍未验收，Block HOLD，top1默认关闭，6次隔离更新授权已用完。

交付入口为本run的`RUN_STATUS.json`、`FULL_UPDATE_SUMMARY.json`、`MODIFIED_FILES.json`、`SOURCE_CHANGES.patch`、`source_before.tar.gz`、`source_after.tar.gz`、`FINAL_SOURCE_PROVENANCE.json`和`DELIVERY_MANIFEST.json`。大状态独立保存在 `/mnt/local2/mcy/GLANCE/results/opd_exact_full_20261018_152931/`；逐次命令、计时、状态审计与6份diagnostic state原样保留。本轮报告已登记到现有报告索引，未覆盖历史报告或权重。
