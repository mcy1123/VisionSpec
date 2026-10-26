# OPD：并行 rollout、rank0 canonical scoring/backward

本轮唯一候选为四卡并行生成完整on-policy轨迹，rank0原slot/anchor顺序重建student图并更新。运行目录：`/mnt/local3/mcy/GLANCE/results/opd_rollout_only_20261018_204925/`；新大状态目录：`/mnt/local/mcy/GLANCE/results/opd_rollout_only_20261018_204925/`。修改前已建立`source_before.tar.gz`，旧结果不覆盖。

## 冻结协议

`PROTOCOL.json`在GPU执行前冻结。沿用原官方方向覆盖：response Bernoulli forward KL、rejected k3、官方池化、decay=0.9、stream1:2、原clamp与anchor plan。Target fp32、Head bf16、CPU fp32 master AdamW、原prompt及生成上限。Block HOLD，top1合并关闭，无batching、anchor减少或其他目标。

### Trace与梯度图

rollout本来就在`no_grad`中执行；完整trace包含ids、committed hidden/fused、完整round/proposal/teacher top1/15个fp32 teacher logp、rejected offsets、commit、boundaries、plan和身份信息。hidden/fused是冻结Target输出，可以detach；不允许传student投影缓存、student KV或复用detached Head forward。rank0通过未经修改的canonical `runtime.score()`按slot0–3与原anchor顺序重新执行Head，bf16 `.grad`跨所有QA累积，然后master sync、clip、step各一次。

wire使用CPU Gloo分块，torch.save保留原dtype/形状，SHA核验完整载荷，接收后校验身份、schema、15个teacher logp、context/token对齐和原分母。response teacher logp仍由rank0 canonical LM-head调用按原anchor计算；不把rejected概率复用为response概率。

### Head版本barrier

初始加载后各rank校验完整参数原始字节SHA；每update开始校验共同Head hash、optimizer step、schedule cursor以及本进程参数mutation版本。trace绑定同一policy与QA/slot/prompt/config身份。全部rollout和trace汇总后才能backward；检查权重仍未改变。只有rank0更新，然后广播Head、重新计算各rank Head SHA并barrier；下一update不得提前开始。不同版本、slot身份、拓扑或恢复配置不匹配直接失败。

### 24次真实更新预算

本条用户请求“24次更新预算，审查可行即执行”作为本轮有界授权，旧6次额度不复用。`BUDGET.json`每次进入optimizer前锁定并fsync reserve，失败不退回、不自动重试；完成后记complete。

| 用途 | 次数 |
|---|---:|
| deterministic单卡、旧四卡anchor、新四卡rollout单步对照 | 3 |
| step90原日程90..93：serial连续4、新方案连续4、新方案2+恢复+2 | 12 |
| default 3组×serial/anchor/rollout完整更新 | 9 |
| 上限 | 24 |

前置三种方案无更新门禁不计入额度，也不构造optimizer。任何门禁失败即停止升级，报告实际已reserve/complete次数；不追加预算、不启动正式第二seed或长期训练。全部新状态为diagnostic_only，不能作为正式step121或自动续训。

### 3×3计时

每个case独立恢复完整原step120，固定原update119四QA。三组顺序为serial→anchor→rollout、anchor→rollout→serial、rollout→serial→anchor。各case完整4QA无更新warmup后，恢复完整Head/master/Adam/RNG再计一次真实update；default只测性能，不用默认kernel梯度噪声作门禁。deterministic原SDPA与`:4096:8`用于语义/梯度/状态验证。

窗口包括rollout、版本barrier、CPU trace卸载、序列化及跨rank传输、分母、canonical scoring/backward、master sync/norm、optimizer、Head广播及版本校验。不含加载、恢复、落盘和窗口外的diagnostic reference比较。阶段和各rank显存独立报告；forward计数由hook实测，wire bytes为实际send tensor/序列化载荷计数，不冒充物理链路流量。GPU-seconds为参与卡数×该case最大rank wall time。

每case前检查GPU UUID及占用，不终止其他任务；运行时记录GPU/主机负载与外部进程。仅使用物理1/2/6/7，单卡为物理1。选定卡有其他任务则停止/标记污染，不给干净速度结论。

性能GO预注册：相对serial和旧anchor三组配对全部更快，中位speedup对两者均>1.05，且对serial>1.3；正确性、连续轨迹及同拓扑恢复必须先通过。否则HOLD，不反向修改阈值。

## 当前状态

本轮已结束：`COMPLETE_GO_MINIMAL_OPT_IN_PROPOSAL_WITH_DISCLOSURES`，真实更新reserve/complete=24/24，上限24。CPU回归27项及初始状态加载292张tensor字节对拍通过。最终门禁、性能和GO/HOLD见下方最终结果；以下纠错记录为实验历史，原产物保留。

### 启动器纠错记录（未进入模型）

首个四卡launcher调用被torchrun参数解析器拒绝：脚本参数`--run`与torchrun自身`--run-path`缩写冲突，未进入脚本main、未加载四卡模型，BUDGET保持0/24。已用四进程CPU `--help`验证`--`分隔符修复。原`execute.py`、`gate_anchor.log`、exit=2及`QUEUE_DONE.json`原样保留；后续由新文件`execute_v2.py`继续，最终队列标记为`QUEUE_V2_DONE.json`，权威当前状态仍为RUN_STATUS。复用已通过的serial门禁，新四卡实际门禁目录为`gate_anchor_v2`；模型/目标/梯度代码及输入冻结SHA均未改。没有重试任何真实optimizer更新。

### 监控退出态误报纠错

anchor无更新门禁实际通过：完整trace、selected logp、58张梯度、RNG均与serial一致。退出阶段两个worker的cmdline变空，被v2监控误报；原始采样显示同PID此前各37次为自有进程，且唯一误报在RESULT写完30秒、梯度文件写完20秒之后。证据保存在MONITOR_FALSE_POSITIVE_AUDIT.json，原log/状态保留。新execute_v3登记已确认自有PID贯穿退场，仍拒绝未知外部进程；模型源码及门禁阈值未改，也未重跑真实更新。后续最终标记为QUEUE_V3_DONE.json，RUN_STATUS仍为权威当前状态。

## 无更新门禁结果

`gate_serial`再次复现前轮legacy的58张梯度；`gate_anchor_v2`与`gate_rollout`均通过完整trace、hidden/fused、分母、58张梯度、selected logp及common RNG精确检查。新方案的rank0 backward event列表严格等于原slot/anchor顺序。新方案调用未改动的canonical runtime.score，58张梯度全部存在，未使用student缓存。

该阶段不构造optimizer，BUDGET仍为0/24；诊断时间不作为完整更新speedup。原anchor仅逻辑梯度P2P载荷为322,976,941,056 bytes；新方案传完整CPU trace，准确序列化载荷见gate_rollout/cursor119/RESULT，Head版本及wire SHA均通过。后续计时还需计入真实optimizer和广播。

### 单卡post-step参考选择纠正（optimizer之前中止）

静态复查发现det_serial的调度参数错误地把no-update gate_serial指定为post-step状态参考，必然把更新后的Head与旧Head比较。该自有诊断进程在BUDGET仍0/24、optimizer尚未进入时被定向SIGINT中止；无其他GPU进程被操作。原命令、log、partial目录和PRE_OPTIMIZER_ABORT.json保留。新execute_v4仅修正调度：单卡对前轮legacy trace/梯度校验后建立post-step基线，anchor/rollout再与该post-step基线比较；输出det_serial_v4，预算case仍det_serial，完整状态根中名称det_serial。冻结模型/梯度/损失源码未改，已通过的三个无更新门禁不重跑。后续最终标记为QUEUE_V4_DONE.json，RUN_STATUS与BUDGET为当前权威。

## 三次deterministic单步更新已通过

真实更新计数3/24。单卡、四卡anchor、四卡rollout均从同一完整step120重载，复用update119。两种四卡方案的58张梯度、Head、CPU fp32 master、Adam全部状态与各rank common RNG、optimizer step及schedule cursor均与单卡直接原始字节一致，证据为det_anchor_STATE_AUDIT.json及det_rollout_STATE_AUDIT.json。

新方案只发送122,965,129 bytes完整CPU trace，原anchor为322,976,941,056 bytes逻辑梯度P2P载荷。此为实际API载荷计数，不是物理链路流量。新方案完整deterministic诊断窗口84.873秒（含optimizer5.702秒与版本barrier），但最终性能结论仍等待default 3×3，不将deterministic诊断计时当作生产speedup。

CPU私有mmap moment原值加0诊断和小预算日志探测均未调用optimizer/Head或GPU，不占真实更新预算。首个单卡optimizer异常尚未完整解释；后续anchor/rollout为5.971/5.702秒，未因此更改加载器或优化器，也未删除原样本。

## default计时前的加载方式冻结补充

在未执行任何default性能更新之前，发现mmap诊断入口的serial optimizer首步出现34.863/97.248秒异常；CPU moment写页和toy optimizer探测不足以完整归因。canonical trainer本来使用非mmap的torch.load，故performance入口显式对齐该加载方式，保证完整CPU状态恢复在计时前完成。新opd_rollout_performance.py与诊断源码唯一差异为初始checkpoint mmap=False，详见BENCHMARK_LOADER_DIFF.patch；三个性能方法均使用同一入口。目标函数、模型算子、精度、warmup、顺序、阈值及24次预算不变。benchmark_context冻结新的源码身份；性能前另做CPU加载状态原始字节对拍。原诊断数据/源码/状态全部保留，不能把两种加载时间口径混称。


2026-10-19 换卡续跑：用户明确要求“启动吧，在别的卡”。账本18/24次已完成；旧调度器已退出，RUNNING为残留状态。新execute_v6以GPU0/2/6/7（UUID绑定，rank0=GPU0）仅执行原预算中未启动的perf1/perf2六次，保留原pair0及全部历史，不重试、不加额度。各新配对组三方案使用相同卡组；原pair0用GPU1/2/6/7，跨组硬件变化必须单独披露，不能宣称三组完全同卡。pair0 rollout有完整结果和完成账本，但旧控制器退出状态未记录，最终审计需保留该限制。最终队列标记QUEUE_V6_DONE；不依据旧RUNNING判断存活。Block HOLD、top1关闭、旧serial默认不变。

## 最终结果（2026-10-19T10:00:23+08:00）

真实更新reserve/complete为**24/24，上限24**。最终队列状态`COMPLETE_GO_MINIMAL_OPT_IN_PROPOSAL_WITH_DISCLOSURES`。全部状态diagnostic_only，不是正式第二seed或正式step121；原checkpoint/Target/数据与日程未覆盖。

- det_anchor：直接原始字节状态审计PASS，检查项`{'head': True, 'master_optimizer': True, 'recipe_identity': True, 'optimizer_step': True, 'schedule_cursor': True, 'per_rank_rng': True}`。
- det_rollout：直接原始字节状态审计PASS，检查项`{'head': True, 'master_optimizer': True, 'recipe_identity': True, 'optimizer_step': True, 'schedule_cursor': True, 'per_rank_rng': True}`。
- continuous：直接原始字节状态审计PASS，检查项`{'head': True, 'master_optimizer': True, 'recipe_identity': True, 'optimizer_step': True, 'schedule_cursor': True, 'per_rank_rng': True}`。
- resume：直接原始字节状态审计PASS，检查项`{'head': True, 'master_optimizer': True, 'recipe_identity': True, 'optimizer_step': True, 'schedule_cursor': True, 'per_rank_rng': True, 'world_size': True, 'variant': True, 'slot_mapping': True}`。

额外CPU审计覆盖no-update、完整单步、连续四步及2+恢复+2的所有trace tensor原始字节（含signed zero），详见FINAL_RAW_TRACE_AUDIT.json。逐参数梯度、Head、master、Adam及RNG签名逐步保存；直接最终状态审计不只比较loss或输出。

### 3×3真实完整更新计时

| 组 | 单卡serial秒 | 四卡anchor秒 | 四卡rollout秒 | 新/serial speedup | 新/anchor speedup |
|---|---:|---:|---:|---:|---:|
| 1 | 197.677 | 89.248 | 72.212 | 2.7375× | 1.2359× |
| 2 | 197.389 | 89.661 | 71.396 | 2.7647× | 1.2558× |
| 3 | 197.917 | 89.303 | 71.442 | 2.7703× | 1.2500× |

| 中位指标 | serial | anchor | rollout |
|---|---:|---:|---:|
| wall秒 | 197.677 | 89.303 | 71.442 |
| GPU-seconds | 197.677 | 357.213 | 285.769 |
| rollout秒 | 180.559 | 53.056 | 52.820 |
| trace_transfer_and_denominator秒 | 0.001 | 0.005 | 0.548 |
| scoring_backward秒 | 8.609 | 26.643 | 8.624 |
| master_sync_and_norm秒 | 2.266 | 2.519 | 2.144 |
| optimizer秒 | 3.724 | 4.592 | 4.612 |
| broadcast_and_version_barrier秒 | 2.489 | 2.743 | 2.703 |
| trace跨rank载荷bytes | 0 | 0 | 122965129 |
| gradient跨rank载荷bytes | 0 | 322976941056 | 0 |
| Head广播源tensor bytes | 0 | 2097252864 | 2097252864 |
| trace D2H逻辑tensor bytes | 190246080 | 190246080 | 190246080 |
| trace H2D逻辑tensor bytes | 190246080 | 190246080 | 190246080 |
| 最大rank allocated GiB | 37.702 | 40.630 | 37.702 |
| 最大rank reserved GiB | 38.955 | 42.795 | 38.871 |

配对中位speedup：对serial **2.7647×**，对anchor **1.2500×**；预注册性能门禁`True`。GPU-seconds与墙钟分别判定，不把多卡时间变短解释成算力消耗下降。

9个样本的实际forward计数是否全同：`True`；首个计数为`{'target_model': 3304, 'lm_head': 3926, 'draft_head': 416}`。各rank显存、所有阶段及原始载荷计数见PERFORMANCE.json与各cursor119/RESULT。计时不含profiler，外部进程、SM clock、功率、温度及主机loadavg采样见*_load.jsonl；这些是共享主机上的配对实测，不声称独占整台服务器。

CPU trace/gradient载荷是实际发送的序列化/通信tensor bytes，不是物理网络计数；Head broadcast记录源tensor bytes和接收端数。Head SHA本身的D2H、master梯度传输及write-back已计入阶段耗时，但未用硬件总线计数器独立计量。所有小tensor创建和原有CPU同步保留。

### 换卡与退出记录限制

原第1组使用GPU1/2/6/7，后两组按用户授权换为GPU0/2/6/7，rank0从GPU1换到GPU0；每组内三方案使用相同GPU。跨组三次不能宣称完全同卡，配对比值与原始秒数分别报告。第1组rollout已写完整RESULT、COMPLETE及预算完成记录，但旧调度器消失，未留下正常exit记录，日志尾有TCPStore关闭告警，原因未确定。该组只作为有退出记录限制的支持证据；后两组均有正常exit与负载监测。最小opt-in建议不表示生产默认已放行，也不能消除该证据限制。

### optimizer异常及范围

首个deterministic单卡optimizer阶段34.863秒，后续deterministic anchor为5.971秒。CPU私有mmap moment add_(0)探测首写2.200秒/重写0.279秒，minor faults分别2,176,356/29，major faults均0；小预算日志落盘探测0.097/0.180秒。这些探测没有Head更新、backward或optimizer step，也未修改真实预算。它们不足以解释首个异常，不能把34.9秒全部归因mmap或日志。正确性诊断未改加载器或优化器，原异常不剔除；default性能入口在9次计时前冻结为canonical eager CPU加载，只有初始化mmap=False这一行变化，三方案一致；完整初始状态逐字节对拍通过。此更正用于排除尚未实化checkpoint的计时风险，不声称CPU探测已完整解释异常。

### GO / HOLD 与最小生产接入

**GO：仅建议进行最小、显式opt-in的生产接入；生产默认尚未切换。** 正确性、连续轨迹、同拓扑恢复与3×3性能门禁均通过。建议仅把canonical trainer的collect段抽为executor接口：默认serial，显式四卡分支使用rollout_parallel的trace协议；rank0继续原scoring/backward/master sync/clip/step循环；每次step后广播Head并完成版本barrier。新CLI开关尚未添加，不能把建议参数当现有命令。checkpoint增加per-rank RNG、拓扑、slot映射、policy/cursor与源码身份；拒绝静默跨拓扑恢复。正式启用和任何新增真实更新需新的明确授权，不据本报告启动第二seed或长训。

Block继续HOLD，top1合并关闭，原canonical serial路径可用。冻结loss、精度、anchors、prompt与on-policy语义均未改；未用新数据、第二seed或长训补性能。

### 可复现入口与回退

相同环境为`/home/mcy/.conda/envs/fin_research/bin/python`。执行前重查GPU UUID和负载，新out必须realpath落在允许挂载点，不能覆盖历史目录。以下只运行无更新门禁：

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=GPU-8af87d4f-b2b2-404f-e7b2-6b39cbedcd55,GPU-be4999dd-0619-aeb4-c83d-97d249fd5476,GPU-4dcc1d4c-5cb3-6751-2307-bbe23795dc81,GPU-41ab56ab-b56f-b007-b7e9-a7bb8f0b02f8
PY=/home/mcy/.conda/envs/fin_research/bin/python
RUN=/mnt/local3/mcy/GLANCE/results/opd_rollout_only_20261018_204925
OUT=/mnt/local3/mcy/GLANCE/results/NEW_UNIQUE_NOUPDATE_DIRECTORY
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -- \
 repro/diagnostics/opd_rollout_only.py --run "$RUN" --out "$OUT" \
 --case manual_no_update --variant rollout --kernel deterministic --updates 0 \
 --reference "$RUN/cases/gate_serial"
```

单卡诊断把CUDA_VISIBLE_DEVICES改为GPU1 UUID，直接python启动、variant改serial；生产fallback仍为原canonical trainer/serial_reference。真实更新及2+resume+2的精确命令和环境保存在*_command.json；本轮预算不允许自动重放，须另建新预算且获得授权。不得用本轮diagnostic state接入正式训练。旧6次和本轮24次为独立、已记录的授权，不混算。

交付包含源码修改前快照、执行版本、最终源码包、差异、CPU测试、完整trace/原始梯度、逐步状态签名、最终完整状态、加载器/监控纠错证据、运行负载与逐文件manifest。大.pt文件在写入时计算SHA并fsync，manifest复用该SHA并检查文件大小；小文件在收尾重新读取计算SHA。该方式明确不同于对全部大文件做第二次独立读盘SHA。恢复与直接状态审计确实重新读取对应完整状态。
