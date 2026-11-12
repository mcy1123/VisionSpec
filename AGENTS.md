# GLANCE 当前交接

## 2026-11-13 首夜已结束（07:42 北京时间检查）

夜间54任务46成功、8失败，738更新训练预算已执行完毕。128帧warm τ7.0384；A/B/C/D的λ10分别7.5257/7.5606/7.5571/7.5587，warm加八臂共360/360逐token无损；各架构λ10优于λ0，但C/D相对B区间跨零。全部128输出到1024上限、无自然EOS。64帧八臂均39/40无损，同视频vgWlRdNOGx4在索引65处分叉，必须保留FAILED；未重试、不能作为无损加速正证据。详见既有实验报告11.7与run的overnight_timeline_20261113/DONE.json。当前没有本夜剩余训练预算。

占卡等待器曾因状态文件短暂读取失败而退出；已归档原日志，增加有限读取重试并按既有授权重新启动。八卡占卡负载不计为训练进度；新增科研任务前仍须先STOP等待器并确认子进程退出。

## 2026-11-13 用户授权训练结束后的空闲占卡

用户明确要求检查并启动 `/czsun/mcy/code-l` 的占卡程序，避免训练结束后空卡。原程序 `/czsun/mcy/code-l/gpu.py` 未运行，现已用tmux `visionspec-idle-fallback` 启动等待器 `repro/local_h100/idle_fallback_after_queue.py`。仅当当前scheduler所有任务均SUCCEEDED/FAILED/BLOCKED、队列有效后，运行原程序的 `--wait-idle --idle-seconds 60 --no-fill-vram`；GPU UUID顺序按NVML物理索引显式对齐。当前排队期间不占卡，避免run_job拒绝外部CUDA进程。此为用户明确新增的空闲负载授权，覆盖此前“不用空算占卡”的限制，仅适用于本等待器。状态/日志位于run的 `control/idle_fallback/`，创建其 `STOP` 可退出。

以后增加新训练前，先设置该STOP并确认fallback所有子进程退出，再入队；后台虽检测新队列会退出，但不能用轮询保证无启动竞争。不得把占卡利用率当训练利用率或科研产出。不修改外部gpu.py，不影响其它watchdog或用户进程。

## 2026-11-13 夜间任务：用户明确授权排队训练

用户要求“一晚上不断去训练”“把任务都安排上，不要做太多验证”，并重申不要大量保存checkpoint。已追加54个有限任务至现有tmux scheduler，协议 `/czsun/mcy/video_opd_pilot/local_h100_20261112/overnight_timeline_20261113/PROTOCOL.json`。复用 `data/full` 的80 train/40 dev和旧事件时间线prompt，真实128帧、max_new1024、fp32 Target、b63/k8、逐token独立AR门禁。数据split为本机seed20261112，旧成熟head不可用，不声称逐项复现。

公共DFlash先做SFT80作为共同起点；A原结构、B同容量文本读出、C视觉spatial_bins2、D视觉spatial_bins1，分别OPD λ0/10各80更新，共八臂，seed20261113。C/D均time_encoding off，D不是新的时间编码架构。共738更新（warm80、正式640、必要gate18），2952 QA。门禁失败仅阻断依赖分支，不自动重试或扩大预算；未上线新视觉加权损失。各头完整128/64帧dev40，单seed开发筛选，不动holdout。

每训练任务save_every=updates，只存终点；warm及八臂终点保留，门禁二进制由配套reporter在对应评测成功后按审计流程退休。SFT为weights-only，OPD保留最终fullstate但不承诺精确resume；不为省空间改运行中训练数学。预计正式checkpoint约130GiB量级，另初始化约8GiB，禁用频繁中间点。tmux `visionspec-overnight-report` 每分钟收集已有结果/清理gate，最长14小时，不读GPU遥测；训练由有限队列自然结束，14小时仅是reporter上限。未恢复人工持续监控。首夜完成架构×λ筛选；第二seed与新多模态loss仍属于后续阶段，未冒充已经运行。

## 2026-11-12 用户授权清理 checkpoint 与稀疏保存

用户明确要求“保存的太多了，清理一些，以及不要这么频繁地保存”。本条覆盖下文禁止删除历史 checkpoint 的规则，但仅限本次 run 经依赖检查确认可退休的中间/诊断二进制；不得删除日志、评测 JSON、曲线或协议。未来训练默认只保存最终/分段终点（显式 `--save-every` 取该任务完整更新预算），只有实际中间评测或恢复需要才额外保存。不得沿用本轮每20步保存为未来默认。

当前六臂 λ10 的 CLI 已冻结，不能动态修改间隔或为了省空间重启训练。通过 `repro/local_h100/checkpoint_retention.py` 自动退休无后续引用的旧点：保留全部正式 step80、待运行/运行中队列引用、初始化、每臂最新发布点以及两份指定恢复诊断 step1。清理仅移除 `model.safetensors`/`training_state.pt`，原元数据原位保留并另存，`RETIRED.json` 标明其已不可加载，历史 SHA 不代表二进制仍存在。审计位于 run 的 `analysis/checkpoint_cleanup_20261112/`，策略位于 `control/CHECKPOINT_POLICY.json`。

首轮退休75个 checkpoint、释放695.31 GiB。有限 tmux `visionspec-checkpoint-retention` 只做文件依赖与回收，等六臂训练及原中间点评测结束后退出，最长12小时；不操作 GPU、不恢复已解除的遥测监控、不启动训练/重试/追加预算。

## 2026-11-12 13:05 UTC 用户授权补齐 λ10

用户明确要求“对，补上吧，你是不是分支搞错了，你现在是哪个分支”。当前仍为 `run/local-h100-fresh-20261112`；fetch 后确认包含 `origin/exp/visual-opd@06ca7a7`、`origin/run/real-video-20261111@2d39df7`、`origin/feat/temporal-evidence-readout@1aa387c` 和 `origin/fix/visual-opd-long-video-protocol@ab6e2fa`。此前缺少的是实际参数中的 λ10 对照，历史辅助项代码已在当前分支。

新增有限授权及队列见本机 run 的 `LAMBDA10_PROTOCOL.json`：A/B/C × 两 seed，各从与原 λ0 相同的初始化 fresh Adam 训练80步，唯一改变 `e2e_tv_weight:0→10`；正式480次更新，另三架构各2步门禁，共486次真实更新、1944 QA。六个真实数据 dry-run 已逐字段证明训练协议仅 λ 不同，禁止 CUDA 初始化的预检通过。沿用原 pilot 入口/数值数学以保持可比，不从 λ0 step80 或 smoke 继续，不冒充旧 checkpoint 的完整 resume。保存20/40/60/80，完整dev40及真实256/512帧部署门禁；全部新增评测732对 AR/SPEC。协议和37个新增有限任务已入队，先架构 full-Adam peak2+dev2 门禁，失败不重试/不加预算，已有任务定义及产物不变。后台 tmux scheduler 继续；下文监控解除仍然有效，不恢复遥测或持续人工监控。

## 2026-11-12 11:34 UTC 用户解除持续监控

用户明确要求“别监控了，你预计一下结果啥时候出来”。已设置本机 run 的 `control/STOP_MONITOR`，结束人工及遥测持续监控；已有 tmux 调度、有限训练与评测队列继续，未设置 STOP_TRAIN/STOP_SCHEDULER，不追加大训练。此条覆盖下文持续监控要求。最新冻结预算另含长输入 1,053 对 AR/SPEC，以及完整恢复诊断六次实际更新、八对部署 AR/SPEC；详见 run 内 `LONG_INPUT_DEV40_PROTOCOL.json`、`LONG_RESUME_GPU_GATE_PROTOCOL.json`。后续进度仍读取 scheduler `STATE.json`；GPU_STATUS 已停止刷新，不能当实时状态。

## 2026-11-12 本机 H100 独立重建与持续监控

用户已明确授权在当前服务器从公共模型重新训练，使用八张 H100 执行实际实验，要求利用率至少 50%，并持续监控直到用户明确解除。本节覆盖下文旧服务器的已耗尽预算及“不持续监控”状态；每个新任务仍须记录明确预算，不盲目重试失败任务，不用空算占卡。加载、保存和 CPU 处理会出现低谷，必须如实报告，不能声称始终满足利用率要求。

- 本机代码 `/czsun/mcy/VisionSpec` 是 Git 工作树，当前分支 `run/local-h100-fresh-20261112`；Python `/czsun/mcy/envs/visionspec/bin/python`。用户已指定本机模型和数据放 `/czsun/mcy`，下文 `/home/mcy`、`/mnt/local*` 属旧服务器，不能用于本机路径推断。
- 本机 run：`/czsun/mcy/video_opd_pilot/local_h100_20261112`。模型为经官方 SHA 核验的 Qwen3-VL-8B-Instruct 与公共 Qwen3-8B-DFlash-b16；没有使用另一服务器的训练 checkpoint。“从零”指本机重新适配公共预训练草稿，不是随机初始化。
- 使用 GPU UUID 与 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。tmux `visionspec-monitor` 每 5 秒记录，`visionspec-scheduler` 调度显式有限任务。实时状态以 `control/GPU_STATUS.json`、`control/scheduler/STATE.json` 为准，不能把本文时间截面的进度当实时结果。队列不允许修改已启动定义；失败及 BLOCKED 不自动重试。
- 正在执行 128 帧、320 train / 40 dev 的 A_plain/B_text/C_visual × SFT/OPD × 两 seed，每臂 80 连续更新、4 QA/update、fresh GPU fp32-master Adam、lr1e-5。全部共用本机 warm20，B/C 新增参数匹配、显式时间编码置零并冻结。主比较是 C−B 及其 SFT/OPD 交互；generic OPD λ0 尚无视觉条件损失加权。协议 `FACTORIAL80_PROTOCOL.json`，小型审计在 `analysis/`。
- 80 个后续 holdout 仅冻结来源，未生成/评测。当前 bounded summary 自然输出约 215–233 token，不能写成千 token 长推理。旧自由生成的重复尾部及高 τ 属诊断，不能混入当前结果。已有 512 帧、24,071 输入 token / 21,504 视觉 token 的单视频独立 AR/SPEC 无损门禁，不构成长输入方法有效性的统计证据。
- `mature_warm80_*` 与同 seed `factorial80_*_A_plain_sft_*` 是误排的重复配置，不是额外强基线或独立 seed。原始结果保留，正式统计只用 factorial ID；差异及解释更正在 `analysis/mature_factorial_duplicate_config_audit.json` 与 `DECISIONS.jsonl`。
- 部署仍必须 fp32、逐 token SPEC=独立 AR。并发运行的墙钟数据只作描述性诊断。现有 12 臂不切换实验性的 packed OPD scorer；该候选仅完成 CPU 检查，bf16 更新数值会变化，尚未获 CUDA 数值/显存验证。
- 本机执行说明追加到[视觉读出实验计划](docs/reports/2026-11-12-temporal-evidence-architecture-plan.md)。不删除既有 checkpoint、原始结果或其它用户进程；下文其余方法契约仍适用。

## 2026-10-19 及后续旧服务器交接

本文件只记录当前有效状态与工作规则。整理前全文见[本次归档](docs/archive/AGENTS-before-cleanup-20261019_135449.md)，切换前交接见[历史 AGENTS](docs/archive/AGENTS-before-pure-default-20261019.md)。历史文档中的“默认 serial”“尚未启动”“正在运行”和旧授权均是当时状态，不是当前指令；完整结果从[报告索引](docs/reports/README.md)查阅。

## 2026-10-26/27 视频长生成 OPD 测试（用户授权）

用户明确要求“直接开训”，并要求所占用卡（物理 3/6/7）不空到 2026-10-27 08:00；2/4/5 与 0/1 未占用。新增显式视频入口 `repro/video_caption/train_video_opd.py`（`--modality video`、三卡 `allow_fresh_start`；ScienceQA 续训入口的冻结保护未动）与数据适配 `repro/video_caption/build_pool.py`。

结果（VideoDetailCaption 439 train、独立 fresh30、平均生成 **270.9 token/条**、fp32/b63k8/AR 逐 token 门禁、n=30）：warm 3.7266 → 1e 3.7180（+0.009，CI 跨零）→ **3e 3.7841（+0.0575，CI [+0.014,+0.102]）** → 4e 3.7104（回落 0.0737）；低 lr（1e-6）从 3e 续训不变，说明 4e 回落是步长而非数据耗尽。SFT 对照崩溃（lr 1e-5：1e 3.3336、3e 1.9168；lr 3e-5：1e 1.9164），两个 SFT 臂都在约 2.5% 权重漂移处落到 τ≈1.917 平台，而 OPD 漂移 ≤0.3%；同 lr/步数 OPD−SFT=+0.3844（30/30 胜）。`video_heldout_30` 30/30 在 train 内，不能当独立集。单 seed、n=30、stage 间 Adam 重置，不外推为“视频领域 OPD 有效”。同卡测速（12 行 ×2、fp32、顺序交替、48 次全无损）：OPD 3e 相对 warm 端到端 1.012×（CI [0.994,1.030]）、decode-only 1.013×（CI [0.992,1.034]）——τ 增益显著但墙钟只有约 1%。 2026-10-27 清理：删本 run 的 15 个中间/诊断 checkpoint 释放 164.1 GiB，保留 `run/step110`、`run_stage2/{step165,step220}`、`run_stage3/step110`、`run_stage5/step110`、`sft_fp32_lr1e5_seed0/{step110,step330}`；清单 `CLEANUP_20261027.json`，eval/曲线 JSON 未动。

绑卡注意：本机数字 `CUDA_VISIBLE_DEVICES` 会被 CUDA 枚举重映射（`3,6,7` 的 rank0 实际落到物理卡 1，导致与别人任务共享并 OOM），必须用 GPU UUID + `CUDA_DEVICE_ORDER=PCI_BUS_ID`。产物：`/mnt/local3/mcy/GLANCE/results/video_caption_opd_20261026/`（`REPORT.md`/`PROTOCOL.json`/`STATUS.md`），报告见[视频长生成 OPD](docs/reports/2026-10-26-video-caption-opd.md)。

## 2026-10-20 新增授权：三卡续训两个 epoch

用户已明确授权从 ScienceQA 全量首轮 SFT/OPD 各自 step1182 继续两个 epoch，每臂新增2364次更新（9452 QA），最终累计step3546。固定预算，不因开发集平台早停；启动后后台独立运行，不持续监控。此授权覆盖下文历史“没有剩余训练授权”，只适用于本次续训，不能自动追加。

独立三卡入口：`repro/opd_official_kl/three_card_trainer.py`；3个物理rank保留4个逻辑slot及4条RNG流，rank0处理slot0/3，其余rank分别处理1/2，保持rank0按slot0–3有序梯度和4 QA/update。原四卡默认入口不变。只能载入显式迁移的三卡checkpoint，迁移保留Head/master/Adam和完整RNG，禁止静默跨拓扑恢复，不声明跨拓扑逐位等价。

产物：`/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020/`。**该轮已于 2026-10-21 18:56 完成并写 `DONE.json`/`FINAL_REPORT.md`**：两臂各到 step3546（新增2364次更新、9452 QA），最终 test200 逐token 200/200 无损，准确数同为162/200；复用confirm200上 OPD τ 6.2206 vs SFT 5.3666（+0.8541，95% CI [+0.7157,+1.0037]），仅为描述性、非独立确认。2026-10-21 另按 10-20 速度报告的同一协议（24图、均衡顺序、fp32、所选卡无外进程）实测 step3546 加速比：AR/SFT 2.295×、AR/OPD 2.522×、**SFT/OPD 1.099× [1.063,1.141]**，144次生成逐token无损；机制是每轮耗时相同（114.5 ms/round）而 OPD 轮数更少（τ 5.306 vs 4.833），细节见报告「加速比复测」。**ChartQA 跨域复测（同协议、ScienceQA step3546 头、短答 4.6 token/题）几乎没有加速：AR/SFT 1.093×、AR/OPD 1.022×、SFT/OPD 0.935× [0.852,0.988]**，原因是 prefill 占 73–82%、答案太短撑不满 block，见报告「ChartQA 跨域复测」。**DocVQA 探针（24 题、跨域头、同一批图只改 prompt）**：短答 6.2 token 时 SFT/OPD 1.012× [0.996,1.031]、AR/OPD 1.144×；改成先推理后作答、答案 139.5 token 后 SFT/OPD 1.036× [1.021,1.052]、AR/OPD 2.221×——加速比由答案长度决定而非数据集，见报告「DocVQA 探针」。数据在 `/mnt/local/mcy/GLANCE/datasets/docvqa_slice/`（公开小样 24 题），计时产物在 `/mnt/local/mcy/GLANCE/results/docvqa_speed_step3546_20261021/`。**2026-10-21 已按用户指令清理中间 checkpoint**：只留 dev τ 最好的 `sft/to_3364/step3364` 与最后的 `sft/to_3546/step3546`、`opd/to_3546/step3546`（3 个均复核 SHA256），删 23 个释放 359.4 GiB，`/mnt/local` 余量 502 GiB；清单与各 checkpoint 的 SHA/config 副本在 `CLEANUP_20261021.json`。`migration/{sft,opd}/step1182`（32 GB）作为输入身份保留。曲线/日志/eval JSON 未动，中间点重新推理已不可行。tmux `glance-sqa-ep23-3gpu` 随完成自行结束。运行期间经历 2026-10-20 21:20 OOM 与 21:25 换绑（物理卡1/2/7、门控38 GiB，每rank实测约36.8 GiB），事件见报告「运行事件与换绑」与 `rebind_20261020/REBIND.json`；`FINAL_RESULT.json` 的 protocol 块仍写0/1/2，那是启动时冻结值，不是实际运行卡。以后任何重启按≥40 GiB空闲或三张全空卡安排，勿再按38 GiB放宽。协议、快照、迁移和可恢复命令见[三卡续训报告](docs/reports/2026-10-20-scienceqa-three-card-continuation.md)。当前confirm200已复用，不作为未来新方法的独立最终确认集。

## 当前默认与方法边界

- ScienceQA image OPD 默认为 `opd_mode=pure_block`、`parallel_rollout=4`。入口：`repro/scienceqa_image/train_sqa_opd.py`；`repro/experimental/train_pure_block_native.py` 是同一 CLI 的别名。少于四张可见 GPU 或非四 rank 启动直接失败，不静默降级。
- 四个 rank 使用同一更新前 Head，各自完成一个 slot 的 rollout；rank0 按 slot0–3、原 anchor 顺序执行带梯度 scoring/backward 和 CPU master optimizer，然后广播 Head。保留 Head version 检查与 barrier。
- pure block-native 直接保留 block KV/hidden/fused，不执行旧 `1+keep` serial 重建；每轮一次 block Target forward，轨迹末尾仍有一次 root materialization。旧 serial 与旧 exact block 已归档，不在 active runtime 中。
- native 是允许训练轨迹、teacher logp 和梯度偏差的近似方法，不声明与 serial 等价。旧 exact block 的 HOLD 只适用于归档方法，不阻拦用户已批准的 native 路径。
- loss 仍为 Bernoulli forward KL＋rejected k3、原池化、decay=0.9、流权重1:2；Target fp32、Head bf16、CPU fp32 master AdamW。保持原 clamp、anchor plan、prompt 与日程；未启用 top1 合并。
- Bernoulli＋k3 是用户批准的方向覆盖，不是 Draft-OPD 基础脚本的默认配方；公式与来源见[KL 契约](docs/methods/draft_opd_official_kl_contract.md)，其中旧入口描述以本文件为准。
- 部署 decoder/verifier 未改，部署输出必须逐 token `SPEC=独立 AR`，零容差。输出无损不证明训练梯度等价。
- checkpoint 身份包含模式、world size、slot map 和 Head version；禁止 serial/native 跨模式或跨拓扑静默 resume。旧视频 recipe 随共享 serial 代码归档，当前视频入口明确失败并提示恢复，不默认切换视频方法。

## 已完成结果与授权

### 当前默认入口 smoke：GO_SMOKE

[默认切换报告](docs/reports/2026-10-19-scienceqa-pure-default-smoke.md)：从原 Warm step2000、原 update0 的四条 QA 完成 **1/1 次**真实更新；loss=4.52362209545263，optimizer_steps=1。CPU import/dry-run/fail-fast 通过，完整 checkpoint 保存及重新加载成功，dev8 **8/8** 逐 token无损。

本次只证明当前入口能跑；未执行梯度等价、连续更新或恢复测试，不声明稳定训练/恢复能力。该 checkpoint 标记 `block_native/smoke/diagnostic_only`，不能用作正式 step121。产物：

- 快照、代码 diff、SHA、日志与回滚脚本：`/mnt/local3/mcy/GLANCE/results/scienceqa_pure_default_smoke_20261019_131714/`。
- smoke checkpoint：`/mnt/local3/mcy/GLANCE/checkpoints/scienceqa_pure_default_smoke_20261019_131714/run/step1/`。

### 切换默认前的 native 120 步实验：已完成

[pure-native 实验报告](docs/reports/2026-10-19-pure-block-native-entry.md)末尾的最终结果为准，前面的启动进度是历史记录。原 Warm step2000、480 QA、seed=20261013、lr=1e-5、4 QA/update，完成 **120/120** 更新；平均 25.74 秒/update（不含 checkpoint 写入），训练总墙钟约55.5分钟。该耗时不代表与 serial 的干净配对加速。

部署 dev8 为8/8、复用 confirm200 为200/200逐 token无损。confirm200 的逐图像有效 token/轮 native=4.44455、serial=4.44569，配对差−0.00114，95% CI [−0.00742,+0.00498]；两者全题准确数均153/200。这是复用集合的描述性比较，不是新独立确认或等价证明。Warm 50 QA观察审计未发现 committed/output 分叉，匹配前缀 teacher logp 最大绝对差0.0001831。

结果目录：`/mnt/local3/mcy/GLANCE/results/pure_block_native_20261019_110343/`；checkpoint 位置读取其中 `STATE_ROOT.txt`，不根据目录名猜测。该批 checkpoint 为 `diagnostic_only/block_native`，与 smoke 及旧 serial 结果分开标记。

**当前没有剩余训练授权。** native 120次、默认入口 smoke 1次，以及此前9/24/6次独立预算均已用完；不得自动追加、重试、启动120步或第二 seed。新增真实训练须用户另行明确授权。本次 AGENTS 整理不包含训练或评测授权。

## 基线与历史证据入口

- 新图像 Warm 由文本 DFlash Head 训练2000步得到，Head SHA：`bd16ebcff1c513d42b14d9b8d89d428d93973ff8620bb78eafa9689509616735`。旧 Warm `fc0ad5a6…` 未恢复，不把新 Warm 当旧权重；最早完整 RNG 恢复点是 step2，不声称有 step0 RNG dump。
- 原 ScienceQA SFT120/serial OPD120 从同一新 Warm、同一480 QA日程出发，结果根：`/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/`。其 `READ_ME_FIRST.md`、最终报告和独立验收为权威证据。
- 原独立 confirm200：`/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/`。I0/SFT/serial OPD均200/200输出无损；有效 token/轮4.2267/4.3708/4.4457；OPD−SFT=+0.0749，95% CI [+0.0444,+0.1063]。准确数均153/200；共享负载下计时不作可靠墙钟加速证据。
- 旧 serial rollout-only 的2.7647×性能结果、确定性梯度协议和恢复检查属于旧方法，不能直接转作 native 验收。见[rollout-only报告](docs/reports/2026-10-18-opd-rollout-only-parallel.md)、[旧 opt-in 接入报告](docs/reports/2026-10-19-opd-production-optin.md)。
- 更早 fixed-AR CE/full-reverse-KL、Task A/P1 等结果不与当前方法混算。原始报告、清理 tombstone 和历史 checkpoint 身份保持不变。

## 存储与执行规则

- 数据集、预处理数据、缓存、权重、checkpoint 和大临时文件必须实际存放于 `/mnt/local`、`/mnt/local2` 或 `/mnt/local3`；写入前核实 `realpath`。不得放入 `/home/mcy` 实体目录；home 中指向这些挂载点的软链接不得替换为实体数据目录。
- 代码根 `/home/mcy/Projects/GLANCE` 不是 Git 工作树；改代码或文档前建立可恢复快照。Python：`/home/mcy/.conda/envs/fin_research/bin/python`。外部 Draft-OPD 与 refs 只读。
- 不覆盖历史 checkpoint、报告、原始结果或运行中的脚本；不删除历史产物，除非用户另行明确授权。执行前检查 GPU 占用，不终止或破坏其他任务。
- lr≤1e-4 使用 fp32 master；记录精度、训练预算与 optimizer。不得改部署 decoder/tree 或权重迎合指标。
- 用户偏好后台独立运行、完成后一次通知，不要求持续人工监控。通知脚本可写 `DONE.txt/DONE.json`，使用可用的 tmux/桌面通知；不能承诺自动聊天消息，不训练、不重试、不加预算。
- 新实验报告登记 `docs/reports/README.md`；不另建平行“最新项目总结”。不自动恢复历史待办、C7、D4或旧训练计划。

## 旧实现归档与回滚

完整切换前代码为 smoke 产物目录中的 `source_before.tar.gz`；旧 serial/1+keep 独立副本在 `archived_legacy/`，SHA 与 `changes.patch` 同目录。恢复旧 active 代码的命令如下，仅在明确需要回滚时执行；它会恢复整个切换前代码快照，不能携带 native/smoke checkpoint 到 serial：

```sh
/home/mcy/.conda/envs/fin_research/bin/python /mnt/local3/mcy/GLANCE/results/scienceqa_pure_default_smoke_20261019_131714/rollback.py
```
