# 真实视频的时间感知视觉读出架构实验

本轮直接验证 draft 的视觉架构：固定 target 的128个真实采样帧，比较原 head、文本上下文读出和视觉读出。交付分支为 `feat/temporal-evidence-readout`，基于 `run/real-video-20261111@ab7f495`。代码与 CPU 验证在开发机完成，真实训练、逐 token 无损和速度测量由 dsh 在有空闲 GPU 的服务器执行。

核心问题是：在同一 OPD 配方下，专门读取视频证据是否优于原结构及同参数的上下文读出。C−A 衡量增加模块后的收益，C−B 判断收益能否由额外上下文读取解释。本轮不同时改变帧数、target、树预算或辅助损失。

## 1 实现与特征来源

新增 `dflash/evidence.py`，由 `dflash/model.py` 在 draft 最后一次 norm 后、词表投影前调用。原 fused-context 主路保留，target 与 decoder/verifier 不变。

```text
同一次 target prefill 的视觉位置特征
                 ↓
      时间分组与每组 2×2 空间池化
                 ↓
       最多256个冻结 memory token
                 ↓
当前 draft hidden → 256维软注意力 → 门控残差 → token logits
```

为直接复用既有采样特征，实际采用**最后一个保留层，即第33层的 hidden**，不是 target 最末层，也不是额外视觉编码器的输出。B/C 从相同层取特征。现有主路先将这些 target hidden 转为 head 的 bf16；池化和新增模块的计算使用 fp32，不声称保留原始 fp32 target hidden 的全部精度。

- **C_visual**：根据实际 `video_grid_thw` 和 spatial merge 恢复时空排列，每个时间组池化为最多4个空间特征；当前128帧通常为64时间组，共256个 memory token。时间编码取对应真实采样帧时间的均值，再除以视频时长。更大的网格会按固定容量同时做时间池化，不复制帧。
- **B_text**：读取同层已经提交的非视觉位置 hidden，随前缀增量更新，最多池化为256个 token。其位置编码是已知前缀内的归一化位置。文本 hidden 可能包含视觉信息，因此它是上下文读出对照，不称完全无视觉。两臂参数和容量上限相同，实际 memory 长度与计算成本不保证相同。
- **残差**：只有输出投影初始化为零；门控初始为0.5。首步仅输出投影可能有非零梯度，后续应检查 query/key/value 的梯度是否打开。
- **缓存**：冻结池化视觉特征绑定输入身份并随每条 trace 搬运。推理时静态视觉 K/V 可缓存，检查请求及参数版本；带梯度 scoring 总是重算投影。B 的上下文随前缀变化，不复用静态 K/V。
- **状态隔离**：训练每次收集4条 QA 后，scoring 对每条重新绑定其 request；部署状态附在请求的 draft cache，支持 crop，禁止用 draft noise 或尚未提交的答案作为 memory。
- **精度**：新增 readout 参数为 fp32，原 head 为 bf16，CPU fp32 master AdamW。使用 Transformers 的严格 fp32 模块保留机制，并有保存/重载逐参数检查。

本轮没有硬 top-k 选择器、接受收益排名监督或新的视觉编码器。OPD 梯度直接训练软注意力及残差，先验证架构是否值得继续。

## 2 固定实验协议

| 项目 | 本轮固定值 |
|---|---|
| 数据 | 既有 `data_parallel`，80 train / 40 dev，128真实帧，按源视频隔离 |
| 任务 | 原 chronological visible-event prompt，无字幕、无强制最短答案 |
| Target | 与 head 匹配的 Qwen3-VL-8B-Instruct，fp32，balanced，多卡 |
| 共同起点 | 上轮128帧 λ10 的 step20 权重，核对路径和 SHA；三臂均 fresh Adam |
| A | 共同起点原结构，继续相同 OPD 配方 |
| B | 同一起点加 text readout |
| C | 同一起点加 visual readout |
| 初始新增参数 | B/C 同 seed、逐 tensor 一致，保存的权重文件 SHA 相同，仅 mode 配置不同 |
| 训练预算 | 每臂连续60 updates，4 QA/update，3 epoch；总计180次真实更新 |
| Seed | 20261112；共享同一份重新生成的训练 schedule |
| 优化 | lr=1e-5，原 Bernoulli/k3 OPD＋λ10 top8-overlap prefix proxy，global_pooled |
| 输出与上下文 | max_new=1024，max_seq_len=32768；先不因截断改变主实验任务 |
| 保存 | step20 / step40 / step60；最终主比较固定 step60，曲线比较用共同步数 |
| 部署评测 | fp32、b63/k8、real time、immediate EOS、每条配独立 AR、逐 token 零容差 |

共同起点已有 OPD 训练经历，本轮60步从权重重新初始化优化器，不是旧 step20 的连续优化器恢复。初始化工具拒绝已经带有其它视觉/跨层适配器的源 head。不要拿32B路径替换当前8B target。

用户本次采纳三臂架构方案并要求交给 dsh 执行；本协议的**单 seed、最多180次真实更新**用于目标执行服务器，覆盖旧 AGENTS 中已经过期的“没有剩余训练授权”。不授权在 GPU 已满的开发机训练，不自动追加第二 seed 或调参训练。失败/调试中已发生的更新也算入180，不通过新目录重置预算。

按上轮262–303秒/update，180步仅训练约13.1–15.2小时，另需初始化、checkpoint、GPU门禁和评测时间。不能承诺一个8小时窗口跑完。旧任务的 `2026-11-12 08:00` 已结束，不把它作为本轮截止。若用户另设绝对截止，设置更早的 `EV_TRAIN_STOP`，为一次最长更新、保存及收口留余量。

## 3 执行服务器准备

代码在 Git worktree 中运行。若现有项目不是 Git 树，先保留可恢复快照并另建 checkout，不覆盖旧脚本和产物。

```bash
git fetch origin
git worktree add -b run/temporal-evidence-pilot ../VisionSpec-evidence-run origin/feat/temporal-evidence-readout
cd ../VisionSpec-evidence-run
git rev-parse HEAD
```

核实旧结果实际位置后设置环境变量。以下路径来自前一轮报告，目标服务器必须用 `realpath` 验证，缺失时定位真实路径，不悄悄换 head 或数据。

```bash
export EV_PY=/home/mcy/.conda/envs/fin_research/bin/python
export EV_TARGET=/mnt/local/mcy/GLANCE/models/Qwen3-VL-8B-Instruct
export EV_DATA=/mnt/local/mcy/GLANCE/results/real_video_20261111/data_parallel
export EV_BASE=/mnt/local/mcy/GLANCE/results/real_video_20261111/arms/seed20261111_lambda10/step20
export EV_RUN=/mnt/local3/mcy/GLANCE/results/temporal_evidence_新时间戳
export EV_SEED=20261112
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-实际UUID一,GPU-实际UUID二,GPU-实际UUID三

realpath "$EV_TARGET" "$EV_DATA" "$EV_BASE"
df -h /mnt/local /mnt/local3
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
```

`EV_RUN` 必须是允许挂载点下的全新目录；不先 `mkdir EV_RUN`，启动器自行创建。磁盘需容纳三臂各3个完整 optimizer checkpoint、初始化头与一次原子保存余量；参考旧 checkpoint 大小预留约200 GiB后再按实际文件估算。不能删除历史文件腾空间。数据已有真实帧 cache，不再重建/下载。

环境要求沿用上一轮可训练环境，尤其 `torch.nn.utils.get_total_norm`。本地旧 PyTorch 2.5.1 没有此 API；CPU测试通过不代替实际训练环境检查。Transformers 版本必须通过新增 fp32 保存/重载测试，不能静默接受 readout 被 bf16 量化。

## 4 CPU预检与后台执行

开发机交付检查：下列13个测试文件联合运行，**137 passed，49 subtests passed**；shell语法和diff检查通过。包含真实decoder调用的小型Qwen3 CPU逐token对照、readout梯度、fp32保存重载、训练trace隔离及wrong-video memory身份校验。尚未执行真实视频GPU门禁或任何真实optimizer更新，不能据此宣称架构已产生收益。

先独立运行 CPU 检查；启动器也会再次执行这组门禁：

```bash
CUDA_VISIBLE_DEVICES='' "$EV_PY" -m pytest -q \
  repro/tests/test_real_video_pilot.py repro/tests/test_pilot_schedule.py \
  repro/tests/test_video_9k_protocol.py repro/tests/test_video_visual_probe.py \
  repro/tests/test_video_eval_merge.py repro/tests/test_visual_context_cache.py \
  repro/tests/test_video_training_stop.py repro/tests/test_evidence_readout.py \
  repro/tests/test_evidence_training.py repro/tests/test_evidence_eval.py \
  repro/tests/test_evidence_analysis.py repro/tests/test_evidence_decoder_cpu.py \
  repro/tests/test_evidence_memory.py

CUDA_VISIBLE_DEVICES='' "$EV_PY" repro/video_caption/init_evidence_heads.py \
  --source "$EV_BASE" --out "$EV_RUN/initial_heads" --seed "$EV_SEED" \
  --evidence-dim 256 --max-tokens 256 --spatial-bins 2 --dry-run
```

`init_evidence_heads.py` 实际执行会逐 tensor 检查旧权重未改变，B/C新增参数和保存权重相同，记录 `INIT_MANIFEST.json`；不载 target、不读取旧 optimizer、不运行 GPU。

将上面的**实际环境变量**写入一个本 run 专用 `launch_evidence.sh`，末尾为：

```bash
cd /实际路径/VisionSpec-evidence-run
exec bash scripts/run_temporal_evidence_pilot.sh
```

由 tmux 启动该脚本，例如：

```bash
tmux new-session -d -s temporal-evidence 'bash /实际路径/launch_evidence.sh'
```

启动器依次执行：CPU回归 → 初始化B/C → 生成60步schedule → 三臂CPU preflight → A/B/C各4条GPU无损门禁 → A/B/C各60步 → 三臂step60完整dev40 → C机制探针 → 配对分析。每个子进程有独立日志，错误即停止，不自动重试。

每次 GPU 子进程前检查所选 UUID 至少40 GiB空闲且没有 compute进程；实际不够时应保存错误并诊断，不能杀别人的任务或绕过门禁。所有输出在 `EV_RUN`，最终 `SUITE_STATUS.json` 区分完成、停止与失败，训练状态以各臂 `RUN_STATUS.json` 为准。

可用 `EV_TRAIN_STOP='实际带时区ISO时间'` 设置训练边界停止；或创建 `$EV_RUN/STOP_TRAINING` 请求保存退出。它不是硬实时中断。入口仍不支持恢复 optimizer/cursor；不得把两次 fresh 30步写成连续60步。任一臂未完成，启动器保留已有结果并停止，其余未启动臂不能写作失败训练。后续仅比较实际存在的共同 checkpoint。

## 5 验收与机制探针

GPU门禁通过才训练。初始化残差为零，因此初始B/C应保持原head预测行为；CPU测试验证数学路径，实际GPU仍必须逐 token 对照独立AR。训练后再次用相同协议检查，任何失配立即保留现场并停止，不放宽精度或零容差。

训练日志 `evidence` 块记录每个模块的梯度、实际参数变化、请求诊断和残差大小。第1步 Q/K/V 零梯度可以正常；多步后仍无输出投影变化，或长期只有通用偏置变化时，先诊断梯度路径，不直接宣称视觉无用。

同一个C checkpoint支持：

| intervention | 改动范围 | 用途 |
|---|---|---|
| normal | 正确视觉 memory | 主条件 |
| bypass | 返回原 draft hidden，不加新增残差 | 检查新增支路贡献；仍做协议/缓存准备，不当移除模块的最优测速 |
| zero | 新支路内容与时间输入置零 | 保留模块参数，检查其对输入的依赖 |
| time_shuffle | 只打乱内容对应的时间标签 | 检查时间关联；不同时重排内容和时间 |
| wrong_video | 从固定映射的另一视频读取冻结 memory | 检查正确视频证据，target输入完全不变 |

错误视频 probe 使用正常C评测导出的冻结 memory bank，身份包含 target配置、输入、特征层、池化配置和源视频；只允许另一视频，拒绝错 target/容量及同输入。导出不运行额外target，序列化时间单列。bank属于机制诊断产物，不作为训练数据，也不把wrong-video运行的加载开销当正常部署速度。

这些probe只改变新增支路，不是旧的 `--zero-visual`，禁止混用。它们不删除原head主路中的视觉信息。单看错误memory导致下降不证明方法全面有效，还要结合C−A、C−B。

## 6 统计与论文结论边界

`summarize_evidence_eval.py` 对齐视频、实际 encoded input hash 和 AR token序列，拒绝不同输入、重复视频、失配输出及错误的A/B/C标签；B/C容量和特征来源必须一致，干预必须使用相同C checkpoint。checkpoint内原子保存 `CHECKPOINT_IDENTITY.json` 并纳入SHA清单，分析器据此检查共同起点、训练步数、数据顺序和优化配置，不从 `step60` 目录名猜预算；旧checkpoint无sidecar时明确标记未核验。以视频为bootstrap单位，输出：

- τ、有效 tokens/round、总输出 token / 总轮数；
- C−A、C−B、B−A及正常C与各干预的配对差和95% CI；
- SPEC encoded-input墙钟、含预处理墙钟；相同首条warmup视频只从计时排除，仍计入接受效率与无损；
- 输出长度、自然EOS和截断数量。

主比较固定step60，step20/40只看学习曲线，不挑各臂不同步数的最佳点。没有完成60时只能对比共同已保存步数并标注协议偏离。一次顺序A/B/C测速仍是诊断，bootstrap不能排除GPU负载、温度和运行顺序偏差；有收益后用相同卡、平衡顺序与重复测量复核。

若C只优于A、与B相当，支持额外读出能力，未证明视觉专用设计；C优于A/B且正确memory优于干预时，支持视觉读出机制。若τ提高但墙钟不降，先处理memory构建/读取开销，不能写整体加速。所有1024截断样本保留，结果只覆盖该输出前缀；另在固定小样本提高cap审阅自然EOS、重复、事件覆盖，不为了架构收益强制答案长度。

dev40仍是开发集，单seed不覆盖训练随机性。本轮最多180更新，不自动增加seed。论文中“架构与OPD协同优于其它训练法”仍需后续稳定SFT/OPD或同结构视觉收益监督对照；当前试验直接回答OPD条件下是否值得加入这一视觉结构。

## 7 给dsh的执行prompt

```text
请在这台有空闲GPU的服务器执行VisionSpec多模态架构验证。
仓库 https://github.com/mcy1123/VisionSpec
分支 feat/temporal-evidence-readout
拉取后记录commit，读AGENTS.md及
docs/reports/2026-11-12-temporal-evidence-architecture-plan.md。

本次按该文档完成代码检查、路径定位、CPU与GPU门禁、A/B/C训练、评测和报告。
授权预算为单seed三臂各连续60updates、共最多180真实optimizer更新；包括失败或
调试中发生的更新，不自动追加seed或通过重启清零。此明确任务覆盖旧AGENTS的
历史无剩余训练授权，只适用于本执行服务器。本轮不是前一晚的08:00截止任务；
若我另设新截止，记录绝对日期和时区，并提前留出保存/评测余量。

用隔离Git worktree，不覆盖旧代码/产物。核实已有真实Video-MME-derived
80train/40dev、128帧缓存、匹配Qwen3-VL-8B target及上轮128帧lambda10 step20。
三臂共享这一起始head的旧参数，均fresh Adam；不是旧训练状态resume。
不要换32B、视频输入、prompt或训练目标，不需要下载新数据。

固定A原结构、B同参数文本上下文readout、C时间分组视觉readout；三臂同schedule、
lr1e-5、lambda10、global_pooled、fp32 target与master、bf16原head、fp32新增模块。
实际特征来自最后保留层33，不是target末层。初始化检查必须证明B/C新增权重一致、
旧权重保留，不能打开旧FOVEA开关替代新模块。

先CPU测试和dry-run，再按脚本三个初始head各4条AR/SPEC逐token门禁，任何失配
停止并诊断。普通路径定位和必要实现修复自主处理，记录diff并重跑相关测试；
不能跳过门禁或更改精度来消除失配。

使用GPU UUID，确认没有其它compute任务，不能终止别人进程。数据、初始化头、
checkpoint、memory bank实际存于已有/mnt/local、/mnt/local2或/mnt/local3。
检查空间，保留历史checkpoint。执行计划写入本run，训练预计超过13小时加评测，
不要假设8小时能完成。长任务必须tmux运行，不依赖聊天在线。

按文档准备实际EV_*环境变量和新EV_RUN，执行
bash scripts/run_temporal_evidence_pilot.sh。
它包含三臂60步、20/40/60保存、dev40与C的bypass/zero/time_shuffle/wrong_video
机制评测及配对分析。训练状态以RUN_STATUS.json为准，STOPPED返回0也不能当完成。
失败不盲目重试；没有optimizer resume，不从step20 fresh加40冒充连续60。

检查多步后的readout梯度、实际参数更新及cache身份，区分未学会与架构无收益。
报告真实视觉token、有效tokens/round、C−A/C−B、逐token无损、EOS/截断、显存与
墙钟。机制探针保持target输入和输出固定，wrong-video bank只能用于诊断。
单次测速标诊断，不把tau当速度。若有新截止，提前停止，保留未配齐状态。

完成后写独立实验报告并登记docs/reports/README.md；将代码修复、报告和小型
逐视频指标/协议摘要push回mcy1123/VisionSpec的独立run分支，记录原始JSON路径。
不要把权重、帧cache或memory bank推入Git。最终说明支持哪条架构结论、哪些证据
不足，不能因为C>A就直接宣称架构与OPD的特有协同。
```

## 11. 本机八卡 H100 独立重建（2026-11-12，进行中）

本节是用户授权的另一次本机实验，不能与上述另一服务器的三臂 60 步方案或旧长生成结果合并。原始 run 为 `/czsun/mcy/video_opd_pilot/local_h100_20261112`；实时进度读取其中 `control/scheduler/STATE.json`。用户要求持续使用八卡执行实际任务、监控到明确解除；5 秒 telemetry 和 tmux 调度已启用。加载与 checkpoint 保存会出现低于 50% 的短暂低谷，尚不能保证每个采样时刻达标。

本机使用官方 SHA 核验的 Qwen3-VL-8B-Instruct 和公共 Qwen3-8B-DFlash-b16 重新适配，未复制另一台服务器的训练 head。Video-MME 来源已扩到 320 train、40 dev；另外 80 holdout 只冻结来源、未做生成。64/128 帧的 640 条 teacher 均自然 EOS、零截断，平均输出分别 214.52/214.73 token。稳定三段摘要任务用于排除重复尾部污染，不能被描述为千 token 长推理任务；早先自由 prompt 的高 τ 只保留作诊断。

当前主实验 `FACTORIAL80_PROTOCOL.json` 固定：

| 项目 | 本机执行 |
|---|---|
| 架构 | A_plain 原草稿；B_text 同容量文本位置读出；C_visual 视觉位置读出 |
| 初始化 | 共同本机 warm20；B/C 新增权重逐 tensor 一致、旧 58 个 tensor 保留 |
| 训练 | SFT/OPD × seed20261112/20261113，共 12 臂；每臂 80 连续更新 × 4 QA，320 源各一次 |
| 精度与优化器 | fp32 Target、bf16 原 Head、fp32 readout、GPU fp32 master/Adam，lr1e-5；fresh optimizer |
| 新增路径 | dim256、memory128、spatial_bins2；显式 time weight 置零并冻结 |
| OPD | 原 Bernoulli forward-KL + rejected k3，global_pooled，λ0；尚未加入视觉条件加权 |
| 评测 | step20/40/80，各完整 dev40，fp32、b63k8、独立 AR 逐 token 无损、自然 EOS |
| 主要比较 | 每 seed 的 C−B，以及 (C_OPD−B_OPD)−(C_SFT−B_SFT)；不把两 seed 当 n=80 |

A_plain 本来就可从融合 Target 特征读到视觉位置；B_text 的文本位置也已受 Target 多模态上下文影响，不能称为“完全无视觉”对照。新增容量的匹配、精度、独立输出门禁、真实输入 SHA 和完整开发集身份由单独审计检查。并发墙钟只能作描述性数据，接受率增益不能直接写成端到端加速。

本机修复了 visual OPD 首个 scoring anchor 的边界：`prompt_len-1` 已包含全部视觉位置，缺少的只是最后一个已知文本 root。新包装器仅接受视觉位置全部已知的这个特例，保持完整请求身份，不补未来特征，不更改 anchor/loss。旧失败 run 保留；新入口两次完整 Adam-state 峰值探针及 dev2 严格无损通过后才进入主实验。

工程上另做固定空间分辨率的嵌套 256/512 帧输入：真实视觉 token 为 10,752/21,504，最长 train 探针实际输入为 12,064/24,071。公共草稿和 warm20 的四次单视频 AR/SPEC 门禁均通过，512 帧峰值 allocated/reserved 约 62.30/63.25 GiB。这只说明本机单卡能运行该长输入协议；默认 processor 会随帧数降低空间分辨率，因此固定像素预算写入独立 `PROCESSOR_PROFILE.json` 并核对实际 grid 和编码 SHA。

排程审计发现 `mature_warm80_*` 与正式同 seed A_plain SFT 的配置完全相同，仅输出目录及 GPU 不同。它们是误排的重复运行，不能计作额外强基线或独立证据；原产物完整保留，正式 factorial 汇总仅使用正式 ID。两者并非逐位相同，数值差异另行审计，不声明训练逐位可复现。更正记录见 `DECISIONS.jsonl` 和 `analysis/mature_factorial_duplicate_config_audit.json`。

独立的 packed OPD scoring 候选仅完成 CPU 证明及回归，没有接入当前 12 臂。bf16 梯度分组会改变数值，不能声称更新等价；启用前还需要单独的 CUDA 数值与显存检查。主结果仍等待完整配对评测，不预判视觉读出或 OPD 协同有效。

### 11.1 扩大训练前的决策窗口（11:32 UTC 截面）

用户于 10:59 UTC 要求尽量两小时内给出扩大训练的准确结论；目标时间为 12:59 UTC，持续监控仍未解除。执行记录为 `DECISION_WINDOW_20261112.json`，不是所有后续评测都必须在该时间前完成的保证。

截至本截面，正式 A/B 两种 SFT 的 step80、两 seed 共 160 次生成全部自然 EOS、逐 token 无损。每视频有效 token/轮分别为 A=2.924449/2.922614，B=2.980026/2.966488；B−A 为 +0.055577、95% CI [0.039416,0.072776]，以及 +0.043874、CI [0.025077,0.062925]。这支持额外读出能力，尚不支持视觉专用收益或 OPD 协同。C 及 OPD 的完整配对结果仍待完成，权威汇总为 `analysis/factorial80_summary.json`；只纳入完整 dev40，不用先完成的行替代全集。

同一 40 个 dev 视频的 256/512 帧数据已完成 80 次实际 CPU 编码，原始像素、真实源帧索引和 PTS 严格嵌套。实际输入分别 12,034–12,065、24,010–24,072 token，视觉 token 为 10,752、21,504。`LONG_INPUT_DEV40_PROTOCOL.json` 固定十二个 step80 头和共同 warm20，逐头最长输入门禁后做完整 dev40；预算 13 个单源门禁、26 个完整条件，共 1,053 对 AR/SPEC，零训练更新。优先 512 帧 C/B，256 帧为次要条件。`summarize_long_dev.py` 同时绑定固定最长源、自身门禁、原始与长输入来源 SHA、预先实际编码的 tensor SHA、Target/decoder/Head 身份和完整配对 AR 输出。不同帧数之间不假设相同 teacher 输出。

恢复审计发现，原 SFT pilot 只有推理权重，原 OPD 虽有 master/Adam 但缺完整 RNG；两者都不能冒充完整断点续训。新隔离入口 `train_video_sft_long.py`、`train_video_opd_long.py` 保留原训练数学，保存完整 Head/master/Adam/Python/NumPy/torch/CUDA RNG、绝对日程游标及严格模型/数据/源码身份，每段写新目录。40 项 CPU 数学与恢复检查通过，真实数据预检不初始化 CUDA。GPU 验证由 `LONG_RESUME_GPU_GATE_PROTOCOL.json` 固定：SFT C 和 OPD C 各从正式 step80 权重新开诊断，连续两步保存 step1/2，再新进程从 step1 恢复只做第二步；共六次实际更新、24 QA，另八对部署 AR/SPEC。恢复后、下一更新前必须零容差回读全部状态；第二步的跨进程数值一致性单独量化，不事后放宽容差，不从 CPU 成功推断 CUDA 逐位等价。此截面 GPU 门禁已入队，尚未完成。

数据扩容也有独立边界：现有 900 源按原 45–600 秒资格并排除 dev40/holdout80 后，最多只有 347 个整段训练源，当前 320 仅能再加 27。另有 399 个未用长源，可按四个等分区间各取一个 151–300 秒真实窗口，理论合计 1,943 个训练样本、746 个独立源。`build_window_probe.py` 已在固定四源的 16 个窗口上完成 32 次训练/评测编码一致性检查，保持源时间轴，每窗 128 独立帧，同源四窗不交；缓存约 0.259 GiB。该小样未使用 dev/holdout，16 窗仍只有四个独立源；全量窗口尚未构建，teacher 输出检查单独入队，不把理论规模写成已就绪。证据见 `analysis/scale_training_feasibility.json`、`analysis/scale_window_expansion_feasibility.json`、`analysis/MULTIWINDOW_PROBE_DELIVERY.json`。

### 11.2 一次性结果检查（12:43–12:46 UTC）

用户要求停止持续监控后再次发起一次检查。未恢复遥测，也未启动大规模训练。十二臂各 80 次更新及其最终 dev40 已全部完成：480 次 SPEC 均逐 token 等于独立 AR，全部自然 EOS。以下为每视频有效 token/轮的均值，两个 seed 使用同一批 40 个开发源，不能视作 80 个独立源。

| 架构 | SFT seed12 | SFT seed13 | OPD seed12 | OPD seed13 |
|---|---:|---:|---:|---:|
| A_plain | 2.924449 | 2.922614 | 2.878231 | 2.882840 |
| B_text | 2.980026 | 2.966488 | 2.920861 | 2.927028 |
| C_visual | 2.975184 | 2.960712 | 2.929128 | 2.932589 |

相对同容量 B，视觉读出在 SFT 的差为 −0.004843/−0.005775，在 OPD 的差为 +0.008268/+0.005562；OPD 第一个 seed 的 C−B 区间跨零，第二个为 [0.000340,0.011389]。交互 `(C_OPD−B_OPD)−(C_SFT−B_SFT)` 为 +0.013110/+0.011337，两个探索性区间都在零以上，但绝对量很小，且 C_OPD 仍低于 C_SFT：差 −0.046055、CI [−0.080969,−0.010081]；以及 −0.028123、CI [−0.059281,+0.003034]。因此现阶段支持“额外读出容量有用，有很弱的视觉/目标交互信号”，不足以支持直接扩大当前视觉 OPD 配方。原假设尚待长输入完整配对，不能把这些接受效率差写成墙钟加速。权威证据为 `analysis/factorial80_summary.json` 和 `analysis/factorial80_objective_contrasts.json`，开发集复用及未做多重比较校正的边界不变。

恢复诊断的六次实际更新已完成，SFT 和 OPD 的 `RESTORE_AUDIT.json` 均为 `EXACT_RESTORE_VERIFIED`，在下一次更新前确认完整 Head/master/Adam/RNG 零容差载入；四个 step2 checkpoint 的 dev2 合计八次部署检查全部无损。连续/恢复第二步的细粒度比较另行生成，完整加载成功本身不等于跨进程更新逐位一致。窗口小样的 16 条 teacher 也全部自然 EOS、零截断，平均 245.125 token；仍只有四个独立视频源。

本次随后完成两份 CPU 细粒度比较：`analysis/sft_long_resume_comparison.json`、`analysis/opd_long_resume_comparison.json` 均为 `CONTINUATION_DIFFERENT`，完整载入硬门禁通过、第二步后的 RNG 仍完全相同，但参数更新未逐位复现。SFT/OPD 的 Head 不同元素分别为 1,376,392/5,813,848，最大绝对差均 3.051758e-5；master 最大绝对差分别 1.444668e-5/1.691468e-5。未追加容差把差异改判为 exact，具体差异来源尚未隔离。因此可报告完整状态可恢复及恢复后部署无损，不能报告 CUDA 续训逐位等价或仅凭此门禁保证长训稳定。

本次发现长输入 OPD 队列有一处 prerequisite 路径错误：该 pilot 把训练协议存于 `run/PROTOCOL.json`，排程误等待不存在的 `step80/TRAINING_PROTOCOL.json`，造成物理 0/6/7 空闲。已逐头核实 checkpoint identity 的协议 SHA 与实际文件一致，只替换 18 个尚未派发任务的只读输入路径；未改已启动定义、评测数学、数据或训练预算。三条相应最长输入门禁已开始，失败记录中的一项首 anchor 历史错误及其两条 BLOCKED 保持原样，未出现新的 GPU 失败。预计关键长输入 OPD 完整对照仍需约 60–90 分钟，不从已完成的 SFT 小子集推断 OPD 的长输入收益。

### 11.3 与历史视频正结果的可比性核查

用户指出历史记录中视频 OPD 有明显收益，复核确认这些正结果确实已记录。上一节的“不直接扩大”仅适用于本轮初始化、三段摘要任务、λ0 和 80 次更新，不应解释为否定旧视频 OPD 路线，更不能称为对旧最佳 λ10 配方的复现失败。核查记录为 `analysis/historical_video_comparability.json`，历史数值来自已归档报告；旧服务器 eval JSON 当前不在本机，尚未重新计算其统计区间。

最直接的正证据是[真实视频夜间报告](2026-11-12-real-video-opd-overnight.md)：相同旧起点、相同真实 128 帧输入和 20 次更新，原 Head τ=6.578，λ0=7.070，λ10=7.171；λ10−λ0=+0.1012、CI [+0.0402,+0.1699]。两种目标都已含 Bernoulli/k3，λ10 额外启用的是历史 top8-overlap prefix proxy，并非本轮新增视觉读出，也不是显式视觉优势加权。该报告没有同协议 SFT 对照，不能由它推出 OPD 一定优于当前 coverage-CE SFT。

| 条件 | 旧真实视频夜间实验 | 本机当前架构筛查 |
|---|---|---|
| 起点 | 已经视频适配及 OPD 多轮训练的 OPD3e Head | 公共文本预训练草稿，先做本机视频 SFT20 |
| 损失比较 | 原结构 λ0 与 λ10 配对 | A/B/C × SFT/OPD；全部 OPD λ0 |
| 任务 | 按时间列事件及时间范围，dev20 全部 1024 cap、无 EOS | 开始/中间/结尾三段摘要，dev40 平均 233.025 token、全部自然 EOS |
| 数据与预算 | 80 train / 40 dev，主评固定 dev20、单 seed、配对 step20 | 320 train / 40 dev、两 seed、主评完整 dev40、配对 step80 |
| 直接回答 | 视频分布适配、prefix 辅助目标是否提高接受效率 | 新视觉读出是否超过同容量文本读出，及本轮 OPD 是否超过本轮 SFT |

本轮并未启用旧正结果的 λ10，且初始化和任务同时变化，因此不能单独归因“视频 OPD 失效”或“λ10 一定能救本轮”。本机原自由 prompt 的固定四源、64/128 帧共八条件已复核：全部 1024 cap，重复 8-gram 窗口比例平均 51.72%；同源三段摘要八条件全部自然 EOS、207–293 token、重复 8-gram 为零。这支持将本机自然生成任务与重复尾部区分，但不能据此断定另一服务器旧 dev20 的全部增益都由重复造成。截断前缀上的接受效率正结果也不能自动外推为完整长报告质量或端到端加速。

另外，[旧 VideoDetailCaption 结果](2026-10-26-video-caption-opd.md)为 warm 3.7266→OPD3e 3.7841（+0.0575），且同 lr/步数的 SFT 退化，OPD−SFT=+0.3844；这与本轮 SFT 更强的结果确实方向不同，但数据、起点、SFT 训练实现和预算并未保持一致。[11月11日整理](2026-11-11-consolidated-video-token-multimodal.md)中 3.8796/3.8740 及 3.9046 的高 token 正结果，输入只有 16 个独立帧复制至 128 帧，亦不能等同本机真实 128 帧长视频。

下一项更有判别力的比较应在本机固定当前起点、视频与 prompt、训练日程和部署协议，给原结构及匹配 B/C 分别补 λ10 与已有 λ0 的同预算对照；先区分辅助目标收益和新增视觉结构收益，再决定是否扩大数据及训练。当前没有由本次历史核查启动任何新训练，也未恢复持续监控。

### 11.4 用户授权的 λ10 同条件补充（13:05 UTC 已入队）

用户随后明确同意补齐 λ10。分支核对：本机仍在 `run/local-h100-fresh-20261112`，fetch 后历史视觉 OPD、真实视频夜间和视觉读出三个远端分支都已是当前 HEAD 的祖先；历史 `runtime.score` 的 λ10 数值计算也已存在，相关 diff 仅含目标名称/边界的注释澄清。此前 λ0-only 架构筛查不等于复现旧最佳辅助目标，本次补齐这一缺项，不切换运行中工作树。

`LAMBDA10_PROTOCOL.json` 固定 A_plain/B_text/C_visual × seed20261112/20261113 六臂，各80连续更新、4 QA/update。逐臂直接复制原 λ0 CLI，只替换输出目录和 `--e2e-tv-weight 10`；初始权重、320源 train、两份完整 shuffle schedule、三段摘要 prompt、原 trainer/scorer、lr1e-5、fp32 Target/master 与 bf16 原 Head、fp32 readout 及同一物理 UUID 都保持一致。每臂 fresh Adam，不能从 λ0 step80 或门禁训练结果接着跑。六个真实数据 CPU dry-run 已通过全协议比较：去除各自协议摘要后，唯一差异为 `config.e2e_tv_weight=10`。没有分配 GPU 或生成模型输出来完成预检。

正式预算480次更新；三架构最长输入重复两个 update 的 full-Adam 门禁另计6次，共486次真实更新、1944 QA。门禁各配dev2，正式各配完整dev40，另每头512帧最长源门禁及256/512帧完整dev40，共732对部署 AR/SPEC；零容差无损，失败停止依赖链，不自动重试。保存20/40/60/80；为尽快回答本问题，主统计固定step80，不追加中间点评测。约27份新 fullstate 的空间已按现有真实 checkpoint 大小及额外150GiB余量核对，未删除历史产物。

队列新增37个有限任务，先三架构peak2和独立AR门禁，再六臂正式训练和128帧dev40，随后长输入；尚未完成的旧 C/B λ0 长输入主要对照保留优先级，已启动任务定义不变。`repro/local_h100/prepare_lambda10.py` 负责原子采纳、CPU预检及预算约束，`summarize_lambda10.py` 对完整40源、实际输入SHA、独立AR、checkpoint identity及新旧训练协议进行严格核验，并在六个正式dev40全部完成后生成 `analysis/lambda10_summary.json`。同时给出每seed λ10−λ0、λ10−SFT、λ10内C−B，以及 `(C10−B10)−(C0−B0)`；不把两seed当80源，不把开发集复用和并发墙钟写成独立确认或正式速度证据。

CPU验证覆盖真实 A/B/C forward 的 full-vocabulary softmax→截取top8→overlap→prefix-product 公式、loss/梯度、原GPU-memory/boundary入口组合，以及拒绝额外更改起点/日程/源码/预算的统计契约。该补充保留原 global_pooled 的基础分母，辅助项只加到 numerator，不能用 Pool 的辅助增广 denominator 偷换归一化。本机任务仍是自然三段摘要；此次对照若为正，只证明旧辅助目标可迁移到这个新任务，不能宣称旧服务器 checkpoint 或千token完整报告已经复现。持续监控维持关闭。

### 11.5 用户授权的 checkpoint 清理与保存策略

用户要求清理过多保存点、降低保存频率。首轮按队列依赖、任务状态与正式终点检查，退休75个中间/诊断 checkpoint，删除746,586,614,034字节（约695.31 GiB），磁盘空闲恢复约2.12 TiB。保留的12份原 factorial 正式 step80 权重逐一检查存在，未发现待运行/运行中引用被退休。训练并发仍会生成新点，不能用首轮数量当作后续实时数量。

清理工具 `repro/local_h100/checkpoint_retention.py` 只删除已发布、已确认无用途保存点中的两种大二进制；检查队列版本、文件身份和路径，删除前归档原JSON并记录清单。所有原日志、评测JSON、曲线、协议保留；已退休点的原SHA仅为历史身份记录，不能继续推理或恢复。保留全部正式step80、实际排队引用、公共初始化、λ10每臂最新发布点及两份指定完整恢复step1。六项CPU测试覆盖依赖保护、最新点、dry-run与幂等、队列/文件变动拒绝和未知/未完成/软链接保护，均通过。

未来训练默认只保存最终或分段终点，显式设置 `--save-every` 等于完整更新预算，需要额外中间评测或恢复时才另定。已运行六臂的save_every20已在进程内冻结；不修改运行中源码/CLI、不重启配对轨迹，由有限tmux自动回收后续不再引用的中间点。该文件回收任务在六臂和原中间点评测结束时退出，最长12小时，不恢复GPU遥测，也不追加训练预算。权威审计为 run 下 `analysis/checkpoint_cleanup_20261112/{deleted.jsonl,STATUS.json,metadata/}`，策略为 `control/CHECKPOINT_POLICY.json`。

### 11.6 旧时间线任务的首夜架构筛选（用户授权已入队）

新增54个有限任务，代码 `prepare_overnight_timeline.py`；复用旧事件时间线prompt和本机真实80/40数据，主128帧、迁移64帧、max_new1024、b63/k8、fp32 Target、独立AR逐token无损。旧成熟checkpoint不可用，本机split也不同，故是任务协议对齐后的重新适配。公共DFlash经SFT80得到共同起点，再进行A原结构/B文本读出/C视觉spatial_bins2/D视觉spatial_bins1 × OPD λ0/10，八臂各80步，同seed20261113、同shuffle日程、fresh Adam。所有证据读出time_encoding off；D只比较池化几何，不声称时间编码创新。总预算738更新、2952 QA，包含18次必要门禁更新；门禁权重不作正式初始化。

每臂仅保存最终点，8个OPD终点约126GiB，加warm约2GiB和匹配初始化约8GiB；门禁产物在自身独立AR评测完成后按原清理审计退休。训练失败不重试，依赖分支停止，其余独立分支继续。八个teacher shard、共同warm、必要门禁、初始化、正式训练、完整dev40的128/64帧评测均已排进现有tmux scheduler。`report_overnight_timeline.py` 独立有限tmux每分钟汇总tau与接受长度直方图、同源配对bootstrap和无损率，并清理已用完gate；最长14小时后退出，不操作GPU或训练进程。产物根为 run 下 `overnight_timeline_20261113/`，含冻结协议、队列快照、STATUS和完成后的DONE。

本轮只做单seed架构筛选与已有λ目标比较，不使用holdout，不声称新视觉依赖加权损失或第二seed已上线。所有训练预算有限，不保证每时每刻八卡满载；公共起点的依赖阶段可能有空卡，不用空算填充。

### 11.7 首夜最终检查（2026-11-12 23:42 UTC / 北京时间11月13日07:42）

54任务为46 SUCCEEDED、8 FAILED；所有训练已完成，失败仅为8个64帧迁移评测的逐token无损检查，非队列路径故障。128帧完整dev40：warm τ7.038422；A λ0/10=7.437747/7.525720；B=7.491915/7.560551；C=7.494529/7.557128；D=7.485142/7.558677。warm及八臂共360/360逐token无损。各架构λ10−λ0为+0.087973/+0.068636/+0.062599/+0.073535，开发集配对95%区间均正；λ10下C−B=−0.003423，CI[−0.018888,+0.012129]，D−B=−0.001875，CI[−0.016440,+0.012396]。因此接受长度已回到7以上，通用辅助目标正向，但视觉读出独立收益未获支持。全部128输出到1024上限、无自然EOS；单seed、复用dev，不声称完整长报告质量或独立泛化结论。

64帧八臂均实际完成40条生成，但同一个视频vgWlRdNOGx4在索引65处SPEC/AR不一致，均39/40无损、EXIT=2，故不能作为无损加速结果。原因尚未定位，未放宽门禁、未自动重试。权威结果为 `overnight_timeline_20261113/DONE.json` 与各job原始result/stdout。

占卡程序曾于队列结束后启动，但22:36 UTC因读取scheduler/STATE.json时短暂FileNotFoundError退出。23:42检查时八卡空闲；已保留原崩溃日志，补充最多5次状态读取重试后安全退出的处理，并按既有授权重新启动tmux等待器。此修复不重试科研任务、不追加训练。持续异常仍释放占卡负载，未来入队仍须先显式STOP并确认子进程退出。

### 11.8 用户再次授权精简最终 checkpoint

北京时间11月13日07:44，确认队列全终态后，删除factorial及night13的20份最终training_state.pt，释放274.332GiB；全部30份jobs内模型权重保留，两份step1恢复诊断完整状态保留。所有原结果/协议/identity/SHA不改，新增COMPACTION.json说明优化器状态已经移除，原SHA为历史身份而非完整文件承诺。权重可用于评测或fresh optimizer初始化，不能恢复已删除的Adam状态。逐文件归档元数据并核对文件身份，审计见run的 `analysis/checkpoint_compaction_20261112T234445Z/`；完成后磁盘空闲约2.305TiB。
