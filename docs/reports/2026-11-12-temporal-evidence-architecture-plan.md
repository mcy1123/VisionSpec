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
