# 真实视频长生成：代码诊断、修复与跨服务器执行方案

状态：**CPU 准备与回归验证；没有在本机运行 GPU 推理或训练。**

代码基线：`origin/exp/visual-opd@06ca7a7`。交付分支：`fix/visual-opd-long-video-protocol`，远端为 `mcy1123/VisionSpec`。本次修改在独立 Git worktree 中完成，原 `main` 工作区及历史结果未改。

本轮用户要求是诊断、修改代码、写方案并推送，实验在另一台服务器执行。本文件区分已经实现的基础修复与尚待实现的新方法，不把 CPU 测试当成 GPU 无损或加速结果。

## 1. 研究问题与已有证据边界

目标：**长视觉上下文中，随生成前缀变化的证据读取，能否与 OPD 配合，改善长文本生成的实际草稿接受效率及端到端速度？**

先用多事件、时间顺序明确的视频报告。输出每一段应需要新画面证据，避免仅要求“至少一千字”。源视频分钟数、真实采样帧数、视觉 token 数、输出 token 数分别记录。

现有证据不足以直接宣称真实长视频加速：

- 旧 128f1x 是 16 个真实帧各复制 8 次；其 τ≈3.88 是高 token 构造下的结果。
- fresh30 已反复用于方法选择，只能继续作为开发回归。
- 旧视频输出平均 270.9 token，但 OPD 对 warm 的端到端增益仅约 1%，CI 跨 1；增加长度不是解决方案本身。
- real/blank 更换 target 输入，会改变 AR 轨迹；两者 τ 差不是某个 draft 模块的纯因果效果。
- 旧“76% 视觉信息在文本 hidden”不能由两个 τ 差的比值推出，且旧置零探针存在位置错位问题。本文件不沿用该解释。

已有 [ParallelVLM](https://arxiv.org/html/2603.19610) 覆盖视频长段落生成，[Sparrow](https://arxiv.org/html/2602.15318) 已训练草稿并评测长视觉输入，[Uni-LaSpec](https://arxiv.org/html/2611.01617) 已结合视觉选择和接受感知训练。因此贡献应落在“证据如何改善当前草稿、如何训练这种选择”及可复核的效率结果，而不是首次训练式视频投机解码。

## 2. 已修复与尚未实现

| 项目 | 诊断与本次处理 |
|---|---|
| 视频时间轴 | trainer/eval 增加显式 `--time-mode real`；使用真实源帧编号及 PTS，拒绝重复采样帧和不满足当前时间表示的输入。旧 `legacy` 默认保留以区分历史实验。 |
| 真实帧数据 | 新增 `build_real_video_pilot.py`，源视频级去重/排除/划分，64/128 嵌套采样、无损 NPZ 缓存、JSONL/可选 parquet 与来源哈希。 |
| 视觉置零探针 | 使用 head 的绝对 stream position，并与 cache length 交叉检查；生成文本位置不会再次套用 prompt 前部的 visual mask。 |
| 跨层 xattn | 旧训练用完整前缀、部署后续轮只用增量 hidden；现将跨层上下文放在每个请求的 draft cache 上，追加已提交 hidden、遵守 crop，不混入草稿 noise 行。 |
| 损失归一化 | 显式 `legacy_per_qa` 与 `global_pooled`。新协议为 `sum(N_i)/sum(D_i)`；legacy 为 `sum(N_i/D_i)`，不是 QA 均值。日志记录实际优化 loss 及 pooled 描述值。 |
| 高输入 trace 内存 | global 模式每条只 rollout 一次，CPU 暂存无梯度 trace，随后按顺序搬回并 backward；未声明与旧收集顺序逐位等价。 |
| 所谓 E2E-TV | 保持旧计算不变，明确命名为 `top8_overlap_prefix_proxy`：完整词表归一化后取 draft top-8 重叠，再作前缀乘积；不是 full-vocab TV 或真实树接受期望。aux 仍加到 numerator，分母仍用基础 OPD response/rejected 计数。 |
| 身份与重启 | CLI 支持显式 target 路径；CPU config 检查拒绝现有 8B head 配 32B target。trainer 记录输入、初始化权重及协议哈希，仍是 `diagnostic_only`、weights-only fresh optimizer，**不支持训练状态 resume**。 |
| 评测 | 保留旧 τ，另报实际 `gen_tokens/rounds`；输出输入哈希、真实帧/token 数、输出文本、EOS/截断、绝对置零记录；AR 不一致保存诊断并非零退出，不覆盖既有结果。 |

**没有实现的新方法部分：** 与输入身份绑定的真实视频 visual memory、新的分段选择器、固定前缀的多证据候选 oracle、接受收益排名监督、B/C 联合训练入口。当前 `train_video_9k.py` 会拒绝依赖外部 visual memory 的 head，以免 `head.visual_memory=None` 静默关闭模块。real eval 也拒绝旧的 `video_name -> tensor` memory。

原 decoder/tree/verifier 没有修改。新增 xattn 缓存修复只作用于已启用 `use_xattn` 的实验 head，会改变该实验分支的旧行为；后续结果必须以新代码身份单独记录。它仍缓存完整跨层上下文，不能声称节省长上下文显存。现有 FOVEA 每轮重投影 memory 的成本也尚未优化。

## 3. 第一阶段：可直接执行的零更新诊断

### 数据与模型

- 先用现有 Video-MME 原视频，优先 medium 中真实时长 240–600 秒的视频。按源视频固定 **80 train / 40 dev**；全部只作开发，本阶段不设最终 test。
- 同一视频的所有 QA/变体必须在同一 split。输入是新任务，名称为 **Video-MME-derived long-form generation**，不是官方选择题评测。
- 64/128 都从真实源视频采样，64 为 128 的子集，长边默认 384、不上采样。processor 后实际 token 数才是测量值。
- 默认无字幕/音频，prompt 已在 builder 中固定。`max_new=1024` 是上限，无 min token 强制。短输出与截断样本保留，不能按新方法收益筛视频。
- 从与现有 head 确实匹配的 Qwen3-VL-8B 开始。32B 放在后续扩展，当前接口还绑定 8B 的层选择，不是改路径即可迁移。
- 当前机器只有 `/czsun` 数据挂载，旧 `/mnt/local*` 不存在且 GPU 已满；**不要在本机物化数据或运行 GPU**。目标服务器先检查真实挂载。缓存/权重/checkpoint 按 AGENTS 规则放 `/mnt/local`、`/mnt/local2`、`/mnt/local3` 的真实路径。

### 目标服务器准备与 CPU 检查

在目标机发现实际路径后填写以下变量；不要把占位值直接执行。若原仓库不是 Git 树，先保存可恢复快照并另建干净 checkout，不覆盖历史代码。

```bash
git fetch origin fix/visual-opd-long-video-protocol
git worktree add -b run/real-video-diagnostics ../VisionSpec-real-video origin/fix/visual-opd-long-video-protocol
cd ../VisionSpec-real-video
git rev-parse HEAD

VS_PY=/实际环境/bin/python
VS_DATA_ROOT=/mnt/local3/mcy/GLANCE
VS_TARGET=/实际路径/Qwen3-VL-8B-Instruct
VS_HEAD=/实际路径/已核实身份的OPD头
VS_SRC=/实际路径/videomme_900videos_2700qa.jsonl
VS_MEDIA=/实际路径/存放原视频的目录
VS_USED=/实际路径/历史使用视频ID清单.txt
VS_RUN="$VS_DATA_ROOT/results/real_video_diagnostic_新时间戳"

realpath "$VS_DATA_ROOT"
df -h "$VS_DATA_ROOT"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv

CUDA_VISIBLE_DEVICES='' "$VS_PY" -m pytest -q \
  repro/tests/test_real_video_pilot.py \
  repro/tests/test_pilot_schedule.py \
  repro/tests/test_video_9k_protocol.py \
  repro/tests/test_video_visual_probe.py \
  repro/tests/test_video_eval_merge.py \
  repro/tests/test_visual_context_cache.py
```

环境需要 PyTorch、transformers、pandas/pyarrow、PyAV、Pillow、ffmpeg/ffprobe、pytest；多卡加载需要 accelerate。保存实际包版本。训练还依赖 `torch.nn.utils.get_total_norm`；本机 PyTorch 2.5.1 没有这个 API，因此本地测试不包含真实 optimizer/GPU 更新。目标机优先使用此前有效环境，不能把 CPU 通过解释为依赖全部满足。

`VS_USED` 必须由历史视频清单生成；若确认无历史重叠才可显式空文件，并记录审计依据。当前机器可参考 `/czsun/mcy/video_opd_pilot/Video-MME/` 的旧 pilot 文件，目标机路径以实际为准。`--source-root` 以原文件 basename 重定位，先检查实际文件布局是否匹配。

```bash
"$VS_PY" repro/video_caption/build_real_video_pilot.py \
  --source-jsonl "$VS_SRC" --source-root "$VS_MEDIA" \
  --exclude-video-list "$VS_USED" \
  --out-dir "$VS_RUN/data" --cache-root "$VS_RUN/frame_cache" \
  --n-train 80 --n-dev 40 --frame-counts 64 128 \
  --min-duration 240 --max-duration 600 --long-side 384 \
  --seed 20261111 --write-parquet --dry-run
```

dry-run 只查容器元数据和划分、不写缓存，也未检查全部解码 PTS。数量不足时先报告可用清单，不复制视频凑数。确认路径后去掉 `--dry-run` 执行物化；只执行一次，工具拒绝覆盖非空目录。默认精确时间模式会拒绝不能由 `frame_id/source_fps` 表达的 VFR 输入。不要用 `--allow-approximate-timeline` 冒充正式真实时间实验；需要支持这类视频时先扩展时间接口或明确另立协议。

物化产物为 `train_64.parquet/dev_64.parquet/train_128.parquet/dev_128.parquet` 及对应 JSONL、完整 manifest、PROTOCOL。绝对路径属于生成数据的服务器，移机应重新物化/验证，不直接复用失效路径。

```bash
"$VS_PY" repro/video_caption/build_pilot_schedule.py \
  --rows "$VS_RUN/data/train_128.jsonl" \
  --epochs 3 --qa-per-update 4 --seed 20261111 \
  --out "$VS_RUN/schedule_128.jsonl"

CUDA_VISIBLE_DEVICES='' "$VS_PY" repro/video_caption/train_video_9k.py \
  --pool "$VS_RUN/data/train_128.parquet" --schedule "$VS_RUN/schedule_128.jsonl" \
  --target "$VS_TARGET" --init-from "$VS_HEAD" \
  --out "$VS_RUN/base_A_RESERVED" --updates 60 --qa-per-update 4 \
  --time-mode real --loss-normalization global_pooled \
  --device-map balanced --master-weights cpu \
  --max-new-tokens 1024 --max-seq-len 32768 \
  --e2e-tv-weight 10 --lr 1e-5 --save-every 20 --seed 20261111 --dry-run

CUDA_VISIBLE_DEVICES='' "$VS_PY" repro/evaluation/eval_video_opd.py \
  --rows "$VS_RUN/data/dev_128.parquet" --head "$VS_HEAD" --target "$VS_TARGET" \
  --time-mode real --require-exact-time --eos-stop immediate \
  --dtype float32 --ar --preflight-only --out "$VS_RUN/preflight_128.json"
```

80 视频 × 3 epoch / 4 QA = **60 updates**。此处仅冻结 schedule 和 dry-run，**没有执行这 60 次更新**。该预算只是下一阶段基线候选配置，不是把去掉 `--dry-run` 作为第一阶段命令。

### GPU 门禁与开发集诊断

只绑定目标机已确认可用的 GPU UUID，`CUDA_DEVICE_ORDER=PCI_BUS_ID`；不沿用旧的物理数字卡号，不终止其他进程。先用 4 条 dev 做小门禁，确认实际输入长度/显存及 4/4 无损，再进行完整 40 条。以下 `CUDA_VISIBLE_DEVICES` 只是形式示例，要替换为实际可用 UUID。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-实际UUID一,GPU-实际UUID二,GPU-实际UUID三

"$VS_PY" repro/evaluation/eval_video_opd.py \
  --rows "$VS_RUN/data/dev_128.parquet" --head "$VS_HEAD" --target "$VS_TARGET" \
  --time-mode real --require-exact-time --eos-stop immediate \
  --dtype float32 --device-map balanced --gpu 0 \
  --n 4 --warmup 0 --max-new 1024 --budget 63 --branch 8 --ar \
  --out "$VS_RUN/gate_128.json"
```

4 条通过后，完整开发诊断仅运行下列 **3 个条件**，各 40 条，均配对 AR：

1. `dev_64.parquet`、原 head。
2. `dev_128.parquet`、原 head。
3. `dev_128.parquet`、同一 head，加 `--zero-visual`。

沿用上述命令，去掉 `--n 4`，各用独立输出名 `baseline_64.json`、`baseline_128.json`、`zero_visual_128.json`。warmup 策略需一致，第一轮可以保留 `--warmup 0` 并将计时标成诊断，不把它当最终速度结论。正式速度再使用独立 warmup、交替顺序与至少两轮配对测量。

所有长进程写进脚本并由 tmux 启动，例如 `tmux new-session -d -s real-video-diag 'bash /实际路径/run_diagnostics.sh'`。脚本自行保存 stdout/stderr、退出码与 DONE/FAILED，不依赖聊天持续在线；失败不自动重试或更改协议。

### 第一阶段应交付的结果

- 实际视频时长、unique 帧数、processor 后视觉 token、总输入长度的分布。
- 输出长度、EOS/截断比例；人工审阅 20 条 AR 文本，检查时间顺序、是否泛泛而谈及是否确实需要多个时间段。
- 逐 token 无损情况，原 τ 与有效 `gen_tokens/rounds` 分开列。
- 128f 原 head 与 zero-visual 的配对差，先核对两组输入哈希、AR ids 完全一致；其余条件不等时拒绝做因果解释。
- 按输出前/中/后段列接受情况。事件切换分析需要人工/标注对齐，不能把所有三等分边界称为事件切换。
- processor/视频预处理耗时单列；现有 `spec_prefill_ms` 实际包括首轮 draft 前的启动窗口，不能称纯 target prefill。主 wall 字段是 encoded input 到输出。多卡 CPU/disk offload 单独标注，不与纯 GPU 时延混算。
- 64/128 改变了 target 输入，因此其 τ 差不是 draft selector 消融。旧 blank/mismatch 结果也不混入新配对结论。

报告写新文件并登记本索引；不得覆盖历史 JSON。第一阶段 **0 次 optimizer update**，不扩大数据集，不自动启动 B/C。

## 4. 第二阶段：待实现的视觉结构与 OPD 协同

第一阶段确认任务可用后，先完成以下代码，再训练；当前分支没有对应“已可执行”的 CLI。

1. **独立 EvidenceReadout**：保留已有 fused-context 主路，复用 FOVEA 式输出残差；视觉 memory 来自完全相同 target 输入，保存 input hash、特征来源/精度、时间组归属。第一版每轮选 2–4 个时间段、总预算 64 token，保留 bypass。不要同时换多种融合结构。
2. **明确缓存生命周期**：推理缓存视觉 K/V；换视频、换输入或更新参数必须失效。训练有梯度路径不得复用 detached 旧投影。建 memory、选择、读出耗时全部计入。
3. **先温启动读出**：只用训练视频让零初始化残差产生有效信号。没有读出能力时，不用全相等的候选收益宣判“选择无用”。
4. **200 个固定前缀探针**：从开发视频覆盖生成前/中/后段，target 输入和已提交前缀固定，比较 bypass、固定、随机、注意力及若干候选中的最好证据。每组自己生成候选树，由完整 target 以相同预算验证，不能用旧候选的 teacher logits 代替新候选验证。每个候选前恢复干净 cache。
5. **接受收益监督**：`A(S)=accepted_prefix(S)-accepted_prefix(bypass)`。先用候选排名监督选择器，OPD 训练草稿和 readout；随着 head 更新刷新 on-policy 数据。无收益时学 bypass。硬 top-k 必须有显式选择信用分配。推理只选一次、draft 一次，无候选枚举。

候选中最好者只是这组候选的 oracle，不是理论上限。如果它也没有收益，先改任务/特征/读出位置；若 oracle 有收益但 learned 接近随机，改监督；若 τ 有益而 wall 无益，优先优化读取成本。

最小训练对照：

| 组 | 内容 | 归因 |
|---|---|---|
| A | 当前结构＋最强已验证通用 OPD 配方 | 强基线 |
| B | 视觉读出＋同一 OPD | 架构增益 |
| C | 与 B 相同结构＋OPD＋接受收益选择监督 | 训练/架构配合的增益，主要看 C−B |

B/C 共享同一温启动 checkpoint；相同训练数据、更新预算、顺序与 head 初始化。第一轮建议 3 epoch，各 epoch 在 dev 选 checkpoint，连续保留 Adam，不以最终测试集调参。额外 teacher 候选验证预算如实报告，不能说三臂计算量相同。先一 seed 排错，再补两个 seed。此为研究预算建议，需要在目标服务器执行任务中明确采用后才启动。

随后补随机/固定/注意力选择、bypass、固定 K，以及“架构有无 × SFT/OPD”的可实现消融。若某目标只对 selector 定义，就不要构造一个不存在的“无 selector 却训练 selector”的 2×2。SFT 先在 dev 校准稳定 lr，不能只靠崩溃对照证明方法。

## 5. 扩展条件与正式论文实验

工程上可预先设继续投入目标：相对最强 OPD 端到端约 ≥5%，三 seed 方向一致，最终独立视频的配对 CI 支持改进；5% 是项目目标，不是领域通用显著性标准。tau 有提升但 wall 无提升时不能写成整体加速。

正式新数据只先加 [YouCook2](https://youcook2.eecs.umich.edu/)：多步骤描述与时间边界适合检验事件切换。先在可获取媒体上做 300 train / 50 dev / 100 独立评估视频，自建子集需明确来源 split；原描述短，仍应审查自然长报告。官方视频可取回率需先检查，不能承诺原链接全部可用。Video-MME-derived 留作跨题材开发/迁移，旧 VideoDetailCaption 留作回归。

最终报告：真实输入预算与输出长度分布、无损、端到端及 decode 分解、视觉读出开销、显存、按输出位置/事件切换的接受情况；bootstrap 以视频为单位，不能把同一视频几百个 prefix 当独立样本。3 seed 共享同一测试集不等于三个独立测试集。

确认机制后再适配 32B；重新配置/训练匹配 head，并在同精度 AR 下检验无损。当前目标是 lossless greedy，不凭逐 token 贪心相等宣称随机采样分布已验证。

## 6. 可复制给另一台服务器代理的执行 prompt

```text
请在这台有可用 GPU 的服务器执行 VisionSpec 的真实视频长生成第一阶段诊断。

仓库：https://github.com/mcy1123/VisionSpec
分支：fix/visual-opd-long-video-protocol
先读 AGENTS.md 和 docs/reports/2026-11-11-real-video-opd-execution-plan.md。

本次授权：拉取该分支到隔离 worktree、定位现有模型/数据、CPU 检查、
物化 80 train/40 dev 的真实帧数据、冻结 schedule、trainer dry-run、
4 条 GPU 无损门禁，以及通过后 3 个条件各 40 条的零更新诊断。
本次 optimizer update 预算为 0；文档第二阶段的新模块及 A/B/C 训练尚未实现，
不要把诊断 trainer 当联合训练入口，也不要自动训练、重试或扩大预算。

请自主完成上述已授权步骤，不要逐条向我确认普通路径和实现选择。
发现缺失资源先查这台机器已有目录；匹配的 8B target/head 身份不清时明确报告，
不能拿 32B 路径直接替换、不能静默换 checkpoint/精度。
若当前目录不是Git树，先快照，再单独checkout，不覆盖历史代码。

数据只用现有Video-MME原视频，按文档真实64/128嵌套采样。
找到并排除历史已用视频，按源视频隔离80/40；所有新数据/缓存/结果实际写在
/mnt/local、/mnt/local2或/mnt/local3，先realpath和df检查。
不要复制16帧凑128，也不要强制最少输出token。VFR时间无法准确表示时报告，
不能偷偷使用近似时间或重复帧过门禁。

显式使用real时间、global_pooled dry-run、fp32 target评测、b63k8、
max_new=1024、eos_stop=immediate和独立AR逐token检查。
先运行文档列出的CPU测试和preflight，再做4条GPU门禁。
任何逐token失配保存现场并停止后续GPU步骤，不降低精度或放宽容差。

GPU用UUID绑定，确认空闲，不结束其他任务。长进程使用tmux，脚本记录日志、
退出码和DONE/FAILED，后台能独立完成，不需要聊天持续在线。

最终写独立报告并登记docs/reports/README.md，汇报实际帧/token/输出长度，
AR文本质量、EOS/截断、lossless、有效tokens/round、输入固定时的zero-visual
配对差以及计时边界。不得把64/128改变target的差当作selector因果收益，
不得重用“76%视觉信息”结论。区分已执行结果与尚未实现的第二阶段，
据结果明确推荐继续做证据读出、修任务，还是先优化运行成本。
```

## 7. 本机验证记录

本机所列六组 CPU 回归 **55 项测试、30 项子测试通过**，覆盖真实视频解码/PTS/NPZ 往返、源视频划分与排除、schedule 预算、梯度归一化、配置兼容、绝对位置视觉掩码、跨层缓存与全前缀重算/rollback/extension 一致性，以及分片协议混合拒绝。`git diff --check` 通过；没有本机 GPU lossless、吞吐或训练恢复结论。

另用本机已有 Qwen3-VL-32B-Thinking 的 processor（只加载处理器，不加载模型）完成 CPU 小样本：4 个合成帧对应源时间 0/3/6/9 秒，实际输出时间标记 1.5/7.5 秒、grid `[2,4,4]`、8 个 visual token，符合 temporal patch 成对平均。它只验证真实时间接口，并非 8B 模型推理或真实长视频验收。

扩大旧测试时，`repro/tests/test_official_kl.py` 因缺失历史 `.official_kl_audit_run` 文件在 collection 阶段失败；没有伪造历史产物来通过该测试，也不声明整个历史测试集通过。

未完成的远端检查：真实 8B processor 的 64/128 token 数及时间标记、fp32 多卡 AR/SPEC 无损、GPU 峰值内存、跨层 xattn 的目标尺寸运行及实际收益、新旧归一化的新配对训练。既有 checkpoint 和旧结果不因 CPU 测试通过而重新标为已验证。

边界：native rollout 靠近 `max_seq_len` 时还需要约一个 draft block 的空间；不能把 `prompt + max_new == max_seq_len` 当作能完整生成的保证。本方案 32768 上限预留了余量，实际输入仍须门禁测量，后续扩大长度需重新核验。
