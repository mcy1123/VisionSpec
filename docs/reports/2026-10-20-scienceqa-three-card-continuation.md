# ScienceQA：三卡续训两个 epoch（2026-10-20）

用户明确授权：使用三张卡，从已完成的 SFT / OPD 各自 step1182 继续训练两个 epoch，启动后独立运行，不持续监控。此次授权为每臂新增 2364 次真实更新、9452 条 QA 曝光；最终累计 step3546。固定训练两轮，不启用平台早停。不会从中途正结果挑选主终点。

产物根：`/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020`。父实验：`/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_restart_20261019_180942`。物理 GPU 0、1、2，使用 UUID 固定设备身份。Python 为 `fin_research`。

> **2026-10-20 21:25 更新**：物理卡改为 1/2/7，空闲门控 40 GiB→38 GiB（用户批准）。原 0/1/2 绑定在 opd_1782 上遇到邻居任务导致 CUDA OOM，详见文末「运行事件与换绑」。世界大小、slot map、RNG、日程、lr、预算与 checkpoint 身份均未改变。

## 三卡与恢复

新增独立入口 `repro/opd_official_kl/three_card_trainer.py`，原四卡默认入口保留。三个物理 rank 执行四个逻辑 QA slot：rank0=[0,3]，rank1=[1]，rank2=[2]。rank0 顺序完成两个 rollout；所有 slot 使用同一个更新前 Head，rank0 仍按 slot0–3 / anchor 原顺序执行 scoring/backward 和一次 CPU master AdamW 更新，再广播 Head。

四条原 per-rank RNG 流显式迁移为四条 per-slot RNG 流，每个 slot 开始时恢复、结束时保存。rank0 在 backward 前恢复 slot0 的 RNG 流，保存时写回该流。新 checkpoint 明确记录 world_size=3、slot map、实现标识和四条 RNG，不允许原四卡入口直接载入。此迁移不声明跨拓扑训练的逐位数值等价。

`prepare_continue.py` 显式迁移两臂训练状态，保留 Head、fp32 master、Adam moments/step、RNG、1182 更新步数；检查迁移保存后对应张量精确相等。旧状态原样保留，迁移及新训练均写独立目录。`MIGRATION.json` 记录源文件 SHA、旧新身份及校验结果。

## 日程与训练协议

保持原 4726 QA / 4352 图像训练池、原固定 AR 标签、lr=1e-5 constant、Target fp32、Head bf16、CPU fp32 master AdamW、4 QA/update。原日程文本完整保留为前缀；epoch2/3 用种子20261014/20261015重新打乱同一4726 QA，并给两臂使用完全相同的顺序。每轮1182次更新，末尾batch2，因此总步数为3546。

第一条真实续训更新分别保存为 step1183，作为启动检查，计入正式预算，无额外 smoke 更新。随后每约200次更新及epoch边界保存完整 checkpoint、评测固定dev32；评测仅记录SPEC效率和覆盖，不做训练期SPEC=AR检查，也不触发早停。两臂固定相同QA曝光及optimizer更新数。

最终step3546用既有独立AR缓存做逐token输出检查，复用confirm200仅用于描述性比较。该集合已参与研究决策，不再称为新的独立最终测试。若以后需要新的独立正式确认，须另行冻结未参与研究的图像集合。

## 后台运行与产物

入口：`repro/scienceqa_image/full_epoch/continue_three_card.py`，启动脚本为产物根 `launch.sh`。tmux会话 `glance-sqa-ep23-3gpu`。无持续人工监控，也没有自动追加预算或失败重试。

`CURRENT.json`/`PROGRESS.json`/`LATEST.json` 记录阶段、两臂共同完成步数和完整checkpoint位置；`logs/`保留每阶段日志；`BUDGET.json`保留每次optimizer尝试与完成；`RECOVER_sft.json`/`RECOVER_opd.json`记录恢复命令。仅已原子保存的完整checkpoint可恢复，未保存的optimizer尝试不得静默重放。

GPU可共享，只要求容纳当前模型；资源不足时等待，不终止别人的任务。真实运行异常会写`STOPPED.json`与`DONE.txt`，保留已保存checkpoint。正常完成写`DONE.json`/`DONE.txt`，并生成`CURVES.png`、`CURVES.csv`、`TRAINING_CURVES.jsonl`、`FINAL_RESULT.json`与`FINAL_REPORT.md`，记录三卡OPD的实际GPU-hours。

状态以产物中的 `STARTUP_OK.json`、`PROGRESS.json`、`DONE.json` 或 `STOPPED.json` 为准。

## 运行事件与换绑（2026-10-20 21:20–21:26）

**第一次停机：资源门控等待 5 小时 44 分（不计入训练）。** 10-20 15:29 完成到共同 step1582 的两臂训练与 dev 评测、并完成 sft_1782 训练与评测后，控制器停在阶段启动前的空闲显存门控（要求被分配的三张卡各 ≥40 GiB 空闲）。当时 GPU2 被同机用户 zcy 的 MEG 任务占用 4.9 GB，只剩 39,616 MiB，比门限少 1,344 MiB；控制器每 30 秒重查一次，共等待 653 轮（`controller.log`），期间没有任何训练或评测。

**第二次事件：opd_1782 CUDA OOM。** 21:12:45 同机用户 wyf 启动新的 video_qa 任务（PID 1680984）。21:13 GPU2 显存释放后门控通过，21:13:56 启动 opd_1782。该任务在 opd_1782 加载 Target 的 6 分钟内涨到约 26 GiB，21:20:12 rank0 在 GPU0 上 OOM（仅剩 143.31 MiB，本进程 23.69 GiB）。控制器按协议写 `STOPPED.json`/`DONE.txt` 后退出，tmux 会话随之结束。

**失败未消耗预算。** `BUDGET.json` 中 opd 最大 index=1581、sft 最大 index=1781；`train/opd/to_1782` 为空，没有半成品 checkpoint。两臂仍分别停在 opd step1582、sft step1782，已完成 400/2364 次新增更新。

**用户批准的换绑。** 物理卡 0/1/2 → 1/2/7（同为 49 GiB 卡），阶段启动门控 40 GiB → 38 GiB（当时 GPU1/2/7 空闲 48,658 / 43,745 / 39,476 MiB）。world_size=3、slot map、四条 RNG 流、日程、lr、预算与 checkpoint 身份逻辑均不变；冻结代码树的 `SOURCE.json` 哈希仍在启动时自检，改动只存在于运行目录副本 `rebind_20261020/continue_three_card_gpus127.py`（原控制器未改）。换绑理由、旧新 UUID、哈希与预算状态记录在 `rebind_20261020/REBIND.json`。

**证据保留。** 失败阶段记录 `stages/opd_1782.failed_20261020T2120.json`、失败日志 `logs/opd_1782.failed_20261020T2120.log`、原 `CURRENT.json`/命令记录/`STOPPED.json`/`DONE.txt` 均在 `rebind_20261020/preserved/`，未覆盖或删除。

**重启与实测占用。** 21:25:50 重建 tmux `glance-sqa-ep23-3gpu` 并运行 `rebind_20261020/launch_rebind.sh`，门控立即通过，opd_1782 从 step1582 继续；`CURRENT.json` 记录的设备已是 1/2/7，`RESOURCES.json` 记录门控策略为 38 GiB。21:29:44 起日志开始输出逐 update JSON（1583、1584、1585…），`BUDGET.json` 同步记录到 opd index 1584，确认是真实训练而非仅加载。

**实测每 rank 占用约 36.8 GiB。** 加载期间 GPU7 空闲从 39,476 MiB 降到 2,627 MiB（其中 zcy 任务占 9,168 MiB），对应本进程约 36.8 GiB/rank；GPU1、GPU2 分别稳定在 8.5 GiB、6.6 GiB 空闲。也就是说这张卡的真实足迹接近 37 GiB，原先 40 GiB 门控只留约 4 GiB 余量，38 GiB 门控在 GPU7 上只剩约 2.6 GiB。若发生 OOM，本阶段仍按失败处理、不消耗预算，重试应按 ≥40 GiB 空闲或三张真正空闲的卡安排。

**残余风险。** 门控只在阶段开始时检查一次；GPU7 只剩约 2.6 GiB，若邻居任务（zcy 的 inventory 任务，当时 9.2 GiB、81% 利用率）在本阶段增长超过该余量，opd_1782 会 OOM 并留下新的失败记录。该风险已如实记录，不通过改权重、缩小模型或降低评测口径来回避。

## 最终结果（2026-10-21 18:56 完成）

固定预算 2364 次新增更新全部跑完，两臂各到 step3546；`DONE.json`、`FINAL_REPORT.md`、`FINAL_RESULT.json` 已写出。换绑后到完成期间**没有发生 OOM、没有门控等待**（`controller.log` 在 REBIND 之后的 waiting 计数为 0）。

| 项目 | SFT | OPD |
|---|---|---|
| 新增更新 / QA | 2364 / 9452 | 2364 / 9452 |
| test200 逐 token 与独立 AR 比对 | **200/200 无损** | **200/200 无损** |
| test200 准确数 | 162/200 | 162/200 |
| test200 有效 token/轮 τ | 5.3666 | **6.2206** |

配对差 **OPD − SFT = +0.8541，95% CI [+0.7157, +1.0037]**（同一复用 confirm200 上的逐图像配对，描述性）。dev32 的 τ 也从 1182 的 4.694/5.092 升到 3546 的 5.034/5.707，但 dev32 只有 32 题、且只记 SPEC 效率。

**结论边界**：两臂准确数相同、输出都逐 token 无损，收益全部体现在解码效率（τ）上，与方法的既有主张一致；但这个 200 题集合已参与研究决策，上表只能作描述性比较，**不能称为新的独立最终确认**。另外 `FINAL_RESULT.json` 的 `protocol` 块记录的 GPU 仍是启动时冻结的 0/1/2，实际 21:25 之后跑在 1/2/7，以「运行事件与换绑」一节为准。

成本（`FINAL_RESULT.json` 的 `cost_gpu_hours`）：OPD 53.48、SFT 6.37、评测与报表 2.36 GPU-hours。产物根累计占用约 438 GB（checkpoint 344 GB），`/mnt/local` 完成时余量 143 GB。tmux `glance-sqa-ep23-3gpu` 随控制器正常退出而结束，不残留会话。

## 加速比复测（2026-10-21，step3546）

最终评测 JSON 里的计时不能用来说加速比：`ar_end_to_end_ms` 在两臂之间 200/200 完全相同（`ar_cached=true`），两次评测都被标记 `timing_contaminated=true`，且两臂相隔 20 分钟，prefill 均值 513 vs 719 ms/题的差异（本该相同）就足以吃掉全部总时间差。因此按 [2026-10-20 速度报告](2026-10-20-scienceqa-inference-speed.md) 的同一协议单独重测：24 图固定子集、三臂顺序均衡、每图 2 次、每臂 2 次预热、AR 现场运行、CUDA 同步外部墙钟、fp32 Target+Head、budget63/branch_k8/immediate EOS，所选卡（GPU2）无其他计算进程。

计时脚本为运行目录内副本 `analysis_speed_20261021/benchmark_speed_step3546.py`（仅把 checkpoint 路径从 `to_1182/step1182` 改为 `to_3546/step3546`），原脚本未改。

| 方法 | 平均完整生成 ms/题 | 完整生成 token/s | AR / 方法 |
|---|---:|---:|---:|
| AR | 7734.44 | 18.38 | 1.000× |
| SFT | 3369.65 | 42.19 | 2.295× |
| OPD | 3067.38 | 46.35 | 2.522× |

| 配对比较 | 总耗时比 | 图像配对 bootstrap 95% CI |
|---|---:|---:|
| ar/sft | 2.295× | [2.055, 2.584] |
| ar/opd | 2.522× | [2.248, 2.874] |
| sft/opd | **1.099×** | [1.063, 1.141] |

144 次正式生成与既有独立 AR token 缓存逐 token 一致（`all_token_exact=true`），所选 GPU 的外进程采样为空（`timing_contaminated=false`）。24 图里 OPD 有 20 图快于 SFT，逐图 SFT/OPD 中位 1.095×。

**机制**：两臂每轮耗时几乎相同（114.5 ms/round），加速全部来自更少的轮数——OPD τ 5.306 vs SFT 4.833，总轮数 1286 vs 1412；AR 为 54.4 ms/token。与 step1182 的 2026-10-20 测量相比，AR/OPD 从 2.387× 升到 2.522×，SFT/OPD 从 1.068× 升到 1.099×。

**范围**：24 图、单卡 A6000、batch1、fp32、不含图片读取/processor/模型加载；不外推为全 200 题或生产服务吞吐。产物：`/mnt/local/mcy/GLANCE/results/scienceqa_speed_step3546_20261021`（`PLAN.json`、`LOAD_START.json`/`LOAD_SAMPLES.json`、`measurements.jsonl`、`RESULT.json`）。

### ChartQA 跨域复测（2026-10-21）

同一计时协议、同一 fp32 配置、同一张空闲卡，把数据换成 ChartQA：从 `data/vl_eval_plain/chartqa.jsonl` 的 2,500 道 test 题里按 seed=20261021 抽 24 题、prompt 为纯问题（无选项、无 `Answer: (X)`），Head 仍是 ScienceQA 训出的 step3546 两臂（**跨域**，从未见过图表图像）。ChartQA 没有现成 AR 缓存，所以先现场跑一遍 AR 生成作为 oracle，计时跑必须逐 token 复现它。

| 方法 | 平均完整生成 ms/题 | 生成 token/s | AR / 方法 |
|---|---:|---:|---:|
| AR | 704.78 | 6.50 | 1.000× |
| SFT | 644.89 | 7.11 | 1.093× |
| OPD | 689.75 | 6.64 | 1.022× |

| 配对比较 | 总耗时比 | 图像配对 bootstrap 95% CI |
|---|---:|---:|
| ar/sft | 1.093× | [1.048, 1.142] |
| ar/opd | 1.022× | [0.913, 1.108] |
| sft/opd | **0.935×** | [0.852, 0.988] |

144 次生成逐 token 无损、所选卡无外进程。24 题里 OPD 只比 SFT 快 8 题，逐图中位 SFT/OPD 0.996×。

**为什么和 ScienceQA 差这么多**：ChartQA 的答案是短数值，**平均每题只生成 4.6 个 token**（ScienceQA 中位 141）。AR 704.8 ms/题里约 0.5 s 是 prefill，decode 只占 27%；SFT/OPD 的 prefill 也各占 82%/81%。也就是说端到端时间被与两臂无关的 prefill 主导，投机解码的收益没有空间。τ 也低（SFT 3.548、OPD 3.333，跨域），总轮数 62 vs 66 基本持平。

**结论**：这不是 ChartQA 上的实现问题，而是任务形态问题——投机解码的加速比随答案长度增长、随视觉前缀变长而稀释。ScienceQA（短图 + 长 CoT）恰好是最有利的一端；ChartQA（短答）在最不利的一端。要在 ChartQA 上展示加速比，得先有一个用 ChartQA/图表域训练过的 Head，并让答案长到能撑满多个 block。

产物：`/mnt/local/mcy/GLANCE/results/chartqa_speed_step3546_20261021`，脚本副本 `analysis_speed_20261021/benchmark_chartqa_speed.py`。

## checkpoint 保留（2026-10-21 清理）

按用户指令只留“最好的和最后一个”：判据为固定 dev32 的有效 token/轮在该臂 13 个网格点中最高。保留 3 个完整 checkpoint——`train/sft/to_3364/step3364`（SFT dev τ 5.0652，最高）、`train/sft/to_3546/step3546`、`train/opd/to_3546/step3546`（OPD dev τ 5.7068，最高且即最终点）；删除其余 23 个中间 checkpoint，释放 359.4 GiB，`/mnt/local` 余量从 143 GiB 回到 502 GiB。

保留的 3 个 checkpoint 已逐个复核 `SHA256.json` 中 `model.safetensors` 的 sha256 一致、`training_state.pt`/`config.json` 在位。`migration/{sft,opd}/step1182`（32 GB）**未删**：它是本轮续训的输入身份（三卡迁移状态），不是网格 checkpoint，删掉会让 `MIGRATION.json` 的迁移记录不可复核；如需一并回收可另行明确。

删除清单连同每个 checkpoint 的 `SHA256.json` 与 `config.json` 副本记在 `CLEANUP_20261021.json`；训练曲线（`TRAINING_CURVES.jsonl`/`CURVES.csv`）、每阶段 eval JSON、日志与 `BUDGET.json` 均未动。因此：曲线与结论可复核，`step3546` 可复现最终评测与继续训练，但中间点的重新推理已不可行（需要时只能按 `BUDGET.json`/日志重跑）。`PROGRESS.json`、`LATEST.json`、`RECOVER_*.json` 指向的仍是 `step3546`，未受影响。

磁盘占用：产物根从 438 GB 降到 79 GB（其中 train 48 GB、migration 32 GB）。父实验 `scienceqa_full_epoch_restart_20261019_180942`（40 GB）与历史 confirm200 等目录未动。

## DocVQA 探针：数据集不是杠杆，答案长度才是（2026-10-21）

按“和 ScienceQA 最像”的判断选了 DocVQA。数据取自公开小样 `nielsr/docvqa_1200_examples`（HF 镜像），解出 24 张文档图与对应问答（seed=20261021），存在 `/mnt/local/mcy/GLANCE/datasets/docvqa_slice/`。Head 仍是 ScienceQA step3546 两臂（跨域），计时协议与前两次完全相同（24 题、顺序均衡、每图 2 次、fp32、现场 AR oracle、所选卡无外进程、144 次生成全逐 token 无损）。

关键设计：**同一批图片、同一组 Head、同一 budget，只改 prompt**，一种要求“最短答案”，一种要求先推理再给 `Answer: <short answer>`。

| prompt | 答案长度 | AR ms/题 | SFT τ | OPD τ | AR/SFT | AR/OPD | SFT/OPD |
|---|---|---:|---:|---:|---:|---:|---:|
| 短答 | 6.2 token | 831.64 | 3.73 | 3.82 | 1.131× | 1.144× | 1.012× [0.996, 1.031] |
| 先推理后作答 | 139.5 token | 7931.27 | 4.59 | 4.80 | 2.144× | 2.221× | **1.036× [1.021, 1.052]** |

短答时 OPD 只在 14/24 题上快于 SFT、逐图中位 1.001×；改成 CoT 后 17/24 题更快、中位 1.034×。两边都零 token 偏差、无外进程污染。

**结论**：同一批 DocVQA 图片，仅因为 prompt 把答案从 6.2 token 拉到 139.5 token，AR 相对加速从 1.14× 跳到 2.22×，OPD 相对 SFT 从“跨零”变成“显著但很小”（1.036×）。这和 ChartQA（4.6 token → 0.935×）以及 ScienceQA（中位 141 token → 1.099×）连起来，按答案长度排成一条单调线。所以“找哪个数据集像 SQA”不如直接问“答案有多长”——短答数据集上投机解码没有 decode loop 可摊薄，这与 GLANCE 论文把 SQA 当作 `below parity` 的边界案例是同一件事。

**为什么 DocVQA+CoT 仍低于 ScienceQA**：Head 是 ScienceQA 训的，跨域 τ 只有 4.59/4.80（ScienceQA 4.83/5.31），OPD 相对 SFT 的优势因此从 1.099× 缩到 1.036×。要拿到干净的跨域结论，需要 DocVQA 域的 Warm/Head 与同预算 SFT 对照，那是新训练，不在本轮授权内。

产物：数据 `/mnt/local/mcy/GLANCE/datasets/docvqa_slice/`（`raw/` 两个 parquet、`images/`、`docvqa_24.jsonl`）；计时 `/mnt/local/mcy/GLANCE/results/docvqa_speed_step3546_20261021/{plain,cot}/`；脚本 `analysis_speed_20261021/benchmark_docvqa_speed.py`。

## 跨数据集与长度上限探针（2026-10-22）

用同一协议（24 题、顺序均衡、每图 2 次、fp32、现场 AR oracle、144 次生成全逐 token 无损）把 step3546 两臂跑到 Draft-OPD 论文库里常见的数据集上。Head 全部是 **ScienceQA 训的、这些数据一条没训过**（跨域）；括号内标注 prompt 风格。

| 数据集 | 答案长度 | τ SFT/OPD | 端到端 AR/SFT | AR/OPD | SFT/OPD |
|---|---:|---:|---:|---:|---:|
| ShareGPT4V（caption） | 252.6 | 3.41/3.46 | 1.782× | 1.821× | 1.022× |
| MathVista testmini（CoT） | 200.4 | 4.34/4.44 | 2.154× | 2.199× | 1.021× |
| DocVQA（CoT） | 139.5 | 4.59/4.80 | 2.144× | 2.221× | 1.036× |
| SQA（CoT，域内） | 142.2 | 4.83/5.31 | 2.295× | 2.522× | 1.099× |
| OCRBench（短答） | 9.6 | 3.08/3.04 | 1.194× | 1.204× | 1.008× |
| TextVQA（短答） | 4.3 | 3.12/3.22 | 1.055× | 1.050× | 0.995× |
| ChartQA（短答） | 4.6 | 3.55/3.33 | 1.093× | 1.022× | 0.935× |

结论与 [ChartQA 跨域复测] 一致：**答案长度是端到端加速比的第一变量**（4–10 token 的短答 ≈1.0×，140–250 token 的 CoT/caption 2.2–2.5×），而 τ 由“答案是否锚定在图上、是否结构化”决定——自由文本 caption 最长但 τ 最低（3.4），和 GLANCE 论文 captioning τ 2.91 的排序一致。对照两篇论文：ViSpec（域内）TextVQA τ 2.95、SQA 2.86；DREAM（每基准 1k 域内样本）OCRBench 4.88、MathVista 5.32、SQA 5.82。我们的跨域 τ 在短答集上与 ViSpec 相当，但低于 DREAM 的域内水平。

**长度上限探针**（只改 `--max-new-tokens`，不重训、不重造标签）：

| 设置 | 平均长度 | 打满 512 | τ SFT/OPD | AR/OPD | SFT/OPD | AR 单题 |
|---|---:|---:|---:|---:|---:|---:|
| SQA @256 | 142.2 | — | 4.83/5.31 | 2.522× | 1.099× | 7.73 s |
| SQA @512 | 150.4 | 0/48 | 4.83/5.29 | 2.484× | 1.078× | 8.13 s |
| ShareGPT4V @256 | 252.6 | 96% | 3.41/3.46 | 1.821× | 1.022× | 14.3 s |
| ShareGPT4V @512 | 372.6 | 25% | 3.43/3.47 | 1.748× | 1.008× [0.987,1.023] | 19.9 s |

因此 **cap 是保真度开关，不是效率杠杆**：放开它只让答案不被截断（SQA +6%、caption +47%），τ 和加速比基本不动，甚至略降；SQA 的自然长度最大只有 338，256 的截断很轻微；ShareGPT4V 在 512 下仍有 25% 打满，自然长度在 400 token 以上。要提 τ/加速比仍需 in-domain 训练或更强 drafter。

产物：`/mnt/local/mcy/GLANCE/results/{sharegpt4v,mathvista,ocrbench,textvqa,sqa512,sharegpt4v512}_speed_step3546_20261021/`；数据切片 `/mnt/local/mcy/GLANCE/datasets/{sharegpt4v,mathvista,ocrbench,textvqa,sqa_probe24}_*/`；脚本 `analysis_speed_20261021/benchmark_generic_speed.py`（新增 `--max-new` 与 `raw/caption` prompt 风格）。
