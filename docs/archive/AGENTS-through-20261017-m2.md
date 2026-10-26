# 数据集与 checkpoint 存储规则（必须遵守）

- 数据集、预处理数据/缓存、模型权重和 checkpoint 必须实际存放在 `/mnt/local`、`/mnt/local2` 或 `/mnt/local3` 下；禁止下载或保存到 home 目录（包括 `/home/mcy`）的实际存储中。
- 下载、解压、缓存和保存 checkpoint 前，先核实目标路径及软链接解析后的真实位置；相关临时大文件也应放在上述 `/mnt` 目录中。
- home 下可以保留指向上述 `/mnt` 目录的软链接，但不能把软链接替换为存放数据或 checkpoint 的真实目录。现有 `/home/mcy/local`、`local2`、`local3` 属于此类软链接。
- 默认 checkpoint 路径为 `/mnt/local/mcy/GLANCE/checkpoints/`；本轮视频 QA 数据位于 `/mnt/local2/mcy/datasets/videoqa_transfer/`。可按容量选择另外两个允许的挂载目录。

# GLANCE 当前状态与交接

最后核对：**2026-10-17，旧图像 Warm 仍不可恢复；已从本地文本 DFlash Head 训练新的图像 Warm 2000 updates，再从同一新 Warm 完成 ScienceQA continue-SFT 与官方 serial KL-OPD 各 120 updates。M1/M2 两臂在同一 dev8 上均 8/8 逐 token SPEC=独立 AR；最终 19 项独立验收 PASS，Target 权重 SHA 未变。block verifier 仍 HOLD。此前“未训练”仅为历史阶段状态。**

本文件是唯一当前进度入口。完整旧记录已归档至 [历史 AGENTS](docs/archive/legacy_opd_pre20261016/AGENTS-before-cleanup.md)，不得把其中的待执行任务视为新授权。

## 当前方法和验证边界

唯一方法说明：[官方 KL 契约](docs/methods/draft_opd_official_kl_contract.md)。

指定的 `run_qwen_gsm8k.sh` 在 HEAD `d6f0ec859321959785c65b59eb311fd25f042158` 中最后修改为 `d1476f5d00a5fb187eb1637dc3f0f610ddbe1e1d`。只读 refs 目录不是独立 Git 根。实际基础脚本 default forward=0/reverse=1/rejected_use_reverse=False，因此两流 k3；交接中的 Bernoulli+k3 需要显式覆盖，不能冒称基础脚本默认。用户已批准“基础脚本+显式方向覆盖”：forward=1/reverse=0/rejected_use_reverse=True，保留 decay=.9、流权重1:2、log最小-10。已实现公共 loss/online rollout，未开展真实模型训练。

ScienceQA 和视频的未来 OPD CLI 均进入同一个 canonical runtime：

```bash
python repro/scienceqa_image/train_sqa_opd.py --dry-run
python repro/train_video_official_kl.py --dry-run
```

dry-run 输出明确配方、不加载模型。执行路径为 cli→trainer→adapters→runtime→loss；缺必要输入/旧开关 fail-fast。旧 `opd_train_video_fullkl.py`、`qa_opd_fair/train_qa_opd.py`、`opd_train_video.py`、`opd_train.py` 直接执行也转入公共入口；历史 helper 为诊断保留，不是 active 方法。旧源码在 `repro/legacy/pre20261016/`，重放须用 `repro/legacy/replay.py --acknowledge-historical-objective` 显式声明。旧图像 trainer 生成器已退休。

## 研究结果与边界

- 图像 ChartQA 复现历史 τ≈5.20；[复现报告](docs/reports/2026-09-25-reproduction-history.md)。
- 视频 A4 aligned SFT 起点 `checkpoints/head_video439_aligned/step1000`，mean τ=3.7377；[报告](docs/reports/2026-10-09-phase-a4.md)。
- Task A/P1 只实现 rejected k3，accepted 仍是全词表 forward KL；不是完整官方对齐，P2/P3 未运行；[报告](docs/reports/2026-10-15-taskA-kl-official-p1.md)。
- ScienceQA 实际目标为 **fixed-AR-anchor CE + full-reverse-KL**。dev I0 4.0050 / SFT 4.3366、4.3394 / OPD 3.9415、3.9495；test I0 4.0723 / SFT 4.4059、4.4309 / OPD 4.0479、4.0583。原配方负结果有效，不能称官方完整 online KL 已失败。
- ScienceQA blocker/EOS/horizon 机制统计存在 bug，“树遗漏极少”的解释撤回；新离线复算的有效 tokens/rounds OPD−SFT 四组仍全部负且 CI 不跨零。parsed-only 答案准确率不可冒充全题准确率；[正式纠错](docs/reports/2026-10-16-scienceqa-implementation-and-metrics-errata.md)。
- 原 ScienceQA、Task A/B、第五轮 QA 的报告、checkpoint、日志和运行时源码证据不改写。历史实验详见[报告索引](docs/reports/README.md)。
- C7、D4、新目标、新种子和追加训练均不自动恢复；旧计划已归档，不能把未勾选项当作待执行命令。

## 当前门禁

18 项 CPU 单测通过（`test_official_kl.py` + `test_official_cleanup.py`）：冻结官方 Bernoulli/k3 数值与梯度、pool和全局response分母、官方anchor计划、合成在线AR/KV/teacher前缀、两个adapter的实际公共调用、入口/指标/存储门禁。SFT保留variable-reveal CE。

SFT/OPD共用恢复契约：head/master/Adam、Python/NumPy/torch/CUDA RNG、cursor/step、日程/数据/模型配置hash。CPU toy SFT与官方KL分别在fp32/bf16参数下4 vs 2+2的loss、梯度、参数和Adam逐位相同；不代表历史checkpoint或真实GPU端到端恢复已通过。

GPU no-update未运行：共享卡可用显存不足约40 GiB模型路径需求。当前 **L2 CPU-only / 待模型门禁**，不是L3，不宣称官方完整训练栈或收益已复现。当前任务不授权训练。

历史证据核验：23份旧仓库报告与任务前快照SHA一致；268个外部产物已读出SHA，其中4份与旧交付manifest不符（包括旧FINAL报告），记录于审计目录，未修改原件，不能声称全体与旧manifest一致。其余详情见纠错报告。后续不得据此自动“修正”旧manifest。

再次可读性检查：公共 runtime/trainer 展开单行语句、明确两流变量；5个兼容入口移除1468行已禁用且不可达的旧训练循环，原件仍在legacy归档。一次性修改脚本和临时格式化环境已清理，核心回归测试保留。另修复视频入口被Python导入后main()漏传modality的问题；18项核心测试和17项旧helper兼容测试通过。

## 目录组织（2026-10-16）

`repro/` 的143个历史顶层文件已按 shared、data、evaluation、diagnostics、tests、historical、tools 分目录；当前 OPD 保留在 opd_official_kl，图像/视频任务目录保持独立。当前入口不再依赖旧 OPD loss/rollout；共享几何与SFT block已抽出。旧报告内容不变，旧路径用 `python repro/run.py --list` 或 `repro/layout.json` 查找。整理验证：29项核心/在线CPU测试、17项旧训练helper测试、19项旧KL测试、图像适配测试通过；227份Python与30份shell语法检查、12条CLI/help门禁通过。整理快照与映射：`/mnt/local3/mcy/GLANCE/results/repro_reorganization_20261016/`。

## 工作规则

- 代码根 `/home/mcy/Projects/GLANCE` 非 Git 工作树；改动前必须可恢复快照。现有环境 `/home/mcy/.conda/envs/fin_research/bin/python`。
- 不删除历史 checkpoint、原始结果、报告快照；不覆盖运行中的 shell 脚本。外部 Draft-OPD 与 refs 仅只读。
- 后续任何真实训练须另行明确授权；lr≤1e-4 使用 fp32 master，target/context 精度、数据预算和优化器逐项记录。
- 不修改生产 decoder/tree 或既有权重来迎合指标；SPEC==独立 AR 必须逐 token 一致。
- 当前 docs/methods 契约优先于历史计划。新增报告加入 docs/reports/README.md；不另建平行“最新项目总结”。
- 审计/恢复根：`/mnt/local3/mcy/GLANCE/results/official_kl_cleanup_20261016_131301`。`changes/restore.sh` 只恢复到新目录，不覆盖当前工作区。

## 独立验收增量（2026-10-16）

[完整验收报告](/mnt/local3/mcy/GLANCE/results/official_kl_final_acceptance_20261016_165617/reports/FINAL_ACCEPTANCE_REVIEW.md)，[放行门禁](/mnt/local3/mcy/GLANCE/results/official_kl_final_acceptance_20261016_165617/release_gate.csv)。上轮两套交付共1001文件SHA吻合；18项原CPU测试复跑通过，补丁后21项通过。ScienceQA/视频实际CLI→trainer→runtime→公共KL已在CPU真实tiny DFlash＋合成Target/媒体下执行；两个no-update均无optimizer.step。图像/视频的SFT与OPD四条实际训练路径均完成独立进程4 vs 2+2，逐步loss/梯度/QA/anchor/reveal/RNG与完整head/master/Adam状态逐位一致。修复SFT提前停止伪最终checkpoint、非网格末步缺完整状态、视频SFT日程hash路径不一致，并增加覆盖保护与CLI有效配方help。

这不代表真实多模态/GPU完整训练已通过：共享GPU无满足约40 GiB的空闲资源，真实图像/视频no-update与新增SPEC=AR未运行。结论hold_pending_gpu_gate，下一轮训练仍须通过GPU门禁并另行授权。265份小型历史原件本轮前后SHA一致；大checkpoint沿用先前审计，本轮未全量重哈希。历史纠错1200anchors与先前结果一致。

## GPU 门禁自动监控（2026-10-16）

用户已授权持续监控并在资源满足时直接占用执行无更新门禁。监控 PID 3166347，每30秒检查；要求无驻留计算进程、可用显存≥40960 MiB、利用率≤5%，启动前再次核对。状态文件：`/mnt/local3/mcy/GLANCE/results/official_kl_gpu_watch_20261016/status.json`；worker日志 `gate.log`，结果 `RESULT.json`。自动任务仅2条训练来源ScienceQA图像＋1条既有训练视频的no-update，不启动10-update或正式训练；失败即退出并保持HOLD，通过后仍需审阅证据再更新等级。脚本在独立审计目录，不改变生产代码。不要另起重复GPU监控或训练任务。

## 真实GPU门禁完成（2026-10-16）

[GPU审阅报告](/mnt/local3/mcy/GLANCE/results/official_kl_gpu_watch_20261016/review/GPU_GATE_REVIEW.md)。GPU0在北京时间21:55:11自动启动，22:04:47完成，退出0；监控与worker均已结束。2条训练ScienceQA图像＋1条原有训练视频通过真实CLI官方KL前反向，Target无梯度，optimizer_steps=0。157/99/60 tokens均含自然EOS，树SPEC、serial rollout和独立AR逐token一致；峰值allocated显存39.904 GiB。5份Target/Head权重文件前后SHA一致。训练Head bf16，树评测用同值提升fp32，Target全程fp32。当前L2_gpu，可建议下一轮另行授权的成对10-update正确性探针；未训练，不是L3。真实GPU更新与恢复尚未验证，不自动开展训练。前文L2_cpu/HOLD为历史检查时状态。

## 正式夜间训练授权与容量收口（2026-10-16）

用户明确授权跳过独立10-update探针，按附件A/B/C/D固定档位执行SFT/官方KL成对正式训练，最长9小时，真实更新/恢复并入前几步；旧“只允许10-update”建议不再是授权限制。

新run：`/mnt/local3/mcy/GLANCE/results/official_kl_scienceqa_overnight_20261016_222121`；[首先查看](/mnt/local3/mcy/GLANCE/results/official_kl_scienceqa_overnight_20261016_222121/READ_ME_FIRST.md)，[最终报告](/mnt/local3/mcy/GLANCE/results/official_kl_scienceqa_overnight_20261016_222121/reports/FINAL_OVERNIGHT_SCIENCEQA.md)。tmux `glance-sqa-night-20261016-222121` 实际执行了资源/数据/吞吐容量审计，pane已正常退出（exit0），不是正式训练进程。状态BLOCKED / blocked_budget_no_feasible_tier，SFT与official KL均0 updates，无新checkpoint、无新dev结果。

现场仅GPU0满足空闲及显存条件。最新真实image no-update循环2 QA耗时108.93秒，估计54.47秒/QA；最低D档1200 QA仅OPD估计18.16小时，加20%余量21.79小时，尚未计SFT/标注/恢复/评测。此为2样本估计，不能宣称精确预测；也不能借旧fixed-anchor OPD约10秒/update推断新online方法可完成。按附件预算硬门禁安全停止，未擅自缩小档位或更改实现。

重新读取官方源图像字节做跨split SHA去重：train6218条，去重隔离后4726 QA/4352互异图像；没有读取test题干/答案/模型输出做方法选择。候选seed顺序与档位前缀已保存schedule.json，但未激活正式日程。源码和I0权重SHA复核无变化，历史资产保留。后续若调整总预算/档位需用户明确；不得自动恢复这个已收口作业。

## 块验证加速独立验收（2026-10-16）

用户本轮仅授权语义保持的加速及无更新验证，不启动正式训练。run：`/mnt/local3/mcy/GLANCE/results/official_kl_block_verify_20261016_223740`；[完整报告](/mnt/local3/mcy/GLANCE/results/official_kl_block_verify_20261016_223740/FINAL_BLOCK_VERIFICATION_OPTIMIZATION.md)。新增teacher_verify_impl=serial_reference|block经两个共同入口接入，默认serial，block在CLI限制为no-update诊断。loss/plan/Draft scoring/生产tree未变。24项CPU测试（含100组边界子案例）通过。

真实GPU0首条冻结题train:69，第13轮d14从token11变13，位于rejected suffix；此前12轮同proposal/teacher top1、全logits最大误差6.8665e-5。到失败轮cache长度、绝对位置与三轴M-RoPE一致，但不能宣称完整语义等价。按硬门禁停止其余五题、cap/重复性、block梯度对拍和profiling；无吞吐或新9小时GO结论。疑似浮点context差异影响bf16 Draft提案，尚未捕获head margin确认根因。

serial首题前反向完成，Target无梯度；未创建optimizer，0更新。五份权重文件前后SHA与前次L2_gpu一致，Head内存hash不变。tmux glance-block-223740已受控退出1，GPU释放，勿自动重启。旧预算两样本外推不是稳定吞吐，后续预算须等正确性与同workload实测。恢复快照/diff和全部失败logits保存在本run，不覆盖历史资产。

## 块验证精确分叉与局部修复（2026-10-16）

[完整报告](/mnt/local3/mcy/GLANCE/results/official_kl_divergence_20261016_225216/FINAL_DIVERGENCE_REVIEW.md)。同prefix/root/cache固定对拍排除位置shift、mask范围和crop行保留错误；无past完整前缀、带past块验证的Teacher top1和commit一致。原始第1轮Target块形状fp32差异进入committed fused，bf16量化后差异累计至第13轮578个坐标；仅交换冻结fused即可逐位重现Head d14由11翻13，margin由+0.005127变−0.010603。

局部修复只在block路径中串行重建keep行已提交KV/fused，完整rejected block评分、loss/anchors/精度不变，serial代码保留。25项CPU通过；最小扰动回归旧代码失败、新代码通过。修复后train:69全部74 tokens/33轮/34anchors/74response+455rejected位置与Teacher top1一致，loss差7.75e-7。整体梯度cosine0.9999885、relative L2 0.004792，但layers.2的input/post_attention_layernorm.weight两张量违反原逐张量max差阈值，未放宽。其余样本、SPEC和测速因硬门禁未通过而不继续，block仍仅no-update。

复验期间原/mnt/local/mcy/GLANCE/checkpoints目录消失，用户确认删除并授权重建。原始梯度失败后收尾SHA因此再次报错；不得掩盖梯度失败或声称最终尝试完整核验前后权重。早期诊断及第一次修复复验有完整不更新SHA证据，所有阶段均无optimizer/0更新。GPU已释放，保留全部失败与局部修复补丁。已重建/mnt/local/mcy/GLANCE/checkpoints，36份同尺寸候选逐SHA均非原Head，未用别的权重替代。恢复结果见本run的head_recovery.json，不能把文本DFlash的新初始化冒充原图像warm Head。

## 新图像 Warm 重建与 ScienceQA 成对训练（2026-10-16 夜间）

用户已明确授权旧图像 Warm 丢失后重新训练新 Warm，并在它完成且门禁通过后从同一权重启动 ScienceQA continue-SFT 与官方 serial KL-OPD；本轮最多 8.5 小时。Warm 训练于 2026-10-17 07:29 CST 完成；其后无更新门禁、诊断和最终审计亦在 08:03:24 截止前完成，全程约 8.30 小时。当前结果请读 [`RUN_STATUS.json`](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/RUN_STATUS.json)、[`READ_ME_FIRST.md`](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/READ_ME_FIRST.md) 和[最终报告](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/reports/FINAL_SCRATCH_WARM_AND_OFFICIAL_KL.md)。上文“未训练”是各旧阶段的当时状态，不能用于否认本轮实际 Warm 更新。

新 Warm 使用本地文本 DFlash 预训练 Head 作为**新图像 Warm 的初始权重**，不是被删除的旧图像 Warm。原 16K COCO/TextVQA Target greedy 数据已核对，图片内容 SHA 与 ScienceQA 全 split 交集为零。第 1–4 步的真实 GPU 连续运行与 2+恢复+2 数值等价门禁通过，游标/RNG/监督计划/Adam 步数完全一致，参数存在已记录的微小 GPU 数值差，不宣称逐位相同。正式 Warm 已完成 2000 updates×8 rows，完整 step2000 Head SHA `bd16ebcff1c513d42b14d9b8d89d428d93973ff8620bb78eafa9689509616735`；Target 四个权重分片前后 SHA 一致，最终训练状态 CPU 独立加载与 Head/状态文件独立 SHA 复算通过。在原授权窗口的剩余时间内，新 Warm 真实 GPU 无更新 SPEC=AR 门禁 train4 为 4/4、dev8 为 8/8 逐 token 一致，dev8 τ=4.0897；固定锚点候选覆盖 top8_first=0.625、L_tree=1.375，不能与在线 τ 混算。新 Warm 上的官方 serial KL 真实 GPU 无更新前向/反向覆盖 4 条 train QA、304 个在线 anchor 计划，loss 4.5291、梯度范数 1.9129、optimizer_steps=0。该小样本不代表独立泛化或成对收益。评测 wrapper 曾因无扩展名输出与 `.json` 读取约定不符退出，原输出、失败日志和修复记录均保留于 run 的 audit，最终状态已按原始结果复核纠正。剩余时间不足以完成匹配 M1，ScienceQA SFT/OPD 各 0 updates，无成对收益结论；不得自动续训。OPD 固定 serial；block verifier 保持 HOLD。本轮结果、恢复命令和可恢复源码补丁都在独立 run 中，历史 checkpoint/结果未覆盖。

## ScienceQA 成对训练 GPU 等待（2026-10-17）

用户在上轮 8.5 小时窗口结束后明确要求继续，并授权先检查 GPU，不够则监控空闲后占用。GPU0 监控启动时仅 25168 MiB 空闲，有其他计算进程 PID 2438600；其余 GPU 均不满足约 40 GiB 模型门槛，因此**成对训练尚未启动**。独立 tmux `glance-sqa-pair-watch-20261017` 正运行一次性监控，见 `/mnt/local3/mcy/GLANCE/results/sqa_pair_gpu_watch_20261017_1115/WATCH_STATUS.json` 和 `watch.log`。每 30 秒检查 GPU0；要求无计算进程、空闲显存≥40960 MiB、利用率≤5%，相隔 5 秒复核，另要求 `/mnt/local2` 空闲≥120 GiB。最多等待 24 小时，满足后调用既有 `scripts/resume_pair_window.sh 30600`，仅开启一次最长 8.5 小时的 SFT/官方 serial KL-OPD 成对窗口；失败或等待超时安全退出。不干扰他人，不另起重复监控。新 Warm step2000 仍位于 `/mnt/local2`，其 Head SHA 与上文一致；SFT/OPD 截至监控启动均为 0 updates。原上文“不得自动续训”的旧窗口限制已由用户本次明确授权覆盖。

## ScienceQA 成对训练 GPU2 启动（2026-10-17）

用户明确指定使用 GPU2。原 GPU0 监控在资源空出时发起一次预检，但 `pair.sh` 将 `worker.log` 预先写入新训练输出目录，触发训练器“新目录必须为空”门禁；该次 0 updates，失败日志及状态已保存在 `/mnt/local3/mcy/GLANCE/results/sqa_pair_gpu_watch_20261017_1115/` 和 run 的 `audit/pair_first_attempt_recovery.json`。已把两臂初次 worker 日志移到 audit，并让输出 symlink 初始化可重复；不修改任何官方 KL 公式、数据日程或模型权重。随后 GPU0 被另一进程占用，重试被门禁拒绝；已停止 GPU0 重试监控，避免重复作业。

GPU2 为同型号 RTX A6000，启动前空闲 48652 MiB、无计算进程。仅把物理卡映射成训练进程内 `cuda:0`：`GLANCE_PHYSICAL_GPU=2`、`CUDA_VISIBLE_DEVICES=2`，两处空闲检查及收尾 GPU 记录也指向物理卡2。tmux `glance-sqa-pair-gpu2-20261017` 正运行，日志 `/mnt/local3/mcy/GLANCE/results/sqa_pair_gpu_watch_20261017_1115/gpu2.log` 与 run 的 `audit/sft_m1_worker.log`。重启仍沿用最初 8.5 小时窗口的截止 epoch `1792246406`，未重新计时。Model binding 通过；SFT step0 完整保存且前 3 个真实更新已在 `train/sft/log.jsonl` 落盘，峰值 allocated 约 38.9 GiB。OPD 尚未开始；只有匹配 M1 两臂及无损 dev 评测完成后才能称为成对结果。`RUN_STATUS.json` 的 `sft_updates` 是最近完整阶段标记，运行中即时进度应读 log.jsonl；block 路径仍 HOLD。

## ScienceQA 成对训练取消时限与自动续跑（2026-10-17）

用户明确取消此前 8.5 小时总时限，授权从相同 Warm 起点把 SFT/官方 serial KL-OPD 成对方案继续跑完。正在运行的 `glance-sqa-pair-gpu2-20261017` 进程启动时已固定旧截止 epoch，不能安全热改，且工作规则禁止覆盖运行中的 shell 脚本。独立 tmux `glance-sqa-pair-no-limit-20261017` 因此只等待旧 worker 完成；状态 `/mnt/local3/mcy/GLANCE/results/sqa_pair_gpu_watch_20261017_1115/UNBOUNDED_CONTINUATION_STATUS.json`，日志同目录 `no_limit.log`，源码同目录 `finish_pair_without_time_limit.py`。它在旧 worker 退出后验证状态、日志 update 游标、完整 checkpoint 三文件 SHA 和 warm SPEC=AR；仅在 GPU2 无其他计算进程、≥40960 MiB 空闲、盘剩余≥80 GiB 时加载模型。继续使用原 seed、日程、数据、lr、fp32 Target、bf16 Head＋fp32 master、官方 serial KL；不设新训练截止时间，目标 SFT/OPD 各 120 updates、30/60/120 等原有 checkpoint 和 dev8 逐 token SPEC=AR。若旧 worker 因正确性失败退出、日志超前 checkpoint 或无损门禁失败，续跑器 HOLD，不改写原日志/权重。当前 SFT 约22/60，OPD 0；即时进度以 `train/{sft,opd}/log.jsonl` 为准，不能据此宣称已有成对收益。不要另起重复续跑器。

2026-10-17 14:09 CST 增量：SFT M1 已完成 60/60 updates，`train/sft/step60` 原子发布并含 SHA256.json，`RUN_STATUS.json` 进入 OPD_M1。实际 OPD 进程仍以相同 Warm step2000 为 `--init-from`，显式 `--teacher-verify-impl serial_reference`；首个真实更新已写入 `train/opd/log.jsonl`，`optimizer_steps=1`、loss=4.529111756、gradient_norm=1.912909，4 条同日程 QA，300 个 serial rollout rounds。按 `runtime.py` 的调用链，4 QA 的 Target model 前向次数可由每题 `2+16×rounds` 推得 4808 次，属源码与日志推算，不是 profiler 计数。OPD 后续仍运行，尚无匹配 M1 dev 结果；上段启动时进度已过时。当前进度只以实际 log/checkpoint/两条 tmux 进程为准。

2026-10-17 14:13 CST 增量：SFT M1 step60 已做独立只读 SHA 与 CPU 训练状态审计，见 run 的 `audit/sft_m1_checkpoint_independent_sha.json`、`audit/sft_m1_training_state_cpu_audit.json`；三文件 SHA 全部匹配，Head SHA 与 Warm 不同，schema/cursor/update=60，58 个 bf16 Head 张量对应 58 个 fp32 master 与 Adam 状态，Adam step 全为60，RNG/identity 齐备。OPD 当时为 2 updates，GPU2 原 worker 与无时限续跑器均存活。新增 `scripts/summarize_pair_compute.py` 只读日志生成 `audit/PAIR_COMPUTE_SUMMARY.json`；最终匹配门禁已补充两臂训练 QA 顺序和各自 SPEC=独立 AR 检查。公式 `2×QA+16×serial rounds` 仅为 Target model 前向调用链推算，非 profiler 实测。该状态未达到匹配 M1，勿报告收益。

2026-10-17 14:35 CST 增量：本轮用户取消 8.5 小时限制后的正式成对训练仍在物理 GPU2 上运行，tmux `glance-sqa-pair-gpu2-20261017`；SFT M1 已完成 60 updates，官方 serial KL-OPD 截至核对时完成 7 个真实 updates，尚未到 step30 checkpoint。无时限接续器 tmux `glance-sqa-pair-no-limit-20261017` 只等待旧 worker 结束后按完整 checkpoint 和 GPU/磁盘门禁续跑，目标两臂各 120 updates。当前原始进度以 `train/{sft,opd}/log.jsonl`、[RUN_STATUS](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/RUN_STATUS.json)及[接续状态](/mnt/local3/mcy/GLANCE/results/sqa_pair_gpu_watch_20261017_1115/UNBOUNDED_CONTINUATION_STATUS.json)为准；`RUN_STATUS` 的 OPD 更新数在现有 worker 内按完成里程碑更新，实时步数看日志。新审计证明 SFT step0 Head 与 Warm step2000 的实际 SHA 相同、OPD 第1步与该 Warm 无更新门禁同4题/300轮/304计划且 loss 完全相同；冻结120步日程有480个不同 train QA、474个图像组。接续器已加上两臂完整训练题目顺序、dev8 同题逐 token SPEC=AR、最终 Target SHA/交付哈希硬门禁；收尾报告区分 Warm 的 bf16 Target 与成对阶段的 fp32 Target，且会将软链 checkpoint SHA 清单纳入交付 manifest。上述只涉及审计、等待脚本和报告模板，活跃训练源码、官方 KL 配方、模型权重未改。当前仍**没有匹配 M1/M2 评测或收益结论**，不得另起重复 GPU 训练/接续进程。

2026-10-17 16:02 CST 增量：SFT 仍为已完成 M1 的 60 updates；官方 KL-OPD 至少完成 31 个真实 updates，当前 worker PID 2728968 与无时限接续器 PID 2880033 均存活，物理 GPU2 只有该训练进程。首个 OPD `train/opd/step30` 已从临时目录原子发布，三文件独立 SHA、CPU 加载的 schema/update/cursor=30、58 个 bf16 Head 与 fp32 master、58 组 Adam 两阶矩且步数全为 30、RNG/恢复身份均通过，见 run 的 `audit/opd_step30_independent_checkpoint_audit.json`。pool、冻结日程与共同 Warm 架构的 SHA 均与 checkpoint 恢复身份一致；配置明确为 on-policy＋serial_reference、accepted Bernoulli forward/rejected k3、decay 0.9、流权重 1:2，见 `audit/opd_step30_identity_and_recipe_audit.json`。前 30 步两臂 120 个训练 QA 顺序逐项等于冻结日程，见 `audit/opd_step30_paired_schedule_audit.json`；OPD 的 loss/梯度有限且优化器计数连续，6995 serial 轮、7115 在线 anchor 计划，见 `audit/formal_opd_first30_sanity.json`。Target 四个权重分片在该步后再次逐 SHA 匹配训练前，见 `audit/target_sha256_at_opd_step30.json`。这些只证明中途正确性与可恢复性，**尚非匹配 M1/M2 评测**；即时步数继续以日志为准，不得启动第二个 GPU 作业。快照：run 的 `audit/AGENTS_before_opd_step30_20261017.md`。

2026-10-17 M1 增量：SFT 与官方 serial KL-OPD 均完成 60 updates，同一冻结日程的前 240 个 QA 顺序、optimizer 计数逐项匹配；OPD 有 13865 轮 serial 验证和 14105 个在线 anchor 计划，见 run 的 `audit/opd_step60_paired_schedule_audit.json`。`train/opd/step60` 已原子提交。两臂在同一 dev8 上均 8/8 逐 token SPEC=独立 Target AR，SFT τ=4.1699，OPD τ=3.9906；这是小型阶段结果，不是独立泛化确认。原有时限的 tmux 已退出，M1 finalizer 将 `RUN_STATUS.json` 标记为 `MATCHED_M1`、成对比较有效、Target SHA 未变。用户已取消 8.5 小时限制；无时限接续 tmux `glance-sqa-pair-no-limit-20261017` 已独立校验来源与 checkpoint，正在物理 GPU2 上从 SFT step60 继续 M2，之后接 OPD step60→120 与同题 dev8 评测。接续状态见 `/mnt/local3/mcy/GLANCE/results/sqa_pair_gpu_watch_20261017_1115/UNBOUNDED_CONTINUATION_STATUS.json`；不得另起重复作业。M1 报告/状态快照在 run 的 `audit/m1_pre_m2_finalization/`，本文件修改前快照在 `audit/AGENTS_before_m1_20261017.md`。

## ScienceQA 新 Warm 成对训练完成（2026-10-17）

用户取消 8.5 小时上限后，物理 GPU2 上的持久续跑已正常结束，接续状态为 `COMPLETE`。唯一正式 run 仍为 `/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843`；[先读状态](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/READ_ME_FIRST.md)、[完整报告](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/reports/FINAL_SCRATCH_WARM_AND_OFFICIAL_KL.md)、[独立验收](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/audit/INDEPENDENT_FINAL_ACCEPTANCE.json)。新 Warm 完成 2000×8 原 COCO/TextVQA 行；SFT 与官方 serial KL-OPD 从同一新 Warm 独立完成各 120 updates×4 QA，共同冻结日程逐题一致。OPD 全程 27766 轮 serial 验证、28246 个在线 anchor 计划；Target model 前向 445216 次是源码调用链推算，不是 profiler 实测。Warm、两臂 step60/120 均有 SHA 和含 fp32 master/Adam/RNG 的完整恢复状态；Target 四分片前后 SHA 一致，官方训练源码冻结哈希一致。

M2 同一 dev8 两臂均 8/8 逐 token SPEC=独立 Target AR：SFT τ=4.2252（1274 tokens/302 rounds），OPD τ=4.3007（1274/296）；8 题来自 8 个图像组，逐题 OPD 高4、低3、同1。M1 在同一 dev8 上 OPD τ=3.9906、SFT τ=4.1699；因此这些只是小型开发集描述结果，不是独立泛化、可靠墙钟加速或旧 ScienceQA 负结果的同起点反证。test 未用于本次方法选择。19 项最终独立验收 PASS；block verifier 依旧 HOLD。旧图像 Warm 未恢复，新权重由本地文本 DFlash 初始化后训练，不得写成旧 Head 的逐位复刻。最终文件哈希见 `reports/DELIVERY_MANIFEST.json`，旧阶段状态以本节为准。
