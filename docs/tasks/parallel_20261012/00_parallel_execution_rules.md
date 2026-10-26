# GLANCE 双方向并行长程探索：共同执行规则

> 两个任务独立运行，A 和 B 不需要等待对方的科学结论；但重型 GPU 任务必须串行。实际路径、命令、检查点和样本应以项目文件审计为准。将此文件和对应方向任务一起交给各 DeepSeek 会话。

## 0. 成功定义

**必须交付真实、可复查的结果**，不意味着必须找到正收益。有效负结果、正确性失败的定位结果、数据质量排除结果，都是结果。不能编造已运行实验、GPU 测量、显著性或成功。每个任务应自主完成审计、实现、测试、运行、报告；仅在确实缺少必需数据/计算资源或不可恢复的门禁失败时给出明确阻塞证据。不得停留在计划。

## 1. 身份、路径和边界

- 根目录 `/home/mcy/Projects/GLANCE`，产物默认 `/mnt/local/mcy/GLANCE`，但先从 `AGENTS.md`、`repro/common.py` 和当前 manifest 核实。
- **启动时完整读取根及适用子目录的 AGENTS.md**，以及 `docs/plans/video-opd.md`、`docs/reports/2026-10-12-gain-stuck.md`、`docs/reports/2026-10-12-current-ar-fixed-anchor.md`、各自所需原始 manifest/代码。聊天中路径、数字不是事实来源。
- C0 对应 aligned step1000；C1/C2 是 phaseC4 两个 step10。验证文件哈希。不得改模型参数、训练、生成训练数据、打开冻结的新测试集、覆盖历史实验。
- 对 C5 B-reference 头8轮和 C6 等距抽样的48轮**保持区分**，禁止拼池；旧 AR 与当前 fp32 AR 不一致，不能使用旧轨迹充作正确当前 continuation。
- 任何用于在线方法决策的信息必须是部署时可用数据，不能用 target 正确token、未来 token、oracle路径来选树或改策略。

## 2. 并行开发隔离

- A 独占目录：`repro/experiments/tree_c7_a/`、`docs/reports/2026-10-12-parallel-A-tree.md`、`results/parallel_A_tree/<独立时间戳>/`。
- B 独占目录：`repro/experiments/temporal_d4_b/`、`docs/reports/2026-10-12-parallel-B-temporal.md`、`results/parallel_B_temporal/<独立时间戳>/`。
- 如目录不存在可新建；先查是否与已有模块冲突。两个任务均**不修改**根 `AGENTS.md`、共享 `docs/plans/video-opd.md`、共同的 `repro/common.py`、`GLANCE_repo/glance/`、共享依赖，除非为修复真实阻塞，且能完全限定为兼容性的独立wrapper，并记录具体补丁。原则上把必要逻辑放在自己目录里通过导入调用。
- 根目录不是 Git 仓库，不依赖 Git 回滚；改动前把相关文件 SHA256、备份副本写入各自运行目录。禁止删除/覆盖历史或他人实验输出。
- A 与 B 各有自己的 `status.json`、`run_manifest.json`、原始日志及失败运行目录；结束前将**可粘贴到 AGENTS.md 的状态摘要**分别写入方向报告，而不是并行编辑 AGENTS.md。两者完成后由用户或单独汇总任务统一更新 AGENTS.md。

## 3. GPU 互斥（务必具体实施）

- 首先检查 `nvidia-smi`、GPU 使用/显存、运行进程、磁盘、RAM；不得杀死任何未知进程。优先轻量 CPU 审计与测试。
- 两方向统一用**同一台主机、同一个绝对路径**的锁，例如 `/mnt/local/mcy/GLANCE/.locks/glance_parallel_gpu.lock`。使用 `flock` 互斥；**GPU代码必须在持锁进程的生命周期内执行**，不能只锁启动语句随后后台放行。示意：`flock -x /mnt/local/mcy/GLANCE/.locks/glance_parallel_gpu.lock bash -c 'nvidia-smi; <完整 GPU 命令>'`；命令实际执行前再次检查所选设备剩余显存，绝不能把锁当作他人的 GPU 也空闲。
- 对外部使用者，锁不能提供保护；仍必须检查实际空闲 GPU。不能运行必然OOM的配置。有限合理重试（比如3轮、每轮几分钟），仍不可用就保存 CPU 结果、环境快照和阻塞证据并收口，不无限等。
- 明确设定 CUDA_VISIBLE_DEVICES 并记下 GPU 型号/设备；同一 GPU 上不同时运行 A/B 的模型任务。禁止 `pgrep -f` 自匹配轮询，记录真实PID/子进程退出码。
- A/B都需要单独控制自己的日志文件和原始产物，禁止在同一个文件上 append。

## 4. 结果质量与一致口径

- 对每个实验记录模型/配置/输入/源码 SHA256、执行命令、seed、环境版本、设备、原始输出、门禁、停止原因。
- `L_raw/L_tree/L_actual` 排除root和bonus；处理EOS/剩余长度。`τ` 的原有定义与这些局部长度分开报告；speed需要可比同卡/同精度/同warmup计时，不能把无插桩历史时间与插桩新时间相比。
- 48个anchors属于6条视频，视频是聚合/重采样单位。小样本CI只作探索性依据；CI跨零不等于等效。
- 结果至少有：完整原始CSV/JSON、按视频汇总、测试日志、可执行复现命令、失败列表、结论等级以及“下一步”。结论写明**观察到了什么/未观察到什么/无法判断什么**。

## 5. 最终汇总协议

各方向只更新独立报告，结束时在 `run_manifest.json` 包括：`direction`, `status`, `outcome_type`, `evidence_paths`, `tests_passed`, `ar_gate`, `limitations`, `next_step`, `requires_shared_update`。如完整实验不能运行，允许 `status=blocked` 但必须尽最大可能执行CPU与环境门禁并记录最小阻塞；不允许把没有做的实验写成已完成。
