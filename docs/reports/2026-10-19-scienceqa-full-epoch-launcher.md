# 当前执行规则：开发集早停，最终才检查 SPEC=AR

用户于 2026-10-19 最新覆盖此前门禁：不再执行 teacher/梯度阈值检查或连续2步 vs1+1恢复对照；旧失败报告保留，不标为通过。训练由用户执行现有入口 `--execute --resume`，本次修改没有自动启动。

- 两臂同一 Warm、QA 日程、lr、精度不变；最多一轮4,726 QA／1,182 updates，最后2 QA不重复。
- 每200步（及最后1182步）完整保存并跑 dev32 的 **SPEC-only** 生成，不生成 dev AR oracle、不检查 SPEC=AR。
- 用图像平均有效 token／验证轮早停：任一臂连续3次评测相对此前最好值提升不超过0.01，两臂在相同步数一起停止。最早step600；若继续改善则完成一轮。下降与平台期采用同一耐心机制，不因单个波动停训。
- 用共同停止点做最终新 confirm200，届时才生成独立 AR 并检查逐 token SPEC=AR；不各挑不同步数的最佳模型。最终不无损则照实报告，不能用不同输出宣称无损加速；开发集的覆盖仅是已生成SPEC前缀覆盖，不冒称AR正确前缀覆盖。
- GPU已有其他任务时等待空闲，不因资源阈值直接判实验失败；不杀其他任务。取消预留整轮磁盘容量的硬门槛。文件损坏、OOM、NaN、磁盘写入失败等不可执行状态仍需处理，不能承诺错误后继续有效训练。保留数据/权重身份、防覆盖、完整状态和更新账本。
- SFT所需冻结Target AR训练标签仍须生成；这是数据准备，不是训练中SPEC=AR检查。原诊断分支不再产生额外optimizer updates。
- `EARLY_STOP.json`保存最好值、无改善次数及历史，`ENDPOINT.json`保存实际停止步数和QA曝光量；早停不会报告为完成全量一轮。旧诊断成本仍计入实际成本。

CPU检查：早停的下降、平台、改善与重复恢复逻辑通过；模拟完整控制流程确认两臂相同步数/曝光量、没有启动诊断/开发集AR、只在最终产生确认集oracle。没有新GPU模型加载或训练。

以下保留初版交付与修正历史；与本节冲突的门禁、阶段和全量终点描述已被上述用户要求替代。

---

# ScienceQA 全量配对训练：用户启动脚本交付

状态：准备完成，未启动 GPU 模型、训练或评测。用户要求简短测试后交付自行运行；本轮执行 CPU 数据核查与代码快检，真实 GPU 门禁留在启动脚本中，失败即停止，不重试或放宽阈值。

入口：`repro/scienceqa_image/run_full_epoch.sh`。不带参数仅显示计划；`--background` 在 tmux `glance-sqa-full` 启动，`--execute` 前台运行。GPU 默认物理编号 `1,2,6,7`，可用 `--gpus` 指定四张空闲 A6000；不抢占其他任务，不使用 24 GiB 卡降精度。

```sh
bash /home/mcy/Projects/GLANCE/repro/scienceqa_image/run_full_epoch.sh --background
```

准备好的唯一 run：`/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_20261019/`。恢复对照分支：`/mnt/local3/mcy/GLANCE/results/scienceqa_full_epoch_20261019_recovery/`。数据、checkpoint、日志实际在挂载盘。历史产物只读。改动前可恢复快照：`/mnt/local/mcy/GLANCE/results/full_epoch_prepare_20261019_140257/source_before.tar.gz`。

## 冻结的数据与预算

重新读取 pinned ScienceQA parquet，核对源文件 SHA、图像可解码性，按图像字节及解码 RGB 像素审计跨 split 重复，排除 685 个跨 split 图像组；相同图像/问题/选项/答案的完全重复 QA 检查未发现额外重复。合法 train 为 **4,726 QA、4,352 张图像**。两臂读取同一 pool 和同一无重复日程，各 **1,182 updates**，最后 2 QA 只在 slot0/1 执行，slot2/3 参与同步但不产生重复曝光。

固定旧开发图像中的 dev32，每 200 步和最终点评测。新 confirm200 来自官方 test，排除本地所有发现的历史 ScienceQA manifest 与评测 CSV 中的图像，按固定 hash 选择；仅完整一轮的两个最终 checkpoint 评测。`DATA_AUDIT.json` 保存发现的历史文件列表与 SHA。这里只证明已发现项目历史与精确像素无交集，不声明语义近重复或基础模型预训练隔离。

同一新 Warm-2000 SHA `bd16ebcff1c513d42b14d9b8d89d428d93973ff8620bb78eafa9689509616735`；seed20261013，lr1e-5 constant、clip1、Target fp32、Head bf16、CPU fp32 master AdamW。SFT 保持 variable-reveal CE、6 anchors/QA；OPD 保持批准的 Bernoulli forward＋rejected k3、decay .9、流权重1:2，不冒称官方基础脚本默认配方。

正式每臂 1,182 updates；恢复对照另计每臂 2 次重复计算，合计最大 optimizer attempts **2,368**，独立 ledger，不扩 seed 或自动重试。通过门禁后，连续两步分支直接续训，正式轨迹始终仅一轮 QA 曝光。

## 精简启动门禁

1. 两条固定训练 QA、Warm 上一次 teacher 检查：独立完整因果重放相同前缀及 proposal，检查所有 teacher logp（含 rejected suffix）、接受标签、loss 与 Head 梯度。冻结工程阈值：teacher logp 最大绝对差≤.002，接受标签完全相同，loss 相对差≤.001，梯度相对 L2≤.01；必须实际覆盖 rejected 位置。失败写报告并停，不能把差异略过。此为小样本近似方法正确性门禁，不是 serial 轨迹/梯度等价证明。
2. 两臂各做连续2步与1＋重启＋1步，Head/master/Adam/RNG/游标和身份逐项精确对比。数值不一致也停止，不用历史恢复结果冒充通过。
3. 两臂 step2 固定 dev32 逐 token SPEC=独立 AR 必须零差异。两臂门禁均通过才继续第3步以后的正式训练。

AR 标签一次性由冻结 Target 生成，四卡分片缓存；这是 SFT 数据准备，可能耗时较长，不是重复验证。SFT 每题仍保留已有 prompt/Target replay 一致性断言。没有承诺整个任务的固定耗时。

## 输出、资源与恢复

完整 checkpoint 在 step2、200/400/600/800/1000/1182；恢复分支另存。启动时主盘预算约253.4 GiB（含原子写入与余量），诊断盘约88.1 GiB。交付时剩余约255.4/97.9 GiB，主盘较紧；脚本会重新检查，不自动删除旧产物。每个阶段记录命令、墙钟、分配卡数及 GPU-hours。OPD 记录 Target model forward 次数与输入 token；SFT 标签准备和 context forward 单列，投影计算不混称完整 Target forward。

每个评测同时用只读观察器记录 online anchor 上 top-k 正确前缀覆盖、实际树前缀覆盖，以及实际有效 token/验证轮、轮数和逐 token 无损；不改部署 decoder/tree。计时包含观察器开销，不用来宣称干净配对墙钟加速。主终点为新 confirm200 的最终 checkpoint 图像级配对有效 token/轮差，固定 seed bootstrap 95% CI。

完成后生成 `CURVES.csv`、`TRAINING_CURVES.jsonl`、`CURVES.png`（若 matplotlib 可用）、`FINAL_RESULT.json`、`FINAL_REPORT.md`、`DONE.json`。中间 checkpoint 正结果不改变终点。

查看进度：run 下 `CURRENT.json`、`logs/`、`LATEST.json`，或 `tmux attach -t glance-sqa-full`。希望更新边界保存停止时：

```sh
touch /mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_20261019/STOP
```

脚本也会在发现所用 GPU 出现其他进程时请求边界停止。停止后先看 `STOPPED.json`；确认资源恢复后移走自己创建的 STOP 标记，再运行同一入口 `--execute --resume`。只跳过成功阶段或从已完整 checkpoint 继续未消费的更新。错误门禁不自动重跑，ledger 已消费但未落盘的更新不自动重试；硬故障只能保证最近完整 checkpoint，不能保证最后一个 update 已保存。`RECOVER_sft.json` / `RECOVER_opd.json` 提供相同配置与 ledger 的续训命令；不允许绕过预算重新执行已消费更新。

本轮 CPU 检查涵盖官方 loss/梯度/池化/anchor 五项、新末批传输与冻结配对日程两项，以及语法、CLI 和预算构建。末批测试发现并修复无 rejected 项时分母返回 int 导致 `.hex()` 失败的边界问题，数值公式未变。GPU teacher、连续恢复与正式训练结果均尚未产生。

## 启动诊断修正（2026-10-19）

用户启动的首轮 teacher 检查在参考评分时触发 student replay 断言，未产生 optimizer update。原诊断错误地要求重算 fused 后的参考 argmax 与 native proposal 相同。v2 使用独立的 fixed_trace_score 对原 proposal 的真实参考概率评分，保留重算 hidden/fused，记录参考 proposal 分叉，继续比较完整 loss 与梯度；生产 scorer 的 current-policy 断言未改，teacher/label/loss/gradient 阈值未改。原失败日志及阶段记录保留，v2 用新阶段名 teacher_gate_v2，必须由用户重新运行；本次未启动 GPU 检查或训练。CPU 合成检查确认相同 context 下两 scorer 的 loss/梯度逐位相同，改变 context 后参考评分有限且记录 proposal 差异，生产 scorer 仍拒绝分叉。
