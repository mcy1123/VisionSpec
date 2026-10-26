# GLANCE 当前交接（2026-10-19）

本文件只记录当前有效状态与工作规则。整理前全文见[本次归档](docs/archive/AGENTS-before-cleanup-20261019_135449.md)，切换前交接见[历史 AGENTS](docs/archive/AGENTS-before-pure-default-20261019.md)。历史文档中的“默认 serial”“尚未启动”“正在运行”和旧授权均是当时状态，不是当前指令；完整结果从[报告索引](docs/reports/README.md)查阅。

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
