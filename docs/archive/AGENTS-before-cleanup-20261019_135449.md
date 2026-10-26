# GLANCE 当前交接（2026-10-19）

以下保留切换前完整交接原文（2026-10-18）；归档副本仍在[切换前AGENTS](docs/archive/AGENTS-before-pure-default-20261019.md)，仅作备份。

## 两个主要目标已完成（2026-10-19）

- **Block verify**：默认已切为 pure block-native；每轮一次 block Target forward；部署 decoder/verifier 未改，dev8/confirm200 逐 token SPEC=独立 AR 通过。
- **多卡训练**：默认 4 卡并行 rollout；rank0 按 canonical slot/anchor 顺序 scoring/backward/optimizer；pure block-native 120 updates 已完成（4 卡，约 25.74 s/update）。
- 默认入口 `repro/scienceqa_image/train_sqa_opd.py`；旧 serial 与旧 1+keep 已归档，不混用；native checkpoint 为 diagnostic_only，不能当正式 step121。
- 不声明生产级稳定或等价性。

## 当前ScienceQA默认

默认 opd_mode=pure_block、parallel_rollout=4；active 默认不含 serial_reference 或旧 1+keep 重建分支。canonical 入口 `repro/scienceqa_image/train_sqa_opd.py`，旧 experimental 入口仅为同一 CLI 别名。

默认切换 smoke 已完成：1 次真实 update + dev8，attempt/complete=(1,1)，不能追加。checkpoint 为 smoke/diagnostic_only/block_native，不能当正式 step121；120-step native 实验已完成，但不声明生产级稳定或等价性。未来真实训练须用户另行明确授权。

部署decoder/verifier不改，SPEC=独立AR逐token零容差。loss仍为Bernoulli forward KL+rejected k3、原池化、decay0.9、流权重1:2，Target fp32/Head bf16/CPU fp32 master。prompt和原日程不变。旧exact block HOLD仅指已归档路径，不是新native的默认门禁。

## 产物、恢复与运行规则

快照/归档/日志：`/mnt/local3/mcy/GLANCE/results/scienceqa_pure_default_smoke_20261019_131714`。完整snapshot source_before.tar.gz，旧serial/1+keep独立归档archived_legacy/，SHA与changes.patch同目录；不覆盖历史权重/报告。完整新checkpoint实际在`/mnt/local3/mcy/GLANCE/checkpoints/scienceqa_pure_default_smoke_20261019_131714`。报告见[默认切换smoke](docs/reports/2026-10-19-scienceqa-pure-default-smoke.md)。恢复active代码：`/home/mcy/.conda/envs/fin_research/bin/python /mnt/local3/mcy/GLANCE/results/scienceqa_pure_default_smoke_20261019_131714/rollback.py`，恢复后不要消费native/smoke checkpoint到serial。

数据、缓存、权重、checkpoint与大临时文件必须实际在/mnt/local、/mnt/local2或/mnt/local3，写前核实realpath；home仅保留软链接，不替换为实体数据目录。代码根不是Git工作树，修改前建立可恢复快照；Python=/home/mcy/.conda/envs/fin_research/bin/python。不得破坏其他GPU任务，不删除历史checkpoint/报告。当前旧视频recipe也已随共享serial代码归档，视频入口明确失败并提示恢复，不静默切换。新报告登记docs/reports/README.md，不建立平行项目总结。

---

<!-- 以下为切换前完整交接原文（2026-10-18），保留以追溯历史状态；当前状态以上文为准。 -->

# GLANCE 当前交接（2026-10-18）

本文件只记录当前有效状态与工作规则。此前逐时进度、失败 trace 和旧授权的原文已原样存于 [2026-10-17 归档](docs/archive/AGENTS-through-20261017-m2.md)；更早记录见[历史 AGENTS](docs/archive/legacy_opd_pre20261016/AGENTS-before-cleanup.md)。归档里的“正在运行”“未训练”和待办均是当时状态，不是当前任务或新授权。

## 存储规则（必须遵守）

- 数据集、预处理数据、缓存、模型权重、checkpoint 和相关大临时文件必须实际存放在 `/mnt/local`、`/mnt/local2` 或 `/mnt/local3`；不得存入 `/home/mcy` 的实体目录。写入前核实目标路径及软链接的 `realpath`。
- home 可以保留指向上述挂载点的软链接，不得将其替换为真实数据目录。现有 `/home/mcy/local`、`local2`、`local3` 属于此类软链接。
- 默认 checkpoint 根为 `/mnt/local/mcy/GLANCE/checkpoints/`，但被删除的旧图像 Warm 未恢复；不要从目录已重建推断旧权重仍在。本轮视频 QA 数据在 `/mnt/local2/mcy/datasets/videoqa_transfer/`。

## 当前结果与可用入口

2026-10-19 pure_block_native 120更新实验已完成；部署dev8/复用confirm200逐token无损通过。真实账本120/120、没有额外更新/重试/第二seed。结果 `/mnt/local3/mcy/GLANCE/results/pure_block_native_20261019_110343`；完整checkpoint在STATE_ROOT/mnt/local2。native与serial允许训练差异、不是exact official KL；默认serial和旧block HOLD未改。完整效果差异与Warm观察审计见本轮报告/EFFECT_COMPARISON/TOKEN_AUDIT_SUMMARY，不自动宣称等价。


2026-10-19 用户明确“启动启动”：授权启动pure_block_native 120次更新与最多50条/1小时无更新token审计；不再要求native训练token/teacher/梯度与serial精确等价，但部署SPEC=独立AR仍逐token零容差。新的BUDGET上限120，与之前9/24/6完全分开。执行结果 `/mnt/local3/mcy/GLANCE/results/pure_block_native_20261019_110343`，完整checkpoint在STATE_ROOT所指/mnt/local2新train目录；GPU0/2/6/7按UUID。默认serial不变，旧block HOLD不放行；本轮为显式native近似变体、全部diagnostic_only/block_native，不作正式step121、不启第二seed、不自动追加或重试。代码冻结见PROTOCOL/execution_snapshot，运行期间不得改冻结模型文件。后台launch120.py完成120步后自动dev8、复用冻结confirm200、50条Warm对拍（排在训练后不抢训练卡）；结果/DONE见本轮目录，不再人工持续监控。


2026-10-19 新增显式 `repro/experimental/train_pure_block_native.py`，用户允许native训练token/teacher/梯度区别于serial，旧block HOLD仍适用于旧精确等价路径；native不是exact official KL。默认serial、部署decoder/无损验证不变。34项CPU与四卡Warm起点4QA零更新通过，optimizer更新0次，120步尚未启动。结果/快照 `/mnt/local3/mcy/GLANCE/results/pure_block_native_20261019_110343`，临时输出实际在 `/mnt/local2/mcy/GLANCE/results/pure_block_native_20261019_110343`。见[pure-native报告](docs/reports/2026-10-19-pure-block-native-entry.md)。新模式直接保留block KV/fused、不做1+keep重建；scalar loss/0.9/1:2/精度不变；native checkpoint强制diagnostic_only/block_native及独立身份，真实更新须有限预算。不再拿serial梯度精确门禁拦native实验；部署SPEC=独立AR仍逐token零容差。


2026-10-19 opt-in生产接入本轮状态 `COMPLETE_EXPERIMENTAL_OPT_IN_GO`；默认parallel_rollout=1，显式4卡路径由rank0 canonical scoring/backward/optimizer。9次预算reserve/complete=9/9，不可追加或重试。源码快照、门禁、清理审计及实际命令在 `/mnt/local3/mcy/GLANCE/results/opd_production_optin_20261019_100539`，完整checkpoint在 `/mnt/local2/mcy/GLANCE/results/opd_production_optin_20261019_100539`。见[生产接入报告](docs/reports/2026-10-19-opd-production-optin.md)。交付最终状态`COMPLETE_EXPERIMENTAL_OPT_IN_GO`，完整checkpoint独立数值审计见DIRECT_CHECKPOINT_AUDIT；实验入口可用性以本轮最终状态为准，不能宣称120步正式复现。旧负结果OPD四份中间optimizer/RNG状态已按授权删除，所有Head及最终完整状态保留；原checkpoint目录有tombstone。本轮所有状态diagnostic_only，不能当正式step121；没有授权或执行120步/第二seed，不重跑3×3。Block HOLD、top1关闭。

2026-10-18 并行rollout＋rank0 canonical scoring/backward/optimizer诊断已结束，结果目录 `/mnt/local3/mcy/GLANCE/results/opd_rollout_only_20261018_204925/`，完整状态 `/mnt/local/mcy/GLANCE/results/opd_rollout_only_20261018_204925/`。见[本轮报告](docs/reports/2026-10-18-opd-rollout-only-parallel.md)。本轮24次上限内reserve/complete为24/24，剩余额度不得自动追加或重试；此前6次仍已用完。决策GO，仅最小显式opt-in接入建议，默认未切换。新方案相对serial/anchor的配对中位speedup为2.7647/1.2500×。原第1组GPU1/2/6/7、后两组GPU0/2/6/7；只作组内配对，跨组非完全同卡。首组rollout完整结果及预算完成但缺调度器正常退出记录，保留此限制；最终交付脚本finalize_v6.py。实际门禁、字节审计、恢复与所有计时见RUN_STATUS及报告；全部产物diagnostic_only，不能当正式step121或第二seed。Block HOLD、top1关闭、原生产serial可用。原历史权重/报告保留，纠错均有独立记录；本轮模型进程已退出，不启动额外真实更新或长期训练。

2026-10-18 OPD效率工作当前采用独立协议：**deterministic原SDPA验证梯度/语义，default kernel测性能**；不再追默认kernel的10次max逐张量门禁。[前轮噪声诊断](docs/reports/2026-10-18-opd-backward-noise.md)已完成，原一次isolated update为diagnostic_only，不能当正式step121。

新一轮产物：`/mnt/local3/mcy/GLANCE/results/opd_exact_protocol_20261018_152931/`，见[确定性分布式协议报告](docs/reports/2026-10-18-opd-exact-distributed-protocol.md)。两卡/四卡原qa_fp32协议的各QA及固定slot传输均与单卡qa_fp32 reference逐位一致，但与legacy的累积算法仍不兼容；新增 **anchor_ordered_bf16** 按原slot/anchor次序在rank0 GPU bf16累加，两卡/四卡固定trace和live rollout的58张梯度均与deterministic legacy逐位一致。全局denominator逐slot精确汇总，teacher fp32 logp保留；生产trainer未切换默认路径。

Block组件诊断确认committed hidden/fused一致，rejected teacher logp差异驱动梯度分叉：换回serial teacher logp可使梯度逐位一致；单行LM-head投影仍未通过原/严格逐张量门槛。**Block继续HOLD**，不把新step120回归当作历史原Head复现。

本轮用户授权的**6次isolated full update已全部完成并用完授权**：deterministic/default的单/2/4卡各1次，每次重载原完整step120、复用原update119。deterministic两卡/四卡的Head、fp32 master、Adam及各rank RNG与单卡逐字节一致。default完整计时初测为201.409/141.817/95.463秒（约1/1.420/2.110×），仅每拓扑一次且warmup不对称，未达正式重复性能验收。状态及原始结果在 `/mnt/local2/mcy/GLANCE/results/opd_exact_full_20261018_152931/`，全部diagnostic_only，不能当正式step121或恢复训练。serial top1三组无更新配对也已完成，speedup为1.029443/1.001455/0.999077，中位1.001455，未达预注册>1.05且全快门槛，默认保持关闭。33项CPU测试通过，所有实验进程已退出、所选GPU已释放。候选未正式放行；不得自动执行额外optimizer更新。


2026-10-18 新确认集无更新评测已完成：`/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/`。[RUN_STATUS](/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/RUN_STATUS.json) 为 `COMPLETE/PASS`，[最终报告](/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/reports/FINAL_CONFIRM200.md)和[独立复核](/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/audit/INDEPENDENT_RESULT_RECHECK.json)已生成，tmux 已退出、GPU 已释放。三份 Head、实际配置、共同训练日程、评测源码 SHA/副本及预注册指标、官方 validation 200 道未参与开发图像题名单均先于新模型评测冻结；名单与此前开发题、精确图像 SHA 簇无重叠。I0/SFT-120/OPD-120 各 200/200 逐 token `SPEC=独立 AR`，权重哈希未变。预注册主指标逐图像有效 token/轮分别为 4.2267/4.3708/4.4457；OPD−SFT 配对均值 +0.0749，95% 图像簇 bootstrap CI [+0.0444,+0.1063]，正/负/同 101/53/46；同为 33,087 tokens，三臂验证轮 7973/7769/7690。全题答案均 153/200，因为输出与 Target AR 相同；不能将 parsed-only 比例冒充全题准确率。三臂计时均受共享负载污染，不报告可靠墙钟加速。原交付 manifest 只有最后写入的 `runner.log` SHA 过早记录；原件保留，[纠正清单](/mnt/local3/mcy/GLANCE/results/sqa_confirm200_20261017_223405/reports/DELIVERY_MANIFEST_CORRECTED.json)和独立复核记录了唯一差异。此确认集只代表固定的 200 图像子集，非全 validation/test。

唯一已完成的新 ScienceQA 成对 run：`/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/`。[先读状态](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/READ_ME_FIRST.md)、[完整报告](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/reports/FINAL_SCRATCH_WARM_AND_OFFICIAL_KL.md)、[19 项独立验收](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/audit/INDEPENDENT_FINAL_ACCEPTANCE.json)和[交付清单](/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/reports/DELIVERY_MANIFEST.json)为权威证据。`RUN_STATUS.json` 为 `M2_COMPLETE / MATCHED_M2`；持久训练、评测、收尾进程与 tmux 均已退出，恢复脚本报告无需续跑。

旧图像 Warm 原 SHA `fc0ad5a6…` 在有限索引检查中未找到。新图像 Warm **不是旧权重的恢复**：从本地文本 DFlash 预训练 Head 出发，用原 COCO/TextVQA Target-greedy 图像数据完成 2000 updates × 8 行，Head SHA 为 `bd16ebcff1c513d42b14d9b8d89d428d93973ff8620bb78eafa9689509616735`。Warm 图像内容与 ScienceQA 全 split 的 SHA 交集为零；真实 GPU 4 vs 2+恢复+2 数值等价门禁、train4/dev8 的逐 token `SPEC=独立 AR` 和官方 KL 无更新前反向均通过。最早完整 RNG 恢复点是 step2；没有单独的 step0 RNG dump，不能声称它存在。Warm Target 用 bf16；成对阶段 Target 用 fp32。

ScienceQA continue-SFT 与官方 KL-OPD 从**同一新 Warm**独立出发，冻结同一 480 道 train QA / 474 图像组日程、lr=1e-5、每步 4 QA，均完成 120 updates。Head bf16、CPU fp32 master/Adam；各阶段完整 checkpoint 含 Head、master、Adam、RNG、游标和身份信息，实际位于 `/mnt/local2`。Target 四个权重分片前后 SHA 一致；正式训练源码哈希未变。OPD 使用 `serial_reference` 的真实 on-policy rollout，accepted Bernoulli forward KL、rejected token-level k3、官方池化、decay 0.9 和流权重 1:2；全程 27,766 轮 serial 验证、28,246 个在线 anchor 计划。Target model 前向 445,216 次是源码与日志**推算**，不是 profiler 实测或等算力结论。

同一 dev8 上，两臂在 M1/M2 均为 8/8 逐 token `SPEC=独立 AR`：M1 SFT τ=4.1699、OPD τ=3.9906；M2 SFT τ=4.2252（1,274 tokens/302 rounds）、OPD τ=4.3007（1,274/296）。I0 与两臂 M1/M2 的全题答案正确数均为 6/8。M2 八题来自八个图像组，OPD 逐题高4、低3、同1。这是已参与开发的小 dev 集描述；独立确认以本文件上方的冻结 validation 200 图像结果为准。此次未用历史 test 选方法。

## 方法、门禁与历史边界

方法以[官方 KL 契约](docs/methods/draft_opd_official_kl_contract.md)为准。指定 Draft-OPD 基础 `run_qwen_gsm8k.sh` 在 HEAD `d6f0ec859321959785c65b59eb311fd25f042158` 中最后修改于 `d1476f5d00a5fb187eb1637dc3f0f610ddbe1e1d`；其实际默认 forward=0/reverse=1/rejected_use_reverse=False（两流 k3）。本项目的 Bernoulli＋k3 是用户批准的**显式方向覆盖**：forward=1、reverse=0、rejected_use_reverse=True，不得称为基础脚本默认。ScienceQA 与视频 OPD 的 CLI 均接入同一 canonical trainer→adapter→runtime→loss；可用 `python repro/scienceqa_image/train_sqa_opd.py --dry-run` 和 `python repro/train_video_official_kl.py --dry-run` 只读检查配方。旧入口转入公共实现，历史源码在 `repro/legacy/pre20261016/`，重放须显式声明历史目标。

**block verifier 仍 HOLD，只能用于 no-update 诊断，正式训练保持 serial。** train:69 的提交前缀分叉已局部修复，但两张 Head layernorm 梯度未达到原逐张量阈值；不得因新 serial 训练成功而放行 block。证据见[分叉报告](/mnt/local3/mcy/GLANCE/results/official_kl_divergence_20261016_225216/FINAL_DIVERGENCE_REVIEW.md)。

历史 ScienceQA 的目标是 fixed-AR-anchor CE＋full-reverse-KL，不是本次官方 online KL；旧负结果及[统计纠错](docs/reports/2026-10-16-scienceqa-implementation-and-metrics-errata.md)仍有效，但不是同起点消融。Task A/P1 只换了 rejected k3，accepted 仍是全词表 forward KL。旧结果、报告和 checkpoint 不得改写；详见[报告索引](docs/reports/README.md)。

## 工作规则

- 代码根 `/home/mcy/Projects/GLANCE` **不是 Git 工作树**；改代码或文档前建立可恢复快照。现有 Python 为 `/home/mcy/.conda/envs/fin_research/bin/python`。
- 不删除历史 checkpoint、原始结果或报告快照，不覆盖运行中的脚本；外部 Draft-OPD 与 refs 只读。当前 run 已完成，**任何新增真实训练仍需用户另行明确授权**。此前取消 8.5 小时限制仅适用于已完成的这次 run。
- lr≤1e-4 使用 fp32 master，逐项记录 Target/context 精度、数据预算和优化器；不得改生产 decoder/tree 或权重迎合指标。`SPEC=独立 AR` 必须逐 token 一致。
- 新报告加入 `docs/reports/README.md`；不另建平行“最新项目总结”。当前方法契约优先于历史计划，不自动恢复 C7、D4、新种子或追加训练。


用户要求不再人工持续监控：训练/审计队列继续由tmux后台独立执行；`notify_done.py`仅在最终交付或中断后生成DONE.txt/DONE.json，并向当前用户已连接tmux客户端及可用桌面通知服务发送一次结果。设置时服务器没有桌面通知服务；不承诺自动发送聊天消息。脚本不训练、不重试、不加预算。
