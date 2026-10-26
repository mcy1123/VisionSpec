# 四卡 rollout-only 的 opt-in 生产接入与小门禁

## 范围与状态

默认 `--parallel-rollout 1` 保留原单卡 serial；`--parallel-rollout 4` 必须通过 torchrun 四进程显式启动。只并行 rollout；rank0 原 slot0–3 / anchor 顺序逐 anchor BF16 backward、fp32 master sync、clip、AdamW、write-back。Target fp32、Head bf16、CPU fp32 master；官方 Bernoulli forward KL / rejected k3 / 官方池化 / decay=0.9 / 权重1:2不变。Block HOLD、top1关闭，不启用其他效率优化。

本轮仅9次 diagnostic 更新预算：step120起点原update119的1次 isolated；step90起点原日程90..93的4次连续；相同起点2次＋同拓扑恢复2次。先四卡生产入口无更新与旧 deterministic serial 对拍，失败不升级。沿用原 SDPA 的 deterministic 验证，不换 math kernel。不再跑3×3性能，不启动120步或第二seed。

当前：已建立可恢复源码快照；32项CPU回归通过。用户随后要求训练前清理存储，GPU门禁尚未启动。清理范围与GPU实际状态以本轮RUN_STATUS、BUDGET及storage_cleanup/RESULT为准。

## 实现与恢复边界

- `cli.py` 新增显式开关，默认值1；四卡仅接受4 QA、serial_reference、fp32 Target及CPU master。
- 原 serial loop保持，新增checkpoint保护，拒绝从diagnostic/four-rank状态静默恢复单卡。
- `parallel_trainer.py` 接收完整CPU Target trace，逐字段检查fp32 teacher logp、隐藏状态、proposal和身份；不传递student KV、logits或投影缓存。
- 原 `runtime.score` 在rank0完整重建student计算图；各rank rollout完成、精确全局分母确定、原slot/anchor顺序不变。
- 每update前检查Head hash/step/cursor及tensor版本；更新前后barrier，step后广播Head并校验全rank版本。
- `--initialize-from` 显式从原完整单卡状态初始化分布式；`--resume-from` 只接受同world size=4及相同slot映射、完整per-rank RNG和身份的checkpoint。初始化与恢复互斥。
- checkpoint只由rank0原子保存，拒绝覆盖；diagnostic_only随状态保存且不能被正式入口提升为正式step121。
- 隐藏的 `--diagnostic-gate` 仅做有预算的生产入口测试，逐步核对trace原始字节、58张梯度、selected logp、loss、分母、累积顺序及Head/master/Adam/per-rank RNG。

## 存储清理

用户授权训练前清除旧效果差状态。选择已关闭的历史全词表KL-OPD两个seed：只删step30/60的4份training_state.pt（optimizer/RNG），保留每步model.safetensors与config、最终step120完整状态、报告及数据。旧中间步将失去精确恢复能力，但推理权重保留。当前官方KL Warm、SFT/OPD step90/120及新回归依赖不在清理范围。删除前检查实际路径、所有者、硬链接、打开的fd和进程依赖，记录原状态与保留Head SHA256，留下逐checkpoint tombstone。原历史报告不改写。

本轮结果目录：/mnt/local3/mcy/GLANCE/results/opd_production_optin_20261019_100539
完整checkpoint目录：/mnt/local2/mcy/GLANCE/results/opd_production_optin_20261019_100539

## 判定规则

32项CPU通过不等于GPU放行。只有生产入口无更新、1次isolated、连续4步与2+恢复+2、dev8逐token SPEC=独立AR全部通过后，才标记“实验入口可用，默认未切换”。这只能证明本轮4步稳定，不证明120步正式复现，后者必须另批授权。

存储清理已完成：实际删除4份旧中间training_state.pt，释放54.6905406415 GiB；完整状态及保留Head的SHA在storage_cleanup/RESULT.json，四处目录均保留STATE_REMOVED_20261019.json。所有原始报告保持原文，清理后的历史中间精确恢复能力不再保留。


用户要求不再人工持续监控：训练/审计队列继续由tmux后台独立执行；`notify_done.py`仅在最终交付或中断后生成DONE.txt/DONE.json，并向当前用户已连接tmux客户端及可用桌面通知服务发送一次结果。设置时服务器没有桌面通知服务；不承诺自动发送聊天消息。脚本不训练、不重试、不加预算。

## 实际验收结果

- no_update：生产入口 PASS，原始字节trace、58张梯度、selected logp、分母与canonical backward顺序检查通过。
- isolated：生产入口 PASS，原始字节trace、58张梯度、selected logp、分母与canonical backward顺序检查通过。
- continuous：生产入口 PASS，原始字节trace、58张梯度、selected logp、分母与canonical backward顺序检查通过。
- split_first：生产入口 PASS，原始字节trace、58张梯度、selected logp、分母与canonical backward顺序检查通过。
- split_resume：生产入口 PASS，原始字节trace、58张梯度、selected logp、分母与canonical backward顺序检查通过。

最终状态：`COMPLETE_EXPERIMENTAL_OPT_IN_GO`，实际更新reserve/complete=9/9。

**实验入口可用，默认未切换。** 新入口9次更新与既有deterministic serial逐步精确一致，同拓扑2+resume+2与连续4步签名一致；dev8 SPEC=独立AR逐token通过。此结论只覆盖4步稳定性，不替代120步正式同seed复现，后者未授权、未启动。不进行新3×3计时。Block HOLD、top1关闭。

## 交付与使用

保存后的完整checkpoint已独立重新读取，对比isolated与旧serial的Head/master/Adam/RNG，以及连续与中断恢复最终checkpoint全部数值状态：`True`。完整原始字节审计见DIRECT_CHECKPOINT_AUDIT.json。

入口：`repro/scienceqa_image/train_sqa_opd.py`（共享canonical CLI/trainer）。`--parallel-rollout 1`为默认旧serial；显式四卡通过`python -m torch.distributed.run --standalone --nproc_per_node=4 -- repro/scienceqa_image/train_sqa_opd.py --parallel-rollout 4 ...`启动，实际UUID、全部参数和恢复命令见各`*_command.json`。`--initialize-from <完整原serial checkpoint目录>`仅用于显式初始化；`--resume-from <四卡checkpoint目录>`用于同world size恢复，二者互斥。checkpoint位于新输出目录，不能覆盖历史。

只读配方检查：`PYTHONDONTWRITEBYTECODE=1 /home/mcy/.conda/envs/fin_research/bin/python repro/scienceqa_image/train_sqa_opd.py --dry-run --parallel-rollout 4 --qa-per-update 4`。CPU命令见cpu_tests.log与测试文件。GPU门禁的每条完整命令已保存；本轮账本用完后不得直接重放真实更新，必须另批新预算。

本轮没有重新执行性能3×3或可选timing，不将此前诊断入口2.76×直接当作新生产入口实测；本轮结论是正确性/恢复接入通过。120步同seed复现、confirm200、切默认、新seed和长训均未执行。Block HOLD、top1关闭。

最终判定：`COMPLETE_EXPERIMENTAL_OPT_IN_GO`。源码差异和文件SHA见SOURCE_CHANGES.patch/MODIFIED_FILES.json，源码快照可恢复；大checkpoint复用canonical save_atomic_checkpoint在写完后读取生成的SHA256.json，交付时复查文件大小和sidecar，明确不同于再次独立读取每个大文件算SHA。推理权重和完整数值内容另有上面的独立加载审计。所有新checkpoint均diagnostic_only，不能作为正式step121。
