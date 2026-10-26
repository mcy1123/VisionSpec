# Draft OPD 官方 KL 唯一当前契约

截至 2026-10-16：**L2 CPU-only / 待真实模型门禁**。用户已批准基础脚本加显式方向覆盖；公共 loss、在线 rollout、图像/视频入口与 CPU 合成重放已接通。未运行真实模型训练，不是 L3，也不是 SGLang/Ray 完整训练栈复现。

## 来源与有效配置

用户指定基础脚本 `verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k.sh`。
本地 Git 根为 `/home/mcy/Projects/Draft-OPD`；`refs/Draft-OPD` 不是独立 Git 仓库，不能把在子目录执行 rev-parse 的输出当成独立仓库版本。

- 整仓库 HEAD：`d6f0ec859321959785c65b59eb311fd25f042158`。
- 基础脚本最后修改提交：`d1476f5d00a5fb187eb1637dc3f0f610ddbe1e1d`；此提交到 HEAD 的该脚本 diff 为空。
- 冻结参考目录中 18 个源码/配置/许可证文件与 HEAD 对应 blob 字节一致（包含新增全局计数与 scheduler 来源）。
- [逐文件 SHA256、函数行号和有效参数](/mnt/local3/mcy/GLANCE/results/official_kl_cleanup_20261016_131301/audit/official_source_manifest.json)。许可证：随冻结源码保存的 `verl/LICENSE`，Apache-2.0；不能丢弃原文件 copyright。

**新发现的基准矛盾**：基础脚本自身不设置 forward/reverse 权重。`ppo_trainer.yaml` 包含 distillation 配置，其默认 `forward_kl_weight=0`、`reverse_kl_weight=1`、`rejected_draft_use_reverse_kl=False`。`_combine_sampled_reverse_forward_losses` 的短路实际选择 response/rejected 两条流的 k3；不能声称该脚本默认 Bernoulli+k3。

`run_qwen_gsm8k_forward-ins.sh` 才显式覆盖 forward=1、reverse=0、rejected_use_reverse=True，同时其默认 decay=.8、rejected_weight=1。不能把 wrapper 的损失方向和基础脚本的 .9/2 超参拼成“基础脚本默认”。用户已明确回复“按照你建议的来”：采用 **基础脚本 + forward=1 / reverse=0 / rejected_use_reverse=True**，保留基础 decay=.9 和流权重1:2，准确称为显式覆盖配方。

基础脚本其余有效值：loss_mode=k3、accepted position decay 默认关闭、rejected decay=.9、response/rejected weights=1/2、first_token_only=False、response_anchor_stride=1、random_anchor=False、rollout temperature=0、teacher temperature=1、log_prob_min_clamp=-10、loss_max_clamp=10。注意 Bernoulli 函数的无配置 fallback=-80 不是这个脚本的有效 -10。

## 官方函数与位置计划

- `losses.py::_local_bernoulli_forward_kl`：两事件 forward KL，真实 teacher 概率参与，非 one-hot CE。
- `core_algos.py::kl_penalty/kl_penalty_forward`：k3 的 log-ratio clamp [-20,20]，结果 clamp [-10,10]。
- `losses.py::_build_response_position_weights/_build_rejected_draft_position_weights`：两套位置权重；`distillation_loss` 外层还有 loss clamp。
- `losses.py::distillation_loss`：跨 DP 的全局计数与 rejected 有效权重和；池化分子/分母，非分别求均值再加权。
- `dflash_student.py::_build_opd_anchor_plan`：从 rollout 的 reject indices 构造 response anchors/segments，包含 prompt_len-1 的起始位置、纠正边界和末端处理，stride=1；rejected anchors 还会补入 plan。不能简化成“只监督 n_acc 个位置”就宣称等价。
- `ray_trainer.py`：构造 rollout 参数和 first_token_only=False。完整 SGLang verification→record→teacher→response plan 的运行等价性仍未证明。

## 实际公共实现与计数契约

- `repro/opd_official_kl/loss.py`：Bernoulli forward 与 k3，teacher detach，外层 clamp，单一加权池化。公式由冻结源码 AST 抽取函数做数值/梯度逐位比较。
- `plan.py`：stride1/nonrandom response plan 与官方 `_build_opd_anchor_plan` 对拍；首个 response anchor 是 prompt_len-1，真实 verify 轮的末提交 token 标边界，**prefill 首 token 不额外伪造 reject 标记**。全接受轮也有 bonus 边界；response 流不等于仅 n_acc 个 draft token。
- `runtime.py`：当前 head 提案、当前 target 串行验证完整 proposal、仅保留已提交前缀 KV/fused hidden、EOS/cap 停止；teacher rejected log-prob 来自 draft-induced prefix，response teacher 来自正确提交轨迹。只有实际到达拒绝点才记录整个 rejected suffix，按官方行为其监督可超出输出 EOS/cap，不能误用旧 valid mask 截断。
- **分母是全局所有有效 response tokens + 2×rejected decay 权重和**，不是只数 anchor 覆盖位置。证据链：`workers/utils/padding.py` 的 loss_mask=response_mask → FSDP `forward_backward_batch` 的 batch_num_tokens → losses.py pooled branch。全接受时 plan 的 block 截断可漏掉某个 response 位置，但分母仍保留该 token。CPU 测试从真实 pooled branch 抽取 AST，另验跨 shard 求和和 DP 平均缩放；未启动分布式运行。
- `adapters.py`：图像沿用 common.encode + VisionPositions.ar_prefill_inputs（由 target 建立 rope_deltas）；视频沿用 videoencode 的 real/legacy 时间轴，不修改帧、生产 decoder/tree。两个 adapter 真正执行同一个 collect→rollout 与 score→pool；合成 target 上逐 token 对照独立 AR，并核对 rejected teacher 前缀。
- `trainer.py`：collect→score→loss→backward→fp32 master AdamW；每次 update 的所有 QA 分子除同一全局分母，逐 anchor backward；head 保持 eval 但启用 autograd，不依赖冻结 AR records。CLI 不接受 `--records`、固定 anchors 或旧 `--rejected-kl` 开关。

这是 GLANCE 串行适配：模型/视觉/fused 结构、显式 JSONL 日程、默认每 update 1 QA、fp32 target/bf16 head/master AdamW，与官方 batch24/accum10/epochs2 的分布式栈不同。lr 默认1e-5，cosine 按显式 updates 计算；不能把相同目标契约称为完整官方训练预算复现。max_seq_len 必须容纳完整 proposal，否则停止新轮；不暗自截短 draft block。无损与显存/吞吐的真实模型门禁仍待验证。

## 当前入口与历史隔离

```bash
/home/mcy/.conda/envs/fin_research/bin/python repro/scienceqa_image/train_sqa_opd.py --dry-run
/home/mcy/.conda/envs/fin_research/bin/python repro/train_video_official_kl.py --help
/home/mcy/.conda/envs/fin_research/bin/python repro/train_video_official_kl.py --dry-run
```

以上入口调用同一个 `repro/opd_official_kl/cli.py`，通过 trainer/adapters/runtime 接入公共 loss。dry-run 不加载模型；缺必要输入或携带旧目标开关退出2。旧 `opd_train_video_fullkl.py`、`qa_opd_fair/train_qa_opd.py`、`opd_train_video.py`、`opd_train.py` 直接执行也转入同一入口。历史 helper API 暂保留给诊断调用，不作为新训练 runtime；未证实无引用的函数不删除。

旧 CE+full-reverse、paper-full-KL、partial-k3 的原件保存在 `repro/legacy/pre20261016/repro/`。精确原命令需要审计备份恢复；显式历史 CLI 为：

```bash
python repro/legacy/replay.py --entry repro/opd_train_video_fullkl.py --acknowledge-historical-objective -- --help
```

历史 replay 的 --help 显示层会转义旧 argparse 文本中的裸 %（旧 full-KL help 原本报错），不改原始源码/训练语义。历史 replay 不能标为 official；它保留历史 loss 和恢复缺陷，仅供历史复现。旧 ScienceQA 生成器已退休，不得再次覆盖 active SFT/OPD。图像适配测试现明确对归档源码做 generator/reversal 检查，另检查 active generator 已退休；这些历史检查不认证新官方 runtime。当前新增门禁为 `repro/tests/test_official_cleanup.py`。

| 历史入口 | 实际目标 | 适用范围 |
|---|---|---|
| ScienceQA 2026-10-15 | 固定 AR anchors、accepted CE、rejected full reverse KL | 此配方的负结果有效，非官方方法否定 |
| Task A P1 | accepted full forward KL、rejected k3 | 部分对齐；P2/P3 没有运行 |
| 旧 paper-full-KL | full forward + full reverse、两流独立归一 | 历史论文式变体 |

## SFT 与恢复

ScienceQA/QA 的 active SFT 保留 variable-reveal CE，不改成 Bernoulli。`repro/shared/training_resume.py` 统一保存 schema、head/master/Adam、四类 RNG、update/cursor、日程内容 hash、数据文件 hash、模型配置和训练配置 hash。恢复用 checkpoint step，显式 CLI step 不符即拒绝；旧 checkpoint 无 schema 时拒绝冒充完整续训。

CPU 微型真实 MasterWeightAdamW 的 4 vs 2+2 对照已通过 loss/gradient/parameter/Adam 逐位检查。此证明不覆盖真实 GPU kernel；无真实模型更新，本轮不提供 L3 证明。所有 checkpoint 输出真实路径强制在 /mnt/local、/mnt/local2 或 /mnt/local3。

## 验证与限制

18 项 CPU 单测通过，覆盖官方公式/梯度、pool/全局分母、50组 anchor plan、verify EOS/cap、当前策略不同访问位置、逐token AR、两个 adapter 实际调用、恢复和指标回归。测试入口：

```bash
PYTHONPATH=repro:GLANCE_repo:scripts/dflash_pkg/dflash-0.1.0 /home/mcy/.conda/envs/fin_research/bin/python -m unittest tests.test_official_kl tests.test_official_cleanup -v
```

GPU no-update 未运行：8卡共享，采样时最大空闲35,278 MiB且利用率64%；空闲利用率为0的卡最多26,512 MiB，低于已知约40 GiB模型路径需求。不降精度或抢占来制造门禁。CPU-only 证明不包含真实图像/视频模型无损、GPU resume 或正式训练收益。

未来另行授权训练所需输入：`--pool` 为含唯一 task、prompt、image_path 或视频 metadata 的 parquet；`--schedule` 为 JSONL（零起始 update、slot、task），每 update 数量等于 qa-per-update；`--init-from` 为已有 head；`--out` 需空目录且真实路径在 /mnt。不读取历史 continuation。`--no-update` 最多执行首个 update 的 forward/backward，既不 step 也不保存权重。完整训练命令的配置通过 `--help` 查看，本轮未执行。

详细纠错见 [ScienceQA 实现与统计纠错](../reports/2026-10-16-scienceqa-implementation-and-metrics-errata.md)。

## 目录迁移

公共工具位于 `repro/shared/`；当前 OPD 使用 `shared/draft_geometry.py` 的草稿前向和EOS工具，SFT使用 `shared/sft_blocks.py`，不再通过历史loss文件获取工具。旧入口已移到 `repro/historical/`，旧命令路径见 `repro/layout.json`，可用 `python repro/run.py <旧文件名> -- <参数>` 解析；原始冻结报告未改写。可复用测试位于 `repro/tests/`。
