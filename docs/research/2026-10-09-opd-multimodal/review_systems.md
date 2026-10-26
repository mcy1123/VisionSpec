# 独立复核：OPD 训练与迁移边界

2026-10-09。由系统文献支线独立审阅 `papers_opd.json` 前八篇、Spec-AUF、VAT，以及 `papers_context.json` 全部机制迁移建议。针对核心项重新打开官方 arXiv HTML 和官方代码 README；未改原 JSON、未复现实验。以下区分已证实的事实、应收紧的表述和研究边界。

## 需要修正或明确的表述

1. **Acceptance-Aware Draft Model Training（2609.24150）：已有树评测，但不等于推导精确覆盖 GLANCE 的确定性树。** 全文 §4/Table 1–6 明确同时评估 tree 与 chain，现卡“链式期望接受目标向 b63k8 树推广需重新定义”方向合理，但应补“原文已在 AR 草稿树上评测；本地 block 边际分布/确定性 Top-k 树仍需重验”。原文 EAL 用 `q(argmax p)` 作为单步接受概率；对随机从 q 提案、target greedy 验证成立，对固定 argmax 草稿实际接受为 0/1，对候选树则由路径覆盖事件决定。因此不得称为当前 GLANCE greedy tree 的精确 τ 目标。其 greedy 收益也远小于 sampling：Table 1 的 KL、KL+0.1EAL、加 GRPO 分别为 4.494、4.503、4.514，WTV 为 4.428。优先级 P0 可表示值得阅读，不能解读为最有把握的改进。[全文 §2.4–4](https://arxiv.org/html/2609.24150v1)

2. **Spec-AUF（2607.01893）必须包含首次失败位置。** “截断至首次失败处”容易被实现成排除 breaker，建议明确“接受前缀 + 首次失败位置保留 CE，之后置零；mask detach”。文中支持集是 `i≤j*`，不是 `i<j*`。全文也说该做法是 teacher-forcing 动机的近似，并不恢复真实 gold-prefix 条件。GLANCE 若用固定 b63k8，单一 argmax 链的首次失败不等于树首次无法覆盖 target，须单独诊断。[全文 §4](https://arxiv.org/html/2607.01893v1)

3. **论文条目存在性/title/version：本次复核前八篇官方页未发现标题或当前版本错配。** VSD 的 MLLM 线索是真实的：全文 F.2 为 LLaVA-1.5 7B/13B 图像任务，不是 Qwen3-VL 视频；其 AR 动态树配置也不等同 GLANCE 的 block 16 / b63k8。保留“额外采样与重新定义路径目标”的限制。[VSD v6 §F](https://arxiv.org/html/2602.05774v6)

## Draft-OPD 版本与目标核实

[官方仓库 README](https://github.com/Simplified-Reasoning/Draft-OPD) 2026-08 明确宣布 E2E TV + first-reject-only 的分支；这是真实后续代码方向，但 README 的性能主张不是本地视频验证。

与此同时，[Draft-OPD v3 全文 §4、附录 A/B](https://arxiv.org/html/2605.29343v3) 仍列 accepted forward KL 与 rejected reverse KL，并仅对 rejected suffix 做位置衰减。因此日期较新的 v3 不能自动与仓库 TV 分支画等号。论文公式语义、官方具体实现和本地预注册 C2 是三层证据；`paper-fullKL` 与 `official-TV-first-reject` 必须另立命名、配置及对照，不能覆写 B1 的既定目标。本地单流为空的处理及 update 级聚合仍应以本地已核实计划和测试为准，不以 README 推断。

## 边界提醒，非已证实错误

- **VAT**：全文的 verification head 是训练期辅助 BCE，累计 survival 标签在首拒后均为 0；重加权保留首拒位置完整权重，并对后缀继续衰减。这不同于 AUF 的后缀完全去除、不同于 Draft-OPD 的双 KL，也不同于 DSpark 推理时 confidence scheduler。现卡方向正确，但实现建议需保留以上区分。其训练期验证模拟不代表每个后缀 teacher 分布来自真实 draft-induced prefix。[VAT §4](https://arxiv.org/html/2608.30135v1)
- **TTS / OnlineSPEC / Aurora**：复用 verifier 信号不意味着 backward、optimizer、热切换与 ensemble 无成本；现卡已作基本限制。测试时更新还需声明请求间是否重置、请求顺序与数据状态，才能与静态 C0/C1/C2 公平比较。[TTS](https://arxiv.org/abs/2605.09329)、[OnlineSPEC](https://arxiv.org/abs/2603.12617)、[Aurora](https://arxiv.org/abs/2602.06932)
- **ALR/IRA**：离轨标签修复和 anchor 恢复值得诊断，但不是“目标模型失配必然是当前 A4 平台期主因”的证据。先核查本地训练来源及有效位置，再谈改动。[论文](https://arxiv.org/abs/2609.38795)
- **context 迁移建议总体合格**：已将一般任务奖励 OPD 与 draft τ 区分，HB-SJD 的图像生成与视频理解区分，IPD 的插值分布精确采样与原 target 无损区分。建议总报告再次强调 IPD 只能作为训练状态生成候选；B0.5 的 bf16/fp32 gradient cosine 不能解释为“梯度有助于 τ”。上述为本地研究推论，不应写成相关论文已经证明 GLANCE 收益。[IPD](https://arxiv.org/abs/2609.37170)、[HB-SJD](https://arxiv.org/abs/2608.18183)
- **避免数据纪律误读**：one-example OPD、teacher intervention、confidence weighting 均不构成减少 untouched test、修改当前 validation 选点或跳过同预算 SFT 的理由。现有卡片已明确，多篇材料合成时不可省略。

本轮未发现足以阻止文献库交付的伪造来源或确定性错误；建议先消除 EAL 的树评测遗漏和 AUF breaker 歧义，再将其作为方法筛选资料交付。此结论仅覆盖本文件声明的复核范围，不是对全部论文理论证明、会议录用或代码正确性的背书。
