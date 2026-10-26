# OPD / 草稿训练文献检索记录

检索日期：2026-10-09。范围：重点2025–2026，补2023–2024的ICLR/ICML核心工作。工具：web search（web.run），随后打开arXiv、OpenReview PDF、PMLR、作者项目页与官方代码库。以首版日期统计年份，版本日期单列。条目见 `papers_opd.json`，共20篇，arXiv ID去重。

本文件记录检索与核实，不是系统综述；搜索引擎不暴露完整结果数，不虚构PRISMA命中总数。选择直接涉及draft训练、on-policy状态、拒绝监督、损失与训练效率的论文。其他无损解码系统由并行子任务整理。

## 实际查询

- `"draft" "on-policy distillation" speculative decoding`
- `speculative decoding acceptance aware training distillation 2026 arxiv`
- `speculative decoding online learning distillation 2025 arxiv`
- `site.arxiv.org speculative decoding "OnlineSPEC"`
- `site.arxiv.org speculative decoding "Aurora"`
- `site.arxiv.org speculative decoding "training" "2026" "distillation"`
- `site.arxiv.org "DistillSpec" "Online Speculative Decoding"`
- `site.arxiv.org "DistillSpec"`
- `site.arxiv.org "Online Speculative Decoding"`
- `site.arxiv.org "Training Language Models with On-Policy Distillation"`
- `site.arxiv.org speculative decoding "2026" "training" acceptance loss`
- `site.arxiv.org "Generalized Knowledge Distillation" "2023"`
- `site.arxiv.org "AdaSPEC"`
- `site.arxiv.org speculative decoding "2025" "Distillation" "Selective"`
- `site.arxiv.org "Recovering Off-Policy Supervision"`
- `site.arxiv.org "MiniLLM" "Knowledge Distillation"`
- `site.arxiv.org "DistiLLM" "2024"`
- `site.arxiv.org "DistiLLM-2"`
- `site.arxiv.org "Speculative Knowledge Distillation"`
- `site.arxiv.org speculative decoding "rejection" "2026" training`
- `site.arxiv.org "Attention Drift" speculative`
- `site.proceedings.mlr.press "When Drafts Evolve"`
- `site.proceedings.mlr.press "DistiLLM-2"`
- `site.openreview.net "On-Policy Distillation of Language Models" ICLR 2024`

## 核实方式与范围

20篇均打开arXiv正页并用正页citation metadata复核标题、作者次序、首版日期、当前版本；固定版本PDF地址写入JSON。方法主要核至官方摘要，VAT与Spec-AUF还读了HTML正文。未声称20篇均完成逐公式精读或独立复现。代码URL只记录论文、作者页及官方仓库明示链接，未猜测GitHub仓库。

Semantic Scholar API对`ARXIV:2605.29343`返回429 Too Many Requests，故本组未完成S2批量核实；使用arXiv ID与标题去重，来源标为官方正页核实，不能写作S2_VERIFIED。OpenReview部分forum页出现浏览器挑战/错误，会议证据使用可访问官方PDF或明确注明作者声明，不将under review当accepted。

证据类型为计算机科学方法论文的理论分析与作者控制实验；不套用医学RCT级别。所有收益是论文作者报告，不是GLANCE复现实验。未额外审计作者金融利益；作者对自己方法评测存在通常的研究者偏向，独立复现状态未知。

## 必须保留的纠错和边界

1. [Draft-OPD](https://arxiv.org/abs/2605.29343) 当前v3为2026-09-18。搜索摘要/官方仓库BibTeX把第二作者写为Yafy，arXiv正页与作者项目页为Yafu Li，采用正页。未据社交转述写EMNLP2026主会已录用。
2. [官方Draft-OPD仓库](https://github.com/Simplified-Reasoning/Draft-OPD) 已从bingyang-lei重定向。README的2026-08更新指出E2E TV loss + first-reject-only扩展；[脚本](https://github.com/Simplified-Reasoning/Draft-OPD/blob/feature/dflash-tv-loss/verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-8b-local.sh) 可访问。它是新的工程目标，不是原论文full-KL复现，不应自动替换Phase C。
3. [AdaFlash](https://arxiv.org/abs/2607.19223) v3为2026-09-26，作者注明COLM workshop spotlight，不能写COLM主会。
4. [VSD](https://arxiv.org/abs/2602.05774) v2曾withdrawn，目前v6为2026-08-12；不得简化为整篇仍撤回，也不得抹掉版本历史。当前摘要为Monte Carlo，旧搜索索引为MCMC；采用v6措辞。
5. [SelecTKD](https://arxiv.org/abs/2510.24021) v2已从SpecKD改名。它与[SKD](https://arxiv.org/abs/2410.11325)不是同一篇，也不能把SpecKD旧名再算一篇。
6. [MiniLLM](https://arxiv.org/abs/2306.08543) v6题名为On-Policy Distillation，ICLR2024正式旧题为Knowledge Distillation；同一arXiv ID只计一次。
7. [DistiLLM-2](https://arxiv.org/abs/2503.07067) arXiv写ICML2025 Spotlight，官方仓库写Oral；只采用ICML2025，不以展示级别排序。
8. [VAT官方仓库](https://github.com/naver-ai/VAT)明确2026-09-25 accepted to NeurIPS2026，但代码仅README且仍internal review。JSON标作者录用声明、代码待公开；不能说已有可运行实现。官方arXiv作者顺序以Geonmo Gu首作者为准，第三方索引顺序错误。

## 直接相关的相互竞争假设

以下只是待测借鉴，不修改既定Phase C：Draft-OPD学习拒绝后缀；Spec-AUF截断至首拒绝；VAT保留拒绝后的衰减监督；AdaSPEC/SelecTKD下调难学或拒绝token；EAL/WTV直接优化连续接受。它们对拒绝位置的处理并不一致，因此应先完成paper-faithful C2，再用同一起点和预算独立对照，避免把多种方法拼接后失去解释。

覆盖限制：本组包含大量最新arXiv论文及作者声明会议状态，不能宣称全是顶会主会或已独立验证；论文主要是文本LLM，VSD/ALR-IRA/SelecTKD/DistiLLM-2才明确涉及多模态。视频Qwen3-VL迁移收益均未知。

## 正式论文集补核

追加query：`"When Drafts Evolve" "proceedings.mlr.press/v306"`、`"When RL Meets Adaptive" "proceedings.mlr.press"`、`"DistiLLM-2" "proceedings.mlr.press/v267"`、`"DistiLLM: Towards" "proceedings.mlr.press/v235"`。

- [OnlineSPEC / ICML2026](https://proceedings.mlr.press/v306/qian26k.html)
- [Aurora / ICML2026](https://proceedings.mlr.press/v306/wang26j.html)
- [DistiLLM / ICML2024](https://proceedings.mlr.press/v235/ko24c.html)
- [DistiLLM-2 / ICML2025](https://proceedings.mlr.press/v267/ko25a.html)

以上均已将JSON从作者声明升级为正式PMLR证据。Aurora会议版作者14人与arXiv v5作者17人不同，JSON authors对应arXiv下载版本，不能直接拼成会议版BibTeX。
