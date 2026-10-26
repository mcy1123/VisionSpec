# 来源与完整性核验报告

核验日期：2026-10-09。

| 检查 | 结果 |
|---|---:|
| 去重后条目 / 中文卡片 / BibTeX条目 | 68 / 68 / 68 |
| 官方arXiv标题与记录规范化后完全匹配 | 68 / 68 |
| 官方landing HTML及citation元数据归档 | 68 / 68 |
| PDF文件头、SHA-256校验通过 | 68 / 68 |
| pdftotext解析成功、非空文本 | 68 / 68 |
| 版本日期不晚于检索日 | 68 / 68 |
| Semantic Scholar成功二次元数据查询 | 2 / 68 |
| 生成目录与卡片本地链接 | 通过 |

Semantic Scholar受429限流影响，失败/跳过情况逐项保存在 `download_manifest.json`，已回退官方arXiv及web来源核验。未将API失败解释为文献不存在，也未声称全部获得独立数据库认证。标题匹配是存在性/元数据检查，不等于科学结果正确。

## 发表身份

- 官方主会/期刊：27。
- 官方Findings/Workshop：3。
- 仅作者声明相关会议身份：3（AdaFlash、Spec-LLaVA、VAT；均保留具体限定）。
- 按预印本记录：35。

期刊/会议身份依据逐项列在目录 `venue_evidence`；OpenReview作者列表与arXiv若有差异，卡片保留说明。优先以当前归档arXiv元数据生成BibTeX，不混用会刊作者表。并未逐篇审计所有公开审稿意见、利益冲突声明或代码。

## 独立交叉审阅与已处理问题

- [系统分支复核](review_systems.md)：检查OPD优先论文、上下文拓展。已补EAL的tree实证范围、surrogate边界及greedy收益限制；Spec-AUF明确包含首失败位置。
- [OPD分支复核](review_opd.md)：检查TIGER、ParallelVLM、HiViS、Sparrow。已补video prefill成本、两种aligned概念的区别。
- [多模态分支复核](review_multimodal.md)：检查DFlash、DDTree、DFlare、SharpDraft、FR-Spec。保留文本→视频迁移与draft/target压缩区别。
- 补充DFlash官方ICML2026论文集证据；DFlare代码指向Tencent/AngelSlim，避免误认同名旧项目。
- Draft-OPD TV扩展与原论文full-KL分开；TV分支固定提交记录于 `metadata/draft_opd_tv_branch.json`。
- GLANCE v1/v3、两篇SpecVLM、HiViS版本差异和SpecFLASH/SelecTKD/MiniLLM更名均保留说明。

交叉审阅是限定范围的AI独立分支复核，不是每篇论文的完整同行评审；没有宣称作者实验已被本地独立复现。

## 覆盖分布与局限

首次提交年份：2026年39篇、2025年16篇、2024年8篇、2023年5篇。2025–2026共55/68（80.9%），符合用户要求优先最新研究的范围，不据此宣称没有更早的重要工作。预印本与最新修订的结论可能变化。

文献建议主要来自摘要与优先项正文核查；下载全文不等于每页已经精读。各方法在GLANCE的效应、显存和耗时仍需实验，不把跨硬件/框架/数据集的速度倍数合并为排名。不更改现有Phase C配置、不启动训练。

本材料由AI辅助web search和本地脚本整理。可复查输入为 `papers_*.json`、`metadata/` 和各分支检索日志；输出为目录、论文卡片、原文与本报告。
