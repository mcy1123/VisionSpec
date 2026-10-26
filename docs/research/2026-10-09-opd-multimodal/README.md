# GLANCE 优化文献库：OPD、多模态投机解码与 Block Drafting

检索截至 **2026-10-09**。围绕 [AGENTS.md](../../../AGENTS.md) 的 aligned SFT → B0.5/B1 → Phase C 整理 **68 篇论文，68 份本地 PDF**。其中 **55 篇首次发表于 2025–2026 年**，包括 2026 年 10 月的新作；另外保留关键早期基础工作。

**建议先读[实验借鉴与优先级](experiment_recommendations.md)，再查[论文总目录](catalog.md)。** 这里的优化建议尚未在 GLANCE 验证，不改变当前预注册训练目标。

## 文件入口

| 内容 | 入口 |
|---|---|
| 针对当前项目的诊断、候选实验、成本和限制 | [experiment_recommendations.md](experiment_recommendations.md) |
| 分类总表，逐篇链接中文卡片和原文 | [catalog.md](catalog.md) |
| 可用表格软件筛选的目录 | [catalog.csv](catalog.csv) |
| 完整结构化元数据、方法、迁移建议、限制 | [catalog.json](catalog.json) |
| arXiv 版本引用，可导入 Zotero 等 | [references.bib](references.bib) |
| 原论文与可搜索文本 | [pdfs/](pdfs/) · [texts/](texts/) |
| 中文论文卡片 | [notes/](notes/) |
| 来源分级、版本修正、下载与校验结果 | [verification_report.md](verification_report.md) |
| 检索范围、纳入标准、关键词及限制 | [search_strategy.md](search_strategy.md) |
| 来源HTML、citation元数据、PDF哈希 | [metadata/](metadata/) · [download_manifest.json](download_manifest.json) |

`references.bib` 引用本地归档的 arXiv 版本；若要正式引用会刊版本，使用对应卡片的官方论文集链接导出 BibTeX，避免将不同版本的作者/标题混在一起。

## 覆盖范围

| 主题 | 篇数 | 主要用途 |
|---|---:|---|
| OPD 与草稿训练 | 20 | KL/TV/接受长度目标、在线蒸馏、拒绝位置监督 |
| 多模态与视频投机解码 | 17 | 视觉特征、视频上下文、draft侧压缩、基准与无损边界 |
| Block 草稿、树验证与系统 | 20 | 并行草稿、块内条件化、动态树预算、实际延迟 |
| 基线、综述与 OPD 拓展 | 11 | GLANCE新版、反证、训练状态分布与跨领域借鉴 |

发表身份：**27 篇官方主会/期刊已核实，3 篇官方 Findings/Workshop，3 篇仅作者声明会议身份，35 篇按预印本记录。** 已核实来源涉及 ICML、ICLR、NeurIPS、CVPR、ACL、EMNLP、TACL；不把Workshop/Findings或作者声明升格为主会。

## 最贴近当前实验的阅读组

- **训练目标和首拒位置**：[Draft-OPD](notes/2605.29343.md)、[VAT](notes/2608.30135.md)、[Spec-AUF](notes/2607.01893.md)、[EAL/WTV](notes/2609.24150.md)。先核实定义，再比较监督范围和损失。
- **直接相关的新训练路线**：[AdaFlash](notes/2607.19223.md)、[ALR-IRA](notes/2609.38795.md)、[TIGER](notes/2607.11131.md)。分别关注在线蒸馏/长度、有效监督、accepted-prefix奖励。
- **视觉路径与视频瓶颈**：[Sparrow](notes/2602.15318.md)、[HiViS](notes/2509.23928.md)、[视频SpecVLM](notes/2508.16201.md)、[MMSpec](notes/2603.14989.md)。先诊断特征与长度，再考虑结构迁移。
- **10月新作**：[D-Loop](notes/2610.06011.md)、[SharpDraft](notes/2610.05106.md)、[AdaSpark](notes/2610.05774.md)。分别为块内条件化、长上下文query scaling、动态验证预算；均按预印本记录。

## 最重要的来源更新

1. **GLANCE 已更新至 v3（2026-10-06）**，本地旧论文为 v1；历史复现数字继续保留原版本口径。[版本记录](notes/2609.00355.md)
2. **Draft-OPD 官方发布了 TV loss + first-reject-only 扩展**，与原论文双KL目标分开记录，适合后续独立对照。[研究建议](experiment_recommendations.md)
3. 两篇 SpecVLM、FLASH/SpecFLASH、SpecKD/SelecTKD 等存在同名或更名问题，目录用 arXiv ID 消歧；HiViS 的 v1/v3 训练描述不同。
4. EAL 论文已测 tree，但 surrogate 不等于本地确定性block树的实际τ；Spec-AUF **包含**首次失败位置。独立复核后已修正文献卡片。

## 核验程度与维护

论文卡片多数是摘要级筛选；部分优先论文进一步核对正文和官方发布说明，未运行作者代码或复现实验。所有迁移建议明确属于研究推论，不承诺本地收益。研究报告由 AI 辅助 web search 整理，三个检索分支完成交叉审阅，详细限制见核验报告。

原始检索分支记录保留为 `papers_*.json` 和 `search_*.md`；`catalog.*` 与 `notes/` 由 `build_catalog.py` 统一生成，修改条目后重新生成，避免维护多份不同版本。

```bash
# 在本文件所在目录运行；本次下载已完成，通常无需重跑。
python fetch_library.py   # 获取缺失PDF；已完成项复用，非自动追新
python build_catalog.py  # 重建目录、卡片和BibTeX
```

本目录是本次检索快照，不是定时更新服务。下一次追新应明确新的截止日期，保留旧版本和结果哈希。本次未启动训练、未改模型或实验脚本。
