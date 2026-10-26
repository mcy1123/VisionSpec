# 检索范围与可复查方法

检索日期：2026-10-09。目的：为本项目 aligned SFT → full-KL B1 → Phase C 提供可借鉴论文；这是一轮有目的的广泛文献检索，不宣称穷尽检索或 PRISMA 系统综述。

## 问题与范围

1. 如何减少草稿训练/实际验证的分布错位，并区分 rollout freshness、KL 方向、监督支持集和优化步长？
2. 对 greedy block/tree 解码，怎样让训练信号更接近连续接受长度，而不是仅降低平均 KL？
3. 视频视觉上下文如何影响草稿特征、位置编码、窗口、候选覆盖和实际耗时？
4. 怎样在保持 target 输出与无损门禁的前提下减少 draft/verify 成本？

以 2025–2026 为主体，补充 2023–2024 的重要基础论文。日期上限为检索日；区分首次投稿日期、最新版本日期与会议年份。特别补搜 2026 年 9 月和 10 月新作。包含 OPD/general KD 的相关理论与反证，但单列“可借鉴思想”与“直接适用于 speculative drafter”。

## 检索途径

按用户要求使用 web search。三个并行检索分支分别覆盖 OPD、多模态/视频、block/tree/system；主线程补最新论文、反证及版本核查。各分支的原始检索式及筛选说明见 `search_opd.md`、`search_multimodal.md`、`search_systems.md`。

主线程实际使用的检索式包括：

- `"Draft-OPD" "On-Policy"`
- `"GLANCE" "Vision Is Not Overhead"`
- `"speculative decoding" "on-policy distillation" 2026`
- `"multimodal" "speculative decoding" 2026 September`
- `site.arxiv.org "speculative decoding" "Oct" "2026" draft`
- `site.arxiv.org "multimodal" "speculative" "Sep" "2026"`
- `site.arxiv.org "on-policy distillation" "Sep" "2026" speculative`
- `site.arxiv.org "speculative decoding" "training" "September" "2026"`
- `"CA-OPD" "Structured Visual Prediction"`
- `"Rethinking On-Policy Distillation" "Phenomenology"`
- `"Rethinking On-Policy Distillation" site:icml.cc 2026`
- `"Rethinking On-Policy Distillation" site:openreview.net/forum`

搜索引擎返回的第三方汇总仅用作发现线索；方法与身份结论回到 arXiv、会议论文集、OpenReview、作者官方项目/仓库。没有依据搜索摘要中的“Published X months ago”推断提交日期。

## 纳入、排除与分级

- 纳入：可定位原论文，且与草稿训练、接受/验证、视觉/视频信息处理或训练成本至少一项相关。
- 排除：无法核实存在、标题与编号错配且不能消解、无明确方法关联的营销页面。作者代码仓库只能支持其实现/发布声明，不能充当独立复现。
- 会议身份区分：官方主会/期刊、官方 Findings/Workshop、作者自述录用、仅 arXiv。OpenReview“在审”不等于已录用。
- 相关性：P0=当前基线/直接训练或验证依据；P1=优先阅读的后续优化；P2=条件性或较高成本迁移；P3=历史或改变无损要求的对照。优先级不是效果保证。
- 把“作者报告的方法”与“对 GLANCE 的迁移推论”分开。将近似接受、target token pruning、图像生成 OPD 单独说明，不能混为原 target 无损视频理解。

## 来源与文件核验

对每篇保存官方 landing page 的 citation 元数据与 HTML 快照；PDF 验证文件头、记录 URL、固定可识别的 arXiv 版本、SHA-256，并用 pdftotext 检查可解析性。结果见 `download_manifest.json` 和 `metadata/`。下载脚本尝试 Semantic Scholar 二次元数据查询；限流/不可达时保留失败记录，以官方页面和 web 核查为准，不把缺索引判为论文不存在。

核实存在/下载成功不等于全文方法已复现：多数卡片为摘要级筛选，优先论文进一步读了正文/代码发布说明；独立复核见 `review_*.md`。本次不运行作者代码，也不估计本地收益。

## 版本与筛选中遇到的问题

- GLANCE 本地 `glance.txt` 明确为 v1；在线最新版为 v3（2026-10-06）。本地历史数字保持原记录，不能替换成新版摘要数字。
- 两篇 SpecVLM 是不同论文，须按 arXiv ID 区分。
- FLASH 当前标题为 SpecFLASH；SpecKD 当前标题为 SelecTKD；MiniLLM 最新 arXiv 标题与 ICLR 2024 会刊题名不同。均保留 ID 与版本来消歧。
- “Rethinking OPD 2604.13016 是 ICML 2026”在第三方页面出现，但本次找到的 OpenReview PDF 标的是 FoGen workshop 在审，故不列为已核实主会论文。
- HB-SJD 用于视觉图像生成 rollout，不是视频理解，降为间接借鉴。
- 不把不同硬件、引擎、精度、采样方式、输出长度下的 speedup 排成统一排行榜。

## 局限与工具披露

本材料由 AI 辅助 web 检索、整理并进行分支交叉审阅；并非人工逐篇复现报告。搜索覆盖受索引与访问条件限制，最新预印本结论可能变化。原始论文作者对自身方法的实验属于作者报告，未逐篇审计利益冲突声明、训练数据或统计代码。源身份核实与实验可信度是两个不同层级；本文不声称完成独立结果复现。
