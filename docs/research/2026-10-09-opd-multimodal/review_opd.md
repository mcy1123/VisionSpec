# 多模态建议交叉审查

审查：OPD文献子任务；日期2026-10-09。独立读取四篇arXiv当前版本摘要，并查看HTML方法/限制部分。审查对象为`papers_multimodal.json`里的方法、迁移建议、限制；不修改该文件。

| 论文 | 结论 | 需保留/补充的边界 |
|---|---|---|
| [TIGER v1](https://arxiv.org/html/2607.11131v1) | 现有建议准确 | 正文Table 1保留完整visual verifier；Table 2/4为辅助视觉接口质量-时延分析，不能当逐token无损证据。只在草稿侧路由、保留完整target验证才适合GLANCE。其group-based训练是另一个目标，不能替换B1后仍称原论文OPD。作者也称长视频仍困难，不宜过推。 |
| [ParallelVLM v2](https://arxiv.org/html/2603.19610v2) | 现有建议准确 | 正文Stage I明确target保留full-context KV、仅draft剪枝。它是training-free并行系统；提升τ的训练优化与并行速度收益要分开。必须单独测预填充、草稿、验证、并发资源，不能直接把论文大模型配对加速倍数用于当前单卡。 |
| [HiViS v3](https://arxiv.org/html/2509.23928v3) | 现有建议准确 | Algorithm 1 target先处理视觉与文本、draft只用融合文本特征；§3.3为step-dependent residual与多步模拟。这里aligned不是GLANCE cached路径修复的同义词；前者纠正独立草稿步语义偏差，后者是本地特征条件错误。建议固定v3 PDF。 |
| [Sparrow v1](https://arxiv.org/html/2602.15318v1) | 方法准确；建议补充端到端限制 | §3递归MTP用自生成hidden加固定视觉anchor，不等于Draft-OPD拒绝流。§6明确prefill仍是瓶颈，未来target视觉剪枝可能损失信息；不能因2.82x decoding就声称同倍端到端加速。严格token无损仍应由本地AR gate证明。 |

共同审查结论：四条已将迁移明确写作研究推断，保留Phase C后独立支线，未承诺实测收益；可通过。建议综合报告特别解释：**draft不显式读visual tokens不意味着target不读visual，也不意味着draft特征与visual无关**。这点与GLANCE旧uncached路径失效直接相关。

Sparrow HTML给出作者代码链接 [ddInference/Sparrow](https://github.com/ddInference/Sparrow)；可由主线程进一步检查可访问性和发布状态后补入目录，当前条目code_url=null并非证据已证明没有代码。
