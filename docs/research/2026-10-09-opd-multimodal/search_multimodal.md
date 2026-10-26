# 多模态 / 视频投机解码检索记录

检索日：2026-10-09。执行 web search；本文件仅记录该分支，最终综合由主文档负责。输出 papers_multimodal.json 共 17 篇。优先 2025–2026，补一篇 2024 基础反例。范围是图像/视频理解输出文本的 speculative decoding；Vegas 为长文本 KV 系统迁移参考。未下载 PDF，pdf_url 是官方公开入口。

## 可复跑查询

- multimodal speculative decoding ViSpec MMSpec SpecVLM arxiv
- video speculative decoding HIPPO Vegas arxiv
- multimodal speculative decoding 2026 arxiv
- site:arxiv.org HIPPO speculative decoding video
- site:arxiv.org Vegas speculative decoding multimodal
- site:arxiv.org multimodal speculative decoding 2026 TIGER
- "SpecVLM" "Verifier-Guided"
- "multimodal" "speculative decoding" "2026" arxiv -site:reddit.com -survey
- "multimodal" "speculative decoding" "2025" arxiv MASSV STAR
- site:arxiv.org "ParallelVLM"
- site:arxiv.org "HiVis"
- site:arxiv.org "Loosely Speculative"
- "HiVis" "speculative decoding"
- "STAR" "speculative decoding" vision
- site:arxiv.org "Spec-LLaVA"
- site:arxiv.org "FLASH" "multimodal" "speculative"
- site:arxiv.org multimodal "speculative decoding" "2024"
- "Vegas: Self-Speculative" site:icml.cc
- "Speculative Decoding Reimagined" site:openaccess.thecvf.com
- "See the Forest for the Trees" site:aclanthology.org "Speculative"
- "Sparrow: Text-Anchored" arxiv
- "SpecVLM: Enhancing" site:aclanthology.org
- "Vegas: Self-Speculative" ICML 2026 poster
- "STAR: Speculative Decoding with Searchable" site:iclr.cc
- "Sparrow: Text-Anchored" site:aclanthology.org/2026
- "TIGER" "2607.11131" github
- "HIPPO" "2601.08273" github
- "STAR" "pMdKnxkFRw" ICLR 2026 accepted
- "Sparse-to-Dense: A Free Lunch" arxiv
- "Sparse-to-Dense: A Free Lunch" site:aclanthology.org

先广泛发现，再访问 arXiv、ACL Anthology、NeurIPS、CVF、PMLR 与作者官方代码。搜索引擎混合输出不提供稳定数据库总命中数，未虚构 PRISMA 数量。搜索片段有过期标题/作者，优先当前官方页面。未做付费数据库全量检索或逐篇全文公式复算，不宣称穷尽。

## 录用核实与版本差异

- [ViSpec 官方 NeurIPS 主会页](https://papers.neurips.cc/paper_files/paper/2025/hash/a7bfdee9544cea324cf183ac03c7d5c0-Abstract-Conference.html)明确 Main Conference Track。
- [SpecVLM 视频版](https://aclanthology.org/2025.emnlp-main.366/)是 EMNLP 2025 主会，arXiv 2508.16201；[图像 elastic-compressor 版](https://arxiv.org/abs/2509.11815)不同作者、不同代码、只标 arXiv。
- [HiViS v3](https://arxiv.org/abs/2509.23928)于 2026-08-23 修订，新增作者 Shuang Qiu；按 CVPR 2026 Findings 官方 PDF 核实，不升格主会。v1 与 v3 的训练表述变化已在条目注明。
- [SpecFLASH v3](https://arxiv.org/abs/2505.12728)于 2026-02-03 改题；旧 FLASH 标题是同一条，不重复计数，不等于 DFlash。
- [MASSV](https://aclanthology.org/2025.findings-emnlp.656/)是 EMNLP Findings。
- [Sparrow](https://aclanthology.org/2026.acl-long.450/)是 ACL 2026 主会。ACL 元数据第三作者写 Hongwanyang，arXiv 写 Wangyang Hong，库中保留 arXiv 拼写并显式说明冲突。
- [ParallelVLM 官方 CVPR PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kong_ParallelVLM_Lossless_Video-LLM_Acceleration_with_Visual_Alignment_Aware_Parallel_Speculative_CVPR_2026_paper.pdf)确认 CVPR 2026；HTML 入口返回 403，未把此访问失败误判为论文不存在。
- [Vegas](https://proceedings.mlr.press/v306/yue26a.html)由 PMLR 确认 ICML 2026，原研究为长文本，属于跨域迁移。
- [LVSpec](https://aclanthology.org/2026.acl-long.1087/)确认 ACL 2026；其 loosely verification 明确不满足本项目逐 token 无损要求，留作诊断启发与排除边界。
- [Sparse-to-Dense](https://aclanthology.org/2025.acl-short.59/)是 ACL 2025 short paper。
- Spec-LLaVA arXiv 作者备注 accepted by ICML TTODLer-FM workshop，未取得独立 workshop 录用证据，仍按 arXiv 收录且不可称 ICML 主会。
- [TIGER](https://arxiv.org/abs/2607.11131)明确 work in progress。其接受前缀奖励与 KL anchoring 是可读的新方向，不构成可覆盖现有预注册 full-KL C2 的依据。

## 排除、去重与证据限制

- STAR（OpenReview pMdKnxkFRw）：官方 PDF 可确认匿名在投稿件的方法摘要，但 forum 被浏览器验证拦截，作者与 ICLR 2026 录用尚未获官方确认；未收入核心 JSON，不能用第三方 Lacuna 的会议标签补足。后续可从[官方页面](https://openreview.net/forum?id=pMdKnxkFRw)继续核实。
- 2608.20743 多模态 diffusion survey 与 GLANCE 2609.00355 由主线程负责，不重复收录。
- Spec-VLA 等动作/图像生成 relaxed acceptance 工作不是图像视频理解的同构基线，本分支未扩张纳入。
- 搜索可见 2026-03 multimodal survey 为发现线索，核心方法事实回到论文官方摘要，不把综述/聚合站当独立复现证据。
- Semantic Scholar API 用 ARXIV:2603.14989 测试返回 HTTP 429；记录 [S2-API-UNAVAILABLE]，按标题+arXiv ID 去重并使用官方 web 来源核实，未声称 S2_VERIFIED。
- 本次是方法导向文献搜索，逐篇 method 为摘要及官方材料支持的事实；glance_use 明确为研究推断，未执行本地训练。无独立复现和完整全文 RoB 审计，作者实验可能存在平台/样本/调参偏倚。
- 发表状态只说明是否同行评议，不证明本地可复现性。17 篇中绝大多数发表于 2025–2026，符合用户最新文献目标，存在时间集中；所有方法多为作者自报实验，不能横向按峰值 speedup 排名。
- AI 辅助完成检索、核实与中文注释；没有复制论文全文。

## 与当前项目的接口边界

priority P0 为立即可借鉴的评测纪律；P1 为 Phase C 之后优先研读；P2 为后续实现/系统分支；P3 为基础反例或不符合 exact 主线的工作。所有条目都不能擅自改变 AGENTS.md 规定的 B0.5 → B1 → C0/C1/C2 顺序、单 seed 与冻结预算。视觉裁剪仅草稿侧可以与完整 target 验证相容；若裁剪 target 则基准模型改变。通过摘要声称 lossless 也不替代本地 fp32 AR token 门禁。

