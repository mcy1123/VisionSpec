# 系统、block 草稿与树验证检索记录

检索日期：2026-10-09。基于 AGENTS.md 的 aligned SFT 起点、尚未证明 OPD 胜过同预算 continue-SFT、固定 b63k8 和 fp32 AR 无损门禁开展。使用 web search 与官方网页 open；这是有目的的文献广搜，不是 PRISMA 系统综述，不声称覆盖所有论文。AI 辅助检索、去重与中文方法解读；未复现论文实验。

## 检索式与筛选

- `site.arxiv.org DFlash speculative decoding block diffusion`
- `site.arxiv.org DSpark speculative decoding`
- `site.arxiv.org EAGLE-3 Scaling up Inference Acceleration`
- `site.arxiv.org speculative decoding 2026 dynamic tree budget`
- `site.arxiv.org EAGLE EAGLE-2 Medusa speculative decoding`
- `site.arxiv.org Sequoia OPT-Tree speculative decoding`
- `site.arxiv.org SpecTr speculative decoding optimal transport`
- `site.arxiv.org speculative decoding 2025 2026 adaptive hardware efficient draft training memory`
- `site.arxiv.org "speculative" "2025" "SpecForge"`
- `site.arxiv.org "speculative decoding" "2026" "dynamic" "September"`
- `site.arxiv.org "Traversal Verification for Speculative Tree Decoding"`
- `site.arxiv.org "Yggdrasil" "Decoding"`
- `site.arxiv.org "FR-Spec"`
- 对 EAGLE/Medusa/Sequoia/SpecTr/Traversal/EAGLE3 追加标题全文与 `proceedings.mlr.press`、`proceedings.neurips.cc`、`neurips.cc`；对 OPT-Tree/FR-Spec/EAGLE2 使用 `aclanthology.org` 验证会刊。
- 从协同检索获得 D-Loop 2610.06011、SharpDraft 2610.05106、AdaSpark 2610.05774，再逐篇打开官方 arXiv 摘要页，核实首次提交日期分别为 2026-10-05、2026-10-04、2026-10-05。
- 从 DFlash 检索滚雪球发现 DFlare 2606.02091，官方页核实 2026-06-01 首发；从最新动态树检索发现 Tsubame 2609.33652，官方页核实 2026-09-27 首发。

收录 20 篇，逐篇详情和直接来源见 [papers_systems.json](papers_systems.json)。条目 `method` 为官方摘要/会刊摘要支持的简述；`glance_use` 明确属于研究推论；`priority` 是阅读与设计相关性，不是立即执行实验的授权。P0 表示当前机制/门禁/对照最相关，P1 为后续直接可借鉴，P2 为条件性扩展或基础背景。

## 核实与版本边界

已核实会刊：EAGLE、Medusa 为 ICML 2024；EAGLE2 为 EMNLP 2024；EAGLE3、Traversal Verification 为 NeurIPS 2025；Sequoia 为 NeurIPS 2024；SpecTr 为 NeurIPS 2023；FR-Spec 为 ACL 2025；OPT-Tree 为 TACL 2025（期刊，不冒称会议）。其余仅作为 arXiv 预印本收录，未把未核实录用的信息提升为顶会。

- [Sequoia arXiv](https://arxiv.org/abs/2402.12374) 标题含 Hardware-aware；[NeurIPS 正式版本](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html) 标题为 “Sequoia: Scalable and Robust Speculative Decoding”。保留 arXiv 标题并明确版本差异。
- [EAGLE2 会刊](https://aclanthology.org/2024.emnlp-main.422/) 与 arXiv 摘要的加速数值有版本差异；本库不混合引用这些数字，也不将其横向排名。
- [Yggdrasil OpenReview](https://openreview.net/forum?id=4E3I17pNEl) 访问返回 browser challenge；论文存在与方法已由 [arXiv](https://arxiv.org/abs/2512.23858) 核实，但未据此声称已录用。
- [SpecForge 官方仓库](https://github.com/sgl-project/SpecForge) 的 2026 年 DFlash/DSpark 新功能属于持续更新的工程资料；[2026-03 论文](https://arxiv.org/abs/2603.18567) 的摘要主要是 EAGLE3，二者不混称同一实验结果。
- 本支线只核实公开页存在及摘要级机制。PDF URL 指向官方全文；不代表已逐页精读或已下载。未核实官方代码的条目留空 `code_url`。

## 排除、交接与反证

AdaSPEC（2510.19779）是与 OPD 最相关的 selective KD 工作，交给训练文献支线，避免重复收录。同名多语言 ADASPEC 是另一篇 AAAI 2026 论文，未与之混同。SpecExit（2509.24248）通过 early exit 缩短推理文本，不能满足本项目原 AR 逐 token 一致的任务约束，未收录。MemSpec（2608.10362）侧重 edge 多 draft 装载调度，与当前固定单 draft/单卡关系较弱，未纳入核心 20 篇。

Tsubame 提醒：stochastic 场景中确定性动态树未必优于 sampled chain；D-Loop 需第二次 draft 前向；DSpark 的高并发收益不直接说明本地 batch-1 有益；FR-Spec 的推理词表裁剪不等于允许改 B1 的全词表 KL。以上均保留于论文卡限制中，避免只收集正面优化结论。

任何后续 tree budget、query scaling、feature fusion、block self-conditioning、词表/KV 稀疏化探索，都应在已冻结 Phase C 完成之后另立对照。当前研究资料不改 B0.5/B1/C 的预注册方案，不宣布新 GPU 实验完成。
