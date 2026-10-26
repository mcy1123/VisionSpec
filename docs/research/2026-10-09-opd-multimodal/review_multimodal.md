# 多模态迁移边界：系统文献独立交叉审查

审查日期：2026-10-09。审查者为负责多模态检索的独立子代理，未编写 papers_systems.json。本轮抽取 5 篇 P0/P1：DFlash、DDTree、DFlare、SharpDraft、FR-Spec；核对官方来源与 AGENTS.md，不声称全文复现或实现正确性审计。

结论：五条迁移建议作为后续假说可以保留；未发现须删除的虚构文献或将近似验证混入 exact 主线的重大问题。需要将“可以读/可以诊断”与“现在获准改训练”区分；没有一条构成更改冻结 Phase C 的授权。

| 文献 | 核实与可迁移部分 | 边界及建议 |
|---|---|---|
| [DFlash](https://arxiv.org/abs/2602.06036) | block 并行草稿是本地直接架构来源；核对 hidden 注入、mask 与位置语义合理。 | 文本模型结果无法证明视觉语义正确；尤其 cached/uncached 前向差异应以本地 aligned 门禁为准。任何新训练都要先验证视频确实影响输入 hidden。 |
| [DDTree](https://arxiv.org/abs/2604.12989) | 一次 block 分布导出候选树，与 b63k8 接口很接近；前缀闭合审计可立即借鉴。 | 独立位置边际乘积只是路径排序 surrogate，不能当真实 AR joint probability；视觉数字/实体上的相关错误可能放大此偏差。原记录已注明 surrogate 最优不等于 τ 最优，可保留。 |
| [DFlare](https://arxiv.org/abs/2606.02091) | 逐草稿层不同的 target feature 融合，是五层固定融合的自然结构对照。 | 论文也增加深度和训练数据，不能把全部收益归给 layer-wise fusion。若改变五层特征/缓存范围，则不是沿用 aligned step1000 的同构 continue training；必须新建结构分支，记录额外 activation/缓存内存。当前条目边界合格。 |
| [SharpDraft](https://arxiv.org/abs/2610.05106) | attention 稀释与 key 数关系可启发长视频长度分桶诊断；仅 draft query scaling 不必改变 target 分布。 | key 数不是视觉信息量；重复帧、跨模态 mRoPE 与 head 的有效 attention mask 会改变含义。应按实际可见 keys 而非原始帧数解释，并在 frozen checkpoint 上单独检验。不能假定长文本理论已证明视频可用。 |
| [FR-Spec](https://aclanthology.org/2025.acl-long.198/) | 大词表草稿投影可能是独立性能瓶颈；推理阶段限制 proposal 支持集与完整 target 验证可分开研究。 | OCR、罕见实体、时间戳与数字属于多模态长尾，频率词表可能适得其反。草稿 proposal 截断绝不等于 teacher/student KL top-k 重归一化；当前 B1 全词表目标必须保留。原条目已正确划界。 |

共同要求：诊断不能用 validation 持续选出新配置后仍称其 untouched test；不能用完整序列 τ、前 60 rounds τ、pooled τ 与 mean per-video τ 混排方法。无损应逐 token 验证，不用任务准确率“不降”代替。上表均为项目迁移推论，并非被论文或本地实验直接证明的效果。

补充元数据核实：2404.08856 已由[官方 CVPR ELVM workshop 条目](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Gagrani_On_Speculative_Decoding_for_Multimodal_Large_Language_Models_CVPRW_2024_paper.html)确认为 CVPR 2024 workshop，已修正 papers_multimodal.json，未标主会。PDF 直接 open 403，但官方搜索结果、workshop 索引和论文页相互一致。

