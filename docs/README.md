# 文档索引

日常交接和进度更新只维护根目录 [AGENTS.md](../AGENTS.md)；运行入口见 [repro/README.md](../repro/README.md)。本目录保存详细计划、证据和参考资料。

## 当前工作

- [官方 KL 唯一契约](methods/draft_opd_official_kl_contract.md)：目前 L2 CPU-only / 待模型门禁，本轮未训练。
- [ScienceQA 实现与统计纠错](reports/2026-10-16-scienceqa-implementation-and-metrics-errata.md)：负结果保留，错误机制解释撤回。
- [历史计划与教程归档](archive/legacy_opd_pre20261016/README.md)：不再作为活跃训练方案。

## 实验历史

完整列表见[按日期排列的报告索引](reports/README.md)。

- [图像复现、Markov 与早期视频记录](reports/2026-09-25-reproduction-history.md)：原 repro README，含已被后续审计替代的历史表述。
- [阶段 A legacy SFT](reports/2026-10-09-phase-a.md)。
- [旧视频 OPD 2×2](reports/2026-10-08-video-opd-2x2.md)。
- [EAGLE3 阶段 0](reports/2026-09-28-eagle3-stage0.md)。
- [视觉适配诊断](reports/2026-10-05-视觉适配诊断.md)：保留指向产物盘的软链接，依赖本机挂载。

## 参考材料

- [2026-10-09 OPD / 多模态投机解码优化文献库](research/2026-10-09-opd-multimodal/README.md)：68 篇论文、PDF、中文卡片与针对当前实验的建议。
- [五篇论文调研](research/five-paper-survey.md)。
- [ViSpec / MMSpec 原文提取](research/vispec-mmspec-extraction.md)。
- [上游库说明](../GLANCE_repo/README.md)：上游结果不等于本地实验结果。
- [历史简历条目](career/简历条目_GLANCE_OPD.tex)：保留编辑源，使用前按当前结论核对数据与措辞。
- 原论文 PDF 和 `glance.txt` 仍在项目根目录；依赖包 README、LICENSE 和 `.spec-workflow` 模板保持原位。

## 2026-10-09 清理记录与恢复

本目录没有 Git 历史。清理前已保存并逐文件校验 [14 份文档原件的压缩包](archive/documentation-before-cleanup-2026-10-09.tar.gz)，对应 [SHA-256 清单](archive/documentation-before-cleanup-2026-10-09.sha256) 使用原始相对路径。归档仅供追溯，不作为当前方法说明。

| 原位置 / 文件 | 处理及理由 |
|---|---|
| `项目经历_GLANCE架构与OPD训练方法.md`、`项目经历_精简版.md` | 与总结和报告重复，且包含旧 OPD 语义；已从活动目录移除，原文在归档内 |
| `面试准备_多模态投机解码项目.md`、`面试方法讲解_无数字.md` | 多份平行叙述容易漂移；移除活动副本，原文在归档内 |
| `repro/README.md` | 改为运行入口；详细旧记录保留为 `reports/2026-09-25-reproduction-history.md` |
| 根目录后续计划、A/A4/B1/2×2/EAGLE3 报告 | 移至 `plans/` 与 `reports/`，保留独立实验依据并更新文档交叉引用 |
| 两份文献笔记、简历 TeX、诊断软链接 | 分别移至 `research/`、`career/`、`reports/`，保留有独立用途的资料 |

安全恢复到独立临时目录（不会覆盖当前文件）：

```bash
# 在项目根目录运行
GLANCE_DOC_RESTORE=$(mktemp -d /tmp/glance-docs-restore.XXXXXX)
tar -xzf docs/archive/documentation-before-cleanup-2026-10-09.tar.gz -C "$GLANCE_DOC_RESTORE"
(cd "$GLANCE_DOC_RESTORE" && sha256sum -c /home/mcy/Projects/GLANCE/docs/archive/documentation-before-cleanup-2026-10-09.sha256)
```

上述 2026-10-09 文档清理未修改训练代码、模型、数据、checkpoint 或外部产物盘报告；2026-10-16 的代码重构见当前契约。
