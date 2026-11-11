# 文档索引

日常交接与运行规则只维护根目录 [AGENTS.md](../AGENTS.md)；训练/评测入口见 [repro/README.md](../repro/README.md)；按日期查实验结论见 [reports/README.md](reports/README.md)。

## 当前有效

| 文档 | 内容 |
|---|---|
| [官方 KL 契约](methods/draft_opd_official_kl_contract.md) | loss、池化、stream 权重与旧入口差异 |
| [组会材料](talks/2026-10-22-group-meeting.md) | 方法、主结果、5 条 baseline 对比（2026-11-05 增补） |
| [三卡续训最终报告](reports/2026-10-20-scienceqa-three-card-continuation.md) | SFT/OPD 各 step3546 的最终数字与协议 |
| [baseline 复现报告](reports/2026-11-04-speculative-baselines.md) | 同一 test200 上的 5 条已发表方法 |

## 目录

| 目录 | 用途 |
|---|---|
| `reports/` | 按日期的实验报告与证据（45 篇，索引即上方链接） |
| `methods/` | 方法契约 |
| `talks/` | 组会材料与配图 |
| `research/` | 文献调研与论文卡片；PDF、抽取全文、metadata 实体在 `/mnt/local/mcy/GLANCE/docs_research/`，此处保留同名软链 |
| `career/` | 简历条目源文件 |
| `plans/` | 只保留一个历史计划跳转页，正文在 `archive/` |
| `archive/` | 旧 AGENTS、旧计划/任务 prompt、清理备份；清单见 [archive/README.md](archive/README.md) |

## 归档原则

- 报告与负结果不删除、不改写；过时的计划、任务 prompt、临时副本移入 `archive/`。
- 大文件（PDF、全文、metadata）按存储规则放 `/mnt/local`，文档树只留索引和小文本。
- 2026-11-05 的清理清单与原路径对照见 [archive/README.md](archive/README.md)。
