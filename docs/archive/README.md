# 归档目录

只用于追溯，不作为当前方法或训练方案。当前状态看 [AGENTS.md](../../AGENTS.md)，方法看 [methods/](../methods/)。

## 内容

| 路径 | 内容 |
|---|---|
| `AGENTS-*.md`（3 份） | 2026-10-17 / 10-19 的历史交接全文 |
| `legacy_opd_pre20261016/` | 10-16 前的旧 OPD 说明、旧视频计划、旧运行教程 |
| `prompts_20261016/` | 根目录移入的两份 Codex 执行 prompt（Warm 重建、官方 KL 夜间训练） |
| `tasks_20261012_20261015/` | 10.12–10.15 的任务 prompt 与并行任务包（含 `parallel_20261012/`） |
| `talks_backup_20261105/` | 组会文档增补 baseline 前的原件 |
| `method-overview-20261015.md` | 截至 10-15 的方法总览快照，已被组会材料与报告取代 |
| `documentation-before-cleanup-2026-10-09.tar.gz` + `.sha256` | 2026-10-09 文档清理的 14 份原件及校验清单 |

## 2026-11-05 清理记录

| 原路径 | 现路径 | 理由 |
|---|---|---|
| `GLANCE_ScienceQA_从零重建Warm_…_Codex_20261016.md` | `prompts_20261016/` | 已完成的执行 prompt，结论已进报告 |
| `GLANCE × ScienceQA：官方 KL-OPD 大规模夜间训练.md` | `prompts_20261016/` | 同上 |
| `docs/tasks/*.md`、`docs/tasks/parallel_20261012/` | `tasks_20261012_20261015/` | 一次性任务说明 |
| `docs/talks/.backup/…` | `talks_backup_20261105/` | 临时副本，不进活动目录 |

同时把文献库的 `pdfs/`、`texts/`、`metadata/`（约 119 MB）移到 `/mnt/local/mcy/GLANCE/docs_research/2026-10-09-opd-multimodal/`，原位置保留软链，文档树从 122 MB 降到约 2 MB。

被移动的任务 prompt 有三处代码内的 source-of-record 引用，已同步改成新路径：

- `repro/scienceqa_image/freeze_prereg.py` → `docs/archive/tasks_20261012_20261015/10.15-1948.md`
- `repro/experiments/temporal_d4_b/package_b.py`、`write_prereg.py` → `docs/archive/tasks_20261012_20261015/parallel_20261012/02_task_B_temporal_response.md`

报告正文里以行内代码提到旧 `docs/tasks/…` 路径的地方按历史记录保留，未改写。

## 恢复

本目录已在 git 跟踪中，移动/删除可用 `git log --follow` 追溯，或直接 `git checkout <commit> -- <path>`。

2026-10-09 的压缩包恢复到独立临时目录（不覆盖当前文件）：

```bash
GLANCE_DOC_RESTORE=$(mktemp -d /tmp/glance-docs-restore.XXXXXX)
tar -xzf docs/archive/documentation-before-cleanup-2026-10-09.tar.gz -C "$GLANCE_DOC_RESTORE"
(cd "$GLANCE_DOC_RESTORE" && sha256sum -c /home/mcy/Projects/GLANCE/docs/archive/documentation-before-cleanup-2026-10-09.sha256)
```
