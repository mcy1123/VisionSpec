# GLANCE 运行入口

当前状态只维护 [AGENTS.md](../AGENTS.md)，方法与验证边界见[官方 KL 契约](../docs/methods/draft_opd_official_kl_contract.md)。当前为 **L2 CPU-only / 待真实模型门禁**，本轮没有启动训练。

## 目录职责

| 目录 | 内容 |
|---|---|
| `opd_official_kl/` | 当前唯一 OPD loss、在线 rollout 和训练循环 |
| `shared/` | 路径/模型加载、草稿前向、SFT block、优化器、恢复和统计工具；无旧 OPD loss 依赖 |
| `scienceqa_image/` | ScienceQA 图像数据适配、SFT/OPD 入口与任务产物工具 |
| `qa_opd_fair/` | 视频 QA 输入、SFT/OPD 入口与任务工具 |
| `data/` | 下载、预处理、数据池与标签构建；实际数据仍必须写到允许的 `/mnt` 挂载点 |
| `evaluation/` | 评测、结果汇总与对比分析 |
| `diagnostics/` | 接受行为、KV、精度、梯度等专项诊断；不作为常规训练入口 |
| `tests/` | 可复用的 CPU 回归测试；含明确标注的历史语义测试 |
| `historical/` | 旧 OPD 配方、旧 SFT 入口、Phase/Task 启动脚本；旧名字的 OPD 兼容入口仍转到当前 runtime |
| `tools/` | GPU 查询与训练监控 |
| `legacy/` | 改动前的冻结源码与显式历史重放；不按当前实现更新 |
| `experiments/`、`eagle3/`、`videoqa_transfer/` | 保持完整的独立实验/迁移任务目录 |

历史文件按当时语义保留，其“official”“online”等旧表述不代表当前契约。新 OPD 不从 `historical/` 导入 loss 或 rollout；草稿几何和 EOS 工具已独立为 `shared/draft_geometry.py`，SFT block 为 `shared/sft_blocks.py`。

## 当前入口

在项目根目录执行：

```bash
/home/mcy/.conda/envs/fin_research/bin/python repro/scienceqa_image/train_sqa_opd.py --dry-run
/home/mcy/.conda/envs/fin_research/bin/python repro/train_video_official_kl.py --help
PYTHONPATH=repro /home/mcy/.conda/envs/fin_research/bin/python -m unittest tests.test_official_kl tests.test_official_cleanup -v
```

SFT 对照仍为 `scienceqa_image/train_sqa_sft.py` 与 `qa_opd_fair/train_qa_sft.py`，保留 variable-reveal CE。当前任务未授权真实模型训练。

## 查找旧文件

143 个旧顶层文件的完整映射在 [layout.json](layout.json)。旧报告和冻结源码原文不改写；其中的旧文件名通过以下命令定位或运行：

```bash
python repro/run.py --list
python repro/run.py eval_chartqa.py -- --help
python repro/run.py opd_train_video_fullkl.py -- --dry-run
```

此 launcher 只解析路径，**不替换历史实验配方**。恢复旧 OPD 训练目标须使用 `legacy/replay.py --acknowledge-historical-objective`；当前兼容入口拒绝旧目标开关。更早的完整运行环境可从审计快照恢复到独立目录。

本次目录整理的映射、快照和检查结果：`/mnt/local3/mcy/GLANCE/results/repro_reorganization_20261016/`。
