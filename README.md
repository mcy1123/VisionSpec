# GLANCE-ScienceQA

冻结 Qwen3-VL-8B-Instruct 的多模态无损投机解码：1.05B / 5 层的 block-diffusion draft head 一次前向覆盖 16 个位置（1 root + 15 预测），target 一次 ancestor-masked 前向验证 63 节点前缀树，输出与贪心 AR 逐 token 相同。

## 结果

ScienceQA-IMG，3 epochs / 3546 updates / 4 QA per update：

| | SFT | OPD |
|---|---:|---:|
| test200 有效 token/轮 | 5.3666 | **6.2206** |
| 答案正确 | 162/200 | 162/200 |
| SPEC = 独立 AR | 200/200 | 200/200 |

- OPD − SFT = **+0.8541**，95% CI [+0.7157, +1.0037]。
- 24 图测速：相对 AR **2.522×**，相对 SFT **1.099×**。
- 跨域：长 CoT 1.02–1.10×，短答 0.94–1.01×；Head 只在 ScienceQA 上训练。

## 目录

| 路径 | 内容 |
|---|---|
| `repro/` | 训练、数据、评测入口；从 `repro/README.md` 开始 |
| `GLANCE_repo/` | 上游 GLANCE 模型/解码代码（MIT） |
| `scripts/dflash_pkg/` | DFlash draft head 包（MIT） |
| `docs/` | 方法、实验报告、组会文档 |
| `AGENTS.md` | 当前交接与运行规则 |

## 运行

Python 3.11；依赖见 `GLANCE_repo/requirements.txt` 与 `scripts/dflash_pkg/dflash-0.1.0/pyproject.toml`。模型、数据和结果默认放在 `/mnt/local/mcy/GLANCE`，可用 `GLANCE_ROOT` 覆盖（见 `repro/shared/common.py`）。

```bash
python repro/scienceqa_image/train_sqa_opd.py --dry-run
python repro/scienceqa_image/train_sqa_sft.py --help
```

## 文档

- 组会文档：[docs/talks/2026-10-22-group-meeting.md](docs/talks/2026-10-22-group-meeting.md)
- 三卡续训最终报告：[docs/reports/2026-10-20-scienceqa-three-card-continuation.md](docs/reports/2026-10-20-scienceqa-three-card-continuation.md)
- 方法契约：[docs/methods/draft_opd_official_kl_contract.md](docs/methods/draft_opd_official_kl_contract.md)

## 第三方代码

`GLANCE_repo/` 和 `scripts/dflash_pkg/` 是上游 MIT 代码，保留原 `LICENSE`。论文 PDF 与抽取全文未纳入仓库。
