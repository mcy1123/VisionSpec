# 新 Warm-2000 训练曲线检查

结论：训练 loss 前期快速下降、后期放缓但仍缓慢下降；没有证据确认开发集接受率已到顶。不能据此断言 Warm 数据一定不足或已经充足。本次只读取历史日志并在 CPU 绘图，没有额外训练/评测，不改变当前 ScienceQA 任务。

[曲线 PNG](/mnt/local/mcy/GLANCE/results/warm_curve_audit_20261019_145826/warm_training_curve.png) · [PDF](/mnt/local/mcy/GLANCE/results/warm_curve_audit_20261019_145826/warm_training_curve.pdf) · [原始数值 CSV](/mnt/local/mcy/GLANCE/results/warm_curve_audit_20261019_145826/warm_training_curve.csv) · [摘要与源 SHA](/mnt/local/mcy/GLANCE/results/warm_curve_audit_20261019_145826/SUMMARY.json)。绘图程序同目录 `plot_warm.py`，执行时需要 `--out` 参数。

源：`/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/warm/train_logs/updates.jsonl`。2000条记录按update1–2000严格连续、无重复；末端rows_seen=16000。训练器使用16000行的一次随机排列，每更新8行，因此这一轮是16,000训练行的一次遍历，不能与原论文6个epoch直接等同。此Warm由已预训练文本DFlash初始化，不是随机初始化Head。

| 更新窗口 | 平均训练loss |
|---|---:|
| 1–250 | 4.44809 |
| 251–500 | 3.96455 |
| 501–750 | 3.81505 |
| 751–1000 | 3.71385 |
| 1001–1250 | 3.65268 |
| 1251–1500 | 3.56499 |
| 1501–1750 | 3.54413 |
| 1751–2000 | 3.51362 |

最后250步相对前250步下降约0.861%。图中浅蓝为每步原始loss，深蓝为100步尾随均值，右图橙线为不重叠250步均值。后期仍有噪声和下降趋势，但不同步使用不同样本，窗口差异不是固定开发集上的显著性检验。

现有Warm运行记录中找到最终dev8评测，但没有随Warm中间checkpoint变化的固定开发集有效token/轮曲线。因此不能用训练loss替代推理收益，也不能声称已经得到“饱和SFT后再OPD”的证据。

原论文图1展示预热后继续SFT趋于平台而OPD继续改善；附录A指定OPD从SFT6epoch checkpoint初始化。这支持论文研究动机，但没有说明每个模型都用在线早停规则找到第6epoch才开始OPD。来源：[Draft-OPD，图1与附录A](https://arxiv.org/html/2605.29343v3)。
