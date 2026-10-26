# 今晚如何启动两个 DeepSeek 会话

在同一机器的两个独立DeepSeek/Codex终端会话中，分别提供以下内容（或者把本包三个 `.md` 放到项目可读路径）：

**会话A首条指令：**

> 在 `/home/mcy/Projects/GLANCE` 工作。完整读取项目 AGENTS.md，并同时读取 `00_parallel_execution_rules.md` 与 `01_task_A_tree_selection.md`（请把两份文件放到项目的只读可见位置，或直接将内容粘贴到会话）。立即按任务A执行直到取得真实可复现的实验结论；无需我的阶段确认。你独占A的目录及报告，不编辑共享 AGENTS.md；GPU使用共同flock锁。只在真实资源/正确性阻塞时报告具体证据，不得以计划作为完成。

**会话B首条指令：**

> 在 `/home/mcy/Projects/GLANCE` 工作。完整读取项目 AGENTS.md，并同时读取 `00_parallel_execution_rules.md` 与 `02_task_B_temporal_response.md`。立即独立执行任务B的真实视频时序干预及target/draft前向，直至得出可复现实验结论；不等待A结果，不需我的阶段确认。你独占B的目录及报告，不编辑共享 AGENTS.md；GPU使用共同flock锁。若干预数据不可判，交付可行性与真实负/阻塞证据，不得制造正结论。

**并发前注意：** 建议先把三个文件复制到项目 `docs/tasks/parallel_20261012/`，确保两个会话都读到相同文本。不要把 `/mnt/data` 当成用户远程主机上的现成路径；这三份文件当前是本次对话的下载产物。

**次日汇总：** 两边完成后，分别检查 `run_manifest.json` 和独立报告；由第三个单独的汇总会话审计证据、统一更新 `AGENTS.md` 与 `docs/plans/video-opd.md`，不合并不同reference/anchor池，不擦除负结果。
