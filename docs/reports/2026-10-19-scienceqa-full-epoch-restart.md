# ScienceQA 全量训练重新启动

用户明确要求“帮我重启吧，不用监控”。本次从原新Warm-2000重新启动两臂，不冒充从未保存的step101续训。旧run `/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_20261019` 的101次更新、预算、日志和step0保持不变。

新run：`/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_restart_20261019_180942`。数据目录软链接复用原来的4726条标签、pool、schedule与固定图像划分；复用已经完成的Warm dev32基线，因此不重新生成标签或重跑Warm评测。输入SHA、源码快照、来源与重新启动授权记录见新run的INPUTS.json、SOURCE.json、source_snapshot.tar.gz、RESTART.json。

最多SFT/OPD各1182次新更新，新增optimizer attempts上限2364。旧101次作为已消耗且未保存的重启成本单列，累计上限2465；不退款、不改旧账本。两臂仍同Warm、同QA顺序、seed20261013、lr1e-5 constant、Target fp32、Head bf16、CPU fp32 master AdamW。

训练期间不执行teacher/梯度等价、SFT缓存AR重放相等、OPD proposal argmax相等、dev SPEC=AR门禁。每200步保存完整checkpoint并使用SPEC-only dev32做早停：任一臂连续3次没有超过0.01的提升，两臂同一步停止；最多覆盖一轮。最终checkpoint才检查新confirm200的独立AR逐token一致性。NaN/Inf、无效输入、checkpoint损坏和写入失败仍会报告执行错误。

通过tmux后台运行，四卡GPU1/2/6/7，当前块先SFT，再OPD。没有创建额外watchdog或持续监控；只做一次启动确认。入口 `/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_restart_20261019_180942/launch.sh`；controller.log、CURRENT.json与logs/记录进展。tmux pane身份写在RUN_SESSION.json。完成／早停结果写DONE.json和DONE.txt。
