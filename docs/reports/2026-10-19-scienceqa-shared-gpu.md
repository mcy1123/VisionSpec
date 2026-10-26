# ScienceQA 共享GPU续训

用户明确要求“只要放得下就行”。控制器由“目标卡没有其他计算进程才执行”改为“剩余显存满足当前模型路径即可执行”，不再因为其他进程存在而等待，不终止其他任务。

本次run：`/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_restart_20261019_180942`。变更前确认SFT/OPD均完成400次更新且完整checkpoint已发布，控制器无子进程、仅在资源等待中；只终止该空闲控制器，然后使用同run的`--resume`继续。已完成的更新和预算不退款、不重跑；未触发早停，仍需完成OPD-400的dev评测。

所需空闲显存按已知模型路径40GiB起算，与本run历史allocated峰值加1GiB取大；当前为40960 MiB。CPU检查证实存在其他进程但显存足够时立即通过，实际GPU1/2/6/7余量也通过。共享任务后续继续申请显存仍可能产生实际OOM，不把“启动时放得下”说成运行全程保证。

源码与运行元数据修改前快照：`/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_restart_20261019_180942/shared_gpu_amendment_20261019_222043`。原先等待控制器日志保留。新入口`resume_shared_gpu.sh`，输出`controller_shared_gpu.log`，仍在tmux `glance-sqa-full`。只做一次后台启动确认，没有额外watchdog或持续监控。早停和最终独立SPEC=AR规则不变。
