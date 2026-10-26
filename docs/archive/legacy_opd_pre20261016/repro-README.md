# GLANCE 复现与实验脚本

当前结论、进度与下一步统一维护在 [AGENTS.md](../../../AGENTS.md)。本文只提供代码与运行入口。旧 README 的详细实验记录保存在[复现历史](../../reports/2026-09-25-reproduction-history.md)。

## 环境与路径

- 代码：`/home/mcy/Projects/GLANCE`；现有 Python：`/home/mcy/.conda/envs/fin_research/bin/python`。
- 模型、数据、checkpoint、日志和结果：`/mnt/local/mcy/GLANCE`。
- `common.py` 支持 `GLANCE_ROOT`，并将 `GLANCE_repo` 和本地 DFlash 包加入导入路径；部分 shell 启动器硬编码了目录。
- GPU 运行依赖本地模型和数据。先检查脚本参数、GPU 占用及输出目录；本页不代表执行后可覆盖已有实验。

## 入口对照

| 任务 | 文件 |
|---|---|
| 图像数据 / 标签 / SFT | `build_train_prompts.py`、`build_eval_data.py`、`gen_target_greedy.py`、`train_head.py` |
| ChartQA / 无损评测 | `eval_chartqa.py`、`audit_lossless.py`、`report.py` |
| 视频数据与污染审计 | `predecode_cache.py`、`merge_pool_shards.py`、`audit_holdout_contamination.py` |
| 视频 aligned SFT | `train_head_video.py --feature-path cached`、`run_phaseA4_aligned_sft.sh` |
| 视频评测与汇总 | `eval_video_opd.py`、`run_phaseA_eval.sh`、`summarize_phaseA.py` |
| 旧版 OPD 2×2 | `run_video_opd_2x2.sh`、`eval_video_opd_2x2.sh`、`summarize_video_opd_2x2.py` |
| context 对齐门禁 | `check_context_interface.py`、`aligned_forward.py` |
| full-KL 原型 / 精度 | `proto_fullkl_b0.py`、`check_b0_fp32_invariance.py`、`opd_fullkl.py` |
| 当前 target 在线 OPD / 无更新门禁 | `opd_train_video_fullkl.py`、`opd_verified_rollout.py`、`test_opd_verified_rollout.py`、`check_opd_verified_gpu.py` |
| B0.5 / 过程监控样本冻结 | `freeze_phase_subsets.py` |
| EAGLE3 支线 | `eagle3/00_validate.sh`、`eagle3/10_capture.sh`、`eagle3/20_train.sh` |

## aligned 起点评测示例

以下命令会占用 GPU 并写入结果；执行前选择空闲 GPU，并为新的运行设置未使用的 `RES`。它只评估已有 step1000，不重新训练。

```bash
cd /home/mcy/Projects/GLANCE
GPU=0 \
OUT=/mnt/local/mcy/GLANCE/checkpoints/head_video439_aligned \
RES=/mnt/local/mcy/GLANCE/results/phaseA4_recheck_newrun \
STEPS=1000 WITH_WARM_REF=0 \
bash repro/run_phaseA_eval.sh
```

脚本使用 fp32、b63/k8、max_new=1024、AR 门禁；30 条视频中首条 warmup 不进入指标聚合。`WITH_WARM_REF=0` 防止额外评估 legacy 起点。训练入口默认 `uncached`，aligned 训练必须显式指定 `cached`。

`opd_train_video_fullkl.py` 默认 `--rollout-mode online`：当前 target 验证、提交并推进上下文，loss 重放检查 teacher/student/接受分组。旧轨迹实验须显式 `--rollout-mode legacy-reference`；历史 Phase C/C4 启动器已固定该选项。新模式输出必须使用新目录，低学习率必须指定 `--master-weights cpu|gpu`。当前采用较慢的逐 token teacher 验证，仍是原 full-KL 目标，不冒充官方 sampled/TV loss。CPU 与2视频6轮无更新 GPU 门禁见[修复报告](../../reports/2026-10-13-opd-online-fix.md)；完整训练显存/吞吐未重新验证，本轮未训练。
