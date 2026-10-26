# ScienceQA image OPD默认切换与一次smoke

结论：**GO_SMOKE**。只证明能跑，不证明等价、稳定训练或恢复能力。native为近似方法，没有serial等价声明。

默认模式：pure_block，parallel_rollout=4；active runtime只有pure block，无serial/1+keep分支。

完整代码根快照：`/mnt/local3/mcy/GLANCE/results/scienceqa_pure_default_smoke_20261019_131714/source_before.tar.gz`（保留软链接、不递归复制挂载点数据）；原文件SHA：FILES_BEFORE.json；独立旧实现归档：`/mnt/local3/mcy/GLANCE/results/scienceqa_pure_default_smoke_20261019_131714/archived_legacy`。归档与checkpoint真实路径均在/mnt/local3。

变更：canonical ScienceQA入口直接使用已测pure-native分支与四卡rollout/rank0顺序评分；删去active serial trainer、runtime serial/1+keep分支及旧等价性入口。旧视频入口明确提示恢复归档，不静默改变视频方法。loss、0.9、1:2、精度、prompt、数据及日程未变；部署decoder/verifier源码未改。

身份：opd_mode、world_size、slot_map、initial_head_version与当前Head version均记录。新checkpoint标block_native；本smoke标smoke/diagnostic_only，不能用于正式step121。禁止跨模式/拓扑恢复；本轮未验证resume。少于四张可见卡/非四rank启动直接失败，无降级路径。

CPU仅import/dry-run/help和四卡fail-fast检查（日志cpu_tests.log/dry_run.json/help.txt），未运行任何梯度等价测试。GPU仅一次真实optimizer update，从原Warm step2000、seed20261013、原update0的4 QA，原lr1e-5 constant/Target fp32/Head bf16/CPU fp32 master。checkpoint只重载，不执行恢复训练。dev8使用原部署验证器与独立AR缓存。

实际结果：`{"result": "GO_SMOKE", "error": null, "attempts": 1, "optimizer_updates": 1, "checkpoint": "/mnt/local3/mcy/GLANCE/checkpoints/scienceqa_pure_default_smoke_20261019_131714/run/step1", "reload": {"passed": true, "checkpoint": "/mnt/local3/mcy/GLANCE/checkpoints/scienceqa_pure_default_smoke_20261019_131714/run/step1", "head_loaded": true, "full_state_loaded": true, "optimizer_steps": 1, "adam_steps": [1.0], "block_native": true, "smoke": true, "diagnostic_only": true, "resume_test": false, "gradient_or_equivalence_test": false, "training_identity": {"opd_mode": "pure_block", "world_size": 4, "slot_map": {"0": [0], "1": [1], "2": [2], "3": [3]}, "initial_head_version": {"checkpoint_model_sha256": "bd16ebcff1c513d42b14d9b8d89d428d93973ff8620bb78eafa9689509616735", "optimizer_step": 0}, "schedule_sha256": "6bbb4c6b28f1890f633a6c0f719cee4dafe2a1066e6c043bb84ca5004a677a6a", "pool_sha256": "c25ddd54b85696bd9e4576025a59dd0490e7407f8e5a2d006ec9bb0cf9a65f49", "model_config_sha256": "9834d608c9ca53d5548b415471ae9e8ebc9aab6cedfc2a7af95b6bd097373102", "config_sha256": "d9519f597c547a4ee6959ac090047190842764a058257421b147247a3c24fd92", "head_version": {"head_sha256": "366e7815773a82c88881f1972f5341493a4cb575b48a18ff57777316d809095e", "optimizer_step": 1, "schedule_cursor": 1}}}, "loss": 4.52362209545263, "gradient_norm": 1.9239343404769897, "dev8_token_exact": 8, "default_changed": true, "claim": "only proves runnable; not equivalence, stability or resume"}`。

回滚命令（恢复active代码，不运行训练）：

```sh
/home/mcy/.conda/envs/fin_research/bin/python /mnt/local3/mcy/GLANCE/results/scienceqa_pure_default_smoke_20261019_131714/rollback.py
```

没有120步、第二seed、2+resume+2、58张梯度或等价性门禁；没有confirm200。旧exact block HOLD仅指归档方法。未来正式训练需另批授权。所有历史checkpoint/报告保留。
