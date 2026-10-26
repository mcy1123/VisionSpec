# pure_block_native 显式实验入口

## 当前状态

用户允许训练中的native block与serial发生token/teacher logp/梯度差异，不再以serial精确等价作为实验门禁；训练配置保持同Warm/日程/seed/lr及精度。最终部署输出仍要求逐token SPEC=独立AR。部署decoder与验证器未修改。

已接入 `repro/experimental/train_pure_block_native.py`：每轮一次block Target forward，直接crop/保留其KV、hidden与fused；不执行原block的1+keep重建。最后一个root的单次materialization仍保留，这不是每轮serial重建。旧serial默认路径、旧block HOLD入口均保留原行为。

训练使用同一Bernoulli forward KL/rejected k3/官方池化函数、decay=0.9和权重1:2。native rejected teacher logp采用原block生成结果；response评分仍沿用原hidden-row投影。按用户最新语义，native模式不要求重投影Target argmax等于native committed token；该模式专属近似不影响serial断言。student自身replay、KV/长度/有限性、完整梯度图检查继续保留。没有引入新loss、减少anchors或改训练精度。

四卡继续rollout-only，rank0 canonical scoring/backward/CPU master optimizer；trace显式标注native模式，配置身份隔离；checkpoint保存diagnostic_only/block_native，不能作为正式step121或混入旧serial结果。新增真实更新必须绑定有上限的实验账本；没有自动运行120步。

## 已执行验证

34项CPU测试通过，包括pure-native EOS/cap/KV状态和每轮恰好一次block调用、旧block训练拒绝、原serial身份/传输/分母/loss回归。首轮新测试导入路径失败原日志保留，修正后完整34项通过；没有模型数值失败或optimizer重试。

四张A6000 GPU0/2/6/7，从原Warm step2000，原日程update0四条QA，Target fp32/Head bf16/CPU fp32 master，生产入口零更新运行成功。loss=4.52362209545263，preclip grad norm=1.9239252805709839，最大allocated=36.864 GiB；真实optimizer_steps=0。未对拍serial梯度，也不把此次不同起点/日程且不含optimizer的耗时当作120步性能收益。无训练checkpoint发布。

## 后续120步实验边界

前一条120步＋50条token审计为用户提出的预算建议；本轮落实最新method语义和入口，尚未启动120步或确认集评测。计划配置仍是原480 QA、seed20261013、lr1e-5、120 updates、4 QA/update、checkpoint30/60/90/120；不启第二seed、不扩数据、不自动追加或重试。native/serial差异统计用于解释结果，不阻拦训练；无损只对最终部署输出逐token判定，不能用“偶尔一个token”容差冒充无损。所有比较必须分开标记native与既有serial结果。

## 复现

CPU：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=repro /home/mcy/.conda/envs/fin_research/bin/python -m unittest tests.test_pure_block_native tests.test_opd_production_parallel tests.test_opd_rollout_parallel tests.test_opd_exact_transport tests.test_opd_efficiency tests.test_official_kl tests.test_block_verification -q`。

只读配方：`/home/mcy/.conda/envs/fin_research/bin/python repro/experimental/train_pure_block_native.py --dry-run --parallel-rollout 4 --qa-per-update 4`。本次真实GPU零更新完整参数见/mnt/local3/mcy/GLANCE/results/pure_block_native_20261019_110343/smoke_command.json，代码修改前快照和CPU日志同目录；不要覆盖已有out。

## 120步启动授权与执行

用户随后明确“启动启动”，激活新的120次更新账本；沿用配置，checkpoint30/60/90/120。真实计数以本轮BUDGET为准，失败不自动恢复/重试。任务在tmux后台独立执行，完成后自动评测和token审计；50 QA审计在训练结束后用独立空闲A6000运行，避免抢训练卡，不阻拦120步。所有评测复用固定输入，不扩训练数据、不增seed。旧“120未启动”段落是接入验证阶段历史状态，本节及RUN_STATUS/BUDGET为当前状态。

启动已确认：GPU0/2/6/7，原Warm seed20261013，账本已完成至少4/120次。RUN_STATUS是阶段状态（计数在阶段切换刷新），实时次数以BUDGET.json和train/log.jsonl为准。TRAIN_DONE.txt标记120次完成，最终DONE.txt/DONE.json标记评测/审计完成或失败。本机无桌面通知服务，不承诺自动聊天消息；有连接tmux客户端时尝试消息通知。后处理脚本已准备，不再逐轮人工监控。

## 120步实验最终结果

120/120次pure block-native更新完成，只有一个seed，全部checkpoint为diagnostic_only/block_native。无额外更新或重试。部署dev8与复用confirm200均逐tokenSPEC=独立AR；这不证明native训练轨迹等于serial。

训练摘要：`{"updates": 120, "update_elapsed_median": 26.1856987689971, "update_elapsed_mean": 25.744744177330478, "update_seconds_sum": 3089.3693012796575, "max_peak_allocated": 40483458048, "loss_first": 4.52362209545263, "loss_last": 3.503765185758673, "grad_norm_min": 0.3035307228565216, "grad_norm_max": 3.0511603355407715, "train_wall_seconds": 3329.699651002884, "timing_scope": "update elapsed includes rollout/scoring/backward/master/optimizer/version barrier, excludes checkpoint writes; total train wall includes loading/checkpoint writes", "GPU_seconds_sum_estimated": 12357.47720511863, "GPU_seconds_definition": "4 multiplied by update elapsed; reserved four GPUs; shared host, no clean serial speedup claim"}`。

效果对照：`{"method": "block-native vs prior serial OPD120, same fixed seed/config/data; not serial-exact", "dev8_effective_tau": 4.304054054054054, "serial_dev8_effective_tau": 4.304054054054054, "dev8_legacy_tau_delta": 0.0, "dev8_legacy_abs_delta_le_005": true, "confirm200_native_mean_image_tokens_per_round": 4.4445538706163115, "confirm200_serial_mean_image_tokens_per_round": 4.445689784442371, "confirm200_paired_mean_delta": -0.0011359138260599, "confirm200_paired95ci": [-0.00741801284983479, 0.004978513320463857], "bootstrap_seed": 20261019, "bootstrap_repetitions": 10000, "unit": "image,200 unique images", "answer_correct_native": 153, "answer_correct_serial": 153, "accuracy_no_drop": true, "interpretation": "Effect differences are reported; the old OPD-minus-SFT CI is not an equivalence margin for native-minus-serial. User accepts training differences; deployment lossless is strict.", "cohort_reused": true}`。

Warm固定50 QA/1h观察审计：`{"completed_qa": 50, "max_qa": 50, "budget_seconds": 3600, "elapsed_seconds": 3058.3946048410144, "committed_fork_qa": 0, "output_different_qa": 0, "parsed_answer_different_qa": 0, "teacher_logp_max_abs_matched_prefix": 0.00018310546875, "effective_tau_delta_mean": -0.0027267404234565707, "gate": false, "optimizer_updates": 0, "checkpoint": "fixed original Warm step2000", "comparison": "live native vs serial, conditional logp comparisons only on matching prefixes/proposal token; divergence is descriptive"}`。差异不是serial精确门禁；不把不同前缀上的概率差归为数值噪声。Head逐张量差异见HEAD_COMPARISON.json。

旧block的HOLD结论未改，serial默认未切换；新native结果为独立近似方法，不与旧serial M1/M2/confirm200合并统计。复用的200图像集合不是新独立确认集。日志逐步保留loss/grad norm/时间/峰值显存，原状态与报告不覆盖。
