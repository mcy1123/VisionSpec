# 当前 OPD 实现审计：proposal 在线，但验证闭环未达到严格 Draft-OPD

日期：2026-10-13。范围：只读正式 C4 的训练入口、数据流、loss、优化器配置及官方代码；运行 CPU 测试和合成反例，不启动真实模型前向、训练或参数更新。

## 结论

**按“类似官方 Draft-OPD、由当前策略与当前 target 在线投机验证产生训练反馈”的要求，当前实现尚未达标。** 它不是早期固定 anchors/top-k 的纯离线训练：每个 update 会重新跑当前 head 的 proposal，teacher 也现场评分 draft-induced prefix。但基础 continuation、接受标签与后续 root 来自磁盘旧轨迹；在线 teacher 不回写接受分组。准确名称是“旧 reference 轨迹上的当前策略 proposal + 在线 teacher 全词表 KL”，不能直接称严格在线投机验证训练。

已完成 QA 评测的 token 门禁和数值仍然成立；其研究对象是这套实现产生的 C4 checkpoint。不能把它升级为标准在线 Draft-OPD 的公平负结果，也没有证据证明本缺口就是全部负结果的原因。

## 1. 正式 C2 实际跑的入口

`repro/run_phaseC4_10upd.sh` 与保存的 `/mnt/local/mcy/GLANCE/checkpoints/phaseC4_C2/opd_fullkl_config.json` 一致：

- `opd_train_video_fullkl.py`，输入 `data/video_greedy_499.pt` 与 `pool_499.parquet`。
- C0 aligned step1000 起点，10 updates，24 anchors/update，最多24轮/视频。
- target/teacher fp32，head bf16，CPU fp32主权重；lr3e-5、gamma0.8、chain rollout。
- 不是早期 `opd_train_video.py --rollout-mode stored`，也不是 phaseC3。

## 2. 关键调用链与发现

| 环节 | 实际代码 | 判断 |
|---|---|---|
| 每次更新生成proposal | `opd_train_video_fullkl.py:1242`起的update循环，`:1260`调用`build_video_trace`；`:637`调用当前head的rollout | 是当前策略，不是固定proposal池 |
| 基础回答轨迹 | `opd_train_video_fullkl.py:587`：`full_ids = record["input_ids"]`，来自磁盘records；用整条旧轨迹重新prefill target | 特征在线重算，但文本轨迹并未在线生成 |
| 接受判定 | `opd_online_rollout.py:121`附近从`seq_ids[pos+1:...]`取`want`；`:39–43`比较draft argmax与`want`的leading match | 不是调用当前target verify得出的`n_acc` |
| 后续状态 | `opd_online_rollout.py:131`按旧参考的`n_acc+1`推进；下一轮root继续来自`seq_ids[pos]`，context来自旧轨迹的重新prefill | 后续状态也绑定旧reference，单改loss不能修好整个闭环 |
| 分组 | `opd_train_video_fullkl.py:468`读取anchor的`n_acc`，`:483`形成accepted/rejected mask | 计划先冻结分组 |
| 在线teacher | `opd_fullkl.py:231`在anchor prefix后接`prop[:k-1]`，对draft-induced prefix重新计算分布；`:805`起score_microbatch调用它 | 这部分确实在线，不是只查离线top-k标签 |
| teacher能否纠正分组 | `opd_train_video_fullkl.py:834`取得`p_logits`后，`:854`仍使用plan.offsets_acc/rej | 没有重新判定/校验接受长度；teacher反馈未闭环 |
| KL与归一 | `opd_fullkl.py:424/436`，`opd_train_video_fullkl.py:218` | 全词表forward/reverse方向、两流全update归一符合本项目选定公式 |

**核心问题（P1）是旧reference与当前teacher之间缺少一致性门禁/实时验证。** 在冻结target、贪心解码且缓存轨迹与当前相同数值路径逐token相同时，缓存正确AR可能是一种有效节省计算的实现；不能因为“用了缓存”就直接判离线错误。但这里没有建立该等价性，且既有真实诊断已表明条件不成立。因此不能使用“target冻结，所以旧轨迹天然等于当前无损轨迹”的理由。

次要数据契约：`build_video_trace`只比较重新编码prompt的长度，不逐token断言与旧records前缀相同。历史70条实际重放已验证prompt相同，故不是已证实的本轮输入错误；未来入口仍应补充完整ID、帧/processor指纹检查。

## 3. 对照官方实现

官方仓库：[Simplified-Reasoning/Draft-OPD](https://github.com/Simplified-Reasoning/Draft-OPD)。本机逐行核对版本为 `d6f0ec859321959785c65b59eb311fd25f042158`，所查三个文件无本地tracked修改；源码快照和SHA保存在审计产物。此次也联网核对官方README，其描述是从真实投机验证暴露的draft-induced states取得反馈，并另列TV/first-reject-only等分支，不能把所有版本称为同一个loss配方。

- `verl/verl/trainer/ppo/ray_trainer.py:1838`调用rollout生成，`:2075`更新actor，`:2112`同步新权重。
- `sglang-dflash/python/sglang/srt/speculative/dflash_worker.py:1325`调用当前target的`forward_batch_generation(..., is_verify=True)`；`:1339`附近由`verify_input.verify(logits_output=...)`得到提交token、commit lengths、下一段target hidden，再更新下一轮状态。
- 本地GLANCE trainer的`rollout`没有这一步，传入的是旧`seq_ids`；后续teacher loss前向不能追溯性地替代真实验证与状态推进。

**loss不是官方代码逐项复刻，但这与在线闭环缺口是两个问题。** 所查官方main版本含采样token/Bernoulli forward KL、k3 reverse估计及池化加权分母；本项目按此前选定的论文Eq.8–13语义使用全词表KL、整段拒绝后缀和两流分别归一。官方又有TV+first-reject-only扩展。全词表形式不因此自动算数学错误，但不能称“与官方训练实现相同”。更不能通过更换成TV或reverse KL来掩盖旧reference问题。

训练使用block16 chain、部署评测b63k8树也是需要保留的配置差异；chain训练本身不自动构成实现bug。

## 4. 本次实际验证

产物：`/mnt/local/mcy/GLANCE/results/opd_implementation_audit/20261013/`。

- 全词表KL CPU测试16/16，trainer规划/归一CPU测试17/17，rollout纯函数测试3/3通过。测试检验局部算术，不证明生产闭环正确。
- 合成反例：固定当前draft预测15个token都与旧reference一致，计划标为15个accepted；构造当前teacher第一位置即不同，其leading agreement为0，但现有plan仍保留15个accepted。反例执行实际`anchor_from_logits`、`build_anchor_plan`，不加载模型。结果为设计契约反例，不是实际数据集错误率。
- 重新从C5 `completed_tree_audit/trajectory_ar_alignment.json`与`d1_reference_mismatch.json`求和：70条旧轨迹中69条与当前部署AR分叉；1680 anchors仅883个在共同prefix；58个anchor的旧接受长度与同旧prefix下当前teacher leading agreement不同，44个旧accepted位置teacher不匹配、16个旧first-rejected位置teacher匹配。该统计是历史诊断集合，不等于C4全部训练数据的标签错误率；A-old条件和正确AR条件不混同。
- 最初尝试pytest，环境未安装，原报错保留在`tests.log`；改用已有standalone CPU测试入口与直接调用3个无fixture测试，未安装依赖。
- 本次真实模型调用0、训练0、checkpoint更新0，未修改训练/部署代码。源码SHA、官方commit、C4配置、日志和合成反例脚本已保存。

## 5. 达到用户要求必须补齐的闭环（尚未实施）

1. 从视频+prompt出发，用当前draft提出block，再用冻结的当前target、固定精度与cached数值路径真实验证；提交matched prefix与target bonus，并据此推进target/draft KV和视觉位置状态。
2. 同一次验证提供proposal、接受mask、拒绝后缀及teacher feedback，分组与teacher不能来自两套reference。拒绝后的teacher须条件于真实draft-induced prefix；第一拒绝位置之前仍是匹配prefix。
3. 每次参数更新后下一批rollout确实使用更新后的head；不把固定anchors或旧response重放冒充在线验证。若保留缓存加速模式，必须先有同路径AR逐token等价门禁，否则明确标记off-policy并阻止以strict-online身份启动。
4. 冻结要复现的官方分支/commit和loss配方；数学形式差异、归一与实际teacher成本要显式记录。先做只读rollout/teacher/无损一致性门禁，再在另行明确训练任务中开展同视频、同监督与teacher预算的SFT/OPD比较，沿用fp32主权重。

因此，当前若考虑恢复OPD训练，优先级应是先修正并验证在线数据/验证闭环，而不是增加updates、换loss或从当前QA弱信号直接重训。此前EOS修复仍是独立的推理评测问题，不能代替这项训练语义修复。本轮没有实施上述改造或启动训练。
