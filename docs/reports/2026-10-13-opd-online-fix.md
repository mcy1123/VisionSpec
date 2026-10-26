# 当前 target 在线 OPD 闭环修复（2026-10-13）

用户授权修改实现；本轮只改代码并做无参数更新验证，不启动训练、不修改 C0/C1/C2 checkpoint。历史 C4 的训练身份仍是混合实现，不能追溯改称严格在线。

## 修改范围与语义

`repro/opd_train_video_fullkl.py` 默认 `--rollout-mode online`，新模块 `repro/opd_verified_rollout.py` 承担验证及状态推进：

1. records 仅提供 task 和 prompt 身份；同时核对长度与逐 token prompt ID。完全忽略旧 continuation 的内容和长度。视频仍使用原 pool question/cache_path/fps 与既有视觉位置路径，未改变缓存时间轴。
2. 当前 head 在当前已提交 context 上生成全块 proposal，当前冻结 target 对 root + proposal 做逐 token causal 验证；接受长度来自当前 teacher argmax，第一处拒绝产生 correction，全接受产生 bonus。
3. KV 裁剪至实际提交前缀，只保留对应融合特征/hidden；最后一个提交 token 作为下一轮 root。拒绝后缀不污染后续上下文。初始 root、接受路径、correction/bonus 的 EOS 均立即终止，支持多个 EOS ID。
4. target 以 token-serial 路径验证，loss pass 也强制 teacher chunk=1；head 每个 anchor 使用 fresh context cache，和梯度重算相同，避开已有 bf16 增量/整前缀投影的数值分歧。head eval 模式关闭 dropout，但保持 autograd。逐 anchor 检查 proposal argmax、teacher argmax 与 n_acc；缺失验证证据或任何漂移均报错，不静默重标。
5. 继续使用原 full-vocabulary 双流 KL、全 update 分母与 fp32 主权重优化器。拒绝后缀按 draft-induced 序列的 EOS 和生成预算确定有效范围，不再要求旧 reference 的未来 token 存在。EOS 本身可监督，后续位置屏蔽。`max_seq_len` 内必须容纳完整验证块，否则停止；`--rollout-max-new-tokens` 默认256，含初始 root、EOS 与 bonus。
6. 默认 online 输出目录必须新建或为空；低学习率必须显式选择 fp32 主权重。每次 update 前保存 `online_rollouts.jsonl`，记录逐视频实际 token、停止原因、proposal、teacher IDs、有效 mask 与接受分组；即使 backward 失败也保留此前证据。同一 update 不允许视频重复覆盖 trace；无法凑齐预算时报错，避免 anchorless 数据无限循环。

旧 `opd_online_rollout.py`、旧 2×2 trainer、生产 `glance/decode.py` 和 loss 数学实现未改。`--rollout-mode legacy-reference` 保留历史路径；Phase C/C4 两个 shell 入口、C5 三处旧诊断调用及旧 GPU smoke 显式选择 legacy。新老模式写入配置，不能混称。

这是在线 full-KL 变体，**不是官方 sampled estimator、pooled denominator 或 TV/first-reject-only 的逐项复刻**。本次未修改这些已有目标选择。

## 实测验证

产物根：`/mnt/local/mcy/GLANCE/results/opd_online_fix/20261013/`。原文件备份在 `before/`，GPU 实测时的准确源码在 `gpu_gate_v1/source/`，最终文件哈希见 delivery manifest。

- 原 full-KL CPU：16/16 通过。
- 原 trainer CPU：17/17 通过。
- 新在线 CPU：11/11 测试方法通过，覆盖多种接受/拒绝边界、全部接受 bonus、accepted/correction/bonus EOS、初始/多 EOS、长度裁剪、轮数上限、拒绝后 KV 回退、独立前缀相关 toy AR、旧 continuation 扰动不变、同长度 prompt 漂移拒绝、无旧未来 token 的监督、teacher/student/标签漂移与证据缺失拒绝、dropout 模式拒绝。
- 439 条训练 records dry-run 通过；未使用新测试集。
- GPU：空闲 RTX A6000 GPU3，seed0，fp32 target + bf16 C0 head，两个预先冻结的训练视频，各3轮，max_new_tokens48，max_seq_len2048。没有 optimizer 实例或 step；只对每视频首 anchor 做一次 backward。

| 视频 | 生成 token 数（含初始 root） | 三轮 n_acc | 独立 AR | backward 梯度范数 |
|---|---:|---|---|---:|
| v_-6dz6tBH77I | 10 | 0 / 4 / 2 | 逐 token 一致 | 4.880867 |
| v_-HpCLXdtcas | 7 | 1 / 0 / 2 | 逐 token 一致 | 3.913594 |

6/6 anchor 的 teacher/student/接受标签重放门禁通过，90个有效监督位置，两流均出现；2次 backward 梯度有限且非零，target 无梯度。峰值分配显存 **38.9101 GiB**，target/head 实际前向 **201/12**，包含独立 AR 与 loss 重算，不当作训练吞吐或部署速度。两个视频均因3轮上限停止，真实 GPU EOS 终止未在本样本发生；EOS 边界由 CPU 状态机用例覆盖。

C0 `model.safetensors` 前后 SHA256：`e2526a249a8ef7e1db3a9990589b727541486093372011e1a0ff6a16e06d23d2`；内存参数 version 未改变。没有加载/修改 C1/C2，没有保存新 checkpoint。

GPU 日志保留了现有 torchvision 解码弃用警告与历史 stats 的 requires_grad 标量转换警告，未安装或修改依赖。没有失败 GPU 运行。GPU 通过后只补充了默认 dtype/模式/缺失证据防护、日志和历史调用兼容；核心验证及提交算法不变，最终 CPU 回归与语法检查通过。

## 验证命令

```bash
/home/mcy/.conda/envs/fin_research/bin/python repro/test_opd_verified_rollout.py
/home/mcy/.conda/envs/fin_research/bin/python repro/test_opd_train_fullkl.py
/home/mcy/.conda/envs/fin_research/bin/python repro/test_opd_fullkl.py
/home/mcy/.conda/envs/fin_research/bin/python repro/check_opd_verified_gpu.py --gpu 3 --out /mnt/local/mcy/GLANCE/results/opd_online_fix/20261013/gpu_gate_v1
```

最后一条输出目录已存在，不可原地重跑；复核须选择新的空目录和空闲 GPU。该脚本不会更新参数。

## 适用边界

代码修复与本轮门禁通过，**不等于证明修复后 OPD 收益更好**。历史 C4/QA 数字不变；修复对最终训练效果仍需另行授权的公平实验。

当前是优先保证数值路径一致的实现：每轮多次 target 前向、每 anchor 重建 draft context，成本高于官方 packed worker。38.91 GiB 仅覆盖2视频小规模 forward/backward，没有分配完整 Adam 主状态，不替代未来完整训练显存门禁；实际24 anchors预算和长程吞吐未运行。若将来优化成 packed verify 或恢复增量 draft cache，必须重新通过 AR/重放门禁，不能直接解除漂移检查。生产 decoder 的 bonus EOS 既有问题属于独立任务，本次没有修改。
