# C2: paper-faithful full-KL OPD trainer (`opd_train_video_fullkl.py`)

本文只说明新增入口本身。项目状态、Phase C 的预注册条件和停止条件仍以
[AGENTS.md](../../AGENTS.md) 与[视频 OPD 计划](../../docs/plans/video-opd.md)为准；本文不修改
`repro/README.md`。

**本入口从未端到端运行过。** 下面所有显存数字都是按模型配置与池内长度做的字节核算，不是实测峰值。
需要实测时按计划第 3 条先跑 ≥3 个 full update 并读日志里的 `peak_pass1_gib` / `peak_pass2_gib`。

## 这是什么

`opd_train_video_fullkl.py` 是 Phase C 的 **C2 臂**：独立、paper-faithful 的全词表 KL OPD
训练入口。它**不修改** `opd_train_video.py`（旧 2×2 配方）的既有行为；两者并存，C1 仍是
aligned continue-SFT。

- 目标函数（arXiv 2605.29343 Eq. 8–13）由 `repro/opd_fullkl.py` 实现并已测试；本入口只负责
  取样、forward、update 级归一、反传、日志与存档。
- teacher 走 decode-aligned 的 cached 路径：`opd_fullkl.target_prefill_shared`。**绝不调用
  `common.target_context`**（`use_cache=False`，文本位置隐状态与视频无关，是 A4 修掉的旧路径）。
- prefix 只 fork 不改：每个 anchor 的 teacher 打分都在共享 full-trajectory cache 的裁剪副本上进行，
  共享 cache 的长度变化会被断言拦下。

## 目标与两个显式偏离

```
L_acc = (1/|I_acc|) · Σ_{k∈I_acc} KL(p_k ‖ q_k)        # 全词表，纯均值，无位置权重
L_rej = (1/Z)       · Σ_{k∈I_rej} w_k · KL(q_k ‖ p_k)  # 全词表，整段被拒后缀，w_k = γ^(k-1)
L     = (L_acc + L_rej) / 2                             # λ_acc = λ_rej = 1
```

单流为空则丢弃该流并把剩余流权重归一到 1（全接受 → `L = L_acc`，不是 `L_acc/2`）；两流皆空则该
update 贡献 0，且**不调用** `optimizer.step()`。

**偏离 1：update 级归一（这是本入口存在的核心理由）。** 不能对不同有效位置数的 microbatch 直接取
均值。训练器先算整个 update 的全局分母，再让每个 anchor 按它缩放：

```
loss_mb = (Σ_acc KL / N_acc_total + Σ_rej w_k·KL / Z_total) / 2   # 两流都在（除数固定为 2）
loss_mb =  Σ_acc KL / N_acc_total                                 # 只有接受流
loss_mb =  Σ_rej w_k·KL / Z_total                                 # 只有拒绝流
loss_mb =  0                                                      # 两流皆空
```

关键点：**除数是 update 的、不是 microbatch 的**。同一条 update 里，一个全接受的 anchor 也要除以 2
（它的接受项除以 `N_acc_total`、拒绝项缺失），否则 microbatch 之和不再等于目标函数。这也是必须
两趟的原因：反传之前就得知道全部分母。`microbatch_scaled_loss` / `plan_update_denominators` /
`paper_objective` 是这三个数的纯函数实现，CPU 可测。

官方实现是按 microbatch 共享一个分母地池化 per-token loss，与上式不同；两者不能混称。本入口采用
上面这个按 update 偏移计数归一、两流等权的读法。

**偏离 2：常数学习率。** 论文是 8 epoch 的 cosine。Phase C 只有几十个 update、439 条训练行，
schedule 基本消耗在 warmup 里，也会让 C1/C2 在相同 update 数上更难比较。因此恒为 `--lr`（默认 3e-4），
优化器与 baseline 一致：`torch.optim.AdamW(betas=common.BETAS, weight_decay=common.WEIGHT_DECAY)`，
裁剪用 `common.GRAD_CLIP`（1.0）。

## 控制流

每个 optimizer update：

1. **Pass 1（无梯度，只跑 head 前向）。** 按 `video_order(seed, epoch)` 的固定顺序取视频：编码 → 取池
   记录里的 committed target trajectory → **一次** `target_prefill_shared`（得到 `fused`、`prefix_kv`、
   `vispos`）→ `opd_online_rollout.rollout(..., verify="chain")` 取 anchors。按 rollout 顺序收集，
   直到 `--anchors-per-update`（默认 24）。每个 anchor 用 `opd_fullkl.stream_split(K, n_acc, valid)`
   记下受监督的接受/拒绝偏移集合。
2. 冻结 `N_acc_total` / `Z_total` 与两流是否存在。
3. **Pass 2（有梯度）。** 每个 anchor 一个 microbatch，**按视频分组**处理（见显存一节）：
   - teacher 在 `no_grad` 下：共享 cache 裁剪到该 anchor（`attach_cropped_prefix`）后交给
     `opd_fullkl.teacher_block_logits`，函数内部 fork；共享 cache 长度不变会被断言。
   - student 有梯度：`draft_block_logits(head, target, fused[:, :pos], anchor_pos=pos,
     root_token=full_ids[pos], prompt_len=...)`，**新建 head cache**。
   - `loss_mb.backward()`，然后立刻释放该 anchor 的图，不保留多个 anchor 的图。
4. 梯度裁剪 → `optimizer.step()` → 清零 → 日志 → 存档。

### student 用 fresh cache 而不是 rollout replay

`rollout` 的 head cache 是逐轮增长的，所以 head 的 `fc` 看到的**行数**与这里一次前向不同。位置、
key/value 和因果注意力都一样，是同一个策略；B0 实测差异为 bf16 下 max|Δlogit| 0.25–0.375、
fp32 下 3.4e-05。**接受/拒绝的划分来自 rollout 的 verifier（`n_acc`），不会从这个前向重算**，所以这点
行数噪声不可能把某个偏移从一流挪到另一流。需要与 rollout 逐位一致的场景用
`proto_fullkl_b0.replay_draft_logits`（B0.5 精度门禁用），训练每步不为它付费。

### teacher 的 offset-1 行

`target_prefill_shared` 只在 cache 上留下整条轨迹最后一行的隐状态。为了每个 anchor 都能拿到
**该 anchor 位置**的 offset-1 行（draft-induced prefix 的第一个 offset 就是 anchor 前缀本身），
`build_video_trace` 额外做一次同样的 cached forward（`_prefill_last_hidden`，`no_grad`），得到
`(1, total, H)` 的逐位置最后隐状态。这不是免费的第二次前向，所以立刻用 `assert_prefill_agrees`
把它与 `target_prefill_shared` 留在 cache 上的那一行比对；不一致就抛错，而不是让 teacher 悄悄落到
另一个分布上。

也可以改成"只 prefill 到第一个 anchor、后续 anchor 用单 token 前向把共享前缀推进到 anchor 位置"，
那样每个 anchor 一次额外前向。当前选的是"一次整轨迹 prefill + 逐 anchor 裁剪"，只多一次整轨迹前向，
且逐位置隐状态同时给了 `fused`，实现上更直白。

## EOS / padding / 截断

偏移 `k`（1-based）监督流位置 `pos+k` 的目标 token，且必须同时满足：

- `pos + k < total`：该位置在存储轨迹内（有目标 token、有缓存上下文行）；
- `pos + k <= first_eos`：不监督任何落在轨迹**首个 EOS 之后**的偏移。**EOS 本身照常监督**（预测它是
  草稿头最后要做的事），EOS 之后的轨迹不是一次停下来的解码会产生的；
- 不补 padding：block 永远恰好 `block_size - 1 = 15` 个偏移。放不下完整 block 的 anchor 直接不用
  （`select_anchors`），所以 chain rollout 的尾部不会变成短 block。

无效偏移经 `valid` 掩码**同时**退出两流和两个分母。每条 update 的 `n_eos_excluded` 记录被这套规则
去掉的偏移数。实测该池 499 条里 487 条以 EOS(151645) 结尾，所以这条规则很少真正生效；它是为剩下
12 条和未来数据池准备的。

`--max-seq-len` 是**守卫不是截断**（与 `target_prefill_shared` 一致）：超长行直接抛错。截断会让每个
anchor 的 `pos` 落到 cache 之外，在视频 token 块内部切断还会让 `get_rope_index` 拒绝该 grid。这条训练
用的 records（`opd_pool_warm499_aligned.pt`）里轨迹长度是 781–1982，全部在默认 2048 内；注意这与
`pool_499.parquet` 的 `input_ids_len`（574–1246，只是 prompt 文本长度）不是同一个量。

## 用法

```bash
R=/mnt/local/mcy/GLANCE
cd /home/mcy/Projects/GLANCE/repro
/home/mcy/.conda/envs/fin_research/bin/python opd_train_video_fullkl.py \
  --records $R/data/opd_pool_warm499_aligned.pt \
  --pool    $R/data/pool_499.parquet \
  --init-from $R/checkpoints/head_video439_aligned/step1000 \
  --out $R/checkpoints/head_video439_opd_fullkl_c2 \
  --updates 50 --anchors-per-update 24 \
  --lr 3e-4 --gamma 0.8 --seed 0 \
  --teacher-dtype float32 --head-dtype bfloat16 \
  --max-seq-len 2048 --clip 1.0 \
  --save-every 10 --log-every 1 \
  --log-file $R/logs/c2_opd_fullkl.jsonl \
  --gpu 0
```

计划里 `--steps` 是同一参数的别名（`--updates` 为准）；两者同时传会被显式拒绝，而不是静默取后者。
`--holdout-parquet` 可额外排除行，但 `split == "train"` 过滤才是主守卫：非 train 行进入训练会直接
assert 失败，训练行数也会与池中 train 行数相减后核对（当前 439 行）。

### 先跑 CPU preflight（不需要 GPU）

```bash
R=/mnt/local/mcy/GLANCE
/home/mcy/.conda/envs/fin_research/bin/python \
  /home/mcy/Projects/GLANCE/repro/opd_train_video_fullkl.py \
  --records $R/data/opd_pool_warm499_aligned.pt \
  --pool $R/data/pool_499.parquet \
  --init-from $R/checkpoints/head_video439_aligned/step1000 \
  --out $R/checkpoints/head_video439_opd_fullkl_c2 --dry-run
```

`--dry-run` 不加载任何模型、不碰 CUDA、不建输出目录：只读同一个 pool / records / split 过滤器，
逐行检查 prompt 与轨迹契约（`prompt_len`、`--max-seq-len`、能否容纳完整 block），并打印轨迹长度分布
与 pass-1 会持有的最坏显存。上面的实测输出：439/499 条 train 行、L 781–1982、24 条视频最坏
**12.80 GiB**（bf16 teacher；fp32 teacher 翻倍为 25.5 GiB）。

`--deterministic` 打开 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 与
`use_deterministic_algorithms(True, warn_only=True)`，与既有训练入口相同。视频顺序来自
`video_order(n, seed, epoch)`，是一条 `(seed, epoch)` 的纯函数排列；anchor 顺序是 rollout 顺序；
不依赖 dict/set 迭代。

## 显存预算（字节核算，非实测）

模型（bf16 target / bf16 head，注意 fp32 teacher 时 target 权重翻倍）：

| 项 | bf16 target | fp32 target |
|---|---|---|
| target 权重 | ~16.4 GiB | ~32.7 GiB |
| head 权重 | 1.95 GiB（1048.6M params） | 同 |
| 梯度（head，bf16→fp32） | ~3.9 GiB | 同 |
| AdamW 一阶+二阶（fp32） | ~7.8 GiB | 同 |

**Pass 1 结束时持有的东西（每个 anchor 的来源视频各一份，`--anchors-per-update` 决定条数）：**

以 `--dry-run` 实测的 records 轨迹长度为准（train 439 行：min 781 / median 1128 / max 1982）：

| 项 | 每视频（L=1128 中位 / 1982 最大） |
|---|---|
| `prefix_kv`（36 层 × K+V），bf16 / fp32 target | 159 / 279 MiB（fp32 翻倍：318 / 558 MiB） |
| `fused`（(L+1) × 5×4096，bf16） | 44.2 MiB（最大 77.7 MiB） |
| `prefix_hidden`（(L+1) × 4096，bf16） | 8.8 MiB（最大 15.5 MiB） |

默认 `--anchors-per-update 24`，且池里每行至少 25 个 anchor，所以最坏情况是 **24 条不同视频**：
`--dry-run` 按池内最长的 24 条算出 bf16 teacher 约 **12.8 GiB**，fp32 teacher 约 **25.5 GiB**。这是
pass 1 必须先付、pass 2 期间继续持有的成本，也是 `--anchors-per-video` 存在的理由：把它设为 4 会把
同时持有的视频数压到 ~6，代价是每条视频只取 4 个 anchor。

**Pass 2 的瞬时峰值**（与 anchor 数无关，因为按视频分组、评完即释放）：

| 项 | bf16 teacher | fp32 teacher |
|---|---|---|
| 已裁剪的 prefix 副本（`torch.cat` 会复制切片：2×36×(pos+1)×8×128×2B） | ≤ 18 MiB | ≤ 36 MiB |
| `teacher_block_logits` 内部 fork（`copy.deepcopy`，按 anchor_pos 长度） | ≤ 279 MiB | ≤ 558 MiB |
| teacher logits `(15, V)` fp32 | 8.7 MiB | 8.7 MiB |
| student logits `(15, V)`（lm_head dtype）+ KL/图 | ~50–100 MiB | ~50–100 MiB |

因此 pass 2 峰值 ≈ `target + head + 梯度 + Adam + 最后一条视频的 trace + 上述临时量`：

- bf16 teacher：16.4 + 1.95 + 3.9 + 7.8 + ~0.3 + ~0.4 ≈ **30.8 GiB**；
- fp32 teacher：32.8 + 1.95 + 3.9 + 7.8 + ~0.6 + ~0.7 ≈ **47.8 GiB**。

**结论（预算层面的警告，不是实测）**：默认的 `--teacher-dtype float32` 单独把 target 权重抬到约
32.8 GiB；即使只看 pass 2，加上 head、梯度和 Adam 状态也已经超过计划里 44 GiB 的目标，而 A6000
只有 48 GiB（`max_memory_allocated` 还不含 CUDA context 与分配器碎片，实际只会更高）。**所以
fp32 teacher 不能默认认为跑得起来**：必须按计划第 3 条真跑 ≥3 个 update 实测
`peak_pass1_gib` / `peak_pass2_gib` 后再定臂配置。bf16 teacher 约 30.8 GiB。需要降预算时：
`--teacher-dtype bfloat16`（会改变精度语义，B0.5 未完成前不要混称）、或 `--anchors-per-video`
压低同时持有的视频数。

另外，`rollout` 自己的 head KV cache 在 pass 1 内按轮增长（最长视频约 0.17 GiB）但每轮 `crop` 回
`pos`，它是单视频内的瞬时量。

## 日志与产物

`--log-file` 写 JSONL，每条 update 一行，至少含：`N_acc_total`、`Z_total`、`acc_present`、
`rej_present`、`empty_update`、两类偏移计数（`n_acc_offsets` / `n_rej_offsets`）、`n_eos_excluded`、
每流 KL 均值与最大值、`grad_norm_preclip`、`clip_scale` / `clip_fraction`、`pass1_s` / `pass2_s`、
`pass1_held_gib`、`peak_pass1_gib` / `peak_pass2_gib`、`empty_update_fraction`、
`teacher_dtype` / `head_dtype`（实际 dtype）。

输出目录只有三类产物：`--save-every` 与最终 `save_pretrained` 的 checkpoint、JSONL 日志、
`opd_fullkl_config.json` + `opd_fullkl_summary.json`。**不落盘任何 logits / 分布 / KV。**

## 测试

```bash
cd /home/mcy/Projects/GLANCE/repro
/home/mcy/.conda/envs/fin_research/bin/python test_opd_train_fullkl.py
```

17 个 CPU 测试覆盖：update 级归一的求和等价（含混合接受数、`gamma ∈ {0.5,0.8,1.0}`、两流皆空的三种
情形）、分母确实是 update 级的反例、单流 anchor 在混合 update 里的缩放、无效偏移同时退出两流与两个
分母、anchor/视频顺序与预算确定性、EOS 与 `--max-seq-len` 的资格判定、`stream_split` 与 `valid` 一致、
`w_k` 调度、单 anchor 的 update 目标等于 `opd_fullkl.opd_fullkl_loss`、CLI 面、`--dry-run` 的 CPU 独立性
与真实池行数，以及池的 split 守卫。当前实测：**17 passed, 0 failed**。

模型相关路径（`build_video_trace`、`attach_cropped_prefix`、`score_microbatch`）需要 GPU，在
`--gpu-smoke` 下，**默认不运行**（本轮不得占用 GPU）。`attach_cropped_prefix` 的 cache 构造逻辑已用
CPU 上的假 cache 手工验证过（1/4/6 行裁剪、追加后源 cache 长度不变），但没有真模型。
