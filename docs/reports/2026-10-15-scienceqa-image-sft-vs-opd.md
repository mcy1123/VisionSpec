# 2026-10-15 GLANCE × ScienceQA：修复版图像 SFT / OPD 公平对照（普通负结果）

**结论：在冻结同一个 Qwen3-VL-8B-Instruct Target、同一个图像 warm Draft 起点、同一批
ScienceQA 图文题、同一 Prompt 与同一 b63k8 验证器的条件下，修复位置契约后的正确在线
OPD **没有**跑赢同预算继续 SFT。两个配对 seed、dev 与冻结 test 上 OPD−SFT 全部为负且
CI 不跨零；同期继续 SFT 相对起点 I0 有稳定约 +0.33 τ 的正增益。按预注册的「普通负结果」
分支关闭这套 ScienceQA 全词表 KL 配方。**

完整产物：`/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/`，
其中唯一权威报告是
[reports/FINAL_SCIENCEQA_SFT_VS_OPD.md](/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/reports/FINAL_SCIENCEQA_SFT_VS_OPD.md)
（由 `make_final_report.py` 从产物直接生成，数字不得手改）。

## 1. 数据入口审计：过去确实「没下载全」

| 目录 | 真实内容 | 行数 | 官方 split 归属 |
|---|---|---:|---|
| `data_sqa_train` | 官方题干 + 选项 + 强制推理指令 + 官方 `solution` | 2,400 | **全部官方 train** |
| `data_sqa_test` | 同上 | 2,017 | **全部官方 test** |
| `data_sqa_full` | 同上 | 4,417 | train 2,400 + test 2,017 的并集 |

- 连接键**不是目录名**，而是 `sha256(图像字节) + question + choices + answer` →
  `(官方 split, 官方行号)`；8,834 张本地 PNG 全部在固定 revision 的官方 parquet 中匹配到
  （`unmatched_rows = 0`）。
- 官方 IMG 可用量为 train 6,218 / validation 2,097 / test 2,017。本地**缺** train 的
  3,818 条与 **validation 全部 2,097 条**。`data_sqa_full` 不是官方全量，也不含 val。
- 官方 `pid` 字面串本机不可得：`github.com` / `raw.githubusercontent.com` 实测不可达，
  可达的四个 HF 镜像都没有 `pid` 列。已改用更强的图像字节键并记录该限制。
- 官方数据大量复用同一张图且**跨 split**（2,797 条记录与另一 split 共享图片）。
  只按图像 SHA 会误报 235 条「train∩test」；按完整记录键为 0 条。构建时对跨 split
  图像簇**整簇排除**。

## 2. 冻结数据与门禁

- 来源固定：`lmms-lab-encoder/ScienceQA@69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50`
  的 ScienceQA-IMG train/validation/test parquet，落盘 `/mnt/local2` 新目录。
- 选样：`split_seed=20261015`，按 `subject × 图像尺寸三分位` 分层 + 最大余数配额 +
  `sha256(split_seed:pid)` 确定性抽取。结果 **train 1200 / dev 240 / test 240**，
  逐条真实可解码，跨集合同图簇 **0**，跨集合重复记录 **0**。
- 冻结 prompt（SHA256 `b97d0b01826685f8…`）只含题干 + `(A)…(E)` 选项 + 一句
  「先推理再以 `Answer: (X)` 结尾」；**不输入** gold `answer`/`lecture`/`solution`/`hint`。
- 许可：ScienceQA 官方 **CC BY-NC-SA 4.0**；HF 镜像页面标的 `cc-by-sa-4.0` 少了 NC，
  本实验以上游为准。

## 3. 阶段 C/D 门禁（全部通过）

| 门禁 | 结果 |
|---|---|
| 图像绑定 | 3 张真实图：换图/置零像素均使 fused hidden 显著变化（末行 max-abs 20.3/14.5/25.8，公共前缀最高 7.5e3），next-token logits 变化，续写在 step 3/28/0 分歧 |
| 长度探针 | 32 条 train：p10/p50/p90 = 83/121.5/227.6 token，自然 EOS 31/32，>256 占 6.2% ⇒ 冻结 `max_new_tokens=256` |
| AR 无损 | b63k8、EOS immediate：SPEC == 独立贪心 AR **8/8 逐 token**，τ=3.664，AR 65.0 vs SPEC 38.2 ms/tok |
| 位置契约复用 | 图像训练器由冻结视频训练器**生成**；反向替换 13 类改动后与冻结源文件**逐字节相等**；24/24 CPU 测试 + 冻结 13/13 + 4/4 通过 |
| 无更新 GPU 重放 | 2 张真实图 × 3 anchors：编码 prompt 与冻结记录逐 token 相等、target replay 复现冻结 AR、规划 `n_acc` == 在线 teacher 重算、梯度 finite 非零、参数逐位未变、`optimizer.step()` 未调用；峰值 38.85 GiB |
| 优化器/恢复 | fp32 master 与 Adam 两阶矩**逐位恢复**，bf16 镜像 == 已发布头；checkpoint 含 `rng`(torch/torch-cuda/numpy/python) + `data_cursor` + `schedule` |

**发现并修复的真实陷阱**：图像 AR 标注若在 prefill 显式传 `position_ids`，Qwen3-VL 不会
写入 `rope_deltas`，后续单 token 解码位置错乱、轨迹退化成 2-token 重复。改用部署路径
（`VisionPositions.ar_prefill_inputs`）后前 12 条 debug 记录与 teacher-forced 全序列前向
**0 处分叉**；`build_training_data.py` 另加断言，禁止把训练器会拒绝的标签冻结进数据。

## 4. 预算修订（训练前登记，对两臂 × 两 seed 一律适用）

原计划 300 updates / 1200 QA 曝光 **未执行**。`amendments.jsonl` 第 1 条在**任何正式训练之前**
登记改为 **120 updates / 480 QA 曝光**，检查点网格 `{30, 60, 120}`，主终点 step120。

- 理由：fp32 AR 标注约 13 秒/条（300-update 方案需 1680 条 ≈ 5 h 独占 GPU）；
  三个 `/mnt` 卷均 98–99% 满（全量 head+master+Adam 检查点约 16 GiB/个）；
  8 张卡中最空闲的 A6000 也常被别人占满，而 fp32 target 本身需约 33 GiB。
- **实测训练只要 7.6–10.8 秒/update**（120 updates ≈ 16–22 分钟/臂），比开工前 smoke 测的
  18.8/22.6 秒更快 ⇒ 算力不是实际约束，标注与存储才是。该结论写入
  `amendments.jsonl` 第 2 条（测量附录），**预算未据此回调**，避免看到训练动态后改预注册。

## 5. 真实训练预算

| arm/seed | updates | 独特 QA | anchors | 监督位置 | teacher 前向 | N_acc | Z_total | 接受为 0 的 update | 峰值 GiB | 秒/upd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sft/20261013 | 120 | 480 | 2,459 | 23,834 | 0 | — | — | — | 38.84 | 8.0 |
| opd/20261013 | 120 | 480 | 2,880 | 36,450 | 2,880 | 5,747 | 8,070.7 | **0** | 38.47 | 10.6 |
| sft/20261014 | 120 | 480 | 2,461 | 23,587 | 0 | — | — | — | 38.84 | 7.6 |
| opd/20261014 | 120 | 480 | 2,880 | 36,450 | 2,880 | 5,732 | 8,072.6 | **0** | 38.47 | 10.8 |

两臂监督位置数不同（SFT 由 variable-reveal 决定，OPD 由 accepted/rejected 分流决定），
teacher 前向只有 OPD 支付；**不能因 updates 相同就宣称 FLOPs 相同**。
注意 OPD 两 seed 都**没有任何一个 update 的接受数为 0**，与历史视频 QA 的「accepted 流
几乎为空」不同。

## 6. 主结果（真实在线 b63k8，逐 token 无损）

`max_new=256`、EOS immediate、fp32 target + fp32 head、`max_pixels=768`。
所有 19 次评测 **全部逐 token 无损**（240/240 或 120/120），且各臂在同一 split 上截断
记录数与自然 EOS 数**完全相同**（dev 38/202，test 42/198）⇒ 无 EOS/截断混淆。
三个 head 的最终可见答案准确率也**完全一致**（dev 0.9163、test 0.9242），符合无损投机
解码不改变 Target 输出的性质。

| split | seed | 比较 | Δτ | 95% CI（图像簇 bootstrap, 2000×） | 跨零 |
|---|---|---|---:|---|---|
| dev | 20261013 | OPD−SFT | −0.3951 | [−0.4368, −0.3557] | 否 |
| dev | 20261013 | OPD−I0 | −0.0635 | [−0.0980, −0.0291] | 否 |
| dev | 20261013 | SFT−I0 | +0.3316 | [+0.2949, +0.3692] | 否 |
| dev | 20261014 | OPD−SFT | −0.3899 | [−0.4308, −0.3498] | 否 |
| dev | 20261014 | OPD−I0 | −0.0555 | [−0.0902, −0.0202] | 否 |
| dev | 20261014 | SFT−I0 | +0.3344 | [+0.2941, +0.3748] | 否 |
| test | 20261013 | OPD−SFT | −0.3581 | [−0.3942, −0.3190] | 否 |
| test | 20261013 | OPD−I0 | −0.0244 | [−0.0600, +0.0092] | **是** |
| test | 20261013 | SFT−I0 | +0.3336 | [+0.2962, +0.3692] | 否 |
| test | 20261014 | OPD−SFT | −0.3727 | [−0.4133, −0.3299] | 否 |
| test | 20261014 | OPD−I0 | −0.0140 | [−0.0504, +0.0213] | **是** |
| test | 20261014 | SFT−I0 | +0.3586 | [+0.3200, +0.3982] | 否 |

绝对量（图像簇均值 τ）：dev I0 **4.0050** → SFT **4.3366 / 4.3394** → OPD **3.9415 / 3.9495**；
test I0 **4.0723** → SFT **4.4059 / 4.4309** → OPD **4.0479 / 4.0583**。

**step30 灾难退出门禁未触发**：dev120 上 OPD30−I0 = +0.0183 / +0.0186（CI 跨零），
SFT30−I0 = +0.1178 / +0.1456，远未触及 −0.50 阈值；两臂同点继续。
中间趋势一致：step60 时 SFT +0.20~+0.23，OPD −0.06。

## 7. 机制线索（固定 anchor 覆盖探针，240 anchors）

| head | top1 | top8 | L_raw | L_tree | L_actual | tree_gap | Φ |
|---|---:|---:|---:|---:|---:|---:|---:|
| I0 | 0.0875 | 0.7500 | 0.7542 | 0.7500 | 0.8625 | 0.0042 | 2.97 |
| SFT120（两 seed） | 0.8833 / 0.8917 | 0.9917 / 0.9792 | 2.98 / 2.72 | 2.275 / 2.171 | 2.396 / 2.296 | 0.708 / 0.550 | 3.54 / 3.61 |
| OPD120（两 seed） | 0.8167 / 0.8167 | 0.9375 / 0.9375 | 1.91 / 1.87 | 1.654 / 1.633 | 1.775 / 1.754 | 0.258 / 0.238 | **6.58 / 6.48** |

- 三臂相对 I0 都改善了固定 anchor 的候选/树覆盖，但 SFT 改善更大，与在线 τ 排序一致。
- **OPD 的树概率代理 Φ 明显高于 SFT（6.5 vs 3.5），而它把正确 path 放进树的 L_tree 反而更低
  （1.63–1.65 vs 2.17–2.28）。** 也就是「代理目标上升、真实接受行为下降」这一失配在本轮
  方向是**负**的：不能据此宣称方法优势。
- 首阻断原因以 `not_in_topk`(121/240) 与 `past_block`(117–119/240) 为主，
  `in_topk_not_in_tree` 仅 0–2 次 ⇒ 本轮瓶颈主要是**候选缺失**，不是树预算损失。

## 8. 速度口径（不构成跨评测加速结论）

共享主机上 19/19 次评测的 GPU 占用在评测前后发生变化，所有计时标
`timing_contaminated`。**本文只用同一次评测内** SPEC 与 AR 背靠背的倍数：
1.53×–2.17×（同卡、同精度、同缓存、同 warmup）。I0 在 dev 端到端 2.03×、test 1.69×。
SFT 因 τ 更高，端到端 ms/req 从 I0 的 4,983 ms 降到 4,751 ms（dev），
OPD 反而升到 6,426 ms；但这些数字跨评测不可比，**不写成加速成果**。

## 9. 判断与不做的事

- **普通负结果**：不自动切 k3 / Bernoulli KL / TV / P2 / P3，不加训练轮次，不扩数据集。
- 唯一保留的后续机制假设（**不启动**）：
  「同一固定 C0 head、同一 top-8、同一 63 节点预算下，把 best-first 的累计概率配额换成
  按深度固定配额，检验 Φ 与真实接受路径是否脱钩」——本轮 OPD 的
  「Φ 更高但 L_tree 更低」是它的反例边界。
- 结论适用范围：ScienceQA 输出中短（p50≈122 token，约 14–16% 撞 256 上限），
  只能称**短/中等长度图文科学问答的投机解码**，不能称长 CoT 加速。

## 10. 交付与复现

- 产物根：`/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/`
  （`audit/ dataset/ preflight/ train/ eval/ reports/ logs/ scripts/`）；
  选定图像 `/mnt/local2/mcy/datasets/scienceqa_glance_20261015/lmms-lab-encoder_ScienceQA@69dd4d6b/images/`（1,680 张）。
- 脚本：`repro/scienceqa_image/`（审计、建数据集、冻结预注册、预检、门禁、两个图像训练器、
  评测、覆盖探针、配对分析、报告生成、交付清单）。
- 命令流水：`<run>/commands.txt`；过程日志：`<run>/logs/`；
  逐文件 SHA256：`<run>/reports/DELIVERY_MANIFEST.json`。
- 历史产物未改：三个原始目录只读、`checkpoints/head` 逐位未变
  （SHA256 `fc0ad5a6…`）、Task B 与第五轮 QA 结果目录未触碰。
- 被清理的只有**本轮自己的**中间产物，且逐个记录了哈希与理由
  （`<run>/pruned_artifacts.json`）：smoke 的 `step0`/`step10`，以及四个正式 arm 的 `step0`
  —— 四份 `step0` 的 `model.safetensors` SHA256 与 `checkpoints/head` **全部相等**（逐字节相同），
  清理只为把本轮推到 100% 占用的三个共享 `/mnt` 卷释放约 32 GiB。
  **注册检查点 `step30/60/120`、全部日志、门禁与结果一个未删**，起点可完全复现。
