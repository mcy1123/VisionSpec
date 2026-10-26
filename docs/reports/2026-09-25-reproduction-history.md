> 历史记录（截至 2026-09-25），由旧 repro/README.md 保留。当前状态以 [AGENTS.md](../../AGENTS.md) 为准。文中的“进行中”、视频留出结论和后续计划可能已被 A/A4 与 2×2 报告替代；ChartQA 主口径为原始问题，3.36× 为 decode-only 加速。文中裸路径以项目根目录或明确给出的产物根目录为基准。

# GLANCE 复现：TextVQA+COCO 训练 → ChartQA 接受率与加速比

论文：*Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative
Decoding in Vision-Language Models*（arXiv `2609.00355`）
代码：`github.com/js-lee-AI/GLANCE`
目标：按论文的配方训练 block head，在 ChartQA 上复现
**τ = 5.16 / 3.42×**（Table 10, budget 63）与 **τ = 4.62 / 2.93×**（Table 1）。

---

## 0. 结论：τ 与加速比都复现了

评测口径与论文 Table 10 一致：batch 1、greedy、256 new tokens、decode-only 计时、
τ 跨 round 池化、首样本作 warm-up 丢弃、budget sweep 在同一进程内共享 AR baseline。

**Qwen3-VL-8B + 本复现训练的 head，ChartQA，A6000：**

| budget N | τ 实测（95% CI） | τ 论文 | 偏差 | speedup 实测（95% CI） | speedup 论文 | 偏差 |
|---|---|---|---|---|---|---|
| 15 | **4.54** [4.37, 4.74] | 4.43 | +2.5% | **2.96×** [2.90, 3.15] | 3.07 | −3.6% |
| 31 | **4.86** [4.69, 5.09] | 4.81 | +1.0% | **3.18×** [3.12, 3.39] | 3.30 | −3.6% |
| 47 | **5.07** [4.91, 5.34] | 5.00 | +1.4% | **3.32×** [3.25, 3.54] | 3.40 | −2.4% |
| **63** | **5.20** [5.03, 5.45] | **5.16** | **+0.8%** | **3.36×** [3.29, 3.57] | **3.42** | **−1.8%** |

**每一个 τ 都在论文的 2.5% 以内，每一个 speedup 都在 3.6% 以内。**

Table 1（τ=4.62 / 2.93×）是 **SGLang 0.5.6 + 32 draft tokens + RTX PRO 6000** 的口径，
本机没有该引擎与卡；同一协议下的 budget-31 实测为 τ=4.86 / 3.18×。

### 另外三项独立复现

| 论文出处 | 论文值 | 本复现实测 |
|---|---|---|
| Table 9 树 vs 链（零混淆对比） | 1.36×（README 说各任务 1.45–1.49×） | **1.44×**（τ 5.20 vs 3.61） |
| Table 7 / App.D fp32 lossless gate | 逐 token 完全一致 | **20/20 identical** |
| §6 可草拟性定律 `holds()` | 四条判据 | **True**（b0 1.60 / b1 1.43 / R²_round 0.131，论文 1.50 / 1.71 / 0.138） |

### 两条内部一致性证据（不依赖论文数字）

- **prompt 数无关**：n=100 给 τ=5.20，n=20 给 5.18。τ 是跨 round 池化的量，
  本就应与 prompt 数无关；它真的无关，说明池化口径写对了。
- **分辨率无关**：评测用 896px（仓库默认）与 768px（App.D 训练侧）都给 τ=5.20。

### 最重要的一处判断：ChartQA 的 prompt 形态

论文只说 "ChartQA under its templated prompts"，**模板本身没有公开**。两个候选差得很远：

| prompt | target 生成长度 | 轮数/题 | τ @63 |
|---|---|---|---|
| 原始问题 | 中位 **136** token | 28 | **5.20** |
| `Answer ... using a single word or phrase. {q}` | **2–5** token | 1.15 | 2.65 |

τ 的定义就是「每轮提交的 token 数」，答案只有 2 个 token 时 τ 数学上不可能超过 2~3。

主口径取**原始问题**，依据是两条**互相独立**的证据，而不是「哪个数好看」：

1. τ 吻合（5.20 vs 5.16）；
2. **生成长度吻合** —— 由论文 Table 12（ChartQA 记 2190 轮、mean a=3.709）反推约
   **103 token/题**，只有原始问题能产生这个量级，模板版差一个数量级。

第 2 条在计算 τ 之前就能得出，构成独立佐证。两个模板的结果都保留在
`results/chartqa_head_shortanswer.json` 与 `REPORT.md` 里。

### 剩余偏离（`$GLANCE_ROOT/CONFIG.md` §8 共 11 条，全列）

最要紧的三条：

1. **reveal 先验底数 β 无一手出处** —— D2SD 只写 `β ∈ (0,1)`，GLANCE 不提。
   取 NeMo 默认 **0.9**（平均 reveal 6.11、P(reveal=1)=0.126），敏感度表在 CONFIG.md §8-10，
   调节开关 `train_head.py --prefix-base`，应急重训脚本 `scripts/19_contingency_beta.sh`。
2. **micro-batch 论文未给** —— 本实现 1 条序列 × accumulation 8 × 2 卡 = global batch 16。
3. **训练卡** —— 目标是 GPU 3 与 6，但 GPU 6 被他人任务占用、可用显存 8–38 GB
   低于实测峰值 ~40 GB，故自动回退为 **GPU 3 + 空闲 A6000**（`scripts/print_gpus.sh`
   按实际空闲显存选卡，GPU 3 恒为第一）。这是按测量结果做的选择，不是随意替换。

配置的逐项溯源见 `$GLANCE_ROOT/CONFIG.md`。

## 1. 目录

```
/home/mcy/Projects/GLANCE/repro/          ← 代码（本仓库）
  common.py                 路径、加载器、编码、常量、视觉 patch-embed 修复
  build_train_prompts.py    8000 COCO + 8000 TextVQA prompt
  fetch_coco_images.py      只下需要的 8000 张 COCO 图
  build_eval_data.py        ChartQA 评测集（模板可切换）
  gen_target_greedy.py      target 自己的 greedy 生成（训练数据来源）
  train_head.py             DFlash variable-prefix 目标训练器
  test_forward.py           自研 forward 与官方 forward 的数值一致性校验
  eval_chartqa.py           Table 10 口径的 budget sweep
  audit_lossless.py         仓库自带的 fp32 lossless gate 封装
  parallel_fetch.py         多连接分块下载（单连接只有 ~3MB/s）
  extract_textvqa.py        TextVQA 图包 CRC 校验 + 解压
  stats.py                  训练语料统计 + 监督量估算
  report.py                 结果 vs 论文对照表（含 bootstrap 95% CI）

/mnt/local/mcy/GLANCE/                    ← 产物（模型/数据/checkpoint/日志）
  models/        Qwen3-VL-8B-Instruct (16.3GB), Qwen3-8B-DFlash-b16 (2.0GB)
  datasets/      coco, textvqa, chartqa, coco_urls
  data/          train_prompts*.jsonl, target_greedy*.jsonl, vl_eval/
  checkpoints/   head/
  results/       chartqa_*.json, audit_*.json, REPORT.md
  logs/          每个阶段一份日志
  scripts/       编排脚本（tmux 里跑）
```

## 2. 复现流程

```bash
R=/mnt/local/mcy/GLANCE
# 端到端（生成 → 门禁 → 训练 → 评测），tmux 里无人值守跑：
tmux new -s glance -n pipeline "bash $R/scripts/20_pipeline.sh 2>&1 | tee $R/logs/20_pipeline.log"
```

分步：

| 步骤 | 命令 | 说明 |
|---|---|---|
| 模型 | `scripts/01_download_models.sh` | 经 hf-mirror（本机代理只有 ~90 B/s） |
| 数据 | `scripts/02b_download_data.sh` | TextVQA 图包 / COCO / ChartQA |
| prompt | `repro/build_train_prompts.py` | 8000 COCO + 8000 TextVQA |
| COCO 图 | `repro/fetch_coco_images.py` | 只下 8000 张（1.3GB）而非 18GB 全量 |
| ChartQA | `repro/build_eval_data.py` | parquet 自带图片，2500 题 |
| 生成 | `TAG=coco PROMPTS=... scripts/10_run_gen.sh` | 2 卡 × 4000 行 |
| 训练 | `scripts/11_run_train.sh` | torchrun ×2, `CUDA_VISIBLE_DEVICES=3,6` |
| 评测 | `HEAD=... scripts/13_run_eval.sh` | budget sweep + chain 消融 + fp32 gate |

## 3. 配方（全部出自论文，详见 CONFIG.md）

| 项 | 值 | 出处 |
|---|---|---|
| target | Qwen3-VL-8B-Instruct，冻结 | §5.1 |
| head | DFlash block head，5 层 / B=16 / 1.05B / kept layers {1,9,17,25,33} | Table 5 |
| warm start | `z-lab/Qwen3-8B-DFlash-b16` | Table 5, §4.3 |
| 数据 | target 自己的 greedy，256 new tokens，8000 COCO-Caption2017 + 8000 TextVQA = 16K teacher-forced | Table 5, §4.3 |
| 图像 | 训练长边 768 px | App.D |
| 目标函数 | coverage CE + `0.8^offset` 权重 + variable-mask block prefix（= DFlash variable-prefix 目标） | Table 5, §4.3 |
| 优化 | AdamW lr 4e-5, betas (0.9,0.95), no wd, clip 1.0, 1 epoch, accum 8, bf16, seed 0 | Table 5, App.D |
| 评测 | batch 1, T=0, 256 new tokens, decode-only 计时, τ 跨 round 池化, 首个样本作 warm-up 丢弃 | §5.1, README |

## 4. 过程中发现并修掉的两个环境问题

这两条都不是论文的问题，是本机的，但**不修就跑不出论文的数**。

### 4.1 视觉塔的 Conv3d 慢了 6 万倍（`common.use_fast_patch_embed`）

`fin_research` 环境（torch 2.9.1+cu128 / cuDNN 9.10.2）在
`(N, 3, 2, 16, 16)` 这个形状上给 Conv3d 选了一个灾难性算法：

```
Conv3d patch-embed     9381 ms      ← 1040 个 patch
F.linear 等价形式        0.03 ms     ← 同样的权重
```

视觉塔除了这层就只有 27 层小矩阵乘，所以这一个卷积**就是**整个 forward：
一次 8 图 batch 从 **84 秒**降到 **1.36 秒**（62×）。同卡另外两个环境
（torch 2.8.0+cu128 同款 cuDNN、torch 2.5.1+cu124）都是 0.14 ms，所以是
这个 build 的问题，不是模型的问题。

修法是**等价的**，不是近似：kernel = stride = 输入尺寸，这层卷积就是把每个
patch 展平后做一次矩阵乘，即 `F.linear(flat, weight.view(out, -1), bias)`。
`use_fast_patch_embed()` 在替换前会先在小 probe 上比对两种形式，不一致就拒绝打补丁。

### 4.2 TextVQA 图包被写坏（重下）

早期该图包的下载同时跑了两个续传进程写同一个 `.part`，结果解压时报
`zlib.error: invalid block type`。改成 `parallel_fetch.py` 分 16 块并发下载
（单连接 ~3MB/s → 并发 ~14MB/s），并对 zip 做完整 CRC 校验后才解压。

### 4.3 一次自己造成的数据损失（记录在案）

重启管线时，第 3 步会重新执行 `10_run_gen.sh`，而该脚本当时**没有传 `--resume`**，
`gen_target_greedy.py` 于是以 `"w"` 打开输出文件，把已经生成的约 6000 行 TextVQA
截断归零。同时 `--resume` 这个能力其实上一轮就已经实现了，只是没接到启动器上。

修法与两条通用教训：

1. **阶段必须幂等**：`10_run_gen.sh` 现在恒定传 `--resume`，重跑只会补差，不会毁数据。
   判据是输出文件以 `[a]`（append）打开，日志里会打印 `[resume] N rows already done`。
2. **不要原地修改正在运行的 shell 脚本**。bash 按字节偏移增量读取脚本，运行中改写会导致
   任意错乱——这与上面那个 bug 同源。因此合并逻辑被抽到独立的
   `repro/merge_generations.py`，追赶逻辑写成新文件 `scripts/16_catchup_textvqa.sh`，
   都不触碰任何正在执行的脚本。

## 5. 关键实现说明：为什么不用官方 forward 训练

`dflash` 的 `DFlashDraftModel.forward` 把 context 和 block 拼成一条序列：
key 是 `cat([k_proj(context), k_proj(block)])`，RoPE 从 0 开始编号，query 取
**最后** `q_len` 个位置。单个 block 因此能正确前向，但每个 anchor 都要重算
一次 `fc`（20480→4096）和整段 context 的 K/V，而这两项就是全部开销。

一条序列的所有 anchor 共享前缀，所以它们的 context K/V 互为前缀。本实现
**每条序列只投影一次 context**，用显式 mask 把该序列所有 anchor 放进一次
forward。`test_forward.py` 校验这个自研 forward 与官方 forward 一致：

* fp32：最大相对差 **3.5e-6**（数学等价）
* bf16：hidden cosine **0.99974**（五层舍入的残差）

## 6. 与论文的偏离

`CONFIG.md` §8 列了全部 9 条。最要紧的几条：

1. **ChartQA 模板没有公开**（论文只说 "templated prompts"）。主口径用
   `Answer the question using a single word or phrase. {q}`，并做敏感性对照。
2. **micro-batch 论文未给**，本实现 1 条序列 × accum 8 × 2 卡 = global batch 16。
3. **评测卡不同**（论文 H100 NVL / RTX PRO 6000，本机 A6000）。论文自己说明
   τ 与引擎无关、speedup 只做同卡比值，所以 τ 可比、speedup 只能同卡自比。
4. **COCO 2017 官方 annotations 在本网络 404**，改用
   `ChristophSchuhmann/MS_COCO_2017_URL_TEXT`（同样的 591,753 caption / 118,287 图）。
5. **训练 768 px / 评测 896 px**（后者是仓库 `main_table.py` 的默认值），
   评测时两档都跑以显示敏感性。

## 7. 复现性检查点

| 检查 | 期望 | 状态 |
|---|---|---|
| AR baseline ms/token | 论文 24.23（Table 1） | **23.88** ✓ 差 1.5% |
| 自研 forward vs 官方 forward | 数学等价 | fp32 3.5e-6 ✓ |
| ChartQA fp32 lossless | 论文 60/60 完全一致 | 见 `results/audit_head.json` |
| ChartQA τ @ budget 63 | 5.16 | 见 `results/REPORT.md` |
| ChartQA speedup @ 63 | 3.42× | 见 `results/REPORT.md` |

---

## 8. 方法改进线：DSpark 的 Markov 偏置（进行中）

复现完成后转入方法改进。目标是 GLANCE 自己承认为**假设**的那一点：
payload 块内 token 条件独立（Assumption 2）。DSpark（arXiv 2607.05147）把由此产生的
损失命名为 "multi-modal collision"，并用一个低秩 logit 偏置 `B = W1 @ W2`（rank 256，
以上一块 token 为条件）修它。证据链与工具如下。

### 8.1 先证明缺的是「条件」而不是「容量」

`probe_prefix_conditioning.py`（ChartQA，block 16，n=20）：

| offset | 解码时 | 给真前缀 | 增益 | 给自采样前缀 |
|---|---|---|---|---|
| 4 | 0.304 | 0.481 | +0.177 | 0.304 |
| 8 | 0.216 | 0.447 | +0.231 | 0.226 |
| 15 | 0.083 | 0.255 | +0.171 | 0.094 |

给**正确的**前缀，深层 offset 准确率翻倍 ⇒ 信息在表示里，缺的是条件化，值得训一个
顺序模块。而给**自己的**前缀几乎不恢复（0.226 vs 0.216）⇒ 未经训练的就地精修没用，
这解释了 `refine_decode.py` 零训练方案为什么是 −0.4% τ。

### 8.2 `steer` 钩子：机制如何进入候选树

`glance/decode.py` 的 `generate()` 新增可选参数 `steer`：一个
`(draft_logits, root) -> draft_logits` 的普通可调用对象，在 top-k 与建树**之前**
改写每 offset 的边缘分布。树的分路径得分因此从「独立边缘之积」变成
「沿一条已实现前缀之积」——正是 Assumption 2 想要却无法表达的量。
传 `None` 时该分支完全跳过，草稿 pass 与原版逐位相同。
把机制做成**通用钩子**而不是把 Markov 代码塞进仓库，是为了让 vendored 仓库保持干净、
并避免同一份实现在两个文件里漂移。

`test_steer_hook.py` 钉住四条性质（全部通过）：

| 检查 | 结果 |
|---|---|
| `steer=None` / 恒等 / 零偏置 三者逐位相同 | ✓ 未启用时确实无副作用 |
| 放大 50× 的 Markov 改变**接受长度日程** | ✓ 钩子真的到达了树得分 |
| 放大的 Markov 下输出 token **仍然完全相同** | ✓ 无损性与草稿器无关 |
| `test_markov.py` 5 项 | ✓ |

**一条设计教训**：这个测试的第一版断言「输出 token 应不同」，结果失败，而机制其实是好的
（τ 4.4000 → 4.3000 而 ids 不变）。原因是 GLANCE 无损：目标总是按自己的 argmax 提交 token，
不同的草稿树只改变**每轮接受几个**，从不改变**提交哪些**。所以敏感的仪器是接受长度日程，
而 ids 相同是**无损性检查**，不是失败的征兆。

### 8.3 判据修订：撤回一个自己读错的次判据

原 `EXEC` 文档的次判据写的是「逐 offset 比值 0.6 → ≥0.72」。这个 0.6 是**误读**：
它是曲线上几个采样点的比值（off1→off4 跨 3 个 offset、off4→off8 跨 4 个、off8→off15 跨 7 个，
依次 0.50 / 0.71 / 0.38），跨的步数都不同，本就不可能是「每 offset 因子」。
实测的逐 offset 几何衰减是 **0.8675**（385 anchors × 20 prompts）。
决定性自洽检验：若真按 0.6 衰减，off8 应为 0.017、off15 应为 0.000，实测 0.216 / 0.083。

**0.72 这个门槛在任何训练发生之前就被基线越过了**，作为判据是空的。
故将其**撤回**（在 Arm M 仍跑在 step 30/1000、尚未见到任何 τ 数字时撤回，不是事后调门槛），
换成 `measure_offset_curve.py`：对任意 head 在**同一条件**下测逐 offset 曲线
（l=1 块、greedy、head 带 `markov_head.pt` 时启用引导），落盘每个 (prompt, anchor) 的对错，
可做**按 prompt 聚类的配对 bootstrap**。基线 **g = 0.8675**，且逐 offset 计数与旧探针
逐格完全一致（234/385, 201/385, …, 32/385）——脚本正确性由此锚定。
判据权重仍只由主判据（τ @ b63 5.20 → ≥5.50）与原有否证条件承担。

**方法学副作用**：该统计量对 anchor 步长极敏感（`--stride 16` 时 off1 = 1.000 且其后 56 次
全错，`--stride 4` 时 off1 = 0.608），故跨 head 比较必须锁死 stride / n / gen / max-pixels
（脚本已把四者写进 JSON）。

### 8.4 待解决的量级问题（这就是下一步要测的）

Markov 偏置在参考实现（DSpark）与本实现里的初始化**完全一致**：`w1` std 0.01，
`w2` 用 PyTorch `Linear` 默认初始化（rank 256 ⇒ std 0.036），合起来偏置 std ≈ **0.0058**，
而块头 logits 是 O(1)。`test_steer_hook` 里需要把偏置放大 **50×** 才会改变树的接受日程。
而 lr 4e-5 的 Adam 每步位移约 lr，1000 步最多把每个权重移动 ~0.04，
即偏置 std 至多长到 ~0.012——**很可能仍然小到改变不了任何 argmax**。

这带来一个必须分开的歧义：Arm M 若为 null，可能是
(a) 机制在 VLM 块草稿 + 树验证下不迁移（结论本身有价值，正是 DSpark 自身 LLM 增益
与 MMSpec survey 多模态回归这一矛盾的答案），也可能是
(b) 偏置从未被驱动到足够大（优化尺度问题，与机制无关）。
因此 `eval_chartqa.py` 与 `measure_offset_curve.py` 都加了 `--markov-scale`，
可在**不重训**的情况下扫偏置量级：若 τ(scale) 在某处升过 5.20，则机制有价值、
缺的是幅度；若所有 scale 都不升过，则负结果稳健。
**评测口径注意**：引导的 15 次顺序小矩阵乘发生在 `timing="decode_only"` 的计时窗**内**，
所以 Arm M 的 ms/token 含真实引导开销（DSpark 报 0.2~1.3%，那是融合实现的数字），
τ 与加速比要分开看，不能让 τ 的增益掩盖加速比的损失。

### 8.5 结果（2026-09-23 完成）：机制不迁移，但原因被定位

四臂跑完。判据写在训练前：主判据 τ @b63 5.20 → ≥5.50。

| arm | 偏置条件 | lr | 步数 | τ @b63 | Δτ | 加速比 | Δ加速比 |
|---|---|---|---|---|---|---|---|
| 基线 | — | — | — | 5.199 | — | **3.36×** | — |
| Arm M | gt | 4e-5 共享 | 1000 | 5.236 | +0.037 | 3.18× | −0.18× |
| Arm S | self | 1e-3 | 1250 | 5.274 | +0.075 | 3.16× | −0.20× |
| Arm S final | self | 1e-3 | 2000 | 5.250 | +0.051 | 3.17× | −0.19× |

主判据**未达标**，否证条款**成立**（Δτ < 0.15 且逐 offset 曲线不动）。
**加速比在每个 budget 上都下降**，因为 15 次顺序引导矩阵乘发生在 `decode_only` 计时窗内。

**但失败原因被精确定位**（这是可写的部分）。三臂的两种口径对照：

| 臂 | 条件 | 训练口径 深层(8-15) | 推理口径 g | Δg (95% CI) |
|---|---|---|---|---|
| Arm F | gt, lr 1e-3 | **0.3017** (+0.191) | 0.8496 | −0.0179 [−0.0378, +0.0025] |
| Arm S | self, lr 1e-3 | 0.1106 (+0.000) | 0.8716 | +0.0040 [−0.0071, +0.0167] |
| Arm M | gt, lr 4e-5 | 0.1164 (+0.006) | 0.8653 | −0.0022 [−0.0110, +0.0067] |

条件信息**确实在表示里**（给真前缀 off8 从 0.216 升到 0.447），
但那个增益**只在真前缀被提供时存在**；推理时块头只能用自己（有错的）前缀，
于是自条件化只到持平（+0.000 / +0.004），gt 条件化则是分布外**伤害**。
Arm S 越训越差（step250→500 的 Δg = −0.0106，CI 排除 0），排除了「训练不够」。

**详细报告**：`/home/mcy/Projects/Draft-OPD/papers/RESULTS_块内衰减修复_2026-09-23.md`
**执行与判据修订记录**：同目录 `EXEC_块内衰减修复_2026-09-23.md`

**三个静默失败（都会让结论反向，已修）**：
1. 评测 `--data` 默认落到 `vl_eval`（"single word or phrase" 模板），把 τ 压到 ~2.65，
   导致首轮 τ 2.78/2.90 的假塌陷；而曲线脚本默认 `vl_eval_plain`，两者跑在不同模板上才互相矛盾。
2. `measure_offset_curve.py --compare` 的 `stats()` 闭包捕获了第一份数据，每次都比较成「+0.0000」。
3. `stepN` 检查点缺 `markov_config.json`，加载器会**静默关掉偏置**，结果看起来像基线。

---

## 9. 视频方向的接入（2026-09-25）

### 9.1 环境坑（都实测踩过）

1. **cv2 的 `CAP_PROP_FRAME_COUNT` 会多算一帧**，seek 到末位返回 False，
   报 `cv2 read failed at frame N of N`（实测 150 条里 13 条中招）。
   **修法：读帧失败就向前回退最多 25 帧**（`predecode_cache.py::normalize`）。
2. **脚本不要命名为 `/tmp/bisect.py`** —— 它会**遮蔽标准库 `bisect`**，
   导致 `random` → `transformers` 导入连锁失败，症状看起来像「环境里 huggingface_hub 坏了」。
3. **`torchcodec` 未安装**，视觉处理器退回 `torchvision` 解码并报弃用警告。
   当前可用，但 transformers 未来版本会移除该回退。暂不安装（避免引入新变量）。

### 9.2 视频通路的两条正确用法（与直觉相反）

- **处理器只认「文件路径 + fps」**。传「已解码帧列表 + `VideoMetadata`」会**静默**给出
  固定网格（8/16/32 帧都得到同一 `video_grid_thw`），帧数完全失效。
- **固定 fps 时帧数随源时长漂移**：同一个 `fps=30`，0.53 s 的片子抽 16 帧，
  8 s 的片子抽 **240 帧**（token 从 959 涨到 11551，**12 倍**）。
  所以必须先把视频**归一化成 `时长 = 帧数/fps`**（见 `predecode_cache.py`，长边 384 / 30 fps / 16 帧）。

### 9.3 判据修正

| 原判据 | 修正后 | 依据 |
|---|---|---|
| `input_ids` ∈ [900, 1000] | 时间维 = 8 且 `input_ids` ∈ [670, 1250] | 窄带是按单条视频定的；整池宽高比不同，实测 671~1247，落窄带仅 53/150 |
| `anchor 数 = max_anchors × 步数` | `anchor 数 = 解码轮数`（建池阶段） | `max_anchors` 是训练时子采样参数；池收集器每轮产一个 anchor |
| 计划里的 `max_rounds=N` | 改用 `max_new_tokens` 预算 | `glance.generate` 没有轮数上限；实测 30 条里 13 条轮数 >120（最高 174），按轮截断会腰斩 43% |

## 10. 视频阶段 2：数据造满 + 三个 bug（2026-09-25）

完整报告：`/home/mcy/Projects/Draft-OPD/papers/视频阶段2_数据量补齐与三个bug_2026-09-25.md`

### 10.1 结论

数据量确实是瓶颈。对齐步数比较，439 训练行 vs 120 训练行：

| arm | τ 中位（锚点 30） | 配对 Δ | t | 正/总 |
|---|---|---|---|---|
| zero-shot（图像侧 head） | 3.1388 | — | — | — |
| 439 行 step500~1500 | 3.45 / 3.41 / 3.43 | **+0.30 / +0.29 / +0.29** | **+5.8 / +5.1 / +5.3** | 26 / 25 / 27 of 30 |
| 120 行最好一步（step1000） | 3.1681 | +0.1326 | +0.58 | 18/30 |
| 120 行 step2500 | 2.8610 | −0.4199 | −3.15 | 1/30 |

独立留出（全新 30 行，**未参与任何选点**）：439 行 **+0.2851（t=+5.68，27/30）**，
120 行 **+0.0011（t=+0.02，13/30）**。

439 行在 **1~3 epoch** 达峰（loss ≈ 2.4~3.4），之后单调回落，6000 步（loss 0.43）已跌回 0。
**CE 目标与 held-out τ 不齐 → 必须早停 + 按留出选点。**

### 10.2 三个 bug（两个否定旧结论，一个否定旧指标）

1. **`train_head_video.py` 没有过滤 split** —— 日志 72 写着 `train rows: 150`，
   而池是 120/30，即**拿留出集训练**。留出 τ 会成为记忆程度的上界。
   已修：按 `split_of` 只留 `train` + `assert` 防回归。
2. **标签与训练/评估的 prompt 差一个词**（`background scenes` vs `overall scene`）→
   prompt_len 差 1 token → 每个 anchor 位置整体错 1 位。
   指纹：`pool_150.parquet` 记的 `input_ids_len = 959`，实际训练编码是 958。
   已修：以池的 `question` 列为唯一真源。同 head 同 30 行下，这个 bug 值 **+0.33 τ**。
3. **`glance.generate` 在视频路径上不是无损的** —— 只有 **b63k8**（标签所用配置）能逐 token
   复现标签（实测 10/10 行）；b63k4 / b63k16 / b31k8 / b31k4 / b31k16 / b15k4 / b127k8
   **全部在第 37 个 token 分叉**，此后 decode 掉进复读循环，同一行 τ 从 3.14 变 8.18。
   原因不是树走错（树与一次全新顺序 forward 在该点一致），而是
   **该处 target 的 top-2 logit 只差一个 bf16 ulp（33.50 vs 33.25）**，
   不同树形状的 attention mask/批布局把 argmax 放到平局另一边。
   → **阶段 1.2 的 τ 2.806 / EOS 中位 329 / 单条 8.9 s 全部口径不可比**（都是 b31k4）。
   评估统一改 b63k8；该口径下池统计：生成 token 中位 **276**、τ 中位 **3.136**、单条 **7.2 s**。

### 10.3 新增/改动的脚本

| 文件 | 作用 |
|---|---|
| `predecode_cache.py` | 新增 `--skip-existing`（复用已归一化文件）、`--reuse-split-from`（冻结旧划分）、`--shards/--shard`（6 路并行，解码 ~30 s/条） |
| `merge_pool_shards.py` | 合并分片池并断言：无重名/无丢行/**旧 split 一字未动** |
| `merge_greedy_labels.py` | 合并两段 rollout，断言无重复且 `max_new/topk/budget/branch_k/head` 完全一致 |
| `opd_rollout.py` | 新增 `--video-prompt`（默认仍取 `question`） |
| `eval_video_head.py` | 视频 τ 评估（跑到 EOS、`--restrict-to` 分离锚点/全新留出、不改图像侧） |
| `analyze_video_eval.py` | 逐行**配对** Δ + t；锚点与全新留出分开报 |
| `train_head_video.py` | split 过滤 + prompt 从池读 |
| `scripts/75..77_*.sh` | 120 行 arm / 439 行 arm / b63k8 评估（`BUDGET`/`BRANCH` 默认 63/8） |

### 10.4 数据上限

本机视频源上限 = **499 行**（VideoDetailCaption 已全部用掉；MLVU summary 已死；
Video-MME τ 退化；其它数据集根目录再无视频源）。
继续加数据只有两条路：**多 prompt（`(video, prompt)` 对，零下载）** 或 **下载新视频源**。
两件更该先做的事：**teacher-forced τ**（消掉平局与复读循环的干扰）、**按 epoch 定训练量 + 留出早停**。
