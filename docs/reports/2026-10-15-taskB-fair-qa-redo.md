# 2026-10-15 · 任务 B：公平重做视频 + 图像 QA 两组对照

**结论**
1. **视频 QA（MSVD-QA / B，20 视频 40 QA，fp32 + real 时间轴 + 两臂各 120/120 updates，同一冻结 480 槽日程）**：
   warm start **3.4700** → 继续 SFT **3.6395**（+0.1695，CI [+0.0901,+0.2472]）→ OPD **1.2060**
   （−2.2640，CI [−2.4429,−2.0915]，20/20 视频变差、轮数 10→29、decode ms/token ×3）。
   **在单协议、单精度、等预算的干净对照下复现了 2026-10-14 的结论**：领域内 QA 适配收益属于 SFT，
   当前 full-KL OPD 配方不仅无额外收益，还会破坏草头。逐 token SPEC==AR：三臂各 40/40、共 **120/120 无损**。
2. **图像 QA（ChartQA，n=100，同一脚本/数据/参数，三臂同起点 `checkpoints/head`，同为 2000 行 / 1000 步 / lr 1e-4）**：
   **warm start 5.1990** → 继续 SFT **5.3676** → OPD-CE **5.3225**。
   SFT−warm **+0.1463** [+0.0713,+0.2232]，OPD-CE−warm **+0.1218** [+0.0570,+0.1830]，
   两者都显著优于不训练；**SFT−OPD-CE = +0.0245，CI [−0.0323,+0.0798] 跨零**（单 seed 不可区分）。
   **§B 的数据量混同被消除**：`head` 不是"另一个 16K 基线"，而是**三臂共同的起点**，
   本轮的 5.1990 就是它在同预算下由同一评测脚本重新测出来的数，报告里没有任何跨预算数字。

- 运行目录：`/mnt/local3/mcy/GLANCE/results/taskB_fair_qa_20261015T085742/`
- 预注册：`prereg/prereg.json`（训练前冻结，含一次明确记录的 amend）＋ `delivery_manifest.json`

---

## 1. 为什么要重做（逐条取证）

### 视频侧（2026-10-14 批次）

| 问题 | 证据 |
|---|---|
| seed20261013 不是配对复制：SFT 120 updates，OPD 只有 60 | `logs/seed1.log:203-207` 在 **step-60** 保存时磁盘写满崩溃（`file write failed`），随后从完整 step30 续训到 step60；报告散文写作 "step90" 属笔误 |
| 训练与评测时间轴不一致 | seed20261014 **训练**用 `--time-mode real`（`logs/seed2_bf16_chain.sh`），而 dev/主矩阵/confirm **评测**全部用 `--time-mode legacy` |
| seed20261014 两臂均为 bf16 | 抢卡回退；评测显示 bf16 下 greedy AR 本身不可复现，故当时**无法给出无损声明** |
| `training_budget_comparison.json` 少算 | 只解析 `opd/log.jsonl`，漏了 `opd_resumed/log.jsonl` |

### 图像侧（2026-09-23/24 批次）

| 问题 | 证据 |
|---|---|
| 基线步数被误记为 1293 | `logs/11_train.log` 结尾是 `done at step 1000`，日志正好打印 1000 行 step；1293 是 append-only `train_head.jsonl` 的**行数**（含一次 293 步的残跑），该字符串被 `13_eval.log:2` 复制传播 |
| 2K 臂 vs 16K 数混同 | `head` 训 16000 行 / lr 4e-5 / accum 8；对照臂 2000 行 / lr 1e-4 / accum 4。两者不可直接相减 |
| 产物全丢 | 除 `chartqa_head_b896.json`(5.199008) 与 `chartqa_plain_control.json`(5.364908) 外，**所有 `chartqa_plain_*.json` 与全部图像 arm checkpoint 均已删除**；`opd_pool_textvqa_2000.pt` 与 plateau 池也已删除 |

## 2. 公平性预注册（第 0 步，先写后跑）

`prereg/prereg.json` 逐项冻结了 AGENTS 要求的八项：训练行数/QA 槽位、optimizer updates、
lr 与 schedule、起点 checkpoint（含 SHA256）、精度（target/teacher/head）、EOS 行为与时间轴模式、
评测集/口径/聚合方式。三臂下限（warm / 继续 SFT / OPD）都已在**本轮重新跑出**，没有引用旧产物。
另有一条**明文 amend**：图像三臂由四臂减为 `{warm_start, sft_control, opd_ce}`，
原因是多租户抢卡下每个 1000 步臂要独占 ≈1h 的 A6000；`opd_ce` 是当年发表比较所用的最好 OPD 变体，
bernoulli 变体当年更差，若补跑将作为**独立标注**的臂、不并入同一张表。

## 3. 视频 QA 重做

### 3.1 冻结配置

| 项 | 值 |
|---|---|
| 起点 | `checkpoints/head_video439_aligned/step1000`，SHA256 `e2526a24…23d2` |
| 日程 | 2026-10-14 冻结的 `train_schedule_seed_20261014.jsonl`（480 槽），**运行前校验 SHA256 `eeafe78e…` 并拒绝不一致** |
| 数据 | `qa_pool_df20261014_real.parquet`（480）+ `target_ar_df20261014_real.pt`（480） |
| 预算 | 120 updates（**两臂相同**）、4 QA/update、6 anchors/QA、lr 3e-5 constant、accum 4 |
| 精度 | target fp32 + head bf16 + **fp32 CPU 主权重**（两臂同） |
| 时间轴 / EOS | **real（训练与评测同一模式）** / `eos_stop="immediate"`（两个 trainer 内置） |
| 评测 | dev 20 视频 40 QA（B），fp32，b63k8，逐 token SPEC==AR；固定 step120，无选择 |
| 本轮未做 | 把任务 A 的 k3 目标搬进 QA-OPD（那会同时改两个变量；任务 B 只回答"公平重做"） |

### 3.2 两臂真实消耗（trainer 自己打印，脚本抽取）

| 臂 | updates | anchors | 监督位置 | 接受 offset | Z（被拒权重和） | teacher 前向 | 峰值 | s/update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QA-SFT | 120/120 | 2338 | **20023** | — | — | — | 39.3 GiB | 21.2（13.0–37.8） |
| QA-OPD | 120/120 | 2880 | — | **879** | 2768.04 | 2880 | 39.2 GiB | 28.8（20.3–42.8） |

机读：`video/training_budget_comparison.json`。**OPD 有 18/120 个 update 的接受 offset 为 0**，
每 update 接受 offset 均值 7.3（0–33）/ 24 anchors ⇒ loss 仍由被拒流主导，
但 fp32+real 下接受流**并非全空**（bf16 的 2026-10-14 seed2 在同 2880 anchors 上只有 481）。

### 3.3 结果（`video/eval/summary_taskB_video.json`、`per_video_taskB_video.md`）

| 臂 | mean per-video τ | median | mean rounds | decode ms/token † | 无损 |
|---|---:|---:|---:|---:|---:|
| C0（warm start） | 3.4700 | 3.4480 | 10.15 | 36.48 | 40/40 |
| **QA-SFT** | **3.6395** | 3.5519 | 9.72 | 37.93 | 40/40 |
| **QA-OPD** | **1.2060** | 1.1893 | 29.45 | 112.80 | 40/40 |

配对（视频级 bootstrap 2000，seed 20261013）：

| 比较 | Δmean τ | 95% CI | + / − / = |
|---|---:|---|---|
| **QA-SFT − C0** | **+0.1695** | **[+0.0901, +0.2472]** | 13 / 3 / 4 |
| **QA-OPD − C0** | **−2.2640** | **[−2.4429, −2.0915]** | 0 / 20 / 0 |
| QA-OPD − QA-SFT | −2.4335 | [−2.2727, −2.6062] | 0 / 20 / 0 |

† 120/120 行都带 `timing_contaminated=True`（评测在 `--allow-foreign` 下跑在共享卡上），
**不作速度声明**。

门禁 `video/eval/dev_real_fp32_b/gates.json`：`pass: true`、`n_ar_unstable: 0`、
`all_lossless: true`、`all_clean_repeats_lossless: true`、3 个 head 的 SHA256 全部记录。
独立复核 `per_question.csv`：120 行、0 行非无损，其中 **95 行以 bonus EOS 停止**
（正是 2026-10-14 EOS 修复要覆盖的路径）。

### 3.4 与 2026-10-14 的关系

2026-10-14 seed2（bf16）为 SFT 3.6343 / OPD-s2-120 1.2215；本轮 fp32/real/120-vs-120 为
3.6395 / 1.2060。**符号、量级、机制都没变**——说明 bf16 回退与 60-vs-120 的不配对
不是该结论的来源，旧结论在更严协议下**被复现而非被推翻**。

**附带发现（值得单独记录）**：warm start 本身随评测时间轴模式移动——C0 在历史 `legacy`
评测里是 3.5417，本轮 `real` 是 3.4700。任何混用两种模式的比较，在训练差异之前就已带
≈0.07 τ 的偏移；这正是要求训练与评测同模式的原因。

## 4. 图像 QA 重做

### 4.1 被删产物的重建

原始 `data/opd_pool_textvqa_2000.pt` 已不存在，本轮按下式重建（它是
(data, head, budget, topk) 的纯函数）：

```
repro/opd_rollout.py --data data/target_greedy_textvqa2000.jsonl \
  --head checkpoints/head --tasks textvqa --limit 2000 --topk 64 \
  --budget 63 --branch-k 8 --max-new 256 --max-pixels 768
```

| 统计量 | 2026-09-23（已删） | 2026-10-15（重建） |
|---|---|---|
| rows / anchors | 2000 / 72418（36.2/行） | 2000 / 72418（36.2/行） |
| accept/round / tau | 2.438 / 3.438 | 2.438 / 3.438 |
| mean decoder tau | 3.567 | 3.567 |
| anchors with a rejection | 72398/72418 (100.0%) | 72398/72418 (100.0%) |
| 文件大小 | 443.1 MB | 443.1 MB（443,090,523 B） |

**每一项日志统计量都复现**；被删文件没有历史 sha256，故匹配的是日志数字与字节大小，
不是哈希（重建文件 sha256 `a9669b64…c6a7a`，见 `image/audit_pool_rebuild.md`）。
`100.0% anchors with a rejection` 也是图像侧的机制约束：被拒流几乎独占 loss。

### 4.2 三臂的公平性

| 项 | 三臂取值 |
|---|---|
| 起点 | `checkpoints/head`（SHA256 `fc0ad5a6…`），**同一目录、同一 SHA** |
| 训练行数 | 2000（`target_greedy_textvqa2000.jsonl`，SHA256 `680b94be…`） |
| updates / accum / mbs | 1000 / 4 / 1（两训练臂相同） |
| lr / schedule | 1e-4 constant（两训练臂相同） |
| 优化器 | AdamW (0.9,0.95) wd 0 clip 1.0；**无 fp32 主权重**（图像侧历史设定，两臂共同承担该缺陷） |
| 精度 | bf16 head + bf16 target（图像侧从来没有 fp32 变体） |
| 评测 | `eval_chartqa.py`、`data/vl_eval_plain`（chartqa.jsonl SHA256 `304da7c0…`）、n=100（首条 warm-up）、budgets 15/31/47/63、branch-k 8、max-new 256、max-pixels 896、bf16、pooled per-round τ |

warm 臂是**本轮重新测的**（14:02 完成），不是引用 09-23 的文件；三个 τ 全部来自同一次评测脚本与参数。
本轮实测 5.1990 与历史 5.199008 一致，属交叉验证而非同一数字的搬运。

### 4.3 结果（`image/eval/summary_taskB_image.json`）

| 臂 | pooled τ @b63 | per-prompt 均值 | median | ms/token † | speedup † |
|---|---:|---:|---:|---:|---:|
| **warm start（不训练）** | **5.1990** | 5.2362 | 5.1374 | 25.13 | 2.82 |
| **继续 SFT（+1000 步）** | **5.3676** | 5.3825 | 5.2715 | 13.69 | 3.45 |
| **OPD-CE（+1000 步）** | **5.3225** | 5.3580 | 5.2313 | 16.65 | 3.39 |

配对（prompt 级 bootstrap 2000，seed 20261013）：

| 比较 | Δmean τ | 95% CI | t | + / − |
|---|---:|---|---:|---|
| **SFT − warm** | **+0.1463** | **[+0.0713, +0.2232]** | +3.61 | 62 / 20 |
| **OPD-CE − warm** | **+0.1218** | **[+0.0570, +0.1830]** | +3.56 | 57 / 26 |
| **SFT − OPD-CE** | **+0.0245** | **[−0.0323, +0.0798]** | +0.84 | 46 / 38 |

† 三臂的 ms/token 在不同抢卡负载下测得（同一脚本、不同时刻），**只作参考，不作速度证据**。

### 4.4 §B 的两处修正落地

* **"拿 2K 的臂去比 16K 的数"不再存在**：`checkpoints/head` 是**三臂共同的起点**，
  四个训练 arm 全部是它的**续训**。AGENTS 要求的"把基线拉到同预算重训一次"在这里的
  正确操作化就是"**在同一次运行里把不训练的 warm start 测出来**"——本轮 5.1990 即为此数，
  报告里没有任何 16K 训练量下的数字参与比较。
* **"1293 步"更正为 1000 步**：`logs/11_train.log` 自证 `done at step 1000`；
  baseline 与对照的真实差异是**行数 8×、lr 4×、任务混合**，不是步数 1.3×。

### 4.5 图像结论的适用范围

* 单 seed、单臂一次训练；`SFT − OPD-CE` 的 +0.0245 **跨零**，因此**不能**声称"OPD 劣于 SFT"，
  只能说**未观察到 OPD 的额外收益**。要区分 09-24 报告的 −0.061±0.014 需要多 seed，本轮未做。
* 评测是 bf16（5.199 基线唯一的协议），bf16 下 greedy AR 不可复现 ⇒ **ChartQA 数字不是无损声明**。
* 两个训练臂都带历史 bf16-AdamW 更新精度缺陷（无 fp32 主权重），是**缺陷对缺陷**的公平比较。

## 5. 与 2026-10-14 / 2026-09-24 结论的关系（旧结论失效范围）

| 旧结论 | 本轮状态 |
|---|---|
| "QA-SFT 的领域内适配收益跨域成立" | **复现并加强**：单协议 fp32/real/120-vs-120 下 MSVD dev +0.1695（旧 +0.2094 seed2 / +0.1620 seed1） |
| "当前 full-KL 在线 QA-OPD 配方无额外收益，长训破坏草头" | **复现**：−2.2640 τ、0/20 视频改善、轮数 ×3 |
| "seed1 两臂 120 vs 60" | 该不配对在本轮**不再存在**（两臂各 120）；旧 seed1 数字仍只作历史 |
| 2026-09-24 "没有任何 OPD 变体超过继续 SFT，3-seed Δ −0.061±0.014" | 本轮单 seed **+0.0245 跨零**，**不推翻也不确认** 3-seed 结论；旧产物（checkpoint/JSON）已删除，无法直接重算 |

## 6. 限制与诚实边界

1. **抢卡**：所有计时带 `timing_contaminated`（视频 120/120 行；图像三臂不同负载），不作速度证据。
2. **单 seed**：视频只有 seed20261014 的干净复制；图像每臂一次训练。
3. **图像池重建**：匹配的是日志统计量与文件大小，无历史哈希可比。
4. **图像臂集缩减**：bernoulli 变体未跑（见 §2 amend）。
5. **confirm 集未重开**：2026-10-14 的 confirm 产物已被删除、且已用于旧结论，
   本轮**不把它当作 untouched test**。
6. **时间轴精度**：`real` 模式在这些素材上是 PTS 派生标记，`real_time_exact` 实测
   `(False, 0.0334s)`，跨视频最大 ~67 ms patch 级误差（与 2026-10-14 timestamp gate 同区间），
   见 `video/audit/real_mode_smoke_20261015.md`。训练与评测同模式，故内部一致；绝对时序精度受此限制。

## 7. 交付物

```
taskB_fair_qa_20261015T085742/
├── prereg/prereg.json                     # 训练前冻结 + 明文 amend
├── commands.txt                           # 逐条命令
├── video/
│   ├── dataset/train_schedule_seed_20261014.jsonl   # 冻结日程副本（SHA 校验通过）
│   ├── training/{sft,opd}/step120/                  # 3 臂中的 2 个训练臂
│   ├── training_budget_comparison.json              # 两臂真实消耗
│   ├── eval/dev_real_fp32_b/{per_question.csv,gates.json,ar.jsonl,inputs.jsonl,...}
│   ├── eval/summary_taskB_video.json + per_video_taskB_video.md
│   ├── audit/real_mode_smoke_20261015.md
│   └── void_runs/                          # 失败留痕（不改写、不并入）
├── image/
│   ├── data/opd_pool_textvqa_2000.pt       # 重建池（sha256 记录）
│   ├── audit_pool_rebuild.md               # 与已删池的逐项对拍
│   ├── training/{sft_control,opd_ce}/step1000/
│   ├── training_budget_comparison.json
│   ├── eval/chartqa_{warm_start,sft_control,opd_ce}.json + summary_taskB_image.json
│   └── void_runs/                          # 失败留痕
└── delivery_manifest.json                  # 全文件 sha256
```

新增脚本：`repro/run_taskB_video_qa.sh`、`repro/run_taskB_video_eval.sh`、
`repro/run_taskB_image_qa.sh`、`repro/summarize_taskB_video.py`、
`repro/summarize_image_qa.py`、`repro/pick_gpu.py`、`repro/make_run_manifest.py`。
