# 2026-10-15 · 任务 A（P1）：把被拒段 KL 换成官方最新 KL 版的 k3 单样本估计量

**结论一句话**：在**只改被拒段目标函数**（全词表 reverse KL → 官方 `run_qwen_gsm8k.sh`
的 k3 单样本估计量）、其余逐项冻结的 10-update 对照里，validation（`video_fresh30`，n=29，
fp32、b63k8、逐 token SPEC==AR）mean τ 由对照 **3.7300** 变为 **3.7480**，
配对 Δ = **+0.0180**，视频级 bootstrap 95% CI **[+0.0002, +0.0381]**，t **+1.83**，
正向/负向视频 **8/4**（29 条中 17 条完全不变）。
**这是一个贴着分辨率的正信号，不是"官方 KL 修好了 OPD"**：效应量 0.5%，单 seed，
CI 下界距 0 只有 0.0002 τ；按本项目一贯判据（phaseC4 把 +0.0027/+0.0179 判为噪声内），
它**不足以**支持"reverse KL 形式是 GLANCE 负结果的主因"，也**不构成任何加速结论**。

- 运行目录：`/mnt/local/mcy/GLANCE/results/taskA_kl_official_20261015T084308/`
- 预注册：`prereg/prereg.json`（训练前冻结；含 A3.2 标度测量摘要与 lr 政策）
- 交付清单：`delivery_manifest.json`

---

## 1. 基准：官方 KL 版最新入口（按用户两轮指示定案）

| 项 | 值 | 取证 |
|---|---|---|
| 入口脚本 | `run_qwen_gsm8k.sh` | `/home/mcy/Projects/Draft-OPD`（`refs/Draft-OPD` HEAD `d6f0ec8`）|
| 最后提交 | `d1476f5`，2026-08-08 | 是「KL 版里最后提交、面向 DFlash」的入口 |
| 被拒段目标 | `DISTILLATION_LOSS_MODE=k3` + `REJECTED_DRAFT_USE_REVERSE_KL=True` | k3 单样本估计量 |
| 官方实现 | `verl/verl/trainer/ppo/core_algos.py:2183-2191`（`kl_penalty_forward`，由 `kl_penalty` 分派） | clamp `kl=±20`、`kld=±10` |
| 被拒段范围 | 整段被拒后缀（`FIRST_TOKEN_ONLY` 默认 `False`） | 与 GLANCE 现状一致 |
| 位置衰减 | 接受段无衰减；被拒段 `γ^(k-1)`（脚本取 0.9） | 与 GLANCE 机制一致（GLANCE γ=0.8 属超参差） |

**TV 路线（`run_qwen_gsm8k_forward-8b-local.sh`）按用户指示不做**，公式仅存档于
`AGENTS.md §A0.1`。

**本次只做 P1。** P2（接受段 → Bernoulli forward KL）与 P3（池化合并 + `w_j=2.0`）
**未实现、未运行**，留作后续独立单变量臂；这与 `AGENTS.md §A2` 的单变量纪律一致。

## 2. 改了什么 / 没改什么

**唯一改动**：新增 `--rejected-kl {full_reverse,k3}`（默认 `full_reverse`＝论文语义，
保持历史可复现），并实现：

* `opd_fullkl.k3_from_logprobs(student_logprob, teacher_logprob)` —— 逐字转写官方 k3；
* `opd_fullkl.k3_reverse_kl(q_logits, p_logits, sample_tokens)` —— 在**轨迹自身的草稿
  token** 上取 `log p`、`log q`（与 `opd_multimodal/opd_train_vlm._k3_reverse_kl` 交叉一致）；
* `score_microbatch(..., rejected_kl=...)` 用 `plan.prop[k-1]` 作为第 k 个 offset 的采样 token。

**未改动（两臂逐项相同）**：接受段全词表 forward KL、位置衰减（`w_k=γ^(k-1)`，仅被拒流）、
两流论文 Eq.13 等权合并、被拒段范围（整段后缀）、数据/起点/updates/lr/精度/优化器/EOS/
rollout 模式/评测口径。

## 3. CPU 逐位验证（A3.1，先于任何 GPU 训练）

`repro/test_opd_fullkl.py` 由 16 项增至 **19 项，全通过**；新增三项直接对拍官方源码：

* **t13**：用 AST 从 `refs/Draft-OPD/.../core_algos.py` **取出并 exec** `kl_penalty` 与
  `kl_penalty_forward` 两个函数（不是凭记忆重写），与 `k3_from_logprobs` 对拍：
  **18 个边界点 + 4001 点密集网格（kl ∈ [−30,30]）逐位相等**；另钉住手算锚点
  （`kl=0 → 0`；`kl=−11 → 10`；`kl=−10 → e^{−10}+9`；`kl=±20 → 10`）。
* **t14**：`rejected_kl='k3'` 的 loss 等于逐 offset k3 的加权均值；缺 `sample_tokens` 报错；
  **默认路径完全忽略 `sample_tokens`（逐位不变）**，保证历史数字不移动。
* **t15**：语义与梯度——`E_q[k3] = KL(q‖p)`（反向，非 `KL(p‖q)`）；并**量化了官方 clamp 的
  系统性低估**：clamp 触发比例仅 `1.03e-4` 时均值就被压掉 0.09（一个 `p/q=1000` 的稀有样本
  真值 ~983 被压到 10）。

`repro/test_opd_train_fullkl.py` 17 项全通过（新增 CLI 断言：默认 `full_reverse`、`k3` 可选、
非法名被 argparse 拒绝）。CPU 测试先于 GPU，且改动后默认路径逐位不变。

## 4. 上 GPU 前的标度测量（A3.2）

`repro/measure_k3_scale.py`（新脚本，复用 trainer 自己的 `score_microbatch`，一次前向同时算出
两种目标）在冻结 C0 策略上取 **6 条训练视频 / 48 anchors / 639 个被拒 offset / 81 个接受 offset**：

| 量 | 全词表 reverse（论文） | 官方 k3 | 比值 |
|---|---:|---:|---:|
| 加权 `l_rej` 均值 | **6.1946** | **4.9673** | **0.802** |
| 方差 / rms | 1.372 / 6.304 | 0.829 / 5.050 | — |
| `l_acc` 均值（31 个有接受 offset 的 anchor） | 0.4073 | 同左（不受 P1 影响） | — |
| `|log p − log q| ≥ 20` 触发比例 | — | **6.42%**（32/48 anchor） | 最大 37.29 |
| k3 `kld ≥ 10` 触发比例 | — | **24.57%**（48/48 anchor） | — |
| 全词表反向 KL `≥ 10` 的比例 | 16.21%（48/48 anchor） | — | — |
| 被拒流占合 loss 份额 | 0.9414（中位 0.9551） | 0.9329（中位 0.9389） | — |

**读法（写进预注册、未事后修改）**：
1. k3 **不是**把被拒段整体缩小一个量级——均值只降到 0.80×；但它是**有界代理**而非近似
   无偏估计：**24.6% 的被拒 offset 直接顶在官方 ±10 clamp 上**（全词表反向 KL 有 16.2% ≥10）。
2. 被拒流在两种目标下都仍占合 loss 的 ~93–94%，与 `AGENTS.md §A2` 的历史事实一致。

## 5. 冻结配置与运行

| 项 | 值 |
|---|---|
| 起点 | `checkpoints/head_video439_aligned/step1000`（C0），SHA256 `e2526a24…23d2` |
| 数据 | `data/video_greedy_499.pt` + `data/pool_499.parquet`，`split=='train'`（439 行）|
| rollout | **`online`**（2026-10-13 修复后的当前闭环入口；非 legacy-reference）|
| updates / anchors | 10 / 每 update ≤24 anchors、单视频 ≤24 |
| lr / 优化器 | 3e-5 constant；AdamW (0.9,0.95)、wd 0、clip 1.0；**fp32 CPU 主权重**（lr≤1e-4 硬前提）|
| 精度 | target fp32 + head bf16（两臂同）|
| EOS / 特征路径 | `immediate` / cached-aligned |
| 评测 | `video_fresh30`，fp32，b63k8，max_new 1024，**`--ar`**（逐 token SPEC==AR），`--warmup 1` → n=29；**固定 step10，无 checkpoint 选择** |

耗时：对照训练 15.6 min（09:01:30→09:17:06），k3 训练 20.8 min（09:17:06→09:37:56）；
对照评测 20.3 min，k3 评测 18.2 min；峰值显存 38.03 / 38.12 GiB（A6000，单卡）。

## 6. 逐 update 机制表（两臂同种子同起点）

| update | N_acc（对照/k3） | Z（对照/k3） | `l_rej` 对照 | `l_rej` k3 | `\|g\|` 对照 | `\|g\|` k3 | clip_scale 对照 | clip_scale k3 | k3 kld clamp |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33 / 33 | 91.19 / 91.19 | 6.6395 | 4.6013 | 3.9109 | 0.8739 | 0.2557 | **1.0000** | 25.5% |
| 2 | 40 / 40 | 84.53 / 84.53 | 5.5662 | 4.7522 | 2.8816 | 1.2163 | 0.3470 | 0.8222 | 22.6% |
| 3 | 29 / 29 | 91.51 / 91.51 | 6.0165 | 4.9161 | 3.3473 | 1.3757 | 0.2988 | 0.7269 | 20.0% |
| 4 | 50 / 50 | 76.80 / 76.31 | 6.5678 | 4.7127 | 4.7788 | 1.1806 | 0.2093 | 0.8470 | 25.4% |
| 5 | 38 / 38 | 86.22 / 86.22 | 6.0973 | 5.2336 | 3.5925 | 1.4855 | 0.2784 | 0.6732 | 27.3% |
| 6 | 28 / 26 | 92.01 / 92.46 | 5.9527 | 4.8503 | 3.4497 | 1.1683 | 0.2899 | 0.8559 | 20.5% |
| 7 | 41 / 43 | 82.58 / 82.34 | 5.3737 | 4.3344 | 3.3954 | 1.3237 | 0.2945 | 0.7555 | 20.3% |
| 8 | 63 / 62 | 69.42 / 68.80 | 6.5729 | 4.6443 | 4.9226 | 1.0414 | 0.2031 | 0.9603 | 21.8% |
| 9 | 66 / 66 | 66.54 / 66.67 | 6.2847 | 5.0604 | 3.5732 | 1.2068 | 0.2799 | 0.8286 | 23.2% |
| 10 | 30 / 30 | 93.60 / 93.24 | 7.0198 | 4.7705 | 3.4118 | 1.1997 | 0.2931 | 0.8335 | 21.8% |

**两个必须一起报的事实**（否则会把"换了目标"误读成"改了有效 lr"）：

1. **原始梯度范数确实塌了**：k3 的 `|g|` 只有对照的 **0.22–0.42×**（均值 1.21 vs 3.83）。
   这正是 `§A3.2` 要求先量、并在预注册里写清的标度风险。
2. **但 clip 1.0 几乎完全吸收了它**：对照 10/10 次都被裁到 applied norm = 1.0；
   k3 有 9/10 次 `|g| ≥ 1` 也被裁到 1.0，只有 update 1 的 applied norm 是 0.874。
   ⇒ **两臂"实际走的一步"几乎一样大**，因此这次比较基本就是一次干净的"只换目标函数"，
   k3 并没有因为标度小而偷到一次更小的有效步长。**这同时说明：预注册里担心的
   "有效 lr 混淆"在 clip=1.0 下基本不成立，无需再加 lr 缩放臂**（预注册已写明该臂为可选）。

## 7. 评测结果（n=29，各 30/30 逐 token 无损）

| 臂 | mean τ | median τ | mean rounds | decode ms/token | 无损门禁 |
|---|---:|---:|---:|---:|---:|
| C0（起点，不训练） | 3.7377 | 3.7778 | 72.83 | 59.98 † | 30/30 |
| **对照（论文 reverse KL）** | **3.7300** | 3.7576 | 73.00 | 37.55 † | **30/30** |
| **P1（官方 k3 单样本）** | **3.7480** | 3.7778 | 72.66 | 35.78 † | **30/30** |

配对（视频级 bootstrap 2000 draws，seed 20261015；以视频为统计单位）：

| 比较 | Δmean τ | 95% CI | t | 正向/负向/变化 |
|---|---:|---|---:|---|
| **p1_k3 − 对照** | **+0.0180** | **[+0.0002, +0.0381]** | **+1.83** | 8 / 4 / 12 |
| p1_k3 − C0 | +0.0103 | [−0.0073, +0.0291] | +1.08 | 10 / 7 / 17 |
| 对照 − C0 | −0.0076 | [−0.0291, +0.0146] | −0.67 | 5 / 11 / 16 |

† `ms/token` 在不空闲的卡上取得（GPU 7 全程有他人 6.6 GiB 进程，util 92%），
**按口径只能作参考，不作速度证据**；rounds 为同一解码协议下的计数，可直接比。
逐视频表：`eval/per_video_taskA.md`；机读：`eval/summary_taskA.json`。

## 8. 判读（对照预注册的可反驳边界）

* **预期命中**：k3 把 `l_rej` 压到 0.77–0.80×、clamp 触发率 ~20–27%、被拒流仍占 ~93%。
* **主结果**：Δτ = **+0.0180，CI 下界 +0.0002**。这是"**贴着分辨率**"的结果：
  效应量 0.5%，单 seed，17/29 条视频完全不变，且 CI 下界距 0 只有 0.0002 τ。
  按项目既有判据（phaseC4 对 +0.0027/+0.0179 判"噪声内，不值得复现"），
  **本结果不能用来宣称"官方 KL 实现带来收益"**；它同样**不能**被读成"证明 reverse KL
  是负结果主因"——恰好相反，它把该假设的空间压到 ≤ +0.04 τ。
* **可反驳边界（预注册）**：若 k3 后 τ 不改善 ⇒ "reverse KL 形式是主因"被否。
  实际是"极小幅改善"。诚实的合并表述：**换成官方 k3 目标在本预算下最多值 +0.02 τ 量级，
  远小于 OPD 与 SFT 之间约 −0.06 的差距；因此不要把"换损失形式"当作解锁 OPD 的路线**，
  这与 `AGENTS.md §A4` 的走向一致（应转向 accepted 流为空 / 数据与目标不匹配 / 树选择）。
* **不构成加速方法**：ms/token 差异在受污染卡上测得，且 τ 增益本身的量级也不足以谈速度。

## 9. 与历史结果的关系（不作跨协议配对）

phaseC4 的 C2（论文 KL，`legacy-reference`、10 updates）为 3.7403。本轮对照是同一目标、
但走**当前 online 闭环**：3.7300（与 C0 差 −0.0076，CI 跨零）。两者**是不同协议**
（rollout 模式不同），只作背景，不作配对。**本轮唯一的配对比较是 control vs p1_k3**，
两臂同批、同起点、同数据、同预算、同精度、同评测。

## 10. 后续单变量臂（未运行，明确登记）

* **P2**：接受段全词表 forward KL → Bernoulli forward KL（`losses.py:432-453`）。
* **P3**：两流合并 → 池化加权均值，且 `w_j` 1.0 → **2.0**（`losses.py:854-888`）。
两者各为独立臂，**不得与 P1 同一次改动一起下结论**；本次未实现、未跑。

## 11. 交付物

```
taskA_kl_official_20261015T084308/
├── prereg/prereg.json              # 训练前冻结（含标度测量摘要、lr 政策、可反驳边界）
├── measure/k3_scale_c0.json        # A3.2 标度测量（48 anchors）
├── training/{control,p1_k3}/       # step5/step10 + jsonl + 日志
├── eval/{control,p1_k3}_u10_fresh30.json
├── eval/summary_taskA.json         # 配对 bootstrap + 逐视频
├── eval/per_video_taskA.md
├── commands.txt                    # 逐条命令
└── delivery_manifest.json          # 全文件 sha256
```

代码改动（三处，均向后兼容、默认行为逐位不变）：
`repro/opd_fullkl.py`、`repro/opd_train_video_fullkl.py`、`repro/test_opd_fullkl.py`；
新增 `repro/measure_k3_scale.py`、`repro/summarize_taskA.py`、
`repro/run_taskA_kl_official.sh`、`repro/pick_gpu.py`、`repro/make_run_manifest.py`。
