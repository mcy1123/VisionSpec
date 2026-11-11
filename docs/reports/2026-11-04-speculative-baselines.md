# 投机解码 baseline 在同一测试集上的复现（2026-11-04）

状态：**五条 baseline 全量完成**（PLD、经典双模型 SD、Medusa、EAGLE-3 发布头、EAGLE-3 同语料微调），
全部 200/200 逐 token 无损、准确数 162/200。EAGLE-3 的数值一致性门禁（对 SpecForge）与端到端门禁均通过。
depth-5 链只有 3 条 smoke，未跑全量。**τ 最高的 baseline 是经典双模型 SD（4.375），其次是 Medusa（3.652）；
EAGLE-3 只有 2.25。** 但 classic SD 每轮要跑 8 次 4B 前向（91.7 ms/token，比 AR 还慢），τ 高不等于快。

## 1. 为什么必须自己复现

用户的问题是"找几篇论文，配置一样就直接用它们的结果，否则复现"。核对结论是：**没有任何已发表数字能在我们的测试集上直接比对**，所以每条 baseline 都必须在本机重跑。

| 论文 / 系统 | 它报了什么 | 与我们的配置差在哪 |
|---|---|---|
| GLANCE（arXiv 2609.00355）Table 2 | Qwen3-VL-8B target，五个任务（caption/TextVQA/InfoVQA/DocVQA/ChartQA）的 τ 与加速比 | 任务不同；我们用的是 ScienceQA image test200，且 prompt 要求逐步推理后 `Answer: (X)`，不是论文里"single letter"那种没有 decode loop 的 ScienceQA 设定 |
| EAGLE-3（NeurIPS'25） | 文本 LLM 上的 acc length；官方 repo 的 HF 路径只有文本 target | 没有 VLM 通路；Qwen3-VL 的 interleaved mRoPE 会静默失效（见 `docs/reports/2026-09-28-eagle3-stage0.md`） |
| EAGLE3-VL 发布头（[AQ-MedAI](https://huggingface.co/AQ-MedAI/Qwen3-VL-8B-Instruct-eagle3)） | MMStar 上 acc length 2.493（SGLang 0.5.6） | 数据集不同；它的 "acc length" 是否含 bonus 未说明；引擎不同 |
| ViSpec（NeurIPS'25，Kang et al.） | Qwen2.5-VL-7B / LLaVA-1.5-7B 上的 τ | target 不是 Qwen3-VL-8B；官方 head 不适用于我们的 target；且它用 typical acceptance，非无损 |
| Medusa（[FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)） | 文本 LLM | 无 VLM 通路，需要自己接 hidden state 并训练 |

GLANCE 论文自己也写了这一点："We train the last two ourselves, since no published VLM drafter releases a head for this target." 对本 target 唯一"开箱即用"的已发布草稿头就是 EAGLE3-VL，我们直接用它做 zero-shot baseline，其余全部本地训练/实现。

## 2. 统一评测口径（同一把尺子）

所有 arm 共用：同一份 frozen test200 manifest、同一 prompt、图片长边 768、fp32 target、贪心、`max_new=256`、immediate EOS、每轮一次 ancestor-masked 打包验证、逐 token 与同一份 fp32 AR oracle 比对。实现见 `repro/baselines/spec_harness.py` 与 `repro/baselines/README.md`。

`τ` 在本报告统一用**有效 tokens/round**（`n_tokens / rounds`），而不是历史 `eval_sqa.py` 里被终止裁剪影响过的 `accepted+1` 字段；两者在 GLANCE 的老结果里差约 0.4。

## 3. 已实现的 baseline

| arm | 论文 | 训练 | 配置 |
|---|---|---|---|
| `ngram_pld` | Prompt Lookup Decoding | 无 | 3-gram 后缀，最多 8 个候选 |
| `eagle3_prod_tree63` | EAGLE-3 + 发布头 | 无（zero-shot） | aux 层 1/17/32，8 步 × top-8 → 63 节点树 |
| `eagle3_prod_chain5` | EAGLE-3 | 无 | 同头，depth-5 链 |
| `classic_sd_qwen3vl4b` | 经典双模型 SD | 无 | Qwen3-VL-4B 贪心 8 token 链 |
| `eagle3_ft_tree63` | EAGLE-3 | 同语料微调 2 epoch | 同上，初始化自发布头 |
| `medusa_tree63` | Medusa（Medusa-1） | 同语料训练 2 epoch | 5 个头（84M 参数，等于论文的 0.08B），tied lm_head，63 节点树 |

## 4. 关键实现坑（都会静默失败）

1. **`d2t` 存的是差值**：SGLang 的 loader 做 `hot_token_id = d2t + arange(32000)`。按"绝对映射"读会把前 119 个 draft id 全映射到 token 0，提案全是垃圾而**无损门禁照样通过**（τ≈1.0，看起来只是"草稿头很弱"）。修正前我们观察到的就是 τ=1.00 + lossless。
2. **aux 层下标**：EAGLE-3 取 `hidden_states[1]/[17]/[32]`（= SGLang 里"该层输入处的 hidden+residual"）；GLANCE 的 block head 用 `hidden_states[i+1]`。差一层不会报错，只会让接受率下降。
3. **`taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3` 不可用**：它的 `t2d` 是 `[32000]` 全 True，`d2t` 也不是差值数组，无法恢复 draft→target 词表映射。AQ-MedAI 的那份才是 `[151936]`/`[32000]` 的正确配对。
4. **EAGLE 轮需要一次额外的 target 前向**：EAGLE 的 draft 输入是 `(emb(x_t), h_t)` 同位置，h_t 只能先由 target 把 root token 前向一次得到。这一趟算进 EAGLE 的墙钟，不影响 τ。
5. **分块 ingest 的因果 mask**：`SDPA(is_causal=True)` 只在 `q_len == kv_len` 时对齐正确，提交尾段的分块 ingest 必须显式给 mask，否则 KV 长度一长就错。

## 5. 数值门禁

- `repro/baselines/eagle3_draft.py` 与 SpecForge 的 `LlamaForCausalLMEagle3.midlayer` 逐元素对比：layer 最大差 `1.5e-5 ~ 6.1e-5`，logits 最大差 `≈2e-5`（n=4/12/64）；分块增量 ingest 与整段前向一致到 `2.3e-5`。
- 端到端：每条 baseline 在每个记录上都做 committed token ids 与 AR oracle 的逐 token 比对；不一致直接判失败（退出码 5）。

## 6. 结果

全部 200 条、全部 200/200 逐 token 无损、准确数都是 162/200。

| arm | 有效 tokens/round | vs OPD（配对 95% CI） | 备注 |
|---|---:|---|---|
| OPD（我们的方法） | **5.476** | — | 已发表 |
| GLANCE（SFT step3546） | **4.960** | −0.516 [−0.609, −0.434] | 已发表 |
| **Classic SD（Qwen3-VL-4B，8 步链）** | **4.375** | −1.101 [−1.306, −0.915] | zero-shot；每轮 8 次 4B 前向，91.7 ms/token（比 AR 慢） |
| **Medusa（同语料训练）** | **3.652** | −1.824 [−1.987, −1.677] | 5 头 × top-8 → 63 节点，训练 35 分钟 + 评测 32 分钟 |
| EAGLE-3 发布头 tree63 | **2.251** | −3.225 [−3.486, −2.992] | 8 步 × top-8 → 63 节点，zero-shot，69 分钟 |
| EAGLE-3 同语料微调 tree63 | **2.288** | −3.188 [−3.448, −2.953] | 2 epoch/单步目标；仅 +0.037 |
| PLD / n-gram 3-gram | **1.254** | −4.222 [−4.481, −3.990] | 训练-free 下限，47 分钟 |
| EAGLE-3 发布头 chain5 | 未跑（smoke 3 条 1.61） | | depth-5 链，队列被主动停在双模型 SD 之后 |

![arms](</mnt/local/mcy/GLANCE/results/baselines_20261104/arms_test200.png>)

### 6.0 落盘的四条（PLD / EAGLE-3 ×2 / 双模型 SD）

- PLD：1.254 tokens/round，200/200 无损，162/200 正确，47 分钟。
- EAGLE-3 发布头（zero-shot，与 GLANCE 同为 63 节点验证预算）：**2.251** tokens/round，
  200/200 无损，162/200 正确，69 分钟。平均每轮 67.2 轮、树 64 个节点。
  这与论文 Table 2 里 EAGLE3-VL 在五个任务上的 2.13–2.92 以及 AQ 头 README 的
  MMStar acc length 2.493 同一量级，说明实现是对的；同时它只有 GLANCE（4.960）的 45%、
  OPD（5.476）的 41%。

### 6.0.3 Medusa：训练过的最强 baseline（τ 4.375 的 classic SD 除外）

Medusa-1（5 个残差矩阵，每个 4096×4096，共 84M 参数，复用 target 冻结的 `lm_head`）在**同一批
4726 条 SQA 训练行**上训练 2 epoch（35 分钟），test200 得到 **3.652** tokens/round，200/200 无损，
评测 32 分钟、53 ms/token——比 EAGLE-3 高 1.6 倍，而且**更便宜**（每轮没有自回归草稿，只有一次
root 前向 + 一次 63 节点验证）。零样本的 classic SD 虽然 τ 更高（4.375），但它每轮要 8 次 4B 前向，
91.7 ms/token 是 AR（52.4）的 1.75 倍；Medusa 是唯一"训练过、τ 又便宜"的对手。

为什么它比 EAGLE-3 强：Medusa 的每个头直接从同一个 root hidden 预测 offset 1..5，**不存在链式误差累积**；
在这个任务上答案开头（"To determine…"、"The image shows…"）虽然有熵，但几个 offset 的联合覆盖
仍然被 63 节点树兜住。EAGLE-3 的第 2 步之后要喂自己的 hidden，分布外就掉。

**两个已知偏差**（都已在 `repro/baselines/medusa_draft.py` 里写明）：
1. 我们用**贪心验证**而不是 Medusa 的 typical acceptance，所以所有 arm 都能过逐 token 无损门禁。
   这个变体只会比原版接受得**更少**，是保守方向。
2. 头是 tied-lm_head 形式（`lm_head(norm(h) + norm(h)W)`），不是每个头一个 622M 的全词表线性层；
   这样 5 个头正好是论文的 0.08B 量级。

### 6.0.1 差距是否集中在某个子集

把 test200 按生成长度三分位、按 subject 分组，EAGLE-3 的 τ 几乎是一条平线：

| 分层 | n | 平均生成 token | EAGLE-3 | GLANCE | OPD |
|---|---:|---:|---:|---:|---:|
| 短（≤~110） | 66 | 94.5 | 2.226 | 5.387 | 6.167 |
| 中 | 66 | 134.3 | 2.229 | 5.682 | 6.756 |
| 长（≥~160） | 68 | 223.1 | 2.282 | 5.040 | 5.754 |

subject 维度：social science（n=104）EAGLE-3 2.22 / GLANCE 5.65 / OPD 6.69；
natural science（n=91）2.28 / 5.12 / 5.80。**逐记录看，EAGLE-3 在 200 条里没有一条超过
GLANCE**（0/200），逐记录 τ 的相关只有 0.25。也就是说差距不是由某个子集或某几道长答案
撑出来的，而是 EAGLE-3 在这个任务上处处低一档——这与论文的机制解释一致：链式草稿在高熵、
自由文本的推理开头会漂移，而 block head 的 15 个 offset 一次给出、再用 63 节点树兜住。

### 6.0.2 同语料微调的 EAGLE-3

在它自己的 4726 条 SQA 训练行上微调 2 epoch（590 更新，lr 2e-5，单步 EAGLE-3 目标），
teacher-forced CE 从 1.74 降到 ~0.25，但 test200 的接受长度**没有变化**：
全量 200 条为 **2.288**，只比 zero-shot 的 2.251 高 0.037（配对 95% CI [+0.020, +0.055]）——统计上不跨零，但相对只有 1.6%，换个数据/seed 就未必复现，不能当成微调有效。200/200 无损。
权重确实动了（fc 相对变化 1.5%，q_proj 2.7%，lm_head 5.0%，cos≈1）。

这条结果的意思是：**EAGLE-3 在这个任务上的短板不是"没在我们的数据上训过"**。
一个解释是链式草稿的误差累积——训练是单步 teacher-forced（TTT=1），第 2 步之后 draft
喂的是自己的 hidden，分布外会掉；发布头本身是 TTT 训的，但两者在这个测试集上落在同一档，
说明瓶颈更可能在"顺序链 + 高熵推理开头"这个 regime，而不是头的拟合能力。
这个解释没有单独实验证伪/证实，已列为后续项（TTT=2、更大 lr 的对照）。

### 6.1 harness 自查（未通过，已放弃该 arm）

把 GLANCE 的 block head 接进 `spec_harness` 当 proposer，6 条记录得到的 tokens/round
是 2.48/2.64/2.81/2.70/1.79/2.38，而 `glance.generate` 同 6 条是 4.58/4.14/4.55/4.50/4.34/4.38，
输出仍然逐 token 无损。差距来自 `glance.generate` 的 head 侧 KV cache 复用
（`kv_head.crop(start)`）——proposer 接口没有暴露这块状态，我的复刻每轮都从空 cache 重新起草。
**这个 arm 只用于自查，不进 baseline 表**：它说明"同一个 head 的起草状态复刻不精确"，
不说明 verifier 有偏差（commit/walk/mask 是同一份代码，且所有 arm 的逐 token 门禁都通过）。
GLANCE/OPD 的数字因此继续用它们自己的 runner，baseline 用本 harness，两边共享同一 verify/commit 语义。

## 6.2 设置对齐与原论文数字核对

这一节回答两个问题：我们的设置和论文是不是同一个尺度；复现出来的数字是不是落在论文的合理区间。

### 6.2.1 尺度（τ 的定义与解码协议）

| 项 | 论文 | 本次 | 一致性 |
|---|---|---|---|
| τ 定义 | GLANCE eq.(1) `τ=E[a+1]`；EAGLE-3 论文 Table 1 的 τ 同义（vanilla=1） | `n_tokens/rounds`，即 committed tokens per verify round | ✓ 同一把尺子 |
| 解码 | greedy（temperature=0），capped 256 | greedy，capped 256 | ✓ |
| target | Qwen3-VL-8B-Instruct，frozen | 同（fp32） | ✓ |
| 验证 | 一轮一次 ancestor-masked 打包验证 | 同 | ✓ |
| 每轮验证预算 | GLANCE 63 节点；EAGLE3-VL 论文 Table 1 固定 32 draft tokens；Medusa 默认树约 64；Classic SD 8 步链（9 行） | GLANCE/OPD 63；EAGLE-3 63（8 步×top-8）；Medusa 63（5 头×top-8）；Classic SD 8 步链（9 行）；PLD ≤9 行 | EAGLE-3 拿到**约 2 倍**于论文的预算（63 vs 32），对 baseline 有利；Medusa/Classic SD 与论文同档 |
| 无损 | GLANCE/EAGLE-3/Classic SD ✓；Medusa/ViSpec ✗（relaxed acceptance） | 全部按**贪心无损**跑 | Medusa 偏差，见下 |

**预算怎么定的**：GLANCE/OPD 的部署预算就是 63 节点，所以训练过的两条重建头都用 63；
EAGLE-3 与 Medusa 也放到 63（EAGLE-3 论文 Table 1 只给 32 draft tokens，等于我们**多给**了约一倍预算）；
Classic SD 保持论文的 8 步链（每轮只验 9 行，比 63 节点便宜得多）。也就是说，
**没有一条 baseline 被我们压小预算**，比较方向对我们的结论是保守的。

反过来，不同预算的 τ 不能直接排名：Classic SD 用 9 行验证就能拿 4.x，而 GLANCE 要 63 行才拿 4.96。
论文的做法是 τ 之外再配一个 within-system 加速比；本 harness 的墙钟不是生产引擎（没有 CUDA graph/融合 kernel），
所以这里只把 τ 当主指标，并把 ms/token 一并列出，不把跨系统的墙钟比当结论。

### 6.2.2 与原论文数字对照

| arm | 原论文/发布方数字 | 出处 | 本次（test200） | 判断 |
|---|---|---|---|---|
| PLD | τ 1.19–2.50（Vicuna-13B 五任务，均值 1.80） | EAGLE-3 论文 Table 1 | 1.254 | 落在下沿；SQA 答案不抄 prompt，合理 |
| EAGLE3-VL | τ 2.13–2.92（同 target）；MMStar accept length 2.493 | GLANCE Table 2；AQ 头 README | 2.251 | ✓ 落在区间 |
| EAGLE-3（文本 LLM） | τ 6.17–7.54（Vicuna-13B）、6.23（LLaMA-3.1-8B） | EAGLE-3 论文 Table 1 | 2.25 | 不可直接比：那是文本 target + 大 draft 预算；VLM 同 target 的参照是 2.13–2.92 |
| Medusa | τ 1.46–1.66（同 target）；2.09–2.78（Vicuna-13B） | GLANCE Table 2；EAGLE-3 Table 1 | 3.652 | 高于区间，见 6.2.3 |
| Classic SD（4B） | τ 3.53–4.47（同 target，8 passes） | GLANCE Table 2 | **4.375** | ✓ 落在区间上沿（修复前 1.522 是 bug，见 6.2.4） |

### 6.2.3 Medusa 的自洽性校验（为什么 3.65 不是虚高）

Medusa 的树只依赖 root hidden，所以可以直接用 teacher-forced 轨迹算出**期望 τ** 并与实测对照：

- test200 前 10 条、1609 个 root 位置，逐 offset 的 top-8 覆盖率：**0.971 / 0.840 / 0.734 / 0.757 / 0.803**
- 由覆盖率推出的期望 τ = **4.168**；实测 3.652
- 差额来自 63 节点预算的截断（5 深 × top-8 的深度-2 就有 64 条路径，装不进 63 个节点），方向正确

所以 3.65 是"头的能力 + 树预算"算得出来的数，不是评测漏洞。它高于论文的 1.46–1.66，原因是任务熵不同：
论文那五行是 caption/TextVQA 这类自由文本，SQA 的答案开头高度模板化（"The image shows…"、"Let's analyze…"），
逐位置熵低——正是论文 §6 entropy law 说的那个机制（越想抄/越确定，τ 越高）。

### 6.2.4 修复记录：Classic SD 的 M-RoPE 位置 bug

第一版 Classic SD（τ 1.522）是**错的**：`ClassicSDProposer` 的增量前向没有传 `position_ids`，Qwen3-VL 的
M-RoPE 带图像 delta，漏传就按默认相位解码。证据：同一条记录 teacher-forced 下的 argmax 是 785（= target 的
真值），增量路径的 argmax 是 2168。修好后 3 条 smoke 的 τ 从 1.52 跳到 4.03，全量 **4.375**，落在论文 3.53–4.47 的区间上沿。
旧产物保留为 `*_INVALID_missing_position_ids.json`，不删。

**这个 bug 正好说明为什么必须拿论文数字反查**：修复前的 1.52 看起来"只是较弱"，无损门禁也照样通过
（输出仍然逐 token 等于 AR），只有对着 GLANCE Table 2 的 3.53–4.47 才会发现不对。

## 6.3 论文格式表：接受率 + 加速比

计时协议：同一进程、同一张卡、fp32 target，**每个 arm 与自己的 AR 基线交替配对测量**（AR 先测，arm 紧随），
预热丢弃后按记录平均；`Speedup = mean(AR ms/记录) / mean(SPEC ms/记录)`，端到端（含 prefill）。
τ 是 test200 全量；加速比在计时子集上（12 条测 10 条 / 5 条测 4 条），子集 τ 见表注。

| Method | Draft params | Train | τ ↑ | ms/token ↓ | Speedup ↑ | Lossless |
|---|---:|---|---:|---:|---:|---|
| AR（参照） | — | — | 1.00 | 102–108 | 1.00× | — |
| PLD (3-gram) | 0 | — | 1.25 | 87 | 1.14× [1.06, 1.20] | ✓ 200/200 |
| EAGLE-3 发布头（tree 63） | 0.40B | — | 2.25 | 148 | 0.66× [0.66, 0.67] | ✓ 200/200 |
| EAGLE-3 同语料微调（tree 63） | 0.40B | 2 ep | 2.29 | 141 | 0.70× [0.67, 0.72] | ✓ 200/200 |
| Medusa-1（5 头，tree 63） | 0.084B | 2 ep | 3.65 | 101 | 0.98× [0.91, 1.07] | ✓ 200/200 |
| Classic SD（Qwen3-VL-4B，8） | 4.4B | — | 4.38 | 116 | 0.85× [0.79, 0.91] | ✓ 200/200 |
| GLANCE（SFT） | 1.05B | 2 ep | 4.96 | 48 | 2.05× [1.87, 2.18] | ✓ 200/200 |
| **OPD（ours）** | 1.05B | 2 ep | **5.48** | **44** | **2.24× [2.15, 2.36]** | ✓ 200/200 |

表注 1：计时子集上的 τ 分别是 PLD 1.30、Classic SD 4.55、Medusa 3.56、EAGLE-3 2.12、EAGLE-3-FT 2.23、
GLANCE 4.35、OPD 4.57，与全量 τ 的排序一致。

表注 2：这张表的绝对加速比**受 harness 限制**（HF/PyTorch 逐轮 Python 循环、无 CUDA graph），
同 target、同任务的论文数字是 GLANCE 2.49×、EAGLE3-VL 2.06×、Medusa 1.07×、Classic SD 0.71×、n-gram 0.92×。
我们这边 GLANCE 2.05× 与之接近，EAGLE-3 只有 0.66× 是因为它每轮要 8 次顺序小前向、在本 harness 里
launch 开销吃掉收益（论文自己也说 "graph capture is what converts the saved passes into saved time"）。
**跨引擎的绝对加速比不可比，τ 才是可比的量。**

表注 3：计时期间卡上仍有别人的任务（利用率 70–100%），所以绝对值低于 2026-10-21 空卡实测的
GLANCE 2.30×、OPD 2.52×；排序与 10 月测量一致。

表注 4：本表的绝对加速比只反映这套 harness 的实现开销，**不是方法优劣**。每轮有一笔按"轮"征收的
固定开销（约 55 ms：PLD 只验证 5 个节点也要 107 ms/轮，而一次 target 前向约 52 ms），前向次数多的
方法被系统性压低。去掉该开销后的投影（保留 fp32、同树、同预算）：OPD 约 3.5–4×、GLANCE 约 3×、
Medusa 约 3.6×、EAGLE-3 约 1.5–1.8×、Classic SD 仍 <1×、PLD 约 1.25×。假设、逐项推导与
"哪些数字能进论文"的边界见 [组会材料 §3.5.1](../talks/2026-10-22-group-meeting.md)。

## 7. 下一步（如果要进论文）

1. **EAGLE-3 从零训练**：当前微调 arm 用发布头初始化，审稿人会问"是不是只吃到通用预训练"。
   SpecForge 的 offline 管线可以吃我们自己 capture 的 3 层 aux 特征（9452 行约 90 GB，
   可只取 4726 行一轮），与 GLANCE/OPD 同语料、同 target、从零训练，才是论文 Table 2 那种
   matched-training 对照。
2. **Medusa 的 typical acceptance 版本**：现在这条是贪心验证（无损）。如果要和论文的 Medusa 数字
   对得上的话需要实现 typical acceptance，会变成非无损、τ 更高。
3. **ViSpec**：官方只发布 Qwen2.5-VL-7B / LLaVA-1.5 的头，要搬到 Qwen3-VL-8B 必须重训；
   可作为"跨 target"对照，或直接在其 home target 上比 GLANCE（论文 Table 2 最后一块）。
4. **视频侧**：本 harness 与数据集无关，换成 VideoDetailCaption 的 fresh30 即可复用；
   你在 10-26 的视频 OPD 结果也需要同一把尺子上的 baseline。
5. **多 seed**：微调 arm 至少 2 seed；zero-shot arm 无 seed 概念，但 test200 的配对 CI 已经给了。

## 8. 局限

- 这是 HF/PyTorch 参考实现，不是 SGLang 生产引擎；τ 可比，墙钟加速比只能当同 harness 内的相对量。
- 4B 双模型 SD 的 draft 只做前向，不训练；修掉 M-RoPE 位置 bug 后 τ=4.375，与论文 3.53–4.47 一致。
  它的 τ 高但**墙钟更慢**（91.7 ms/token vs AR 52.4，decode 加速 0.56×），这正是论文说的
  "acceptance ≠ speed"：每轮 8 次 4B 顺序前向把省下的验证次数吃掉了。
- 微调 arm 是单步 EAGLE-3 目标（无 TTT），会低估 EAGLE-3 的天花板；这条已在报告里标注为偏差。
- Medusa 用贪心验证（无损变体），不是原论文的 typical acceptance，因此它的 τ 是**保守**的。
- 所有墙钟都是同一个 HF/PyTorch harness 内的相对量：Medusa 53 ms/token、EAGLE-3 127 ms/token、
  GLANCE 21.6 ms/token，这些数不能当生产引擎的加速比用（生产里树验证会被 CUDA graph/专用 kernel 摊薄）。
