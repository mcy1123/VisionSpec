# 并行任务 B：GLANCE 视频时序条件响应与候选覆盖诊断（2026-10-12）

**结论等级：`未观察到稳定差异`（并在同一批数据上量化了"全时序反转"与"共享文本 prefix"之间的可行性冲突）。**
本方向不训练、不改 head/verifier、不改生产解码、不打开冻结测试集；未修改任何共享文件。
产物根：`/mnt/local/mcy/GLANCE/results/parallel_B_temporal/20261012T233231/`，交付索引 `run_manifest.json`。

---

## 0. 一句话结论

在**唯一可用作因果比较**的 9 个"共享 prefix 在 O/T/K 下都成立"的比较单元上，冻结 target 确实对
16 帧全时序反转有明显响应（候选集内概率质量移动 TV=0.162，重编码噪声底仅 0.042），而 C0 draft 的
相对响应**在概率口径下与"保持时序的镜像对照"不可区分**（`R_T−R_K = −0.089`，视频级 bootstrap
95% CI `[−0.289, +0.093]`，留一视频后可翻号）。因此**没有证据支持"draft 对时序条件变化的响应系统性弱于 target"**
这一假设，也没有观察到 C6 的 `not_in_topk` 缺口在共享 prefix 上可用的位置被复现。
本报告同时给出：为什么只有 9/86 个候选位置能进入因果比较（**prefix 与时序干预的语义冲突**），
以及在这 9 个位置之外的 20 个"prefix 已被 T 否证"的次要单元上的结果。

---

## 1. 审计：C6 事实、`not_in_topk` 定义与正确 AR 来源

全部结论来自本地文件，不采信任务提示中的路径简写。

| 项 | 核实结果 |
|---|---|
| C6 正式产物 | `results/phaseC6/current_ar_closed_loop_20261012T221104_gpu_scores/`，`manifest.json` SHA256 `573f8397…`，`trace_current_ar.jsonl` `0ffabf03…`，`token_metrics_current_ar.jsonl` `e802cbc5…`（全部见 `stageA/audit_c6.json`） |
| C6 冻结样本 | 6 条 validation 视频、各 8 个等距轮次 = 48 轮，其中 44 轮有有效 draft offset |
| `not_in_topk` | 不是网络错误率，而是 C6 `diagnostic_acl_metrics.local_oracles` 对**首个阻断事件**的分类：`raw = #(正确 token 依次落在 draft top-8)`；`raw_candidate_missing`（报告称"下一正确 token 不在 top-8"，24 次）与 `tree_selection_loss`（"正确 token 在 top-8 但树未包含"，18 次） |
| 正确 AR 来源 | `reference_mode = B_current_incremental_AR`：C6 用独立增量贪心 AR 复算当前 fp32 部署轨迹，并先与已存部署 AR 逐 token 对齐；**不是**旧 continuation，也不是 teacher-forced surrogate |
| C0 身份 | `checkpoints/head_video439_aligned/step1000`，`model.safetensors` SHA256 `e2526a24…`；本次运行前后哈希一致 |
| C5 B-reference 与 C6 | 两者样本不同（前 8 轮 vs 完整轮次等距抽样）；本方向只复用 **C6**，未与 C5 拼池 |
| target | 冻结 `Qwen3-VL-8B-Instruct`，fp32，SDPA，`use_cache=True` 的解码缓存路径 |

**视频数据事实（决定干预能否成立的关键）**：6 条视频的 `cache_path` 是 `predecode_cache.normalize`
用 `np.linspace(0, total-1, 16)` 从**原始长视频**抽 16 帧后重编码的 0.53 s / 30 fps 封装。
原始视频长度 18.0 s–224.6 s（539–6028 帧）。因此：

- 缓存文件的 0.53 s 封装时间**不是**原视频时间轴；16 帧是原视频时间轴上的**单调抽样**，帧间隔 1.17–15.0 s。
- 本方向的干预因此定义在**这 16 帧的观测序列**上，并保存每帧的原始帧号/时间
  （`frames/frame_manifest.json`），不声称复现原视频的连续时间轴。
- 原始 mp4 与逐帧联系表均可复核（`frames/sheet_*.png`）。

**人工/视觉可核验性（模型无关判断）**：6 条视频都能看出事件先后——`v_-D1gdv_gQyw`（说话→搬木→生火→火旺→庆祝）、
`v_0DFz3sgfda0`（空厨房→备菜→端盘）、`v_1kwIYXB4b8s`（进店→理发中→成品）、`v_0x6ZV9Y8g_c`（涂抹→擦掉）、
`v_0X0u_3xUBPE`（杨式太极连续架势）、`v_1KONAdZGdWU`（昏暗室内聚会，事件最弱）。全部通过抽帧联系表核对。

---

## 2. 预注册与冻结设计（在任何前向之前写盘）

`prereg.json`（含 `prereg_v1_superseded.json`）。核心：

### 2.1 比较单元（"共享 prefix"）

每个单元 =（视频, 共享文本 prefix）。主分析位置固定为该 prefix 之后**第一个由 draft 预测的 token**
（root 属于 prefix；`opd_fullkl.draft_block_logits` 的第 0 行），即部署中一个真实轮次的 offset 1。

候选池（只用 C6 既有 O 条件记录与 prefix 文本，不看任何 O/T/K 前向结果）：

1. C6 的 44 个有效 anchor（offset 1）；
2. 同一 6 条视频 round-0 block 的 offset 2–8（prefix = question + C0 贪心前缀）。

过滤与排序规则：prefix 无时序措辞（"family N"）且 C6 原条件下该位置 target entropy ≥ 0.30 nats；
按 entropy 降序、（来源, 索引）升序；每视频每族最多 4 个。
prefix 含时序措辞者归入 family C，同样测量但**不作为因果证据**。

### 2.2 三条件

| 条件 | 构造 |
|---|---|
| **O** | 原始缓存 clip，未改动（与 C6 读取的是同一文件） |
| **T** | 16 帧**精确倒序**，同一 cv2 `mp4v` writer、同一 384 长边几何、同一 16 帧/30 fps |
| **K** | 同一 16 帧、**同一时序顺序**、水平镜像（保持事件先后关系的对照扰动） |
| **O_rec** | 同一 16 帧、同一顺序、同一 writer 重写一遍（**仅门禁**，用于量化重编码/管线噪声底） |

三个写出的文件都经重新解码逐帧验证（`conditions/conditions.json`）：帧序为预期置换、逐帧平均绝对像素差 ≤ 3.2/255。

### 2.3 响应指标

- 候选集：O/T/K 三条件 target top-8 的并集（k=8 固定），再删掉"在 O/T/K 下概率都 < 0.005"的 token（概率下限）。
- target/draft 各自在 O→X 的 log-prob 变化向量 → `L1_target_X`、`L1_draft_X`、`R_X = L1_draft/L1_target`
  （`L1_target_X < 0.05` 记 NA）、`cos_X`。
- 次要口径（全部同时报告）：(a) 未加概率下限的原始并集 log-prob 量；(b) 概率空间的 L1（= 2×TV）与余弦。
- 聚合单位是**视频**：单元先按视频等权平均，再做 n=6 视频级 bootstrap（seed 20261012，2000 次）。
- 负对照：O→O_rec 的同一指标即管线噪声底；任何单元噪声底不低于其 O→T target L1 者被标注。

### 2.4 指标修订（v1 → v2，必须披露）

v1 把主候选集定义为**未过滤**的 top-8 并集。2 条视频的门禁 pilot（`pilot_v1_metric/`，保留、
**不作为证据**）显示：log-prob L1 被 `log p ≈ −25` 的尾部 token 主导，其量级淹没真实质量移动，
以致 O→O_rec 的"噪声"与 O→T 同量级。v2 因此加入 0.005 概率下限作为**主口径**，并把 v1 口径完整保留为
次要报告。修订发生在 2/6 视频门禁之后、**任何 6 视频结果与任何 O/T/K 比较被读取之前**；
干预、单元、阈值、聚合、停止规则均未改动；`prereg_v1_superseded.json` 原样保留。

### 2.5 冻结后修订记录（`prereg_amendments.json`）

冻结后只发生一处**纯元数据**修正：`audit_select.py` 把"最后一个视频"的 `prompt_ids` 写进了
`stageA/prefixes.json` 的每个条目。修复后重跑 stage A，并核对 `selection.json` 的单元内容
**逐位相同**（仅 `created` 时间戳不同）。`run_forward.py` 从不读取 `prefixes.json`，它按视频读取
C6 的 `input_<video>.json` 并重新调用真实 processor，因此**模型输入、logit 与任何指标都不受影响**。
`main/analysis/prefix_provenance.csv` 已按修复后的 prefix 重新生成。协议本身（单元、条件、指标、
阈值、聚合、停止规则）**未改动**。

---

## 3. 数据可行性：全时序反转与共享 prefix 的语义冲突（本轮最重要的机制性发现）

"共享 prefix 在 O/T/K 语义都成立"与"全时序反转"存在**结构性冲突**：caption 前缀一旦断言时间关系，
反转就使其为假。实测覆盖率：

| 候选来源 | 候选数 | 有效 offset1 | prefix 时序中立 | 中立且 target H≥0.3 |
|---|---:|---:|---:|---:|
| C6 anchor 44 个 | 44 | 44 | 13 | **7** |
| round-0 offset 2–8 | 42 | 42 | 12 | **8** |
| 合计 | 86 | 86 | 25 | **15** |

- 6 条视频里只有 `v_0x6ZV9Y8g_c` 的 caption 是非时序叙述式（"In a cozy, warmly lit indoor setting…"），
  其余 5 条的 caption 均以 `The video begins/opens with …` 开头，**任何非空前缀都带时序声称**，
  因此这 5 条视频的 C6 anchor 中"中立且可用"的位置数为 **0**。
- 只有 9 个位置最终入选主分析（≤4/视频上限）：5 条视频各 1 个（prefix = `"The video"`，round-0 offset 2），
  外加 `v_0x6ZV9Y8g_c` 的 4 个 C6 anchor（含 prefix = `"In"` 的 round-0 anchor）。
- **9/9 主单元在 C6 原条件下正确 token 都在 draft top-8 内**（`top8_offset1_C6 = True`），
  即本方向的因果子集**没有**落在 `not_in_topk` 缺口上。C6 那 24 次 `raw_candidate_missing` 的位置
  恰好都落在"prefix 已被反转否证"的区域（family C）。这与任务要求的"优先选择真实 `not_in_topk` 样本"
  存在不可同时满足的张力，本报告按"不凑数"原则如实记录覆盖率，不做替代挑选。

---

## 4. 门禁结果（全部通过）

| 门禁 | 内容 | 结果 |
|---|---|---|
| **G1** | 对原始 cache 重跑真实 processor，复现 C6 保存的输入 | **6/6** `input_ids` 与 `pixel_values_videos` SHA256 逐位一致 |
| **G2** | 每个单元的 O 条件 target argmax 复现 C6；C6-anchor 单元的 draft offset-1 top-8 复现 C6 trace | **29/29 通过**；target entropy 最大绝对差 **2.01e-5** |
| **G2b** | 在 C6 round-0 block（fused=prompt、noise block root=a₁）复现 **15 个 offset** 的 draft top-8 | **6/6 视频 15/15 offset 全同**（head 调用路径的独立实现门禁） |
| **G3** | 同一视频 O/O_rec/T/K 的 `input_ids` 必须相同（只允许像素变） | **全部一致** |
| **G4** | 写出的 T/K 文件重新解码后帧序 = 预期置换 | 6/6 通过（`v_1kwIYXB4b8s` 首末帧均为黑帧，argmin 歧义，逐帧容差门禁仍通过） |
| **G5** | O→O_rec 管线噪声底 | 已报告（概率 TV：T 0.162 vs noise 0.042），无一单元噪声底高于其 O→T 响应 |
| pilot | 前 2 条视频（`v_-D1gdv_gQyw`, `v_1kwIYXB4b8s`）先过门禁再扩展 | 通过后才跑余下 4 条 |

> G2b 存在的原因：`round0_offset` 单元的 anchor 位于 `prompt_len + j − 1`，是 C6 未采样过的轮次边界；
> 那里 head 的 fused 上下文含 a₁…a_j，而 C6 round-0 block 只有 a₁ 在 noise block 内，两者**故意不同**，
> 因此这些单元的 draft 行不可与 C6 offset j 直接比较。它们的 **target** 侧位置与 C6 相同（已逐单元门禁），
> **draft** 侧由 G2b + 结构一致性覆盖。

---

## 5. 主结果：family N（9 单元 / 6 视频，因果口径）

| 指标（视频级等权，n=6） | 点估计 | bootstrap 95% CI |
|---|---:|---|
| `R_T`（log-prob，主口径） | 1.273 | [0.812, 1.830] |
| `R_K`（log-prob） | 2.492 | [1.250, 4.564] |
| **`R_T − R_K`（log-prob）** | **−1.219** | **[−2.824, −0.320]** |
| `R_T`（概率口径 TV） | 1.105 | [0.519, 2.051] |
| `R_K`（概率口径 TV） | 0.731 | [0.500, 0.890] |
| **`R_T − R_K`（概率口径）** | **−0.089** | **[−0.289, +0.093]** |
| `cos_T`（draft vs target 方向） | 0.592 | [0.374, 0.750] |
| `cos_K` | 0.256 | [−0.022, 0.514] |
| target 响应 TV（O→T） | 0.162 | [0.120, 0.210] |
| draft 响应 TV（O→T） | 0.138 | [0.091, 0.208] |
| target 响应 TV（O→K） | 0.087 | [0.046, 0.147] |
| draft 响应 TV（O→K） | 0.095 | [0.044, 0.166] |
| **管线噪声底 target TV（O→O_rec）** | **0.042** | [0.024, 0.064] |
| 目标 greedy 变化比例（O→T / O→K） | 2/9 / 2/9 | — |
| draft top-8 覆盖变化（O→T / O→K） | **0/9 / 0/9** | — |

**逐单元（family N，概率 TV）**

| 单元 | prefix 尾部 | H_t(O) | TV tgt T | TV drf T | R_T(log) | pR_T | cos_T | TV tgt K | TV drf K | pR_K | cos_K | noise T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `-D1gdv_gQyw::round0_offset::0::1` | `The video` | 0.93 | 0.175 | 0.139 | 1.73 | 0.80 | 0.64 | 0.086 | 0.074 | 0.87 | −0.28 | 0.056 |
| `1kwIYXB4b8s::round0_offset::0::1` | `The video` | 1.73 | 0.135 | 0.074 | 0.89 | 0.55 | 0.61 | 0.077 | 0.055 | 0.71 | 0.49 | 0.026 |
| `0X0u_3xUBPE::round0_offset::0::1` | `The video` | 1.49 | 0.270 | 0.114 | 1.13 | 0.42 | 0.83 | 0.059 | 0.053 | 0.90 | 0.70 | 0.009 |
| `1KONAdZGdWU::round0_offset::0::1` | `The video` | 1.76 | 0.183 | 0.084 | 0.62 | 0.46 | 0.76 | 0.052 | 0.015 | 0.29 | 0.18 | 0.045 |
| `0DFz3sgfda0::round0_offset::0::1` | `The video` | 1.77 | 0.089 | 0.305 | 2.51 | 3.42 | 0.07 | 0.018 | 0.268 | NA | −0.05 | 0.026 |
| `0x6ZV9Y8g_c::C6_anchor::9::42` | `…colorful cartoon` | 1.45 | 0.253 | 0.179 | 0.56 | 0.71 | 0.33 | 0.589 | 0.080 | 0.14 | 0.17 | 0.248 |
| `0x6ZV9Y8g_c::C6_anchor::57::209` | `…captures the scene` | 1.31 | 0.078 | 0.018 | 0.76 | 0.23 | 0.84 | 0.119 | 0.039 | 0.33 | 0.47 | 0.040 |
| `0x6ZV9Y8g_c::C6_anchor::47::173` | `…kitchen utensils. A` | 1.28 | 0.083 | 0.225 | 0.79 | 2.70 | 0.75 | 0.146 | 0.194 | 1.33 | 0.77 | 0.022 |
| `0x6ZV9Y8g_c::C6_anchor::0::0` | `In` | 1.05 | 0.076 | 0.021 | 0.93 | 0.28 | 0.65 | 0.058 | 0.101 | 1.76 | 0.63 | 0.043 |

图：`main/analysis/fig_tv_N.png`（各视频 target/draft 在 T、K 下的概率质量移动与噪声底）。

**读法（严格按预注册）**
- 预注册的"支持时序特异弱响应"要求 **`R_T < 1` 且 `R_T < R_K` 且 `R_T − R_K` 的 CI 整体小于 0**。
  两个口径都**不满足第一条**（`R_T` = 1.27 / 1.10，均 > 1）。
- log-prob 口径下 CI 整体小于 0（`[−2.82, −0.32]`），方向上"draft 相对响应在 T 下弱于 K"；
  但该量级由单条视频主导：留一法去掉 `v_0DFz3sgfda0`（其 `R_K` 异常大）后 `R_T − R_K` 从 −1.22 降到 **−0.43**，
  其余 5 次留一样本维持在 −1.32…−1.42，**符号稳定、量级不稳**。
- 概率口径（有界、不受尾部支配）下 CI **跨零**（`[−0.289, +0.093]`），且留一去掉 `v_0X0u_3xUBPE` 后
  点估计翻号为 **+0.009** → 判为"不可区分"。
- `cos_T = 0.592`（CI 不含 0）明显高于 `cos_K = 0.256`（CI 含 0）：draft 的响应**方向**在时序反转上
  反而更贴合 target，不支持"时序上是弱项"。
- 9/9 单元 draft top-8 覆盖未变；O→T 只有 2/9 改变了 target 的贪心 token。

**结论等级：`未观察到稳定差异`。** 既不能声称 draft 对时序变化响应系统性更弱，也不能声称二者等效
（n=6 视频、9 单元，CI 跨零）。

---

## 6. 次要结果：family C（20 单元 / 5 视频，prefix 已被 T 否证）

| 指标 | 点估计 | 95% CI |
|---|---:|---|
| `R_T`（log-prob） | 1.471 | [0.962, 2.023] |
| `R_K`（log-prob） | 1.616 | [0.993, 2.358] |
| `R_T − R_K`（log-prob） | −0.470 | [−1.411, +0.350] |
| `R_T`（概率 TV） | 1.024 | [0.690, 1.314] |
| `R_K`（概率 TV） | 0.923 | [0.685, 1.161] |
| `R_T − R_K`（概率 TV） | −0.100 | [−0.375, +0.175] |
| `cos_T` / `cos_K` | 0.522 / 0.372 | [0.311,0.733] / [0.229,0.553] |
| target/draft 响应 TV（O→T） | 0.194 / 0.161 | — |
| 管线噪声底 target TV | 0.059 | [0.043, 0.078] |

方向与 family N 一致但更弱，两个口径的差 CI 均跨零。**这些 prefix 在 T 下已是假命题（"The video opens with …"
反转后不再成立），因此这 20 个单元只作为完整性与污染检查列出，不构成因果证据。**

---

## 7. 探索性：额外 offset（offset 2–8，teacher-forced 后缀）

按预注册单列为探索分析（这些更深的位置需要把 C0 的 AR token teacher-force 进 target，**不是**在线轮次）。
逐视频 `mean R_T`（log-prob 口径）：family N 为 3.71 / 2.07 / 3.87 / 0.64 / 2.40 / 0.84（末条是
`v_0x6ZV9Y8g_c`，其 family N 单元含 4 个 C6 anchor）；family C 为 0.64 / 1.13 / 1.13 / 0.65 / 0.97 / 0.84。
逐 offset 明细见 `per_offset_exploratory.csv`。**方向随视频与 offset 剧烈变化，没有跨视频一致的
`R_T<R_K` 或 `R_T>R_K` 模式**；target 贪心 token 在这些 teacher-forced 位置的变化比例最高 28.6%
（family N 的 `v_0X0u_3xUBPE`），说明它们与在线真实轮次不可混为一谈。不据此下任何结论。

---

## 8. 稳健性与负对照小结

1. **管线噪声**：概率口径下重编码噪声底 TV = 0.042，约为 O→T target 响应的 26%；没有单元的噪声底
   超过其 O→T 响应，说明所有 O→T 差异不是编解码伪影。
2. **两个口径分歧**：log-prob 差 CI 不含 0、概率差 CI 含 0。这是尾部量级敏感度造成的，
   已在 §2.4 预先披露并同时报告；**不选取对结论有利的那一个**，结论按"不可区分"收口。
3. **留一视频**：log-prob 差符号稳定但量级从 −1.22 变到 −0.43；概率差在去掉 `v_0X0u_3xUBPE` 后翻号。
4. **单元嵌套**：9 个单元里 4 个来自同一条视频。所有 CI 以视频为重采样单位，C6 的 48 轮与本方向的 29 个单元从未
   被当作独立样本。
5. **未做/不可做**：没有在线解码，因此没有 τ、rounds、ms/token；没有新测试集；没有训练。

---

## 9. 限制（必须随结论一起引用）

- 因果子集只有 **9 个单元 / 6 条视频**，且 5 条视频各只有 1 个位置（同一个 prefix `"The video"`），
  位置同质、n 极小；CI 跨零只能写"不可区分"，不能写"等效"。
- 因为 caption 模板本身的时序措辞，**C6 的 `not_in_topk` 位置没有一个能进入因果子集**。
  本报告给出的是覆盖率证据，不是对 `not_in_topk` 机制的检验。
- 干预作用在 16 帧抽样序列上，帧间隔 1.17–15.0 s；**不是**原视频连续时间轴上的事件重排。
- K 是**空间**镜像而非时序微扰：它保持时序但改变左右关系。选择它的理由是一个"保持事件先后"的
  时序重排在本数据上只能是恒等操作；代价是 K 不是纯"同时序、同模态"的对照。镜像会使画面内文字
  （`v_0X0u_3xUBPE` 的标题卡、各处水印）反向，已记录。
- prefix 有效性用"时序措辞词表 + 人工视觉复核"判定，属保守判断，不是形式化语义蕴含证明。
- 只有 1 个 target 精度（fp32）、1 个 draft（C0）、1 个温度（0）、1 组候选 top-8。

---

## 10. 是否建议下一步训练（明确回答）

**不建议基于本方向启动时序条件化的相对偏好蒸馏或任何新训练。**理由：

1. 本探针在唯一因果可用的 9 个位置上**没有可重复的"draft 时序响应更弱"信号**（概率口径 CI 跨零，
   留一即翻号）；在没有稳定缺口时做相对偏好蒸馏会是在噪声上拟合。
2. 更根本的是**数据供给问题**：本轮已量化，在这些验证视频的 caption 前缀上，全时序反转会让
   本轮冻结的 29 个单元中有 20 个的前缀会被全反转否证；要构造"prefix 在 O/T/K 都成立且 target 高熵"的位置，
   需要**非时序叙述式 caption**或**非时序干预**（例如只在 prefix 未描述的后缀上做反转，但这需要
   精确的 caption↔帧时间对齐，本轮测得线性对齐误差可达 ~4 帧，不足以支撑）。
3. 若仍要走路线 B，建议先做的是**数据/标注侧准备**而不是 loss：为候选视频人工标注句级
   "该句描述了哪些帧区间"，据此构造"prefix 未覆盖后缀反转"的干预；在此之前不应指定 loss。

**唯一有价值的即时后续**（纯 CPU/离线，沿用本目录产物）：在 `v_0x6ZV9Y8g_c`（唯一叙事式中立 caption 的视频）
的 4 个中性单元上重复同协议，检验 `cos_T > cos_K` 是否可重复；以及在 `pool_499` 中按
"caption 以非时序模板开头"筛出更多视频，扩大 6 视频上限下的因果子集。二者都不训练。

---

## 11. 交付与复现

```bash
cd /home/mcy/Projects/GLANCE
RUN=/mnt/local/mcy/GLANCE/results/parallel_B_temporal/20261012T233231
PY=/home/mcy/.conda/envs/fin_research/bin/python
# 新一次运行请用新时间戳目录：
bash repro/experiments/temporal_d4_b/run_b.sh <新时间戳>
# 只重跑分析与图（CPU，不占 GPU）：
$PY repro/experiments/temporal_d4_b/analyze_b.py --run "$RUN" --out "$RUN/main/analysis"
$PY repro/experiments/temporal_d4_b/make_figures.py --run "$RUN"
# package_b 在未给 --outcome 时会沿用已存在的 run_manifest.json 结论；新目录才自动推断
$PY repro/experiments/temporal_d4_b/package_b.py --run "$RUN" --main main
# CPU 测试（13/13）
cd /home/mcy/Projects/GLANCE && B_RUN="$RUN" $PY -m unittest discover -s repro/experiments/temporal_d4_b -p "test_*.py" -v
```

产物清单（`run_manifest.json` 内含 SHA256）：

| 路径 | 内容 |
|---|---|
| `prereg.json` / `prereg_v1_superseded.json` / `prereg_amendments.json` | 冻结协议（v2）、v1 留痕、冻结后纯元数据修订记录 |
| `stageA/audit_c6.json` | C6/模型/基线哈希与来源 |
| `stageA/candidates.csv` | 86 个候选位置及入选/排除理由 |
| `stageA/selection.json` | 冻结单元（family N 9 / family C 20） |
| `frames/frame_manifest.json` + `frames/sheet_*.png` | 16 帧原始帧号/时间、O/T/K 联系表 |
| `conditions/conditions.json` + `*.mp4` | O_rec/T/K 文件、SHA256、逐帧置换验证 |
| `main/forward_records.jsonl` | 29 单元 × 4 条件的完整 logit 记录与指标 |
| `main/gates.json` | G1/G2/G2b/G3/G4/G5 与 pilot 门禁 |
| `main/calls.json` | 调用计数、耗时、峰值显存、checkpoint 前后哈希 |
| `main/analysis/` | `per_unit.csv`、`per_video.csv`、`per_offset_exploratory.csv`、`prefix_provenance.csv`（逐单元 prefix ID/hash、anchor、M-RoPE 与视觉绑定说明）、`summary.json`、`fig_tv_N/C.png` |
| `pilot_v1_metric/` | v1 指标口径下的作废 pilot（独立留痕，不计入证据） |
| `pilot.log` / `main.log` / `tests.log` | stdout/err |

计算开销：真实前向 24 次 target + 122 次 head（含 G2b 门禁），processor 24 次；主循环 65.5 s，
峰值显存 38.53 GiB（GPU0，fp32 target + fp32 head）；前后 checkpoint SHA256 一致。无梯度、无优化器、
无生产解码文件修改；未触碰 `GLANCE_repo/glance/`、`repro/common.py`、根 `AGENTS.md`。
GPU 全程在共享锁 `/mnt/local/mcy/GLANCE/.locks/glance_parallel_gpu.lock` 内执行，并在取锁后复查实际空闲显存。

---

## 12. 可粘贴到 AGENTS.md 的状态块

```markdown
| **phase 并行任务 B：视频时序条件响应** | **已完成，`未观察到稳定差异`** | 复用 C6 冻结 6 视频；自建 O/T/K(=全时序反转/保持时序镜像)+O_rec 噪声门禁，帧级置换已复核；G1 6/6、G2 29/29、G2b 6 视频×15 offset 全同、G3 全同。**可行性主发现：全时序反转会使 caption 前缀成假，86 个候选位置只有 15 个"前缀时序中立且 target H≥0.3"，最终 9 个（5 条视频各 1 个 prefix="The video" + v_0x6ZV9Y8g_c 4 个）；C6 的 24 次 not_in_topk 无一进入因果子集。** 主结果（n=6 视频）：target 对反转响应 TV=0.162（噪声底 0.042），draft 相对响应 `R_T=1.10` vs 镜像 `R_K=0.73`，`R_T−R_K=−0.089 CI[−0.289,+0.093]` 跨零、留一翻号；log-prob 口径 `R_T−R_K=−1.22 CI[−2.82,−0.32]` 但 `R_T>1` 且量级由单一视频主导 → 按预注册判为不可区分。`cos_T=0.59` 反高于 `cos_K=0.26`。**不建议据此启动时序相对偏好蒸馏**；建议先做句级 caption↔帧区间标注，再构造"未描述后缀反转"干预。产物 `results/parallel_B_temporal/20261012T233231/`，`run_manifest.json` 为入口；报告 `docs/reports/2026-10-12-parallel-B-temporal.md`。未训练、未改 head/verifier/共享文件，未打开新测试集。 |
```
