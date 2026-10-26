# GLANCE 视频 OPD 后续计划

> 2026-10-09 文档整理补注：当前进度统一见 [AGENTS.md](../../../AGENTS.md)。
> B1 公式核实已完成，见[核实记录](../../reports/2026-10-09-b1-formula.md)；`repro/opd_fullkl.py` 已存在，尚不能据此认定完整训练与测试通过。
> B0.5 已有 `results/phaseB05/subsets.json`（8 条训练视频与 8 条监控视频），精度/梯度校准仍待完成。
> 下文保留原始预注册计划及执行约束；裸 `repro/`、`docs/` 路径相对项目根，`results/` 等路径相对产物根。

更新日期：2026-10-09（最新修订：新增 §10 DeepSeek 长程执行指令，覆盖 B0.5、B1 与 Phase C 首轮）  
状态：阶段 A/A4 完成，B0 机制核验收口；正式起点为 **aligned step1000**；B0.5 样本已冻结、校准待完成；B1 公式已核实、模块已有实现，完整训练验证与 Phase C 待完成  
依据：`docs/reports/2026-10-08-video-opd-2x2.md`、`docs/reports/2026-10-09-phase-a.md`

> **执行入口：§10「DeepSeek 长程执行指令」**。本次交付是执行计划，未启动训练。
> 本轮范围为 B0.5 → B1 → seed-0 三臂，训练臂先各 50 updates，满足条件才延长至 100。
> §10 是本轮执行范围、精度选择及继续/停止规则的最新依据；§6 多种子及 §7 转向仅为后续规划。

> **2026-10-09 阶段 A 结果**：见 `docs/reports/2026-10-09-phase-a.md`。
> clean `video_fresh30`（fp32, b63k8, n=29, lossless 30/30）上
> step500/1000/1500 = **3.5389 / 3.5785 / 3.5246**，极差 0.0538 > 0.03 且
> **先升后降**——按 §3 A3 第三条取 clean 泛化峰值 **step1000**，不是 plateau。
> 但 step1000 相对 step500 仅 +0.0396（t=1.56），是"当前观测峰值"而非稳固唯一最优点。
>
> **2026-10-09 关键修订（作废上述起点）**：这三个 checkpoint 全部由旧的
> `use_cache=False` 路径训练。`repro/check_context_interface.py` 实测该路径下
> **文本位置 hidden 对视频输入逐位无关**（置零视频后 `max|Δ| = 0.000e+00`，
> 3 行均为 0），而部署解码走 cached 路径（同口径下 `max|Δ|` 为 2.4e2~9.8e3）。
> 同时已证 `target_context_cached` 与 `target_context_aligned` **逐位相同**
> （`max|Δ| = 0.000e+00`，cos=1.000000），即两臂可以共用同一个 target prefill 接口。
> 因此**阶段 B/C 的正式起点改由 §3.1 阶段 A4（aligned SFT 重新定峰）给出**；
> `head_video439_plateau/*` 降级为历史路径基线保留。
>
> **2026-10-09 阶段 A4 结果**：见 `docs/reports/2026-10-09-phase-a4.md`。
> aligned 曲线（fp32, b63k8, n=29, 六个 checkpoint 全部 lossless 30/30）：
> step250/500/750/1000/1250/1500 = **3.5840 / 3.6517 / 3.7125 / 3.7377 /
> 3.7129 / 3.6776**，极差 0.1537、**先升后降** → 按 A4.4 取 aligned 泛化峰值
> **`step1000`（mean τ 3.7377）**。
>
> **并行的最重要结论**：同一步数上 aligned 比 legacy 高 **+0.1128 / +0.1592 /
> +0.1530 τ**（step500/1000/1500，配对 t = 5.33 / 7.30 / 5.29，20~27/29 行为正）。
> **仅修特征路径不一致就是本项目迄今最大的单一效应**，比 2×2 的 online 主效应
> （+0.0710）约大 2.2 倍（以 step1000 的 +0.1592 比较），不是一个数量级。
>
> **阶段 B/C 起点 = `head_video439_aligned/step1000`**。

## 1. 当前结论与研究问题

当前只能判定：**现有 Bernoulli + teacher top-64 + first-reject forward-KL 配方在
`video_fresh30` 的 fp32、全长度、GLANCE tree b63/k8 主口径上无效**。不能据此判定
“OPD 在视频领域无效”。

已确认的事实：

1. online rollout 有真实但不足的收益：2×2 实验的 online 主效应为 `+0.0710 τ`，
   95% CI `[+0.0477,+0.0943]`。
2. 最佳 online arm 仍比 warm start 低约 `−0.27 τ`。
3. chain/tree 训练拓扑不是当前失败的充分解释；online+chain 与 online+tree 最终 τ
   几乎相同。
4. 当前 rejected reverse-KL stream 为空（`n_k3_rej=0`），与 Draft-OPD 原论文目标不同。
5. online 后 teacher top-64 外 proposal 比例升到约 18%，说明截断 teacher 在最重要的
   on-policy 状态上更加不精确。

下一阶段要回答两个问题：

- **Q1：从 aligned SFT 泛化峰值附近开始，完整 OPD 能否改善全长度投机解码效率？**
  A4 支持 step750–1250 的宽峰，不能据此检验“严格 plateau 是否为 OPD 的必要条件”。
- **Q2：使用论文忠实的 online teacher scoring 和 mixed KL 后，视频 OPD 能否超过同起点、
  同预算的继续 SFT？**

## 2. 关于 SFT plateau 的当前判断

现有旧验证集上的 SFT 曲线：

| checkpoint | mean τ @b63/k8 |
|---|---:|
| step500 | 3.6464 |
| step1000 | 3.6408 |
| step1500 | 3.6383 |
| step2000 | 3.5962 |
| step3000 | 3.5231 |
| step6000 | 3.2295 |

`step500–1500` 已基本压平，之后持续退化，所以 `step1000` 更接近“早期泛化最优”而不是
明显欠训练。但是，目前只有 `step1000` 在真正独立的 `video_fresh30` 上做过严格 fp32
评测，因此**不能把它写成已被严格证明的 plateau**。

这里按行为定义 plateau，不照搬论文的 epoch 数：若连续三个 checkpoint 在 clean set 上
没有单调提升，且最大差 `<0.03 τ`，则视为视频 SFT plateau。视频数据只有 439 行，论文的
“6 个 SFT epoch”不能直接作为视频侧判据。

## 3. 阶段 A：补齐 clean SFT plateau 曲线

### A1. checkpoint

从相同初始头、相同 409/439 行训练数据重新训练并保存：

- step500；
- step1000；
- step1500；
- 可选 step2000，用于确认平台之后是否开始过拟合。

若历史 checkpoint 可恢复且配置哈希一致，则直接复用；否则重训，不能用只有旧验证结果的
checkpoint 冒充 clean 曲线。

### A2. 评测

- 数据：`video_fresh30.parquet`；禁止用已与训练集重叠的历史 `video_heldout_30` 作主结论。
- dtype：fp32。
- decoder：GLANCE tree b63/k8，跑到 EOS。
- 每个 checkpoint 记录：mean/median τ、逐样本 τ、round 数、生成长度、EOS 命中率。
- fp32 speculative 输出必须与 AR target 逐 token 一致；不一致则该结果无效。

### A3. 判定

- 若 step500/1000/1500 最大差 `<0.03 τ` 且无单调上升：确认 plateau，选其中 clean mean τ
  最高的 checkpoint。
- 若仍单调上升：继续 SFT 到首次平台，再进入阶段 B。
- 若先升后降：选择 clean 泛化峰值，不为了满足“6 epochs”继续过拟合。

## 3.1 阶段 A4：aligned SFT 重新定峰（2026-10-09 新增）

阶段 A 的曲线作废为起点，原因是特征路径而非数字本身。

### A4.1 动机（已实测，不再沿用旧报告转述）

`repro/check_context_interface.py` 在 3 条 `video_fresh30` 行上实测：

| 路径 | 文本位置 hidden 置零视频后 `max|Δ|` | 说明 |
|---|---:|---|
| `common.target_context`（旧训练路径） | **0.000e+00**（3/3 行） | 文本行对视频逐位无关 |
| `common.target_context_cached` | 2.440e+02 / 3.190e+02 / 9.778e+03 | 文本行依赖视频 |

即：**过去所有视频头都在"文本行不依赖视频"的特征上训练，却在"文本行依赖视频"的
特征上评测**。这类 checkpoint 不能作为 paper-faithful OPD 的强 SFT 起点。

同时已证两条对齐实现是同一个对象：

| 对比 | `max|Δ|` | 逐位相同 | cos |
|---|---:|---|---:|
| `target_context_cached` vs `target_context_aligned` | **0.000e+00** | 是（3/3 行） | 1.000000 |

**因此两臂可以、也必须共用一个 target prefill 接口。**

### A4.2 特征路径口径

- **`aligned` 是语义要求**：训练 hidden 必须与解码 hidden 一致。
- **`cached` 是实现方式**：可缓存或现场计算，只要张量逐位一致。
- **统一为 cached/decode-aligned，OPD 不退回旧路径；continue-SFT 必须一起改**，
  否则 SFT 与 OPD 之差会混入"特征路径修复"的收益。

两臂共用：

```text
一次 cached target prefill
├── fused hidden  → SFT 和 OPD 的 head context
└── prefix KV     → OPD draft-induced prefix teacher scoring
```

**门禁**：同一行上 SFT context 与 OPD context `max|Δ| = 0`。
`train_head_video.py` 新增 `--feature-path {uncached,cached}`（默认 `uncached`，
保证历史数字可复现）；`opd_train_video.py --aligned-forward` 走对齐路径。二者已实测
逐位一致。

### A4.3 执行

- 起点：图像 warm head `checkpoints/head`（与阶段 A 相同）。
- 唯一变量：`--feature-path cached`。其余（439 行、目标函数、reveal 先验、block
  布局、`--max-anchors 96 --lr 3e-4 --seed 0`）与阶段 A 逐项相同。
- 保存 step250/500/750/1000/1500（实现取 `--save-every 250`，故另含 step1250）。
- 评测：`video_fresh30`，fp32，tree b63/k8，跑到 EOS，`--ar` 无损门禁。

### A4.4 判定

与 A3 同一套预注册规则（极差 `<0.03` 且非单调上升 → plateau；单调上升 → 继续；
先升后降 → 取 clean 峰值）。**峰值即阶段 B/C 的正式起点**；旧
`head_video439_plateau/*` 仅作历史路径基线。

## 4. 阶段 B：实现 paper-faithful OPD

### B0. full-KL 可行性原型（2026-10-09 新增，先于完整 trainer）

不要一上来写完整 trainer。先用最小规模验证机制与显存：

- 1 条视频；
- 4 个 chain anchors；
- prefix KV fork；
- full-vocabulary forward/reverse KL；
- teacher logits 按 anchor/chunk 即算即用。

必须验证：

- cached scorer 与独立完整 target forward 的 logits 一致；
- target 无梯度；
- rejected suffix 的上下文确实包含此前的 draft proposal；
- A6000 峰值显存保持在约 44 GiB 以下；
- 不落盘完整 logits/KV。

B0 不通过则不进入 B1。

> **2026-10-09 B0 执行结果**（`repro/proto_fullkl_b0.py`，
> `results/phaseB0/proto_fullkl_b0.json`）：机制成立且便宜——1 条视频、4 个 chain
> anchor、每 anchor 一次 prefill + `copy.deepcopy` 分叉 prefix KV，把 drafter 自己
> 的 proposal 推进分叉得到**全词表** teacher logits；峰值 **19.19 GiB**，teacher 路径
> 合计 10.1 s。
>
> | 门禁 | 结果 |
> |---|---|
> | G2 target 无梯度 | PASS（requires_grad 0 / grad 0） |
> | G3 drafted-induced prefix | PASS（见下） |
> | G4 显存 <44 GiB | PASS（bf16 19.19 GiB） |
> | G5 不落盘 logits/KV | PASS |
> | G1 增量 == 独立完整前向 | **bf16 下非逐位**：`max|Δlogit| = 0.375`（`|logit|max 51.2`），TV 0.092，60 个 offset 中 1 个 argmax 翻转 |
>
> G1 的根因已定位并**独立复核**：bf16 GEMM 对 batch 行数敏感，不是结构错误。
> 证据：(i) 同一增量方法按 1-token 步进 vs 15-token 整块推进，差异同为 0.375；
> (ii) mRoPE 位置逐位复现（脚本内 assert）；(iii) 独立微实验：bf16 下
> `M=1 vs M=16` 的 `lm_head` GEMM 差 3.1e-2，而 **fp32 下为 0.000e+00**。
>
> **B1 的取舍（已实测）**：把 teacher 前向换成 fp32 后，同一门禁的差异从
> `0.375` 降到 **`3.43e-05`**（`|logit|max 40.7`，相对 ~8e-7），chunking 对照同为
> `3.72e-05`，峰值 **34.09 GiB**（原型阶段仍 <44）。不能把 bf16 的 logit 差异
> 概括成固定“0.4% 目标值漂移”：TV 和 KL/梯度影响需要单独校准。
> **最新决策：B1 优先 fp32 teacher**，先测含 backward 和已初始化 optimizer 状态的
> 真实训练峰值；精度替代与校准规则见 §10。fp32 前向允许合理数值容差，
> 部署解码的 AR 逐 token 无损门禁仍须精确通过。
>
> **G3 的重要修正**：第一个 rejected offset 的 draft-induced prefix 与 target greedy
> prefix **按定义相同**（`n_acc` 就是匹配前缀的长度），所以在那里测"两者不同"必然为 0
> ——这是正确性证明，不是缺陷。真正的分歧从 `k = n_acc + 2` 才开始出现：
> 首个 rejected offset 处 KL 仅 5.1e-4（噪声底），而 `n_acc+2` 处 KL 达
> **19.6 / 42.2 / 23.6 / 0.87**（4 个 anchor）。**结论：paper-faithful 的
> draft-induced 前缀语义只在 `k >= n_acc+2` 处真正生效。**
>
> **对 B1 最重要的实质结论**：该窗口内 accepted 流 mean forward KL 仅 **0.102**
> （5 个 offset），而 rejected 流 mean reverse KL 达 **9.05**（55 个 offset，单个
> offset 最高 36.7 nats）。**当前配方把这条流整个置零（`n_k3_rej=0`），而它承载了
> 该小窗口的大部分原始 KL 总量。** 该统计不等同于实际加权目标或梯度贡献；
> 不得推广为 aligned 起点的总体结论。B0.5 将核实原型配置并在 aligned 训练样本上校准。

### B1. 必须实现的语义

1. **current-policy rollout**：每个训练 step 使用当前 head，不能反复消费 warm-start 静态
   anchor pool。
2. **draft-induced prefix teacher scoring**：target 必须在 drafter 实际提出的前缀上打分；
   rejected suffix 不能查 target greedy 轨迹同位置的缓存 logits 冒充。
3. **full-vocabulary accepted loss**：accepted positions 使用
   `KL(p_target || q_draft)`。
4. **full-vocabulary rejected loss**：从第一个 rejected token 到 block 末尾使用
   `KL(q_draft || p_target)`。
5. **待原文/官方实现核对的权重约定**：`w_k = 0.8^(k-1)`，accepted/rejected 两个
   stream 等权并分别归一；B1 编码前须核实 γ 的作用位置、分母、整体系数及空流行为，见 §10。
6. target 冻结且无梯度；只更新 draft head。
7. 第一版固定 chain/block rollout，以最接近 Draft-OPD 原论文；确认有效后再接 tree。

### B2. 工程实现顺序

1. 给 target scoring 建立 prefix KV cache，避免每个 anchor 重算完整视频 prompt。
2. 在一个 proposal block 内构造 draft-induced prefix，计算 target full-vocab logits。
3. 实现独立的 `paper_fullkl` loss，不修改当前负基线 loss 的默认行为。
4. 将 teacher logits 即算即用，不落盘完整词表张量。
5. 支持 gradient accumulation；记录 target scoring 和 draft replay 各自耗时/峰值显存。

若 full-vocabulary 显存不可接受，可增加一个**明确标记为近似**的 `topk+tail` 诊断 arm；
不得再把“只在 top-k 内重归一化”称为论文忠实 KL。

### B3. 单元和集成门禁

- full-vocab forward/reverse KL 与直接 PyTorch 公式一致。
- `branch_k=1` 时 tree rollout 与 chain 的 proposal/n_acc 一致。
- teacher 行和 student offset 无 off-by-one。
- optimizer 更新一次后，online anchors/proposals 至少有一处可变化；frozen control 不变。
- target 参数梯度始终为空。
- 一条视频完成 rollout → replay → backward，loss 有限、显存不持续增长。
- fp32 解码保持逐 token lossless。

## 5. 阶段 C：先做单 seed 筛选

### C1. 实验矩阵（2026-10-09 修订：缩为三臂）

**不再**跑原计划的"三个旧起点 × 两方法"6 个 arm（旧起点已被 A4 作废）。改为从
**A4 选出的 aligned SFT 峰值**单一启动，seed 0，共 3 个 arm：

| arm | 说明 |
|---|---|
| **no-training start** | A4 峰值本身，不训练；给 Δτ 提供零位移参照 |
| **aligned continue-SFT** | 同预算继续 SFT，**必须同样走 cached/aligned 路径** |
| **paper-faithful online OPD** | B1 语义，同预算 |

三个 arm 必须使用：

- 相同训练行与样本顺序；
- 相同 optimizer step 数和有效 anchor/position 预算；
- 相同 checkpoint 起点（A4 峰值）；
- **相同特征路径（cached/decode-aligned）**——这是 A4 修订的核心，不得让
  continue-SFT 走旧路径而 OPD 走对齐路径；
- 相同评测样本和 fp32 decoder；
- OPD 额外 target scoring 成本单独报告，不能用 wall-clock 相等代替 step/样本预算相等。

### C1.1 主对比口径

主对比是 `paper-faithful OPD − aligned continue-SFT`；`no-training start` 只用于
说明同预算训练本身带来多少位移。**不允许**把"相对 no-training/frozen 有提升"写成
"相对继续 SFT 有提升"（见 §8）。

### C2. 主次指标

主指标：

- `video_fresh30` 全长度 mean τ；
- paired `Δτ = OPD − continue-SFT`；
- fp32 lossless 通过率。

次指标：

- 前 60 verification rounds 的 τ/覆盖率；
- accepted/rejected positions、每 offset acceptance；
- teacher entropy 分桶后的 Δτ；
- decode-only speedup 和 wall-clock；
- teacher scoring 中 rejected suffix 的概率质量与 KL。

### C3. 预注册筛选判据

- 进入多 seed：A4 峰值起点上，全长度 `Δτ = OPD − aligned continue-SFT >= +0.05`，
  逐样本配对方向为正，且 fp32 lossless 100%。
- 仅“r60 为正、全长度不增”：记为前段 curriculum 效应，不能宣称视频 OPD 全局有效。
- 本轮 50/100 updates 的继续/停止规则以 §10 为准；短预算负结果只支持当前配置下的
  判断，不直接宣称整个视频 OPD 或纯 KL 路线无效。
- OPD 相对 continue-SFT 为正但 `<0.05`：记录为统计/工程价值过小，除非吞吐收益明显，否则
  不进入昂贵复现。

## 5.1 评测集纪律：`video_fresh30` 已升格为 validation（2026-10-09 新增）

`video_fresh30` 已被用于**选择** A4 的峰值检查点，因此从现在起它只能算
**validation/选点集**，不能再当最终 test。另外核对：剩余 30 条 heldout 也已出现在
大量历史评测结果里（`results/opd_task/` 等）。**因此现有 60 条 heldout 都不是严格
untouched test。**

- **开发和筛选阶段**：继续用 `video_fresh30`，这是允许的。
- **最终宣称"视频 OPD 有效"之前**：必须另外**冻结一批未参与任何选点的视频测试集**
  （不参与 A4 定峰、不参与阶段 C/D 的任何 arm 选择、不参与超参或配方筛选），并且
  在冻结时记录其来源、划分 seed 与"从未被任何选点流程读取"的证据。
- 未冻结新测试集之前，所有结论表述限定为 **validation 口径**，不得写成 test 结论。

## 6. 阶段 D：通过后做 3 seeds

只有阶段 C 达标才执行：

- A4 选出的 aligned SFT 峰值起点；
- aligned continue-SFT vs paper-faithful OPD（同特征路径）；
- seeds 0/1/2；
- 逐 seed、逐样本配对；
- 报告 mean Δ、95% CI、t 值、正向样本数以及训练/评测成本。

若 3 seeds 后 OPD 稳定超过继续 SFT，且冻结的新测试集确认收益，才支持在所测模型与数据
条件下的泛化收益结论；否则表述限定为
“online teacher feedback 有信号，但未形成稳定的全长度方法收益”。

## 7. 忠实版失败后的转向

若阶段 C/D 失败，按以下顺序转向，不再继续微调 KL 权重：

1. **直接 acceptance objective**：优化 verifier 接受概率/接受长度，而不是假设 KL 下降会
   自动转化为 τ 上升。
2. **SFT 数据改造**：使用 sampled target traces、多 prompt、长响应 prompt，检验 greedy
   trace 的一一对应退化和数据量瓶颈。
3. **entropy-aware curriculum**：先训练低熵、可学习位置，再逐步加入高熵视频描述位置。
4. **verification-side 方法**：若允许质量/保证权衡，再评估 relaxed verification；必须与
   GLANCE 的 lossless 主卖点分开报告。

## 8. 当前禁止项

- 不再为现有 Bernoulli/top-64/first-reject-forward 配方补更多 seeds。
- 不先扫 tree budget、branch factor 或 learning rate 来掩盖 loss 语义问题。
- 不使用 bf16 输出路径作为最终 lossless τ 结论。
- 不用训练集重叠的历史 holdout 作为泛化主指标。
- 不把“相对 frozen OPD 有提升”写成“相对继续 SFT 有提升”。
- **不让阶段 C 的两条臂走不同特征路径**（continue-SFT 必须与 OPD 同为
  cached/decode-aligned），否则方法差会混入特征路径修复收益。
- **不把 `video_fresh30` 当最终 test**（它已是选点集），未冻结新测试集前不得宣称
  test 结论。见 §5.1。
- 不在 B0 原型通过之前直接跑完整 paper-faithful trainer。

## 9. 交付物

- clean SFT plateau 曲线与逐样本结果（阶段 A，已完成）；
- **A4：aligned SFT 曲线、峰值 checkpoint 与特征路径门禁证据**；
- **B0：full-KL 可行性原型与显存/一致性门禁结果**；
- `paper_fullkl` 实现及门禁测试；
- seed-0 三臂筛选汇总（no-training / aligned continue-SFT / paper-faithful OPD）；
- 若达标，3-seed 配对报告；
- **一批冻结的、未参与任何选点的新视频测试集**（最终 test 口径的前提）；
- 最终结论必须明确区分：相对 warm、相对 continue-SFT、全长度、r60 四种口径，并标注
  validation 还是 test。

## 10. DeepSeek 长程执行指令（2026-10-09 最新任务）

请在 `/home/mcy/Projects/GLANCE` 中执行本节长程任务。负责检查、实现、测试、运行实验、
分析结果和更新文档；不要只给方案或在每个阶段结束后等待确认。

**总目标**：从已完成的 A4/B0 出发，完成可信的 paper-faithful 视频 OPD 实现，并通过
aligned 起点下的 Phase C seed-0 三臂实验，判断完整 OPD 是否改善投机解码效率。
本节未勾选项目均为待执行任务，不代表已有实验结果。

### 10.1 先核实当前状态

- [ ] 阅读适用的 `AGENTS.md`、相关代码、本计划及以下产物：
  - `docs/reports/2026-10-09-phase-a4.md`
  - `repro/proto_fullkl_b0.py`
  - `repro/check_b0_fp32_invariance.py`
  - `/mnt/local/mcy/GLANCE/results/phaseB0/` 下的配置和结果。
- [ ] 保留已有修改和历史复现能力。实际文件和运行产物优先于聊天描述；发现冲突时记录
  证据并修正工作假设。

已知起点为 `/mnt/local/mcy/GLANCE/checkpoints/head_video439_aligned/step1000`，
validation mean τ 为 3.7377。A4 的 step750–1250 是宽峰，不是已证明的严格 plateau。
OPD 与 continue-SFT 均采用 cached/aligned 特征语义。
B0 已验证 prefix KV fork、draft-induced prefix、target 无梯度及不落盘机制；
bf16 teacher 存在增量/独立完整前向数值差异，fp32 明显降低差异。
现有 60 条 heldout 均不得称为 untouched test。

### 10.2 B0.5：在正确起点校准

- [ ] 核实 B0 实际 head、样本和精度，不把旧 head 少量 anchors 的 KL 统计推广到新起点。
- [ ] 从训练集固定抽取 8 个视频，每个视频最多 4 个有效 anchors；使用 aligned step1000。
  固定 seed，保存样本 ID、anchor 位置和配置。核实与验证集无重叠，不使用验证得分选择样本。
- [ ] 在相同 rollout tokens、prefix 和 mask 上比较 bf16/fp32 teacher，避免混入 rollout 变化。
- [ ] 汇总 accepted/rejected 有效位置数、accepted 长度、两流 KL 的均值及
  P50/P90/P99/max、teacher 分布 TV、argmax 差异、总 loss 的绝对/相对差异、显存和用时。
- [ ] 固定 student 状态，在代表性 anchors 上比较两种 teacher 精度产生的 head 梯度
  cosine 和范数比。逐 tensor 累计内积/范数，不额外拼接整个模型梯度。

只保存汇总统计、token ID 和配置，不保存完整 logits、teacher 分布或 KV。
本阶段用于诊断，不根据结果反复调出最优验证指标。

### 10.3 B1：实现与公式核对

- [x] 查阅项目已有论文原文/官方实现；必要时获取原论文或官方仓库。逐项核对公式、
  KL 方向、γ 的作用位置、归一化、空流处理、rollout 和 masking，记录具体依据。
  先前建议的 `1/2` 系数、两流分别取均值等为待核对假设，不可未经核实硬编码。
  论文、官方代码与项目设计有差异时，明确选择及原因。
- [ ] 新增独立 full-KL 实现和训练入口，保留旧 trainer 的行为。

实现要求：

- aligned step1000 初始化，cached/aligned context。
- 尽可能共用一次 target prefill，获得 fused hidden 和 prefix KV。
- 当前 draft policy 在线 rollout，即算即用；第一版采用 chain/block rollout。
- accepted 为全词表 forward-KL，rejected 为全词表 reverse-KL。
- `γ=0.8`，应用位置遵循核实后的定义；teacher stop-gradient。
- `log_softmax` 与 KL 运算采用 fp32；不以 top-k 重归一化替代全词表 KL。
- 不落盘 logits、分布和 KV；正确处理 EOS、padding、无 accepted、无 rejected 和截断。
- 正确实现 draft-induced prefix：第一处 rejected token 的条件前缀仍与 target prefix
  相同，分叉在后续位置才产生。

梯度累积要求：单次 optimizer update 内固定 draft 参数；各 microbatch 的 loss 缩放
必须等价于声明的整个 update 归一化，不能简单平均不同 token 数 microbatch 的均值。
不能为精确归一化保留所有 microbatch 的计算图而导致显存失控。

- [ ] 增加必要测试：小分布手工 KL 对照；mask/γ/分母/空流；target 无梯度且 head
  梯度有限非零；prefix fork 不污染共享缓存；teacher 增量/独立前向合理容差；
  起点加载和可训练参数范围；optimizer 只实际更新 head；累积梯度与不拆分计算一致。

区分 teacher 浮点一致性与部署无损性：前者允许合理浮点容差，后者的 AR 逐 token
门禁必须精确通过，不得互相替代。

### 10.4 teacher 精度与真实训练可行性

- [ ] 科学主实验优先 fp32 teacher，student head 可用 bf16；记录各模块实际 dtype，
  防止 autocast 无意降低 teacher 精度，核实 dtype 边界与共享权重处理。
- [ ] 至少完成 3 次真实 optimizer updates，覆盖 rollout、teacher scoring、head backward
  和 optimizer.step，必须测到 Adam 状态初始化后的显存峰值。目标峰值 `<44 GiB`。

显存不足时依次尝试：减小 microbatch anchors 并正确累积；及时释放 teacher 中间结果、
KV fork 和无用图；采用不改变目标的全词表分块或 student activation checkpointing。
不得删掉 rejected loss、截断 top-k 或改变特征路径来节省显存。

若 fp32 仍不可行，记录完整证据；可实现 bf16 工程替代，并用 B0.5 同轨迹比较报告误差。
该结果必须标为精度近似，不能据其负结果宣称高精度 paper-faithful OPD 无效。
不得临时放宽阈值把已观察误差包装成通过。继续完成可运行替代的实现和诊断；若精度证据
不足以支撑正式比较，则以精度阻塞报告收口。

### 10.5 Phase C：先冻结 seed-0 配置

- [ ] 正式训练前写入配置及本计划：起点路径/哈希、样本顺序/seed、optimizer/LR/scheduler/
  clip、anchors/update、累积及归一化方式、各模块 dtype、checkpoint/评测节点和停止规则。

默认目标为 24 anchors/update，microbatch 由显存测试决定。若必须减少有效 anchors/update，
在正式实验前统一确定，同步应用两条训练臂。

| Arm | 设置 |
|---|---|
| C0 | aligned step1000，不训练 |
| C1 | 同一起点 continue-SFT，cached/aligned |
| C2 | 同一起点 paper-faithful OPD，在线 chain rollout |

C1/C2 尽量配平视频顺序、更新次数、anchor 采样预算和优化器设置；核实两臂 anchor 定义
是否可比，无法配平之处如实记录。报告视频曝光数、anchor 数、有效监督位置数、updates、
GPU 时间及显存，不能把 updates 相同等同于计算预算相同。
两臂用于 head 的 context 精度/语义也须对齐；若升级 teacher 精度影响 context，记录并
验证共同接口，避免将数值路径变化混入 loss 对比。

沿用既定超参数；不存在约定时根据项目基线选择并记录。首轮不搜索 LR。
clip 默认沿用 1.0，不私自加入改变目标的 KL 截断。

### 10.6 完成首轮训练与评测，满足条件再延长

- [ ] C1/C2 各运行 50 optimizer updates，保存 10/25/50；记录两流 loss、数量、KL 分布、
  梯度范数、clipping 比例、entropy、非有限值、耗时和峰值显存。
- [ ] 从 `video_fresh30` 预先固定 8 条作为过程监控子集，选择规则不得依据模型得分。
  评测 C0 及 C1/C2 的 10/25/50，形成逐视频配对结果。
- [ ] 评测统一 fp32、b63k8、跑到 EOS、`--ar`。核实 30 条输入仅 29 条进入 τ 聚合的
  原因，跨臂一致处理并报告排除情况。

停止及延长规则：

1. 非有限值、target 梯度、prefix 错误或 AR token mismatch：立即停止受影响实验，定位
   修复，明确作废受影响结果。
2. 小子集上 C2 相对 C0 连续两个节点 mean τ 下降超过 0.05：暂停延长，在完整 validation
   上复核最新节点后决定收口。不得因单个小子集节点轻微波动宣称显著收益或方法失败。
3. 完成 50 updates 后，在完整 `video_fresh30` 比较 C0/C1/C2。C0 可复用完全一致配置的
   已有结果；C1/C2 固定使用 update50，不根据验证成绩任意选择最好 checkpoint。
4. 输出逐视频配对 Δτ、均值、中位数、正向视频比例和配对 bootstrap 95% CI；bootstrap
   单位为视频，不能把 token 当独立样本。主要比较为 C2−C0 和 C2−C1。
5. 同时报无损门禁和计时，不能仅凭 τ 宣称端到端加速。
6. 若 update50 的 C2 同时高于 C0/C1，且无正确性或数值异常，自动将 C1/C2 延长至
   update100，并以完整 validation 复核。这是探索性延长，不要求 50 步时已统计显著。
7. 否则不自动扩展长程训练或多种子，完成原因分析并收口。

**本轮上限：1 个正式训练 seed，每条训练臂最多 100 optimizer updates；不做超参数搜索，
不自动展开 3 seeds。** 修复错误后的重跑须记录原因及作废结果。

### 10.7 新测试集纪律

- [ ] 盘点本地已有新视频来源、检查 train/validation 重复，形成新测试集候选与冻结方案。
- [ ] 若有足够且来源明确的新视频，可生成独立候选 manifest、哈希及抽样规则；本轮不在
  候选测试集上运行模型评测。

不未经授权大规模下载或付费获取数据。若没有合适来源，明确记录依赖，不阻塞 B1/Phase C
的 validation 实验。当前 60 条 heldout 均不可称为 untouched test。

### 10.8 运行与资源约束

- 检查 GPU 占用与项目自身进程，只用满足内存需求的空闲 GPU；不终止他人进程或抢显存。
- 默认一次一个重型训练/评测任务，避免并发影响计时及 OOM。
- 采用可恢复的后台方式，保存日志、PID、配置和输出路径，并持续检查直到本轮收口。
- 空闲 GPU 可用时持续推进；不可用时适当间隔检查，不忙轮询；长期资源阻塞如实报告。
- 不删除历史 checkpoint/结果腾空间；不足时记录具体需求。
- 优先使用现有环境，不安装或升级共享环境核心依赖。
- 中断/作废运行不得混入正式汇总。

### 10.9 交付与结论边界

- [ ] B0.5 校准脚本、配置、结果。
- [ ] 独立 B1 trainer/full-KL 实现及必要测试结果。
- [ ] 精度选择、真实 backward 显存和速度报告。
- [ ] Phase C 三臂配置、日志、checkpoint、逐样本评测及汇总。
- [ ] 机制、数值限制和实验结果报告。
- [ ] 更新本计划的完成状态、复现命令、产物路径、下一步及具体阻塞证据。

文档与结论统一遵守：

- `+0.159` 相对 `+0.071` 约为 2.2 倍，不是一个数量级。
- A4 支持 aligned SFT 泛化宽峰，不支持严格 plateau 表述。
- 单视频旧 head 的 rejected KL 大不代表新起点总体性质，也不直接代表梯度贡献。
- validation 结果不得称为 untouched test。
- OPD 若不如 C0 但优于 C1，只能说缓解继续训练退化。
- 收益 CI 跨零时报告趋势及不确定性。
- 负结果限定于所测模型、数据、预算和实现，不泛化为整个视频领域 OPD 无效。

**请从 §10.1 开始，按顺序自主推进到本轮收口，阶段间无需等待确认。只有缺少必要数据/
硬件、涉及越权外部操作，或穷尽合理修复仍无法继续时，才报告具体阻塞。**

---

## 11. §10 执行结果回填（2026-10-10）

本轮已按 §10.1–§10.9 自主推进到收口。逐项状态：

| 小节 | 状态 | 证据 |
|---|---|---|
| 10.1 状态核实 | 完成 | AGENTS.md 已更新；A4 报告与 B0 产物已复核 |
| 10.2 B0.5 校准 | 完成 | `docs/reports/2026-10-09-b1-precision-and-phasec-config.md` §1.2；`results/phaseB05/b05_calibration.json` |
| 10.3 公式核实 | 完成 | `docs/reports/2026-10-09-b1-formula.md` |
| 10.3 B1 实现 | 完成 | `repro/opd_fullkl.py`（16/16 测试）、`repro/opd_train_video_fullkl.py`（17/17 测试） |
| 10.4 精度与可行性 | 完成 | fp32 实测 OOM（含 1-anchor 对照）→ bf16 替代实测峰值 26.42 GiB |
| 10.5 配置冻结 | 完成 | `docs/reports/2026-10-09-b1-precision-and-phasec-config.md` §2 |
| 10.6 首轮训练与评测 | 完成，触发停止规则 2 | `docs/reports/2026-10-10-phase-c-seed0.md` |
| 10.7 新测试集 | 完成盘点（未跑评测） | `results/phaseC/testset_candidates/INVENTORY.md` + `hashes.json` |
| 10.8 资源约束 | 遵守 | 全程串行、只用空闲 GPU、未删历史产物、未改共享依赖 |
| 10.9 交付 | 完成 | 见下 |

### 11.1 Phase C seed-0 主结果（n=29，三臂各 30/30 无损）

| arm | mean τ | median τ |
|---|---:|---:|
| C0 不训练（aligned step1000） | **3.7377** | 3.7778 |
| C1 aligned continue-SFT 50 updates | 3.6579 | 3.6471 |
| C2 paper-faithful OPD 50 updates | 3.5095 | 3.5135 |

- C2 − C0 = **−0.2281**，95% CI [−0.2830, −0.1738]，正向 1/29
- C2 − C1 = **−0.1483**，95% CI [−0.2037, −0.0953]，正向 5/29
- C1 − C0 = **−0.0798**，95% CI [−0.1269, −0.0311]，正向 5/29

**触发 §10.6 规则 2（监控子集 C2−C0 连续两节点低于 −0.05：−0.1742 / −0.1530）→ 暂停延长，
完整 validation 复核后收口。不扩 100 updates，不展开 3 seeds，不搜索超参数。**

### 11.2 本轮新增的固定语义与限制（后续引用必须带上）

1. **C2 的 teacher 是 bf16，是精度近似**：fp32 单卡不可行（实测 OOM）。逐 anchor loss
   相对差可达 9.5%，且 **bf16 rollout 会重选 anchors（8/8 视频不同）**，所以不得把
   C2−C1 全部归因于 KL 语义。
2. **两臂的「anchor」定义不可比**：C1 = 1200 个被监督位置，C2 = 18000 个被监督 offset
   （accepted 2043 + rejected 15957）。同一个「24 anchors/update」不是同一份预算。
3. **50 updates 的预算本身在退化**：C1 相对 C0 已显著为负，本实验比较的是「谁退化得少」，
   **没有测试论文 8-epoch 量级的预算**。
4. **计时不可用**：三臂评测跨越约 2 小时、落在不同 GPU 争用条件下，τ 与 ms/tok 方向不一致，
   只报无损门禁，不宣称端到端加速。
5. **全部结论是 validation 口径**：`video_fresh30` 已用于 A4 选点，60 条 heldout 均非
   untouched test。

### 11.3 §10.7 新测试集盘点结论

- `pool_499` **就是** `lmms-lab/VideoDetailCaption`（499/500 文件名吻合），**本机没有新的
  同域视频**（仅剩 1 个未用文件）。
- 两个全新**异域**来源：Video-MME（30 条，其中 20 条未被任何既往运行使用）与 MLVU（396 条）。
- 候选已冻结并带哈希（`results/phaseC/testset_candidates/hashes.json`）：
  A `vmme20_fresh`（n=20，id_list sha256 `367bb701…`）、
  B `mlvu47_untouched`（n=47，`63ddf30e…`）、C `combined67`（n=67，`da975d0a…`）。
- 选择规则完全基于「文件系统/标注集合差」，与任何模型得分无关；**本轮未在候选集上跑评测**。
- MLVU 的既往使用需注意：GLANCE_v2 训练并留出了 349 条 MLVU（本项目的 Phase C 选点
  不含 GLANCE_v2 head，所以不影响本轮；候选 B 已额外排除该 349 条，两种情况下都安全）。
