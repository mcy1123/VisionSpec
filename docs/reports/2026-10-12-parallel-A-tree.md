# 任务 A：可部署树选择与接受路径失配（2026-10-12 并行实验）

**结论标签：`未发现净局部改善`（negative）。** 在冻结 C0、原始 GPU top-8 候选与 b63k8 预算下，
唯一预注册的固定深度配额树 A1 在 C6 的 6 视频/48 anchor 上**没有任何一条 anchor 改善**正确
AR 前缀覆盖（0 更好 / 32 相同 / 16 更差，ΣΔL_tree = **−20**，均值 **−0.4167**，视频级
bootstrap 95% CI **[−0.4792, −0.3333]**）。三条预注册门禁全部未通过，因此**按停止规则收口，
不进入 Stage 2/3，不消耗 GPU，不调配额**。本报告同时给出一个可复核的机理解释：现有 best-first
已经是它所优化的 draft 概率代理 `Φ` 的最优前缀闭合解，而该代理与"正确 AR 前缀能被覆盖"
（真实接受路径深度）**系统性失配**。

本任务只做 CPU 分析与测试，**未训练、未更新任何参数、未改共享代码、未打开新测试集、
未使用 C5 B-reference 的 8 轮样本**。

- 方向目录：`repro/experiments/tree_c7_a/`
- 结果目录：`/mnt/local/mcy/GLANCE/results/parallel_A_tree/20261012T233048/`
  （一次独立复现运行：`.../20261012T233539/`，逐字段一致）
- 预注册：`prereg.json`（**在计算任何 A1 结果之前写入**，SHA256 见下）

## 1. 审计：真实调用链与语义

| 项目 | 核实结果（读代码，不依赖任务书简写） |
|---|---|
| 构树函数 | `GLANCE_repo/glance/tree.py::build_budget_tree`（SHA256 `5e02038d…ab63`） |
| 生产调用点 | `GLANCE_repo/glance/decode.py::generate` 第 258–260 行（SHA256 `0bcbdba6…c2f5`） |
| 候选来源 | block head 一次前向给出各 offset 边缘分布，`torch.topk(F.log_softmax(draft_logits.float()), 8)`，ids/logp 移到 CPU 构树 |
| budget 语义 | 63 **只计树节点，不含 root**；root 作为 packed 位置 0 单独进入 verify，故 verify pack 为 64 个位置 |
| depth 语义 | depth = block offset，取值 1..15（block size 16）；depth 0 是已提交 root，不建节点 |
| 路径分数 | root→node 的**各单位置边缘 log-prob 之和**；不同 depth 的节点从不合并分数 |
| best-first | 以负累计分数建最大堆，depth-1 的 K 个候选入堆；弹出→追加节点→把它的 K 个子节点以 depth+1 入堆。同分按堆元组的 (depth, parent, rank, 插入序) 决定，即插入序稳定 |
| 前缀闭合 | 子节点只会从已追加节点推入，因此 packed 顺序天然拓扑有序（parent index < child index） |
| verifier/commit | 一次祖先掩码 target 前向；commit 沿 target 自身贪心 token 走树，遇第一个缺失子节点即停止 |
| commit gap | 正确实现的贪心 commit 必然 `L_actual = L_tree`，非零即 verifier/commit 正确性失败 |
| root/bonus | root 是已提交 token；bonus 是接受前缀之后 target 的额外 token，不计入 L_raw/L_tree/L_actual |

**A0 是否已是其代理目标下的最优分配：是。** 因为 `log p ≤ 0`，沿路径延长永不提高路径概率，
所以按累计分数 best-first 弹出的顺序，正是最大化 `Φ(T)=Σ_{node∈T} exp(累计 log p)`
的前缀闭合集合的贪心顺序。因此本报告**不声称 A1 更好地求解了同一目标**；A1 检验的是
"把预算按深度配额重排，能不能换来更多**正确 AR 路径覆盖**"。

**A0 重放等价性（写预注册之前就已核实）**：用未改动的生产 builder 在 C6 持久化的 GPU top-8
ids/logp 上重建，48/48 anchor 与落盘的 63 节点树**逐字段相同**。Stage 0 把这作为正式门禁。

**样本区分**：本任务只用 C6 的 6 视频/48 轮（`trace_current_ar.jsonl` 中 `arm == "C0"`）。
C5 B-reference 的前 8 轮是不同 anchor 集合，**未使用、未拼池**。

## 2. 冻结的唯一实验策略

| | 构树方式 | 不变因素 |
|---|---|---|
| **A0** | 生产 best-first，累计 path log-prob 排序，前缀闭合预算 63 | C0、原始 top-8 候选/分数、root/prefix、深度上限 15、预算 |
| **A1** | 固定深度配额：depth 1–3 各 5 节点、depth 4–15 各 4 节点（3×5+12×4=63）；由浅入深单趟填充；某深度只能扩展上一层已保留父节点；同层按**与 A0 相同的累计分数**排序，同分依次按 (父节点在上一层保留列表中的槽位, 该 offset 的 rank) 决定；配额不足时不从更深层回填 | 同 A0；不使用 target logits/正确 token/oracle 路径/样本收益，不新增候选、不追加 head 调用 |

`Φ(T) = Σ_{node∈T} exp(Σ_path log p_d)`，只用持久化的每 offset GPU log-prob 计算。

**未搜索第二套配额，未事后改阈值。** 预注册（`prereg.json`）在计算任何 A1 数量之前冻结，
内容包含：输入/样本/checkpoint 哈希、深度与预算定义、A0/A1 构造与 tie-break、`L_raw/L_tree/
L_actual/Φ/first_missing_depth/recovered/lost` 定义、三项门禁的精确数值规则、bootstrap 口径、
停止规则。承认这是在已反复查看的 validation（6 视频）上开发，**不宣称独立验证**。

## 3. Stage 0（CPU 重放）门禁：10/10 通过

只重放 C6 保存的 GPU 候选，不重新做 CPU log_softmax、不新增 head 调用。

| 检查 | 结果 |
|---|---|
| A0 重建 == 落盘 63 节点树（48/48，token/depth/parent/rank） | 通过 |
| 独立走树得到的 `L_tree` == 记录 `oracle.L_tree_oracle` | 通过 |
| 独立重算 `L_raw`（同 cap/EOS 规则）== 记录 `oracle.L_raw` | 通过 |
| 用记录 `verification_argmax` 独立贪心 walk 得到 `L_actual` == 记录 | 通过 |
| 树形状：parent < child、depth = parent.depth+1、节点取自本 offset 候选行、长度 63 | 通过 |
| Σ(L_raw − L_tree) == **40** | 通过 |
| mean L_tree == **2.8958333333333335** | 通过 |
| Σ commit_gap == **0**（逐 anchor 为 0） | 通过 |
| anchor 数 48、48 棵树均为 63 节点 | 通过 |

失败清单文件 `stage0_failures.json` 全为空。**Stage 0 通过，A0/A1 比较有效。**

## 4. Stage 1（CPU 反事实）：净损失，三门禁全不过

### 4.1 总览

| 指标 | 值 |
|---|---:|
| ΣΔL_tree（A1−A0，48 anchor） | **−20** |
| mean ΔL_tree | **−0.4167** |
| 视频级 bootstrap 95% CI（seed 20261012，2000 次） | **[−0.4792, −0.3333]** |
| 逐 anchor 分布 | 0 更好 / 32 相同 / **16 更差**（−1 有 12 条，−2 有 4 条） |
| ΣL_tree A0 → A1 | **139 → 119** |
| ΣΦ_A0 → ΣΦ_A1 | **161.4431 → 139.9173**（A0 ≥ A1，符合 best-first 最优性） |
| A1 实际节点数 | 3024 = 48×63，无 anchor 短于预算 |

### 4.2 按视频（聚合单位=视频，8 anchor 嵌套）

| 视频 | mean L_tree A0 | mean L_tree A1 | mean ΔL_tree | 正/负/变化 anchor | 早期(depth≤4)净损失 | 阈值 | 严重早期损失 |
|---|---:|---:|---:|---|---:|---:|---|
| v_-D1gdv_gQyw | 3.750 | 3.250 | −0.500 | 0/2/2 | 4 | 1.75 | 是 |
| v_1kwIYXB4b8s | 2.875 | 2.500 | −0.375 | 0/3/3 | 3 | 1.75 | 是 |
| v_0X0u_3xUBPE | 3.000 | 2.750 | −0.250 | 0/2/2 | 1 | 2.00 | 否 |
| v_1KONAdZGdWU | 2.000 | 1.625 | −0.375 | 0/3/3 | 2 | 1.75 | 是 |
| v_0DFz3sgfda0 | 2.375 | 1.875 | −0.500 | 0/3/3 | 4 | 1.75 | 是 |
| v_0x6ZV9Y8g_c | 3.375 | 2.875 | −0.500 | 0/3/3 | 4 | 2.00 | 是 |

6 条视频的 mean ΔL_tree **全部为负**；`recovered_paths` 全库合计 **0**，`lost_paths` 合计 **19**。

### 4.3 三项预注册门禁

| 门禁 | 规则 | 实测 | 通过 |
|---|---|---|---|
| 1 净收益 | ΣΔL_tree > 0 | −20 | 否 |
| 2 非单视频推动 | ≥2 条视频 mean ΔL_tree > 0 | 0 条 | 否 |
| 3 早期覆盖 | 无视频在 depth≤4 的净损失 > `2×(有效anchor数)/8` | 5/6 视频超阈值 | 否 |

**`all_pass = false` → 按预注册停止规则与任务书 §2/§3，不进入 Stage 2，不做 GPU 局部验证，
不做在线 A0/A1 解码，不调配额。** 这不是"所有树选择方法都无效"的结论，只否定**这一套固定
配额方案**。

### 4.4 逐 anchor 表摘录（最差 5 条）

| 视频 | round | L_raw | L_tree A0 | L_tree A1 | Δ | first_missing_A1 | lost paths |
|---|---:|---:|---:|---:|---:|---:|---|
| v_-D1gdv_gQyw | 34 | 8 | 4 | 2 | −2 | 3 | 3;4 |
| v_-D1gdv_gQyw | 68 | 3 | 3 | 1 | −2 | 2 | 2;3 |
| v_0DFz3sgfda0 | 48 | 6 | 4 | 2 | −2 | 3 | 3;4 |
| v_0x6ZV9Y8g_c | 9 | 3 | 2 | 0 | −2 | 1 | 1;2 |
| v_1kwIYXB4b8s | 7 | 6 | 4 | 3 | −1 | 4 | 4 |

注意 A1 的 `first_missing_depth` 全部落在 **depth 1–4**，即固定配额给得"最多"的区域：
损失不是因为 A1 不够深，而是因为 A1 在这些深度上换了节点。

## 5. 机理解释（post-hoc，非预注册门禁）

产物：`mechanism_analysis.json`。全部只用同一批持久化候选，属顺序/预算性质。

1. **预算去向（48 anchor 合计）**
   - A0 节点深度直方图：depth1=176、2=495、3=716、4=715、5=523、6=245、7=104、8=44、9=6 —— 重心在 depth 3–5，最长 9。
   - A1 节点深度直方图：depth1–3 各 240，depth4–15 各 192 —— 平坦配额，末段强制铺到 depth 15。
2. **覆盖去向（正确 token 落在哪一深度）**：A0 covered 深度直方图 1:43、2:40、3:29、4:16、5:7、6:4；
   A1 为 1:42、2:36、3:22、4:10、5:6、6:3。正确前缀主要活在 depth≤4，且 A0 在每个深度都更多或持平。
3. **比较器镜子自检**：用脚本内实现的 best-first 顺序对 A0 自己的节点集合重排，**48/48 还原未改动的
   生产树**，说明后续重排不是手写比较器的伪影。
4. **并集重排**：把 A0 与 A1 的节点并集按同一生产累计分数比较器重排取 top-63，
   **48/48 恰好等于 A0**，且平均有 **0.0** 个 A1 新节点进入该 top-63。也就是说：A0 已经把这个候选池里
   分数最高的前缀闭合集合选走了，A1 提出的节点在分数意义上都更差。固定配额做的事是
   **用分数更高的浅层节点换更深的低分节点**。
5. **覆盖前缀性**：A1 覆盖的正确 token 序列在 **48/48** anchor 上是 A0 的前缀（A1 从不覆盖 A0 未覆盖的位置）。
6. **代理与真实目标的失配（本轮核心机制读数）**：`Φ_A0 ≥ Φ_A1`（161.44 vs 139.92）而
   `ΣL_tree A0 > ΣL_tree A1`（139 vs 119）。即 **A0 在概率代理上严格更优，在"正确路径能否被树覆盖"
   上却仍留下 40 个 token 的 tree_gap**；而把预算改去加深、牺牲代理分数，只会让真实覆盖更差。
   在已测数据上，第 40 个 gap token **不可能靠"更深的配额"回收**：18 条有 tree_gap 的 anchor 平均
   ΔL_tree = **−0.778**，反而比 30 条无 gap anchor（−0.200）损失更大。

这不构成"draft 概率目标错误"的一般性证明，也不排除**改变候选集合**（而非重排已有候选）或
**用部署可得特征做代价感知分配**的空间；它只说明：在冻结候选与 63 节点预算下，按深度配额重排
不可能改善正确路径覆盖，因为该预算已经被概率代理的最优解占满。

## 6. 复现

```bash
cd /home/mcy/Projects/GLANCE
PY=/home/mcy/.conda/envs/fin_research/bin/python
# 一键复现（新时间戳目录；CPU only，不需要 GPU）
bash repro/experiments/tree_c7_a/run.sh
# 或分步
"$PY" -m unittest -v repro.experiments.tree_c7_a.test_fixed_quota_tree
"$PY" -u repro/experiments/tree_c7_a/run_stage01.py   --out <RUN>
"$PY" -u repro/experiments/tree_c7_a/analyse_mechanism.py --out <RUN>
```

测试：**24/24 通过**（配额表、硬上限、前缀闭合、拓扑序、候选来源、无重复、
tie-break 确定性、小规模穷举对照、零配额、父节点不足不回填、EOS/末端 cap、Φ 与覆盖、以及
在冻结真实候选上的 A0 精确重建与 L_tree 一致）。日志 `tests.log`。

输入哈希：trace `0ffabf0352f502305f88dff3ba1d913e54ee8463f50ba6fd082f9617bc20340c`；
C0 `e2526a249a8ef7e1db3a9990589b727541486093372011e1a0ff6a16e06d23d2`；
prereg `5e1a7c180e6337d35af8cf29c86bda172218699776e8e455dfced27faaec9d44`。
任务 A 代码哈希：`fixed_quota_tree.py` `f402997b…3bd07`、`run_stage01.py` `7cd57da3…1ada2`、
`analyse_mechanism.py` `8e9e81d2…3a4874`、`test_fixed_quota_tree.py` `b3b8d4f4…dd805c`。
两次独立运行（`20261012T233048`、`20261012T233539`）的 stage01 与 mechanism 输出**逐字段一致**。

## 7. 交付物

`/mnt/local/mcy/GLANCE/results/parallel_A_tree/20261012T233048/`：

| 文件 | 内容 |
|---|---|
| `prereg.json` | 冻结的预注册（参数、口径、门禁、停止规则） |
| `stage01_summary.json` | Stage 0 门禁 + Stage 1 汇总 + `per_video` + `gates` |
| `stage0_failures.json` | Stage 0 失败清单（全空） |
| `stage1_per_anchor.csv` | 48 行逐 anchor：L_raw/L_tree_A0/L_tree_A1/Δ/Φ/首漏深度/回收·损失路径/深度计数/实际节点 |
| `stage1_per_video.csv` | 6 行逐视频聚合（含早期阈值与严重损失标记） |
| `mechanism_analysis.json` | §5 的顺序/预算机理读数 |
| `a1_trees.jsonl` | 每 anchor 的 A1 节点表与逐深度计数 |
| `tests.log` / `stage01.log` / `mechanism_analysis.log` | 原始日志 |
| `source_*.py` | 运行时的源码快照 |
| `run_manifest.json` | 交付索引与状态字段 |

## 8. 观察到了什么 / 未观察到什么 / 无法判断什么

- **观察到**：A0 在冻结候选与 63 节点预算下已是概率代理 `Φ` 的最优前缀闭合解；唯一预注册的
  固定深度配额树在 48 anchor、6 视频上一致变差；损失集中在 depth 1–4 的节点替换，且
  18 条本就有 tree_gap 的 anchor 损失更大。
- **未观察到**：任何 anchor 的覆盖改善；任何视频的正向均值；任何"更深的树能回收 gap"的证据。
- **无法判断**：这套负结果不能外推到 (a) 改变候选集合/新增候选、(b) 使用部署时可得特征的
  代价感知或动态配额、(c) 其它预算或 block size、(d) 其它模型/数据。也不能说"树方法已无空间"。
- 本探针的 `L_tree` 是**离线局部覆盖**，不等于在线 τ 或速度；本轮**未产生任何 GPU/在线数字**，
  因此不报告加速或端到端结论。

## 9. 下一轮方法研究是否值得开展及为什么

**不值得继续投入"固定深度配额"这一支。** 三条独立理由：预注册三项门禁全不过；机理上该预算
已被 `Φ` 最优解占满，重排只能牺牲代理分数；且损失集中在正确前缀真正所在的 depth 1–4。

**值得（若继续）走的方向**：既然 24/42 的阻断是"正确 token 不在原始 top-8"而 18/42 是"树遗漏"，
而树遗漏在本探针下无法用**重排**回收，下一步应先做**候选缺失**方向的研究准备（任务 B 的时序条件
响应正是其入口），或在新的预注册里检验**部署时可得的候选/位置特征**能否预测哪些 anchor 需要
更深而非更宽的预算——但这需要新的候选或新的预算语义，属新实验，**不在本任务内启动，也不在
本结果之后自动恢复 OPD 或另开超参搜索**。

## 10. 可直接粘贴到 AGENTS.md 的状态更新

> **任务 A（并行，2026-10-12）：固定深度配额树选择 = 负结果，未使用 GPU。**
> 目录 `repro/experiments/tree_c7_a/`、结果 `results/parallel_A_tree/20261012T233048/`、
> 报告 `docs/reports/2026-10-12-parallel-A-tree.md`。冻结 C0/top-8/b63k8，只重放 C6 保存的 GPU 候选：
> Stage 0 十项门禁全过（A0 重建 48/48、Σtree_gap=40、mean L_tree=2.895833、commit_gap=0）。
> 唯一预注册的 A1（depth1–3 各 5、depth4–15 各 4）在 6 视频/48 anchor 上
> ΣΔL_tree=**−20**、mean **−0.4167**、CI[−0.4792,−0.3333]，0 更好/32 相同/**16 更差**，
> 三条门禁全不过 → 按停止规则不进 Stage 2/3，不调配额。机理：`Φ_A0≥Φ_A1` 而 L_tree A0>A1，
> 并集按同一比较器重排 top-63 仍**恰好等于 A0**（48/48），说明该预算已被概率代理最优解占满，
> 正确前缀活在 depth≤4 而 A1 在那里换掉了节点。**结论：不支持继续"固定深度配额"支线；
> 也未证明所有树选择无空间**（改变候选/代价感知分配仍未检验）。C7 该方案收口，
> 下一优先仍是候选缺失/时序方向（任务 B）。测试 24/24，未训练、未改共享代码、未动 C1/C2。
