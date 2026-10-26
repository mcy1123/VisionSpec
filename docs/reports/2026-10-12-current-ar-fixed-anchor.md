# 当前 AR 固定 anchor 诊断闭环（2026-10-12）

**已完成当前 reference 门禁、B-current 三臂评分及真实局部 trace/oracle。未训练、未更新参数，
未运行 D4、完整 chain 评测或冻结新测试集。** 本报告是阶段实验记录；项目状态仍只维护在 AGENTS.md。

主产物：`/mnt/local/mcy/GLANCE/results/phaseC6/current_ar_closed_loop_20261012T221104_gpu_scores/`。
`delivery_manifest.json` 链接配置/哈希、门禁、调用量和原始文件。初始失败运行保留在不带
`_gpu_scores` 后缀的同名目录，**未混入正式指标**。

## 1. 冻结样本、模型及模型门禁

从已有非 warmup validation 29条按 `(prompt_len, video_id)` 排序，分3个等量长度层，
`Random(20261012)` 每层抽2条，再交错层顺序。抽样只使用输入长度和ID，不查看收益。
每条在原始 C0 b63k8 的已存轮次序列上取8个整数等距索引，覆盖起始、中段与末轮。
前两条先过模型门禁，之后才执行其余四条：

| 视频 | prompt长度 | C0轮次索引 | 有有效draft offset的anchors |
|---|---:|---|---:|
| v_-D1gdv_gQyw | 766 | 0,11,22,34,45,57,68,80 | 7 |
| v_1kwIYXB4b8s | 766 | 0,7,14,21,28,35,42,49 | 7 |
| v_0X0u_3xUBPE | 958 | 0,10,21,31,42,52,63,74 | 8 |
| v_1KONAdZGdWU | 766 | 0,9,18,27,37,46,55,65 | 7 |
| v_0DFz3sgfda0 | 958 | 0,16,32,48,65,81,97,114 | 7 |
| v_0x6ZV9Y8g_c | 958 | 0,9,19,28,38,47,57,67 | 8 |

共48个真实轮次，其中44个有有效offset，另4个为无待预测token的末轮边界，不补造替代anchor。
三臂各635个有效offset；执行144个唯一 anchor×head 评分，其中132个有有效预测位置、
12个仅保留边界记录。下列局部长度平均保留全部48轮（末轮有效长度为0）；
token指标仅使用有效位置，各视频先聚合、再平均。CSV同时列出 n_anchor、n_predictive_anchor
与 n_valid_offsets，不能把48轮都称为48个有效预测样本。

模型身份从 phaseC5 manifest 和 phaseC4 正式结果追溯并核对 SHA-256：
C0=`head_video439_aligned/step1000`；C1=`phaseC4_C1/step10`；C2=`phaseC4_C2/step10`。
使用现有 `fin_research` Python，单独使用空闲GPU0，fp32 target/fp32 head、SDPA、cached
context、greedy temperature=0、block16/b63/k8/max_new1024。checkpoint前后哈希一致。

| 模型级检查 | 结果 |
|---|---|
| processor输入ID、像素与上一轮精确输入哈希 | 6/6一致 |
| 当前独立 `greedy_generate` 与已保存部署AR | 6/6逐token一致 |
| 实际生产 `generate` 的 C0输出、选中anchor位置 | 6/6匹配AR及原轮次位置 |
| 两条pilot固定输入重复head前向 | logits max差0，top32 IDs一致 |
| pilot增量head缓存 vs 同一past-context整段重算 | max差1.19e-5 / 1.29e-5，top32 IDs一致 |
| pilot重复树验证、贪心路径/bonus | 一致；logits max差0；与实际C0捕获的验证argmax一致 |
| 144个局部树正确路径判断 vs 独立当前AR | token全部一致；对应full-logit最大差1.1444e-4 |
| 144份落盘trace独立复算 | 树、mask祖先闭合、路径/bonus、裁剪、oracle全部通过 |

没有把logit容差替代token门禁；不要求不同计算形状的中间张量逐位相等。
检查只针对正确祖先路径，不能要求错误分支上的teacher判断与正确AR相同。

生产解码文件未修改：使用可移除hook捕获 C0真实候选、verification前的target KV、真实
head context片段。每个anchor的context严格截止 `pos-1`；root为已知 `pos` token；
noise block为 root+15个MASK。其他head使用同一真实prefix/context/root/mask，
没有传入未来hidden states或答案。正确未来token仅供reference及离线评分。

## 2. A-old/A-relabel：同输入重分组，没有重训

单独复用1680个旧anchors的旧prefix、冻结C0 proposal、相同teacher argmax及已有双向KL。
A-old按旧n_acc分组；A-relabel按该teacher对冻结proposal的leading greedy agreement分组，
三臂共用重标结果。仍有58个anchor的分组边界改变；**不是实际训练过程标签错误率**。

轨迹来源（旧保存/当前部署）与teacher条件（draft-induced/正确AR）是两个轴。
旧轨迹可作为teacher-forced/off-policy输入，不因与当前AR不同就失去所有训练意义。
首拒后draft-induced与正确AR teacher不同是预期条件差异，不是本轮发现的新bug。

validation n=29，以下均为 C2−C0、同一旧输入：

| 指标 | A-old | A-relabel |
|---|---:|---:|
| 两流full-KL replay Δloss | −0.129751 | −0.130496 |
| accepted forward-KL Δ | −0.010069 | −0.010020 |
| 首拒 reverse-KL Δ | −0.305124 | −0.308496 |
| 首拒严格rank Δ | −0.104885 | −0.084770 |
| 首拒top1 Δ | +3.736 pp | +4.023 pp |
| 首拒rank≤8代理 Δ | −0.144 pp | −0.144 pp |

曝光并集20条、未曝光20条的loss差异方向也不变；未曝光组首拒rank≤8代理
从−0.625 pp到−0.833 pp。**本次重标未改变原来的局部判断**：旧参考loss改善存在，
但不能据此证明validation有效候选边界改善。不能外推为已排除训练中所有标签影响。

KL方向、原始offset权重及两流归一已有数据可重算；实际top8 IDs、概率、entropy等旧产物
未保存的量仍N/A，`rank≤8`不能改称实际成员。详见 `A_relabel/per_video_A.csv` 和
`A_relabel/summary_A.json`。本轮不把A与B强配成“同上下文只改reference”的对照。

## 3. B-current：当前正确AR上的候选与实际树覆盖

主比较C2−C0；辅助C1−C0、C2−C1。三臂均条件在同一批真实C0状态；
其他head的局部verification不代表其各自在线轨迹。full-KL/JS使用独立增量AR的全词表
logits与draft logits作fp32归约，**不是C2原生draft-induced训练loss**。

实际候选在GPU按生产相同调用计算并保存 top8、top32 IDs及log-probability；
top8使用独立 `topk(8)`，不是从 `topk(32)` 截取。每位置记录严格rank、实际成员、
cutoff ties、target entropy/margin、正确token在draft的log-probability。
三种标签分开：A旧/重标、B当前C0 top1 leading agreement、B实际C0-tree覆盖/下一未覆盖位置。
主分组跨臂固定为C0-tree，正常终止不计rejection。

| 模型 | mean L_raw=L_candidate | mean L_tree | mean L_actual | mean tree_gap | commit_gap | 全有效offset top8覆盖² |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 3.729167 | 2.895833 | 2.895833 | 0.833333 | 0 | 55.739% |
| C1 | 3.604167 | 2.833333 | 2.833333 | 0.770833 | 0 | 55.570% |
| C2 | 3.645833 | 2.812500 | 2.812500 | 0.833333 | 0 | 55.427% |

² 先算每视频有效offset比例，再平均6视频；不是将所有token视作独立样本。
L排除root/bonus，并按EOS、输出上限、block深度裁剪；不是历史 accepted+1 的τ。

| 视频级配对 | mean Δ | bootstrap 95% CI | 正/负/不变视频 |
|---|---:|---|---|
| C2−C0 L_raw | −0.083333 | [−0.229167,+0.104687] | 1 / 4 / 1 |
| C2−C0 L_actual | −0.083333 | [−0.208333,0] | 0 / 2 / 4 |
| C1−C0 L_actual | −0.062500 | [−0.145833,0] | 0 / 2 / 4 |
| C2−C1 L_actual | −0.020833 | [−0.145833,+0.104167] | 1 / 2 / 3 |
| C2−C0 所有offset实际top8 | −0.312 pp | [−1.264,+0.641] pp | 1 / 2 / 3 |
| C2−C0 C0下一未覆盖位置top8 | −2.381 pp | [−9.524,+4.762] pp | 1 / 2 / 3 |

bootstrap单位=视频，seed20261012、2000次；n=6只是探针，不能把CI端点0解释为等效，
也不能把嵌套的48轮或635个offset当独立样本。C2−C0 的B forward-KL为
+0.043347（CI[−0.002109,+0.102437]）；这是另一批context/teacher条件，不是A-loss的反转消融。

**本探针未显示C2改善关键候选边界或实际树覆盖。** 原始连续覆盖仅1/6视频改善；
实际局部提交没有正向视频。不是对所有数据/训练配置作“OPD永远无效”的结论。

Top8集合并非完全静止：C2有442/635个offset的集合与C0相同，C1为352/635。
但“集合不同”不等于正确token越过边界。在全部有效offset集合都相同的anchor中，
没有观察到 L_tree 改变；这些数含末轮空集合的平凡相同，原始记录单独列 H_valid。
不能推广为候选分数/排序一般不影响树。`candidate_set_changes.csv` 保留逐anchor证据。

## 4. 首个阻断与局部空间

当前独立offset结构最多15个draft tokens，预算63不含root，无额外条件分支限制。
正确单链至多15节点，故 `L_candidate=min(L_raw,63)=L_raw`，不另造多余oracle。

| 分类 | C0 | C1 | C2 |
|---|---:|---:|---:|
| 下一正确token不在实际top8 | 24 | 24 | 23 |
| 正确token在top8，但完整正确路径未进入树 | 18 | 18 | 19 |
| 验证/选择/提交差距 | 0 | 0 | 0 |
| 正常终止/边界 | 6 | 6 | 6 |

C0在42个非终止阻断位置中，预测缺失24次（57.1%），树选择18次（42.9%）。
**两者都实测存在，提交差距本探针未显示。** 不能依据 `15-L_raw` 比gap大就宣布
预测是主要可训练瓶颈，也不能据原始缺失略多就忽略明显树遗漏。

C0的tree_gap合计40个局部draft tokens，18个anchor上为正、最大5；
跨全部48轮均值0.8333（仅有有效offset的44轮为40/44≈0.9091）。
这只是已知正确reference的局部空间，**不是可部署选择器已实现的收益，也不是全局Δτ**。

可重放实例（事后选作说明，不影响冻结样本与主统计）：

- `v_1KONAdZGdWU`, round37：L_raw=7、L_tree=L_actual=2；首个未覆盖在offset3，
  正确路径存在于原始候选组合，但预算选择的实际树遗漏；tree_gap=5。
- `v_-D1gdv_gQyw`, round34：8 / 4 / 4；tree_gap=4。

每条trace保存top8分数、完整nodes(parent/depth/rank)、packed IDs/positions、
prefix可见列数、完整packed祖先可见矩阵、各节点verification argmax、选中路径、
raw/effective draft tokens、bonus和终止原因。节点选择使用现有best-first构造，没有独立
后处理剪枝。`verify_current_ar_trace.py` 可完全重算局部oracle与提交，不需要模型。
完整词表logits只临时计算；重算full-KL仍需按脚本重新前向，不能从top32伪恢复full-KL。

## 5. 故障、额外调用与测试

首个追踪器版本把GPU logits移到CPU后重新log_softmax、重建“实际树”。第二条pilot的
round28触发packed节点不一致断言。只修独立诊断路径：直接捕获GPU实际候选分数与树，
并让其他head使用相同GPU候选计算。修复后原冻结6条名单不变，2条门禁重过，再完成全6条。
这是诊断器实现修正，不是新算法，也不是发现生产tree存在bug。失败产物/脚本/日志保留，
原28条部分记录作废，29次部分评分尝试另计，未拼接进正式144条。

| 调用类型 | 成功运行 | 失败pilot额外 |
|---|---:|---:|
| C0整轨迹head前向（含被复用的正式C0评分） | 455 | 131 |
| 固定context的C1/C2 head前向 | 96 | 16 |
| 重复/一致性gate额外head前向 | 4 | 4 |
| C0生产轨迹target前向（prefill+verify） | 461 | 133 |
| 独立增量AR target前向/teacher分布 | 1616 | 463 |
| 局部树target verification（含gate重复） | 100 | 20 |

成功运行总计555次head、2177次target前向；含失败额外成本后706/2793。
100次局部验证中的4次是gate重复，不重复加账；当前AR分布从独立AR前向安全复用，
不再单独teacher-force产生另一种reference。**144是唯一固定评分数，不是总调用数。**
详见 `all_call_accounting.json`。

成功运行主循环带trace耗时282.23秒（不含加载与初始文件哈希），峰值37.79GiB；
这是诊断成本，不是正式decode耗时或速度收益。新CPU测试4/4；另有144/144实际落盘trace
复算和两条模型pilot门禁。无参数梯度/optimizer更新，未改共享依赖、生产解码代码或他人进程。

## 6. 四个回答与唯一后续实验

1. **旧标签重分组改变原判断？本探针未显示。** A同输入可比较的loss、KL、rank/top1代理
   变化很小；不能计算旧实际top8/entropy，也不能推断训练过程错误率或把A/B作因果配对。
2. **当前AR上C2改善候选或实际树？本探针未显示。** n=6、44个有预测offset的真实C0
   anchors中，按全部48轮边界一致统计，ΔL_raw与ΔL_actual均−0.0833；不称全局Δτ。
3. **首先阻断路径的情形？支持原始缺失与树遗漏同时存在；提交缺陷本探针未显示。**
   C0非终止事件24/18/0，tree_gap合计40；局部oracle不保证可实现收益。
4. **下一步优先什么？支持先做一次固定head/候选/预算的树选择对照。**
   这是因为已有正确候选的路径遗漏在本探针占42.9%，不是因为已经证明某种新树更好。

**唯一建议（本轮未启动）**：冻结C0、raw top8及63节点验证预算，把现有全局累计路径概率
best-first树，与一个预先固定的按深度配额树作对照。例：depth1–3各保留5个节点、depth4–15
各4个节点，总63；每层只从已保留父节点扩展，按累计概率取配额，保持前缀闭合。
只测试这个固定设置，不搜配额，不利用正确reference选部署候选。

假设：预算分配而非新增候选可回收部分已观测的树遗漏。最小规模仍为本次6视频/48冻结
anchors：先补48次新树target验证，测ΔL_tree/L_actual、早期匹配损失与视频方向；随后同6条
分别做两种树的真实在线decode（共12条decode，AR逐token门禁），确认局部收益是否转化为
全局接受长度。记录构造成本，不能仅凭oracle承诺速度。

若新树不能减少固定anchor的tree_gap，或局部改善不能转化为跨视频一致的在线接受长度改善，
就推翻“这个固定配额方案值得继续投入”的建议，停止推进该方案；这不等于证明所有树选择
都无空间。不得在此结果之后自动恢复长程OPD或另开超参搜索。

## 7. 复现入口

新增独立脚本：`repro/current_ar_closed_loop.py`、`current_ar_analysis.py`、
`verify_current_ar_trace.py`、`test_current_ar_closed_loop.py`、`package_current_ar.py`。
原始manifest冻结名单与源码哈希；`manifest_final.json`记录完成与head前后哈希；
`delivery_manifest.json`为交付索引。正式表：`per_video_current_ar.csv`、
`diagnosis_current_ar.csv`；另有 `token_metrics_current_ar.jsonl`、`per_video_C0_tree_groups.csv`。

```bash
cd /home/mcy/Projects/GLANCE
PY=/home/mcy/.conda/envs/fin_research/bin/python
RUN=/mnt/local/mcy/GLANCE/results/phaseC6/current_ar_closed_loop_NEW_ID
# 先确认GPU0确为空闲且满足fp32容量；RUN必须不存在。
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PY" -u repro/current_ar_closed_loop.py --out "$RUN" --gpu 0 > "${RUN}.log" 2>&1
"$PY" repro/current_ar_analysis.py relabel --out "$RUN/A_relabel"
"$PY" repro/verify_current_ar_trace.py --run "$RUN"
"$PY" repro/current_ar_analysis.py analyse --out "$RUN"
"$PY" -m unittest discover -s repro -p test_current_ar_closed_loop.py -v
```

`package_current_ar.py`用于本次包含失败pilot的交付记账（`--run`成功目录、`--prior`失败目录），
不需要为了复现重新制造失败。新一次运行应使用独立目录，保留已完成证据。
