# 2026-10-15 QA 单路径接受、候选树与梯度方向诊断

产物：`/mnt/local3/mcy/GLANCE/results/qa_tree_semantics_20261015`。只读模型、无 optimizer step；生产代码未修改，未访问冻结新测试集。GPU 7 共享运行经过用户明确授权，所有计时不作速度证据。

## 结论与历史判断修正

存在明确实现错位，不能将 Task B 负结果视为正确在线 Draft-OPD 的公平否定。直接观测到 OPD checkpoint 在固定正确 AR prefix 上丢失有效候选；teacher 错位造成局部梯度冲突的证据很强，但尚不能定量判定各缺陷对长期 τ 退化的因果贡献。SFT 的候选覆盖改善可观测，其收益是否主要来自某类 reveal/位置尚不能判断。

本次审计对象 `qa_opd_fair/train_qa_opd.py` 是独立 QA trainer，不是 Task A 使用的 `opd_train_video_fullkl.py`。通用模块的既有 19/17 项测试及 Task A 的 k3 小正信号，不能替 QA trainer 的位置契约背书。Task A 本轮未重跑，也未据 QA 缺陷推翻其数值。

10月14日 FINAL/NEXT 报告及10月15日 Task B 的部署 τ、逐 token 无损结果继续有效，但下述机制描述失效：①首拒为0直接推出 top-8 缺失；②QA loss 是整段被拒后缀 Eq.8–13；③当前 QA trainer 的接受状态就是当前 target 对对应位置的验证；④仅凭 loss 流占比断言 rejected 梯度必然方向错误。当前发现是实现错误，不是新颖方法机制。

## 1. 真实语义审计

| 对象 | 实际语义与证据 | 判定 |
|---|---|---|
| 部署 root/prefix | `decode.py:243–310`：prefix KV 为 `full[:p]`，root=`full[p]` 已由 target 提交；draft 的 offset k 预测 `full[p+k]`，首轮 p=prompt_len。 | root 不计入单路径或树的 L。 |
| proposal / top-8 / tree | 同一次 block16 前向产生15个位置的独立 marginal；proposal 是各位置 argmax；GPU 实际 `log_softmax(...).topk(8)` 构建63节点前缀闭合树；ancestor mask 验证后沿 target argmax 行走。 | proposal 失败不蕴含 top-8 或树失败。 |
| QA anchors | 实际调用 `sft_candidate_positions` 后等距取样，使用存储 target continuation；不是当前策略真正访问的 rollout rounds。首 anchor p=L−1 是最后一个 prompt token，能训练第一答案 token，但不是部署首轮状态；本次单列。 | 是固定轨迹上的当前 proposal，不能称完整在线 rollout。 |
| n_acc | caller 在 raw_verifier 前复制 root，`_verify:399–408` 虽计算 `idx=p+k−L`，实际比较却用 `verifier[k]`；它总在比答案开头，只有 p=L−1 时索引对齐。 | CPU原函数反例 + 真实同prefix重放证实错误。 |
| valid | `valid.append(alive and can)`，alive 在首拒后为false。首拒本身有效，后续全部无效。 | Task B 原日志2880/2880 anchors均恰好1个 rejected offset；不是整段后缀。 |
| rejected teacher | `_anchor_loss:452–477` 从 p+1 输入 proposal，并取输出第 k−1 行；该行预测 p+k+1，却与预测 p+k 的 draft 第 k−1 行算 KL。 | teacher 条件多消费了当前 proposal，且teacher 行对应 p+k+1 而 draft 预测 p+k；原函数 recording stub 和真实模型对照均验证。 |
| accepted teacher | `_anchor_loss:434–440` 是正确轨迹 token 的 CE，即 one-hot KL；不是 soft target 全词表 forward KL。 | 实现与报告中的 Eq.8–13 命名不符。 |
| SFT | `train_qa_sft.py:269–286` 继承同一额外 prompt-boundary anchor；CE 标签位置为 p+k，variable reveal、0.8位置权重。caller 的 valid_end=cand[-1]=L+EOS_index−1。 | 实际排除 EOS 标签，旧 helper 测试用正确 valid_end，未覆盖 caller 错一位；非本轮主归因。 |

原始证据：`cpu_semantic_counterexamples.json`、`extra_cpu_contracts.json`、`historical_log_audit.json`。没有修改上述生产函数。日志不含完整历史 proposal，因此不能从日志复算原训练2880 anchors的精确误分组率。

## 2. 冻结样本、身份与门禁

从 Task B 使用的480槽训练池与 MSVD-B 既有开发集按已排序video_id、Random(20261015)各抽3视频，每视频取按字符串question_id排序的第一题。选择不看收益。训练/开发视频不相交，prompt模板一致但问题/视频内容不同；这是同域 train/dev，不是跨域泛化测试。

| split | video | QA | prompt长度 | 当前答案长度 | anchors |
|---|---|---|---:|---:|---|
| train | 5EXVeP6R3yc_105_108.avi | 0005909 | 995 | 43 | [994, 995, 1015, 1036] |
| train | 5W02895vT8c_312_322.avi | 0006322 | 993 | 47 | [992, 993, 1015, 1038] |
| train | aCKNdip53W4_0_25.avi | 0029870 | 999 | 29 | [998, 999, 1012, 1026] |
| dev | bmOy6p87TWI_26_35.avi | 0000165 | 999 | 64 | [998, 999, 1030, 1061] |
| dev | hSgGBHbJrmE_0_17.avi | 0004960 | 997 | 30 | [996, 997, 1011, 1025] |
| dev | ibSwITK4jjQ_14_24.avi | 0005542 | 994 | 44 | [993, 994, 1015, 1036] |

每视频4 anchors：L−1（训练边界，单列）、L、L+(答案长度−2)//2、最后一个非EOS token。部署可用子集共18 anchors（每split 9），每split 93个有效offset；EOS-only末端anchors每split 3个，保留并明确标注。所有 L 排除 root/bonus，含首个EOS预测、排除之后位置，最多15；不把局部L冒充在线τ。

完整输入、像素/grid、正确prefix、融合特征、target KV、M-RoPE delta保存于 `context_0..5.pt`；输入重复哈希一致，当前独立 greedy AR各重复两次一致，全轨迹cached prefill的argmax也与当前AR一致。target固定fp32；主表head fp32对应部署协议，额外完整重放bf16 head对应训练精度，层集合{1,9,17,25,33}、同mask/reveal=1、real时间轴、EOS immediate。三主臂18个部署anchors的 L_raw/L_tree/n_acc 在两种head精度下全部相同。

原存储训练reference在3条中2条与当前AR不同（首分叉答案位置29和17，0-based），输入prompt逐token一致；原因尚未隔离，不能宣称由某一精度导致，也不能称为普遍标签错误。首轮失败保留于 `prepare.log`；在任何head评分前用 `amend_reference_gate.json` 明确切换到当前AR，并将旧reference差异单列于 `reference_audit.jsonl`。此处的原QA目标梯度是当前正确prefix上的诊断重放，不是假称重现历史每次update的梯度。

| checkpoint | 真实路径 | SHA256 |
|---|---|---|
| C0 | `/mnt/local/mcy/GLANCE/checkpoints/head_video439_aligned/step1000` | `e2526a249a8ef7e1db3a9990589b727541486093372011e1a0ff6a16e06d23d2` |
| B_SFT120 | `/mnt/local3/mcy/GLANCE/results/taskB_fair_qa_20261015T085742/video/training/sft/step120` | `84fe984cc21f7ad5f7ee9fffc91d15ed8b0960f66159b694601e57e5a311d014` |
| B_OPD120 | `/mnt/local3/mcy/GLANCE/results/taskB_fair_qa_20261015T085742/video/training/opd/step120` | `f2dc7d15a5f55af91aa64e46d04840f974b50d4d0965bd71a1fc4b9b865f6d60` |
| old_SFT_s1_120 | `/mnt/local2/mcy/GLANCE/results/qa_opd_fair_20261013_195500/training/seed_20261013/sft/step120` | `c5a65583e972581faa220cfa03fe3f805cf71e34e8306d7883d102f2abe161f6` |
| old_OPD_s1_30 | `/mnt/local2/mcy/GLANCE/results/qa_opd_fair_20261013_195500/training/seed_20261013/opd/step30` | `7fb4de49f525bef7672302a083171e1870394a210c72df652db3ed756ccd0f05` |
| old_OPD_s1_60 | `/mnt/local2/mcy/GLANCE/results/qa_opd_fair_20261013_195500/training/seed_20261013/opd_resumed/step60` | `dd1f5038bed14812bf32a2bdcbcb05e7133e64da4f0f32695769d97e500f9f70` |
| old_SFT_s2_120 | `/mnt/local2/mcy/GLANCE/results/qa_opd_fair_20261013_195500/training/seed_20261014/sft/step120` | `641aba6ce164b54ddc41dcdd88156061e8ad7d8d9bdcd670447f132d7e7ab110` |
| old_OPD_s2_120 | `/mnt/local2/mcy/GLANCE/results/qa_opd_fair_20261013_195500/training/seed_20261014/opd/step120` | `9f2a05e1f1c851e688ab580276c664698adfe98844f6231e81c6541de58410a3` |

Task B仅保存step0/120，历史部分step30/60/90已删除；未重建缺失checkpoint，也未训练填空。历史seed1的OPD30/60、SFT120及seed2的SFT120/OPD120只作辅助轨迹，在本轮同一real/fp32协议下重测；不与Task B当作同seed连续阶段混池。这里train指Task B训练池，不表示每个早期checkpoint都已见过该QA；按冻结日程截断到各checkpoint update的实际曝光表见 `checkpoint_exposure.csv`。

384份局部trace（24 anchors×8 heads×2 precisions）通过实际GPU top8、节点/parent/depth重建、完整ancestor mask、L_actual=L_tree、正确AR路径和bonus检查，重复head前向逐位一致。原始候选与分数直接在GPU捕获，未用CPU重新softmax替换。
额外18次未改动生产decode首轮重放：18/18 root与正确AR一致、18/18路径一致、18/18 L_tree与固定prefix诊断一致；top8 IDs完全一致 18/18。最大top8 logp差 4.3869019e-05，数值差异原样保留。

## 3. 固定正确prefix的完整主指标

先按视频聚合，再在每split 3视频取均值；token命中率以该视频全部有效offset为分母。首位置命中率、首缺失位置、每anchor完整raw trace另见CSV/JSONL。

| split | head | 训练n_acc | 正确单路径L | top1 | top8 | L_raw | L_tree=L_actual | tree_gap | 首阻断 missing/tree/边界 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| train | C0 | 0.0000 | 0.7778 | 24.73% | 56.99% | 3.4444 | 1.5556 | 1.8889 | 3/3/3 |
| dev | C0 | 0.0000 | 1.2222 | 26.88% | 68.82% | 3.6667 | 2.4444 | 1.2222 | 3/3/3 |
| train | B_SFT120 | 0.0000 | 0.8889 | 31.18% | 63.44% | 3.5556 | 2.0000 | 1.5556 | 4/2/3 |
| dev | B_SFT120 | 0.0000 | 1.2222 | 25.81% | 73.12% | 3.8889 | 2.6667 | 1.2222 | 3/3/3 |
| train | B_OPD120 | 0.0000 | 0.3333 | 4.30% | 36.56% | 0.3333 | 0.3333 | 0.0000 | 6/0/3 |
| dev | B_OPD120 | 0.0000 | 0.3333 | 4.30% | 47.31% | 1.4444 | 0.5556 | 0.8889 | 4/2/3 |

C0训练verifier在18个部署anchors均给0，而正确单路径有12个非零；扣除EOS-only边界后仍有6/12个错分。正常的“正确单路径L=0但tree>0”在非末端anchors为训练2/6、开发1/6，必须与索引bug分开。首个proposal被拒不能代替候选树失败。

OPD相对C0：训练top8从56.99%降至36.56%，开发从68.82%降至47.31%，每组3/3视频均下降。训练丢失22、恢复3个正确top8成员；开发丢失24、恢复4个。SFT分别丢失1/恢复7、丢失1/恢复5。开发SFT的top1略降而top8/L_tree上升，进一步表明不能用单条argmax proposal概括树收益。

| 配对（均为OPD120−C0） | ΔL_raw [视频bootstrap95%CI] | ΔL_tree [CI] | Δtop8 [CI] |
|---|---|---|---|
| train n=3 | −3.1111 [−5.3333,0] | −1.2222 [−2.0000,0] | −20.43pp [−29.03,−6.45] |
| dev n=3 | −2.2222 [−3.6667,−0.6667] | −1.8889 [−2.3333,−1.6667] | −21.51pp [−29.03,−16.13] |

bootstrap=2000，seed20261015，单位视频；n=3非常小，CI只是探针内部描述，不是稳健总体效应。SFT−C0开发ΔL_tree=+0.2222 [0,+0.3333]、Δtop8=+4.30pp [−3.23,+9.68]，不据此额外声称显著收益。所有臂/精度/指标的完整配对表在 `paired_video_bootstrap.csv`。

## 4. 历史checkpoint辅助重放

| head | split | L_raw | L_tree | top8 |
|---|---|---:|---:|---:|
| old_SFT_s1_120 | train | 3.2222 | 2.1111 | 60.22% |
| old_SFT_s1_120 | dev | 4.5556 | 2.6667 | 73.12% |
| old_OPD_s1_30 | train | 3.4444 | 1.5556 | 55.91% |
| old_OPD_s1_30 | dev | 3.6667 | 2.2222 | 69.89% |
| old_OPD_s1_60 | train | 2.5556 | 1.3333 | 47.31% |
| old_OPD_s1_60 | dev | 3.3333 | 1.5556 | 61.29% |
| old_SFT_s2_120 | train | 3.5556 | 2.0000 | 63.44% |
| old_SFT_s2_120 | dev | 3.8889 | 2.6667 | 72.04% |
| old_OPD_s2_120 | train | 0.3333 | 0.3333 | 31.18% |
| old_OPD_s2_120 | dev | 1.1111 | 0.3333 | 45.16% |

早期OPD30可能主要改变排序/树覆盖，OPD60及120出现更广候选退化；不能把不同seed/时间轴/训练target精度拼成单一连续学习曲线。C6是描述任务的phaseC4三头，原24/18 raw/tree首阻断统计保持历史适用范围，本轮QA统计不与之混池。

## 5. 完整参数梯度方向（不更新）

测量所有58个可训练参数张量，target冻结，head bf16、teacher fp32；同一prefix/root/视觉特征/reveal=1输入图。比较 accepted CE、实际 rejected KL、仅teacher位置对齐后的 rejected KL、相同rejected offsets的CE、全有效offset 0.8加权CE。逐流归一的正标量不改变单anchor流的余弦；原 `_anchor_loss` 实际调用与诊断重构的loss数值门禁通过。

这里的CE是相同部署mask下的SFT式正确token目标，不是历史随机reveal、多anchor batch、完整update的原始梯度；未测Adam预条件后的更新方向。分流梯度与teacher对齐对照不代表已经修复n_acc或训练。

首部署root每视频1个，共18份（3头×6视频）；其accepted全空，因此 accepted/rejected 余弦为N/A。为直接测两流，另在同6视频的真实训练首anchor p=L−1测18份，单列，不混入部署root样本。零流不补造、余弦不填0。

| head | split | 首root n | actual rejected vs同offset CE均值（负向数） | aligned teacher vs同offset CE均值（负向数） |
|---|---|---:|---:|---:|
| C0 | train | 3 | -0.0625 (2/3) | +0.7610 (0/3) |
| C0 | dev | 3 | -0.9287 (3/3) | +0.9590 (0/3) |
| B_SFT120 | train | 3 | -0.5175 (3/3) | +0.5607 (0/3) |
| B_SFT120 | dev | 3 | -0.9573 (3/3) | +0.8545 (0/3) |
| B_OPD120 | train | 3 | -0.7077 (3/3) | +0.7759 (0/3) |
| B_OPD120 | dev | 3 | -0.8311 (3/3) | +0.8962 (0/3) |

| head | split | 训练首anchor n | accepted非空 | accepted vs rejected均值 | accepted vs全offset CE均值 |
|---|---|---:|---:|---:|---:|
| C0 | train | 3 | 2 | +0.3188 | +0.6688 |
| C0 | dev | 3 | 2 | +0.2428 | +0.7928 |
| B_SFT120 | train | 3 | 2 | -0.0342 | +0.0370 |
| B_SFT120 | dev | 3 | 2 | -0.0921 | +0.6610 |
| B_OPD120 | train | 3 | 1 | +0.1396 | +0.4298 |
| B_OPD120 | dev | 3 | 2 | -0.1303 | +0.2656 |

完整范数、原始loss、teacher argmax、正确token、每张量dot/norm²与余弦见 `gradient_results.jsonl`；分组全指标见 `gradient_summary.csv`。参数无optimizer步骤，内存version门禁及磁盘SHA门禁均记录。首版CPU归约过慢，保留已完成记录，停止本人的只读进程后以等价GPU归约续跑；bf16乘积可在fp32精确表示，再fp64累加，CPU等价测试见 `reduction_audit.json`。未丢弃任何不利梯度样本。例如OPD120的一个训练边界anchor中，对齐teacher与同位置CE余弦仍为−0.0153；对齐并不保证所有参数梯度都与greedy CE同向。

## 6. 不同解释与证据边界

| 解释 | 支持/反证 | 可下结论 |
|---|---|---|
| 只因训练单路径失败、部署用树 | C0的top8可恢复部分top1失败；但QA verifier还有独立索引错误。 | 有结构差异，但不能把全部失配归为拓扑差异。 |
| OPD只破坏树选择，候选仍好 | 同prefix top8成员明确丢失、L_raw明显下降；tree_gap反而缩小。 | 不支持“只坏在树选择”；当前直接瓶颈是候选覆盖。 |
| teacher错位导致监督方向冲突 | 原函数CPU契约、真实teacher行位置和参数梯度对照一致；仅对齐teacher后与同位置CE方向更一致。 | 强局部机制证据；尚不能定量归因全部长期τ下降。 |
| accepted/rejected天然相互冲突 | 首部署root accepted为空且分组存在索引错误；训练首anchor的两流方向单列，空流不计入余弦。 | 不能宣称两流天然冲突或reverse KL必然无用。 |
| 仅训练/开发分布差异 | 相同模板、视频不重叠；训练组与开发组均丢候选。 | 不支持仅由泛化差异解释；不能外推其他任务。 |
| 仅bf16舍入造成 | 两head精度重放主要长度结论一致；Task B已有fp32主权重。 | 本探针不支持舍入是主要解释，但不取消历史训练数值差异。 |

## 7. 唯一下一步实验（建议，未启动）

先把QA训练的位置契约修正并通过无更新门禁，再做一次固定10-update的“正确位置契约QA-OPD vs 等预算QA-SFT”救援对照：同C0起点、同QA日程、当前fp32 AR、同head/主权重精度、同时间轴/EOS及实际监督预算记录。修复内容预先列明（接受索引、teacher row、valid语义及accepted目标名称），不同时引入tree-aware loss、k3或新树；旧缺陷checkpoint仅作历史参照。主终点为冻结开发prefix的top8/L_raw与完整无损b63k8接受，预注册失败即停止，不延长训练。

这是实现正确性的救援对照，不是算法创新。若要分离每个bug的因果贡献，需以后另行授权单变量实验；本次不自动启动，也不把修复收益当作OPD理论收益。

## 8. 交付与复现

`prereg.json` + 三份明文amend、所有源码与头SHA、CPU反例、6份固定context、384份原始tree traces、逐anchor/逐video CSV、视频bootstrap、36份参数梯度、18份生产首轮门禁、失败/续跑日志、命令与 `delivery_manifest.json`。

复算（无需GPU）：`python analyse.py`；原函数CPU审计：`python extra_cpu_audit.py`。GPU脚本在全新run目录复制预注册与脚本后依次执行：`probe_v2.py prepare`、`probe_v2.py score`、`probe_fast.py grad`、`boundary_and_deploy_gate_fast.py`；固定使用物理GPU7，运行前按当时授权检查资源。不得覆盖本次原始结果或将缓存写回home。
