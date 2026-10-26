# 数据集与 checkpoint 存储规则（必须遵守）

- 数据集、预处理数据/缓存、模型权重和 checkpoint 必须实际存放在 `/mnt/local`、`/mnt/local2` 或 `/mnt/local3` 下；禁止下载或保存到 home 目录（包括 `/home/mcy`）的实际存储中。
- 下载、解压、缓存和保存 checkpoint 前，先核实目标路径及软链接解析后的真实位置；相关临时大文件也应放在上述 `/mnt` 目录中。
- home 下可以保留指向上述 `/mnt` 目录的软链接，但不能把软链接替换为存放数据或 checkpoint 的真实目录。现有 `/home/mcy/local`、`local2`、`local3` 属于此类软链接。
- 默认 checkpoint 路径为 `/mnt/local/mcy/GLANCE/checkpoints/`；本轮视频 QA 数据位于 `/mnt/local2/mcy/datasets/videoqa_transfer/`。可按容量选择另外两个允许的挂载目录。

# GLANCE 项目总结与进度

最后核对：**2026-10-16（ScienceQA 图像 SFT/OPD 公平对照已完成：普通负结果）**。本文是项目当前状态、下一步工作和交接约定的唯一维护入口；历史报告用于追溯证据。记录来自本地代码、实验报告及结果文件。

> **2026-10-16 ScienceQA 图像对照：三目录审计 → 补下载 → 冻结 1200/240/240 → 门禁 → 两 seed × 两臂 120 updates → 无损评测。结论：OPD 未跑赢同预算继续 SFT（普通负结果，配方关闭）。**
> 审计：`data_sqa_train` 2,400 条**全部官方 train**、`data_sqa_test` 2,017 条**全部官方 test**、
> `data_sqa_full` 是二者并集；**官方 validation 一条都没有，train 被截断到 2400/6218** ⇒「没下载全」成立。
> 连接键为 `sha256(图像字节)+question+choices+answer`（8,834/8,834 匹配，0 未匹配）；官方 `pid` 字面串因
> `github.com`/`raw.githubusercontent.com` 本机不可达而不可得。官方同图**跨 split** 复用 2,797 条，已整簇排除。
> 从 `lmms-lab-encoder/ScienceQA@69dd4d6b…` 补下载缺失 split，冻结 train 1200 / dev 240 / test 240，
> 跨集合同图簇 **0**、重复记录 **0**，全部真实可解码。
> 门禁：图像像素确实驱动 fused hidden 与续写；长度探针 p50 **121.5** token、自然 EOS 31/32 ⇒ `max_new_tokens=256`；
> b63k8 SPEC == 独立 AR **8/8 逐 token**；图像训练器由冻结视频训练器**生成**，反向替换 13 类改动后与冻结源**逐字节相等**
> （24/24 + 冻结 13/13 + 4/4 CPU 测试）；2 张真实图无更新 GPU 重放通过（`n_acc` == 在线重算、梯度有限非零、
> 参数逐位未变、`optimizer.step()` 未调用）；fp32 master + Adam 逐位恢复。
> 预算：原 300 updates **未执行**；训练前登记统一修订为 **120 updates / 480 QA 曝光**（另存测量附录，未据训练动态回调）。
> 结果（图像簇配对 bootstrap 2000×，全部逐 token 无损，各臂截断/EOS 计数完全相同，目标答案准确率三臂一致）：
> dev I0 **4.0050** / SFT **4.3366,4.3394** / OPD **3.9415,3.9495**；test I0 **4.0723** / SFT **4.4059,4.4309** / OPD **4.0479,4.0583**。
> **OPD−SFT：dev −0.3951[−0.4368,−0.3557] / −0.3899[−0.4308,−0.3498]；test −0.3581[−0.3942,−0.3190] / −0.3727[−0.4133,−0.3299]，
> 全部为负且 CI 不跨零。同期 SFT−I0 = +0.332/+0.334（dev）、+0.334/+0.359（test），全部正向且 CI 不跨零。**
> step30 灾难退出门禁未触发（OPD30−I0 +0.018；SFT30−I0 +0.118~+0.146）。机制线索：固定 anchor 上 OPD 的
> `L_tree` **1.63–1.65** 低于 SFT **2.17–2.28**，但其树概率代理 **Φ 6.5 高于 SFT 3.5**——
> 「代理目标上升、真实接受行为下降」，方向对 OPD 不利，不构成方法优势。
> 判断：**按预注册「普通负结果」关闭该 ScienceQA 全词表 KL 配方**，不自动切 k3/Bernoulli KL/TV/P2/P3、不加轮次、不扩数据集；
> 唯一保留（不启动）的后续假设是固定头部/候选/预算下 best-first 换按深度固定配额的树选择对照。
> 计时 19/19 标 `timing_contaminated`，只用同一次评测内倍数（1.53×–2.17×）说明，不作加速结论。
> [完整报告](../../reports/2026-10-15-scienceqa-image-sft-vs-opd.md)（由产物生成，勿手改数字）；
> 产物 `/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/`；
> 图像 `/mnt/local2/mcy/datasets/scienceqa_glance_20261015/lmms-lab-encoder_ScienceQA@69dd4d6b/images/`；
> 脚本 `repro/scienceqa_image/`。历史产物未改，`checkpoints/head` 逐位未变；仅清理本轮自己的中间产物（smoke step0/step10 与四份 step0；四份 step0 与 `checkpoints/head` 逐字节相同）并逐个留哈希记录。

> **2026-10-15 第五轮：QA位置契约已修复，10-update公平验证完成。**
> 修复接受绝对索引、draft-induced teacher预测行及整段rejected suffix mask；SFT同步包含EOS标签。13项新CPU测试+4项既有测试、6份无更新完整梯度与独立逐token teacher门禁通过。
> 以 **Task B QA-SFT-120**（`84fe984c…`）为共同起点，40 QA槽、两臂各10 updates、lr3e-5、fp32 target/bf16 head/CPU主权重、fresh Adam、real时间轴；原QA loss公式保留，未引入k3/P2/P3。
> dev20视频40QA：ZERO **3.639499** / SFT10 **3.677040** / OPD10 **3.642907**；OPD−SFT **-0.034133 CI[-0.117019,+0.042432]**；OPD−ZERO **+0.003408 CI[-0.061922,+0.070757]**。
> 主评测120/120、复测120/120 SPEC=AR；GPU7共享经授权，计时不作速度证据。未获得超过 continue-SFT 的可靠证据；保留位置契约正确性结果，不宣称方法优势，不继续训练。
> 历史checkpoint/评测产物未改；未使用冻结新测试集、未追加种子或训练。[完整报告](../../reports/2026-10-15-qa-contract-fix-10update.md)；产物 `/mnt/local3/mcy/GLANCE/results/qa_contract_fix_20261015`。

> **2026-10-15 第四轮审计补正：Task B 的 QA-OPD 独立 trainer 存在真实实现错误。**
> `train_qa_opd.py` 的接受索引未随 anchor 移动；rejected teacher logits 预测 p+k+1，而 draft 预测 p+k；
> `valid` 在首拒后关闭，原训练日志 **2880/2880 anchors 均只有1个 rejected offset**；accepted 实际为 greedy CE，而非 soft forward KL。
> 新冻结 **3 train + 3 dev 视频 / 24 anchors / 8 heads / 2种head精度**，384份局部trace门禁、18次生产首轮复核、36份无更新梯度测量完成。
> Task B OPD120 的 dev 正确top8覆盖 **68.82%→47.31%**，L_raw **3.6667→1.4444**，L_tree **2.4444→0.5556**；训练组亦退化。
> C0首部署root的 rejected梯度与同位置CE余弦 **5/6为负，均值−0.4956**；仅teacher对齐后 **6/6为正，均值+0.8600**。
> **历史部署τ与无损数字仍有效，但不能据此否定正确实现的OPD；此前“整段错误后缀主导”的机制解释撤回。**
> Task A 使用另一入口，其结果不因这次QA审计自动失效。主因的定量因果贡献仍尚不能判断。
> [完整诊断、样本、门禁及原始产物](../../reports/2026-10-15-qa-tree-semantics-diagnosis.md)。生产代码未改、optimizer steps=0、冻结新测试集未访问；该诊断时点修复/10-update对照**未启动**；第五轮现已完成，见上方最新状态。

> ## ✅ 2026-10-15 派发的两个任务**已执行完毕**（同日第三轮）
> - **任务 A（P1）已完成**：被拒段目标 全词表 reverse KL → 官方 `run_qwen_gsm8k.sh`
>   (`d1476f5`) 的 **k3 单样本估计量**，CPU 与官方源码逐位对拍、上 GPU 前标度测量、
>   预注册、同预算对照训练与 fp32/b63k8/AR 评测全部完成。
>   **结果：Δτ = +0.0180，CI [+0.0002,+0.0381]，t +1.83（n=29，两臂各 30/30 无损）**
>   —— 贴着分辨率的小幅正信号，**不足以**支持"reverse KL 形式是负结果主因"，
>   也**不构成加速结论**。P2/P3 未实现、未跑（保持单变量纪律）。
>   → [报告](../../reports/2026-10-15-taskA-kl-official-p1.md)；
>   产物 `results/taskA_kl_official_20261015T084308/`
> - **任务 B 已完成**：视频 QA（seed20261014，**两臂各 120/120**，fp32，**训练与评测同为
>   real 时间轴**）与图像 QA（**三臂同起点** `checkpoints/head`，2000 行/1000 步/lr 1e-4，
>   全部在本轮重测）两组对照重做完毕。
>   **视频**：warm 3.4700 → QA-SFT **3.6395**（+0.1695 [+0.0901,+0.2472]）→ QA-OPD
>   **1.2060**（−2.2640 [−2.4429,−2.0915]，0/20 视频改善），三臂共 **120/120 逐 token 无损**。
>   **图像**：warm **5.1990** → SFT **5.3676**（+0.1463）→ OPD-CE **5.3225**（+0.1218）；
>   SFT−OPD-CE = +0.0245 **CI 跨零**（单 seed 不可区分）。
>   **§B 的数据量混同已消除**：`head` 是三臂共同起点，没有任何 16K 数字参与比较。
>   → [报告](../../reports/2026-10-15-taskB-fair-qa-redo.md)；
>   产物 `/mnt/local3/mcy/GLANCE/results/taskB_fair_qa_20261015T085742/`
>
> 两任务的约束、预注册要求、历史坑清单仍在下方对应章节，供追溯。

## 项目定位

本项目复现 GLANCE 多模态无损投机解码，并研究 Draft-OPD 训练能否进一步提高草稿头的接受长度与实际速度。基础解码架构来自 GLANCE/DFlash；本地工作包括复现训练与评测管线、图像/视频接入、训练方法对照、特征路径修复和无损审计。

冻结 Qwen3-VL-8B-Instruct，以其第 `{1,9,17,25,33}` 层融合视觉与语言的隐状态作为 DFlash 草稿头输入。草稿头约 1.05B、5 层、block size 16；一次前向生成各位置的候选分布，构建前缀闭合的候选树，再由 target 验证并提交匹配前缀及 bonus token。常用评测树为 budget 63、branch 8。最终输出是否无损必须通过 AR 逐 token 对照验证。

**当前结论：图像复现已取得接近参考表格的结果；视频 aligned SFT 已完成并确定新的实验起点；尚无证据证明视频 OPD 优于同预算 continue-SFT。**

## 已有结果与边界

| 工作 | 结果 | 解释与证据 |
|---|---|---|
| 图像复现 | ChartQA b63：τ≈5.20，decode-only 加速 3.36×；fp32 审计 20/20 一致 | 原始问题 prompt、A6000；不是端到端加速，也不是生产引擎 Table 1 的直接复现。[复现记录](../../reports/2026-09-25-reproduction-history.md) |
| Markov 条件化探索 | τ 小幅增加但耗时上升，未达预注册目标 | 当前实验未取得速度收益，已结束该轮探索；见复现记录 §8.5 |
| 图像 OPD 历史探索 | 修正配方改善了部分旧配置，但不足以据此声称超过继续 SFT | 多版方法说明已归档；旧 top-k/静态池实现不能称为本轮全词表在线 OPD |
| 视频旧版 2×2 | online 主效应 +0.0710 τ；四臂均低于旧 warm start | 单 seed、200 updates、旧损失与旧初始化；不支持“只改 online 就有效”。[2×2 报告](../../reports/2026-10-08-video-opd-2x2.md) |
| 阶段 A（legacy SFT） | step1000 mean τ=3.5785 | 历史特征路径基线，已被 A4 替代。[A 报告](../../reports/2026-10-09-phase-a.md) |
| 阶段 A4（aligned SFT） | step1000 mean τ=**3.7377**，median=3.7778；六个 checkpoint 各 30/30 无损 | step750–1250 为宽峰；同一步数相对 legacy +0.1592 τ。[A4 报告](../../reports/2026-10-09-phase-a4.md) |
| B0 teacher 原型 | fp32 增量与独立前向最大 logit 差约 3.43e-5，峰值 34.09 GiB | 只证明原型前向可行，不代表含 Adam 状态及 backward 的完整训练显存已通过 |

图像复现的 τ 按所有 round 池化；A/A4 表中的 mean τ 是逐视频 τ 的算术均值，两者不可混用。A4 输入 30 条，首条 warmup 不参与指标聚合，故 n=29；无损门禁仍检查全部 30 条。旧 2×2 warm 参考按两个 shard 排除 warmup，其配对比较使用 n=28，不能直接与 A4 n=29 当作同一组样本相减。

A4 指标已核对本地 `results/phaseA4/summary_phaseA4.json`。其余历史结果按对应报告保留，引用时应继续查对应原始产物与配置。

## 当前进度

| 阶段 | 状态 | 交接说明 |
|---|---|---|
| 图像复现、Markov 探索、旧视频 2×2 | 已完成该轮实验 | 保留正负结果及限制，不自动扩展结论 |
| A / A4 | 已完成 | 正式起点改为 `checkpoints/head_video439_aligned/step1000` |
| B0 | 原型机制与精度检查已有结果 | 见 `results/phaseB0/`；旧 head、少量 anchors 的统计不能外推到 aligned 起点 |
| B0.5 | 已完成 | 8 条训练视频 / 32 anchors，同轨迹·prefix·mask 比较 bf16 vs fp32 teacher：总 loss 相对差 1.2e-3、**逐 anchor 可达 9.5%**、teacher argmax 97.7%、梯度 cosine 0.99986（纯 teacher 效应）；**bf16 会重选 anchors（8/8 视频不同）**。[报告](../../reports/2026-10-09-b1-precision-and-phasec-config.md) |
| B1 | 已完成 | `repro/opd_fullkl.py`（16/16 测试）+ `repro/opd_train_video_fullkl.py`（17/17 测试），实现论文 Eq.8–13；**fp32 teacher + fp32 head** 实测 OOM（含 1-anchor 对照）；**fp32 teacher + bf16 head 实测可跑（峰值 43.84 GiB）**；bf16/bf16 峰值 26.42 GiB |
| Phase C（seed 0） | 已完成，**负结果**（τ 排序不变） | C0/C1/C2 = **3.7377 / 3.6579 / 3.5095**（n=29，各 30/30 无损）；C2−C0 −0.2281 CI[−0.2830,−0.1738]，C2−C1 −0.1483 CI[−0.2037,−0.0953]，C1−C0 −0.0798 CI[−0.1269,−0.0311]。触发预注册停止规则 2 → 收口。[报告](../../reports/2026-10-10-phase-c-seed0.md) |
| 报告审计与补正 | 已完成 | 用户核对代码后指出四处解释问题，全部用代码/配置/实测复核：①C1 真实监督 **11648 个位置**（非 1200），C2/C1 = **1.55×** 不是 15×；②**fp32 teacher + bf16 head 实测可跑**（3 次真实 update，峰值 **43.84 GiB**），原「fp32 不可行」应限定为 fp32 head；③C2 的 anchors 是**在线当前策略**产生，非固定 fp32 池，「bf16 重选 anchors」未被控制；④「50 步太短」属过度解读。[补正报告](../../reports/2026-10-10-phase-c-corrections.md) |
| continue-SFT 小步校准 | 已完成，找到可用设置 | lr 3e-5 / 10 updates / 96 anchors；完整 validation **3.7438 vs C0 3.7377 = +0.0061** CI[+0.0000,+0.0137]，过预注册门槛 −0.02。**但权重只移动 4.86e-05（比 lr 3e-4 小 179 倍）、29 条里仅 3 条 τ 变化 → 属「基本不变」型控制臂**。[预注册](../../reports/2026-10-10-phase-c2-prereg.md) |
| **参数更新精度诊断** | 已完成，**发现优化器精度缺陷** | 训练入口一直是 `AdamW(bf16 params)`、**无 fp32 主权重**。真实 head 实测：**lr 3e-5 下第一步只实现 0.223 的 fp32 理想更新范数，96.9% 的元素步长 < 半个 bf16 ulp 被舍掉且不累积**（10 步 loss 降幅仅 fp32 主权重的 1/5.2）；lr 3e-4 第一步不受影响（比值 1.042）。修复：`repro/master_optim.py` + `--master-weights {none,cpu,gpu}`（默认 none 保持历史行为）。**加主权重后同一 lr 的权重位移提高约 12 倍**（lr3e-5 25 步：6.70e-05 → 8.10e-04）。**历史上所有训练（A4 1500 步、Phase C 50 updates）都在该缺陷下跑过**。[报告](../../reports/2026-10-10-update-precision.md) |
| fp32 主权重复校准（phaseC3） | **已完成** | 6/6 监控候选全部通过 −0.02 门槛（修复前多为 0.000）；冻结规则选出 **lr 3e-5 / u10**。完整 validation（n=29，30/30 无损）**3.7453 vs C0 3.7377 = +0.0077，CI[−0.0095,+0.0243]，t +0.86，9/29 正向；相对 C0 有 14/29 条 τ 改变（修复前仅 3/29）**。相对 C1 phaseC +0.0875 CI[+0.0391,+0.1328]、相对 C2 +0.2358 CI[+0.1869,+0.2862]。**CI 跨零 → 只能说"与起点不可区分"，不能称"不退化"**（按用户判据属「基本不变」型控制臂）。流水线 `run_phaseC3_pipeline.sh` 串起了「6 评测 → 冻结规则选择 → 完整 validation」 |
| **phaseC4 冻结 10-update 正式对照** | **已完成** | C1/C2 同起点（aligned step1000）、**同 context 数值路径（C1 `--target-dtype float32` == C2 `--teacher-dtype float32`）**、**同 fp32 主权重（CPU）**、lr 3e-5、10 updates、24 anchors/update、1 video/update。完整 validation（n=29，各 30/30 无损）：C0 **3.7377** / C1 **3.7225** / C2 **3.7403**。配对：**C2−C0 = +0.0027 CI[−0.0203,+0.0274] t +0.21（7/29 正向）**；**C2−C1 = +0.0179 CI[−0.0146,+0.0543] t +0.99（11/29 正向）**；C1−C0 = −0.0152。**两项都为正但都跨零、效应量 ~0.003/0.018 τ、正向视频仅 7/29 与 11/29 → 判为噪声内，不值得复现**；按约定转入数据与目标诊断。真实监督预算：C1 = 2404 位置/权重和 1008；C2 = 3600 offset（acc 485 + rej 3115），**比 1.50×** |
| 优化器集成验证（C2 主权重） | 已完成 | `opd_train_video_fullkl.py` 接入 `--master-weights {none,cpu,gpu}`；**组合门禁**（fp32 teacher + bf16 head + CPU 主权重）3 次真实 update 通过，**峰值 39.7 GiB**（比无主权重的 43.84 GiB 多约 4 GiB 余量），host RSS 35.1 GiB，两流非空；裁剪改在 fp32 主权重梯度上。checkpoint 含 `optimizer_master.pt`（fp32 参数 + fp32 Adam 矩，实测 `exp_avg` 非零、step=3），`load_master` **逐位恢复**且 bf16 前向参数不变；"朴素续训"会从零重启 Adam |
| **phaseC5 ACL 收益诊断审计** | **本阶段审计/CPU 补证完成；当前AR诊断后由C6补齐** | 复用70视频×24旧anchors，输入prompt重放70/70一致；旧continuation与fp32部署AR **69/70分叉（validation 30/30）**，仅883/1680 anchors处于共同prefix，58个anchors的旧接受长度与当前teacher不符。现有“chain”是b1k1单候选探针，不是block16链。三臂训练组tree已自然完成，各40/40无损；加validation各30/30，共70/70，跨臂AR相同。C2−C0 τ：本轮C2曝光10条+0.0950、未曝光训练池20条+0.0665、validation29条+0.0027；曝光组相对完整未曝光组的差中差CI跨零，不能证明收益只集中于曝光。C1/C2曝光集合交集0。先交付参考路径不一致阻塞，不扩展D4/训练。[报告](../../reports/2026-10-12-gain-stuck.md)；产物 `results/phaseC5/diagnostic_acl_20261012T213006/`，最终表`diagnosis_table_final.csv`，交付索引`delivery_manifest.json` |
| **phaseC6 当前AR固定anchor闭环** | **已完成，无训练** | 冻结6条validation/48真实C0轮次（44条有有效offset），144份三臂trace通过独立复算、当前AR/packed判断门禁；A重标未改变原局部判断。C2−C0局部ΔL_raw/ΔL_actual均−0.0833，非全局Δτ；C0首阻断24次raw缺失、18次树遗漏、commit_gap均0。下一步仅建议固定head/候选/预算的单一树选择对照，未启动；D4/完整chain/新测试集未运行。[阶段报告与交付入口](../../reports/2026-10-12-current-ar-fixed-anchor.md) |
| **phaseC5 B-reference 对齐实验** | **已完成，无训练** | 6 条 validation / 48 个**真实 fp32 C0 b63k8 部署轮次**的前 8 轮（非磁盘旧 token）。门禁 16/16 通过后继续；**C0 重放 48/48 精确复现**（提交长度/ token / bonus / draft top-8 / tree 节点全同），插桩 round-loop 与未改动的 `glance.generate` 逐 token 一致。**离线 rank-1 恰好解释 block-16 chain（48/48，均值 2.8333/2.75/2.9167 与 chain commit 相同）**；tree 的提交长度**不能仅由候选覆盖解释**：首拒原因 `in_topk_not_in_tree` **29/28/29**、`not_in_topk` 19/20/19、**`in_tree` 恒为 0**；事后 top-8 上限比 tree 提交高约 +1.7~2.0 token。**block-16 chain 与 tree 差很大**：Δτ(chain−tree) = **−1.27 [−1.39,−1.14]** → 此前的 b1k1 单候选探针会误导。**分支：提交仍无法由候选解释 → 下一步做固定 head/候选/预算的树选择对照，不启动训练**。仅 12/48 anchors 处于三臂真共同前缀（36 条为分叉后反事实重放，已标注不混池）。**§12.2 的"离线未校准到 tree"收窄为跨 anchor 分布假象**：固定对齐上下文下首 token 二者是同一事件。[报告 §13](../../reports/2026-10-12-gain-stuck.md) |
| **OPD在线语义代码审计（2026-10-13）** | **审计时尚未实现；后续修复见下一行** | C4入口确实每update重算当前head proposal，teacher在线评分draft-induced prefix；但`full_ids`来自旧`video_greedy_499.pt`，`n_acc`由proposal匹配旧token得到，后续root/context沿旧轨迹推进，在线teacher不重新确定分组。因此不是纯固定池离线训练，也不等同官方真实target verify的完整在线Draft-OPD。CPU16+17+3测试通过，合成反例证实旧reference分组可与当前teacher冲突；独立汇总C5仍为69/70轨迹分叉、58/1680 anchor接受长度不同。loss算术通过不代表闭环正确；现有QA数值有效，但不能作为标准在线OPD公平负结果。恢复训练前须先修并验证此闭环；未改训练代码、未训练。[审计报告](../../reports/2026-10-13-opd-online-implementation-audit.md)。 |
| **OPD在线闭环修复（2026-10-13）** | **代码与小规模无更新门禁完成，未训练** | 默认`online`从prompt启动，当前target逐token验证并推进KV/融合特征；teacher/student/接受标签重放不一致即报错，旧continuation不参与。保留显式`legacy-reference`供历史Phase C/C4及诊断重放。CPU11+17+16通过；A6000、fp32 target/bf16 C0，2训练视频6轮独立AR一致、6/6重放通过，2次backward梯度有限非零，峰值38.91GiB，optimizer step=0、C0哈希不变。逐token验证/fresh draft cache成本较高，未验证完整Adam状态及长程吞吐，不代表收益改善。[修复报告](../../reports/2026-10-13-opd-online-fix.md)。 |
| **视频 QA 跨任务迁移（2026-10-13）** | **已完成，无训练** | MSVD/MSRVTT/NExT-QA多选，各20视频40QA、A/B两格式；正式2160/2160、smoke324/324无损。B的C2−C1分别 **+0.0588 CI[−0.0147,+0.1393] / +0.0229 [−0.0307,+0.0770] / +0.0337 [−0.0139,+0.0796]**。A全部≤8 tokens；MSVD/NExT A仅各2/20视频改善，净省轮均来自EOS终止轮。NExT B decode-only有弱信号−0.8258 ms/token CI[−1.7542,−0.0154]，但τ及含prefill总耗时CI跨零、重复不稳且曾资源中断恢复，不能判稳定优势。暂不建议立即启动公平QA训练；唯一建议是先独立修复bonus EOS停止边界并无训练复核，未执行。三头SHA及生产代码未变。证据：[完整报告](/mnt/local/mcy/GLANCE/results/videoqa_transfer/20261013T134000/final_report.md)。 |
| **任务 A / P1：官方 KL 版 k3 被拒段目标（2026-10-15）** | **已完成** | 唯一改动 `--rejected-kl full_reverse|k3`（默认 `full_reverse`，历史逐位不变）。CPU：k3 与官方 `core_algos.py:2183-2191` **逐位相等**（18 边界点 + 4001 点网格；clamp 边界、`kl=0`、`kl=-11` 均钉住），`test_opd_fullkl.py` 19/19、`test_opd_train_fullkl.py` 17/17。上 GPU 前标度测量（6 视频 48 anchors）：`l_rej` 均值 full **6.1946** vs k3 **4.9673**（比 0.802），**24.6% 被拒 offset 顶在官方 ±10 clamp**、6.4% 触 ±20；被拒流仍占合 loss **94%**。冻结 10-update 对照（C0 起点、online 闭环、fp32 target + bf16 head + fp32 CPU 主权重重、lr 3e-5、24 anchors/update）：validation n=29 各 **30/30 无损**，mean τ 对照 **3.7300** / k3 **3.7480**，**Δ=+0.0180 CI[+0.0002,+0.0381] t +1.83（8/4 正向，17 条不变）**；k3 原始 `\|g\|` 只有对照的 0.22–0.42×，但 **clip 1.0 把两臂 applied norm 都归一**（对照 10/10 裁到 1.0，k3 9/10 亦为 1.0）⇒ 基本是干净的目标函数单变量比较。判读：**贴着分辨率的小幅正信号，不足以称“官方 KL 修好了 OPD”，也不构成加速结论**；P2/P3 未实现未跑。[报告](../../reports/2026-10-15-taskA-kl-official-p1.md)；产物 `results/taskA_kl_official_20261015T084308/` |
| **任务 B：视频 + 图像 QA 公平重做（2026-10-15）** | **已完成** | 预注册先写后跑（`prereg.json` + 一条明文 amend）。**视频**：seed20261014、冻结 480 槽日程（SHA 校验）、**两臂各 120/120 updates**、fp32 + **训练与评测同为 real 时间轴**、eos_stop=immediate、同起点 `e2526a24…`。真实消耗：SFT 20023 监督位置 / OPD 2880 anchors、N_acc 879、18/120 update 接受为 0。dev 20 视频 40 QA（B）：warm **3.4700** → QA-SFT **3.6395**（+0.1695 [+0.0901,+0.2472]，13/3/4）→ QA-OPD **1.2060**（−2.2640 [−2.4429,−2.0915]，**0/20 视频改善**、轮数 10→29）；三臂 **120/120 逐 token 无损**，95 行以 bonus EOS 停止。**图像**：被删的 `opd_pool_textvqa_2000.pt` 按原命令重建，**日志统计量逐项复现**（2000 行/72418 anchors/τ 3.438/443.1 MB）；三臂同起点 `checkpoints/head`、2000 行/1000 步/lr 1e-4、全部在本轮重测：warm **5.1990** → SFT **5.3676** → OPD-CE **5.3225**；SFT−warm +0.1463 [+0.0713,+0.2232]、OPD-CE−warm +0.1218 [+0.0570,+0.1830]、**SFT−OPD-CE +0.0245 CI 跨零**。**§B 的数据量混同已消除**（`head` 即三臂共同起点，无 16K 数字参与比较；基线步数更正为 1000）。全部计时标 `timing_contaminated`，不作速度证据。[报告](../../reports/2026-10-15-taskB-fair-qa-redo.md)；产物 `/mnt/local3/mcy/GLANCE/results/taskB_fair_qa_20261015T085742/` |
| **QA 训练—树验证语义/梯度诊断（2026-10-15）** | **已完成，无训练；发现 QA trainer 缺陷** | 3 train+3 dev，24固定当前AR anchors，8 heads×fp32/bf16共384局部trace；18生产首轮、36参数梯度门禁完成。接受索引错位、teacher偏一位、每anchor只有1 rejected监督；OPD120丢候选，teacher对齐后局部梯度明显更接近CE。历史QA负结果不能视为正确OPD的方法否定。详见[报告](../../reports/2026-10-15-qa-tree-semantics-diagnosis.md)；产物 `/mnt/local3/mcy/GLANCE/results/qa_tree_semantics_20261015/`。该诊断时点未修生产代码、未训练；后续第五轮修复与10-update已完成。 |
| **QA位置契约修复与10-update验证（2026-10-15）** | **已完成并收口** | QA-SFT-120共同起点，ZERO/SFT10/OPD10 τ=3.639499/3.677040/3.642907；未获得超过 continue-SFT 的可靠证据；保留位置契约正确性结果，不宣称方法优势，不继续训练。 240次SPEC=AR，原始局部候选/树与梯度/teacher预算齐备。[完整报告](../../reports/2026-10-15-qa-contract-fix-10update.md)。 |
| **ScienceQA 图像 SFT/OPD 公平对照（2026-10-15 派发，2026-10-16 完成）** | **已完成，普通负结果，配方关闭** | 三目录审计确认 train 只是官方 2400/6218 子集、**validation 完全缺失**、`data_sqa_full` = 子集+test；从 `lmms-lab-encoder/ScienceQA@69dd4b…` 补下载并冻结 train 1200 / dev 240 / test 240（跨集合同图簇 0、重复 0）。起点 `checkpoints/head`，预算由 300→**120 updates / 480 QA 曝光**（训练前登记）。dev I0 **4.0050** → SFT **4.3366/4.3394** → OPD **3.9415/3.9495**；test I0 **4.0723** → SFT **4.4059/4.4309** → OPD **4.0479/4.0583**。**OPD−SFT 四个组合全为负且 CI 不跨零（−0.358~−0.395）**，SFT−I0 全为正且 CI 不跨零（+0.332~+0.359）。全部逐 token 无损、截断/EOS 计数与目标答案准确率三臂一致。机制：OPD 的 `L_tree` 更低而树概率代理 Φ 更高（6.5 vs 3.5）。不切 k3/TV/P2/P3、不加轮次。[报告](../../reports/2026-10-15-scienceqa-image-sft-vs-opd.md)；产物 `/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955/`，脚本 `repro/scienceqa_image/`。 |
| **phaseC7 固定树预算对照** | **待执行，仅整合计划** | 固定C0/top-8/b63：best-first vs depth1–3各5、depth4–15各4的单一配额树；CPU重建/反事实通过后才做GPU验证与12条在线解码，不搜索第二组配额。见下方分阶段门禁。 |
| **D4 视频时序条件干预** | **未执行，有条件后续路线** | 待C7结果决定后续准备；先检查实现一致性，再验证target/draft条件响应差异，不能预设时序理解缺陷。 |
| Phase D / 新测试集 | 候选已冻结，未评测 | `results/phaseC/testset_candidates/`（A=20、B=47、C=67，带哈希）；同域新视频已耗尽（pool_499 即 VideoDetailCaption）；当前 heldout 不可作为 untouched test |
| EAGLE3 对比支线 | 已有阶段 0 准备记录 | 未核实到完整训练与最终对照结果，不能标记对比完成 |

本表描述已核实的文件状态，不代表后台任务的实时运行状态。第五轮QA修复与10-update实验已完成，见最新报告；下述10月13日闭环说明为历史阶段记录。2026-10-13视频QA迁移评测已按MSVD→MSRVTT→NExT-QA逐级门禁完成。本轮没有遗留后台评测或训练。最新用户授权的OPD在线闭环修复已完成代码、CPU及2视频无更新GPU门禁；未启动训练，完整训练显存/吞吐和修复后的收益尚未验证。EOS停止边界修复仍为独立评测建议，尚未启动。原C7 / ACL路线保留为有条件计划，不自动恢复；[视频 OPD 历史计划 §10](../../plans/video-opd.md#10-deepseek-长程执行指令2026-10-09-最新任务)用于追溯。旧计划未勾选项不代表需要继续训练。

## 当前诊断结论与适用范围

**2026-10-15 最新状态（QA 专用 trainer）：** 位置契约修复、无更新门禁及以QA-SFT-120为起点的两臂10-update实验均完成。未获得超过 continue-SFT 的可靠证据；保留位置契约正确性结果，不宣称方法优势，不继续训练。 当前仅保留证据，不启动C7/D4、新目标或额外训练。详见[完整报告](../../reports/2026-10-15-qa-contract-fix-10update.md)。第四轮诊断仍解释历史QA负结果受实现缺陷污染，但其“未修复/未训练”只描述当时状态。

**2026-10-13代码审计补充：当前C4 C2是旧reference轨迹上的当前策略proposal与在线teacher KL，不是严格当前target投机验证闭环。** 现有部署无损/QA评测数值不受此命名修正影响，但不能据此否定标准在线Draft-OPD；数据reference与接受分组须先修复并门禁验证，再谈后续公平训练。详见[代码审计](../../reports/2026-10-13-opd-online-implementation-audit.md)。**后续修复已让训练入口默认使用当前target在线闭环，小规模无更新门禁通过；这不会改变历史C4 checkpoint身份或补出训练收益证据。** 见[修复报告](../../reports/2026-10-13-opd-online-fix.md)。

研究目标是在冻结 target 贪心输出不变的前提下，降低视频生成的验证轮数与真实解码成本。当前停止扩展 full-KL OPD；**Phase C6及2026-10-13视频QA迁移评测均已完成；QA未形成可靠的额外接受长度/总耗时联合收益，截至该历史阶段没有启动C7、D4或训练**；此后第五轮QA位置修复和两臂10-update已完成，本轮未启动C7/D4或追加训练。三组B结果不支持把历史OPD负结果直接归因于描述任务更难；A受短长度与EOS边界限制，既有缓存时间编码也限制时序解释。以下保留2026-10-12 ACL研究路线；附件中的后续路线仍是计划，不是新增实测结果。实验数字以本地报告、manifest和原始指标为准。

- **C4：受控比较，不是只差 loss 的纯消融。** 三臂 validation τ 为 3.7377 / 3.7225 / 3.7403（n=29）；C2−C0=+0.0027，原报告 CI[−0.0203,+0.0274]，不采用备忘录中略有不同的近似区间。C1/C2 的实际曝光集合和监督位置预算不同；现有结果不支持可靠收益，也不能证明 OPD 普遍无效。依据：[C4 报告](../../reports/2026-10-12-phase-c4-frozen-10update.md)。
- **C5：轨迹来源与 teacher 条件是两个轴。** 旧 continuation 与当前部署 AR 69/70 分叉；58/1680 旧 anchors 的分组边界不同于同输入 teacher 的 leading agreement。这不是训练标签错误率，也不否定旧轨迹作为 teacher-forced/off-policy 输入的用途。首拒后 draft-induced 与正确 AR teacher 的差异本属正常。三臂完整 tree 各 70/70 SPEC=AR，跨臂 AR 相同。依据：[C5 报告](../../reports/2026-10-12-gain-stuck.md)。
- **C6：A 重标与 B-current 分开。** 同旧输入的 A-old/A-relabel 重算未改变原局部判断；B-current 在当前正确 AR 的固定 C0 状态评分，不能将 A/B 当作同上下文只改 reference 的因果对照。旧参考 loss 降低不等于当前 AR 下有效候选或接受长度改善，B-reference KL 也不是 C2 原生训练 loss。

C6 冻结6条 validation、48个真实 C0 轮次（44个有有效预测位置），144份三臂 trace 通过复算。下表为全部48轮的局部有效 draft-token 长度均值；以视频为统计单位，不能把轮次/token 当独立样本。

| 模型 | L_raw=L_candidate | L_tree=L_actual | mean tree_gap | commit_gap |
|---|---:|---:|---:|---:|
| C0 | 3.7292 | 2.8958 | 0.8333 | 0 |
| C1 | 3.6042 | 2.8333 | 0.7708 | 0 |
| C2 | 3.6458 | 2.8125 | 0.8333 | 0 |

C2−C0 的 ΔL_actual=−0.0833，视频级 CI[−0.2083,0]；0条增加、2条下降、4条不变，**本探针未显示改善，不等于证明等效**。C0 的42次非终止阻断中，24次原始 top-8 缺失（57.1%）、18次树遗漏（42.9%）；另6次正常终止/边界。Σtree_gap=40，18轮为正、最大5。**支持候选缺失与树遗漏并存；本探针未显示提交缺陷。** 这40个 token 是离线局部空间，不是可保证回收的收益，不能直接换算全局 τ 或速度。24比18多也不能决定下一步投入的成本效益，更不能预先断定 suffix 梯度无用。

C5 B-reference 支线使用前8轮，C6 使用跨完整轮次的等距抽样；两者样本与轨迹条件不同，不混池、不混用 chain 数字。当前 C7 仅复用 C6。详细证据、脚本、失败记录、额外调用量见 [C6 报告](../../reports/2026-10-12-current-ar-fixed-anchor.md) 与 `results/phaseC6/current_ar_closed_loop_20261012T221104_gpu_scores/delivery_manifest.json`。不再把全面追溯旧轨迹分叉原因列为当前实验的必需前置。

## 下一步：Phase C7 固定树预算对照（待执行）

**C7原定实验（待执行，视频QA收口后不自动恢复）：固定 C0、真实 GPU top-8 与63节点验证预算，检验一次预先固定的深度配额方案。** 本轮不启动该实验，不训练、不延长 OPD、不打开冻结新测试集。

当前模型身份固定为 `checkpoints/head_video439_aligned/step1000`（C0）、`checkpoints/phaseC4_C1/step10`（C1）、`checkpoints/phaseC4_C2/step10`（C2）；不得用 phaseC3 代替 C1。运行前复核实际路径、哈希、环境与空闲资源。

| 方案 | 构造方式 | 不变因素 |
|---|---|---|
| A：现有 best-first | 现有累计 draft 路径概率排序、前缀闭合预算选择 | C0、原始候选/分数、root/prefix、target、预算63（不含root） |
| B：固定深度配额 | depth1–3各5节点，depth4–15各4节点；逐层仅扩展保留父节点，按同一累计概率取配额 | 同 A；不使用 AR reference 选节点，不新增候选 |

配额总额 `3×5+12×4=63`；EOS、合法扩展不足、有效深度限制下实际节点数可能较少，必须记录。**不搜索第二套配额，不把失败变成调参起点。**

同时记录 `Φ(T)=Σ_{node∈T} exp(Σ_{path} log p_d)`。若 A 确实在同候选、同前缀闭合预算下最优化该 draft 概率代理，则 B 不应在 Φ 上优于 A；真实路径改善只能提示代理排序与 target 接受行为可能失配，不能称为 draft 概率目标更优。A 未选中事后正确路径本身不等于构树 bug，也尚未证明视频特有机制。

### 分阶段门禁与最小计算规模

| 阶段 | 执行内容 | 继续/停止条件 |
|---|---|---|
| Stage 0：CPU 重放 | 仅复用 C6 冻结6视频/48轮及捕获的GPU候选，精确重建 A 的节点、parent/depth、前缀闭合与长度汇总 | 必须复现 Σtree_gap=40、mean L_tree=2.895833、commit_gap=0；不能复现先定位，停止有效 A/B 比较 |
| Stage 1：CPU 反事实 | 同48轮构建 B，记录逐视频/anchor 的 ΔL_tree、毛回收、路径损失、净和、首遗漏深度、早期2–4 token覆盖、Φ_A/Φ_B | 仅总净增量>0、收益不完全由单一anchor/视频撑起且无严重早期覆盖损失时进入GPU；否则收口，不改配额。执行前冻结“严重损失”的判据，不能看结果后定门槛 |
| Stage 2：GPU 局部验证 | 仅前关通过后，最多48次 B 树 target 验证；核对 ancestor mask、节点argmax、路径/bonus、EOS/裁剪及 L_actual_B=L_tree_B | 正确AR路径逐token门禁通过；不一致先定位，停止收益解释；trace耗时不作为速度 |
| Stage 3：真实在线 A/B | 仅局部验证通过后，同一空闲GPU/精度/warmup/计时规则下，两树各完整解码6视频，共12条任务 | 逐token SPEC=AR；报告各自在线τ、rounds、ms/token、构树与verify成本，可比时报告端到端延迟；负载不一致不得给加速结论 |

C6 首版追踪器曾因 CPU 重新 log_softmax 导致树重建不一致，修复为捕获实际 GPU 候选及分数。C7 必须沿用原始捕获数据，不以 CPU 新 top-8 冒充生产候选；tie-breaking 或数值差异需显式核查。失败/作废 pilot 独立留痕，不拼入正式统计。C7 不修改生产解码文件、不改 C1/C2、不使用未来 target 答案构树、不修改共享依赖或他人任务。

最小交付：manifest、复现脚本与必要测试、命令/日志、逐视频及逐anchor表。表至少含 `L_raw, L_tree_A, L_tree_B, ΔL_tree, Φ_A, Φ_B, first_missing_depth, 回收/损失路径`；进入GPU后加 L_actual/门禁，进入在线后加 τ、rounds、ms/token、tree build/verify cost、SPEC=AR。未进入阶段填 N/A 并说明原因。

### 停止规则与后续分支

- CPU 无净局部收益：停止此固定配额方案，转入候选缺失与 D4 的研究准备；不代表所有树选择方法都无空间。
- 局部改善但验证或在线接受长度未改善：定位早期路径损失、轨迹变化、开销或实现不一致，不继续调配额；实现门禁失败时先处理阻塞，不扩展 D4。
- 在线 τ 改善但可比时延/吞吐无改善：只保留机制证据，不能称为加速方法。
- 在线接受长度与可比速度均改善：进入下述路线 A 的机制研究及强基线比较，仍不能宣称 ACL 创新已成立。

## ACL 研究路线与证据门槛（有条件计划）

### 路线 A：草稿概率代理与真实接受路径失配

触发条件是 C7 获得可重复的受控在线接受收益，最好同时有真实速度收益。待检验问题：差异能否由候选校准、位置、熵或视觉依赖等特征预测，而非固定配额偶然有利？可考虑低开销预算分配或候选分数校准；部署只能使用当时可获得的信息，诊断用 target 熵等量若部署获取需额外前向，必须计入成本，不能偷用未来答案。

证据链：**候选不变 → 节点分配改变 → 正确 AR 前缀进入树 → 在线无损验证 → wall-clock 下降**。与动态树/代价感知树方法比较；DDTree、OPT-Tree、CaDDTree、CAST 仅为待核查文献清单，不据此断言发表状态或新颖性归属。

### 路线 B：视频时序条件响应差异（D4 未执行）

触发条件是 C7 无可靠在线收益，或进一步证据指向候选预测缺失。**“视频事件变化引起 target 偏好变化，而 draft 响应不足”仅是假设**，不是已证明的 grounding/时序理解缺陷，也不能自动解释 phaseC4 的结果。

D4 首轮最多6条合格训练/validation视频、每条最多4个有效共享prefix，提前冻结选择规则。选有候选缺失且事件可解释的视频，构造原视频、经核验的时序干预，以及保持关键事件顺序的控制扰动：

1. 每种视频条件重新生成 target context 与融合特征，核验帧顺序、时间编码、视觉token对齐；不能复用原视频缓存，不能把缓存16帧的封装时间当原视频真实时间轴。
2. 使用两个条件下仍然成立的同一文本prefix、offset与mask；矛盾prefix排除或单列。先检查第一个实际由draft预测的位置，记录 target/draft 分布变化、target greedy是否改变、各条件正确token的log-prob/rank/实际top-k、entropy、margin与offset。
3. 比较时序干预与控制扰动，控制target熵、预测位置、上下文长度与原始难度，避免把一般不确定性误称时序依赖。
4. 无事实标注/人工核验，只能讨论 target/draft 响应差异，不能宣称 target 理解正确、draft 理解错误。若先发现明确实现不一致，停止扩展干预。

仅在稳定缺口被证实后才考虑成对视频条件下的相对偏好蒸馏或小规模时序证据接口；须与普通配对数据 SFT、困难样本采样、首拒采样作等预算对照，核对真实曝光、监督位置与权重。当前不启动这些训练，不先指定 loss 再挑支持样本。

### 共同投稿门槛与论文组织

- **新颖性与可反驳机制**：OPD换KL、固定深度配额、普通自适应树、把图像消融搬到视频，均不足以自动构成创新。先有机制及反例，再设计方法；6视频探针需在新的开发数据复核。
- **强基线与公平成本**：合理优化的 continue-SFT、相关 OPD 变体及树方法；报告真实监督量、teacher计算、额外前向和推理开销。不能用不同视频曝光或位置预算冒充只差目标的消融。
- **无损与真实速度**：冻结target、AR逐token一致，分开在线τ、decode-only与端到端；重型trace及不同负载的历史耗时不作正式速度证据。
- **泛化与统计**：n=29是已用于开发的validation；“本轮未曝光”不等于“从未训练”。方法、基线和停止规则冻结后才打开独立测试；以视频为主统计单位，训练比较考虑多个seed。不显著不等于等效，不把局部oracle换算全局收益。
- **论文结构（路线成熟后）**：研究问题 → 问题/成本定义 → 可重复诊断 → 针对性方法 → 因果消融与强基线 → 无损在线收益及局限。区分target理解准确性与draft接受性；如实报告样本、时间轴、硬件负载、跨域覆盖。两条待验证路线不能合并写成已成立机制。

## 固定的实验语义

- **优化器硬前提**：若后续获准训练，lr ≲ 1e-4 必须使用 `--master-weights cpu|gpu`，统一两臂实现与teacher/context精度，实测显存并审计优化器恢复；本轮不训练。历史 bf16 AdamW 无主权重存在更新舍入；lr3e-4 首步范数接近理想不代表方向无误差（余弦0.792），不能由此证明修复后排序必然不变。
- **在线训练语义**：`opd_train_video_fullkl.py`默认`online`，当前target逐token验证/提交、推进KV与融合特征，重放核对teacher/student argmax及接受分组；不使用旧continuation。旧checkpoint及显式`legacy-reference`仍属于历史混合语义。固定anchors只用于诊断。严格在线loss仍是full-KL变体，非官方sampled/TV公式；未来优化数值路径须重过AR及重放门禁。
- **局部长度语义**：`L_raw/L_tree/L_actual` 排除root和bonus，按EOS/输出上限/block深度裁剪；当前15个独立offset、预算63下 `L_candidate=L_raw`。固定C0状态的局部ΔL不是各臂在线Δτ，历史τ可能含accepted+1及裁剪前计数。

- **特征路径**：aligned SFT 显式传 `--feature-path cached`；`train_head_video.py` 默认仍是 `uncached`，直接使用默认值会回到旧路径。cached 与 aligned context 在既有门禁中逐位一致；旧路径曾出现文本 hidden 不依赖视频的现象。
- **B1 损失选择**：按[公式核实记录](../../reports/2026-10-09-b1-formula.md)采用论文语义。接受位置为全词表 forward KL 的无衰减均值；整段被拒后缀为全词表 reverse KL，以 `0.8^(k-1)` 加权并按权重和归一；两流存在时等权平均。单流为空时按剩余流重归一，两流皆空时贡献零。不能简单平均不同有效位置数的 microbatch 均值。
- **论文与官方实现有差异**：采样 token 估计量、池化归一、top-k 重归一化都不能与上述全词表、两流分别归一的目标混称；常数学习率等偏离应显式记录。
- **条件前缀**：第一个 rejected 位置之前仍是匹配前缀；draft-induced prefix 的差异从后续位置出现，不能要求第一处 rejection 的 prefix 不同。
- **精度与无损**：teacher 前向比对允许明确的数值容差；部署解码与 AR 的 token 门禁要求精确一致。bf16 输出分叉时不能据其 τ 宣称速度收益。
- **数据纪律**：`video_fresh30` 已用于选点，只是 validation；历史 `video_heldout_30` 与 pool_499 train 重叠，现有 60 条 heldout 均非严格 untouched test。prompt 以池内 `question` 为准，split 与 token 对齐必须审计。
- **指标纪律**：区分 pooled τ、mean per-video τ、全长度、前 60 rounds、decode-only 与端到端计时。提高 τ 不等于提高实际速度。

## 目录与入口

| 路径 | 职责 |
|---|---|
| `GLANCE_repo/glance/` | 本地 GLANCE 库：tree、decode、head、vl、metrics、law；包含本地扩展，不能当作可随意覆盖的纯上游副本 |
| `repro/` | 数据构建、训练、评测、诊断与实验启动脚本；见[运行说明](../../../repro/README.md) |
| `repro/common.py` / `aligned_forward.py` | 路径、模型加载、context 与前向对齐 |
| `repro/opd_online_rollout.py` / `opd_train_video.py` | 旧 2×2 在线 rollout 与训练入口，保留历史行为 |
| `repro/opd_fullkl.py` / `proto_fullkl_b0.py` | 新 full-KL 模块及 teacher prefix-KV 原型 |
| `repro/eagle3/` | EAGLE3 特征采集、补丁、训练启动与验证 |
| `docs/` | [文档索引](../../README.md)：计划、实验报告、文献笔记和可恢复归档 |
| `scripts/dflash_pkg/`、`scripts/head_ref/` | 本地依赖与参考实现 |
| `glance-2609.00355.pdf`、`glance.txt` | 原论文及提取文本，保留原路径供检索 |

代码根目录：`/home/mcy/Projects/GLANCE`。产物根目录默认 `/mnt/local/mcy/GLANCE`，可由 `common.py` 的 `GLANCE_ROOT` 环境变量覆盖；部分 shell 脚本仍硬编码路径，运行前检查。产物按 `models/`、`datasets/`、`data/`、`checkpoints/`、`results/`、`logs/`、`scripts/` 分开。

常用现有环境为 `/home/mcy/.conda/envs/fin_research/bin/python`。项目根及 `GLANCE_repo/` 当前都不是 Git 工作树；不能依靠 Git 撤销更改。

## 文献检索入口（2026-10-09）

已完成一轮 web search，整理 [68 篇优化文献与 PDF](../../research/2026-10-09-opd-multimodal/README.md)，含[针对当前实验的借鉴建议](../../research/2026-10-09-opd-multimodal/experiment_recommendations.md)。来源和会议身份分级、版本核对及方法限制均在文献库内。

需注意：本地 GLANCE 原论文是 v1，在线已更新至 v3（2026-10-06）；Draft-OPD 官方有 TV loss + first-reject-only 扩展，与本轮 paper-full-KL 目标不同。新增方法均为待验证候选，未改现有 Phase C 配置，也未运行新实验。

## 后续维护约定

- `docs/reports/` 的报告统一命名为 `YYYY-MM-DD-主题.md`，日期沿用报告日期；同日按主题排列。新增或改名时同步维护[报告索引](../../reports/README.md)和项目内引用，历史产物盘及归档快照保持原样。

- 每次完成实验或改变判断，更新本文的日期、状态、下一步和证据路径；不要再新增平行的“项目总结/精简版/最新进展”。
- 只有日志、配置、结果和门禁齐全才标“完成”；存在脚本、模块或 checkpoint 不等于验证通过。记录失败、作废运行及阻塞原因。
- 当前状态集中在本文；详细计划维护 `docs/plans/video-opd.md`；阶段报告保留实验时点的原始事实，修正用明确补注说明。
- 新报告写明数据划分、起点/哈希、seed、精度、采样数与 warmup、指标聚合、命令和结果路径。若覆盖旧结论，指明旧结论失效的范围。
- 不删除历史 checkpoint/原始结果；不原地改写运行中的 shell 脚本。重跑用独立输出目录，防止覆盖证据。
- 重型任务默认串行，只用满足显存要求的空闲 GPU；优先使用现有环境，不修改共享依赖或他人进程。

## ★ 2026-10-15：KL 口径核对结论 + 图像 QA 对比一致性审计

### A. KL 口径：**三条轴要分清；其中两条与官方代码不同，需按官方改动**

> ⚠️ **本节曾于 2026-10-15 初稿写成"GLANCE 的 KL 不需要改"，那是错的。**
> 初稿只核了"位置衰减"这一条轴，就下了"KL 无需改动"的结论，**漏掉了另两条轴**。
> 用户指出后重核并改正。**下面三条轴各自独立，不得合并成一句话。**

#### A0. ★★ 基准：**保留 KL 目标，取 KL 版里最新的入口**

用户指示（2026-10-15，两轮）：
① "官方哪个入口新就用哪个"；② **"先不要 TV loss，保留 KL 的最新的"**。
⇒ 基准 = **KL 目标 + KL 版里最后提交的那个入口**。

按 git 取证（`cd /home/mcy/Projects/Draft-OPD && git log -1 --format='%ad' --date=short -- verl/examples/...`，
全部在 `main` 工作树）：

| 入口脚本 | 最后提交 | 提交 | 是否 KL 版 | `REJECTED_DRAFT_POSITION_DECAY` | `rejected_draft_stream_weight` | `FIRST_TOKEN_ONLY` |
|---|---|---|---|---|---|---|
| `run_qwen_gsm8k_forward-8b-local.sh` | 2026-09-04 | `c9b4896` | ❌ **TV 版**（`USE_TV_LOSS=True`） | 0.9 | — | True |
| `run_qwen_eagle3.sh` | 2026-09-04 | `a737810` | KL（EAGLE3 专用） | 1.0 | **0.0** | — |
| **★ `run_qwen_gsm8k.sh`（基础）** | **2026-08-08** | **`d1476f5`** | ✅ **KL，DFlash** | **0.9** | **2.0** | **False** |
| `run_qwen-30b-a3b.sh` | 2026-06-12 | `1cbbac0` | KL | — | — | — |
| `run_qwen_gsm8k_forward-ins.sh`（README 入口） | 2026-05-31 | `b89cc7e` | KL，DFlash | 0.8 | 1.0 | False |

⇒ **基准 = `run_qwen_gsm8k.sh`（`d1476f5`, 2026-08-08）**：
* 它是**最新提交的、面向 DFlash 的 KL 版入口**
  （`eagle3.sh` 虽同为 09-04，但是 **EAGLE3 架构**、`rejected_draft_stream_weight=0.0`，
  不是同一套 drafter，故不作基准）。
* **注意与 README 入口的差别**：README 指向的 `forward-ins.sh` 是
  **5 月（最旧）**，且默认 `rejected_draft_stream_weight=1.0 / decay=0.8`；
  而最新的 KL 版用 **`2.0` / `0.9`**。**两者不是同一套超参，不要混引。**
* 该脚本显式设 `REJECTED_DRAFT_POSITION_DECAY_ENABLED=True`、`DECAY=0.9`，
  且**不设** `ACCEPT_DRAFT_POSITION_DECAY_ENABLED` ⇒ 默认 `False`（接受段无衰减）。

> **TV loss 已按用户指示排除**（`-8b-local.sh` / `USE_TV_LOSS` 路线不做）。
> 下表 §A0.1 的 TV 公式**保留作存档**，仅用于解释"官方还有另一条更强但不同的路线"，
> **不作为本任务的基准**。

#### A0.0 官方 KL 版目标（本任务基准，逐字核对）

来自 `verl/verl/trainer/distillation/losses.py` + `verl/verl/trainer/ppo/core_algos.py`：

| 项 | 官方 KL 版实现 | 依据 |
|---|---|---|
| 接受段 | `_local_bernoulli_forward_kl`：把分布**二值化**为 {该 token, 非该 token} 的 forward KL；log-prob clamp `min=−80`、`max=log1p(−eps)` | `losses.py:432-453` |
| 被拒段 | `kl_penalty(..., "k3")`：`kl=clamp(ref_logprob−logprob, −20, 20)`；`ratio=exp(kl)`；`clamp(ratio−kl−1, −10, 10)` | `core_algos.py:2183-2191` |
| 被拒段范围 | **整段被拒后缀**（`rejected_draft_first_token_only` 默认 **False**，`-8b-local.sh` 才设 True） | `ray_trainer.py:1415` |
| 位置衰减 | 接受段**无**（`accept_draft_position_decay_enabled` 默认 False）；被拒段 `γ^(k−1)`，**基线脚本取 0.9** | `losses.py:532-593` |
| 两流合并 | **池化加权均值** `(w_r·Σ_r + w_j·Σ_j)/(w_r·N_r + w_j·N_j^eff)`，`w_r=stream_weight=1.0`、`w_j=rejected_draft_stream_weight=2.0`（基线脚本） | `losses.py:854-888` |
| 空流 | 池化式天然处理；两流皆空 ⇒ 零 loss 且在图上 | `losses.py:884-890` |

#### A0.1 【存档·不作基准】官方 TV loss 目标（用户已指示不做）


`verl/verl/models/transformers/dflash_student.py`：

```python
# :40-53  全词表 TV 距离（target 被 detach，梯度只走 draft）
draft_probs  = F.softmax(draft_logits.float(), dim=-1)
target_probs = F.softmax(target_logits.detach().float(), dim=-1)
return 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)

# :1349-1367  每个 block 的"期望接受长度"与损失
cumulative_overlap = torch.cumprod((1.0 - block_tv).clamp(0.0, 1.0), dim=0)
expected_accept_length = cumulative_overlap.sum()
block_losses.append(1.0 - expected_accept_length / float(block_tv.numel()))
```

* 这是**端到端**目标：`L_block = 1 − E[acc]/K`，**直接优化期望接受长度**，
  而不是在某个 teacher 分布上做 KL；`_exact_tv_distillation_loss` 以
  `Σ block_losses / Σ block_mask` 聚合（`losses.py:626-637`）。
* 目标侧 logits 在 **draft-induced prefix** 上现算且 `no_grad`（`:1338-1341`）；
  梯度检查点用 `torch.utils.checkpoint`（`:1336-1340`）。
* **`response_anchor_stride=1`**（`:1512-1516` 原文注释："Exact TV is defined on every
  rollout speculative block. Sampling or striding anchors would change that objective."）
  ⇒ 不能子采样 anchor，否则改的是目标本身。
* `first_reject_only=True` 时，原 KL 版里那条"被拒后缀"流根本不再参与
  （`rejected_draft_use_reverse_kl` 那条路径与 TV 分支互斥）。

> **更正我先前的判断**：我曾在 §7.1 写"TV loss 依赖 6 个私有组件 ⇒ 整体复刻成本高、
> 不建议"。读了新入口后要修正为：**成本主要来自**（a）需要在 Python 里复现
> `_build_tv_position_plan` / teacher-branch hidden 那套位置计划，
> （b）`response_anchor_stride=1` 使监督位置数远大于现在的 anchor 子采样，
> **显存与耗时都会明显上升**。**但公式本身很简单**（上面两段就是全部），
> 所以"不能复刻"不成立，只是"要按新预算重做"。

#### A1'. 那 GLANCE 与"最新 KL 版基准"差在哪

基准 = `run_qwen_gsm8k.sh`（`d1476f5`）。**下方 §A2+ 的"三条轴"即本表展开**：

| 轴 | 官方 KL 版基准 | GLANCE 现状 | 差 |
|---|---|---|---|
| **① 接受段 KL** | Bernoulli forward KL（`losses.py:432`） | 全词表 forward KL（`opd_fullkl.py:424`） | ❌ **不同** |
| **② 被拒段 KL** | k3 单样本（`core_algos.py:2183-2191`） | 全词表 reverse KL（`opd_fullkl.py:436`） | ❌ **不同** |
| **③ 位置衰减** | 接受段无衰减；被拒段 `γ^(k−1)`，脚本取 **0.9** | 接受段无衰减；被拒段 `γ=0.8` | 衰减**机制一致**；数值 0.8 vs 0.9 属超参差 |
| **④ 两流合并** | 池化加权均值，`w_j=2.0`（`losses.py:854-888`） | 论文 Eq 13，等权 | ❌ 不同 |
| 被拒段范围 | 整段后缀（`FIRST_TOKEN_ONLY` 默认 False） | 整段后缀 | ✅ 一致 |

> 注意：GLANCE 的 `γ=0.8` 来自**论文 Appendix A**；官方最新 KL 版脚本用 **0.9**。
> 这属超参差异，**不要**与轴 ③ 的"机制一致性"混为一谈。

起因：Draft-OPD 侧（`opd_multimodal`）2026-10-15 修掉一处位置衰减偏差 —— 官方
`accept_draft_position_decay_enabled` **默认 False**（接受段**无** position decay），
而该侧此前把同一个 `decay` 也乘到了接受段上。以此为起点逐条核对了 GLANCE 与官方的**三条独立轴**。

| 轴 | Draft-OPD 官方代码 | GLANCE 现状 | 一致？ |
|---|---|---|---|
| **① 接受段 forward KL 形式** | `_local_bernoulli_forward_kl`（`losses.py:432`）——把分布**二值化**为 {该 token, 非该 token} 的 Bernoulli forward KL | `full_kl_forward`（`opd_fullkl.py:424`）——**全词表** `Σ p(log p − log q)` | ❌ **不同** |
| **② 被拒段 reverse KL 形式** | `kl_penalty(..., kl_penalty="k3")`（`core_algos.py:2183-2191`）——**k3 单样本估计量** `r − log r − 1`（`r = p/q`），`kl` clamp ±20、`kld` clamp ±10；**只需采样 token 的 logprob** | `full_kl_reverse`（`opd_fullkl.py:436`）——**全词表** `Σ q(log q − log p)` | ❌ **不同** |
| **③ 位置衰减（position decay）** | 接受段**无**衰减；被拒段 `w_k = gamma**(k-1)`，入口默认 `0.8`（`losses.py:532-561`） | 接受段 plain mean 无衰减；被拒段 `w_k = gamma**(k-1)`，`gamma=0.8` | ✅ **一致** |

另有一条**合并方式**的差异（属①/②的后果，单列以免混进上面三轴）：

| 轴 | 官方代码 | GLANCE 现状 |
|---|---|---|
| **④ 两流合并** | **池化加权均值** `(w_r·Σ_r + w_j·Σ_j)/(w_r·N_r + w_j·N_j^eff)`（`losses.py:854-888`），与跨 DP rank 的 `_global_sum` 计数归约耦合 | 论文 Eq 13：**两流各自归一后按 λ 加权平均** → `(λ_acc·L_acc + λ_rej·L_rej)/(λ_acc+λ_rej)`，λ=1,1 ⇒ `(L_acc+L_rej)/2`（`opd_fullkl.py:563-564`） |

**③ 一致的依据**（这部分初稿是对的）：`opd_fullkl.py` 里接受段是
`full_kl_forward(...).mean()`（plain mean、无权重），只有被拒段用 `gamma**(k-1)`；
该模块 docstring（文件头 22–36 行）明确记录：原型 `proto_fullkl_b0.weighted_stream_mean`
当初"把 `w_k` 用到两条流上"是 pre-verification guess，**已被纠正**；
`repro/test_opd_train_fullkl.py` 有 `gamma=1.0 (no decay)` 一档断言。
**注意：这也正是 Draft-OPD 侧同期修掉的同一个坑。**

**为什么 GLANCE 现在长这样（不是随手写的）**：`docs/reports/2026-10-09-b1-formula.md`
是逐条对过 **arXiv 2605.29343 原文（Eq 8–13、Appendix B）** 与官方 `losses.py` 的核实记录，
结论是 **GLANCE 显式选择了"论文语义"**：论文 Eq 10/11 与 App. B 明确写
`D_KL(p‖q) = Σ_{y∈V} …`、`D_KL(q‖p) = Σ_{y∈V} …`，**求和范围是完整词表 V**；
而官方代码的默认路径是"采样 token 估计量"（verl 集成侧只拿到采样 token logprob 的近似）。
旁证：官方 `core_algos.py` 的 `if kl_penalty == "full":` 分支是 **`raise NotImplementedError`**
—— 官方**没有**实现全词表版本。

⇒ **准确表述**：**论文的公式与官方代码的默认实现不是同一个目标函数**；
GLANCE 按论文实现了全词表版本。
**不宜**说成"官方代码错了"或"官方与论文矛盾"——那更像是未写进论文的集成近似
（池化归一与 verl 的分布式计数归约耦合）。也**没有**看到官方作者本人承认过此差异。

⇒ **当前状态**：轴 ③ 一致；轴 ①②④ 与官方代码不同。**按用户 2026-10-15 指示，
接下来要"按官方代码的实现去改动"** —— 见下方新任务 §（B）。

### B. ★ 图像 QA 对比一致性审计（逐条取证，含一处真实混同）

**纪律背景**：做 OPD vs SFT 对照时，**两臂的训练数据量/步数/lr/起点必须逐项一致**，
否则测的是数据量而不是方法。以下为对 2026-09-23/24 图像 OPD 实验的逐条核验
（证据全部来自 `/mnt/local/mcy/GLANCE/logs/` 的原始训练日志）。

| 臂 | 日志 | 数据行数（日志原文） | steps | lr | 结果 |
|---|---|---|---|---|---|
| **基线 GLANCE head** | `11_train.log` | `row filter: 16000 -> 16000`，**8000 rows/rank × 2 rank** | 1293 optimizer steps（1 epoch = 1000） | — | ChartQA τ **5.199** |
| **对照：继续 SFT** `head_sft_control` | `41_train_control.log` | **2000** | 1000 | 1e-4 | τ 5.365 |
| 同上 seed1 / seed2 | `47_control_s1/s2.log` | **2000** | 1000 | 1e-4 | 5.352 / 5.350 |
| **OPD**（各变体） | `42/43/44/45/46/48_*.log` | pool `opd_pool_textvqa_2000.pt`：**2000 rows** | 1000 | 1e-4 | 5.02–5.32 |

**核验结论（分三层，务必按此引用）**：

1. ✅ **对照 vs OPD 这一组本身是公平的**：两侧都是 **2000 行 / 1000 步 / lr 1e-4**，
   而且 `41_train_control.log` 与 `42_train_opd.log` 的启动时间戳
   **完全相同**（`2026-09-23 20:53:15`），说明是同一批次、同一起点并行跑的。
   ⇒ 报告 §6 里"两组同 2000 行数据"的说法**成立**，
   **"SFT 训 8000 条而 OPD 只训 2000 条"在 control-vs-OPD 这对里并未发生。**
2. ❌ **但"与基线 head 比"是混同的（真实问题在这里）**：
   基线 `head` 训的是 **16000 行（8000 COCO + 8000 TextVQA）、1293 步**，
   而 OPD 实验的对照只训 **2000 行 TextVQA、1000 步**。
   所以拿 OPD 臂去比 **τ 5.199**（**基线**的公开数字）**同时混了"数据量 + 训练量 +
   任务混合"三件事**。`RESULTS_OPD_on_GLANCE_2026-09-24.md` 有一句
   "池只有 2000 行、1000 步（论文 16K prompt × 8 epoch）；但对照臂用的是同样规模" ——
   它只论证了 control↔OPD 可比，**没有**论证"OPD 臂 ↔ τ 5.199 可比**。引用时不得把
   `head_sft_control`(2000行) 与 `head`(16000行) 当成同一档基线。**
3. ⚠️ **口径混杂**：`head_sft_control`(τ 5.365) 反而**高于**基线 `head`(5.199)，
   但两者数据量差 8×、训练量差 ~1.3×、数据混合也不同（TextVQA-only vs COCO+TextVQA）。
   该差值**不能**被解读为"TextVQA-only 更好"，因为从未做过等数据量/等训练量的对照。

**⇒ 因此图像 QA 的结论只允许这样表述**：
> "在 **2000 行 / 1000 步 / lr 1e-4 / 同起点** 这一等预算条件下，
> 继续 SFT（5.355±0.008, 3 seed）优于所有 OPD 变体（最好 5.295, 3 seed, t=−7.49），
> OPD 稳定差约 0.06；两者都高于同口径的 warm start。"
> **不得**表述为"OPD 比 GLANCE 基线差"或"继续 SFT 比基线好"——那三对量不可比。

---

## ★ 新任务 A（**已于 2026-10-15 执行完毕**，见[报告](../../reports/2026-10-15-taskA-kl-official-p1.md)；下方为派发时的原始约束，保留供追溯）：把 KL 目标改成**官方最新 KL 版代码的实现**

> 授权日期 2026-10-15。用户两轮指示：①"官方哪个入口新就用哪个"；
> ② **"先不要 TV loss，保留 KL 的最新的"**。
> ⇒ **基准 = `run_qwen_gsm8k.sh`（`d1476f5`, 2026-08-08）= KL 版里最新提交的 DFlash 入口**
> （取证见 §A0；**TV 路线 `-8b-local.sh` 明确不做**）。
> **先读本节全部约束再动手。**

### A0. 基准与优先级（**先记住这条**）

**基准脚本**：`run_qwen_gsm8k.sh`，默认值（**这些是本次要对齐的数**）：

```
DISTILLATION_LOSS_MODE=k3
REVERSE_KL_WEIGHT=0.0          FORWARD_KL_WEIGHT=1.0
REJECTED_DRAFT_USE_REVERSE_KL=True
REJECTED_DRAFT_POSITION_DECAY_ENABLED=True   REJECTED_DRAFT_POSITION_DECAY=0.9
（不设 ACCEPT_DRAFT_POSITION_DECAY_ENABLED ⇒ 默认 False）
stream_weight=1.0              rejected_draft_stream_weight=2.0
（不设 FIRST_TOKEN_ONLY ⇒ 默认 False = 整段被拒后缀）
```

| 优先级 | 改什么 | 依据 |
|---|---|---|
| **P1（第一步，只改这一处）** | **被拒段：全词表 reverse KL → k3 单样本估计量** | `core_algos.py:2183-2191` |
| P2（第二步，单独一臂） | **接受段：全词表 forward KL → Bernoulli forward KL** | `losses.py:432-453` |
| P3（第三步，单独一臂） | **两流合并：Eq 13 → 池化加权均值**，`w_j` 1.0 → **2.0** | `losses.py:854-888`；`run_qwen_gsm8k.sh` |
| —（**不动**） | 位置衰减（接受段无衰减；被拒段 `γ^(k−1)`） | 轴 ③ 已与官方一致 |

> ⚠️ **`rejected_draft_stream_weight` 是 2.0（不是 1.0）**。
> README 指向的 `forward-ins.sh` 用 1.0/decay 0.8，**那是 5 月的最旧版本**，
> 不要拿它当真值。P3 里把这个 2.0 一起对齐。

### A1. P1–P3 的逐项细节（逐字核对）

以 /home/mcy/Projects/Draft-OPD/refs/Draft-OPD（HEAD d6f0ec8）
的 `verl/verl/trainer/distillation/losses.py` + `verl/verl/trainer/ppo/core_algos.py` 为准：

| # | 项 | 现在（论文语义） | **改成（官方 KL 版）** | 官方依据 |
|---|---|---|---|---|
| 1 | **接受段 KL** | 全词表 `Σ p(log p − log q)` | **Bernoulli forward KL**：`tp·(log tp − log sq) + (1−tp)·(log(1−tp) − log(1−sq))` | `losses.py:432-453`（clamp：`min=−80`、`max=log1p(−eps)`） |
| 2 | **被拒段 KL** | 全词表 `Σ q(log q − log p)` | **k3 单样本**：`kl = clamp(ref−logp, −20, 20)`；`ratio=exp(kl)`；`clamp(ratio−kl−1, −10, 10)` | `core_algos.py:2183-2191` |
| 3 | 被拒段范围 | 整段后缀 | 整段后缀（`first_token_only` 默认 False） | 一致 |
| 4 | 位置衰减 | `w_k = γ^(k−1)`，只乘被拒流（γ=0.8） | **不变** | `losses.py:532-561`；轴 ③ 已一致 |
| 5 | **两流合并** | 论文 Eq 13：各自归一后 `(L_acc+L_rej)/2` | 池化加权均值（同 P4） | `losses.py:854-888` |
| 6 | 空流处理 | 丢弃空流、按剩余 λ 重归一 | 池化式天然处理；保留"两流皆空 ⇒ 零 loss 但在图上" | `losses.py:884-890` |

### A2. ★ 单变量纪律（关键，别一次改完就下结论）

P1/P2/P3 是**三个独立改动**，必须能分开归因：

* **第一步只改 P1（被拒段：全词表 reverse KL → k3 单样本）**，
  接受段与合并方式保持不动。理由：先前诊断
  （`docs/reports/2026-10-13-opd-online-implementation-audit.md`、
  `reports/FINAL_QA_SFT_VS_OPD.md`）反复指向"**loss 由 rejected reverse-KL 主导**"
  （QA-OPD seed1 首拒深度 720/720 为 0、accepted 流几乎为空）⇒ 被拒段是主要嫌疑。
* **P2（接受段 Bernoulli）与 P3（池化合并 + `w_j=2.0`）各自单独作为后续单变量臂**，
  **不得与 P1 同一次改动一起下结论**。
* 每个臂**只改一处**，其余（数据/起点/updates/lr/精度/EOS/时间轴/评测口径）逐项冻结。

### A3. 必须先做的验证（不通过不许上 GPU）

**A3.1 公式层（CPU，逐位/数值对拍）**

1. **k3 实现与官方逐位一致**：从 `core_algos.py:2183-2191` **直接取函数**对拍，
   **必须覆盖 clamp 边界**：`kl = ±20` 两端（`ratio = exp(±20)` 会溢出/下溢）、
   `kld` 触 `±10` 的情形、以及 `logprob == ref_logprob`（`kld = 0`）。
   `opd_multimodal/opd_train_vlm.py::_k3_reverse_kl` 是逐字照抄官方的移植，
   可作**交叉参照**，但**判定基准必须是官方源码**。
2. **Bernoulli 实现与官方逐位一致**（做 P2 时）：从 `losses.py:432-453` 取函数，
   覆盖 `tp→1`（`log1p(−tp)` 的数值处理）、`tp→0`、以及 log-prob 被
   clamp 到 `min=−80` / `max=log1p(−eps)` 的边界。
3. **池化合并与官方一致**（做 P3 时）：`(w_r·Σ_r + w_j·Σ_j)/(w_r·N_r + w_j·N_j^eff)`，
   含**空流**（`N=0`）与`w_j=2.0` 两组取值；并断言与论文 Eq 13 的差异
   （仅当 `|I_acc| == Z_rej` 时两者相等）。
4. **退化情形钉住**：全接受块、全被拒块、单 offset 块、
   `valid` 掩码下部分 invalid、以及"两流皆空 ⇒ 零 loss 且仍在图上"。

**A3.2 标度层（上 GPU 前的必测项）**

k3 是**单样本、有界**估计量；全词表 reverse KL 对整条分布取期望。
两者**数值标度不同**，会直接影响有效学习率。⇒ 上 GPU 前必须先量：

* 同一批 anchor 上，`l_rej` 在两种实现下的**均值与方差**（及比值）；
* **k3 的 clamp 触发比例**（历史事实：被拒段会触到上界 10.0）；
* `l_acc` 与 `l_rej` 的相对贡献（历史事实：loss 的 ~97% 由反向项贡献）。
* 若标度差异很大，**学习率是否需要跟着改必须在预注册里写清**，
  否则会把"换了目标"和"改了有效 lr"混成一件事。

**A3.3 开关与默认值**

保留"论文语义/现行"版本作为**可选开关**（如 `--objective {paper_kl, kl_official}`），
**默认值选哪个必须在报告里写明**；并断言各开关内部自洽，
以免把"改了实现"和"改了默认值"混成一件事故。

### A4. 预期与判读（跑之前登记，防止事后附会）

* 预期：**k3 会把被拒段的数值标度显著压低**（单样本 + clamp ±10，
  而全词表 reverse KL 无上界）。⇒ τ 若变化，**必须先用 A3.2 的标度数据解释**，
  不能直接归因于"目标函数更对"。
* **可反驳边界**：
  * 若改成官方 KL 实现后 τ **仍不改善** ⇒ "reverse KL 形式是 GLANCE 负结果的主因"
    这一假设被否掉，应转向 accepted 流为空 / 数据与目标不匹配 / 树选择等方向；
    **不要**靠继续换损失形式来找结果。
  * 若 τ 改善但**可比时延无改善**，只保留机制证据，**不得**称为加速方法。
* 与对照臂的比较必须**同数据量、同 updates、同 lr、同精度、同 EOS/时间轴**。

### A5. 交付物

* 独立 run 目录（新时间戳），含 `prereg.json`、命令、日志、逐视频/逐题表、manifest；
* 报告须写明：**基准脚本与提交**（`run_qwen_gsm8k.sh` / `d1476f5`）、
  改了哪条（P1/P2/P3）、哪条**没改**、`l_acc`/`l_rej` 标度与 clamp 触发率、
  起点 SHA、精度、updates、评测口径与聚合方式；
* 必须与**同一批数据上的对照臂**同时报告，且对照臂与实验臂**数据量/步数/lr 逐项相同**。

---

## ★ 新任务 B（**已于 2026-10-15 执行完毕**，见[报告](../../reports/2026-10-15-taskB-fair-qa-redo.md)；下方为派发时的原始约束，保留供追溯）：公平重做视频 + 图像 QA 两组对照

> 授权日期 2026-10-15。**先读本节全部约束再动手**；未满足公平性约束的运行一律作废，
> 不许"先跑了再说"。

### 目标

1. **视频 QA**：在 2026-10-14 那批实验的基础上（改动后的代码 + 修好的 EOS/时间轴），
   **重做一遍**，产出可引用的 τ / 轮数 / 时延对照。
2. **图像 QA**：**重做**，并且**消灭 §B 第 2/3 条的数据量与训练量混同**。

### 第 0 步：公平性预注册（**必须先写、后跑**，写进独立 run 目录的 `prereg.json`）

对**每一个** arm 逐项记录并在运行前冻结：

| 必须一致的项 | 说明 |
|---|---|
| 训练行数 / QA 槽位 | 两臂**逐条相同**（同一 schedule，只有顺序可按 seed 打乱） |
| optimizer updates | 相同 |
| lr 与 schedule | 相同、constant |
| 起点 checkpoint | 相同（含 SHA256） |
| 精度（target/teacher/head） | 相同；若某臂必须用 bf16 才能跑，**两臂必须同为 bf16** |
| EOS 行为、时间轴模式 | 相同（视频侧 `eos_stop` 与 real/legacy 必须一致） |
| 评测集 / 口径 / 聚合方式 | 相同；**明确说明是 pooled τ 还是 per-video 均值** |

**图像 QA 特加约束（针对 §B 的教训）**：
* 若要**与 GLANCE 基线 τ 5.199 比较**，则**必须把基线也拉到同预算重训一次**
  （同 2000 行 / 同 1000 步 / 同起点），**或者**把 OPD 实验整体做到 16K 行规模。
  **二选一，不许再拿 2K 的臂去比 16K 的数。**
* 三臂至少：**继续 SFT（等预算对照）**、**OPD**、**warm start（不训练）**。
  报告里必须三臂同时给出。
* 训练与评测都要**逐条记录**数据量（打印 `rows` / `qa_slots` / `updates`），
  并在报告首段复述，供人核对。

### 沿用已验证的资产（不要重写）

* KL 目标：`repro/opd_fullkl.py`（**已核对无需改**，见 §A）；
* 在线闭环入口：`opd_train_video_fullkl.py`（默认 `online`，2026-10-13 修复并过门禁）；
* EOS：`glance/decode.py` 的 `eos_stop={"legacy","immediate"}`（默认 legacy 保历史可复现，
  **新实验统一用同一个模式并在报告里写明**）；
* 时间轴：真实 PTS 模式已过编码门禁（`pixel_values_videos` 逐字节不变）；
* 视频公平性模板：`results/qa_opd_fair_20261013_195500/experiment_protocol.json`
  —— 它的 `fairness_constraints`（same_schedule_per_seed / same_updates_lr_optimizer_precision_eos_timestamp /
  no_arm_specific_tuning）就是本次要复用的写法；
* 评测门禁：逐 token SPEC==AR 无损检查、`check_acl_completed_eval.py` 一类的齐备性检查。

### 明确要避免的坑（历史教训，逐条对照）

1. **数据量不等**（本次审计发现，见 §B）；
2. **评测文件被冒烟测试写穿**（历史上发生过符号链接写穿真实 results）；
3. **`train_head.py` 没有 split 过滤**（视频侧 2026-09-25 的 bug：日志自证
   `train rows: 150` 而池是 120/30，等于**拿留出集训练**）；
4. **标签与训练 prompt 差一个词**（`background scenes` vs `overall scene`，
   prompt_len 差 1 ⇒ 每个 anchor 错一位，实测值 +0.33 τ）；
5. **视频路径只有 b63k8 无损**（其余树形在第 37 token 分叉，同一行 τ 3.14 → 8.18）；
6. **accepted 流为空的工况**：若首拒深度恒为 0，则 loss 完全由 reverse KL 主导、
   必须先报告这一事实再解释结果（2026-10-14 的机制就是它）。

### 交付物

* 独立 run 目录（新时间戳），含 `prereg.json`、命令、日志、逐视频/逐题表、manifest；
* 报告：写明数据划分、起点 SHA、seed、精度、updates、评估口径与聚合方式、
  以及**每一个 arm 的数据量**；
* **必须同时报告三臂**（warm start / SFT / OPD），不得只报有利的；
* 若结论与 2026-10-14 不同，**指明旧结论失效的范围**，不删除旧产物。

### 阻塞与资源

* 图像复现与评测需要 ≈40 GiB 空闲显存的 A6000；该机器多租户，
  **历史上多次因他人占满而排队**。开工前先确认空闲卡，**不要打断他人进程**，
  也不要用不同负载下的耗时分作速度证据（计时需同负载、或标注 `timing_contaminated`）。
* 重型任务默认**串行**。

## 维护记录

- 2026-10-15（第五轮）：按用户授权修复QA位置契约并完成无更新门禁，随后以Task B QA-SFT-120共同起点执行continue-SFT/修复OPD各10 updates。新40条当前AR双解码+全轨迹重放，原数据未覆盖。dev20视频完整三臂b63k8、240次SPEC=AR，局部实际top8与L_raw/L_tree/L_actual、逐视频bootstrap、参数梯度及teacher成本交付。未获得超过 continue-SFT 的可靠证据；保留位置契约正确性结果，不宣称方法优势，不继续训练。 GPU7共享获明确授权，不作速度结论；未开新测试集。[完整报告](../../reports/2026-10-15-qa-contract-fix-10update.md)。

- 2026-10-15（第四轮，QA单路径/树语义诊断）：按用户要求完成 AGENTS、Task A/B、10月14日QA报告与C6复核。独立原函数CPU反例确认QA verifier固定比较答案开头、teacher多消费一个proposal后预测下一位置、valid把被拒后缀截成首拒；接受流实际为CE，SFT caller另有EOS监督边界偏一位。用户明确允许GPU7共享诊断。冻结6视频、24anchors、8现存heads两精度，共384trace；18未改生产decoder首轮与固定重放一致。36份完整参数梯度，无optimizer；C0首root同位置CE余弦由原teacher−0.4956变为对齐teacher+0.8600。2/3训练历史continuation与当前重复fp32 AR分叉，已按跑前amend只用新正确AR配对。全部原始结果与失败/归约续跑留痕；所有head与生产源SHA前后不变。撤回QA“整段错误后缀”与“可否定正确OPD”的机制解读，保留历史部署数值。位置修复及唯一建议10-update救援对照未执行；不访问冻结新测试集。报告：[docs/reports/2026-10-15-qa-tree-semantics-diagnosis.md](../../reports/2026-10-15-qa-tree-semantics-diagnosis.md)。

- 2026-10-15（第三轮：**两个任务实际执行完毕**）：按用户"完成上面最新的任务A和任务B"执行。
  **任务 A（P1）**：只改被拒段目标（`--rejected-kl`，默认 `full_reverse` 保持历史逐位不变）。
  CPU 用 AST 从官方 `core_algos.py` 取出并 exec `kl_penalty`/`kl_penalty_forward` 对拍，k3 逐位相等
  （19/19 + 17/17 测试）；上 GPU 前做标度测量（`repro/measure_k3_scale.py`，48 anchors：
  `l_rej` 6.1946→4.9673、**24.6% 顶 ±10 clamp**、被拒流仍占 94%）；预注册后跑冻结 10-update 对照
  （online 闭环、fp32+bf16+CPU 主权重、lr 3e-5）：C0 3.7377 / 对照 3.7300 / k3 3.7480，
  **Δ=+0.0180 CI[+0.0002,+0.0381]**，两臂各 30/30 无损。**关键机制发现**：k3 原始 `|g|` 掉到 0.22–0.42×，
  但 clip 1.0 让两臂 applied norm 都归一（对照 10/10、k3 9/10 裁到 1.0），所以这是一次干净的
  "只换目标函数"比较，**预注册里担心的有效 lr 混淆在 clip=1.0 下基本不成立**。判读为
  **贴分辨率的小幅正信号**，不宣称收益、不宣称加速、也不推翻"reverse KL 是主因"（只是把空间压到 ≤+0.04 τ）。
  **任务 B**：视频 QA 用固定 seed20261014 + 冻结 480 槽日程 + **两臂各 120/120** + fp32 +
  **训练与评测同为 real** 重做；图像 QA 先**按原命令重建被删的 2000 行 TextVQA OPD 池**
  （日志统计量逐项复现：72418 anchors / τ 3.438 / 443.1 MB），再以**三臂同起点** `checkpoints/head`
  重跑 warm/SFT/OPD-CE。结果：视频 warm 3.4700 → SFT **3.6395**（+0.1695）→ OPD **1.2060**
  （−2.2640，0/20 视频改善），**120/120 逐 token 无损**；图像 warm **5.1990** → SFT **5.3676** →
  OPD-CE **5.3225**，SFT−OPD-CE +0.0245 **CI 跨零**。**同时更正 §B 的基线步数 1293 → 1000**
  （1293 是 append-only jsonl 行数），并明确**"把基线拉到同预算"的正确操作化 = 在同一次运行里
  测不训练的 warm start**（`head` 就是四臂共同起点，报告里没有任何 16K 数字参与比较）。
  执行期遇到并修复/留痕的工程问题：`nvidia-smi` 一度无法通信（改用 torch 的 `repro/pick_gpu.py`
  选卡）、`nvidia-smi` 忽略 `CUDA_VISIBLE_DEVICES` 导致 `eval_qa_models.py` 内存门禁查错卡
  （改为传物理 index、不设 CUDA_VISIBLE_DEVICES）、图像 SFT 一次抢卡 OOM（记入 `image/void_runs/`
  并删除半成品 checkpoint，未评测）、`aggregate_eval.py` 表头/行错位（另写 `summarize_taskB_video.py`
  从 `per_question.csv` 直接算）。新增报告 2 份、脚本 7 个；未删除任何历史产物。
- 2026-10-15（KL 口径核对 **+ 一处自我更正** + 图像 QA 对比一致性审计 + 派发两个任务）：
  起因是 Draft-OPD 侧（`opd_multimodal`）修掉一处位置衰减偏差（官方
  `accept_draft_position_decay_enabled` 默认 False ⇒ 接受段**无** position decay）。
  **★ 自我更正**：本节初稿写成"GLANCE 的 KL 不需要改"—— **那是错的**，
  当时只核了"位置衰减"一条轴就下了结论。用户指出后重核，确认 KL 有**三条独立轴**，
  其中**两条与官方代码不同**：① 接受段官方是 **Bernoulli 二值** forward KL，
  GLANCE 是**全词表**；② 被拒段官方是 **k3 单样本估计量**（`core_algos.py:2183-2191`），
  GLANCE 是**全词表 reverse KL**；③ 位置衰减（`w_k` 只乘被拒流、γ=0.8）**两者一致**
  —— 初稿只有这条是对的，而它恰好也是 Draft-OPD 侧同期修掉的同一个坑。
  另有一条合并方式差异：官方是**池化加权均值**（`losses.py:854-888`），
  GLANCE 是论文 Eq 13（两流各自归一后按 λ 平均）。
  **原因不是随手写的**：`docs/reports/2026-10-09-b1-formula.md` 是逐条对过
  arXiv 2605.29343 原文（Eq 8–13、App. B，明确 `Σ_{y∈V}`）与官方 `losses.py` 的核实记录，
  **GLANCE 显式选择了"论文语义"**；旁证是官方 `core_algos.py` 的 `kl_penalty=="full"`
  分支为 `raise NotImplementedError`（官方未实现全词表）。⇒ 准确说法是
  **"论文公式与官方代码默认实现不是同一个目标函数"**，而非"官方代码错了"。
  **按用户 2026-10-15 指示，新增「新任务 A：把目标函数改成官方代码实现」**。
  **★ 同日追加（第 2 轮，用户改口）：**用户先指示"官方哪个入口新就用哪个"，
  我据此把基准定成最新入口 `run_qwen_gsm8k_forward-8b-local.sh`（2026-09-04）
  的 **TV 端到端目标**（`L_block = 1 − E[acc]/K`，`E[acc]=Σ cumprod(1−TV)`，
  `TV=0.5·Σ|p_draft−p_target|`，`dflash_student.py:40-53 / :1349-1367`；
  `stride=1`，官方注释"Exact TV is defined on every rollout speculative block"）。
  **随后用户明确"先不要 TV loss，保留 KL 的最新的"** ⇒ 基准改为
  **KL 版里最后提交的入口 `run_qwen_gsm8k.sh`（2026-08-08, `d1476f5`）**，
  其默认值：`MODE=k3`、`REVERSE_KL_WEIGHT=0.0`、`FORWARD_KL_WEIGHT=1.0`、
  `REJECTED_DRAFT_USE_REVERSE_KL=True`、被拒 decay **0.9**、
  `stream_weight=1.0`、**`rejected_draft_stream_weight=2.0`**、
  `FIRST_TOKEN_ONLY` 默认 False（整段后缀）。
  **更正我先前的取证疏漏**：上一轮我只把 `-8b-local.sh` 与 `forward-ins.sh` 列为候选，
  **漏了 `run_qwen_gsm8k.sh`（08-08）才是最新的 KL 版**；
  也要注意 README 指向的 `forward-ins.sh` 是**最旧（05-31）**且超参不同
  （1.0 / decay 0.8），**两者不可混引**。
  任务 A 据此重排为 **P1 = 被拒段 → k3 单样本**（第一步只改这一处）、
  P2 = 接受段 → Bernoulli、P3 = 池化合并 + `w_j=2.0`；
  TV 公式仅作**存档**（§A0.1），**不作为基准**。
  含逐项改动表（含官方 clamp 常数）、**单变量纪律**、
  CPU 逐位等价验证（含 clamp 边界）、**上 GPU 前的标度必测项**
  （k3 与全词表 reverse 的 `l_rej` 均值/方差比、clamp 触发率 —— 标度差异会改变有效 lr，
  必须在预注册里写清 lr 是否跟着改）、以及跑前登记的预期与可反驳边界。
  另**图像 QA 逐条取证**（证据取自
  `/mnt/local/mcy/GLANCE/logs/` 原始训练日志）：① control↔OPD 这对**本身公平** ——
  两侧均 `2000 行 / 1000 步 / lr 1e-4`，且 `41_train_control.log` 与 `42_train_opd.log`
  启动时间戳完全相同（2026-09-23 20:53:15）；② **但"与基线比"是混同的** ——
  基线 `head`（τ 5.199）训的是 `11_train.log` 所记 **16000 行（8000 COCO + 8000 TextVQA）、
  1293 optimizer steps**，而 OPD 实验的对照只训 2000 行 TextVQA / 1000 步，
  拿它们对比同时混了"数据量+训练量+任务混合"三件事，**引用时不得把
  `head_sft_control`(2K) 与 `head`(16K) 当同一档基线**；③ `head_sft_control`(5.365)
  反而高于基线 `head`(5.199)，但从未做过等数据量/等训练量对照，**该差值不可解读**。
  据此在本文新增 §A/§B 与**两个待执行任务**：
  **「新任务 A：把目标函数改成官方代码实现」**（§A 轴①②）
  与**「新任务 B：公平重做视频 + 图像 QA 两组对照」**，
  含第 0 步公平性预注册表、图像 QA 特加约束（比 5.199 就必须把基线拉到同预算重训，
  或整体做到 16K，二选一）、要复用的已验证资产与 6 条历史坑。本次**只改文档**，
  未运行任何训练或评测、未改任何代码。

- 2026-10-14（三数据集正式评测补齐）：在 2026-10-14 的 QA-SFT/QA-OPD 公平训练基础上，完成 **MSVD-QA / MSRVTT-QA / NExT-QA 三个 B 格式数据集的 fp32 无损评测**（每数据集 20 视频/40 QA，5 个 arm：C0、QA-SFT-s1/s2-120、QA-OPD-s1-30、QA-OPD-s2-120，共 600 条 QA、1200 次 SPEC 解码全部逐 token 等于独立 AR），并补齐 **独立确认集**（MSVD 官方 val 中与 dev/训练视频级不相交的 60 视频/120 QA）。视频级配对（bootstrap 2000）`QA-SFT − C0`：MSVD **+0.1620** [+0.0915,+0.2349]（seed1）/ +0.0925 [−0.0052,+0.1902]（seed2）；MSRVTT **+0.2094** [+0.1316,+0.2949] / **+0.2098** [+0.1199,+0.3138]；NExT **+0.1018** [+0.0537,+0.1496] / **+0.1380** [+0.0872,+0.1856]；确认集 **+0.2106** [+0.1400,+0.2820] / **+0.2011** [+0.1329,+0.2750]。`QA-OPD − C0`：seed1-30 在 MSVD **−0.1922** [−0.2546,−0.1229]、MSRVTT +0.0364 [−0.0478,+0.1269]、NExT −0.0632 [−0.1328,+0.0060]（跨域不可区分）、确认集 **−0.1311** [−0.1908,−0.0674]、对 QA-SFT **−0.3391**；seed2-120 在三个数据集全部 **−1.97 ~ −2.34 τ**（20/20 视频变差、轮数涨到 31–39）。**判定：QA-SFT 的领域内 QA 适配收益跨域成立；当前 full-KL 在线 QA-OPD 配方无额外收益，长训甚至破坏草稿头，属任务文档 §9 判据 B（明确负面）而非"证据不足"。** 机制：OPD 的 accepted 流几乎为空（首拒深度 seed1 720/720 为 0、seed2 2724/2880 为 0），loss 由 rejected reverse-KL 主导。全部产物、逐视频/逐题表、门禁与命令在 `results/qa_opd_fair_20261013_195500/`（报告 `reports/FINAL_QA_SFT_VS_OPD.md`、`reports/NEXT_ACL_DECISION.md`）。磁盘清理只删除本轮冗余中间 checkpoint（删除前已存 SHA256 表），历史 checkpoint 归档未动。

- 2026-10-14（QA-SFT/QA-OPD 公平训练，已完成训练与主结论）：按用户授权执行 `docs/tasks/10.13-19.09prompt.md`。产物根 `/mnt/local/mcy/GLANCE/results/qa_opd_fair_20261013_195500/`（`training/` 因 `/mnt/local` 写满已迁至 `/mnt/local2` 并留软链接）。**EOS 缺陷已修**：`glance/decode.py` 新增 `eos_stop={"legacy","immediate"}`（默认 legacy 保持历史可复现），真实 72 次解码复测 36/36 对命中 bonus-EOS、7 对 legacy 多跑一轮（244→237 轮），两模式 token 完全相同、SPEC==AR 全通过，CPU 分支测试 8/8。**时间轴适配已做且过门禁**：只传真实 `VideoMetadata` 无效（transformers 4.57.3 会重新推断），必须传已解码 16 帧；real 模式下 `pixel_values_videos`/`video_grid_thw` 与 legacy 逐字节相同、时间标记改为真实 PTS（patch 级误差 ≤0.045s）。**数据**：官方 MSVD-QA train 1161/1161 全量取得并 cache16，`readiness_gate.json` 十项硬门禁全 true；两 seed 共用同一 480 槽 QA 日程（仅顺序不同）；confirm = 官方 val 中与 dev/训练视频级不相交的 60 视频/120 QA，模型输出前冻结。**训练**：QA-SFT seed20261013 fp32 120/120、seed20261014 bf16 120/120；QA-OPD seed1 fp32 60（step90 在磁盘写满时损坏，从完整 step30 精确续训）、seed2 bf16 120/120；均保存可恢复的 head+fp32 主权重/Adam 状态。**主结论：QA-SFT 正向、QA-OPD 负向且随步数恶化。** dev（MSVD-B 20 视频/40 QA，全逐 token 无损）mean τ：seed1 SFT-120 **3.7037** / SFT-90 3.6678 / SFT-60 3.6257 / SFT-30 3.5551 / C2 3.5452 / C0 3.5417 / C1 3.5111 / OPD-30 **3.3495** / OPD-60 **2.6119**；seed2 SFT-90 3.6351 / SFT-120 3.6343 / OPD-30 3.3772 / OPD-60 2.9781 / OPD-90 2.1157 / OPD-120 1.18。机制：OPD 的 accepted 流几乎为空（首拒深度 seed1 720/720 为 0、seed2 2724/2880 为 0），loss 由 rejected reverse-KL 主导。**ACL 方向**：唯一推荐"首预测位置/候选边界失配"诊断（见 run 内 `reports/NEXT_ACL_DECISION.md`），在诊断分支明确前不恢复 OPD 调参。bf16 下 greedy AR 不可复现，故无损声明只在 fp32 给出；A6000 长期被他人任务占用，计时逐条标 `timing_contaminated`。三数据集正式评测在 run 内 `eval/` 与 `reports/FINAL_QA_SFT_VS_OPD.md`。

- 2026-10-13（报告文件命名整理）：19份报告（含1个软链接）统一日期前缀，新增按时间排列的报告索引并同步项目内引用；正文仅调整交叉引用，产物盘原件及历史manifest保持不变。改名前备份及名称映射：`/mnt/local/mcy/GLANCE/results/documentation_rename/20261013T193131`。

- 2026-10-13（用户授权OPD闭环修复）：新增当前target在线验证/状态推进模块并设为full-KL入口默认；旧Phase C/C4及C5诊断显式legacy。添加EOS/预算/缓存/轨迹独立性/重放拒绝测试，CPU44项通过，439训练记录dry-run通过；空闲GPU3实际2视频6轮AR精确一致、6份loss重放通过、2次backward无step，峰值38.91GiB，C0磁盘SHA及内存参数version不变。无训练、无生产decoder修改、无新测试集访问。原文件备份、GPU准确源码、manifest、逐轮token和日志在`/mnt/local/mcy/GLANCE/results/opd_online_fix/20261013/`；结论与成本限制见[修复报告](../../reports/2026-10-13-opd-online-fix.md)。

- 2026-10-13（OPD实现审计）：核实C4配置与实际入口，对照官方d6f0ec8训练/verify/loss及在线README；确认当前proposal与teacher评分在线，但基础trajectory与n_acc仍依赖旧reference，未形成真实target验证/状态推进闭环。36项CPU测试及合成契约反例完成，C5原始统计独立求和复现；pytest缺失记录保留，未改共享依赖。只写审计报告与状态，训练/部署代码未修改、真实模型调用0、参数更新0。产物`/mnt/local/mcy/GLANCE/results/opd_implementation_audit/20261013/`。

- 2026-10-13（视频QA迁移，已完成）：完整读取附件及C4/C5/C6证据，审计历史图像OPD并核实C4三头；冻结并完成MSVD/MSRVTT/NExT-QA多选各20视频40QA，两格式，正式2160/2160与smoke324/324逐token无损。CPU独立复算5202诊断轮、60视频原始/缓存SHA、各组80context及12个pilot/full AR均通过；三头权重/config与生产解码/树文件前后不变。target/head真实调用39376/18018，processor435次，含恢复warmup。NExT full因外来GPU进程在138/240后自动退出，自然空闲后在新目录补齐剩余102条，原失败/138条原始证据保留、输入哈希80/80重现；时延保留批次及1题资源边界不确定标记。ffprobe、代理/TLS、初始资源阻塞及CPU汇总失败均保留。B三组主τ区间均跨零；NExT decode-only弱信号不同时满足稳定τ/总耗时收益。当前建议先做独立EOS停止修复的无训练复核，未启动；不建议立即开展新QA训练，不宣称OPD普遍无效。模型仍沿用缓存压缩时间轴，不冒充真实时序输入。完整产物、逐视频表与SHA索引见[最终报告](/mnt/local/mcy/GLANCE/results/videoqa_transfer/20261013T134000/final_report.md)。全部新数据在/mnt，home/local系列原本是软链接、无需搬迁；没有参数更新、生产解码修改、C7/D4或冻结新测试集访问。

- 2026-10-12（ACL路线整合，仅文档）：按用户研究备忘录整理当前诊断结论、C7 Stage 0–3门禁/停止规则、两条有条件研究路线及投稿证据门槛；移除“下一步”中已完成的C3/C4待办，保留历史进度与报告。C4区间沿用本地原报告；A-old/A-relabel重算归属C6，C5支线与C6样本分开。C7、D4仍未执行，未启动模型实验或训练。

- 2026-10-12（ACL诊断审计）：完整读取诊断附件，核查真实根目录、环境、phaseC4来源与phaseC5原始产物；建立此前不存在的收益报告。无训练，未改共享依赖、未重跑已有模型评测。CPU补做目标归一/三段rank/视频bootstrap、真实曝光顺序和监督预算、旧轨迹vs已存AR、70条processor输入重放、候选oracle算子与16项测试；head前后哈希一致。确认旧轨迹69/70分叉、58个旧接受标签不符teacher，b1k1非完整链；D4按门禁不运行。期间原有g1/g2评测自然结束，复用三臂各40/40无损并与validation一起核实跨臂70/70 AR一致。C2曝光10条与完整未曝光20条的真实τ收益差异证据不足（差中差+0.0286，CI[-0.0586,+0.1171]），不把首10条子组较强关联当稳健泛化结论。详见`docs/reports/2026-10-12-gain-stuck.md`和`results/phaseC5/diagnostic_acl_20261012T213006/`。仍未取得D1正确AR分布、D2局部oracle和D4真实前向证据，不宣称全轮实验已完成。
- 2026-10-09：建立本进度入口；合并四份重复项目/面试说明至可恢复归档；精简 repro README；分类保存报告、计划与文献。补正 B0.5/B1 状态、validation 边界及 decode-only 计时口径。本次只整理文档，未运行新实验。
- 2026-10-09：完成 OPD/多模态投机解码文献检索，归档 68 篇论文及 PDF，新增逐篇借鉴与候选实验说明；代码与实验进度不变。
- 2026-10-10：执行计划 §10 长程任务并收口。完成公式核实（发现官方代码与论文在「采样估计量」与「池化归一」两处分歧，已显式选择论文语义）、B0.5 精度校准、B1 模块与 trainer（16/16 + 17/17 测试）、fp32 可行性实测（OOM，bf16 替代峰值 26.42 GiB）、Phase C seed-0 三臂训练与评测。**主结果：C2 同时低于 C0 与 C1，触发预注册停止规则 2，收口不延长。** 顺带修掉 B1 trainer 两个真实 bug（EOS 守卫被 prompt 内的 `<|im_end|>` 触发导致空转全数据集；per-anchor `vispos` 绑定长度不符）。新增报告 2 份，新增候选测试集盘点 1 份。未删任何历史产物。
- 2026-10-12（第五轮，停止扩展 OPD + 收益卡点诊断）：用户叫停继续扩展这套 OPD 训练，改为**不更新参数**的诊断，并纠正我两处表述——①"从 −0.2281 到 +0.0027 说明旧退化主要不是 OPD 语义"**不成立**：本轮同时改了 lr、步数、teacher/context 精度、优化器精度与视频曝光，未分离，只能说结论对配置高度敏感；②**7/29、11/29 是"正向视频数"，不是"发生变化的视频数"**（后者是 C2 16/29、C1 20/29）。已在 phaseC4 报告与本文修正。三项诊断（复用 C0 与 phaseC4 C1/C2 checkpoint，不追加训练）：**D1 固定上下文的预测变化**（同一 prefix/anchor/mask，把位置拆成 accepted / **首个 rejected** / rejection 之后的 suffix，分别报 KL、target greedy token 排名、top-1 命中、首拒位置；用 C0 生成的 anchors，离线诊断，正式在线解码仍由各自策略生成）、**D2 chain vs tree**（三模型同视频分别评 chain 与现有 b63k8 tree，均保留 AR 门禁）、**D3 曝光 vs 泛化**（本轮实际训练过的视频 / 同数量未曝光训练视频 / 现有 validation；后者是历史训练集，**不得称为 held-out**）。目标交付是一张 **loss 变化 → token 排名变化 → 首拒位置变化 → 实际 τ 变化** 的配对诊断表。新测试集继续不动。
- 2026-10-12（第四轮，C2 主权重接入 + 组合门禁 + 冻结对照）：把 fp32 主权重接进 C2（`opd_train_video_fullkl.py`），验证四项集成不变量（裁剪在 fp32 主权重梯度上、主权重与模块参数写回逐位一致、optimizer 只含 head 的 58 个张量、checkpoint 带 `optimizer_master.pt` 且 `load_master` 逐位恢复而朴素续训会重启 Adam）。组合门禁 `fp32 teacher + bf16 head + CPU 主权重` 3 次真实 update 通过，峰值 **39.7 GiB**、host RSS 35.1 GiB。发现并修复用户指出的关键混杂：C1 原先用 bf16 target context，与 fp32 teacher 的 C2 不是"只差 loss"，新增 `--target-dtype` 并统一两臂到 fp32，同时补上 C1 的 lm_head/embedding dtype 边界。跑冻结 10-update 对照：C0 3.7377 / C1 3.7225 / C2 3.7403，**C2−C0 +0.0027、C2−C1 +0.0179 均为正但都跨零**（t 0.21/0.99，正向 7/29 与 11/29）→ 判为噪声内、不复现，转数据与目标诊断。**相对 Phase C 结果明显不同，但不得断言"旧退化主要不是 OPD 语义"**——本轮同时改了 lr、步数、teacher/context 精度、优化器精度与视频曝光，未分离；只能说结论对配置高度敏感。注意区分**正向视频数**（C2−C0 7/29、C2−C1 11/29）与**发生变化的视频数**（C2 16/29、C1 20/29）。
- 2026-10-10（第三轮续，方向误差 + 复校准收口）：补测**更新方向**——第 1 步同状态下，lr 3e-5 的范数比 0.223 且**逐张量余弦中位数仅 0.221**（31% 张量更新恰为 0），lr 3e-4 范数比 1.042 但**余弦仅 0.792**（22.4% 零更新）。因此修正「lr 3e-4 第一步不受影响」为「幅度基本没损失、**方向误差仍显著**」，并明确范数比不等于有效更新比例、比较对象是 bf16 模块参数 vs fp32 主权重。phaseC3 复校准收口：6/6 候选过门槛，选出 lr 3e-5/u10，完整 validation **3.7453 vs C0 3.7377（+0.0077，CI 跨零，14/29 条 τ 改变 vs 修复前 3/29）**。新增 `run_phaseC3_pipeline.sh`（6 评测 → 冻结规则 → 完整 validation 串成一条流水线），修正 `run_phaseC3_eval_retry.sh` 的 GPU 感知排队。未重启 C1/C2。
- 2026-10-10（第三轮，更新精度）：用户提出「低学习率下几乎不动，可能不是训练温和，而是 bf16 把小更新舍掉了」。真实 head + 真实 batch 实测确认：训练入口一直是 `AdamW(bf16 params)`、无 fp32 主权重，**lr 3e-5 第一步只实现 0.223 的 fp32 理想更新范数、96.9% 的元素拿到 <半个 ulp 的步长且不累积**（10 步 loss 降幅只有 fp32 主权重的 1/5.2）；**lr 3e-4 第一步不受影响（1.042）**，所以 C0>C1>C2 排序不变，但 A4 与 Phase C 的长跑都带上一个未量化的优化器精度混杂。新增 `repro/master_optim.py`（fp32 主权重，支持 CPU offload）与 `--master-weights {none,cpu,gpu}`（默认 none 保持历史行为）；加主权重后同一 lr 的 25 步权重位移提高约 12 倍。修正我此前「退化主因已证明是 lr」与「179 倍说明学习率效应」两条过强表述。已用主权重重跑 lr {3e-5,1e-5}×25 updates 校准，评测因所有 A6000 被他人占用（fp32 eval 需 ~40 GiB 空闲）而排队在 GPU 感知的重试脚本里。新增报告 1 份，未重启 C1/C2，未删历史产物。
