# 2026-10-15 QA-OPD 位置契约修复与 10-update 公平验证

未获得超过 continue-SFT 的可靠证据；保留位置契约正确性结果，不宣称方法优势，不继续训练。

共同起点是 **Task B QA-SFT-120**，不是 C0。两臂各40 QA槽/10 optimizer updates，保留零更新 ZERO；评测20视频40 QA，统计单位为视频。所有新增模型、缓存与大文件位于 `/mnt/local3/mcy/GLANCE/results/qa_contract_fix_20261015`。历史checkpoint/评测产物未修改，未打开冻结新测试集。

## 1. 修复范围与保持的目标

| 项 | 修复/约定 |
|---|---|
| n_acc | raw verifier[i] 对应绝对位置 prompt_len+i；offset k 用 i=p+k−prompt_len，删除重复root。接受只计连续前缀。 |
| teacher | cache截止root之前，输入 `[root]+proposal[:last−1]`，第k−1行预测p+k；保持draft-induced条件，首拒后的prefix不替换为正确AR。 |
| suffix mask | rejection只结束accepted，不终止valid；监督整段被拒后缀，保留首个reference/proposal EOS，排除其后及答案/块边界。 |
| SFT EOS | caller的valid_end由最后anchor改为最后anchor+1，包含EOS标签；与OPD采用一致EOS边界。 |
| 原公式保留 | accepted为one-hot CE（并非soft forward KL）；rejected为全词表KL(q∥p)，γ=0.8；全update分别按N_acc/Z归一，两流存在时等权平均。无k3、P2、P3。 |
| 训练轨迹 | 共同固定AR anchors上的当前proposal；不是各自策略访问轮次的完整在线rollout。每anchor的接受长度由当前target在线重算并核对。 |
| 输入和标签门禁 | encoded prompt逐token相等；当前target全轨迹replay逐token复现冻结AR，否则停止。每anchor规划/梯度前向proposal一致，否则停止。 |

两臂保留各自原监督协议：SFT为variable-reveal加权CE，OPD为reveal=1两流目标。相同QA、updates和lr不等于监督位置数或teacher计算量相同；实际预算见下表。SFT每QA归一后累计，OPD按update两流归一，未额外改动这些公式。

## 2. 第一阶段：测试与无更新门禁

13项新CPU测试与4项既有anchor/guard测试通过。覆盖非零anchor、n_acc=0、部分/全部接受、prompt边界、reference/proposal EOS、末尾裁剪、无有效位置、旧重复root契约拒绝，以及独立手算公式的loss/梯度对拍。测试调用实际trainer函数。

3训练视频×2 anchors，共6份同输入旧/新完整参数梯度对照。旧函数从备份AST原样载入，不改历史代码快照。独立逐token teacher与批量teacher所有argmax一致，在线n_acc与修复规划一致；最大logit差 3.6239624e-05。全部58参数张量测量，内存版本不变，optimizer steps=0。

| video / anchor | n_acc旧→新 | valid旧→新 | loss旧→新 | 梯度cos(旧,新) |
|---|---:|---:|---:|---:|
| 5EXVeP6R3yc_105_108.avi / 994 | 1→1 | 2→15 | 4.5350→4.6234 | +0.2763 |
| 5EXVeP6R3yc_105_108.avi / 995 | 0→2 | 1→15 | 22.7339→3.3061 | -0.0311 |
| 5W02895vT8c_312_322.avi / 992 | 1→1 | 2→15 | 11.0935→4.3967 | -0.1796 |
| 5W02895vT8c_312_322.avi / 993 | 0→0 | 1→15 | 15.9624→8.2788 | -0.5392 |
| aCKNdip53W4_0_25.avi / 998 | 0→0 | 1→15 | 32.1984→5.0896 | +0.2421 |
| aCKNdip53W4_0_25.avi / 999 | 0→2 | 1→15 | 14.1688→3.4443 | -0.2878 |

以上loss/mask/分流共同改变，不能把loss下降解释为训练收益或单一bug的因果贡献。原始teacher token IDs、mask、accepted/rejected offsets、58张量dot/norm²均在 `gates/same_input.jsonl`。

## 3. 冻结起点、数据与预注册

- 起点：`/mnt/local3/mcy/GLANCE/results/taskB_fair_qa_20261015T085742/video/training/sft/step120`，SHA256 `84fe984cc21f7ad5f7ee9fffc91d15ed8b0960f66159b694601e57e5a311d014`。实际58张量、1,048,626,432参数、磁盘BF16。
- 来源：Task B seed20261014、120 updates、real时间轴、fp32 target、bf16 head、fp32 CPU主权重；原配置副本在 `dataset/start_training_config.json`。
- 两臂均从同一已发布bf16权重建立新的fp32主权重，Adam矩重置；不是恢复历史SFT optimizer。ZERO就是同一发布权重的零更新评测。
- 日程为原seed20261014前40槽，4 QA/update、最多6 anchors/QA；同lr3e-5 constant、AdamW(0.9,0.95)、wd0、clip1、fp32 target/bf16 head/CPU fp32 master。
- 重新用部署greedy路径生成40条当前fp32 AR，每条独立生成两次且完整prefill重放一致；两臂使用同一个新pool/records，旧标签不改写。prompt/视觉编码/real时间轴/EOS immediate固定。
- 新旧标签复核：40条中26条历史continuation与当前稳定AR不同，prompt全部相同；详见gates/historical_AR_comparison.json。差异原因未单独隔离，不归咎某一精度；两臂统一使用新标签。
- dev为已有MSVD-B 20视频40 QA，与本轮训练视频不重叠；不是untouched test。QA日程与评测manifest SHA在prereg；训练前数据/源码SHA在training_seal。
- 原局部计划为3 train+3 dev探针；在训练和任何结果评分前明文amend为3 train+全部20 dev视频（40 QA）。每QA固定p=L−1、首部署root、答案中点、最后非EOS root；prompt边界单列，主表排除p<L。
- 预注册固定10步，不挑checkpoint、不延长；OPD−ZERO的τ CI上界<0为明确退化并收口；不超过SFT不称方法优势。单seed即使CI为正也不称稳定优势。

## 4. 真实训练预算

| arm | updates | anchors | 监督位置 | accepted / rejected | context调用 / tokens | proposal teacher调用 / tokens | peak GiB |
|---|---:|---:|---:|---|---|---|---:|
| sft | 10 | 197 | 1725 | N/A / N/A | 40 / 40142 | 0 / 0 | 39.22 |
| opd | 10 | 240 | 2774 | 470 / 2304 | 40 / 40142 | 240 / 2774 | 39.20 |

240个OPD anchors中，200个监督多个rejected offsets，27个无rejected监督，13个仅一个；完整直方图见gates/rejected_suffix_histogram.json。teacher调用上表只指训练：另有数据构建的双AR与完整replay、第一阶段验证和评测，原命令/逐样本调用规则保留，不把这些成本隐去。每update梯度范数、裁剪系数、loss、分流计数及teacher位置在各臂log.jsonl；预算机读 `training_budget.json`。

| update | SFT梯度范数 / clip | OPD梯度范数 / clip |
|---|---|---|
| 1 | 12.6117 / 0.0793 | 3.4228 / 0.2922 |
| 2 | 9.0011 / 0.1111 | 3.6613 / 0.2731 |
| 3 | 8.4999 / 0.1176 | 4.2459 / 0.2355 |
| 4 | 10.9836 / 0.0910 | 5.2958 / 0.1888 |
| 5 | 12.2598 / 0.0816 | 4.6112 / 0.2169 |
| 6 | 8.9748 / 0.1114 | 3.3387 / 0.2995 |
| 7 | 10.0376 / 0.0996 | 3.6803 / 0.2717 |
| 8 | 9.9377 / 0.1006 | 3.4926 / 0.2863 |
| 9 | 9.3610 / 0.1068 | 3.1738 / 0.3151 |
| 10 | 10.7083 / 0.0934 | 3.7266 / 0.2683 |

## 5. 真实在线 b63k8 与无损验证

三个头本轮全部重新评测，fp32 target/head、real、EOS immediate、max_new160、b63k8。每QA一次主要SPEC与一次复测均对独立AR逐token校验；主记录120/120，复测120/120无损。先取每视频两QA均值，再对20视频等权平均。

| arm | mean视频τ | 平均轮数 | 实际tokens/round | decode-only ms/token† |
|---|---:|---:|---:|---:|
| ZERO | 3.639499 | 9.7250 | 3.685005 | 33.238 |
| SFT10 | 3.677040 | 9.6500 | 3.705939 | 35.428 |
| OPD10 | 3.642907 | 9.7500 | 3.682579 | 34.711 |

† GPU7共享已获用户明确授权，计时污染行 120/120；不提供加速结论。τ采用现有decoder计数，实际tokens/round同时报告，局部长度不包含root/bonus，二者不混用。

| comparison | Δ视频τ | 视频bootstrap95%CI | 正/负/不变 |
|---|---:|---|---|
| SFT10-ZERO | +0.037540 | [-0.027792, +0.108585] | 9/4/7 |
| OPD10-ZERO | +0.003408 | [-0.061922, +0.070757] | 6/7/7 |
| OPD10-SFT10 | -0.034133 | [-0.117019, +0.042432] | 5/7/8 |

bootstrap 10000，seed20261015，单位视频；完整逐视频数值与各指标配对差异在 `per_video_online.csv` / `paired_video_online.csv`。

## 6. 同正确prefix的候选与树指标

固定同视频、QA、正确AR prefix、融合特征、mask和fp32精度；实际GPU top8成员资格（不是rank阈值代替）。L_raw是连续正确top8覆盖；L_tree为正确路径在实际预算树中的深度；L_actual为packed target验证后的有效draft token长度，排除root/bonus并按EOS/答案上限/15个offset裁剪。共516份trace（43 QA×4 anchors×3 heads）；主表为387份部署可用trace。捕获GPU候选和分数后CPU独立重建节点/路径及完整ancestor mask哈希，516/516通过，全部局部L_tree=L_actual。

| split | arm | top1 | top8 | L_raw | L_tree | L_actual | 正确单路径L |
|---|---|---:|---:|---:|---:|---:|---:|
| train | ZERO | 31.183% | 63.441% | 3.555556 | 2.000000 | 2.000000 | 0.888889 |
| train | SFT10 | 33.333% | 63.441% | 4.111111 | 2.222222 | 2.222222 | 2.333333 |
| train | OPD10 | 36.559% | 64.516% | 3.555556 | 2.555556 | 2.555556 | 2.444444 |
| dev | ZERO | 23.101% | 58.747% | 3.450000 | 2.233333 | 2.233333 | 1.233333 |
| dev | SFT10 | 24.158% | 59.770% | 3.608333 | 2.258333 | 2.258333 | 1.300000 |
| dev | OPD10 | 24.720% | 58.622% | 3.516667 | 2.266667 | 2.266667 | 1.300000 |

每视频先汇总；train仅3个既有训练池视频，dev20视频。这3个train QA均未在本轮40槽曝光，逐QA记录见local_train_exposure.json；不能用此子组推断已曝光训练样本的变化。局部Δ不等于各臂真实在线τ差异。首个raw/tree缺失位置、逐位置候选IDs和完整树在 `anchor_results.jsonl`；逐视频与全部指标bootstrap见 `per_video_local.csv`、`local_summary.json`。单proposal失败不等于树失败。

| dev comparison | metric | Δ | 视频95%CI |
|---|---|---:|---|
| OPD10-ZERO | top8 | -0.1254 pp | [-1.3762, +1.0580] pp |
| OPD10-ZERO | top1 | +1.6188 pp | [+0.3038, +2.9715] pp |
| OPD10-ZERO | L_raw | +0.0667 | [-0.0583, +0.1917] |
| OPD10-ZERO | L_actual | +0.0333 | [-0.0417, +0.1250] |
| OPD10-SFT10 | top8 | -1.1480 pp | [-2.4050, +0.0223] pp |
| OPD10-SFT10 | top1 | +0.5619 pp | [-0.7505, +1.8591] pp |
| OPD10-SFT10 | L_raw | -0.0917 | [-0.2585, +0.0500] |
| OPD10-SFT10 | L_actual | +0.0083 | [-0.0667, +0.0917] |

OPD在固定开发prefix的top1有小幅提升，但top8覆盖、L_raw和L_actual相对ZERO的区间均跨零；OPD相对SFT的top8、L_raw和L_actual区间也都跨零。局部argmax改善没有形成可靠的候选覆盖或树接受收益，不能把它当作全局方法优势。

## 7. 判断、边界与下一步

未获得超过 continue-SFT 的可靠证据；保留位置契约正确性结果，不宣称方法优势，不继续训练。

本次同时修复接受索引、teacher位置和suffix有效范围，并统一SFT EOS边界、刷新两臂AR标签；因此是实现正确性救援对照，不能分解单个缺陷的因果贡献。起点是QA-SFT-120，不能与历史C0起跑的120-update负结果作单变量差分；Task A入口和k3结论不受本轮自动更改。共享资源不允许速度结论，单seed及已使用开发集不允许泛化/独立测试声明。

## 8. 交付、复现与失败留痕

代码：`repro/qa_opd_fair/train_qa_opd.py`、`train_qa_sft.py`、`qa_anchors.py`；测试 `test_position_contract.py`。部署decode/tree与通用KL公式未改。原代码在 `before/`，实际训练版本在 `source/`，差异在 `source_changes.patch`。

完整命令见 `commands.txt` / `scripts/run_experiment.sh`。无更新GPU门禁：`scripts/prepare_gate.py gates`；数据构建：同脚本 `data`；CPU复算：`scripts/analyse.py`。复现必须使用新输出目录并重新核实GPU授权/空闲资源，不覆盖本轮或历史结果。
失败记录：首次冻结脚本把dataset名写为msvd而非MSVD-QA，断言阻止继续；第一次GPU门禁启动因尚无prereg在加载模型前退出。修正后独立日志v2成功。所有失败日志保留，不混入正式统计。

`delivery_manifest.json` 列出原始数据、缓存、checkpoint、配置、门禁、逐视频统计、脚本与日志哈希；`final_gates.json`记录源checkpoint、部署代码、训练封存版本及计数完整性。AGENTS.md同步本轮最终状态。
