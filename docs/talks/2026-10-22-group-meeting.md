# 块并行 Draft + 在线蒸馏：多模态无损投机解码

组会材料 · 2026-10-22

## 一句话

冻结 Qwen3-VL-8B-Instruct，训练一个 1.05B、5 层的块扩散 draft head：一次 draft 前向覆盖 16 个位置（1 个 root + 15 个预测），一次 target 前向验证 63 节点候选树，输出与贪心 AR 逐 token 相同。ScienceQA 训练 3 个 epoch 后，OPD 臂题均每轮提交 6.22 个 token（SFT 5.37），端到端 2.52×（相对 AR）、1.10×（相对同预算 SFT，24 图测速子集）；test200 准确数与 SFT 相同（162/200），200/200 输出无损。

## 1. 任务

- **目标**：多模态大模型图像问答的长文本生成，减少解码阶段的串行验证轮数。
- **约束**：target 全程冻结，只训练 draft；提交的每个 token 必须等于 target 贪心 argmax。
- **指标**：有效 token/验证轮 τ、端到端墙钟加速、SPEC 与独立 AR 逐 token 一致性、答案准确率。
- **主场景**：ScienceQA 图像 CoT，答案中位约 141 token，图像 grounding 强。
- **边界**：TextVQA、ChartQA 这类 4–10 token 的短答，decode 只占端到端的一小部分（ChartQA 27%，TextVQA 更低），投机解码没有可摊薄的循环。

## 2. 方法

### 2.1 一轮解码

1. target prefill 一次，缓存 KV 和融合 hidden。
2. draft head 读 target 第 {1, 9, 17, 25, 33} 层 hidden（5×4096 拼接，20480→4096），以最后一个已提交 token 为 root，一次前向并行输出 15 个未来位置的分布（连同 root 共 16 个位置）。
3. 每个位置取 top-8，按路径概率和保留 63 个前缀闭合节点，组成候选树。
4. target 做一次 ancestor-masked forward 验证整棵树；从 root 起取与 target argmax 连续一致的最长前缀提交；全部接受时用最后一个位置的预测再提交 1 个 bonus token；遇到 EOS 立即停。
5. 只保留已提交前缀的 KV/hidden，进入下一轮。

块内双向注意力，对上下文因果；15 个位置并行预测，位置之间不做自回归。

### 2.2 模型与图像嵌入

- target 先做一次多模态 prefill：vision tower（27 层 ViT，patch 16，2×2 merge，hidden 1152→4096）把图切成视觉 token，替换 prompt 里的 image placeholder，和文本 token 拼成一条序列；deepstack 把 vision tower 第 8/16/24 层的特征注入 text tower。
- prefill 后取 text tower 第 {1, 9, 17, 25, 33} 层 hidden，每个位置 5×4096，`fc` 投到 4096，作为 draft 的视觉-语言上下文。
- draft 不重新编码图像，也没有 visual adapter；视觉编码只在 target prefill 付一次。每轮 draft 只算 16 个 block 位置的 query（root + 15 mask），上下文 KV 由缓存的 fused hidden 投影得到；提交前缀后 KV/hidden 直接裁剪复用。
- 训练和部署走同一条 prefill 路径；图像长边 768，M-RoPE 位置在 prefill 时绑定（`VisionPositions.ar_prefill_inputs`）。
- SFT 每题一次 context forward，6 个 anchor 共享同一份 fused hidden；OPD 每轮验证 forward 的 block hidden 直接写回下一轮缓存。
- 视觉 token 量：数据里的 750×429 地图 299 个，692×171 磁铁图 110 个，536×398 表格 204 个。

| | 配置 |
|---|---|
| Target | Qwen3-VL-8B-Instruct；36 层 text，hidden 4096，32 Q / 8 KV，vocab 151936；vision 27 层，hidden 1152，patch 16，merge 2；冻结；SFT/OPD 与评测 fp32（warm 阶段 bf16） |
| Draft head | 5 层 block-diffusion，hidden 4096，32 Q / 8 KV，FFN 12288，head_dim 128，RoPE θ=1e6；block 16；mask token 151669；输入 5 层 concat 20480→4096；1.049B 可训练参数；embedding 与 lm_head 复用 target 的冻结权重；SFT/OPD max_seq 4096 |
| 解码 | budget 63，branch 8，greedy（temperature 0），max_new 256，图像长边 768 |

### 2.3 训练

三个阶段共用同一个 head 架构。

**Warm（2000 updates）**

- 初始化自文本块扩散头 Qwen3-8B-DFlash-b16。
- 数据：COCO 8000 + TextVQA 8000 = 16000 条 image+prompt，target 贪心轨迹（≤256 token）。
- 目标：variable-reveal coverage CE；随机 reveal 长度，`0.8^(k−reveal)` 加权。
- 优化：batch 1 × accum 8，lr 4e-5，AdamW（0.9, 0.95），wd 0，clip 1，bf16 target/head + fp32 CPU master；数据训 1 遍。
- 结果：dev8 SPEC=AR 8/8；Warm-2000 作为两臂共同起点。

**SFT 基线（3 epochs）**

- 数据：ScienceQA-IMG 去重后 4726 QA / 4352 图；target 贪心标签一次生成并缓存。
- 每题 6 个 anchor；block 16；reveal ~ 截断几何分布（base 0.9，1..15）；对 reveal 之后的位置做 CE，权重 `0.8^(k−reveal)`。
- 4 QA/update，lr 1e-5 constant，clip 1；3546 updates。

**OPD（3 epochs）**

- 同一 warm、同一数据/顺序/预算。
- 每题用当前 head 在 target 上完整 rollout 一条 CoT，上下文由实际提交前缀推进。
- 每轮 block forward 同时给出验证 argmax、提交决策、teacher 在提议 token 上的 log-prob；提交前缀的 KV/hidden 直接复用。
- Loss：
  - 接受位置：two-event Bernoulli forward KL，只在提交 token 上算，不做全词表 KL。
  - 首个拒绝之后的后缀：k3（reverse KL 估计），权重 `0.9^(k−1)`，流权重 ×2。
  - 分母 = 接受 token 数 + 2×后缀权重和；4 个 QA 共用一个全局分母；teacher detach；clamp [−10, 10]。
- teacher 概率取自本轮 draft 前缀上的 block forward，不额外串行重放。
- 优化：4 QA/update，lr 1e-5 constant，clip 1，AdamW（0.9, 0.95），wd 0，target fp32 / head bf16 / fp32 CPU master。

**并行与恢复**

- OPD 的 4 个 rank 各跑一个 QA slot，使用同一更新前 head；rank0 按 slot 顺序 backward，做一次 master AdamW step，再广播 head。
- 续训阶段 3 卡跑 4 个逻辑 slot（rank0 承担 slot0/3），每个 slot 独立 RNG 流。
- checkpoint 保存 head/master/Adam/4 条 RNG/游标；按 step 精确续训。

### 2.4 数据与预算

| 阶段 | 数据 | 轮数 | 更新 / QA |
|---|---|---|---|
| Warm | COCO 8000 + TextVQA 8000 | 1 pass | 2000 updates（bs 1 × accum 8） |
| SFT | ScienceQA-IMG 4726 QA / 4352 图 | 3 epochs | 3546 updates，4 QA/update，14178 QA |
| OPD | 同 SFT | 3 epochs | 3546 updates，4 QA/update，14178 QA |

- 数据审计：按图像字节和像素查重，排除 685 个跨 split 图像组；SFT/OPD 两臂使用完全相同的 QA 顺序（seed 20261013，epoch 2/3 用 seed 20261014/20261015 重新打乱）。
- prompt：题干 + 选项 + “先推理，最后 `Answer: (X)`”；不输入 gold rationale/lecture。
- 前 1 个 epoch 与后 2 个 epoch 分两次运行；step1182 的 head/master/Adam/RNG 完整迁移后继续。
- dev32 每约 200 步评一次，只看 SPEC 效率；test200 来自官方 test，排除历史 manifest 中的图像，最终用独立 AR oracle 检查逐 token。

### 2.5 数据示例

训练池按图像字节和像素去重后是 4726 QA / 4352 图；下面三题覆盖地图、示意图、表格三种图像，选项数 2–4。

**例 1 · social science / map（4 选项）**

![Montana 地图](assets/sqa_example_map.png)

- Q: What is the capital of Montana?
- Options: (A) Juneau (B) Saint Paul (C) Honolulu (D) Helena
- Answer: (D) Helena
- Prompt 388 token（299 视觉 + 89 文本）；target 写 116 token：先认出高亮州是 Montana，再逐个排除其他州的首府。

**例 2 · natural science / diagram（3 选项）**

![磁铁距离](assets/sqa_example_magnets.png)

- Q: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?
- Options: (A) The magnetic force is stronger in Pair 2. (B) The strength of the magnetic force is the same in both pairs. (C) The magnetic force is stronger in Pair 1.
- Answer: (C)
- Prompt 233 token（110 视觉 + 123 文本）；target 写 140 token：从图中读出 2 in vs 4 in，按距离判断。

**例 3 · language science / table（2 选项）**

![故事表](assets/sqa_example_table.png)

- Q: Based on the table, which story is set in the eighteenth century?
- Options: (A) The Time Machine (B) Rip Van Winkle
- Answer: (B)
- Prompt 296 token（204 视觉 + 92 文本）；target 写 104 token：直接从表格里取 "New York in the late eighteenth century" 一行。

- 训练池：natural science 2451、social science 2164、language science 111；grade 1–12；选项 2–5；图种包括地图、示意图、表格、照片。
- 答案长度约 100–150 token；很多题的答案只存在于图里（高亮州、2 in/4 in、表格单元格），draft 必须读 target 的视觉状态。
- 示例图来自 ScienceQA-IMG（CC BY-NC-SA 4.0）。

## 3. 结果

### 3.1 主结果（test200，3 epochs，fp32）

| 指标 | SFT | OPD |
|---|---:|---:|
| 有效 token/轮（题均） | 5.3666 | **6.2206** |
| 相对 SFT | — | **+0.8541**，95% CI [+0.7157, +1.0037] |
| 平均生成 token/题 | 151.35 | 151.35 |
| 验证轮数/题 | 30.5 | 27.6 |
| 答案正确 | 162/200 | 162/200 |
| SPEC = 独立 AR | 200/200 | 200/200 |

- 1182 步到 3546 步：SFT ≈4.96→5.37，OPD ≈5.55→6.22；两臂差距 +0.59→+0.85。
- OPD 的收益全部在 τ，准确数不变，输出都无损。
- test200 在 1182 已用于一次对比，3546 是在同一 200 题上的复测。

### 3.2 训练曲线（固定 dev32）

![dev32 有效 token/轮](assets/sqa_dev32_tau.png)

- SFT：4.27（200 步）→ 4.69（1182）→ 5.03（3546）。
- OPD：4.47（200 步）→ 5.09（1182）→ 5.71（3546）。
- OPD 从 200 步起就高于 SFT，差距随训练扩大。

### 3.3 速度（24 图，A6000，fp32，budget 63）

| 方法 | 完整生成 ms/题 | 生成 token/s | 相对 AR |
|---|---:|---:|---:|
| AR | 7734.4 | 18.38 | 1.000× |
| SFT | 3369.7 | 42.19 | 2.295× [2.055, 2.584] |
| OPD | 3067.4 | 46.35 | **2.522× [2.248, 2.874]** |

- SFT/OPD = **1.099× [1.063, 1.141]**；24 图中 20 图 OPD 快于 SFT，逐图中位 1.095×。
- 每轮耗时两臂相同（114.5 ms），差别在轮数：48 次生成共 1412 轮（SFT）vs 1286 轮（OPD）。
- 协议：24 张固定图、每图 2 次、每臂 2 次预热、AR 现场跑、CUDA 同步外部墙钟、所选卡无其他计算进程；144 次生成逐 token 一致。
- 计时为 batch 1 的完整生成（含 prefill），不含图片读取/processor/模型加载。

### 3.4 跨数据集与长度边界（Head 只在 ScienceQA 上训练，其余零样本）

| 数据集（prompt） | 答案长度 | τ SFT/OPD | AR/SFT | AR/OPD | SFT/OPD |
|---|---:|---:|---:|---:|---:|
| ScienceQA（CoT） | 142.2 | 4.83 / 5.31 | 2.295× | 2.522× | **1.099×** |
| DocVQA（CoT） | 139.5 | 4.59 / 4.80 | 2.144× | 2.221× | 1.036× |
| MathVista（CoT） | 200.4 | 4.34 / 4.44 | 2.154× | 2.199× | 1.021× |
| ShareGPT4V（caption） | 252.6 | 3.41 / 3.46 | 1.782× | 1.821× | 1.022× |
| OCRBench（短答） | 9.6 | 3.08 / 3.04 | 1.194× | 1.204× | 1.008× |
| TextVQA（短答） | 4.3 | 3.12 / 3.22 | 1.055× | 1.050× | 0.995× |
| ChartQA（短答） | 4.6 | 3.55 / 3.33 | 1.093× | 1.022× | 0.935× |

- **AR/方法**由答案长度和 prefill 占比决定：4–10 token 短答约 1.0×，140–250 token 的 CoT/caption 为 1.8–2.5×；prefill 占 73–82% 时 decode 收益被摊薄。
- **SFT/OPD** 基本等于 τ 之比：两臂每轮成本相同，τ 高多少就快多少。
- τ 由任务形态决定：grounded、结构化的答案 τ 高（SQA 5.31），自由 caption τ 低（ShareGPT4V 3.46）。
- DocVQA 同图、同 head，只改 prompt：答案 6.2→139.5 token，AR/OPD 1.14×→2.22×，SFT/OPD 1.01×→1.04×。
- max_new 256→512 只增加答案长度（SQA +6%，caption +47%），τ 和加速比基本不变或略降。

### 3.5 训练成本

- 续训两个 epoch：SFT 6.37 GPU-h（单卡），OPD 53.48 GPU-h（3 卡），约 8.4×。
- OPD 每题要跑完整 rollout（随 τ 提高，约 30–50 轮 target block forward）；SFT 每题只做 1 次 context forward + 1 次 6-anchor 的 head forward。

## 4. 结论

- 块并行 draft 在 ScienceQA 上把 τ 从 5.37 提到 6.22，端到端 2.52×（AR）、1.10×（同预算 SFT），答案不变、输出逐 token 不变。
- OPD 的收益主要来自训练分布与推理一致：SFT 在 target 轨迹上 teacher forcing，OPD 在自己诱导的轨迹上训练，并对被拒后缀做加权纠正；同预算 SFT 达不到同样的 τ。
- 收益上限由答案长度和 prefill 占比决定。短答集上 decode 只占几个百分点到 27%，投机解码没有可摊薄的循环，这是任务形态决定的边界。
- OPD 训练成本约 8× SFT，换推理端 1.10×；训练成本是下一步要压的部分。

## 5. 下一步

- 短答/图表域的 domain warm + head 训练，看 τ 能否从 3.3 提到 5 附近，并观察 SFT/OPD 在短答集上是否仍有区分度。
- 降 OPD 训练成本：rollout 复用、减少每 update 的验证轮数、多卡异步调度。
- 新的独立 test 集 + bf16 部署 + batch>1 吞吐，补上服务侧数字。
