# 视频 OPD 的视觉依赖复现（2026-11-04）

状态：已完成。结论：**图像侧的阴性结论在视频上复现，而且更强**。

## 1. 动机

图像 ScienceQA 上，位置加权、cutout、mismatch、2×2/3×3 clue、target-attention 权重六种输入侧干预都没有统计支持的 high-g 增益。这里在已训练好的视频 OPD checkpoint 上做同样的视觉依赖测试：

- 数据：VideoDetailCaption `video_fresh30`（30 条独立视频，16 帧，384 长边，AR 平均 271 token）。
- Head：warm（`taskB_fair_qa_20261015T085742/video/training/sft/step0`）与 OPD 3e（`video_caption_opd_20261026/run_stage2/step220`）。
- 对照：真实视频 vs 同尺寸全灰 16 帧视频（blank），prompt/帧数/fps/分辨率不变。

## 2. Blank-frame 诊断

对每条 fresh30，用 OPD3e 的 AR answer 做 teacher-forcing，真实帧和灰帧各算一次 target log-prob：

\[
g_i=\max\left(0,\ \log p_{\text{real}}(y_i)-\log p_{\text{blank}}(y_i)\right)
\]

每条视频取 answer token 的 `g_mean`，与 warm/OPD3e 的 per-video τ 连接：

- `g_mean` 均值 **0.6039**，中位 0.5809（和图像的 0.607 几乎一样）。
- warm τ 3.7266 → OPD3e 3.7841，Δ=**+0.0575**，95% CI **[+0.0123,+0.1018]**。
- 高 g 半（n=15）：Δ=**+0.0645**，CI [−0.0088,+0.1385]（跨零）。
- 低 g 半（n=15）：Δ=**+0.0504**，CI [+0.0028,+0.0995]（显著）。
- `Spearman(g, Δ)=−0.0375`。

**OPD 的增益没有集中在视觉依赖的视频上**；高 g 组不显著，低 g 组反而显著。与图像诊断一致。

## 3. Blank-video 行为对照

把 fresh30 的全部帧替换为同尺寸灰帧（30 条 blank mp4），用同一 eval 协议（fp32、b63k8、AR 逐 token 门禁）分别评 warm 和 OPD3e。两臂 **30/30 逐 token 无损**。

| arm | 真实 τ | blank τ | 真实−blank（visual reliance） |
|---|---:|---:|---:|
| warm | 3.7266 | 3.0747 | 0.6519，CI [0.5623,0.7398] |
| OPD 3e | 3.7841 | 3.2826 | 0.5015，CI [0.3979,0.6056] |
| OPD−warm | **+0.0575**，CI [+0.0134,+0.1024] | **+0.2079**，CI [+0.1945,+0.2210] | **−0.1504**，CI [−0.1945,−0.1057] |

- OPD 在**灰视频**上提升 **+0.2079**，远大于真实视频的 **+0.0575**。
- 因此 OPD 的视觉依赖度（真实 τ − blank τ）从 0.652 **降到** 0.502；difference-in-differences = **−0.1504**，CI 完全在负侧。
- 也就是说：**OPD 主要改善的是“没有视觉信息时的语言先验/通用解码”，不是视觉 grounding。** 这比图像侧的 high-g 不显著更直接、更强。

## 4. 结论

视频上复现并放大了图像的阴性结论：

1. OPD 的 +0.0575 真实视频增益不是视觉特异性的；blank 视频上增益 +0.2079，视觉依赖反而下降。
2. 高 g 视频没有显著增益，低 g 视频显著；`Spearman` 近零偏负。
3. 因 n=30 且效应方向相反，没有再烧 3 小时跑视频加权训练臂：blank-video 行为对照已经比一个 underpowered 的加权臂更有判别力；视频加权臂对 <0.05 的效应没有检验力。
4. **输入侧视觉加权/扰动/clue 这条路线在图像和视频上都不成立。** 若继续视频方向，应换目标（visual attention 对齐/draft 侧 copy）或换视觉证据更必要、独立集更大的数据，而不是重复输入侧权重。

## 5. 边界

- 单 seed、n=30 视频、一个独立集 fresh30；`video_heldout_30` 已污染，不能当独立确认。
- 灰帧是 OOD 的 blank 输入；它测的是“没有视觉内容时的语言先验解码”，不能证明模型对所有视觉信息都不敏感。
- blank eval 的 AR oracle 是各 head 自己的 AR；两臂都 30/30 无损，τ 可比。
- 没有跑视频训练干预；这是有意的取舍，原因见 §4.3。

## 6. 产物

- 诊断：`repro/diagnostics/visual_dependence_video.py`
- 结果：`/mnt/local3/mcy/GLANCE/results/video_opd_visual_20261105/`
  - `blank_videos/`（30 条灰视频）
  - `video_fresh30_blank.parquet`
  - `eval/fresh30_blank.json`、`fresh30_blank.csv`
  - `eval/warm_blank.json`、`eval/opd330_blank.json`
  - `eval/warm_opd_real_blank_pairs.csv`
- 真实/blank 评测脚本：`run_blank_evals.sh`
- 复用真实 τ：`video_caption_opd_20261026/eval/{warm_fp32,opd330}.json`
- 代码分支 `exp/visual-opd`；`main` 未动。
