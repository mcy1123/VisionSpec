# ScienceQA 最终 SFT / OPD 实测推理速度（2026-10-20）

使用完整一轮训练的两臂 step1182，在同一张 RTX A6000 上测独立 AR、SFT-SPEC、OPD-SPEC。固定抽取此前 confirm200 的 24 张图像，随机种子 20261020，每张重复两次。六种三臂运行顺序均衡分配，第二次反转顺序；预热每臂两次，不计入结果。

目标与 Head 均 fp32，batch=1，图像长边上限 768，贪心生成最多 256 个新 token，budget=63、branch_k=8、immediate EOS，与此前最终评测配置一致。训练时 Head bf16；此处沿用最终部署评测的 fp32 Head。每次重新 prefill，无 KV 复用。每次 AR 都现场运行，缓存只用来核对输出 token，绝不复用缓存计时。

主计时使用 CUDA 同步的外部墙钟，包含视觉与文本 prefill、首 token、全部草稿生成和验证，排除图片读取、processor、输入搬运、模型加载和 Head 搬运。Head 切换在计时外进行，每次仅一个 Head 驻留 GPU；AR 测量也保留同样的驻留 Head。

| 方法 | 平均完整生成耗时 ms/题 | 完整生成 token/s | AR / 方法 |
|---|---:|---:|---:|
| AR | 7686.72 | 18.50 | 1.000× |
| SFT | 3437.72 | 41.35 | 2.236× |
| OPD | 3220.16 | 44.15 | 2.387× |

| 配对比较 | 总耗时比 | 图像配对 bootstrap 95% CI |
|---|---:|---:|
| ar_over_sft | 2.236× | [2.038, 2.485] |
| ar_over_opd | 2.387× | [2.134, 2.711] |
| sft_over_opd | 1.068× | [1.038, 1.104] |

全部 144 次正式生成与既有独立 AR token 缓存逐 token 一致：True。每张图的两次重复先取平均，再按图像配对重采样 10,000 次；加速比为总耗时之比，并非逐题比值的平均。

所用 GPU 上采样到其他计算进程：0。其他卡仍有任务，本次属于单张空闲卡上的实测，整机不是完全独占环境。进程占用每张图完成后采样，不能排除采样间短任务。

辅助 decode-only 耗时保留在原始记录中，沿用现有 decoder 口径；SPEC 排除首次草稿准备、AR 排除 prefill，因此不作为主要加速结论。底层 greedy_generate 的 ms_per_token_wall 已含 prefill，使用时须避免再加一次 prefill；原 eval_sqa.py 使用其 decode-only ms_per_token 另加 prefill。本次直接测外部完整生成墙钟，未修改 decoder 或历史评测结果。

结果范围：24 张固定子集、单 GPU、当前 PyTorch/SDPA 路径与 fp32 配置；不外推为全 200 题或生产服务速度，不涉及量化、批处理、网络和排队。接受长度的增加不是同百分比的墙钟加速。未调整树或模型迎合计时。

产物：`/mnt/local/mcy/GLANCE/results/scienceqa_speed_20261020`。`PLAN.json` 固定采样和运行顺序，`CHECKPOINTS.json`/`SOURCE_SHA256.json` 固定身份，`measurements.jsonl`/`.csv` 保留每次耗时和输出，`LOAD_SAMPLES.json` 保留占用与频率，`RESULT.json` 为汇总。

复现命令（改用新的输出目录，禁止覆盖已有记录）：
```sh
env CUDA_VISIBLE_DEVICES=GPU-8af87d4f-b2b2-404f-e7b2-6b39cbedcd55 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false /home/mcy/.conda/envs/fin_research/bin/python /home/mcy/Projects/GLANCE/repro/scienceqa_image/benchmark_sqa_speed.py --run /mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_restart_20261019_180942 --out /mnt/local/mcy/GLANCE/results/scienceqa_speed_REPEAT --uuid GPU-8af87d4f-b2b2-404f-e7b2-6b39cbedcd55
```

> **2026-10-21 复测（step3546）**：同一协议、同一 24 图、同一 fp32 配置，把两臂换到两 epoch 续训的终点重测。AR 7734.44 ms/题（18.38 tok/s）、SFT 3369.65（42.19）、OPD 3067.38（46.35）；AR/SFT 2.295× [2.055, 2.584]、AR/OPD 2.522× [2.248, 2.874]、SFT/OPD 1.099× [1.063, 1.141]；144 次生成全部逐 token 无损，所选 GPU 外进程采样为空。与本文 step1182 结果相比：AR/OPD 2.387→2.522、SFT/OPD 1.068→1.099。细节见[三卡续训报告](2026-10-20-scienceqa-three-card-continuation.md)「加速比复测」。产物：`/mnt/local/mcy/GLANCE/results/scienceqa_speed_step3546_20261021`；复现脚本副本在运行目录 `analysis_speed_20261021/benchmark_speed_step3546.py`。
```
