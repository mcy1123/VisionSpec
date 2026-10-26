# 训练阶段严格重放断言清理

用户要求去除训练时不必要的严格相等检查，并排查类似条件。已确认当前无训练进程后修改；未启动训练、未重置预算。源码回滚快照：`/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_20261019/training_assertion_cleanup_20261019_180709/source_before.tar.gz`；完整分类审计：[training_assertion_cleanup_20261019_180709/AUDIT.json](/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_20261019/training_assertion_cleanup_20261019_180709/AUDIT.json)。

SFT 去除 Target 全序列重放 argmax 与固定 AR 标签必须完全相同的断言，并去除专用于该检查的 hidden hook 和 lm_head 投影；固定训练标签、上下文前向、CE、anchor、精度和optimizer均保持原语义。

OPD 去除 scorer 中重算 argmax 与已记录 proposal 必须一致的立即／deferred断言。仍按记录的 proposal token 计算 student logp 和 rejected loss，没有强行改写 logits、替换 token 或丢弃样本。另取消零梯度范数报错，只保留非有限梯度报错。

已排查 active SFT、OPD runtime/parallel trainer、trace contracts及full_epoch controller。此前teacher／梯度门禁、恢复对照、训练期SPEC=AR已不在active控制流程执行，历史诊断函数与旧报告保留。保留数据与标签对齐、missing records、长度／形状、NaN/Inf、梯度图、四卡Head版本和trace身份、KV长度、跨rank池化分母、checkpoint完整性与防覆盖检查。这些用于防止损坏输入或状态下错误训练，不要求训练输出与独立AR相等。最终独立SPEC=AR检查不变。

简短CPU检查：记录的proposal与重算argmax刻意不一致时，scorer在普通及旧deferred调用方式下均可求loss和梯度，与独立fixed-token scorer逐位一致；SFT多余hook和重放投影已移除，仍使用原targets作为标签；三个修改模块语法编译通过。没有加载GPU模型或做optimizer更新。

当前历史run仍有101/101次optimizer尝试／完成，只有step0 checkpoint。修改代码不能恢复未保存的101步；本次未重跑、未退款或增加预算，也未修改旧日志／checkpoint。原入口的Source manifest已同步本次源码，原manifest在快照目录中保留；对旧run执行resume仍受到已消费但未保存更新的保护，不能冒充从step101续训。
