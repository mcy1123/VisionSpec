"""Render completed dataset numerical tables without deciding the research conclusion."""
import pathlib,argparse,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();R=pathlib.Path(a.run)
s=pd.read_csv(R/'all_summary_by_dataset_format.csv');q=pd.read_csv(R/'all_per_question.csv',dtype={'question_id':str,'video_id':str});perf=pd.read_csv(R/'performance_paired_video_ci.csv')
def val(x):return 'N/A' if pd.isna(x) else f'{x:.4f}' if isinstance(x,float) else str(x)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(val,r))+' |\n' for r in rows)
def ci(r,p):return f'{r[p+"_mean"]:+.4f} [{r[p+"_ci_low"]:+.4f}, {r[p+"_ci_high"]:+.4f}]'
text='# 实测数值附表\n\n均为冻结20视频开发集；95% CI按视频bootstrap 2000次、seed20261013，未作多重检验校正。所有A/B样本保留。\n\n'
rows=[]
for _,r in s.iterrows():
 z=q[(q.dataset==r.dataset)&(q['format']==r['format'])&(q.arm=='C0')];rows.append([r.dataset,r['format'],int(r.n_videos),int(r.n_QA),f'{r.AR_length_mean:.2f}/{r.AR_length_median:.1f}',f'{r.fraction_short_le8:.1%}/{r.one_round_fraction_C0:.1%}',f'{r.tau_C0:.4f}/{r.tau_C1:.4f}/{r.tau_C2:.4f}',ci(r,'C2-C1'),ci(r,'C2-C0'),f'{r.top8_C0:.3f}/{r.top8_C1:.3f}/{r.top8_C2:.3f}',r.target_entropy,f'{r.decode_ms_token_C0:.2f}/{r.decode_ms_token_C1:.2f}/{r.decode_ms_token_C2:.2f}',r.SPEC_AR])
text+=table(['dataset','format','视频','QA','AR长度均值/中位','≤8/单轮(C0)','τ C0/C1/C2','C2−C1 [CI]','C2−C0 [CI]','top8 C0/C1/C2','target熵(nats)','decode ms/token C0/C1/C2','SPEC=AR'],rows)
text+='\n## 三种配对比较及方向计数\n\n';rows=[]
for _,r in s.iterrows():
 for p in ['C2-C1','C2-C0','C1-C0']:rows.append([r.dataset,r['format'],p,ci(r,p),int(r[p+'_positive']),int(r[p+'_negative']),int(r[p+'_unchanged'])])
text+=table(['dataset','format','比较','Δτ [CI]','正','负','相等'],rows)
text+='\n## 实际性能\n\n干净计时每输入/臂重复2次；先按题平均，再按视频平均。下面C2−C1负值表示C2更快。模型调用wall包含prefill及首轮draft/build，**不包含processor或CPU→GPU输入拷贝，不是完整端到端**。生产decode-only还排除了首轮draft/build，短输出尤其敏感。NExT跨资源门禁恢复批次，其中1题时延可能受中断边界干扰；计时不作稳定加速证明。\n\n';rows=[]
for _,r in s.iterrows():
 for metric,key in [('decode_ms_token','decode_ms_token'),('wall_ms_token','wall_ms_token'),('wall_ms','wall_ms')]:
  d=perf[(perf.dataset==r.dataset)&(perf['format']==r['format'])&(perf.comparison=='C2-C1')&(perf.metric==metric)].iloc[0];rows.append([r.dataset,r['format'],metric,r[key+'_C0'],r[key+'_C1'],r[key+'_C2'],f'{d["mean"]:+.4f} [{d.ci_low:+.4f}, {d.ci_high:+.4f}]'])
text+=table(['dataset','format','计时口径（wall_ms为单次总ms）','C0','C1','C2','C2−C1 [视频CI]'],rows)
text+='\n## 长度、轮次与质量辅助表\n\nMC按保留选项的字母解析计分；开放式短答规范化字面匹配仅作辅助，不能替代语义正确性。三臂输出相同，因此每项质量值相同。\n\n';rows=[]
for _,r in s.iterrows():
 z=q[(q.dataset==r.dataset)&(q['format']==r['format'])&(q.arm=='C0')];rows.append([r.dataset,r['format'],r.fraction_first_draft_EOS,z.rounds.mean(),z.rounds.median(),r.eos_fraction,r.MC_accuracy_AUX,r.normalized_EM_AUX,z.target_entropy_excluding_eos.mean(),z.first_answer_target_entropy.mean()])
text+=table(['dataset','format','首draft预测EOS占比','C0轮数均值','轮数中位','EOS占比','MC正确率','开放短答EM辅助','排EOS熵','首答案token熵'],rows)
text+='\n## 有效长度与pooled口径\n\nτ沿用生产裁剪前accepted+1；effective为实际输出token/轮数，root、draft、bonus无重复计数。两者分别报告，不能混用。\n\n';rows=[]
for _,r in s.iterrows():
 for arm in ['C0','C1','C2']:rows.append([r.dataset,r['format'],arm,r['tau_'+arm],r['tau_effective_'+arm],r['pooled_tau_'+arm],r['one_round_fraction_'+arm]])
text+=table(['dataset','format','臂','mean per-video τ','mean effective','pooled τ','单轮占比'],rows)
(R/'numerical_tables.md').write_text(text)
