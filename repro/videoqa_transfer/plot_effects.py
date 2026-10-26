"""Standalone research figure of all completed exploratory contrasts."""
import pathlib,sys,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
R=pathlib.Path(sys.argv[1]);s=pd.read_csv(R/'all_summary_by_dataset_format.csv');p=pd.read_csv(R/'performance_paired_video_ci.csv');order={'MSVD-QA':0,'MSRVTT-QA':1,'NExT-QA':2};s=s.assign(order=s.dataset.map(order)).sort_values(['order','format']).reset_index(drop=True)
fig,axes=plt.subplots(1,2,figsize=(10,2+len(s)*.5),sharey=True)
labels=[]
for i,r in s.iterrows():
 name='NExT-QA (MC)' if r.dataset=='NExT-QA' else r.dataset
 labels.append(f'{name} | {r["format"]} | L={r.AR_length_mean:.1f}')
 for ax,metric in zip(axes,['tau','decode_ms_token']):
  x=p[(p.dataset==r.dataset)&(p['format']==r['format'])&(p.comparison=='C2-C1')&(p.metric==metric)].iloc[0];mu=x['mean'];ax.errorbar(mu,i,xerr=[[mu-x.ci_low],[x.ci_high-mu]],fmt='o' if r['format']=='A' else 's',capsize=4,color='#245889');ax.axhspan(i-.4,i+.4,color='#f5f5f5',zorder=-2)
for ax,title in zip(axes,['Acceptance length: C2 - C1 (higher is better)','Decode ms/token: C2 - C1 (lower is better)']):
 ax.axvline(0,color='black',lw=1,ls='--');ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(nbins=5));ax.tick_params(axis='x',labelsize=9)
axes[0].set_yticks(range(len(labels)),labels);axes[0].invert_yaxis();fig.suptitle('Fixed C4 checkpoints: exploratory video-QA transfer',fontsize=13);fig.text(.5,.01,'20 videos/dataset; paired video bootstrap 95% CI, 2000 draws; no multiplicity adjustment. A: short answer, B: brief evidence.\nNExT timing spans a resource-gated restart; one boundary QA has uncertain timing contamination.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(R/'paired_effects.svg');fig.savefig(R/'paired_effects.png',dpi=180)
