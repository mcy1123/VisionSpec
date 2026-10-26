"""Additional prespecified video-level length/uncertainty/performance tables."""
import argparse,json,pathlib,sys,math,statistics as st,collections,csv
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from protocol import paired
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();R=pathlib.Path(a.run)
import pandas as pd,numpy as np
parts=[]
for sub in ['', 'msrvtt','nextqa_mc']:
 d=R/sub
 if (d/'full/gates.json').exists() and json.loads((d/'full/gates.json').read_text()).get('pass'):
  df=pd.read_csv(d/'per_question.csv',dtype={'question_id':str,'video_id':str});df['run_directory']=str(d);parts.append(df)
if not parts:raise RuntimeError('No complete dataset results')
q=pd.concat(parts,ignore_index=True);q['spec_model_wall_ms']=q.spec_model_wall_ms_token*q.AR_length;q['spec_decode_ms']=q.spec_decode_ms_token*q.AR_length;q.to_csv(R/'all_per_question.csv',index=False)
v=q.groupby(['dataset','format','video_id','arm'],sort=True).agg(n_QA=('question_id','size'),tau=('tau','mean'),tau_effective=('tau_effective','mean'),AR_length=('AR_length','mean'),entropy=('target_entropy','mean'),first_entropy=('first_draft_target_entropy','mean'),top8=('top8_cover','mean'),decode_ms_token=('spec_decode_ms_token','mean'),wall_ms_token=('spec_model_wall_ms_token','mean'),first_draft_is_eos=('first_draft_is_eos','mean'),wall_ms=('spec_model_wall_ms','mean'),decode_ms=('spec_decode_ms','mean')).reset_index();v.to_csv(R/'all_per_video.csv',index=False)
perf=[];pairrows=[];summ=[]
for (ds,fmt),g in v.groupby(['dataset','format'],sort=True):
 s={'dataset':ds,'format':fmt,'n_videos':g.video_id.nunique(),'n_QA':int(g[g.arm=='C0'].n_QA.sum()),'AR_length_mean':float(g[g.arm=='C0'].AR_length.mean())}
 qq=q[(q.dataset==ds)&(q['format']==fmt)];q0=qq[qq.arm=='C0'];s.update(AR_length_median=float(q0.AR_length.median()),fraction_short_le8=float((q0.AR_length<=8).mean()),fraction_first_draft_EOS=float(q0.first_draft_is_eos.mean()),target_entropy=float(g[g.arm=='C0'].entropy.mean()),SPEC_AR=bool(qq.lossless.all()),eos_fraction=float(q0.hit_eos.mean()),MC_accuracy_AUX=float(q0.MC_correct.mean()) if q0.MC_correct.notna().any() else None,normalized_EM_AUX=float(q0.normalized_short_answer_match.mean()) if q0.MC_correct.isna().all() and q0.normalized_short_answer_match.notna().any() else None)
 for arm in ['C0','C1','C2']:
  x=g[g.arm==arm];z=qq[qq.arm==arm];s.update({f'tau_{arm}':float(x.tau.mean()),f'tau_effective_{arm}':float(x.tau_effective.mean()),f'decode_ms_token_{arm}':float(x.decode_ms_token.mean()),f'wall_ms_token_{arm}':float(x.wall_ms_token.mean()),f'wall_ms_{arm}':float(x.wall_ms.mean()),f'decode_ms_{arm}':float(x.decode_ms.mean()),f'top8_{arm}':float(x.top8.mean()),f'one_round_fraction_{arm}':float((z.rounds<=1).mean()),f'pooled_tau_{arm}':float((z.tau*z.rounds).sum()/z.rounds.sum())})
 for left,right in [('C2','C1'),('C2','C0'),('C1','C0')]:
  for metric in ['tau','tau_effective','decode_ms_token','wall_ms_token','decode_ms','wall_ms']:
   a1=g[g.arm==left].set_index('video_id')[metric].to_dict();b1=g[g.arm==right].set_index('video_id')[metric].to_dict();b=paired(a1,b1);perf.append({'dataset':ds,'format':fmt,'comparison':left+'-'+right,'metric':metric,**b})
   if metric=='tau':
    for k,val in b.items():s[f'{left}-{right}_{k}']=val
    for vid in a1:pairrows.append({'dataset':ds,'format':fmt,'video_id':vid,'comparison':left+'-'+right,'delta_tau':a1[vid]-b1[vid]})
 summ.append(s)
pd.DataFrame(summ).to_csv(R/'all_summary_by_dataset_format.csv',index=False);pd.DataFrame(perf).to_csv(R/'performance_paired_video_ci.csv',index=False);pd.DataFrame(pairrows).to_csv(R/'all_paired_differences.csv',index=False)
# Auxiliary groups are fixed AR-length bins, never redefine or replace the main sample.
aux=[]
for (ds,fmt,lb,arm),g in q.groupby(['dataset','format','length_bin','arm']):
 vv=g.groupby('video_id').tau.mean();aux.append({'dataset':ds,'format':fmt,'length_bin':lb,'arm':arm,'n_videos':len(vv),'n_QA':len(g),'mean_video_tau':vv.mean(),'mean_AR_length':g.AR_length.mean(),'mean_target_entropy':g.groupby('video_id').target_entropy.mean().mean()})
pd.DataFrame(aux).to_csv(R/'length_strata.csv',index=False)
# Descriptive video correlations, not causal attribution or independent token tests.
cor=[]
for (ds,fmt),g in v.groupby(['dataset','format']):
 pivot=g.pivot(index='video_id',columns='arm',values='tau');c0=g[g.arm=='C0'].set_index('video_id');delta=pivot.C2-pivot.C1
 for x in ['AR_length','entropy','first_entropy']:
  corr=float(delta.corr(c0[x],method='spearman')) if delta.nunique()>1 and c0[x].nunique()>1 else None
  cor.append({'dataset':ds,'format':fmt,'covariate':x,'spearman_deltaC2C1':corr,'n_videos':len(delta),'exploratory':True})
pd.DataFrame(cor).to_csv(R/'uncertainty_length_correlations.csv',index=False)
# Explicit per-video counterexamples to any overall ranking; retain every sign.
pd.DataFrame(pairrows).query("comparison == 'C2-C1'").sort_values(['dataset','format','delta_tau']).to_csv(R/'per_video_counterexamples.csv',index=False)
# Exact duplicates by raw bytes are detectable; absence is not absence of shared source.
raw=[]
for sub in ['', 'msrvtt','nextqa_mc']:
 f=R/sub/'dataset_manifest_20videos.jsonl'
 if f.exists():raw+=list(map(json.loads,f.read_text().splitlines()))
unique={(x['dataset'],x['video_id']):x for x in raw};byhash=collections.defaultdict(list)
for k,x in unique.items():byhash[x['hash']].append(k)
(R/'cross_dataset_overlap.json').write_text(json.dumps({'exact_raw_file_duplicate_groups':[x for x in byhash.values() if len(x)>1],'raw_videos':len(unique),'source_lineage_status':'overlap_unknown except IDs documented in manifests; exact hash inequality does not prove no common source'},indent=2))
print(pd.DataFrame(summ)[['dataset','format','n_videos','n_QA','AR_length_mean','tau_C0','tau_C1','tau_C2','C2-C1_mean','C2-C1_ci_low','C2-C1_ci_high']].to_string(index=False))
