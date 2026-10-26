"""Secondary descriptive tables; no sample filtering in the main results."""
import pathlib,argparse,json,statistics as st
import pandas as pd
from protocol import paired
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();R=pathlib.Path(a.run);q=pd.read_csv(R/'all_per_question.csv',dtype={'question_id':str,'video_id':str})
rows=[]
for (ds,fmt),g in q.groupby(['dataset','format']):
 for arm,gg in g.groupby('arm'):
  v=gg.groupby('video_id')[['AR_length','target_entropy','target_entropy_excluding_eos','tau','top8_cover_non_eos','first_predicted_top8_non_eos']].mean()
  for cov in ['AR_length','target_entropy','target_entropy_excluding_eos']:
   rows.append(dict(dataset=ds,format=fmt,arm=arm,covariate=cov,n_videos=len(v),spearman_with_tau=v[cov].corr(v.tau,method='spearman') if v[cov].nunique()>1 and v.tau.nunique()>1 else None,exploratory=True))
pd.DataFrame(rows).to_csv(R/'task_covariate_correlations.csv',index=False)
rows=[]
for ds,g in q.groupby('dataset'):
 for arm,gg in g.groupby('arm'):
  v=gg.groupby(['video_id','format']).mean(numeric_only=True).reset_index()
  for metric in ['tau','tau_effective','AR_length','target_entropy','target_entropy_excluding_eos','rounds','spec_decode_ms_token','spec_model_wall_ms_token','top8_cover_non_eos']:
   x=v[v['format']=='B'].set_index('video_id')[metric].dropna().to_dict();y=v[v['format']=='A'].set_index('video_id')[metric].dropna().to_dict();keys=set(x)&set(y)
   if keys:rows.append({'dataset':ds,'arm':arm,'metric':metric,'comparison':'B-A','exploratory':True,**paired({k:x[k] for k in keys},{k:y[k] for k in keys})})
pd.DataFrame(rows).to_csv(R/'format_B_minus_A_video_ci.csv',index=False)
rows=[]
for (ds,fmt,typ),g in q.groupby(['dataset','format','question_type']):
 v=g.groupby(['video_id','arm']).tau.mean().unstack();b=paired(v.C2.to_dict(),v.C1.to_dict());rows.append({'dataset':ds,'format':fmt,'question_type':typ,'n_QA':len(g[g.arm=='C0']),'exploratory':True,**b})
pd.DataFrame(rows).to_csv(R/'question_type_paired_aux.csv',index=False)
rows=[]
for (ds,fmt,lb),g in q.groupby(['dataset','format','length_bin']):
 v=g.groupby(['video_id','arm']).tau.mean().unstack();rows.append({'dataset':ds,'format':fmt,'length_bin':lb,'n_QA':len(g[g.arm=='C0']),**paired(v.C2.to_dict(),v.C1.to_dict())})
pd.DataFrame(rows).to_csv(R/'length_bin_paired_aux.csv',index=False)
rows=[];checks=[]
for sub in ['', 'msrvtt','nextqa_mc']:
 d=R/sub
 if not (d/'full/gates.json').exists() or not json.loads((d/'full/gates.json').read_text()).get('pass'):continue
 for r in map(json.loads,(d/'full/results.jsonl').read_text().splitlines()):
  c=r['clean_repeats'];x=[z['stats']['ms_per_token'] for z in c];y=[1000*z['wall_s']/len(z['tokens']) for z in c];rows.append(dict(dataset=r['dataset'],format=r['mode'],video_id=r['video_id'],question_id=r['question_id'],arm=r['arm'],decode_repeat_relative_range=(max(x)-min(x))/st.mean(x),wall_repeat_relative_range=(max(y)-min(y))/st.mean(y),decode_repeat1=x[0],decode_repeat2=x[1],wall_repeat1=y[0],wall_repeat2=y[1]))
  checks.append({'dataset':r['dataset'],'format':r['mode'],'video_id':r['video_id'],'question_id':r['question_id'],'arm':r['arm'],'input_key':r['input_key'],'diagnostic_SPEC_AR':r['lossless'],'clean_repeat_checks':len(c),'tokens_diagnostic':r['diagnostic']['tokens'],'tokens_clean_identical':all(z['tokens']==r['diagnostic']['tokens'] for z in c)})
f=pd.DataFrame(rows);f.to_csv(R/'timing_repeatability.csv',index=False);f.groupby(['dataset','format','arm'])[['decode_repeat_relative_range','wall_repeat_relative_range']].agg(['mean','median','max']).to_csv(R/'timing_repeatability_summary.csv');(R/'token_gate_index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in checks))
# Production stopping boundary audit, descriptive only; never subtract work from measured speed.
q.groupby(['dataset','format','arm']).agg(n_QA=('question_id','size'),EOS_bonus_fraction=('eos_was_bonus','mean'),mean_rounds_at_or_after_EOS=('rounds_at_or_after_eos','mean'),mean_clipped_draft=('clipped_draft','mean'),mean_actual_rounds=('rounds','mean')).to_csv(R/'eos_boundary_summary.csv')
# Exact accounting partition, not a counterfactual decoder or a measured EOS fix.
savings=[]
for (ds,fmt),g in q.groupby(['dataset','format']):
 for left,right in [('C1','C2'),('C0','C2')]:
  a1=g[g.arm==left].set_index(['video_id','question_id']);b1=g[g.arm==right].set_index(['video_id','question_id']);assert set(a1.index)==set(b1.index)
  dr=a1['rounds']-b1['rounds'];dt=a1['rounds_at_or_after_eos']-b1['rounds_at_or_after_eos'];savings.append({'dataset':ds,'format':fmt,'comparison':left+' minus '+right,'n_QA':len(dr),'total_verification_rounds_saved':float(dr.sum()),'terminal_root_EOS_rounds_saved':float(dt.sum()),'remaining_rounds_saved':float((dr-dt).sum()),'note':'accounting decomposition of actual original-decoder runs, not counterfactual fixed-decoder performance'})
pd.DataFrame(savings).to_csv(R/'round_savings_decomposition.csv',index=False)
# Segment means are descriptive; a video spanning a resource restart appears in both.
segments=[]
for (ds,fmt,segment),g in q.groupby(['dataset','format','execution_segment']):
 v=g.groupby(['video_id','arm'])[['spec_decode_ms_token','spec_model_wall_ms_token']].mean()
 row={'dataset':ds,'format':fmt,'execution_segment':segment,'n_videos':g.video_id.nunique(),'n_QA':len(g[g.arm=='C0']),'boundary_uncertain_QA':int(g[g.arm=='C0'].timing_boundary_uncertain.sum())}
 for metric in ['spec_decode_ms_token','spec_model_wall_ms_token']:
  for arm in ['C0','C1','C2']:row[metric+'_'+arm]=v.xs(arm,level='arm')[metric].mean()
  row[metric+'_C2-C1']=row[metric+'_C2']-row[metric+'_C1']
 segments.append(row)
pd.DataFrame(segments).to_csv(R/'timing_segment_summary.csv',index=False)
