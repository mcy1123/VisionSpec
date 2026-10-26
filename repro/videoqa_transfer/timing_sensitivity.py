"""Posthoc timing robustness checks; never replace the frozen primary sample."""
import pathlib,sys,json,pandas as pd
from protocol import paired
R=pathlib.Path(sys.argv[1])
q=pd.read_csv(R/'all_per_question.csv',dtype={'video_id':str,'question_id':str})
t=pd.read_csv(R/'timing_repeatability.csv',dtype={'video_id':str,'question_id':str})
out=[]
for (ds,fmt),g in t.groupby(['dataset','format']):
 for metric in ['decode','wall']:
  for rep in [1,2]:
   v=g.groupby(['video_id','arm'])[f'{metric}_repeat{rep}'].mean().unstack()
   out.append({'dataset':ds,'format':fmt,'metric':metric+'_ms_token','clean_repeat':rep,'analysis':'posthoc repeat stability; all frozen videos','exploratory':True,**paired(v.C2.to_dict(),v.C1.to_dict())})
pd.DataFrame(out).to_csv(R/'timing_repeat_contrast_aux.csv',index=False)
g=q[(q.dataset=='NExT-QA')&(q['format']=='B')&(~q.timing_boundary_uncertain)]
z=[]
for metric in ['spec_decode_ms_token','spec_model_wall_ms_token','spec_model_wall_ms']:
 v=g.groupby(['video_id','arm'])[metric].mean().unstack()
 z.append({'metric':metric,'n_QA':len(g[g.arm=='C0']),'posthoc':True,'note':'Sensitivity excluding one resource-boundary QA from timing only; primary sample and tau unchanged',**paired(v.C2.to_dict(),v.C1.to_dict())})
(R/'nextqa_mc/timing_boundary_sensitivity_aux.json').write_text(json.dumps(z,indent=2))
