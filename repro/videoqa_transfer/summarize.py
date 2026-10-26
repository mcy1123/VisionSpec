"""Video-clustered paired summaries; no failed/partial pool silently treated as full."""
import argparse,json,pathlib,sys,csv,collections,statistics as st
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from protocol import paired,normalize_answer,choice_answer
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',choices=['smoke','full'],default='full');p.add_argument('--dataset-manifest',required=True);a=p.parse_args();R=pathlib.Path(a.run);O=R/a.stage

def write(name,rows,fields):
 with (R/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
fields=['dataset','format','video_id','question_id','arm','AR_length','tau','tau_effective','rounds','hit_eos','speed_informative','accepted_raw','accepted_effective','new_root_count','bonus_count','clipped_draft','top8_cover','first_predicted_top8_cover','target_entropy','target_margin','spec_decode_ms_token','spec_model_wall_ms_token','ar_decode_ms_token','normalized_short_answer_match','lossless','status','question_type','gold_answer','AR_text','MC_prediction','MC_correct','first_round_top8_cover','first_draft_target_entropy','target_entropy_excluding_eos','processor_s','spec_end_to_end_estimated_ms','tree_build_diagnostic_ms','head_forward_diagnostic_ms','verify_diagnostic_ms','first_draft_is_eos','first_answer_target_entropy','length_bin','top8_cover_non_eos','first_predicted_top8_non_eos','first_rejection_offset_mean','rounds_at_or_after_eos','eos_was_bonus','execution_segment','timing_boundary_uncertain']
if not (O/'results.jsonl').exists():
 write('per_question.csv',[],fields);write('per_video.csv',[],['dataset','format','video_id','arm','n_QA','tau']);write('paired_differences.csv',[],['dataset','format','video_id','comparison','delta_tau'])
 write('summary_by_dataset_format.csv',[{'dataset':d,'format':f,'status':'blocked' if d=='MSVD-QA' else 'not_run','n_videos':0,'n_QA':0,'reason':'No successful model inference; see gates.json'} for d in ['MSVD-QA','MSRVTT-QA','NExT-QA (MC)'] for f in ['A','B']],['dataset','format','status','n_videos','n_QA','reason']);sys.exit()
manifest={x['input_key'] if 'input_key'in x else (x['video_id'],x['question_id'],x['prompt_mode']):x for x in map(json.loads,pathlib.Path(a.dataset_manifest).read_text().splitlines())}
inputs={x['key']:x for x in map(json.loads,(O/'inputs.jsonl').read_text().splitlines())}
ars={x['input_key']:x for x in map(json.loads,(O/'ar.jsonl').read_text().splitlines())}
raw=list(map(json.loads,(O/'results.jsonl').read_text().splitlines()));qs=[]
boundary_key=None
if (O/'resume_provenance.json').exists():
 previous=pathlib.Path(json.loads((O/'resume_provenance.json').read_text())['source'])
 boundary_key=json.loads((previous/'results.jsonl').read_text().splitlines()[-1])['input_key']
for r in raw:
 d=r['diagnostic'];ref=ars[r['input_key']];m=manifest.get(r['input_key'],manifest.get((r['video_id'],r['question_id'],r['mode'])));tokens=ref['tokens'];hits=[];first=[];non_eos_hits=[];non_eos_first=[]
 for rr,tr in zip(d['rounds'],d['trees']):
  rel=rr['pos']-d['rounds'][0]['pos']
  for k,candidates in enumerate(tr['top8'],start=1):
   if rel+k>=len(tokens):break
   hit=int(tokens[rel+k] in candidates);hits.append(hit)
   if k==1:first.append(hit)
   if tokens[rel+k]!=151645:
    non_eos_hits.append(hit)
    if k==1:non_eos_first.append(hit)
 cs=d['clipped_rounds'];cr=r['clean_repeats'];arcr=ref['clean_repeats'];eos=151645
 q={'dataset':r['dataset'],'format':r['mode'],'video_id':r['video_id'],'question_id':r['question_id'],'arm':r['arm'],'AR_length':len(tokens),'tau':d['stats']['tau'],'tau_effective':len(tokens)/max(1,d['stats']['rounds']),'rounds':d['stats']['rounds'],'hit_eos':bool(tokens and tokens[-1]==eos),'speed_informative':r['speed_informative'],'accepted_raw':sum(x['accepted_raw'] for x in cs),'accepted_effective':sum(x['accepted_effective'] for x in cs),'new_root_count':sum(x['new_root'] for x in cs),'bonus_count':sum(x['bonus_emitted'] for x in cs),'clipped_draft':sum(x['clipped_draft'] for x in cs),'top8_cover':st.mean(hits) if hits else None,'first_predicted_top8_cover':st.mean(first) if first else None,'target_entropy':st.mean(ref['entropy']),'target_margin':st.mean(ref['margin']),'spec_decode_ms_token':st.mean(x['stats']['ms_per_token'] for x in cr),'spec_model_wall_ms_token':st.mean(x['wall_s']*1000/len(tokens) for x in cr),'ar_decode_ms_token':st.mean(x['stats']['ms_per_token'] for x in arcr),'normalized_short_answer_match':int(normalize_answer(ref['text'])==normalize_answer(m['gold_answer'])) if r['mode']=='A' else None,'lossless':r['lossless'],'status':a.stage}
 q.update(execution_segment=r.get('execution_segment','uninterrupted'),timing_boundary_uncertain=bool(boundary_key and r['input_key']==boundary_key))
 q.update(question_type=m['question_type'],gold_answer=m['gold_answer'],AR_text=ref['text'],MC_prediction=choice_answer(ref['text'],m['options']) if m.get('options') else None)
 q['MC_correct']=int(q['MC_prediction']==m['gold_answer']) if m.get('options') else None
 q.update(first_draft_is_eos=bool(len(tokens)>1 and tokens[1]==eos),first_answer_target_entropy=ref['entropy'][0],length_bin='le8' if len(tokens)<=8 else '9_32' if len(tokens)<=32 else 'gt32')
 q.update(first_round_top8_cover=first[0] if first else None,first_draft_target_entropy=ref['entropy'][1] if len(ref['entropy'])>1 else None,target_entropy_excluding_eos=st.mean(ref['entropy'][:-1]) if tokens[-1]==eos and len(tokens)>1 else st.mean(ref['entropy']),processor_s=inputs[r['input_key']]['processor_s'],spec_end_to_end_estimated_ms=1000*(inputs[r['input_key']]['processor_s']+st.mean(x['wall_s'] for x in cr)),tree_build_diagnostic_ms=d['tree_build_ms'],head_forward_diagnostic_ms=d['head_forward_ms'],verify_diagnostic_ms=d['verify_ms'])
 q['top8_cover_non_eos']=st.mean(non_eos_hits) if non_eos_hits else None
 q['first_predicted_top8_non_eos']=st.mean(non_eos_first) if non_eos_first else None
 # Offset16 is the block horizon, not a failed draft position.
 rejected=[x['first_rejected_offset'] for x in cs if x['first_rejected_offset'] is not None and x['first_rejected_offset']<=15]
 q['first_rejection_offset_mean']=st.mean(rejected) if rejected else None
 q['rounds_at_or_after_eos']=sum(x['root_position']>=len(tokens)-1 for x in cs) if q['hit_eos'] else 0
 q['eos_was_bonus']=any(x['root_position']+x['accepted_raw']+1==len(tokens)-1 and x['bonus_emitted'] for x in cs) if q['hit_eos'] else False
 qs.append(q)
write('per_question.csv',qs,fields)
groups=collections.defaultdict(list)
for q in qs:groups[(q['dataset'],q['format'],q['video_id'],q['arm'])].append(q)
vs=[]
for (d,f,v,arm),rows in groups.items():
 vs.append({'dataset':d,'format':f,'video_id':v,'arm':arm,'n_QA':len(rows),**{k:st.mean(r[k] for r in rows if r[k] is not None) if any(r[k]is not None for r in rows) else None for k in ['tau','tau_effective','AR_length','top8_cover','target_entropy','spec_decode_ms_token']}})
write('per_video.csv',vs,list(vs[0]))
sums=[];diffs=[];comparisons=[]
for d,f in sorted(set((r['dataset'],r['format']) for r in vs)):
 pool=[r for r in vs if r['dataset']==d and r['format']==f];by={arm:{r['video_id']:r['tau'] for r in pool if r['arm']==arm} for arm in ['C0','C1','C2']}
 if len(set(tuple(sorted(x)) for x in by.values()))!=1:raise RuntimeError('Incomplete arms: cannot compute a full paired result')
 s={'dataset':d,'format':f,'n_videos':len(by['C0']),'n_QA':sum(r['n_QA'] for r in pool if r['arm']=='C0'),'status':a.stage}
 for arm in by:
  armq=[q for q in qs if q['dataset']==d and q['format']==f and q['arm']==arm];s['tau_'+arm]=st.mean(by[arm].values());s['pooled_tau_'+arm]=sum(q['tau']*q['rounds'] for q in armq)/sum(q['rounds'] for q in armq)
  s['effective_tau_'+arm]=st.mean(q['tau_effective'] for q in armq);s['decode_ms_token_'+arm]=st.mean(q['spec_decode_ms_token'] for q in armq);s['SPEC_AR_'+arm]=all(q['lossless'] for q in armq);s['fraction_one_round_'+arm]=st.mean(q['rounds']<=1 for q in armq)
  covers=[q['top8_cover'] for q in armq if q['top8_cover'] is not None];s['top8_cover_'+arm]=st.mean(covers) if covers else None
 for left,right in [('C2','C1'),('C2','C0'),('C1','C0')]:
  b=paired(by[left],by[right]);c=left+'-'+right
  s.update({c+'_'+k:v for k,v in b.items()});comparisons.append({'dataset':d,'format':f,'comparison':c,**b})
  for v in by[left]:diffs.append({'dataset':d,'format':f,'video_id':v,'comparison':c,'delta_tau':by[left][v]-by[right][v]})
 q0=[q for q in qs if q['dataset']==d and q['format']==f and q['arm']=='C0'];s['AR_length_mean']=st.mean(q['AR_length'] for q in q0);s['AR_length_median']=st.median(q['AR_length'] for q in q0);s['fraction_short_le8']=st.mean(q['AR_length']<=8 for q in q0);s['target_entropy']=st.mean(q['target_entropy'] for q in q0);sums.append(s)
write('summary_by_dataset_format.csv',sums,list(sums[0]));write('paired_differences.csv',diffs,list(diffs[0]));(R/'paired_bootstrap.json').write_text(json.dumps(comparisons,indent=2));print(json.dumps(sums,indent=2))
