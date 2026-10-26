"""Read-only QA transfer adapter around unmodified GLANCE generate/greedy_generate.
Diagnostic forward hooks and clean timing repeats are strictly separate.
"""
import argparse,collections,contextlib,hashlib,json,os,pathlib,subprocess,sys,time,traceback
HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
from protocol import cache_key,clipped_rounds,exact_tokens,digest

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def append(path,obj):
 with path.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush()
def gpu_snapshot():
 return {k:subprocess.check_output(['nvidia-smi','--query-'+k+'='+fields,'--format=csv,noheader,nounits'],text=True) for k,fields in [('gpu','index,uuid,name,memory.total,memory.free,utilization.gpu'),('compute-apps','gpu_uuid,pid,process_name,used_memory')]}
def gpu_gate(index):
 s=gpu_snapshot();g=next(x for x in s['gpu'].splitlines() if x.split(',')[0].strip()==str(index));v=[x.strip() for x in g.split(',')]
 foreign=[x for x in s['compute-apps'].splitlines() if x.split(',')[0].strip()==v[1] and int(x.split(',')[1])!=os.getpid()]
 if foreign or int(v[4])<40000:raise RuntimeError('GPU_RESOURCE_GATE: fp32 needs >=40000 MiB free and no foreign compute process: '+g+'; '+str(foreign))
 return s

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--dataset-manifest',required=True);ap.add_argument('--stage',choices=['smoke','full'],required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--resume-from');a=ap.parse_args();R=pathlib.Path(a.run);out=R/a.stage
 out.mkdir(exist_ok=False);dump(out/'command.json',{'argv':sys.argv,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 try:
  dump(out/'gpu_start.json',gpu_snapshot());gpu_gate(a.gpu)
  import torch,numpy as np,transformers
  from shared import common; import glance; import glance.decode as dec
  from glance.vl import VisionPositions
  from transformers import AutoProcessor
  from safetensors.torch import load_file
  os.environ['TOKENIZERS_PARALLELISM']='false';torch.set_num_threads(8)
  torch.manual_seed(20261013);torch.set_grad_enabled(False)
  dev=f'cuda:{a.gpu}';torch.cuda.set_device(a.gpu)
  rows=[json.loads(l) for l in pathlib.Path(a.dataset_manifest).read_text().splitlines()]
  if a.stage=='full':
   gate=json.loads((R/'smoke/gates.json').read_text());assert gate['pass']
  M=json.loads((R/'run_manifest.json').read_text());sampling=json.loads((R/'sampling_manifest.json').read_text())
  expected=sampling['smoke_video_ids'] if a.stage=='smoke' else sampling['video_ids']
  assert set(r['video_id'] for r in rows)==set(expected)
  assert all(r['split'] in ['val','dev','train'] for r in rows)
  target,proc=common.load_target(device=dev,dtype=torch.float32)
  assert next(target.parameters()).dtype==torch.float32
  eos=proc.tokenizer.eos_token_id;processor_hash=digest(M['target_files']);inputs={};ars={};calls=collections.Counter();results=[]
  previous=pathlib.Path(a.resume_from) if a.resume_from else None
  old_inputs={}
  if previous:
   assert a.stage=='full'
   assert not json.loads((previous/'gates.json').read_text())['pass']
   assert 'foreign GPU task arrived' in (previous/'error.log').read_text()
   old_inputs={x['key']:x for x in map(json.loads,(previous/'inputs.jsonl').read_text().splitlines())}
   ars={x['input_key']:x for x in map(json.loads,(previous/'ar.jsonl').read_text().splitlines())}
   results=list(map(json.loads,(previous/'results.jsonl').read_text().splitlines()))
   assert len({(x['input_key'],x['arm']) for x in results})==len(results)
   calls.update(json.loads((previous/'calls.json').read_text()))
   for x in results:
    assert x['lossless'] and x['diagnostic']['tokens']==ars[x['input_key']]['tokens']
    assert all(c['tokens']==ars[x['input_key']]['tokens'] for c in x['clean_repeats'])
    x['execution_segment']='initial';append(out/'results.jsonl',x);append(out/'diagnostic.jsonl',{k:v for k,v in x.items() if k!='clean_repeats'})
   for x in ars.values():append(out/'ar.jsonl',x)
   dump(out/'resume_provenance.json',{'source':str(previous),'reused_arm_records':len(results),'reused_AR_inputs':len(ars),'new_warmup_required':True,'timing_segments_not_conflated':True})
  dump(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'device':torch.cuda.get_device_name(),'target_dtype':str(next(target.parameters()).dtype),'eos':eos})
  def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
  def encode(r):
   t=time.perf_counter();text=proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':r['prompt']}]}],add_generation_prompt=True,tokenize=False)
   x=proc(text=[text],videos=[r['cache_path']],return_tensors='pt',fps=float(r['fps']))
   h={k:thash(v) for k,v in x.items() if torch.is_tensor(v)}
   return x,h,time.perf_counter()-t
  for i,r in enumerate(rows):
   x,h,sec=encode(r);key=cache_key(r,r['prompt_mode'],processor_hash);assert key not in inputs
   if previous:assert h==old_inputs[key]['hashes']
   if i==0:
    xx,hh,_=encode(r);assert h==hh
   assert int(x['video_grid_thw'][0,0])==8,'16 input frames / temporal patch 2 required'
   inputs[key]=(x,h,sec);r['input_key']=key
   append(out/'inputs.jsonl',{'key':key,'video_id':r['video_id'],'question_id':r['question_id'],'mode':r['prompt_mode'],'hashes':h,'grid':x['video_grid_thw'].tolist(),'input_ids':x['input_ids'].tolist(),'processor_s':sec})
  def tensors(r):
   x=inputs[r['input_key']][0];return x['input_ids'].to(dev),x['pixel_values_videos'].to(dev),x['video_grid_thw'].to(dev)
  def pos(pv,vg):return VisionPositions(pixel_values_videos=pv,video_grid_thw=vg)
  def sync():torch.cuda.synchronize()
  def count_target(m,args,outp):calls['target_forwards']+=1
  counter=target.register_forward_hook(count_target)
  def ar(r,diagnostic=False,warm=False):
   ids,pv,vg=tensors(r);ent=[];margin=[];greedy=[]
   def capture(m,args,output):
    z=output.logits[0,-1].float();lp=z.log_softmax(-1);pr=lp.exp();vals=pr.topk(2).values
    ent.append(float(-(pr*lp).sum()));margin.append(float(vals[0]-vals[1]));greedy.append(int(z.argmax()))
   handle=target.register_forward_hook(capture) if diagnostic else None
   try:
    sync();t=time.perf_counter();v,st=dec.greedy_generate(target,ids,16 if warm else M['config']['max_new_tokens'][r['prompt_mode']],[eos],positions=pos(pv,vg),return_stats=True);sync();wall=time.perf_counter()-t
   finally:
    if handle:handle.remove()
   tok=v[0,ids.shape[1]:].tolist()
   if diagnostic:assert greedy==tok
   return {'tokens':tok,'text':proc.tokenizer.decode(tok,skip_special_tokens=True),'stats':st.to_dict(),'wall_s':wall,'entropy':ent,'margin':margin}
  # All AR diagnostic values come from the independent greedy oracle, not off-path verify nodes.
  warm_ar=ar(rows[0],warm=True);calls['AR_warmup_sequences']+=1
  if previous:
   assert len(ars)==len(rows)
   assert warm_ar['tokens']==ars[rows[0]['input_key']]['tokens'][:16]
  for r in rows:
   if r['input_key'] in ars:continue
   d=ar(r,diagnostic=True);calls['AR_diagnostic_sequences']+=1;clean=[]
   for _ in range(2):
    t=ar(r);calls['AR_clean_sequences']+=1;assert exact_tokens(t['tokens'],d['tokens']);clean.append(t)
   d['clean_repeats']=clean;d['input_key']=r['input_key'];ars[r['input_key']]=d;append(out/'ar.jsonl',d)
  def spec(r,head,diagnostic=False,warm=False):
   ids,pv,vg=tensors(r);rr=[];trees=[];ctx=[];timers=[];head_events=[];target_events=[];handles=[]
   original=dec.build_budget_tree
   def build(top_ids,top_lp,budget,branch_k):
    t=time.perf_counter();nodes=original(top_ids,top_lp,budget,branch_k);timers.append(time.perf_counter()-t)
    trees.append({'top8':top_ids.tolist(),'top8_logp':top_lp.tolist(),'nodes':[vars(x) for x in nodes]});return nodes
   def hpre(m,args,kwargs):
    ev=torch.cuda.Event(enable_timing=True);ev.record();head_events.append([ev,None])
    if not ctx:ctx.append(thash(kwargs['target_hidden']))
   def hpost(m,args,o):
    ev=torch.cuda.Event(enable_timing=True);ev.record();head_events[-1][1]=ev
   def tpre(m,args):
    ev=torch.cuda.Event(enable_timing=True);ev.record();target_events.append([ev,None])
   def tpost(m,args,o):
    ev=torch.cuda.Event(enable_timing=True);ev.record();target_events[-1][1]=ev
   if diagnostic:
    dec.build_budget_tree=build;handles=[head.register_forward_pre_hook(hpre,with_kwargs=True),head.register_forward_hook(hpost),target.register_forward_pre_hook(tpre),target.register_forward_hook(tpost)]
   try:
    sync();t=time.perf_counter();v,st=glance.generate(head,target,ids,16 if warm else M['config']['max_new_tokens'][r['prompt_mode']],[eos],budget=63,branch_k=8,temperature=0.,positions=pos(pv,vg),return_stats=True,timing='decode_only',on_round=rr.append if diagnostic else None);sync();wall=time.perf_counter()-t
   finally:
    dec.build_budget_tree=original
    for h in handles:h.remove()
   tok=v[0,ids.shape[1]:].tolist();d={'tokens':tok,'stats':st.to_dict(),'wall_s':wall}
   if diagnostic:
    d.update(rounds=rr,trees=trees,first_context_hash=ctx[0],clipped_rounds=clipped_rounds(rr,int(ids.shape[1]),len(tok)),tree_build_ms=sum(timers)*1000,head_forward_ms=sum(s.elapsed_time(e) for s,e in head_events),target_prefill_ms=target_events[0][0].elapsed_time(target_events[0][1]),verify_ms=sum(s.elapsed_time(e) for s,e in target_events[1:]))
   return d
  # Smoke order C0 then C1 then C2. Full measurements rotate arm order by input;
  # loading and transfer are outside timing. Clean runs must reproduce diagnostic tokens.
  jobs=[(arm,rows) for arm in ['C0','C1','C2']] if a.stage=='smoke' else [(arm,[r]) for i,r in enumerate(rows) for arm in (['C0','C1','C2'][i%3:]+['C0','C1','C2'][:i%3])]
  first_ctx={};head=None;loaded=None;head_cache={};warmed=set()
  done={(x['input_key'],x['arm']) for x in results}
  if previous:
   planned=[(r['input_key'],arm) for arm,batch in jobs for r in batch]
   assert [(x['input_key'],x['arm']) for x in results]==planned[:len(results)]
  for x in results:
   k=x['input_key'];h=x['diagnostic']['first_context_hash']
   if k in first_ctx:assert first_ctx[k]==h
   first_ctx[k]=h
  for arm,batch in jobs:
   batch=[r for r in batch if (r['input_key'],arm) not in done]
   if not batch:continue
   if arm!=loaded:
    if head is not None:head.to('cpu')
    if arm not in head_cache:
     head_cache[arm]=common.load_head(M['heads'][arm],device='cpu',dtype=torch.float32).eval()
     head_cache[arm].requires_grad_(False)
     def count_head(m,args,o):calls['head_forwards']+=1
     head_cache[arm].register_forward_hook(count_head)
    head=head_cache[arm].to(dev);loaded=arm
    assert next(head.parameters()).dtype==torch.float32 and head.block_size==16
    if arm not in warmed:
     warm_spec=spec(rows[0],head,warm=True);calls['SPEC_warmup_sequences']+=1;warmed.add(arm)
     if previous:assert warm_spec['tokens']==ars[rows[0]['input_key']]['tokens'][:16]
   for r in batch:
    # Fail immediately if another compute job arrives; no waiting or killing.
    s=gpu_snapshot();uuid=next(x.split(',')[1].strip() for x in s['gpu'].splitlines() if x.split(',')[0].strip()==str(a.gpu))
    foreign=[x for x in s['compute-apps'].splitlines() if x.split(',')[0].strip()==uuid and int(x.split(',')[1])!=os.getpid()]
    assert not foreign,'foreign GPU task arrived: '+str(foreign)
    d=spec(r,head,diagnostic=True);calls['SPEC_diagnostic_sequences']+=1
    ref=ars[r['input_key']];good=exact_tokens(d['tokens'],ref['tokens'])
    record={'dataset':r['dataset'],'video_id':r['video_id'],'question_id':r['question_id'],'mode':r['prompt_mode'],'arm':arm,'input_key':r['input_key'],'lossless':good,'diagnostic':d,'gpu':s}
    if previous:record['execution_segment']='resume1'
    append(out/'diagnostic.jsonl',record)
    assert good,'SPEC != AR: '+r['input_key']+' '+arm
    if r['input_key'] in first_ctx:assert first_ctx[r['input_key']]==d['first_context_hash']
    else:first_ctx[r['input_key']]=d['first_context_hash']
    clean=[]
    for _ in range(2):
     t=spec(r,head);calls['SPEC_clean_sequences']+=1;assert exact_tokens(t['tokens'],ref['tokens']);assert t['stats']['rounds']==d['stats']['rounds'];clean.append(t)
    assert sum(x['new_root']+x['accepted_effective']+x['bonus_emitted'] for x in d['clipped_rounds'])==len(d['tokens'])
    record['clean_repeats']=clean;record['speed_informative']=len(d['tokens'])>8 and d['stats']['rounds']>1;results.append(record);append(out/'results.jsonl',record);dump(out/'calls.json',dict(calls))
    print(a.stage,arm,r['video_id'],r['question_id'],r['prompt_mode'],'n',len(d['tokens']),'tau',d['stats']['tau'],'lossless',good,flush=True)
  assert len(results)==len(rows)*3
  dump(out/'gates.json',{'pass':True,'status':'complete','n_inputs':len(rows),'n_videos':len(expected),'spec_ar_checks':len(results)*3,'cross_arm_context_hash':True,'calls':dict(calls),'max_memory_GiB':torch.cuda.max_memory_allocated()/2**30})
 except Exception:
  if 'calls' in locals():dump(out/'calls.json',dict(calls))
  err=traceback.format_exc();(out/'error.log').write_text(err);dump(out/'gates.json',{'pass':False,'status':'blocked' if 'GPU_RESOURCE_GATE' in err else 'failed','reason':err})
  print(err,flush=True);raise
if __name__=='__main__':main()
