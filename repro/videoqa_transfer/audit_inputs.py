"""Real CPU processor audit on the frozen pilot videos, preserving internal metadata."""
import sys,pathlib,json,argparse,hashlib,dataclasses,time
sys.path[:0]=[str(pathlib.Path(__file__).resolve().parent),str(pathlib.Path(__file__).resolve().parents[1])]
from protocol import cache_key,digest
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();R=pathlib.Path(a.run)
import torch,transformers,numpy as np
from transformers import AutoProcessor
from shared import common
torch.set_num_threads(4)
def safe(x):
 if dataclasses.is_dataclass(x):return safe(dataclasses.asdict(x))
 if isinstance(x,(torch.Tensor,np.ndarray)):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 return x
def sha(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
proc=AutoProcessor.from_pretrained(common.TARGET_PATH,local_files_only=True);meta={};original=proc.video_processor.preprocess

def observe(*args,**kwargs):
 out=original(*args,**kwargs);meta['value']=safe(out.get('video_metadata'));return out
proc.video_processor.preprocess=observe
rows=[json.loads(x) for x in pathlib.Path(a.manifest).read_text().splitlines()];m=json.loads((R/'run_manifest.json').read_text());ph=digest(m['target_files']);seen=set();video_hash={};out=[]
with (R/'input_audit.jsonl').open('x') as f:
 for r in rows:
  text=proc.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':r['prompt']}]}],add_generation_prompt=True,tokenize=False)
  def process():return proc(text=[text],videos=[r['cache_path']],return_tensors='pt',fps=float(r['fps']))
  x=process();metadata=meta['value'];xx=process();assert all(torch.equal(x[k],xx[k]) for k in x if torch.is_tensor(x[k]))
  key=cache_key(r,r['prompt_mode'],ph);assert key not in seen;seen.add(key)
  h={k:sha(v) for k,v in x.items() if torch.is_tensor(v)};vh=h['pixel_values_videos']
  if r['video_id'] in video_hash:assert video_hash[r['video_id']]==vh
  else:video_hash[r['video_id']]=vh
  assert int(x['video_grid_thw'][0,0])==8
  rec={'video_id':r['video_id'],'question_id':r['question_id'],'format':r['prompt_mode'],'input_key':key,'input_ids':x['input_ids'].tolist(),'tensor_hashes':h,'grid':x['video_grid_thw'].tolist(),'metadata':metadata,'repeat_exact':True,'input_tokens':x['input_ids'].shape[1]};out.append(rec);f.write(json.dumps(rec)+'\n');f.flush();print(r['video_id'],r['question_id'],r['prompt_mode'],rec['input_tokens'],'exact',flush=True)
(R/'input_gates.json').write_text(json.dumps({'pass':True,'real_videos':len(video_hash),'real_inputs':len(out),'processor_calls':len(out)*2,'unique_input_keys':len(seen),'same_video_pixels_across_qa_formats':True,'repeat_inputs_exact':True,'frames':16,'feature_model_gate':'not_run CPU-only check','environment':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__}},indent=2))
