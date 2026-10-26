"""Finalize evidence indices only after all launched subprocesses have terminated."""
import pathlib,json,hashlib,collections,time,shutil,subprocess
R=pathlib.Path(__import__('sys').argv[1]);repo=pathlib.Path('/home/mcy/Projects/GLANCE')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
M=json.loads((R/'run_manifest.json').read_text());heads={a:sha(pathlib.Path(p)/'model.safetensors') for a,p in M['heads'].items()};assert all(heads[a]==M['checkpoint_hashes'][a]['model.safetensors'] for a in heads)
assert all(sha(pathlib.Path(p)/'config.json')==M['checkpoint_hashes'][a]['config.json'] for a,p in M['heads'].items())
target_metadata_after={p:sha(pathlib.Path(p)) for p in M['target_files']}
assert target_metadata_after==M['target_files']
M['target_metadata_unchanged']=True
prod={}
for s in ['GLANCE_repo/glance/decode.py','GLANCE_repo/glance/tree.py','repro/shared/common.py','repro/data/predecode_cache.py']:
 snap=R/'snapshots'/s.replace('/','__');prod[s]={'before':sha(snap),'after':sha(repo/s)};assert prod[s]['before']==prod[s]['after']
snapdir=R/'snapshots/final_adapters';snapdir.mkdir(exist_ok=True)
for p in (repo/'repro/videoqa_transfer').glob('*.py'):shutil.copy2(p,snapdir/p.name)
status={};total=collections.Counter();allgen=[];processor=0;allmanifest=[];allsampling={}
for sub,ds in [('', 'MSVD-QA'),('msrvtt','MSRVTT-QA'),('nextqa_mc','NExT-QA (MC)')]:
 d=R/sub;stages={};calls=collections.Counter();ds_processor=0
 for stage in ['smoke','full']:
  p=d/stage/'gates.json'
  if p.exists():
   gate=json.loads(p.read_text());stages[stage]=gate
   cf=d/stage/'calls.json';calls.update(gate.get('calls',json.loads(cf.read_text()) if cf.exists() else {}))
   inp=d/stage/'inputs.jsonl'
   if inp.exists():ds_processor+=len(inp.read_text().splitlines())+1
   provenance=d/stage/'resume_provenance.json'
   if provenance.exists():
    src=pathlib.Path(json.loads(provenance.read_text())['source'])
    ds_processor+=len((src/'inputs.jsonl').read_text().splitlines())+1
   if gate.get('pass'):
    for z in map(json.loads,(d/stage/'ar.jsonl').read_text().splitlines()):allgen.append({'dataset':ds,'stage':stage,'kind':'AR','input_key':z['input_key'],'tokens':z['tokens'],'text':z['text'],'raw_file':str(d/stage/'ar.jsonl')})
  else:stages[stage]={'pass':False,'status':'not_run'}
 ig=d/'input_gates.json'
 if ig.exists():ds_processor+=json.loads(ig.read_text())['processor_calls']
 processor+=ds_processor
 complete=stages['full'].get('pass',False)
 if complete:
  assert stages['smoke'].get('pass',False), f'{ds}: full without passed smoke'
  assert json.loads(ig.read_text())['pass'], f'{ds}: input gate missing/failed'
  assert len((d/'full/results.jsonl').read_text().splitlines())==240
  assert len((d/'full/ar.jsonl').read_text().splitlines())==80
 status[ds]={'status':'complete' if complete else 'partial' if d.exists() else 'not_run','gates':stages,'calls':dict(calls)};total.update(calls)
 if d.exists() and sub:
  dm=json.loads((d/'run_manifest.json').read_text());dm.update(status=status[ds]['status'],actual_model_calls=dict(calls),actual_processor_calls=ds_processor,checkpoint_hash_after=heads,checkpoint_unchanged=True,production_files_unchanged=True,completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'));dm.pop('gpu_stage',None);dm.pop('actual_model_calls_status',None);dm.pop('dataset_status',None)
  dm['adapter_code_hashes']={str(p.relative_to(repo)):sha(p) for p in (repo/'repro/videoqa_transfer').glob('*.py')}
  dm['hardware_records']=[str(d/s/'gpu_start.json') for s in ['smoke','full'] if (d/s/'gpu_start.json').exists()]
  write(d/'run_manifest.json',dm);write(d/'gates.json',status[ds])
 if complete:
  shutil.copy2(d/'dataset_manifest_20videos.jsonl',d/'dataset_manifest.jsonl')
  allmanifest.extend(map(json.loads,(d/'dataset_manifest_20videos.jsonl').read_text().splitlines()))
  allsampling[ds]=json.loads((d/'sampling_manifest.json').read_text())
M.update(status='complete' if all(x['status']=='complete' for x in status.values()) else 'partial',completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),dataset_status=status,actual_model_calls=dict(total),actual_processor_calls=processor,checkpoint_hash_after=heads,checkpoint_unchanged=True,production_files_unchanged=True,production_hash_verification=prod,adapter_code_hashes={str(p.relative_to(repo)):sha(p) for p in (repo/'repro/videoqa_transfer').glob('*.py')},timing_notes='two clean repeats after diagnostic warm execution; full rotates arms per input; CPU head transfers outside timing; diagnostic component times not official latency')
M.pop('gpu_stage',None);M.pop('actual_model_calls_status',None);M['gpu_at_delivery']=subprocess.check_output(['nvidia-smi'],text=True);write(R/'run_manifest.json',M)
write(R/'gates.json',{'pass':all(x['status']=='complete' for x in status.values()),'datasets':status,'checkpoint_unchanged':True,'production_files_unchanged':True,'no_training':True})
(R/'raw_generations.jsonl').write_text(''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in allgen))
(R/'all_dataset_manifest.jsonl').write_text(''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in allmanifest))
write(R/'all_sampling_manifests.json',allsampling)
files=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json']
write(R/'delivery_manifest.json',{'generated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'status':M['status'],'files':files,'raw_video_storage':'/mnt/local2/mcy/datasets/videoqa_transfer/; per-video SHA256 in dataset manifests','no_training':True,'checkpoint_unchanged':True,'production_files_unchanged':True})
print(json.dumps({'status':M['status'],'calls':dict(total),'processor_calls':processor,'files':len(files)},indent=2))
