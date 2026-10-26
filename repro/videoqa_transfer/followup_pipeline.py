"""Sequential follow-up datasets. Only launch after full MSVD gate actually passes."""
import json,pathlib,subprocess,sys,time,shutil,traceback
R=pathlib.Path(sys.argv[1]);PY=sys.executable;HERE=pathlib.Path(__file__).resolve().parent
assert json.loads((R/'full/gates.json').read_text())['pass'],'MSVD full gate required'
for ds,sub,index in [('MSRVTT-QA','msrvtt','msrvtt_zip_index.json'),('NExT-QA','nextqa_mc','next_zip_index.json')]:
 D=R/sub;D.mkdir(exist_ok=False);(D/'logs').mkdir();m=json.loads((R/'run_manifest.json').read_text());m.update(status='active',dataset=ds,actual_model_calls={'target':0,'head':0});(D/'run_manifest.json').write_text(json.dumps(m,indent=2))
 def run(name,args):
  print(time.strftime('%H:%M:%S'),ds,name,flush=True)
  with (D/'logs'/f'{name}.log').open('x') as f:subprocess.run([PY,'-u',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  base=[HERE/'prepare_zip_dataset.py','--run',D,'--sources',R/'sources','--zip-index',R/'sources'/index,'--dataset',ds]
  run('freeze',[*base,'--stage','freeze'])
  run('prepare_smoke',[*base,'--stage','download','--n-videos',3])
  run('input_audit',[HERE/'audit_inputs.py','--run',D,'--manifest',D/'dataset_manifest_3videos.jsonl'])
  run('smoke',[HERE/'evaluate.py','--run',D,'--dataset-manifest',D/'dataset_manifest_3videos.jsonl','--stage','smoke','--gpu',0])
  assert json.loads((D/'smoke/gates.json').read_text())['pass']
  run('prepare_full',[*base,'--stage','download','--n-videos',20])
  run('full',[HERE/'evaluate.py','--run',D,'--dataset-manifest',D/'dataset_manifest_20videos.jsonl','--stage','full','--gpu',0])
  assert json.loads((D/'full/gates.json').read_text())['pass']
  run('summary',[HERE/'summarize.py','--run',D,'--dataset-manifest',D/'dataset_manifest_20videos.jsonl','--stage','full'])
  shutil.copy2(D/'dataset_manifest_20videos.jsonl',D/'dataset_manifest.jsonl');print(ds,'COMPLETE',flush=True)
 except Exception:
  (D/'pipeline_error.log').write_text(traceback.format_exc());raise
