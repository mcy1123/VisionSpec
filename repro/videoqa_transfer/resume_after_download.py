"""Resume after recorded MSRVTT network failure; no repeats of completed GPU stages."""
import pathlib,json,subprocess,sys,traceback,time,shutil
R=pathlib.Path(sys.argv[1]);H=pathlib.Path(__file__).resolve().parent;PY=sys.executable;D=R/'msrvtt'
def run(d,name,args):
 print(time.strftime('%H:%M:%S'),d.name,name,flush=True)
 with (d/'logs'/f'{name}.log').open('x') as f:subprocess.run([PY,'-u',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 assert json.loads((R/'full/gates.json').read_text())['pass'] and json.loads((D/'smoke/gates.json').read_text())['pass']
 m=json.loads((D/'sampling_manifest.json').read_text());assert all((pathlib.Path('/mnt/local2/mcy/datasets/videoqa_transfer/MSRVTT-QA')/v).exists() for v in m['video_ids'])
 base=[H/'prepare_zip_dataset.py','--run',D,'--sources',R/'sources','--zip-index',R/'sources/msrvtt_zip_index_refreshed.json','--dataset','MSRVTT-QA']
 run(D,'prepare_full_retry1',[*base,'--stage','download','--n-videos',20]);run(D,'full',[H/'evaluate.py','--run',D,'--dataset-manifest',D/'dataset_manifest_20videos.jsonl','--stage','full','--gpu',0]);assert json.loads((D/'full/gates.json').read_text())['pass'];run(D,'summary',[H/'summarize.py','--run',D,'--dataset-manifest',D/'dataset_manifest_20videos.jsonl','--stage','full']);shutil.copy2(D/'dataset_manifest_20videos.jsonl',D/'dataset_manifest.jsonl')
 print('MSRVTT COMPLETE',flush=True)
 D=R/'nextqa_mc';D.mkdir(exist_ok=False);(D/'logs').mkdir();m=json.loads((R/'run_manifest.json').read_text());m.update(status='running',dataset='NExT-QA (MC)',actual_model_calls={});(D/'run_manifest.json').write_text(json.dumps(m,indent=2));base=[H/'prepare_zip_dataset.py','--run',D,'--sources',R/'sources','--zip-index',R/'sources/next_zip_index.json','--dataset','NExT-QA']
 run(D,'freeze',[*base,'--stage','freeze'])
 for n,stage in [(3,'smoke'),(20,'full')]:
  run(D,'download_'+stage,[H/'download_frozen_zip_ranges.py','--run',D,'--index',R/'sources/next_zip_index.json','--n-videos',n]);run(D,'prepare_'+stage,[*base,'--stage','download','--n-videos',n])
  if n==3:run(D,'input_audit',[H/'audit_inputs.py','--run',D,'--manifest',D/'dataset_manifest_3videos.jsonl'])
  run(D,stage,[H/'evaluate.py','--run',D,'--dataset-manifest',D/f'dataset_manifest_{n}videos.jsonl','--stage',stage,'--gpu',0]);assert json.loads((D/stage/'gates.json').read_text())['pass']
 run(D,'summary',[H/'summarize.py','--run',D,'--dataset-manifest',D/'dataset_manifest_20videos.jsonl','--stage','full']);shutil.copy2(D/'dataset_manifest_20videos.jsonl',D/'dataset_manifest.jsonl');print('NExT-QA MC COMPLETE',flush=True)
except Exception:
 (D/'resume_pipeline_error.log').write_text(traceback.format_exc());raise
