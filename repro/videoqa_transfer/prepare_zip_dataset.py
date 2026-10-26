"""Val-only deterministic selection and selective ZIP video extraction for follow-up datasets."""
import argparse,pathlib,json,sys,csv,hashlib,zipfile,collections,time
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from protocol import select,prompt
from remote_zip import RemoteZipFile
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['MSRVTT-QA','NExT-QA'],required=True);p.add_argument('--run',required=True);p.add_argument('--sources',required=True);p.add_argument('--stage',choices=['freeze','download'],required=True);p.add_argument('--n-videos',type=int,default=3);p.add_argument('--zip-index',required=True);a=p.parse_args();R=pathlib.Path(a.run);R.mkdir(exist_ok=True,parents=True);S=pathlib.Path(a.sources);D=pathlib.Path('/mnt/local2/mcy/datasets/videoqa_transfer')/a.dataset;D.mkdir(exist_ok=True)
if a.stage=='freeze':
 if a.dataset=='MSRVTT-QA':
  raw=json.loads((S/'msrvtt_val.txt').read_text());rows=[{'dataset':a.dataset,'split':'val','video_id':r['video'],'question_id':r['question_id'],'question':r['question'],'question_type':r['answer_type'],'options':None,'gold_answer':r['answer'],'source_id':None,'overlap_status':'overlap_unknown'} for r in raw]
 else:
  raw=list(csv.DictReader((S/'next_val.txt').open()));mapping=json.loads((S/'next_map.txt').read_text());rows=[{'dataset':a.dataset,'split':'val','video_id':str(r['video'])+'.mp4','question_id':r['video']+'_'+r['qid'],'question':r['question'],'question_type':r['type'],'options':[r['a'+str(i)] for i in range(5)],'gold_answer':chr(65+int(r['answer'])),'gold_answer_text':r['a'+r['answer']],'source_id':'vidor:'+mapping[r['video']],'overlap_status':'overlap_unknown'} for r in raw]
 vids,selected=select(rows);m={'frozen_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'dataset':a.dataset,'split':'val','seed':20261013,'video_ids':vids,'qa_rows':selected,'smoke_video_ids':vids[:3],'source_version':'LAVIS converted val' if a.dataset=='MSRVTT-QA' else 'author NExT-QA val.csv MC','n_available_annotation_videos':len(set(r['video_id'] for r in rows)),'n_available_QA':len(rows),'selection':'same protocol.select video type/question-length stratification as MSVD; unknown duration before acquisition; no resampling','max_new':{'A':64,'B':160},'templates':{mode:prompt({'question':'{question}','options':['{option_'+chr(65+i)+'}' for i in range(5)] if a.dataset=='NExT-QA' else None},mode) for mode in ['A','B']},'license':'see parent dataset_audit.md; original-source research use only, no redistribution'}
 with (R/'sampling_manifest.json').open('x') as f:json.dump(m,f,indent=2)
 print('frozen',len(vids),len(selected),vids[:3]);sys.exit()
m=json.loads((R/'sampling_manifest.json').read_text());vids=m['video_ids'][:a.n_videos];idx=json.loads(pathlib.Path(a.zip_index).read_text());by=collections.defaultdict(list)
for member in idx['members']:by[pathlib.Path(member['name']).name].append(member)
for v in vids:assert len(by[v])==1,('ambiguous or missing frozen video',v)
print('planned compressed bytes',sum(by[v][0]['compressed_size'] for v in vids),flush=True)
if any(not (D/v).exists() for v in vids):
 f=RemoteZipFile(idx['url']);z=zipfile.ZipFile(f)
else:
 from types import SimpleNamespace
 f=SimpleNamespace(downloaded=0);z=None
meta={}
import cv2,numpy as np,av
for v in vids:
 dest=D/v;member=by[v][0]
 if not dest.exists():
  data=z.read(member['name']);assert len(data)==member['size'];dest.write_bytes(data)
 cap=cv2.VideoCapture(str(dest));assert cap.isOpened();n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);sw,sh=int(cap.get(3)),int(cap.get(4));assert n>0 and fps>0
 scale=384/max(sw,sh);w,h=max(int(sw*scale)//2*2,2),max(int(sh*scale)//2*2,2);ids=[];frames=[]
 for i in np.linspace(0,n-1,16).astype(int):
  for back in range(25):
   j=int(i)-back
   if j<0:raise RuntimeError('undecodable frame')
   cap.set(cv2.CAP_PROP_POS_FRAMES,j);ok,frame=cap.read()
   if ok:break
  if not ok:raise RuntimeError('undecodable frame')
  ids.append(j);frames.append(cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA))
 cap.release();cache=D/(dest.stem+'.cache16.mp4')
 if not cache.exists():
  wr=cv2.VideoWriter(str(cache),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h));assert wr.isOpened()
  for frame in frames:wr.write(frame)
  wr.release()
 c=av.open(str(dest));streams=[{'type':s.type,'codec':s.codec_context.name,'time_base':str(s.time_base),'duration':s.duration,'frames':s.frames} for s in c.streams];duration=c.duration/av.time_base if c.duration else n/fps;pts=[float(fr.pts*fr.time_base) for fr in c.decode(video=0)];c.close();ts=[pts[j] for j in ids]
 meta[v]={'video_id':v,'raw_path':str(dest),'cache_path':str(cache),'hash':hashlib.sha256(dest.read_bytes()).hexdigest(),'cache_hash':hashlib.sha256(cache.read_bytes()).hexdigest(),'video_duration':duration,'source_fps':fps,'source_frame_count':n,'sample_frame_ids':ids,'sample_timestamps':ts,'sample_timestamp_method':'PyAV decoded frame pts*time_base','source_probe':{'streams':streams,'duration':duration},'fps':30,'n_frames':16,'width':w,'height':h,'zip_member':member}
 print('readable',v,duration,flush=True)
recs=[]
for q in m['qa_rows']:
 if q['video_id'] in meta:
  for mode in ['A','B']:recs.append({**q,**meta[q['video_id']],'prompt_mode':mode,'prompt':prompt(q,mode)})
with (R/f'dataset_manifest_{a.n_videos}videos.jsonl').open('x') as out:
 for x in recs:out.write(json.dumps(x)+'\n')
(R/f'acquisition_{a.n_videos}.json').write_text(json.dumps({'range_bytes_read':f.downloaded,'videos':len(vids),'inputs':len(recs)},indent=2));print('prepared',len(recs))
