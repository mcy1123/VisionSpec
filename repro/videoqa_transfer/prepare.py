"""Freeze val-only samples before model inference; fetch only explicit video ranges."""
import argparse,json,pathlib,sys,requests,hashlib,time,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from protocol import select,source_id,prompt,digest
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',choices=['freeze','download'],required=True);p.add_argument('--n-videos',type=int,default=3);a=p.parse_args();R=pathlib.Path(a.run)
D=pathlib.Path('/mnt/local2/mcy/datasets/videoqa_transfer/MSVD-QA');D.mkdir(parents=True,exist_ok=True)
if a.stage=='freeze':
 raw=json.loads((R/'sources/msvd_val.txt').read_text());rows=[]
 for x in raw:
  rows.append({'dataset':'MSVD-QA','split':'val','video_id':x['video'],'source_id':source_id(x['video']),'question_id':x['question_id'],'question':x['question'],'options':None,'gold_answer':x['answer'],'question_type':x['answer_type'],'overlap_status':'overlap_unknown'})
 vids,selected=select(rows)
 hist=[]
 import pandas as pd
 for fn in ['pool_499.parquet','video_fresh30.parquet']:
  f=pathlib.Path('/mnt/local/mcy/GLANCE/data')/fn
  if f.exists():hist+=pd.read_parquet(f)['video_name'].tolist()
 sources={source_id(str(x)) for x in hist}
 for r in selected:r['known_source_overlap']=r['source_id'] in sources
 manifest={'frozen_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'dataset':'MSVD-QA','split':'val','seed':20261013,'n_available_annotation_videos':len(set(r['video_id'] for r in rows)),'n_available_qa':len(rows),'video_ids':vids,'qa_rows':selected,'selection':'round-robin question-type × representative question-length tertile; within-stratum seeded video shuffle; up to two distinct questions, different type preferred','duration_stratification':'not possible before selective video download; report durations after freeze, do not resample','smoke_video_ids':vids[:3],'max_new':{'A':64,'B':160},'templates':{m:prompt({'question':'{question}'},m) for m in ['A','B']},'source_overlap':'compared extracted source IDs with pool499/fresh30; missing genealogy remains overlap_unknown','known_source_overlap_count':sum(r['known_source_overlap'] for r in selected),'license':'MSVD original public research download; site requests citation, no explicit video redistribution license found. LAVIS code BSD-3-Clause is not a video license. Local research only, no redistribution.'}
 with (R/'sampling_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
 print('FROZEN',len(vids),len(selected),vids[:3])
else:
 m=json.loads((R/'sampling_manifest.json').read_text());vids=m['video_ids'][:a.n_videos];idx={pathlib.Path(x['name']).name:x for x in map(json.loads,(R/'sources/msvd_tar_index.jsonl').read_text().splitlines())};missing=[v for v in vids if v not in idx]
 if missing:raise RuntimeError('tar metadata index not yet reached frozen videos: '+str(missing))
 print('planned video bytes',sum(idx[v]['size'] for v in vids),flush=True)
 meta=[]
 for v in vids:
  x=idx[v];dest=D/v
  if not dest.exists():
   st=x['data_offset'];end=st+x['size']-1;r=requests.get(x['url'],headers={'Range':f'bytes={st}-{end}'},timeout=60,stream=True)
   assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {st}-{end}/')
   with dest.with_suffix('.partial').open('wb') as f:
    for chunk in r.iter_content(1<<20):f.write(chunk)
   assert dest.with_suffix('.partial').stat().st_size==x['size'];dest.with_suffix('.partial').rename(dest)
  import cv2,numpy as np
  cap=cv2.VideoCapture(str(dest));assert cap.isOpened();total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);sw,sh=int(cap.get(3)),int(cap.get(4));assert total>0 and fps>0
  scale=384/max(sw,sh);w,h=max(int(sw*scale)//2*2,2),max(int(sh*scale)//2*2,2)
  cache=D/(dest.stem+'.cache16.mp4');ids=[];ts=[];frames=[]
  for i in np.linspace(0,total-1,16).astype(int):
   for back in range(25):
    j=int(i)-back
    if j<0:raise RuntimeError('cannot decode frozen frame')
    cap.set(cv2.CAP_PROP_POS_FRAMES,j);ok,f=cap.read()
    if ok:break
   if not ok:raise RuntimeError('cannot decode frozen frame')
   ids.append(j);ts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000);frames.append(cv2.resize(f,(w,h),interpolation=cv2.INTER_AREA))
  cap.release()
  if not cache.exists():
   wr=cv2.VideoWriter(str(cache),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h));assert wr.isOpened()
   for f in frames:wr.write(f)
   wr.release()
  import av
  container=av.open(str(dest));streams=[{'type':z.type,'codec':z.codec_context.name,'time_base':str(z.time_base),'duration':z.duration,'frames':z.frames} for z in container.streams]
  duration=container.duration/av.time_base if container.duration is not None else total/fps
  pts=[float(fr.pts*fr.time_base) for fr in container.decode(video=0)];container.close()
  probe={'format':{'duration':duration},'streams':streams,'reader':'PyAV existing environment (ffprobe CLI unavailable)'}
  cv_ts=ts;ts=[pts[j] for j in ids]
  assert all(t>=0 for t in ts) and all(ts[i]<=ts[i+1] for i in range(15))
  meta.append({'video_id':v,'raw_path':str(dest),'cache_path':str(cache),'hash':hashlib.sha256(dest.read_bytes()).hexdigest(),'cache_hash':hashlib.sha256(cache.read_bytes()).hexdigest(),'video_duration':float(probe['format']['duration']),'source_fps':fps,'source_frame_count':total,'sample_frame_ids':ids,'sample_timestamps':ts,'sample_timestamp_method':'PyAV decoded frame pts*time_base indexed by actual sampled frame ID','opencv_timestamps':cv_ts,'fps':30,'n_frames':16,'width':w,'height':h,'source_probe':probe,'download_range':x})
  print('readable',v,total,fps,ids,flush=True)
 metaby={x['video_id']:x for x in meta};recs=[]
 for q in m['qa_rows']:
  if q['video_id'] in metaby:
   for mode in ['A','B']:
    rec={**q,**metaby[q['video_id']],'prompt_mode':mode,'prompt':prompt(q,mode)};recs.append(rec)
 fn=R/f'dataset_manifest_{a.n_videos}videos.jsonl'
 with fn.open('x') as f:
  for x in recs:f.write(json.dumps(x)+'\n')
 print('prepared',len(recs),'inputs',fn)
