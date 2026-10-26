"""Deterministic first-five-video contact sheet and corresponding AR text for qualitative audit."""
import pathlib,json,argparse,cv2
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();r=pathlib.Path(a.run)
rows=list(map(json.loads,(r/'dataset_manifest_20videos.jsonl').read_text().splitlines()));ids=list(dict.fromkeys(x['video_id'] for x in rows))[:5]
inputs=list(map(json.loads,(r/'full/inputs.jsonl').read_text().splitlines()));ars={x['input_key']:x for x in map(json.loads,(r/'full/ar.jsonl').read_text().splitlines())};keys={(x['video_id'],x['question_id'],x['mode']):x['key'] for x in inputs}
im=Image.new('RGB',(1000,len(ids)*215),'white');draw=ImageDraw.Draw(im);texts=[]
for i,vid in enumerate(ids):
 m=next(x for x in rows if x['video_id']==vid);cap=cv2.VideoCapture(m['cache_path']);draw.text((4,i*215+2),vid,fill='black')
 for j,idx in enumerate([0,5,10,15]):
  cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,f=cap.read();assert ok
  f=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));f.thumbnail((248,185));im.paste(f,(j*250,i*215+25))
 cap.release()
 for x in rows:
  if x['video_id']==vid and x['prompt_mode']=='B':
   z=ars[keys[(vid,x['question_id'],'B')]];texts.append({'video_id':vid,'question':x['question'],'options':x.get('options'),'gold':x['gold_answer'],'AR_answer':z['text']})
im.save(r/'first5_visual_frames.png');(r/'first5_visual_answers.json').write_text(json.dumps(texts,indent=2,ensure_ascii=False));print(json.dumps(texts,indent=2,ensure_ascii=False))
