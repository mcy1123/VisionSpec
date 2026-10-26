"""Index official uncompressed tar with 512-byte range requests, never full download."""
import argparse, json, pathlib, tarfile, requests, time
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
s=requests.Session(); offset=0; n=0
if out.exists():
 records=[json.loads(x) for x in out.read_text().splitlines()]
 if records: offset=records[-1]['next_offset'];n=len(records)
with out.open('a') as f:
 while True:
  for retry in range(3):
   try:
    r=s.get(a.url,headers={'Range':f'bytes={offset}-{offset+511}'},timeout=20,stream=True)
    if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'): raise RuntimeError(f'Range refused: {r.status_code} {r.headers.get("Content-Range")}')
    b=r.content; assert len(b)==512;break
   except Exception:
    if retry==2:raise
    time.sleep(1)
  if b==bytes(512): break
  t=tarfile.TarInfo.frombuf(b,'utf-8','strict')
  rec={'name':t.name,'size':t.size,'offset':offset,'data_offset':offset+512,'next_offset':offset+512+((t.size+511)//512)*512,'type':t.type.decode(),'url':a.url}
  f.write(json.dumps(rec)+'\n');f.flush();offset=rec['next_offset'];n+=1
  if n%100==0:print(n,offset,t.name,flush=True)
print('complete',n,offset,flush=True)
