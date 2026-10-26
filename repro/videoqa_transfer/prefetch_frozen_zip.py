"""Bounded two-requester prefetch of frozen ZIP members, one validated range/member."""
import argparse,pathlib,json,requests,struct,zlib,os,hashlib,concurrent.futures,time
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--index',required=True);p.add_argument('--last',type=int,default=8);a=p.parse_args();R=pathlib.Path(a.run);m=json.loads((R/'sampling_manifest.json').read_text());idx=json.loads(pathlib.Path(a.index).read_text());D=pathlib.Path('/mnt/local2/mcy/datasets/videoqa_transfer')/m['dataset'];by={pathlib.Path(x['name']).name:x for x in idx['members']};selected=[v for v in m['video_ids'][-a.last:] if not (D/v).exists()]
def fetch(v):
 x=by[v];start=x['header_offset'];end=min(idx['archive_size']-1,start+30+2*65535+x['compressed_size']-1);assert end-start+1<32*1024*1024
 t=time.time();failures=[]
 for attempt in range(3):
  try:
   with requests.get(idx['url'],headers={'Range':f'bytes={start}-{end}'},timeout=(10,45),stream=True) as response:
    assert response.status_code==206 and response.headers['Content-Range'].startswith(f'bytes {start}-{end}/'), 'Refuse non-range response'
    b=response.raw.read(end-start+2);assert len(b)==end-start+1
   break
  except Exception as err:
   failures.append(repr(err));print(json.dumps({'video_id':v,'attempt':attempt+1,'error':repr(err)}),flush=True)
   if attempt==2:raise
   time.sleep(2)
 h=struct.unpack('<IHHHHHIIIHH',b[:30]);assert h[0]==0x04034b50;method=h[3];n,e=h[-2:];name=b[30:30+n].decode('utf8');assert name==x['name'];body=b[30+n+e:30+n+e+x['compressed_size']];assert len(body)==x['compressed_size'];raw=zlib.decompress(body,-15) if method==8 else body if method==0 else None;assert raw is not None and len(raw)==x['size'] and zlib.crc32(raw)==x['CRC']
 dest=D/v;tmp=D/(v+'.prefetch.partial');tmp.write_bytes(raw)
 try:os.link(tmp,dest)
 except FileExistsError:assert dest.read_bytes()==raw
 tmp.unlink();out={'video_id':v,'source_url':idx['url'],'range':[start,end],'range_bytes':len(b),'video_bytes':len(raw),'CRC':x['CRC'],'sha256':hashlib.sha256(raw).hexdigest(),'wall_s':time.time()-t,'status':'verified','prior_failures':failures};print(json.dumps(out),flush=True);return out
out=[];errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for f in concurrent.futures.as_completed([ex.submit(fetch,v) for v in selected]):
  try:out.append(f.result())
  except Exception as err:errors.append(repr(err))
  (R/'prefetch_safe_acquisition.json').write_text(json.dumps({'selected_frozen_video_ids':selected,'completed':out,'errors':errors,'max_workers':2,'training_or_model_calls':0},indent=2))
if errors:raise RuntimeError(errors)
