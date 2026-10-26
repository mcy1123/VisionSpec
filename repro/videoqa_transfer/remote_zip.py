"""HTTP range-only ZIP/ZIP64 reader. Refuses whole-file responses."""
import io,requests,zipfile,json,argparse,pathlib
class RemoteZipFile(io.RawIOBase):
 def __init__(self,url):
  self.url=url;self.session=requests.Session();self.pos=0;self.downloaded=0
  r=self.session.get(url,headers={'Range':'bytes=0-0'},timeout=30,stream=True)
  if r.status_code!=206:
   r.close();raise RuntimeError('server refuses range')
  self.size=int(r.headers['Content-Range'].split('/')[-1]);r.close()
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if not 0<=self.pos<=self.size:raise ValueError('seek outside archive')
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if not n:return b''
  if n>32*1024*1024:raise RuntimeError('refuse >32MiB single read; never download complete archive')
  start=self.pos;end=start+n-1;r=self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=60,stream=True)
  if r.status_code!=206 or not r.headers['Content-Range'].startswith(f'bytes {start}-{end}/'):
   r.close();raise RuntimeError('invalid range response')
  b=r.content
  if len(b)!=n:raise RuntimeError('short range read')
  self.pos+=n;self.downloaded+=n;return b
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--index',required=True);a=p.parse_args();f=RemoteZipFile(a.url)
 with zipfile.ZipFile(f) as z:idx=[{'name':x.filename,'size':x.file_size,'compressed_size':x.compress_size,'header_offset':x.header_offset,'CRC':x.CRC} for x in z.infolist()]
 pathlib.Path(a.index).write_text(json.dumps({'url':a.url,'archive_size':f.size,'metadata_bytes_read':f.downloaded,'members':idx},indent=2));print(len(idx),f.downloaded)
