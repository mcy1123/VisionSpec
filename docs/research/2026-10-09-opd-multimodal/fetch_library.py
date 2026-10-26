#!/usr/bin/env python3
"""Download verified paper URLs; retain source metadata, PDF hashes and failures.
Run after editing papers_*.json. Uses only Python's standard library.
"""
import concurrent.futures as cf
import difflib
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import threading
import urllib.request

ROOT = Path(__file__).resolve().parent
S2_LIMITED = threading.Event()
class Metadata(HTMLParser):
    def __init__(self):
        super().__init__(); self.values = {}
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'meta' and a.get('name', '').startswith('citation_'):
            self.values.setdefault(a['name'], []).append(a.get('content', ''))
def get(url, limit=60_000_000):
    req = urllib.request.Request(url, headers={'User-Agent':'GLANCE-literature-review/1.0 (personal research)'})
    with urllib.request.urlopen(req, timeout=35) as r:
        data = r.read(limit + 1)
        if len(data) > limit: raise ValueError('Response exceeds download limit')
        return data, r.geturl()
def norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())
def fetch(p):
    key = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(p['id']))
    logpath = ROOT/'metadata'/f'{key}.json'
    if logpath.exists():
        old = json.loads(logpath.read_text())
        if old.get('pdf_status') == 'downloaded': return old
    r = {'id':p['id'], 'requested_title':p['title'], 'source_url':p['url'], 'checked_on':'2026-10-09'}
    pdf_url = p.get('pdf_url')
    try:
        raw, final = get(p['url'], 5_000_000)
        text = raw.decode('utf-8', errors='replace')
        (ROOT/'metadata'/f'{key}.html').write_text(text)
        parser = Metadata(); parser.feed(text)
        r['landing_url'] = final; r['citation_metadata'] = parser.values
        titles = parser.values.get('citation_title')
        if titles:
            r['online_title'] = titles[0]
            r['title_similarity'] = round(difflib.SequenceMatcher(None,norm(p['title']),norm(titles[0])).ratio(),4)
        versions = re.findall(r'(\d{4}\.\d{4,5}v\d+)',text)
        arxiv = re.search(r'(\d{4}\.\d{4,5})', p['url'])
        if arxiv:
            versions = [v for v in versions if v.startswith(arxiv.group(1))]
            if versions:
                version = max(set(versions),key=lambda v:int(v.split('v')[-1]))
                r['arxiv_version'] = version
                pdf_url = f'https://arxiv.org/pdf/{version}'
        r['metadata_status'] = 'fetched'
    except Exception as exc:
        r['metadata_status'] = 'failed'; r['metadata_error'] = str(exc)
    arxiv = re.search(r'(\d{4}\.\d{4,5})',p['url'])
    if arxiv and not S2_LIMITED.is_set():
        try:
            raw,_ = get('https://api.semanticscholar.org/graph/v1/paper/ARXIV:'+arxiv.group(1)+'?fields=title,year,externalIds',200_000)
            r['semantic_scholar'] = json.loads(raw)
        except Exception as exc:
            r['semantic_scholar_error'] = str(exc)
            if '429' in str(exc): S2_LIMITED.set()
    else: r['semantic_scholar_status'] = 'skipped: rate-limited earlier or no arxiv ID'
    try:
        if not pdf_url: raise ValueError('No verified PDF URL supplied')
        raw, final = get(pdf_url)
        if not raw.startswith(b'%PDF-'): raise ValueError('Response is not a PDF')
        pdf = ROOT/'pdfs'/f'{key}.pdf'
        pdf.write_bytes(raw)
        r.update(pdf_status='downloaded', pdf_url=final, local_pdf=str(pdf.relative_to(ROOT)), bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        txt = ROOT/'texts'/f'{key}.txt'
        result = subprocess.run(['pdftotext','-layout',str(pdf),str(txt)],capture_output=True,text=True)
        r['text_status'] = 'extracted' if result.returncode==0 else 'failed'
        if result.returncode==0: r['local_text']=str(txt.relative_to(ROOT))
        else: r['text_error']=result.stderr[:500]
    except Exception as exc:
        r['pdf_status']='failed'; r['pdf_error']=str(exc)
    logpath.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(p['id'],r['metadata_status'],r['pdf_status'],r.get('title_similarity',''),flush=True)
    return r

def main():
    for d in ('pdfs','texts','metadata'): (ROOT/d).mkdir(exist_ok=True)
    papers={}
    for path in sorted(ROOT.glob('papers_*.json')):
        for p in json.loads(path.read_text()): papers.setdefault(p['id'],p)
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(fetch,papers.values()))
    (ROOT/'download_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('Total:',len(results),'PDFs:',sum(r.get('pdf_status')=='downloaded' for r in results))
if __name__=='__main__': main()
