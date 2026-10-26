#!/usr/bin/env python3
"""Build the human-readable library from verified branch records and downloads."""
from pathlib import Path
import collections,csv,hashlib,json,re
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parent
GROUPS={'papers_opd.json':'OPD与草稿训练','papers_multimodal.json':'多模态与视频投机解码','papers_systems.json':'Block草稿、树验证与系统','papers_context.json':'基线、综述与OPD拓展'}
papers={}
for fname,group in GROUPS.items():
 for p in json.loads((ROOT/fname).read_text()):
  assert p['id'] not in papers, p['id']
  p=dict(p);p['group']=group
  m=json.loads((ROOT/'metadata'/f"{p['id']}.json").read_text())
  meta=m.get('citation_metadata',{})
  p['authors']=meta.get('citation_author',p['authors'])
  p['first_submitted']=meta.get('citation_date',[p['date'][:10]])[0].replace('/','-')
  p['latest_updated']=meta.get('citation_online_date',[p['first_submitted']])[0].replace('/','-')
  p['download_version']=m.get('arxiv_version',p.get('latest_version',p.get('version','not resolved')))
  p['pdf_url']=m.get('pdf_url',p['pdf_url'])
  p['local_pdf']=m.get('local_pdf','');p['local_text']=m.get('local_text','')
  p['sha256']=m.get('sha256','');p['pdf_status']=m.get('pdf_status','missing')
  p['source_metadata']=f"metadata/{p['id']}.json"
  p['checked_on']='2026-10-09'
  host=urlparse(p.get('venue_evidence') or '').hostname or ''
  if any(t in p['venue'] for t in ['声明','未独立核实','作者标注']): level='作者声明（非独立录用核验）'
  elif host in ['aclanthology.org','openaccess.thecvf.com','proceedings.mlr.press','proceedings.neurips.cc','papers.neurips.cc','papers.nips.cc','openreview.net','proceedings.iclr.cc']:
   level='官方Findings/Workshop' if ('Findings' in p['venue'] or 'Workshop' in p['venue']) else '官方主会/期刊'
  else: level='预印本'
  p['publication_evidence_level']=level
  papers[p['id']]=p
ordered=sorted(papers.values(),key=lambda p:(list(GROUPS.values()).index(p['group']),p['priority'],p['latest_updated']),reverse=False)
(ROOT/'catalog.json').write_text(json.dumps(ordered,ensure_ascii=False,indent=2)+'\n')
fields=['id','title','group','priority','first_submitted','latest_updated','download_version','venue','publication_evidence_level','venue_evidence','url','pdf_url','local_pdf','code_url','method','glance_use','limitations']
with (ROOT/'catalog.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(ordered)
(ROOT/'notes').mkdir(exist_ok=True)
for p in ordered:
 evidence=f"[录用/身份依据]({p['venue_evidence']})" if p.get('venue_evidence') else '无单独会议录用依据'
 code=f"[作者代码入口]({p['code_url']})" if p.get('code_url') else '本轮未核实代码入口；不代表一定没有代码'
 md=f"""# {p['title']}

[返回总目录](../catalog.md) · {p['group']} · **{p['priority']}**

- 作者：{'；'.join(p['authors'])}。
- arXiv ID：`{p['id']}`；首次提交：{p['first_submitted']}；最新修订：{p['latest_updated']}；本地版本：`{p['download_version']}`。
- 发表状态：**{p['venue']}**。证据级别：{p['publication_evidence_level']}；{evidence}。
- [论文主页]({p['url']}) · [在线PDF]({p['pdf_url']}) · [本地PDF](../{p['local_pdf']}) · [提取文本](../{p['local_text']})。
- 代码：{code}。

## 方法事实

{p['method']} [来源]({p['url']})

## 对当前 GLANCE 的借鉴

{p['glance_use']}

## 不能直接外推的地方

{p['limitations']}

## 核验与使用边界

{p['verification']}

PDF下载状态：{p['pdf_status']}。本地原文下载不等于全文复现或代码可运行。仅提供下一轮研究候选，不改当前 Phase C 的固定目标、验证器或数据划分。

[元数据与下载校验](../{p['source_metadata']})；SHA-256：`{p['sha256']}`。
"""
 for k in ['metadata_note','code_status']:
  if p.get(k): md+=f"\n补充：{p[k]}\n"
 (ROOT/'notes'/f"{p['id']}.md").write_text(md)
lines=['# 论文总目录','',f'共 **{len(ordered)} 篇**；检索日期 2026-10-09。点击题名阅读中文卡片，下载版本与首发/修订日期见卡片。','', '[研究建议](experiment_recommendations.md) · [CSV筛选表](catalog.csv) · [BibTeX](references.bib) · [核验报告](verification_report.md)','', 'P0=直接基线/训练依据，P1=优先候选，P2=条件性迁移，P3=历史/非无损对照。优先级不代表已验证收益。']
for group in GROUPS.values():
 lines+=['',f'## {group}','','| 优先级 | 论文 | 首发 / 最新修订 | 发表状态 | 原文 |','|---|---|---|---|---|']
 for p in sorted([x for x in ordered if x['group']==group],key=lambda p:(p['priority'], ''.join(chr(255-ord(c)) for c in p['latest_updated']))):
  lines.append(f"| {p['priority']} | [{p['title']}](notes/{p['id']}.md) | {p['first_submitted']} / {p['latest_updated']} | {p['venue']} | [PDF]({p['local_pdf']}) |")
(ROOT/'catalog.md').write_text('\n'.join(lines)+'\n')
def tex(s):return s.replace('\\','').replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#')
bib=['% These entries cite the archived arXiv versions, not substituted conference metadata.','% For the published version, use each card\'s venue_evidence and its publisher BibTeX.']
for p in ordered:
 bib.append('@misc{arxiv'+p['id'].replace('.','')+',\n'+f"  title = {{{{{tex(p['title'])}}}}},\n  author = {{{' and '.join(tex(a) for a in p['authors'])}}},\n  year = {{{p['first_submitted'][:4]}}},\n  eprint = {{{p['id']}}},\n  archivePrefix = {{arXiv}},\n  url = {{{p['url']}}},\n  note = {{Archived version {p['download_version']}; accessed 2026-10-09}}\n"+'}')
(ROOT/'references.bib').write_text('\n\n'.join(bib)+'\n')
counts=collections.Counter(p['publication_evidence_level'] for p in ordered)
years=collections.Counter(p['first_submitted'][:4] for p in ordered)
(ROOT/'library_stats.json').write_text(json.dumps({'total':len(ordered),'publication_evidence':counts,'first_submission_years':years,'pdfs':sum(p['pdf_status']=='downloaded' for p in ordered)},ensure_ascii=False,indent=2)+'\n')
print('Built',len(ordered),'cards; publication levels:',dict(counts),'years:',dict(years))
