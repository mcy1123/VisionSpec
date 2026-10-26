"""Pure protocol helpers: no model, training, test split or gold-conditioned prompt."""
import hashlib, json, random, re, collections
SEED=20261013
TEMPLATES={
 'A':'Answer the question directly with a short answer. Do not add an explanation.',
 'B':'First answer the question directly. Then explain the visible evidence in the video in 1–2 short sentences. If the video does not allow you to confirm the answer, say that you cannot confirm it. Do not invent details.'}
def prompt(row, mode):
 text='Question: '+row['question']
 if row.get('options'): text+='\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(row['options']))
 return text+'\n'+('Give the option letter first. ' if row.get('options') else '')+TEMPLATES[mode]
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def cache_key(row,mode,processor_hash):
 return digest({'video_hash':row['hash'],'frames':row['sample_frame_ids'],'timestamps':row['sample_timestamps'],'prompt':prompt(row,mode),'processor':processor_hash})
def source_id(name):
 s=name[2:] if name.startswith('v_') and len(name)==13 else name
 # Dataset filenames preserve YouTube's 11-character ID, including '_' or '-'.
 return s[:11] if re.match(r'^[a-zA-Z0-9_-]{11}(?:_|\.|$)',s) else None

def select(rows,n=20,seed=SEED):
 """Video-stratified round-robin by representative question type and length tertile.
 Questions are picked across types where possible. No answer/model scores used.
 """
 by=collections.defaultdict(list)
 for r in rows:
  assert r['split'] in ('train','val','dev')
  by[r['video_id']].append(r)
 ids=sorted(by)
 ordered=sorted(ids,key=lambda v:(sum(len(r['question'].split()) for r in by[v])/len(by[v]),v))
 bins={v:min(2,3*i//len(ordered)) for i,v in enumerate(ordered)}
 groups=collections.defaultdict(list)
 for v in ids:
  types=collections.Counter(r['question_type'] for r in by[v]);typ=sorted(types,key=lambda x:(-types[x],x))[0]
  groups[(typ,bins[v])].append(v)
 rng=random.Random(seed)
 for g in sorted(groups):rng.shuffle(groups[g])
 picked=[]
 while len(picked)<min(n,len(ids)):
  for g in sorted(groups):
   if groups[g] and len(picked)<n:picked.append(groups[g].pop())
 result=[]
 for v in picked:
  qs=sorted(by[v],key=lambda x:str(x['question_id']));rng.shuffle(qs)
  first=qs[0];second=next((r for r in qs[1:] if r['question']!=first['question'] and r['question_type']!=first['question_type']),None)
  if second is None:second=next((r for r in qs[1:] if r['question']!=first['question']),None)
  for q in [first]+([second] if second else []):result.append({**q,'question_length_stratum':bins[v],'sample_seed':seed})
 return picked,result

def exact_tokens(a,b):return a==b

def clipped_rounds(rounds,n_in,gen_tokens):
 """Classify final emitted positions once: first root, accepted draft, bonus.
 Later roots are already-emitted prior bonus tokens, never count them twice.
 """
 out=[]
 for r in rounds:
  root=r['pos']-n_in;raw=r['accepted'];accepted=max(0,min(raw,gen_tokens-root-1))
  bonus_pos=root+raw+1
  out.append({'root_position':root,'root_present':int(root<gen_tokens),'new_root':int(root==0 and gen_tokens>0),'accepted_raw':raw,'accepted_effective':accepted,'bonus_emitted':int(bonus_pos<gen_tokens),'clipped_draft':raw-accepted,'first_rejected_offset':raw+1 if bonus_pos<gen_tokens else None,'terminal_clipping':max(0,root+raw+2-gen_tokens)})
 return out

def normalize_answer(s):return re.sub(r'[^\w\s]','',s.lower()).strip()

def video_average(rows,key):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['video_id']].append(float(r[key]))
 return {v:sum(x)/len(x) for v,x in sorted(groups.items())}

def paired(a,b,n=2000,seed=SEED):
 import numpy as np
 if set(a)!=set(b) or not a:raise ValueError('nonempty identical video identities required')
 d=np.array([a[v]-b[v] for v in sorted(a)]);rng=np.random.default_rng(seed)
 draws=d[rng.integers(len(d),size=(n,len(d)))].mean(1)
 return {'mean':float(d.mean()),'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),'positive':int((d>0).sum()),'negative':int((d<0).sum()),'unchanged':int((d==0).sum()),'n_videos':len(d)}


def choice_answer(text,options):
 """Conservative MC parsing; ambiguous output stays unparseable, never uses gold."""
 first=text.strip().splitlines()[0] if text.strip() else ''
 m=re.match(r'^(?:answer\s*:\s*|option\s+)([A-E])\b|^([A-E])(?:[.):](?:\s|$)|$)|^\(([A-E])\)',first,re.I)
 if m:return next(x for x in m.groups() if x).upper()
 n=normalize_answer(first)
 matches=[chr(65+i) for i,x in enumerate(options) if normalize_answer(x)==n or n.startswith(normalize_answer(x)+' ') or n.startswith(normalize_answer(x)+'.')]
 return matches[0] if len(matches)==1 else None
