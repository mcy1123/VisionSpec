#!/usr/bin/env python3
"""Frozen current-AR C0 anchor probe. No backward, optimiser, or model writes.

Observe the unmodified production decoder using removable hooks. C0's actual
head logits, target cache BEFORE root verification, and past-only fused context
are captured. Other heads are evaluated at those exact states. Independent
production greedy_generate supplies full-vocabulary B reference distributions.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import DynamicCache
from shared import common
import glance.decode as dec
from glance.tree import build_budget_tree, children_of, tree_mask
from glance.vl import VisionPositions
from historical import opd_fullkl
from diagnostics.diagnostic_acl_metrics import distribution_metrics, local_oracles


OLD=Path(common.RESULTS)/'phaseC5/diagnostic_acl_20261012T213006'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def emit(p,v):
    with open(p,'a') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
def cache_cpu(cache):return [(l.keys.detach().cpu(),l.values.detach().cpu()) for l in cache.layers]
def cache_gpu(data,device):return DynamicCache(ddp_cache_data=[(k.to(device),v.to(device)) for k,v in data])
def leading(values):
    n=0
    for v in values:
        if not v:break
        n+=1
    return n


def candidates(q):
    # Preserve the deployment device's reduction and topk call sizes. Computing
    # log_softmax again on CPU can change heap order even when IDs agree.
    lp=q.float().log_softmax(-1)
    t8=lp.topk(8,dim=-1);t32=lp.topk(32,dim=-1)
    return {'lp':lp.cpu(),'ids8':t8.indices.cpu(),'val8':t8.values.cpu(),
            'ids32':t32.indices.cpu(),'val32':t32.values.cpu()}


def freeze(out):
    """Select only on input length and ID; round index uniformly covers C0 run."""
    old=read(OLD/'completed_tree_audit/manifest.json')
    source=Path(common.RESULTS)/'phaseA4/sft439_step1000_fresh30_tree_float32.json'
    ev=read(source)
    eligible=sorted([r for r in ev['rows'] if not r['warmup']],key=lambda r:(r['prompt_len'],r['video']))
    rng=random.Random(20261012)
    strata=np.array_split(np.arange(len(eligible)),3)
    picks=[]
    for st in strata:
        chosen=rng.sample(list(map(int,st)),2)
        picks.extend([eligible[i] for i in sorted(chosen)])
    # Interleave strata, so pilot covers short and middle lengths.
    picks=[picks[i] for i in [0,2,4,1,3,5]]
    videos=[]
    for r in picks:
        indices=np.linspace(0,len(r['anchor_rel'])-1,min(8,len(r['anchor_rel'])),dtype=int).tolist()
        videos.append({'video':r['video'],'prompt_len':r['prompt_len'],'round_indices':indices,
                       'anchor_rel':[r['anchor_rel'][i] for i in indices],
                       'rounds':r['rounds'],'generated_length':r['gen_tokens']})
    m={'created':datetime.now().astimezone().isoformat(),'root':common.ROOT,'python':sys.executable,
       'seed':20261012,'videos':videos,'pilot_videos':[x['video'] for x in videos[:2]],
       'selection':'sort 29 nonwarmup by (input_length,video_id); 3 equal-count strata, sample2 each using Random(20261012); interleave strata; integer linspace over saved C0 rounds',
       'checkpoint_identity':old['head_checkpoints'],'baseline_eval':str(source),'baseline_sha256':sha(source),
       'target':common.TARGET_PATH,'target_dtype':'float32','head_dtype':'float32','attention':'sdpa',
       'feature_layers':list(common.KEPT_LAYERS),'feature_path':'actual decoder cached past-only context',
       'temperature':0.0,'budget':63,'branch':8,'block':16,'max_new_tokens':1024,
       'formal_head_score_limit':144,'formal_C0_scores_reused_from_actual_decode':True,
       'scripts':{str(Path(__file__).resolve()):sha(__file__),
                  str(Path(common.REPO)/'glance/decode.py'):sha(Path(common.REPO)/'glance/decode.py'),
                  str(Path(common.REPO)/'glance/tree.py'):sha(Path(common.REPO)/'glance/tree.py')},
       'command':sys.argv,'status':'frozen_before_new_model_results'}
    for arm,c in m['checkpoint_identity'].items():
        assert sha(Path(c['path'])/'model.safetensors')==c['model_sha256'],arm
    write(out/'manifest.json',m)
    return m,ev


class Capture:
    def __init__(self,head,target,selected,calls):
        self.head=head;self.target=target;self.selected=set(selected);self.calls=calls
        self.index=0;self.chunks=[];self.saved={};self.lastq=None;self.handles=[]
    def head_pre(self,module,args,kw):
        self.calls['C0_decode_head_forwards']+=1
        self.chunks.append(kw['target_hidden'].detach().cpu())
        if self.index in self.selected:
            pos=kw['position_ids'][0,-1].item()-15
            fused=torch.cat(self.chunks,dim=1)
            assert fused.shape[1]==pos
            self.saved[self.index]={'pos':pos,'fused':fused,'head_position_ids':kw['position_ids'].cpu().tolist(),
                                    'head_cache_len':kw['past_key_values'].get_seq_length()}
    def steer(self,logits,root):
        if self.index in self.selected:
            self.saved[self.index].update(q=logits.detach().cpu(),root=int(root[0]),candidates=candidates(logits))
        return logits
    def build_tree(self,ids,logp,budget,branch):
        nodes=self.original_builder(ids,logp,budget,branch)
        if self.index in self.selected:
            s=self.saved[self.index];s['nodes']=nodes
            assert torch.equal(ids,s['candidates']['ids8']) and torch.equal(logp,s['candidates']['val8'])
        return nodes
    def target_pre(self,module,args,kw):
        self.calls['C0_decode_target_forwards']+=1
        if self.index in self.selected and kw.get('attention_mask') is not None:
            s=self.saved[self.index];cache=kw['past_key_values']
            assert cache.get_seq_length()==s['pos']
            s['kv']=cache_cpu(cache)
            s['packed_input_ids']=args[0].cpu().tolist()[0]
            s['packed_positions']=kw['position_ids'].cpu().tolist()
            mask=kw['attention_mask'][0,0]
            assert bool((mask[:,:s['pos']]==0).all())
            s['packed_visibility']=(mask[:,s['pos']:]==0).cpu().tolist()
    def target_post(self,module,args,kw,output):
        if self.index in self.selected and kw.get('attention_mask') is not None:
            self.saved[self.index]['packed_logits']=output.logits[0].detach().cpu()
    def on_round(self,r):
        if self.index in self.selected:self.saved[self.index]['round']=r
        self.index+=1
    def __enter__(self):
        self.original_builder=dec.build_budget_tree
        dec.build_budget_tree=self.build_tree
        self.handles=[self.head.register_forward_pre_hook(self.head_pre,with_kwargs=True),
                      self.target.register_forward_pre_hook(self.target_pre,with_kwargs=True),
                      self.target.register_forward_hook(self.target_post,with_kwargs=True)]
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        dec.build_budget_tree=self.original_builder


@torch.inference_mode()
def reference(target,ids,pv,vg,saved,expected,calls):
    need=set()
    for s in saved.values():
        rel=s['pos']-ids.shape[1]
        need.update(range(rel+1,min(rel+16,len(expected))))
    logits={};index=[0]
    def hook(module,args,kw,output):
        i=index[0];index[0]+=1;calls['independent_AR_target_forwards']+=1
        if i<len(expected):assert int(output.logits[0,-1].argmax())==expected[i],('AR mismatch',i)
        if i in need:logits[i]=output.logits[0,-1].detach().cpu()
    h=target.register_forward_hook(hook,with_kwargs=True)
    try:
        ar=dec.greedy_generate(target,ids,1024,[151645],positions=VisionPositions(pixel_values_videos=pv,video_grid_thw=vg))
    finally:h.remove()
    assert ar[0,ids.shape[1]:].tolist()==expected
    return logits


@torch.inference_mode()
def verify(target,s,nodes,device,calls):
    cache=cache_gpu(s['kv'],device)
    packed=torch.tensor([[s['root']]+[n.token for n in nodes]],device=device)
    # C0's root M-RoPE position fixes the delta for the same prefix in all arms.
    root_pos=torch.tensor(s['packed_positions'],device=device)[:,:,0:1]
    depths=torch.tensor([0]+[n.depth for n in nodes],device=device)[None,None,:]
    positions=root_pos+depths
    mask=tree_mask(s['pos'],nodes,device,torch.float32)
    assert cache.get_seq_length()==s['pos']
    result=target(packed,position_ids=positions,past_key_values=cache,use_cache=True,
                  attention_mask=mask,output_hidden_states=False)
    calls['local_tree_verification_target_forwards']+=1
    return result.logits[0].cpu(),positions.cpu().tolist(),(mask[0,0,:,s['pos']:]==0).cpu().tolist()


def walk(nodes,logits):
    child=children_of(nodes);cursor=-1;path=[]
    while True:
        token=int(logits[cursor+1].argmax())
        nxt=next((i for i in child[cursor] if nodes[i].token==token),None)
        if nxt is None:return path,token
        path.append(nxt);cursor=nxt


def metrics_and_trace(arm,video,idx,s,q,p,ref,logits,nodes,positions,visible,prompt_len,c0length,cand):
    h=len(ref);lp=cand['lp']
    path,bonus=walk(nodes,logits);actual=[nodes[i].token for i in path]
    oracle=local_oracles(cand['ids8'].tolist(),[asdict(n) for n in nodes],ref,actual,
        budget=63,remaining_after_root=h,root_token=s['root'],eos_id=151645,reference_verified=True)
    assert oracle['L_actual']==oracle['L_tree_oracle'],oracle
    assert oracle['L_candidate_oracle']==oracle['L_raw']
    # Packed logits only have B conditions along the correct path, not other branches.
    good=[];cursor=-1;ch=children_of(nodes)
    for k,token in enumerate(ref):
        assert int(logits[cursor+1].argmax())==token,('packed vs AR',video,idx,arm,k)
        good.append(float((logits[cursor+1]-p[k]).abs().max()))
        nxt=next((i for i in ch[cursor] if nodes[i].token==token),None)
        if nxt is None:break
        cursor=nxt
    vals=distribution_metrics(p,q[:h]) if h else {}
    vals={k:v.tolist() for k,v in vals.items()}
    tokens=[]
    for k in range(h):
        cutoff={}
        for n in [1,8,32]:
            cut=cand['val8'][k,7] if n==8 else cand['val32'][k,n-1]
            cutoff[str(n)]={'equal_count':int((lp[k]==cut).sum()),
                            'straddles_boundary':bool((lp[k]>cut).sum()<n<(lp[k]>=cut).sum())}
        group='tree_covered' if k<c0length else ('next_uncovered' if k==c0length else 'later_offsets')
        tokens.append({'offset':k+1,'reference_token':ref[k],'C0_tree_group':group,
                       **{key:value[k] for key,value in vals.items()},
                       'top1':bool(cand['ids8'][k,0]==ref[k]),
                       'top8':bool((cand['ids8'][k]==ref[k]).any()),
                       'top32':bool((cand['ids32'][k]==ref[k]).any()),'cutoff_ties':cutoff})
    return {'video':video,'arm':arm,'round_index':idx,'pos':s['pos'],'anchor_rel':s['pos']-prompt_len,
            'reference_mode':'B_current_incremental_AR','trajectory_source':'current_deployment_AR',
            'teacher_condition':'correct_AR_prefix','root':s['root'],'H_valid':h,
            'draft_context_last_position':s['pos']-1,'future_context_used':False,
            'context_sha256':tsha(s['fused']),'top32_ids':cand['ids32'].tolist(),'top32_logp':cand['val32'].tolist(),
            'top8_ids':cand['ids8'].tolist(),'top8_logp':cand['val8'].tolist(),
            'nodes':[asdict(n) for n in nodes],'budget_excludes_root':63,'packed_root_cost':1,
            'packed_input_ids':[s['root']]+[n.token for n in nodes],
            'packed_positions':positions,'packed_prefix_visible_columns':s['pos'],
            'packed_visibility':visible,'verification_argmax':logits.argmax(-1).tolist(),
            'selected_node_path':path,'raw_accepted_tokens':actual,'effective_draft_tokens':actual[:h],
            'bonus':bonus,'bonus_survives_output_clip':len(actual)<h,
            'reference_tokens':ref,'metrics':tokens,'oracle':oracle,
            'tree_gap':oracle['L_candidate_oracle']-oracle['L_tree_oracle'],
            'commit_gap':oracle['L_tree_oracle']-oracle['L_actual'],
            'top1_leading_agreement':leading(q[:h].argmax(-1).tolist()[i]==ref[i] for i in range(h)),
            'packed_vs_AR_max_abs_logit':max(good,default=None),'gate':'PASS',
            'termination':'normal_boundary' if oracle['L_actual']==h else oracle['reason'],
            'native_C2_training_loss':None,'B_KL_label':'full-vocabulary AR-reference diagnostic; not draft-induced native loss'}


@torch.inference_mode()
def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    manifest,ev=freeze(out)
    calls=Counter();gates=[];formal=0
    torch.set_num_threads(4)
    torch.manual_seed(20261012)
    device=f'cuda:{a.gpu}'
    target,proc=common.load_target(device=device,dtype=torch.float32)
    heads={arm:common.load_head(c['path'],dtype=torch.float32,device='cpu').eval().requires_grad_(False)
           for arm,c in manifest['checkpoint_identity'].items()}
    pool=pd.read_parquet(Path(common.DATA)/'video_fresh30.parquet').set_index('video_name')
    oldrows={r['video']:r for r in ev['rows']}
    start=time.perf_counter()
    try:
        for vi,v in enumerate(manifest['videos']):
            task=v['video'];row=pool.loc[task];baseline=oldrows[task]
            print('VIDEO',vi,task,'pilot' if vi<2 else 'extension',flush=True)
            if vi==2:
                assert len(gates)==2 and all(g['pass'] for g in gates)
                write(out/'pilot_gate.json',{'pass':True,'videos':gates.copy(),'calls':dict(calls)})
            text=proc.apply_chat_template([{'role':'user','content':[{'type':'video'},
                 {'type':'text','text':str(row.question)}]}],add_generation_prompt=True,tokenize=False)
            ins=proc(text=[text],videos=[row.cache_path],fps=float(row.fps),return_tensors='pt')
            ids=ins['input_ids'].to(device);pv=ins['pixel_values_videos'].to(device);vg=ins['video_grid_thw'].to(device)
            assert ids.shape[1]==v['prompt_len']
            past_inputs=[json.loads(x) for x in (OLD/'input_replay_v2.jsonl').read_text().splitlines()]
            previous=next(x for x in past_inputs if x['video']==task)
            assert tsha(ids[0])==previous['input_ids_sha256'] and tsha(pv)==previous['pixel_sha256']
            write(out/f'input_{task}.json',{'video':task,'input_ids':ids.cpu().tolist(),
                  'input_ids_sha256':tsha(ids[0]),'pixel_sha256':tsha(pv),'grid':vg.cpu().tolist(),
                  'video_sha256':sha(row.cache_path),'question':str(row.question),'previous_metadata':previous['video_metadata']})
            head=heads['C0'].to(device)
            with Capture(head,target,v['round_indices'],calls) as cap:
                generated=dec.generate(head,target,ids,1024,[151645],budget=63,branch_k=8,
                    positions=VisionPositions(pixel_values_videos=pv,video_grid_thw=vg),
                    on_round=cap.on_round,steer=cap.steer,temperature=0.0)
            assert generated[0,ids.shape[1]:].tolist()==baseline['ar_ids']
            assert [cap.saved[i]['pos']-ids.shape[1] for i in v['round_indices']]==v['anchor_rel']
            ref_logits=reference(target,ids,pv,vg,cap.saved,baseline['ar_ids'],calls)
            # Model gate: repeat a late fixed anchor, so cached-vs-full context is exercised.
            gate={'video':task,'pass':True,'current_AR_equals_saved':True,'C0_SPEC_equals_AR':True}
            if vi<2:
                s=cap.saved[v['round_indices'][3]];fused=s['fused'].to(device)
                qs=[]
                for _ in range(2):
                    qs.append(opd_fullkl.draft_block_logits(head,target,fused,anchor_pos=s['pos'],root_token=s['root']).cpu())
                    calls['gate_extra_head_forwards']+=1
                gate['fixed_repeat_max_abs']=float((qs[0]-qs[1]).abs().max())
                gate['cached_vs_fresh_max_abs']=float((qs[0]-s['q']).abs().max())
                gate['repeat_top32_exact']=torch.equal(qs[0].topk(32).indices,qs[1].topk(32).indices)
                gate['cached_vs_fresh_top32_exact']=torch.equal(qs[0].topk(32).indices,s['q'].topk(32).indices)
                assert gate['repeat_top32_exact'] and gate['cached_vs_fresh_top32_exact'],gate
                nodes=s['nodes']
                vv=[]
                for _ in range(2):
                    vv.append(verify(target,s,nodes,device,calls)[0]);calls['gate_verification_subset']+=1
                gate['verification_repeat_max_abs']=float((vv[0]-vv[1]).abs().max())
                gate['verification_repeat_argmax_exact']=torch.equal(vv[0].argmax(-1),vv[1].argmax(-1))
                gate['verification_matches_capture']=torch.equal(vv[0].argmax(-1),s['packed_logits'].argmax(-1))
                assert gate['verification_repeat_argmax_exact'] and gate['verification_matches_capture'],gate
                assert walk(nodes,vv[0])==walk(nodes,vv[1])
                del qs,vv,fused
            gates.append(gate);write(out/'gates.json',gates)
            c0_lengths={}
            for arm in ['C0','C1','C2']:
                heads['C0'].cpu() if arm!='C0' else None
                head=heads[arm].to(device)
                for idx in v['round_indices']:
                    s=cap.saved[idx];rel=s['pos']-ids.shape[1]
                    ref=baseline['ar_ids'][rel+1:rel+16]
                    p=torch.stack([ref_logits[i] for i in range(rel+1,rel+1+len(ref))]) if ref else torch.empty(0,151936)
                    if arm=='C0':q=s['q'];cand=s['candidates']
                    else:
                        qgpu=opd_fullkl.draft_block_logits(head,target,s['fused'].to(device),anchor_pos=s['pos'],root_token=s['root'])
                        cand=candidates(qgpu);q=qgpu.cpu();del qgpu
                        calls['fixed_context_other_head_forwards']+=1
                    formal+=1;assert formal<=144
                    nodes=build_budget_tree(cand['ids8'],cand['val8'],63,8)
                    if arm=='C0':
                        logits=s['packed_logits'];positions=s['packed_positions'];visible=s['packed_visibility']
                        assert [s['root']]+[n.token for n in nodes]==s['packed_input_ids']
                        path,bonus=walk(nodes,logits)
                        assert [nodes[i].token for i in path]==s['round']['path_tokens']
                        c0_lengths[idx]=min(len(path),len(ref))
                    else:logits,positions,visible=verify(target,s,nodes,device,calls)
                    record=metrics_and_trace(arm,task,idx,s,q,p,ref,logits,nodes,positions,visible,ids.shape[1],c0_lengths[idx],cand)
                    emit(out/'trace_current_ar.jsonl',record)
                head.cpu();torch.cuda.empty_cache()
            write(out/'calls.json',{'calls':dict(calls),'formal_head_scores':formal,'elapsed_with_trace_s':time.perf_counter()-start,
                                    'timing_not_speed_evidence':True,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
            del cap,ref_logits,ids,pv,vg,generated
            gc.collect();torch.cuda.empty_cache()
            print('DONE',task,'formal scores',formal,dict(calls),flush=True)
        manifest['status']='completed'
        manifest['formal_head_scores']=formal;manifest['calls']=dict(calls)
        manifest['checkpoint_hashes_after']={arm:sha(Path(c['path'])/'model.safetensors') for arm,c in manifest['checkpoint_identity'].items()}
        assert all(manifest['checkpoint_hashes_after'][arm]==c['model_sha256'] for arm,c in manifest['checkpoint_identity'].items())
        write(out/'manifest_final.json',manifest)
    except Exception as e:
        write(out/'failure.json',{'error':repr(e),'calls':dict(calls),'formal_head_scores':formal,'video':locals().get('task'),
                                 'round_index':locals().get('idx'),'gate':locals().get('gate')})
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True);ap.add_argument('--gpu',type=int,default=0)
    run(ap.parse_args())
