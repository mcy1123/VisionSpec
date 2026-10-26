#!/usr/bin/env python3
"""Bounded GPU preparation/checks, invoked only by the user-started controller."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import configure
configure()
import numpy as np
import pandas as pd
import torch
from shared import common
from scienceqa_image.full_epoch.data import WARM, write_json


def labels(run, shard, shards):
    from scienceqa_image.build_eval_ar import encode
    import glance
    target, proc = common.load_target(device='cuda:0', dtype=torch.float32)
    torch.set_grad_enabled(False)
    eos = int(proc.tokenizer.eos_token_id)
    rows = pd.read_parquet(run/'dataset/pool.parquet').to_dict('records')
    dest = run/'dataset/labels'; dest.mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        if i % shards != shard: continue
        path = dest/(row['pid'].replace(':','_')+'.pt')
        if path.exists(): continue
        start = time.time()
        ids, pos, _ = encode(proc,row,768,'cuda:0')
        full = glance.greedy_generate(target, ids, 256, [eos], positions=pos)
        # API can return (tokens, stats) only when requested; guard unexpected variants.
        if isinstance(full, tuple): full = full[0]
        tokens = full[0,ids.shape[1]:].tolist()
        if full.shape[1] > 4096: raise ValueError(f"{row['pid']}: over sequence limit")
        record = dict(task=row['pid'], input_ids=full[0].cpu(), prompt_len=ids.shape[1],
            eos_pos_in_answer=tokens.index(eos) if eos in tokens else None,
            n_answer_tokens=len(tokens), verifier_argmax=tokens)
        tmp=path.with_suffix('.pending'); torch.save(record,tmp); tmp.replace(path)
        with (run/f'labels_cost_{shard}.jsonl').open('a') as f:
            f.write(json.dumps(dict(task=row['pid'], seconds=time.time()-start,
                target_forward_calls=len(tokens), target_input_tokens=int(ids.numel())+max(0,len(tokens)-1)))+'\n')
        print(f'labels {i+1}/{len(rows)} {row["pid"]}',flush=True)


def teacher_gate(run):
    from opd_official_kl import adapters, runtime
    from opd_official_kl.plan import verify
    from glance.head import context_feature
    from glance.vl import VisionPositions
    from shared.draft_geometry import eos_ids
    from transformers import DynamicCache
    target, proc = common.load_target(device='cuda:0', dtype=torch.float32)
    target.eval().requires_grad_(False)
    head = common.load_head(WARM,device='cuda:0',dtype=torch.bfloat16).eval()
    rows = pd.read_parquet(run/'dataset/pool.parquet').iloc[:2]
    reports=[]; rejected_total=0
    for row in rows.to_dict('records'):
        inputs=adapters.encode(proc,row,'image',max_pixels=768)
        trace=adapters.collect(target,head,inputs,max_new_tokens=256,max_seq_len=4096)
        x={k:v.to('cuda:0') for k,v in inputs.items() if hasattr(v,'to')}
        def replay(ids):
            positions=VisionPositions(x['pixel_values'],x['image_grid_thw'])
            positions.bind(target,ids)
            with torch.no_grad():
                return target.model(ids,past_key_values=DynamicCache(),use_cache=True,
                    output_hidden_states=True,**positions.ar_prefill_inputs(ids))
        ref=dict(trace,rounds=[])
        max_lp=0.; rejected_lp=0.; labels_equal=True
        with torch.no_grad():
            for r in trace['rounds']:
                seq=torch.cat([trace['ids'][:r['pos']], trace['ids'].new_tensor([int(trace['ids'][r['pos']])]+r['prop'])])[None]
                out=replay(seq)
                logits=target.lm_head(out.last_hidden_state[:,r['pos']:])[0].float()
                logp=logits[:-1].log_softmax(-1).gather(1,seq.new_tensor(r['prop'])[:,None]).squeeze(1)
                difference=(logp-r['teacher_logp']).abs()
                max_lp=max(max_lp,float(difference.max()))
                if r['rejected_offsets']:
                    rejected_total+=len(r['rejected_offsets'])
                    rejected_lp=max(rejected_lp,float(difference[seq.new_tensor(r['rejected_offsets'])-1].max()))
                decision=verify(r['prop'],logits.argmax(-1).tolist(),eos_ids(target),
                    min(256-(r['pos']+1-trace['prompt_len']),4096-(r['pos']+1)))
                labels_equal &= all(decision[k]==r[k] for k in decision)
                ref['rounds'].append(dict(r,teacher_logp=logp.detach()))
                del out,logits
            out=replay(trace['ids'][None])
            ref['hidden']=out.last_hidden_state.detach()
            ref['fused']=context_feature(out.hidden_states,layer_ids=common.KEPT_LAYERS).detach()
            del out
        proposal_differences=[]
        preliminary=dict(task=row['task'],max_teacher_logp_abs=max_lp,
            max_rejected_teacher_logp_abs=rejected_lp,acceptance_labels_equal=labels_equal,
            gradient_status='pending')
        write_json(run/'TEACHER_GATE.json',dict(passed=False,status='scoring',rows=reports+[preliminary]))
        def backward(t, reference=False):
            head.zero_grad(set_to_none=True); value=0.
            if reference:
                from scienceqa_image.full_epoch.fixed_trace_score import score
                stats=score(target,head,t,draft_fn=adapters.draft,diagnostics=proposal_differences)
            else:
                stats=runtime.score(target,head,t,draft_fn=adapters.draft)
            for stat in stats:
                loss=stat.numerator/runtime.denominator(t); value+=float(loss.detach()); loss.backward()
            return value
        native_loss=backward(trace)
        native_grad={n:p.grad.detach().cpu().clone() for n,p in head.named_parameters() if p.grad is not None}
        reference_loss=backward(ref,reference=True)
        delta=norm=0.
        for n,p in head.named_parameters():
            if p.grad is None:
                if n in native_grad: raise ValueError('gradient support mismatch')
                continue
            g=p.grad.detach().cpu().float(); other=native_grad.pop(n).float()
            delta+=float((g-other).double().square().sum()); norm+=float(g.double().square().sum())
        if native_grad: raise ValueError('gradient support mismatch')
        grad_rel=(delta/max(norm,1e-30))**.5
        loss_rel=abs(native_loss-reference_loss)/max(abs(reference_loss),1e-12)
        passed=labels_equal and max_lp<=0.002 and loss_rel<=0.001 and grad_rel<=0.01
        reports.append(dict(task=row['task'],max_teacher_logp_abs=max_lp,
            max_rejected_teacher_logp_abs=rejected_lp,acceptance_labels_equal=labels_equal,
            native_loss=native_loss,reference_loss=reference_loss,loss_relative=loss_rel,
            gradient_relative_l2=grad_rel,reference_proposal_differences=proposal_differences,passed=passed))
        write_json(run/'TEACHER_GATE.json',dict(passed=False,rows=reports,
            explanation='Same token prefixes, independent full causal replay. Tolerances are engineering gates, not serial equivalence proof.'))
        if not passed: raise RuntimeError('teacher/loss/gradient gate failed; no threshold relaxation or automatic retry')
    if rejected_total==0: raise RuntimeError('quick gate did not exercise rejected teacher; stop rather than claim coverage')
    write_json(run/'TEACHER_GATE.json',dict(passed=True,rows=reports,rejected_positions=rejected_total,
        teacher_atol=.002,loss_rtol=.001,gradient_relative_l2_limit=.01,
        scope='2 QA at Warm; independent same-prefix full causal replay; not serial rollout equivalence'))


def compare(run, arm, a, b):
    # mmap keeps the two ~14-GiB states from being duplicated in anonymous RAM.
    left=torch.load(a/'training_state.pt',map_location='cpu',weights_only=False,mmap=True)
    right=torch.load(b/'training_state.pt',map_location='cpu',weights_only=False,mmap=True)
    differences=[]
    def check(x,y,path):
        if torch.is_tensor(x):
            if x.dtype!=y.dtype or x.shape!=y.shape or not torch.equal(x,y): differences.append(path)
        elif isinstance(x,np.ndarray):
            if not np.array_equal(x,y): differences.append(path)
        elif isinstance(x,dict):
            if x.keys()!=y.keys(): differences.append(path+'.keys'); return
            for k in x: check(x[k],y[k],path+'.'+str(k))
        elif isinstance(x,(tuple,list)):
            if len(x)!=len(y): differences.append(path+'.length'); return
            for i,(u,v) in enumerate(zip(x,y)): check(u,v,path+f'.{i}')
        elif x!=y: differences.append(path)
    keys=['head','master_optimizer','rng','update_step','data_cursor','resume_identity']
    if arm=='opd': keys+=['parallel_state','training_identity']
    for k in keys: check(left[k],right[k],k)
    write_json(run/f'RESUME_{arm}.json',dict(passed=not differences,comparison='continuous2 vs 1+restart+1; exact states',differences=differences[:100]))
    if differences: raise RuntimeError(f'{arm} recovery mismatch; stop: {differences[:5]}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['labels','teacher','compare'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--arm');p.add_argument('--left',type=Path);p.add_argument('--right',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    if a.mode=='labels': labels(a.run,a.shard,a.shards)
    elif a.mode=='teacher':
        try:
            teacher_gate(a.run)
        except BaseException as exc:
            path=a.run/'TEACHER_GATE.json'
            report=json.loads(path.read_text()) if path.exists() else {}
            report.update(passed=False,status='failed',error=repr(exc),traceback=__import__('traceback').format_exc())
            write_json(path,report)
            raise
    else: compare(a.run,a.arm,a.left,a.right)
