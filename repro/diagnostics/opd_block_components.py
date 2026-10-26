"""Frozen deterministic Block intervention diagnostics. Never updates weights.
Raw block traces, row-shaped LM-head projection and hybrid scoring are archived.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import copy
import hashlib
import json
import os
import torch
from shared import common,training_resume
from opd_official_kl import runtime,adapters
from opd_official_kl.trainer import move_trace
from opd_official_kl.efficiency import capture_gradients,compare_gradients,validate_trace
from opd_efficiency_gate import BASE,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();training_resume.validate_output(args.out);args.out.mkdir(parents=True,exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic environment required')
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    manifest=json.loads((args.baseline/'manifest_before.json').read_text());cfg=manifest['config'];slots=manifest['slots']
    for p,sha in manifest['hashes'].items():
        if training_resume.file_hash(p)!=sha:raise RuntimeError('input changed')
    import pandas as pd
    rows=pd.read_parquet(cfg['pool']).set_index('task',verify_integrity=True)
    target,processor=common.load_target(device='cuda:0',dtype=torch.float32);target.eval().requires_grad_(False)
    head=common.load_head(str(BASE/'step120'),device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(True)
    state=torch.load(BASE/'step120/training_state.pt',map_location='cpu',weights_only=False,mmap=True)
    head.load_state_dict(state['head']);training_resume.restore_rng(state['rng'])
    refs=torch.load(args.baseline/'fixed_traces.pt',map_location='cpu',weights_only=False)
    blocks=[];rowed=[];contexts=[];pending=[];rowlogp={}
    def before(mod,a):
        if a[0].shape[-2]==16:pending.append(a[0].detach().clone())
    handle=target.lm_head.register_forward_pre_hook(before)
    def observe(pos,ids,logits):
        if len(pending)!=1:raise RuntimeError('block projection hook alignment failed')
        hidden=pending.pop()
        # Same block hidden; change ONLY LM-head call shape to serial's single row.
        rowlogp[pos]=torch.stack([target.lm_head(hidden[:,i:i+1])[0,0].float().log_softmax(-1)[ids[i+1]] for i in range(15)]).cpu()
    try:
        for s,ref in zip(slots,refs):
            rowlogp={};inputs=adapters.encode(processor,rows.loc[s['task']].to_dict(),'image',max_pixels=cfg['max_pixels'])
            tr=adapters.collect(target,head,inputs,max_new_tokens=cfg['max_new_tokens'],max_seq_len=cfg['max_seq_len'],teacher_verify_impl='block',verification_observer=observe)
            tr=move_trace(tr,'cpu');identity=dict(ref['efficiency_identity'])
            identity['config_sha256']=hashlib.sha256((identity['config_sha256']+'block_component_diagnostic').encode()).hexdigest()
            tr['efficiency_identity']=identity;tr['diagnostic_verifier_impl']='block'
            validate_trace(tr,identity)
            if not torch.equal(tr['ids'],ref['ids']):raise RuntimeError('token divergence')
            for k in ('plan','boundaries','prompt_len','final_cache_length'):
                if tr[k]!=ref[k]:raise RuntimeError('discrete trace divergence: '+k)
            if len(tr['rounds'])!=len(ref['rounds']):raise RuntimeError('round count differs')
            for a,b in zip(ref['rounds'],tr['rounds']):
                for k in a:
                    if k!='teacher_logp' and a[k]!=b[k]:raise RuntimeError('round discrete divergence: '+k)
            contexts.append(dict(slot=s['slot'],hidden_bitwise=torch.equal(ref['hidden'],tr['hidden']),
                fused_bitwise=torch.equal(ref['fused'],tr['fused']),hidden_max_abs=float((ref['hidden']-tr['hidden']).abs().max()),
                fused_max_abs=float((ref['fused']-tr['fused']).abs().max()),
                teacher_block_max_abs=max(float((a['teacher_logp']-b['teacher_logp']).abs().max()) for a,b in zip(ref['rounds'],tr['rounds'])),
                teacher_row_projection_max_abs=max(float((a['teacher_logp']-rowlogp[a['pos']]).abs().max()) for a in ref['rounds'])))
            blocks.append(tr);rowed.append(rowlogp)
            print(json.dumps(contexts[-1]),flush=True)
    finally:handle.remove()
    torch.save(blocks,args.out/'block_traces.pt');torch.save(rowed,args.out/'row_projection_logp.pt')
    write(args.out/'CONTEXT_COMPONENTS.json',contexts)
    def hybrid(context,teacher,name):
        out=[]
        for s,(tr,source) in enumerate(zip(context,teacher)):
            cp=dict(tr);cp['diagnostic_intervention']=name
            cp['rounds']=[dict(r,teacher_logp=(source['rounds'][i]['teacher_logp'] if 'rounds' in source else source[r['pos']])) for i,r in enumerate(tr['rounds'])]
            out.append(cp)
        return out
    variants={'serial_reference':refs,'block_full':blocks,
        'serial_context_block_teacher':hybrid(refs,blocks,'teacher_only'),
        'block_context_serial_teacher':hybrid(blocks,refs,'context_only'),
        'block_context_row_projected_teacher':hybrid(blocks,rowed,'row_lm_head_only')}
    reference=torch.load(args.reference/'legacy.pt',map_location='cpu',weights_only=False)
    denom=sum(runtime.denominator(t) for t in refs);reports=[];selected_reference=None
    original_pool=runtime.pool;selected=[]
    def capture(*a,**kw):
        selected.append(tuple(x.detach().cpu().clone() for x in a[:4]));return original_pool(*a,**kw)
    runtime.pool=capture
    for name,traces in variants.items():
        head.zero_grad(set_to_none=True);selected=[];num=0.
        if sum(runtime.denominator(t) for t in traces).hex()!=denom.hex():raise RuntimeError('denominator changed')
        for tr in traces:
            gpu=move_trace(tr,'cuda:0')
            for stat in runtime.score(target,head,gpu,draft_fn=adapters.draft):
                (stat.numerator/denom).backward();num+=float(stat.numerator.detach())
            del gpu,stat
        grads=capture_gradients(head);torch.save(grads,args.out/(name+'_gradients.pt'))
        report=compare_gradients(reference,grads)
        report.update(variant=name,loss=num/denom,bitwise=all(torch.equal(g,reference[n]) for n,g in grads.items()),
            strict_1e6=all(r['max_abs']<=1e-6 for r in report['tensors']),optimizer_steps=0,block_status='HOLD')
        if selected_reference is None:selected_reference=selected
        diffs=[0.]*4
        if len(selected)!=len(selected_reference):raise RuntimeError('selected anchor count differs')
        for a,b in zip(selected_reference,selected):
            for i,(x,y) in enumerate(zip(a,b)):
                if x.shape!=y.shape:raise RuntimeError('selected shape differs')
                if x.numel():diffs[i]=max(diffs[i],float((x-y).abs().max()))
        report['selected_logp_max_abs']=dict(zip(['response_student','response_teacher','rejected_student','rejected_teacher'],diffs))
        write(args.out/(name+'.json'),report);reports.append(dict(name=name,bitwise=report['bitwise'],relative_l2=report['relative_l2'],strict_1e6=report['strict_1e6']))
        print(json.dumps(reports[-1]),flush=True);del grads
    runtime.pool=original_pool
    after={p:training_resume.file_hash(p) for p in manifest['hashes']}
    if after!=manifest['hashes']:raise RuntimeError('input changed')
    write(args.out/'RESULT.json',dict(status='DIAGNOSED_HOLD',reports=reports,optimizer_steps=0,input_hashes_unchanged=True))

if __name__=='__main__':main()
