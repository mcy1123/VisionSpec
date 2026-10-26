"""Diagnostic tensor hooks and repeated single-parameter autograd.grad. No updates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import json
import torch
import torch.nn.functional as F
from shared import common,training_resume
from opd_official_kl import adapters,runtime
from opd_official_kl.trainer import move_trace
from opd_efficiency_gate import BASE,write


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); training_resume.validate_output(args.out); args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    target,_=common.load_target(device='cuda:0',dtype=torch.float32);target.eval().requires_grad_(False)
    head=common.load_head(str(BASE/'step120'),device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(True)
    traces=torch.load(args.baseline/'fixed_traces.pt',map_location='cpu',weights_only=False)
    denom=sum(runtime.denominator(t) for t in traces)
    # Last anchor of slot0: keep full original rounds, committed tensors and global denominator.
    trace=dict(traces[0]);trace['plan']=[trace['plan'][-1]];trace=move_trace(trace,'cuda:0')
    refs={}; rows=[]; current={}; calls={}; handles=[]; sdpa_inputs=[]
    def record(name,tensor):
        value=tensor.detach().cpu().clone()
        if name not in refs:refs[name]=value
        current[name]=dict(max_abs=float((value.float()-refs[name].float()).abs().max()),bitwise=torch.equal(value,refs[name]))
    def watch(name,tensor):
        record(name+'/forward',tensor)
        if tensor.requires_grad:
            tensor.register_hook(lambda g: record(name+'/backward',g))
    for name,module in head.named_modules():
        if any(name.endswith(s) for s in ('input_layernorm','post_attention_layernorm','k_proj','v_proj','q_norm','k_norm','o_proj')):
            def hook(mod,inputs,output,name=name):
                index=calls.get(name,0);calls[name]=index+1
                watch(f'{name}/{index}',output)
            handles.append(module.register_forward_hook(hook))
    original=F.scaled_dot_product_attention
    def wrapped(q,k,v,*a,**kw):
        i=calls.get('sdpa',0);calls['sdpa']=i+1
        for label,tensor in zip(('q','k','v'),(q,k,v)):watch(f'sdpa{i}/{label}',tensor)
        output=original(q,k,v,*a,**kw);watch(f'sdpa{i}/output',output)
        return output
    F.scaled_dot_product_attention=wrapped
    try:
        for repeat in range(10):
            current={};calls={};head.zero_grad(set_to_none=True)
            stats=list(runtime.score(target,head,trace,draft_fn=adapters.draft))
            (stats[0].numerator/denom).backward()
            rows.append(dict(repeat=repeat,tensors=current,loss=float(stats[0].numerator.detach())/denom))
            write(args.out/f'hooks{repeat:02d}.json',rows[-1])
    finally:
        F.scaled_dot_product_attention=original
        for handle in handles:handle.remove()
    names=['layers.0.input_layernorm.weight','layers.0.self_attn.k_proj.weight','layers.1.self_attn.k_norm.weight']
    params=dict(head.named_parameters());single=[]
    for name in names:
        head.zero_grad(set_to_none=True)
        # Identical forward graph reused: isolates backward from forward changes.
        stats=list(runtime.score(target,head,trace,draft_fn=adapters.draft));loss=stats[0].numerator/denom
        first=None;diffs=[]
        for repeat in range(10):
            grad,=torch.autograd.grad(loss,params[name],retain_graph=repeat<9)
            grad=grad.detach().cpu()
            if first is None:first=grad.clone()
            diffs.append(float((grad.float()-first.float()).abs().max()))
        single.append(dict(name=name,max_abs_distribution=diffs))
    head.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as prof:
        stats=list(runtime.score(target,head,trace,draft_fn=adapters.draft))
        (stats[0].numerator/denom).backward()
    attention_ops=sorted({e.name for e in prof.events() if any(x in e.name.lower() for x in ('attention','flash','fmha'))})
    prof.export_chrome_trace(str(args.out/'single_anchor_profile.json'))
    write(args.out/'ATTENTION_OPS.json',attention_ops)
    # Math SDPA changed replay on the frozen trace: diagnose forward only, never bypass score checks.
    from torch.nn.attention import sdpa_kernel,SDPBackend
    math_divergence=None
    with torch.no_grad():
        for slot,tr in enumerate(traces):
            gpu=move_trace(tr,'cuda:0')
            for rd in gpu['rounds']:
                pos=rd['pos'];root=int(gpu['ids'][pos])
                normal=adapters.draft(head,target,gpu['fused'],anchor_pos=pos,root_token=root)
                with sdpa_kernel(SDPBackend.MATH):
                    alternative=adapters.draft(head,target,gpu['fused'],anchor_pos=pos,root_token=root)
                if normal.argmax(-1).tolist()!=rd['prop']:raise RuntimeError('default forward itself diverged')
                changed=(normal.argmax(-1)!=alternative.argmax(-1)).nonzero().flatten().tolist()
                if changed:
                    math_divergence=dict(slot=slot,anchor_pos=pos,changed_offsets=[i+1 for i in changed],
                        max_logit_diff=float((normal-alternative).abs().max()),
                        normal_top2=[dict(values=normal[i].topk(2).values.tolist(),tokens=normal[i].topk(2).indices.tolist()) for i in changed],
                        math_top2=[dict(values=alternative[i].topk(2).values.tolist(),tokens=alternative[i].topk(2).indices.tolist()) for i in changed])
                    break
            del gpu
            if math_divergence is not None:break
    write(args.out/'MATH_FORWARD_DIVERGENCE.json',dict(first_divergence=math_divergence,loss_checks_bypassed=False))
    write(args.out/'RESULT.json',dict(status='COMPLETE',optimizer_steps=0,slot=0,anchor=trace['plan'],
        hooks='tensor hooks synchronize CPU copies; separate diagnostic, not uninstrumented baseline',
        single_parameter_autograd_grad=single,
        changing_hook_tensors=sorted({n for r in rows for n,v in r['tensors'].items() if not v['bitwise']})))

if __name__=='__main__': main()
