"""Independent noise-aware diagnostic gates. Never changes production/old gates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import gc
import json
import os
import torch
from shared import common,training_resume
from opd_official_kl import runtime,adapters
from opd_official_kl.trainer import move_trace
from opd_official_kl.efficiency import capture_gradients,compare_gradients,SlotGradientSum
from opd_serial_candidate import compare_trace
from opd_efficiency_gate import BASE,write


def noise_gate(report, calibration):
    thresholds={r['name']:r['proposed_noise_threshold'] for r in calibration['tensors']}
    rows=[dict(name=r['name'], max_abs=r.get('max_abs'), threshold=thresholds[r['name']],
        passed=r['max_abs']<=thresholds[r['name']] if 'max_abs' in r else r['passed']) for r in report['tensors']]
    return dict(tensors=rows,passed=all(r['passed'] for r in rows) and report['cosine']>=.9999
        and report['relative_l2']<=.01 and report['norm_relative']<=.002)


def discrete_trace(ref,tr):
    if not torch.equal(ref['ids'],tr['ids']):raise RuntimeError('output token divergence')
    for k in ('prompt_len','plan','boundaries','final_cache_length','efficiency_identity'):
        if ref[k]!=tr[k]:raise RuntimeError('discrete trace mismatch: '+k)
    if len(ref['rounds'])!=len(tr['rounds']):raise RuntimeError('round length differs')
    maximum=0.
    for a,b in zip(ref['rounds'],tr['rounds']):
        if a.keys()!=b.keys():raise RuntimeError('round fields differ')
        for k in a:
            if k=='teacher_logp':
                if a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape:raise RuntimeError('teacher dtype/shape differs')
                maximum=max(maximum,float((a[k]-b[k]).abs().max()))
            elif torch.is_tensor(a[k]):
                if not torch.equal(a[k],b[k]):raise RuntimeError('round tensor differs: '+k)
            elif a[k]!=b[k]:raise RuntimeError('round metadata differs: '+k)
    return maximum


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--deterministic',action='store_true')
    args=ap.parse_args();training_resume.validate_output(args.out);args.out.mkdir(parents=True,exist_ok=False)
    cal=json.loads((args.calibration/'CALIBRATION.json').read_text())
    if args.deterministic and (cal['mode']!='deterministic' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):raise ValueError('kernel/calibration mismatch')
    if not args.deterministic and cal['mode']!='default':raise ValueError('kernel/calibration mismatch')
    # This immutable copy is published BEFORE any candidate forward/backward.
    write(args.out/'FROZEN_GATE.json',dict(calibration=cal,calibration_sha=training_resume.file_hash(args.calibration/'CALIBRATION.json'),
        old_gate_unchanged=True,block_release=False,global_thresholds=dict(cosine_min=.9999,relative_l2_max=.01,norm_relative_max=.002)))
    torch.set_num_threads(8);torch.use_deterministic_algorithms(args.deterministic)
    manifest=json.loads((args.baseline/'manifest_before.json').read_text());cfg=manifest['config'];slots=manifest['slots']
    for p,sha in manifest['hashes'].items():
        if training_resume.file_hash(p)!=sha:raise RuntimeError('input SHA mismatch')
    import pandas as pd
    rows=pd.read_parquet(cfg['pool']).set_index('task',verify_integrity=True)
    target,processor=common.load_target(device='cuda:0',dtype=torch.float32);target.eval().requires_grad_(False)
    head=common.load_head(str(BASE/'step120'),device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(True)
    state=torch.load(BASE/'step120/training_state.pt',map_location='cpu',weights_only=False)
    initial,rng=state['head'],state['rng'];del state
    refs=torch.load(args.baseline/'fixed_traces.pt',map_location='cpu',weights_only=False)
    refgrads=torch.load(args.calibration/'reference_gradients.pt',map_location='cpu',weights_only=False)
    results=[];selected_reference=None
    original_pool=runtime.pool
    selected=[]
    def capture_pool(*a,**kw):
        selected.append(tuple(x.detach().cpu().clone() for x in a[:4]))
        return original_pool(*a,**kw)
    runtime.pool=capture_pool
    for variant in ('heldout_legacy','qa_fp32_reference','batch_ids','block_diagnostic'):
        selected=[]
        head.load_state_dict(initial);head.zero_grad(set_to_none=True);training_resume.restore_rng(rng)
        teacher_diff=0.
        if variant in ('heldout_legacy','qa_fp32_reference'):traces=refs
        else:
            traces=[]
            for slot,ref in zip(slots,refs):
                inputs=adapters.encode(processor,rows.loc[slot['task']].to_dict(),'image',max_pixels=cfg['max_pixels'])
                tr=adapters.collect(target,head,inputs,max_new_tokens=cfg['max_new_tokens'],max_seq_len=cfg['max_seq_len'],
                    batch_teacher_ids=variant=='batch_ids',teacher_verify_impl='block' if variant=='block_diagnostic' else 'serial_reference')
                tr['efficiency_identity']=ref['efficiency_identity'];tr=move_trace(tr,'cpu')
                try:
                    if variant=='batch_ids':compare_trace(ref,tr)
                    else:teacher_diff=max(teacher_diff,discrete_trace(ref,tr))
                except RuntimeError as e:
                    torch.save(tr,args.out/f'{variant}_divergence_slot{slot["slot"]}.pt')
                    write(args.out/f'{variant}.json',dict(status='DISCRETE_HOLD',error=str(e),optimizer_steps=0))
                    raise
                traces.append(tr);del inputs
        denom=sum(runtime.denominator(t) for t in traces)
        if denom.hex()!=sum(runtime.denominator(t) for t in refs).hex():raise RuntimeError('denominator mismatch')
        agg=SlotGradientSum([n for n,_ in head.named_parameters()]);num=0.
        for slot,tr in enumerate(traces):
            if variant=='qa_fp32_reference':head.zero_grad(set_to_none=True)
            gpu=move_trace(tr,'cuda:0')
            for stat in runtime.score(target,head,gpu,draft_fn=adapters.draft):
                (stat.numerator/denom).backward();num+=float(stat.numerator.detach())
            if variant=='qa_fp32_reference':agg.add(slot,capture_gradients(head))
            del gpu,stat
        grads=agg.finish() if variant=='qa_fp32_reference' else capture_gradients(head)
        old=compare_gradients(refgrads,grads);noise=noise_gate(old,cal)
        if selected_reference is None:selected_reference=selected
        if len(selected)!=len(selected_reference):raise RuntimeError('selected logp anchor count differs')
        selected_diff=0.
        for a,b in zip(selected_reference,selected):
            for x,y in zip(a,b):
                if x.shape!=y.shape or x.dtype!=y.dtype:raise RuntimeError('selected logp shape/dtype differs')
                if x.numel():selected_diff=max(selected_diff,float((x-y).abs().max()))
        bitwise=all((g is None and refgrads[n] is None) or (g is not None and refgrads[n] is not None and torch.equal(g,refgrads[n])) for n,g in grads.items())
        report=dict(variant=variant,gradient_bitwise_exact=bitwise,selected_logp_max_abs=selected_diff,selected_logp_pass=selected_diff<=.001,legacy_gate=old,noise_gate=noise,loss=num/denom,
            loss_pass=abs(num/denom-cal['losses'][0])<=1e-4+1e-3*abs(cal['losses'][0]),
            max_teacher_logp_diff=teacher_diff,teacher_logp_pass=teacher_diff<=.001,
            denominator_hex=denom.hex(),optimizer_steps=0,discrete_trace_exact=True,
            block_status='HOLD' if variant=='block_diagnostic' else None)
        report['diagnostic_passed']=noise['passed'] and report['loss_pass'] and report['selected_logp_pass'] and report['teacher_logp_pass']
        write(args.out/f'{variant}.json',report);results.append(dict(variant=variant,old_pass=old['passed'],noise_pass=noise['passed'],diagnostic_passed=report['diagnostic_passed']))
        print(json.dumps(results[-1]),flush=True)
        del grads,agg,traces;head.zero_grad(set_to_none=True);gc.collect()
        if variant=='heldout_legacy' and not report['diagnostic_passed']:
            raise RuntimeError('heldout baseline failed preregistered gate; stop before candidates')
    after={p:training_resume.file_hash(p) for p in manifest['hashes']}
    write(args.out/'inputs_after.json',dict(unchanged=after==manifest['hashes'],hashes=after))
    if after!=manifest['hashes']:raise RuntimeError('input changed')
    write(args.out/'RUN_STATUS.json',dict(status='DIAGNOSTIC_COMPLETE',results=results,optimizer_steps=0,block_status='HOLD'))

if __name__=='__main__':main()
