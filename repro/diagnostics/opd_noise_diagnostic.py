"""Frozen-trace baseline noise calibration; never steps an optimizer.
Run each kernel mode in a fresh process. All model/loss semantics stay unchanged.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import argparse
import contextlib
import gc
import json
import os
import time
import torch
from shared import common, training_resume
from opd_official_kl import adapters, runtime
from opd_official_kl.trainer import move_trace
from opd_official_kl.efficiency import capture_gradients, compare_gradients, validate_trace
from opd_efficiency_gate import BASE, write


def run(args, out):
    torch.set_num_threads(8)
    deterministic = args.mode in ('deterministic', 'deterministic_math')
    if deterministic and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process startup')
    torch.use_deterministic_algorithms(deterministic)
    manifest = json.loads((args.baseline/'manifest_before.json').read_text())
    hashes = dict(manifest['hashes'])
    hashes[str(args.baseline/'fixed_traces.pt')] = training_resume.file_hash(args.baseline/'fixed_traces.pt')
    for p, sha in hashes.items():
        if training_resume.file_hash(p) != sha:
            raise RuntimeError('input changed: '+p)
    write(out/'inputs.json', hashes)
    write(out/'environment.json', dict(mode=args.mode, torch=torch.__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32, repeats=args.repeats,
        accumulation='per-anchor GPU fp32 (diagnostic only)' if args.mode == 'fp32_accumulator' else 'legacy bf16',
        target_dtype='float32', head_dtype='bfloat16'))
    target, _ = common.load_target(device='cuda:0', dtype=torch.float32)
    target.eval().requires_grad_(False)
    head = common.load_head(str(BASE/'step120'), device='cuda:0', dtype=torch.bfloat16).eval().requires_grad_(True)
    state = torch.load(BASE/'step120/training_state.pt', map_location='cpu', weights_only=False)
    initial, rng = state['head'], state['rng']
    del state
    traces = torch.load(args.baseline/'fixed_traces.pt', map_location='cpu', weights_only=False)
    for tr in traces:
        validate_trace(tr, tr['efficiency_identity'])
    denom = sum(runtime.denominator(t) for t in traces)
    ref = lo = hi = None
    reports = []
    from torch.nn.attention import sdpa_kernel, SDPBackend
    context = sdpa_kernel(SDPBackend.MATH) if 'math' in args.mode else contextlib.nullcontext()
    with context:
        for repeat in range(args.repeats):
            head.load_state_dict(initial)
            head.zero_grad(set_to_none=True)
            training_resume.restore_rng(rng)
            started = time.perf_counter()
            numerator = 0.
            accumulated = dict.fromkeys(n for n, _ in sorted(head.named_parameters()))
            for trace in traces:
                gpu = move_trace(trace, 'cuda:0')
                for stat in runtime.score(target, head, gpu, draft_fn=adapters.draft):
                    (stat.numerator/denom).backward()
                    numerator += float(stat.numerator.detach())
                    if args.mode == 'fp32_accumulator':
                        for name, p in sorted(head.named_parameters()):
                            if p.grad is not None:
                                g = p.grad.detach().float()
                                if accumulated[name] is None:
                                    accumulated[name] = g.clone()
                                else:
                                    accumulated[name].add_(g)
                        head.zero_grad(set_to_none=True)
                del gpu, stat
            grads = ({n: None if g is None else g.cpu() for n,g in accumulated.items()}
                     if args.mode == 'fp32_accumulator' else capture_gradients(head))
            if ref is None:
                ref = {n: None if g is None else g.clone() for n,g in grads.items()}
                lo = {n: None if g is None else g.clone() for n,g in grads.items()}
                hi = {n: None if g is None else g.clone() for n,g in grads.items()}
                torch.save(ref, out/'reference_gradients.pt')
            report = compare_gradients(ref, grads)
            report.update(repeat=repeat, loss=numerator/denom, denominator_hex=denom.hex(),
                          diagnostic_seconds=time.perf_counter()-started,
                          bitwise=all((g is None and ref[n] is None) or
                                      (g is not None and ref[n] is not None and torch.equal(g,ref[n]))
                                      for n,g in grads.items()))
            for n, g in grads.items():
                if (g is None) != (ref[n] is None):
                    raise RuntimeError('gradient presence changed')
                if g is not None:
                    torch.minimum(lo[n], g, out=lo[n]); torch.maximum(hi[n], g, out=hi[n])
            write(out/f'repeat{repeat:02d}.json', report)
            reports.append(report)
            print(json.dumps(dict(mode=args.mode, repeat=repeat, bitwise=report['bitwise'],
                legacy_pass=report['passed'], loss=report['loss'], seconds=report['diagnostic_seconds'])), flush=True)
            del grads, accumulated
            head.zero_grad(set_to_none=True); gc.collect()
    rows = []
    for n,g in ref.items():
        noise = 0. if g is None else float((hi[n]-lo[n]).abs().max())
        rows.append(dict(name=n, max_pairwise_noise=noise, proposed_noise_threshold=2*noise+1e-6,
            reference_comparison_distribution=[next(r['max_abs'] for r in rep['tensors'] if r['name']==n)
                for rep in reports] if g is not None else [],
            legacy_failure_count=sum(not next(r['passed'] for r in rep['tensors'] if r['name']==n) for rep in reports)))
    write(out/'CALIBRATION.json', dict(mode=args.mode, repeats=args.repeats, tensors=rows,
        all_repeats_bitwise=all(r['bitwise'] for r in reports),
        losses=[r['loss'] for r in reports], optimizer_steps=0,
        global_thresholds=dict(cosine_min=.9999, relative_l2_max=.01, norm_relative_max=.002),
        rule='2 * maximum pairwise baseline noise + 1e-6; independent of old gates'))
    after = {p: training_resume.file_hash(p) for p in hashes}
    write(out/'inputs_after.json', dict(hashes=after, unchanged=after==hashes))
    if after != hashes:
        raise RuntimeError('input changed')
    write(out/'RUN_STATUS.json', dict(status='CALIBRATED', optimizer_steps=0, repeats=args.repeats))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=['default','deterministic','math','deterministic_math','fp32_accumulator'], required=True)
    ap.add_argument('--repeats', type=int, default=10)
    args=ap.parse_args()
    if args.repeats < 2: raise ValueError('at least two repeats required')
    training_resume.validate_output(args.out); args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/'RUN_STATUS.json', dict(status='RUNNING',optimizer_steps=0))
    try: run(args,args.out)
    except BaseException as e:
        write(args.out/'RUN_STATUS.json', dict(status='ERROR', error=repr(e), optimizer_steps=0))
        raise

if __name__ == '__main__': main()
