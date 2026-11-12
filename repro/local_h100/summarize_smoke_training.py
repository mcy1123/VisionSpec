#!/usr/bin/env python3
"""Read-only CPU audit of finite SFT smoke checkpoints, not efficacy evidence."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

import torch
from safetensors import safe_open
from local_h100.train_video_sft import safe_output, write_json_exclusive
from shared import common, training_resume


def compare_parameters(base, saved, master=None, chunk=1048576):
    """Exact element counts and float64 reductions, with bounded temporary RAM."""
    if base.shape != saved.shape or (master is not None and master.shape != base.shape):
        raise ValueError('parameter shape mismatch')
    base, saved = base.detach().reshape(-1), saved.detach().reshape(-1)
    master = master.detach().reshape(-1) if master is not None else None
    stat = dict(elements=base.numel(), changed=0, base_squared_l2=0., saved_delta_squared_l2=0.,
                max_abs_saved_delta=0., master_changed=0, master_delta_squared_l2=0.,
                master_changed_but_saved_unchanged=0, master_to_saved_mismatches=0)
    for offset in range(0, base.numel(), chunk):
        b, s = base[offset:offset + chunk], saved[offset:offset + chunk]
        changed = b != s
        stat['changed'] += int(changed.sum())
        difference = s.double() - b.double()
        stat['base_squared_l2'] += float(b.double().square().sum())
        stat['saved_delta_squared_l2'] += float(difference.square().sum())
        stat['max_abs_saved_delta'] = max(stat['max_abs_saved_delta'], float(difference.abs().max()))
        if master is not None:
            m = master[offset:offset + chunk]
            if m.dtype != torch.float32:
                raise ValueError('master parameter must be fp32')
            master_changed = m != b.float()
            stat['master_changed'] += int(master_changed.sum())
            stat['master_delta_squared_l2'] += float((m.double() - b.double()).square().sum())
            stat['master_changed_but_saved_unchanged'] += int((master_changed & ~changed).sum())
            stat['master_to_saved_mismatches'] += int((m.to(s.dtype) != s).sum())
    return stat


def finish_stats(stat):
    out = dict(stat)
    n, b2 = stat['elements'], stat['base_squared_l2']
    out['changed_fraction'] = stat['changed'] / max(1, n)
    out['saved_delta_l2'] = math.sqrt(stat['saved_delta_squared_l2'])
    out['relative_l2'] = math.sqrt(stat['saved_delta_squared_l2'] / b2) if b2 else None
    out['master_changed_fraction'] = stat['master_changed'] / max(1, n)
    out['master_relative_l2'] = math.sqrt(stat['master_delta_squared_l2'] / b2) if b2 else None
    out['master_changed_but_saved_unchanged_fraction'] = stat['master_changed_but_saved_unchanged'] / max(1, n)
    out['saved_to_master_delta_norm_ratio'] = (
        math.sqrt(stat['saved_delta_squared_l2'] / stat['master_delta_squared_l2'])
        if stat['master_delta_squared_l2'] else None)
    return out


def add_stats(total, stat):
    for key, value in stat.items():
        total[key] = max(total.get(key, 0), value) if key == 'max_abs_saved_delta' else total.get(key, 0) + value


def semantic_update(h, name, tensor):
    h.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(',', ':')).encode())
    flat = tensor.detach().reshape(-1)
    for offset in range(0, flat.numel(), 1048576):
        h.update(flat[offset:offset + 1048576].contiguous().view(torch.uint8).numpy().tobytes())


def audit_run(run, public_weights, public_sha):
    protocol = json.loads((run / 'PROTOCOL.json').read_text())
    status = json.loads((run / 'RUN_STATUS.json').read_text())
    logs = [json.loads(line) for line in (run / 'log.jsonl').read_text().splitlines() if line.strip()]
    if status['status'] != 'COMPLETED' or status['completed_updates'] != 2 or len(logs) != 2:
        raise ValueError(f'{run}: expected a completed two-update smoke')
    if [row['update'] for row in logs] != [1, 2]:
        raise ValueError('training log update sequence differs')
    if any(row['protocol_sha256'] != protocol['protocol_sha256'] for row in logs):
        raise ValueError('log/protocol identity mismatch')
    if protocol['hashes']['initial_head_weights'] != {public_weights.name: public_sha}:
        raise ValueError('smoke did not initialize from the selected public checkpoint')
    for row in logs:
        for name in ('loss', 'gradient_norm', 'written_update_norm'):
            if not math.isfinite(row[name]) or (name != 'loss' and row[name] <= 0):
                raise ValueError(f'{run}: nonfinite or missing update at {name}')
    ckpt = Path(status['last_checkpoint']).resolve()
    if ckpt != (run / 'step2').resolve():
        raise ValueError('unexpected checkpoint path')
    manifest = json.loads((ckpt / 'SHA256.json').read_text())
    verified = {}
    for name in ('config.json', 'model.safetensors', 'CHECKPOINT_IDENTITY.json'):
        actual = training_resume.file_hash(ckpt / name)
        if manifest.get(name) != actual:
            raise ValueError(f'{ckpt}: SHA mismatch for {name}')
        verified[name] = actual
    identity = json.loads((ckpt / 'CHECKPOINT_IDENTITY.json').read_text())
    if identity['protocol_sha256'] != protocol['protocol_sha256'] or identity['optimizer_step'] != 2:
        raise ValueError('checkpoint sidecar/protocol mismatch')
    # mmap avoids materializing all Adam moments. These are our own local
    # checkpoints; do not use unrestricted pickle loading on external files.
    state = torch.load(ckpt / 'training_state.pt', map_location='cpu', mmap=True, weights_only=False)
    if state['update'] != 2 or state['protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('training-state identity mismatch')
    master = state['master_optimizer']['master']
    head = common.load_head(ckpt, dtype=torch.bfloat16, device='cpu')
    loaded = dict(head.named_parameters())
    totals, by_tensor = {}, {}
    semantic = hashlib.sha256()
    with safe_open(public_weights, framework='pt', device='cpu') as baseline, \
            safe_open(ckpt / 'model.safetensors', framework='pt', device='cpu') as saved:
        if set(loaded) != set(baseline.keys()) or set(loaded) != set(saved.keys()) or set(loaded) != set(master):
            raise ValueError('parameter keys differ across public/saved/reloaded/master')
        for name in sorted(loaded):
            value = saved.get_tensor(name)
            if not torch.equal(loaded[name].detach(), value):
                raise ValueError(f'CPU bf16 reload changed tensor {name}')
            if not torch.equal(state['head'][name], value):
                raise ValueError(f'training_state head differs from safetensors at {name}')
            semantic_update(semantic, name, value)
            stats = compare_parameters(baseline.get_tensor(name), value, master[name])
            add_stats(totals, stats)
            by_tensor[name] = finish_stats(stats)
    if totals['master_to_saved_mismatches']:
        raise ValueError('fp32 master rounded to bf16 differs from saved head')
    total = finish_stats(totals)
    del loaded, head, state, master
    gc.collect()
    eval_path = run.parent.with_name(run.parent.name + '_dev_gate') / 'result.json'
    evaluation = None
    if eval_path.is_file():
        evaluated = json.loads(eval_path.read_text())
        artifacts = evaluated['protocol']['artifacts']
        if (artifacts['head_weights_sha256'] != {'model.safetensors': verified['model.safetensors']} or
                artifacts['checkpoint_identity_sha256'] != verified['CHECKPOINT_IDENTITY.json']):
            raise ValueError('evaluation did not load this saved checkpoint identity')
        erows = evaluated['rows']
        evaluation = dict(path=str(eval_path), n=len(erows),
                          all_lossless=all(r['lossless'] and r['spec_ids'] == r['ar_ids'] for r in erows),
                          all_truncated=all(r['truncated'] for r in erows),
                          mean_tau=sum(r['tau'] for r in erows) / len(erows),
                          rows=[{k: r[k] for k in ('video', 'tau', 'rounds', 'gen_tokens', 'lossless')} for r in erows],
                          head_identity_matches_saved=True, scientific_efficacy_claim=False)
    return dict(name=run.parent.name, run=str(run), config=protocol['config'],
                protocol_sha256=protocol['protocol_sha256'], status=status,
                manifest_verified=verified,
                training_state_sha256_rehashed=False,
                training_state_audit='mmap-read: identity/head exact equality and fp32-master-to-bf16 exact equality',
                cpu_reload_all_parameters_exact=True, parameter_content_sha256=semantic.hexdigest(),
                full_parameter_drift=total, by_tensor=by_tensor,
                training_updates=[{k: r[k] for k in ('update', 'loss', 'gradient_norm', 'written_update_norm', 'seconds')} for r in logs],
                dev_gate=evaluation)


def comparisons(results):
    pairs = []
    for i, left in enumerate(results):
        for right in results[i + 1:]:
            lc, rc = left['config'], right['config']
            same_pool = lc['pool'] == rc['pool']
            if same_pool and ((lc['seed'] == rc['seed']) != (lc['lr'] == rc['lr'])):
                pairs.append(dict(left=left['name'], right=right['name'],
                                  differing_factor='lr' if lc['seed'] == rc['seed'] else 'seed',
                                  identical_parameter_contents=left['parameter_content_sha256'] == right['parameter_content_sha256'],
                                  identical_model_file=left['manifest_verified']['model.safetensors'] == right['manifest_verified']['model.safetensors']))
    return pairs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--public-head', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cpu-threads', type=int, default=2)
    a = p.parse_args(argv)
    if a.cpu_threads < 1:
        raise ValueError('cpu-threads must be positive')
    torch.set_num_threads(a.cpu_threads)
    out = safe_output(a.out)
    if out.exists():
        raise FileExistsError(out)
    public = Path(a.public_head) / 'model.safetensors'
    public_sha = training_resume.file_hash(public)
    runs = sorted(Path(a.run_root).glob('jobs/sft_smoke_*f_lr*_seed*/run'))
    if len(runs) != 8:
        raise ValueError(f'expected eight smoke runs, found {len(runs)}')
    results = []
    start = time.perf_counter()
    for run in runs:
        print(f'[audit] {run.parent.name}', flush=True)
        result = audit_run(run, public, public_sha)
        results.append(result)
        stats = result['full_parameter_drift']
        print(json.dumps(dict(name=result['name'], relative_l2=stats['relative_l2'],
                              changed_fraction=stats['changed_fraction'],
                              master_relative_l2=stats['master_relative_l2'],
                              master_changed_but_saved_unchanged_fraction=stats['master_changed_but_saved_unchanged_fraction'])), flush=True)
    summary = dict(schema=1, purpose='engineering_smoke_only_no_n2_efficacy_conclusions',
                   public_head=str(public.parent), public_weights_sha256=public_sha,
                   source_sha256=training_resume.file_hash(__file__), elapsed_seconds=time.perf_counter() - start,
                   runs=results, pair_comparisons=comparisons(results),
                   all_eight_parameter_contents_distinct=len({r['parameter_content_sha256'] for r in results}) == 8,
                   limitations=['Two optimizer updates only; different QA batches make step1-vs-step2 loss non-comparable.',
                                'Two dev videos only; identical tau does not imply identical weights or predictions everywhere.',
                                'No efficacy or best-lr selection from this smoke.',
                                'Training-state file is mmap-inspected; its large-file SHA was not rehashed in this analysis.'])
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(out, summary)
    print(str(out), flush=True)
    return summary


if __name__ == '__main__':
    main()
