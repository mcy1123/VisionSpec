#!/usr/bin/env python3
"""Read-only CPU audit of finite evidence engineering runs; no target load."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()
import torch
from safetensors import safe_open
from shared import common
from local_h100.evidence_sft import freeze_time_after_unfreeze, minimal_config

PREFIX = 'evidence_readout.'
DEFAULT_JOBS = [f'evidence_{engine}128_{arm}_smoke2'
                for engine in ('sft', 'opd') for arm in ('B_text', 'C_visual')]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            value.update(data)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def evidence_weights(path):
    with safe_open(str(Path(path) / 'model.safetensors'), framework='pt', device='cpu') as file:
        return {name: file.get_tensor(name) for name in file.keys() if name.startswith(PREFIX)}


def checkpoint_audit(path, initial):
    """Compare saved FP32 evidence to actual production CPU bf16-head load."""
    declared = read(path / 'SHA256.json')
    selected = ['model.safetensors', 'config.json', 'CHECKPOINT_IDENTITY.json']
    actual = {name: sha(path / name) for name in selected}
    if any(actual[name] != declared[name] for name in selected):
        raise ValueError(f'checkpoint checksum mismatch: {path}')
    saved, before = evidence_weights(path), evidence_weights(initial)
    if not saved or saved.keys() != before.keys():
        raise ValueError('checkpoint evidence tensor names differ from initialization')
    result = {'path': str(path), 'verified_file_sha256': actual,
              'training_state_not_loaded_or_rehashed': True,
              'architecture': minimal_config(path), 'tensors': {}}
    for name, value in saved.items():
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('evidence weights must be finite FP32')
        delta = value - before[name]
        result['tensors'][name[len(PREFIX):]] = dict(
            shape=list(value.shape), dtype=str(value.dtype),
            nonzero_count=int(torch.count_nonzero(value)),
            l2=float(value.norm()), initial_delta_l2=float(delta.norm()),
            changed_elements=int(torch.count_nonzero(delta)), sha256=tensor_sha(value))
    head = common.load_head(path, dtype=torch.bfloat16, device='cpu')
    loaded = dict(head.evidence_readout.named_parameters())
    exact = all(loaded[name[len(PREFIX):]].dtype == value.dtype and
                torch.equal(loaded[name[len(PREFIX):]].detach(), value)
                for name, value in saved.items())
    if not exact:
        raise ValueError('production CPU head loading changed FP32 evidence')
    head.requires_grad_(True)
    freeze_time_after_unfreeze(head)
    result['production_load'] = dict(
        device='cpu', head_requested_dtype='torch.bfloat16',
        evidence_dtype='torch.float32', all_evidence_tensors_exact=True,
        time_requires_grad_after_explicit_policy=head.evidence_readout.time.weight.requires_grad,
        time_gradient_is_none=head.evidence_readout.time.weight.grad is None,
        request_is_none_on_new_instance=head.evidence_readout.request is None,
        caveat='requires_grad and request bindings are not serialized; this verifies reload plus explicit freeze policy, not a saved runtime flag')
    del head
    gc.collect()
    return result


def cached_prompt_audit(root, key, target):
    path = root / 'teacher_cache' / (key + '.json')
    cache = read(path)
    ids = torch.tensor(cache['prompt_ids'], dtype=torch.int64)[None]
    recorded = cache['identity']['encoded_inputs']['input_ids']
    if tensor_sha(ids) != recorded['sha256'] or list(ids.shape) != recorded['shape']:
        raise ValueError('teacher prompt IDs do not match recorded actual input SHA')
    video_id = int(read(Path(target) / 'config.json')['video_token_id'])
    mask = ids[0].eq(video_id)
    locations = mask.nonzero().flatten()
    if not len(locations):
        raise ValueError('cached teacher prompt has no video tokens')
    return dict(teacher_cache_key=key, cache_file_sha256=sha(path),
                actual_input_sha256=recorded['sha256'], prompt_tokens=ids.numel(),
                visual_tokens=int(mask.sum()), visual_token_id=video_id,
                visual_mask_sha256=tensor_sha(mask), first_visual_position=int(locations[0]),
                last_visual_position=int(locations[-1]),
                all_visual_known_before_last_prompt_root=bool(locations[-1] < ids.numel() - 1),
                final_prompt_token_is_visual=bool(mask[-1]),
                observation='mask reconstructed from checksum-verified actual cached teacher prompt IDs; no live request object was serialized')


def log_audit(logs):
    result = []
    for log in logs:
        item = dict(update=log['update'], loss=log['loss'], gradient_norm=log['gradient_norm'])
        if 'qa' in log:
            item['qa'] = [{key: value for key, value in qa.items()
                           if key in ('task', 'prompt_tokens', 'teacher_cache_key', 'evidence')}
                          for qa in log['qa']]
            item['per_parameter_gradient_observation'] = 'NOT_RECORDED'
        else:
            item['tasks'] = log['tasks']
            item['input_tokens_per_qa'] = log['input_tokens_per_qa']
            item['visual_tokens_per_qa'] = log['visual_tokens_per_qa']
            item['evidence'] = log['evidence']
            item['per_parameter_gradient_observation'] = 'RECORDED_IN_SHARED_OPD_DIAGNOSTICS'
        result.append(item)
    return result


def source_checks(protocol, sft):
    paths = ({'trainer': REPRO / 'local_h100/train_video_sft.py',
              'evidence_sft': REPRO / 'local_h100/evidence_sft.py',
              'evidence_sft_entry': REPRO / 'local_h100/train_video_evidence_sft.py'} if sft else
             {'trainer': REPRO / 'video_caption/train_video_9k.py',
              'local_evidence_policy': REPRO / 'local_h100/evidence_sft.py',
              'local_evidence_entry': REPRO / 'local_h100/train_video_evidence_opd.py'})
    checks = {name: dict(path=str(path), recorded_sha256=protocol['hashes'][name],
                        actual_sha256=sha(path)) for name, path in paths.items()}
    for item in checks.values():
        item['matches_recorded_source'] = item['recorded_sha256'] == item['actual_sha256']
    return checks


def audit_job(root, name):
    job = (root / 'jobs' / name).resolve()
    if job.parent != (root / 'jobs').resolve():
        raise ValueError('job must be one direct child of this run/jobs')
    pro = read(job / 'run/PROTOCOL.json')
    exit_record = read(job / 'EXIT.json')
    logpath = job / 'run/log.jsonl'
    logs = [json.loads(line) for line in logpath.read_text().splitlines() if line.strip()] if logpath.exists() else []
    if any(line['protocol_sha256'] != pro['protocol_sha256'] for line in logs):
        raise ValueError('logged protocol identity mismatch')
    sft = 'sft' in name
    result = dict(job=name, exit=exit_record, protocol_sha256=pro['protocol_sha256'],
                  protocol_file_sha256=sha(job / 'run/PROTOCOL.json'),
                  initialization=pro['config']['init_from'],
                  architecture=pro['evidence_architecture'], source_checks=source_checks(pro, sft),
                  updates=log_audit(logs), optimizer=pro['optimizer'])
    if exit_record['exit_code'] != 0:
        stdout = (job / 'stdout.log').read_text()
        result.update(status='FAILED_ORIGINAL_RUN', failure_tail=stdout.splitlines()[-35:],
                      checkpoint=None, completed_logged_updates=len(logs),
                      conclusion='No successful update/checkpoint is evidenced; preserve this failed run independently of any corrected new run.')
        return result
    state = read(job / 'run/RUN_STATUS.json')
    if state['status'] != 'COMPLETED' or state['completed_updates'] != len(logs):
        raise ValueError('EXIT0 does not match a complete update log')
    path = Path(state['last_checkpoint']).resolve()
    if not path.is_relative_to(job):
        raise ValueError('checkpoint must be inside this job')
    result.update(status='COMPLETED_AUDITED', completed_logged_updates=len(logs),
                  checkpoint=checkpoint_audit(path, Path(pro['config']['init_from'])))
    result['runtime_evidence_limits'] = dict(
        target_freeze=('Verified source freezes target and asserts all target gradients None after every QA; successful completion is guard evidence.' if sft else
                       'Verified setup source freezes target; no per-update target-gradient telemetry was recorded.'),
        request_cleanup='Verified entry clears request in finally and refuses checkpoint with a live binding; successful checkpoint is guard evidence, not a serialized live trace.',
        time_freeze='Verified entry re-freezes time after blanket unfreeze; checkpoint guard requires zero FP32 time, requires_grad=False, grad=None.',
        gradients=('SFT has no per-parameter gradient log. Saved nonzero Q/K/V drift is observed, but is not a direct recorded gradient norm.' if sft else
                   'Shared OPD logs per-parameter gradient norms and missing gradients; time.weight is intentionally absent.'),
        not_tested='No target GPU rerun, training-state resume, final quality or formal speed claim.')
    if sft:
        result['cached_input_masks'] = [dict(task=qa['task'], **cached_prompt_audit(root, qa['teacher_cache_key'], pro['config']['target']))
                                        for log in logs for qa in log['qa']]
    else:
        result['actual_visual_mask_limit'] = 'OPD logs token counts, but does not serialize actual prompt IDs/mask in these logs. SFT teacher masks are not represented as direct OPD mask observations.'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--jobs', nargs='+', default=DEFAULT_JOBS)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error("CPU-only audit requires CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(2)
    root = args.run_root.resolve()
    out = (args.out or root / 'analysis/evidence_engineering_audit.json').resolve()
    if out.parent != root / 'analysis':
        parser.error('audit output must be a new file directly in this run/analysis')
    if out.exists():
        raise FileExistsError('refusing to overwrite an existing audit')
    jobs = [audit_job(root, name) for name in args.jobs]
    failed = [item['job'] for item in jobs if item['status'] != 'COMPLETED_AUDITED']
    result = dict(schema='local_evidence_engineering_audit_v1',
                  status='PARTIAL_WITH_FAILED_ORIGINAL_RUN' if failed else 'COMPLETED_ENGINEERING_AUDIT',
                  read_only=True, gpu_used=False, tool_sha256=sha(__file__),
                  scope='Finite engineering smoke only; no efficacy, final quality, or formal speed conclusion.',
                  failed_jobs=failed, jobs=jobs)
    out.parent.mkdir(exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(out=str(out), status=result['status'], failed_jobs=failed)))


if __name__ == '__main__':
    main()
