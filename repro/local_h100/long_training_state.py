"""New, strict single-device training-state contract; never imports pilot state.

Only independent long-run entries use this module. A complete state makes a
restart test possible; it does not itself establish CUDA bitwise equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import uuid

import numpy as np
import pandas as pd
import torch

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()
from shared import common, training_resume
from shared.evidence import evidence_settings
from shared.model_contract import validate_model_contract
from local_h100 import evidence_sft
from local_h100.generate_teacher_cache import file_identity
from video_caption.train_video_9k import load_schedule, validate_rows

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
SCHEMA = 'local_video_long_state_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def safe_path(path):
    value, root = Path(path).resolve(), ROOT.resolve(strict=True)
    if value == root or not value.is_relative_to(root):
        raise ValueError('long-run output/state must be beneath this authorized run root')
    return value


def write_json(path, value):
    path = safe_path(path)
    temporary = path.with_name('.' + path.name + '-' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def argument_parser(description, objective):
    if objective not in ('sft', 'opd'):
        raise ValueError('objective must be sft or opd')
    p = argparse.ArgumentParser(description=description)
    for name in ('pool', 'schedule', 'target', 'out'):
        p.add_argument('--' + name, required=True)
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument('--init-from')
    start.add_argument('--resume-from')
    p.add_argument('--total-updates', type=int, required=True)
    p.add_argument('--run-updates', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--qa-per-update', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--max-seq-len', type=int, default=16384)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--time-mode', choices=('real',), default='real')
    p.add_argument('--master-weights', choices=('gpu',), default='gpu')
    p.add_argument('--evidence-mode', choices=('auto', 'plain', 'text', 'visual'), default='auto')
    p.add_argument('--deadline-utc')
    p.add_argument('--stop-file')
    p.add_argument('--dry-run', action='store_true')
    if objective == 'sft':
        p.add_argument('--anchors-per-qa', type=int, default=32)
        p.add_argument('--teacher-cache')
        p.add_argument('--device-map', choices=('single',), default='single')
    else:
        p.add_argument('--device-map', choices=('balanced',), default='balanced')
        p.add_argument('--loss-normalization', choices=('global_pooled',), default='global_pooled')
        p.add_argument('--e2e-tv-weight', type=float, choices=(0.,), default=0.)
        p.add_argument('--max-pixels', type=int, default=768)
    return p


def _weights(path):
    files = sorted(Path(path).glob('*.safetensors')) + sorted(Path(path).glob('pytorch_model*.bin'))
    if not files:
        raise ValueError('model weight files are absent')
    return files


def target_hashes(path):
    """Read-only reuse of locally verified full hashes, never trust a path alone."""
    files = _weights(path)
    before = {p.name: file_identity(p) for p in files}
    cache = ROOT / 'teacher_cache/_target_identity' / (digest(before) + '.json')
    if cache.is_file():
        record = json.loads(cache.read_text())
        hashes = record.get('weight_sha256', {})
        if record.get('schema') != 1 or record.get('file_identity') != before or set(hashes) != set(before):
            raise ValueError('invalid locally verified Target hash cache')
        if any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in hashes.values()):
            raise ValueError('invalid Target SHA')
    else:
        hashes = {p.name: training_resume.file_hash(p) for p in files}
    if before != {p.name: file_identity(p) for p in files}:
        raise RuntimeError('Target changed during identity check')
    return hashes


def _protocol_hash(protocol):
    return digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'})


def cudnn_version_without_device_probe():
    # torch.backends.cudnn.version() in this installed PyTorch also probes
    # every visible GPU capability, which initializes CUDA during dry-run.
    # This library-version getter reads the same value without that probe.
    binding = getattr(torch.backends.cudnn, '_cudnn', None)
    return None if binding is None else binding.getVersionInt()


def _rng_valid(rng, cuda_count):
    if not isinstance(rng, dict) or set(rng) != {'torch', 'numpy', 'python', 'torch_cuda'}:
        raise ValueError('complete Python/NumPy/torch/CUDA RNG is required')
    states = [rng['torch'], *rng['torch_cuda']]
    if len(rng['torch_cuda']) != cuda_count or any(
            not torch.is_tensor(t) or t.device.type != 'cpu' or t.dtype != torch.uint8 or t.ndim != 1
            for t in states):
        raise ValueError('RNG device/topology/state layout mismatch')
    # Validate without changing this process's generators.
    torch.Generator(device='cpu').set_state(rng['torch'])
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])


def read_checkpoint(path, *, allow_cpu_test=False):
    """Verify committed files before loading trusted local pickle state on CPU."""
    path = safe_path(path)
    manifest = json.loads((path / 'SHA256.json').read_text())
    required = {'config.json', 'training_state.pt', 'CHECKPOINT_IDENTITY.json', 'TRAINING_PROTOCOL.json'}
    if not isinstance(manifest, dict) or not required <= set(manifest) or not any(
            name.endswith('.safetensors') for name in manifest):
        raise ValueError('checkpoint lacks complete long-run files; pilot states cannot resume')
    for name, expected in manifest.items():
        if Path(name).name != name or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('invalid checkpoint checksum manifest')
        file = path / name
        if file.is_symlink() or not file.is_file() or training_resume.file_hash(file) != expected:
            raise ValueError('checkpoint file checksum mismatch: ' + name)
    identity = json.loads((path / 'CHECKPOINT_IDENTITY.json').read_text())
    protocol = json.loads((path / 'TRAINING_PROTOCOL.json').read_text())
    if identity.get('schema') != SCHEMA or identity.get('resume_supported') is not True:
        raise ValueError('only the new complete long-run resume schema is accepted')
    if protocol.get('protocol_sha256') != _protocol_hash(protocol):
        raise ValueError('checkpoint protocol hash mismatch')
    state = torch.load(path / 'training_state.pt', map_location='cpu', mmap=True, weights_only=False)
    if (state.get('long_state_schema') != SCHEMA or state.get('resume_schema') != 1
            or state.get('resume_identity') != protocol or state.get('training_protocol') != protocol):
        raise ValueError('long-run state schema/protocol mismatch')
    step = state.get('update_step')
    if isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= protocol['config']['total_updates']:
        raise ValueError('invalid completed update')
    if any(state.get(k) != step for k in ('update', 'data_cursor', 'optimizer_step', 'next_schedule_update')):
        raise ValueError('checkpoint absolute cursor/update mismatch')
    if identity.get('optimizer_step') != step or identity.get('protocol_sha256') != protocol['protocol_sha256']:
        raise ValueError('checkpoint sidecar step/protocol mismatch')
    if (identity.get('data_cursor') != step
            or identity.get('qa_exposures') != step * protocol['config']['qa_per_update']
            or identity.get('parameter_layout_sha256') != digest(state.get('parameter_layout'))):
        raise ValueError('checkpoint sidecar parameter layout/cursor mismatch')
    execution = state.get('execution', {})
    if execution.get('device_type') != 'cuda' and not allow_cpu_test:
        raise ValueError('CPU fixture checkpoints cannot enter production resume')
    _rng_valid(state.get('rng'), 1 if execution.get('device_type') == 'cuda' else 0)
    if state.get('phase') != 'after_complete_optimizer_update' or state.get('pending_request') is not False:
        raise ValueError('checkpoint is not at an empty complete-update boundary')
    return state, protocol, digest(manifest)


def prepare(args, objective, source_paths):
    """No output writes, no model loads, no CUDA allocation; complete preflight."""
    for name in ('total_updates', 'run_updates', 'qa_per_update', 'max_new_tokens', 'max_seq_len', 'save_every'):
        if getattr(args, name) <= 0:
            raise ValueError(name + ' must be positive')
    if args.max_seq_len <= args.max_new_tokens or not all(math.isfinite(v) and v > 0 for v in (args.lr, args.clip)):
        raise ValueError('invalid sequence or optimizer budget')
    if objective == 'sft' and args.anchors_per_qa < 1:
        raise ValueError('anchors_per_qa must be positive')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not re.fullmatch(r'GPU-[A-Za-z0-9-]+', visible[0]):
        raise ValueError('exactly one explicit CUDA GPU UUID is required')
    out = safe_path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('segment output must be new or empty')
    state, previous, parent_hash = (read_checkpoint(args.resume_from) if args.resume_from else (None, None, None))
    init = Path(args.resume_from or args.init_from)
    rows = pd.read_parquet(args.pool).set_index('task', verify_integrity=True)
    if 'split' not in rows or not rows['split'].eq('train').all():
        raise ValueError('long training requires explicitly train-only rows')
    schedule = load_schedule(args.schedule, rows, args.total_updates, args.qa_per_update)
    inputs = validate_rows(rows, schedule, args.time_mode)
    dimensions = validate_model_contract(args.target, init)
    settings = evidence_settings(init)
    if settings['mode'] != 'none':
        settings.update(evidence_sft.minimal_config(init))
    else:
        from local_h100.train_video_sft import validate_head
        validate_head(init)
    requested = 'none' if args.evidence_mode == 'plain' else args.evidence_mode
    if requested != 'auto' and requested != settings['mode']:
        raise ValueError('evidence mode differs from the actual head')
    operational = {'out', 'run_updates', 'init_from', 'resume_from', 'dry_run', 'deadline_utc', 'stop_file',
                   'save_every', 'teacher_cache', 'evidence_mode'}
    config = {k: v for k, v in vars(args).items() if k not in operational}
    for key in ('pool', 'schedule', 'target'):
        config.pop(key, None)  # Exact content, not path aliases, is the identity.
    target = Path(args.target)
    source_paths = list(source_paths) + [__file__, REPRO / 'shared/common.py', REPRO / 'shared/master_optim.py',
        REPRO / 'local_h100/train_video_opd.py', REPRO / 'local_h100/train_video_sft.py',
        REPRO / 'local_h100/evidence_sft.py',
        REPRO / 'local_h100/evidence_opd_prompt_boundary.py',
        REPRO / 'shared/model_contract.py', REPRO / 'local_h100/generate_teacher_cache.py',
        REPRO / 'shared/training_resume.py', REPRO / 'shared/evidence.py', REPRO / 'qa_opd_fair/videoencode.py',
        REPRO / 'shared/sft_blocks.py', REPRO / 'shared/draft_geometry.py',
        REPRO / 'opd_official_kl/runtime.py', REPRO / 'opd_official_kl/loss.py', REPRO / 'opd_official_kl/plan.py',
        REPRO / 'opd_official_kl/adapters.py', REPRO / 'video_caption/train_video_9k.py',
        REPRO.parent / 'GLANCE_repo/glance/decode.py', REPRO.parent / 'GLANCE_repo/glance/vl.py',
        REPRO.parent / 'GLANCE_repo/glance/head.py', REPRO.parent / 'GLANCE_repo/glance/tree.py',
        REPRO.parent / 'scripts/dflash_pkg/dflash-0.1.0/dflash/model.py',
        REPRO.parent / 'scripts/dflash_pkg/dflash-0.1.0/dflash/evidence.py']
    sources = {str(Path(p).resolve().relative_to(REPRO.parent)): training_resume.file_hash(p)
               for p in source_paths}
    initial = previous['initialization'] if previous else dict(
        kind='weights_only_new_experiment', config_sha256=training_resume.file_hash(init / 'config.json'),
        weights_sha256={p.name: training_resume.file_hash(p) for p in _weights(init)})
    import transformers
    protocol = dict(schema=SCHEMA, objective=objective, resume_supported=True,
        resume_validation='implementation candidate; CPU/CUDA equivalence must be separately validated',
        config=config, initialization=initial, evidence=settings, model_dimensions=dimensions,
        topology=dict(world_size=1, slot_map={'0': list(range(args.qa_per_update))}, cuda_rng_states=1),
        precision=dict(target='float32', head='bfloat16', evidence='float32', master='float32'),
        optimizer=dict(where='gpu', lr=args.lr, betas=list(common.BETAS), eps=1e-8,
                       weight_decay=common.WEIGHT_DECAY, clip=args.clip),
        environment=dict(torch=torch.__version__, transformers=transformers.__version__,
            cuda=torch.version.cuda, cudnn=cudnn_version_without_device_probe(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        inputs=inputs, hashes=dict(pool=training_resume.file_hash(args.pool),
            schedule=training_resume.file_hash(args.schedule), target_config=training_resume.file_hash(target / 'config.json'),
            target_weights=target_hashes(target), decoder=training_resume.file_hash(REPRO.parent / 'GLANCE_repo/glance/decode.py'),
            processor={p.name: training_resume.file_hash(p) for p in sorted(target.glob('*.json'))
                       if p.name not in ('download_manifest.json', 'DONE.json') and 'model.safetensors' not in p.name},
            sources=sources))
    protocol['protocol_sha256'] = _protocol_hash(protocol)
    if previous is not None and previous != protocol:
        raise ValueError('resume experiment/data/schedule/Target/config/source/environment identity mismatch')
    start = state['update_step'] if state else 0
    if start >= args.total_updates or args.run_updates > args.total_updates - start:
        raise ValueError('finite segment budget exceeds remaining immutable experiment budget')
    if state and state['execution'].get('cuda_uuids') != visible:
        raise ValueError('initial resume support requires the same physical GPU UUID')
    return dict(rows=rows, schedule=schedule, protocol=protocol, init_path=str(init), resume_state=state,
                start_step=start, end_step=start + args.run_updates, out=out, gpus=visible,
                parent_checkpoint=args.resume_from, parent_manifest_sha256=parent_hash, objective=objective)


def start_segment(prepared, args):
    out = prepared['out']
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'PROTOCOL.json', prepared['protocol'])
    write_json(out / 'SEGMENT.json', dict(schema=SCHEMA, start_step=prepared['start_step'],
        end_step=prepared['end_step'], parent_checkpoint=prepared['parent_checkpoint'],
        parent_manifest_sha256=prepared['parent_manifest_sha256'], cuda_uuids=prepared['gpus'],
        deadline_utc=args.deadline_utc, stop_file=args.stop_file, save_every=args.save_every,
        protocol_sha256=prepared['protocol']['protocol_sha256']))


def parameter_layout(head):
    return [dict(name=n, shape=list(p.shape), dtype=str(p.dtype), requires_grad=p.requires_grad)
            for n, p in head.named_parameters()]


def validate_runtime(head, optimizer, *, require_cuda=True):
    expected = 'cuda' if require_cuda else 'cpu'
    if optimizer.where != ('gpu' if require_cuda else 'cpu'):
        raise ValueError('optimizer placement differs from execution contract')
    if {p.device.type for p in head.parameters()} != {expected} or {p.device.type for p in optimizer.master.values()} != {expected}:
        raise ValueError('head/master must be on the single execution device')
    if len({str(p.device) for p in head.parameters()}) != 1 or len({str(p.device) for p in optimizer.master.values()}) != 1:
        raise ValueError('split head/master placement unsupported')
    if {p.device for p in head.parameters()} != {p.device for p in optimizer.master.values()}:
        raise ValueError('head and master execution devices differ')
    if require_cuda and any(p.dtype != (torch.float32 if name.startswith('evidence_readout.') else torch.bfloat16)
                            for name, p in head.named_parameters()):
        raise ValueError('production requires bf16 backbone and fp32 evidence readout')
    if any(v.dtype != torch.float32 for v in optimizer.master.values()):
        raise ValueError('master parameters must be fp32')
    if getattr(head, 'evidence_readout', None) is not None:
        evidence_sft.assert_time_off(head)
        if head.evidence_readout.request is not None:
            raise ValueError('cannot save/restore an active evidence request')


def _prevalidate_optimizer(head, optimizer, state):
    blob = state['master_optimizer']
    names = [n for n, _ in head.named_parameters()]
    if state.get('parameter_layout') != parameter_layout(head) or list(blob['master']) != names or optimizer._names != names:
        raise ValueError('ordered parameter layout/freeze policy mismatch')
    if blob.get('where') != optimizer.where or blob.get('clip') != optimizer.clip:
        raise ValueError('master placement/clip mismatch before mutation')
    before, incoming = optimizer._opt.state_dict(), blob['optimizer']
    if len(before['param_groups']) != len(incoming['param_groups']):
        raise ValueError('optimizer group count mismatch')
    for current, saved in zip(before['param_groups'], incoming['param_groups']):
        if current != saved:
            raise ValueError('optimizer hyperparameter/order mismatch before mutation')
    params = dict(head.named_parameters())
    if set(state['head']) != set(head.state_dict()):
        raise ValueError('head state keys mismatch')
    for name, actual in head.state_dict().items():
        value = state['head'][name]
        if value.shape != actual.shape or value.dtype != actual.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError('head tensor layout/nonfinite mismatch')
    for name, value in blob['master'].items():
        p = params[name]
        if value.shape != p.shape or value.dtype != torch.float32 or value.requires_grad or not bool(torch.isfinite(value).all()):
            raise ValueError('master tensor layout/nonfinite mismatch')
        if not torch.equal(value.to(dtype=p.dtype), state['head'][name]):
            raise ValueError('saved head differs from dtype-cast fp32 master')
    ids = [i for g in incoming['param_groups'] for i in g['params']]
    if len(ids) != len(names) or len(set(ids)) != len(ids) or not set(incoming['state']) <= set(ids):
        raise ValueError('optimizer state parameter mapping mismatch')
    for index, item in incoming['state'].items():
        name = names[ids.index(index)]
        if not params[name].requires_grad:
            raise ValueError('frozen parameter unexpectedly has Adam state')
        if set(item) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('unsupported Adam state layout')
        step = item['step']
        if not torch.is_tensor(step) or step.numel() != 1 or not 0 < float(step) <= state['update_step'] or float(step) != int(float(step)):
            raise ValueError('Adam step inconsistent with completed updates')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = item[key]
            if value.shape != params[name].shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError('Adam moment tensor invalid')
        if bool((item['exp_avg_sq'] < 0).any()):
            raise ValueError('Adam second moment is negative')


def _exact_tree_check(actual, expected, path, counts):
    """Read back one tensor at a time, with zero numerical tolerance."""
    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not torch.equal(actual.detach().cpu(), expected.detach().cpu())):
            raise ValueError('restored tensor is not exact: ' + path)
        counts['tensors'] += 1
        counts['elements'] += expected.numel()
        counts['bytes'] += expected.numel() * expected.element_size()
    elif isinstance(expected, np.ndarray):
        if (not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype
                or actual.shape != expected.shape or not np.array_equal(actual, expected)):
            raise ValueError('restored array is not exact: ' + path)
        counts['arrays'] += 1
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or list(actual) != list(expected):
            raise ValueError('restored mapping/order differs: ' + path)
        for key in expected:
            _exact_tree_check(actual[key], expected[key], path + '.' + str(key), counts)
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError('restored sequence differs: ' + path)
        for index, value in enumerate(expected):
            _exact_tree_check(actual[index], value, path + '.' + str(index), counts)
    else:
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('restored scalar differs: ' + path)
        counts['scalars'] += 1


def audit_restored_state(prepared, head, optimizer, saved):
    """Prove complete loading before another update; does not draw random values."""
    actual = dict(head=head.state_dict(), master_optimizer=dict(master=optimizer.master,
        optimizer=optimizer._opt.state_dict(), where=optimizer.where, clip=optimizer.clip),
        rng=training_resume.capture_rng())
    checks = {}
    for key, value in actual.items():
        counts = dict(tensors=0, elements=0, bytes=0, scalars=0, arrays=0)
        _exact_tree_check(value, saved[key], key, counts)
        checks[key] = counts
    result = dict(status='EXACT_RESTORE_VERIFIED', complete_state_exact=True,
        next_update_started=False, restored_step=saved['update_step'],
        protocol_sha256=prepared['protocol']['protocol_sha256'],
        parent_manifest_sha256=prepared['parent_manifest_sha256'], checks=checks)
    write_json(prepared['out'] / 'RESTORE_AUDIT.json', result)
    return result


def restore_if_needed(prepared, head, optimizer):
    cpu_test = prepared.get('_test_allow_cpu', False)
    validate_runtime(head, optimizer, require_cuda=not cpu_test)
    state = prepared['resume_state']
    if state is None:
        return 0
    if state.get('resume_identity') != prepared['protocol']:
        raise ValueError('resume identity differs from prepared experiment')
    _prevalidate_optimizer(head, optimizer, state)
    _rng_valid(state['rng'], 0 if cpu_test else 1)
    if not cpu_test:
        # Constructor/validation uses an independent generator, never the
        # global stream. Malformed CUDA state must fail before parameter copy.
        torch.Generator(device=optimizer._dev).set_state(state['rng']['torch_cuda'][0])
    # shared.restore restores RNG last, after head/master/Adam writes.
    step = training_resume.restore(state, head, optimizer, prepared['protocol'], prepared['start_step'])
    validate_runtime(head, optimizer, require_cuda=not cpu_test)
    for item in optimizer._opt.state.values():
        for key in ('exp_avg', 'exp_avg_sq'):
            if item[key].device != optimizer._dev:
                raise ValueError('restored Adam moment is not on the master device')
    prepared['restore_audit'] = audit_restored_state(prepared, head, optimizer, state)
    prepared['resume_state'] = None
    return step


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def save_checkpoint(prepared, step, head, optimizer):
    cpu_test = prepared.get('_test_allow_cpu', False)
    validate_runtime(head, optimizer, require_cuda=not cpu_test)
    if not prepared['start_step'] < step <= prepared['end_step']:
        raise ValueError('checkpoint step exceeds this finite segment')
    destination = safe_path(prepared['out'] / f'step{step}')
    if destination.exists():
        raise FileExistsError(destination)
    if not cpu_test:
        torch.cuda.synchronize()
    protocol = prepared['protocol']
    rng = training_resume.capture_rng()
    _rng_valid(rng, 0 if cpu_test else 1)
    state = dict(head=_cpu_tree(head.state_dict()), master_optimizer=_cpu_tree(optimizer.state_dict()),
        **training_resume.extras(step, protocol), long_state_schema=SCHEMA, update=step,
        optimizer_step=step, next_schedule_update=step, training_protocol=protocol,
        parameter_layout=parameter_layout(head), phase='after_complete_optimizer_update', pending_request=False,
        execution=dict(device_type='cpu' if cpu_test else 'cuda', cuda_uuids=prepared['gpus']),
        parent_manifest_sha256=prepared['parent_manifest_sha256'])
    state['rng'] = rng
    _prevalidate_optimizer(head, optimizer, state)
    metadata = dict(schema=SCHEMA, optimizer_step=step, data_cursor=step,
        qa_exposures=step * protocol['config']['qa_per_update'], protocol_sha256=protocol['protocol_sha256'],
        resume_supported=True, optimizer_state_saved=True,
        resume_validation='candidate; no CPU/CUDA bitwise equivalence claim',
        parent_manifest_sha256=prepared['parent_manifest_sha256'],
        parameter_layout_sha256=digest(state['parameter_layout']))
    if not cpu_test:
        metadata['cuda'] = dict(max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                               max_memory_reserved_bytes=torch.cuda.max_memory_reserved())
    temporary = destination.with_name('.' + destination.name + '.tmp-' + uuid.uuid4().hex)
    temporary.mkdir()
    # Independent publication keeps active shared save/protocol functions intact.
    head.save_pretrained(temporary)
    torch.save(state, temporary / 'training_state.pt')
    for name, value in [('TRAINING_PROTOCOL.json', protocol), ('CHECKPOINT_IDENTITY.json', metadata)]:
        write_json(temporary / name, value)
    files = sorted(p for p in temporary.iterdir() if p.is_file())
    for path in files:
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    write_json(temporary / 'SHA256.json', {p.name: training_resume.file_hash(p) for p in files})
    temporary_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(temporary_fd)
    finally:
        os.close(temporary_fd)
    if destination.exists():
        raise FileExistsError(destination)
    os.rename(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination


def finish_segment(prepared, completed, reason, last_checkpoint):
    value = dict(status='COMPLETED' if completed == prepared['protocol']['config']['total_updates'] else 'STOPPED',
        stop_reason=reason, completed_updates=completed, segment_start=prepared['start_step'],
        segment_end=prepared['end_step'], total_updates=prepared['protocol']['config']['total_updates'],
        last_checkpoint=str(last_checkpoint) if last_checkpoint else None,
        resume_supported=True, protocol_sha256=prepared['protocol']['protocol_sha256'])
    write_json(prepared['out'] / 'RUN_STATUS.json', value)
    return value
