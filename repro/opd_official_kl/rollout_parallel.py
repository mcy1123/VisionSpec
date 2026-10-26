"""Native-only rollout transport, Head-version checks and bounded update ledger."""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .native_contracts import validate_trace

CAP = 64 * 1024 * 1024
TRACE_FIELDS = {'ids', 'prompt_len', 'hidden', 'fused', 'rounds', 'plan',
                'boundaries', 'final_cache_length', 'efficiency_identity', 'policy_identity',
                'teacher_verify_impl', 'training_identity'}


def tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def raw_digest(tensor):
    h = hashlib.sha256()
    for part in tensor.detach().reshape(-1).split(4 * 1024 * 1024):
        h.update(part.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), sha256=h.hexdigest())


def tree_digest(value):
    """Exact tensor byte signatures, including signed zero; no floating reductions."""
    import numpy as np
    if torch.is_tensor(value):
        return raw_digest(value)
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): tree_digest(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_digest(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def head_digest(head):
    rows = {name: raw_digest(p) for name, p in sorted(head.named_parameters())}
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def parameter_versions(head):
    return tuple((name, p._version) for name, p in head.named_parameters())


def check_versions(records):
    required = {'head_sha256', 'optimizer_step', 'schedule_cursor'}
    if (not records or any(set(r) != required for r in records)
        or any(type(r['optimizer_step']) is not int or r['optimizer_step'] < 0
               or type(r['schedule_cursor']) is not int or r['schedule_cursor'] < 0
               or not isinstance(r['head_sha256'], str) or len(r['head_sha256']) != 64
               for r in records)
        or any(r != records[0] for r in records)):
        raise ValueError('Head policy version/hash disagreement')
    return records[0]


def validate_envelope(trace, identity, policy, metadata):
    if (set(trace) != TRACE_FIELDS or trace.get('teacher_verify_impl') != 'pure_block_native'
        or trace.get('training_identity') != dict(metadata, head_version=policy)
        or metadata.get('opd_mode') != 'pure_block' or metadata.get('world_size') != 4
        or metadata.get('slot_map') != {str(i):[i] for i in range(4)}):
        raise ValueError('native-only trace/mode/topology/Head identity required')
    validate_trace(trace, identity)
    if trace['policy_identity'] != policy:
        raise ValueError('stale or cross-update Head trace')
    check_versions([policy])
    if identity['checkpoint_sha256'] != policy['head_sha256']:
        raise ValueError('trace checkpoint and policy Head disagree')
    for t in tensors(trace):
        if t.requires_grad or t.grad_fn is not None or t.device.type != 'cpu':
            raise ValueError('wire trace must contain detached CPU rollout tensors only')
    if trace['hidden'].dtype != torch.float32 or trace['fused'].dtype != torch.float32:
        raise ValueError('frozen fp32 Target state precision changed')
    if (trace['ids'].dtype != torch.long
        or trace['hidden'].shape[1] != trace['ids'].numel()
        or trace['fused'].shape[1] != trace['ids'].numel()
        or trace['final_cache_length'] != trace['ids'].numel()):
        raise ValueError('trace token/Target context alignment changed')


def pack_trace(trace):
    buffer = io.BytesIO()
    torch.save(trace, buffer)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def unpack_trace(payload, digest):
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('trace wire checksum mismatch')
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)


def send_trace(trace, destination=0):
    payload, digest = pack_trace(trace)
    dist.send_object_list([dict(bytes=len(payload), sha256=digest)], destination)
    view = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    for chunk in view.split(CAP):
        dist.send(chunk, destination)
    return len(payload)


def receive_trace(source):
    header = [None]
    dist.recv_object_list(header, source)
    n = header[0]['bytes']
    if type(n) is not int or not 0 < n <= 1024 ** 3:
        raise ValueError('invalid trace wire length')
    view = torch.empty(n, dtype=torch.uint8)
    for chunk in view.split(CAP):
        dist.recv(chunk, source)
    payload = view.numpy().tobytes()
    return unpack_trace(payload, header[0]['sha256']), n


def budget_event(path, case, event, index, expected=None):
    """Reserve before optimizer entry. A failed attempt is never refunded/retried."""
    path = Path(path)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text())
        spec = data['cases'][case]
        if expected is not None and any(spec.get(k) != v for k, v in expected.items()):
            raise ValueError('case does not match authorized update budget')
        key = dict(case=case, index=index)
        if not 0 <= index < spec['updates']:
            raise ValueError('case update allowance exceeded')
        if event == 'reserve':
            if len(data['attempts']) >= data['max_attempts'] or key in data['attempts']:
                raise ValueError('update budget exhausted or attempt already consumed')
            data['attempts'].append(key)
        elif event == 'complete':
            if key not in data['attempts'] or key in data['completed']:
                raise ValueError('invalid completion')
            data['completed'].append(key)
        else:
            raise ValueError('unknown budget event')
        tmp = path.with_suffix('.pending')
        with tmp.open('w') as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        return len(data['attempts']), len(data['completed'])


@torch.no_grad()
def broadcast_head(head, group):
    # Same chunked broadcast as the tested native rollout-only implementation.
    for _, p in sorted(head.named_parameters()):
        for chunk in p.detach().reshape(-1).split(CAP // p.element_size()):
            dist.broadcast(chunk, 0, group=group)
