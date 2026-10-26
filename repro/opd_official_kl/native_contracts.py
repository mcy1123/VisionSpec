"""Native trace metadata and unchanged slot denominator calculation."""
import math
import torch
from . import runtime


def denominator_record(trace, slot, task):
    offsets = [list(r['rejected_offsets']) for r in trace['rounds']]
    record = dict(slot=slot, task=task,
                  response_count=trace['ids'].numel() - trace['prompt_len'],
                  ordered_rejected_offsets=offsets,
                  denominator_hex=runtime.denominator(trace).hex())
    return record


def global_denominator(records, expected_tasks):
    if len(records) != len(expected_tasks):
        raise ValueError('missing/extra slot')
    by_slot = {}
    for r in records:
        s = r['slot']
        if type(s) is not int or s in by_slot or not 0 <= s < len(expected_tasks):
            raise ValueError('duplicate/invalid slot')
        if r['task'] != expected_tasks[s]:
            raise ValueError('task identity mismatch')
        if type(r['response_count']) is not int or r['response_count'] < 0:
            raise ValueError('invalid response count')
        offsets = [o for row in r['ordered_rejected_offsets'] for o in row]
        if any(type(o) is not int or o < 1 or o > 15 for o in offsets):
            raise ValueError('invalid rejected offset')
        value = float(r['response_count']) + 2 * sum(0.9 ** (o - 1) for o in offsets)
        if not math.isfinite(value) or value.hex() != r['denominator_hex']:
            raise ValueError('denominator bit mismatch')
        by_slot[s] = value
    total = sum(by_slot[s] for s in range(len(expected_tasks)))
    if not math.isfinite(total) or total <= 0:
        raise ValueError('invalid global denominator')
    return total


def bind_trace(trace, identity):
    required = {'checkpoint_sha256', 'task', 'slot', 'prompt_sha256', 'config_sha256'}
    if set(identity) != required or not all(identity[k] for k in required - {'slot'}):
        raise ValueError('incomplete trace identity')
    trace['efficiency_identity'] = dict(identity)
    validate_trace(trace, identity)
    return trace


def validate_trace(trace, identity):
    if trace.get('efficiency_identity') != identity:
        raise ValueError('trace identity mismatch')
    for r in trace['rounds']:
        p = r['teacher_logp']
        if p.dtype != torch.float32 or p.requires_grad or p.shape != (len(r['prop']),):
            raise ValueError('teacher logp dtype/shape/autograd mismatch')
        if len(r['prop']) != 15 or len(r['teacher_ids']) != 16:
            raise ValueError('incomplete proposal/teacher rows')
        if not bool(torch.isfinite(p).all()):
            raise ValueError('nonfinite teacher logp')
        if any(not 1 <= o <= len(r['prop']) for o in r['rejected_offsets']):
            raise ValueError('invalid rejected offset')
