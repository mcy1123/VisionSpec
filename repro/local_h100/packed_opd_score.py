"""Experimental fixed-trace OPD scoring with at most four anchors per forward.

Only the draft forward is packed. The original runtime.score still chooses
recorded response/rejected actions, frozen teacher log-probabilities, clamps,
Bernoulli/k3 terms, and uncovered-token accounting. A group sums its original
numerators before one backward through the shared graph; the caller retains
the original global QA denominator. No SFT labels or CE objective are used.

Packing changes floating-point reduction/GEMM shapes. Algebraic agreement and
CPU regression tolerances do not establish bitwise CPU/CUDA equivalence.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from local_h100 import evidence_sft as evidence
from local_h100.evidence_opd_prompt_boundary import known_visual_prefix
from opd_official_kl import runtime
from opd_official_kl.loss import Pool
from shared import sft_blocks, training_resume


@dataclass
class PackedPool(Pool):
    """One shared autograd graph; backward once on numerator/global_D."""
    anchors: tuple
    anchor_pools: tuple
    logits: torch.Tensor


def mode_of(head):
    config = head.config.to_dict()
    draft = config.get('dflash_config', {})
    if any(draft.get(name, config.get(name, False)) for name in evidence.OTHER_BRANCHES):
        raise ValueError('packed OPD supports only plain or minimal time-off evidence heads')
    mode = draft.get('evidence_mode', config.get('evidence_mode', 'none'))
    if mode not in ('none', 'text', 'visual'):
        raise ValueError('unsupported evidence mode')
    if mode != 'none':
        evidence.assert_time_off(head)
    return mode


def validate_trace(head, trace):
    ids, fused, hidden = trace['ids'], trace['fused'], trace['hidden']
    if ids.ndim != 1 or ids.dtype != torch.long or fused.ndim != 3 or hidden.ndim != 3:
        raise ValueError('invalid single-sequence trace tensor layout')
    if fused.shape[:2] != (1, len(ids)) or hidden.shape[:2] != (1, len(ids)):
        raise ValueError('trace features must align exactly with recorded token IDs')
    if fused.requires_grad or hidden.requires_grad:
        raise ValueError('frozen teacher trace features must be detached')
    prompt = int(trace['prompt_len'])
    if not 1 < prompt < len(ids):
        raise ValueError('trace must contain a prompt and generated response')
    plan = list(trace['plan'])
    if not plan or len({pos for pos, _ in plan}) != len(plan):
        raise ValueError('nonempty unique original anchor plan required')
    block = int(head.block_size)
    for pos, count in plan:
        if isinstance(pos, bool) or isinstance(count, bool) or not isinstance(pos, int) or not isinstance(count, int):
            raise ValueError('anchor positions/counts must be integer metadata')
        if not prompt - 1 <= pos < len(ids) or not 0 <= count < block or pos + count >= len(ids):
            raise ValueError('anchor precedes known prompt root or exceeds recorded response')
    if sum(count for _, count in plan) > len(ids) - prompt:
        raise ValueError('response supervision exceeds recorded response length')
    if len({row['pos'] for row in trace['rounds']}) != len(trace['rounds']):
        raise ValueError('duplicate recorded rollout positions')
    for row in trace['rounds']:
        if len(row['prop']) != block - 1 or row['teacher_logp'].shape != (block - 1,):
            raise ValueError('recorded proposal/teacher log-probability geometry mismatch')
        if row['teacher_logp'].requires_grad:
            raise ValueError('recorded teacher log-probabilities must be detached')
        if any(not isinstance(offset, int) or isinstance(offset, bool) or not 0 < offset < block
               for offset in row['rejected_offsets']):
            raise ValueError('invalid recorded rejected offsets')
    if mode_of(head) != 'none':
        request = trace.get('evidence_request')
        if not isinstance(request, dict) or len(request['prompt_visual_mask']) != prompt:
            raise ValueError('trace evidence request must retain the complete prompt identity')
    return plan


def forward_anchors(head, ctx, block_embed, anchors, prompt_len):
    """One actual head.forward, with per-anchor known-prefix evidence memory.

    This is the independent OPD counterpart of evidence_sft.forward_anchors:
    its sole boundary extension admits prompt_len-1. At that boundary only
    known_visual_prefix may adapt visual pooling; the request mask is intact.
    """
    block = int(head.block_size)
    if not anchors or ctx.ndim != 3 or ctx.shape[0] != 1:
        raise ValueError('one nonempty packed sequence required')
    if block_embed.ndim != 3 or block_embed.shape[:2] != (1, len(anchors) * block):
        raise ValueError('packed noise must contain complete root-plus-mask blocks')
    if any(a < prompt_len - 1 or a > ctx.shape[1] for a in anchors):
        raise ValueError('packed anchor lies outside its known context')
    ctx_positions = torch.arange(ctx.shape[1], device=ctx.device)[None]
    query_positions = (torch.tensor(anchors, device=ctx.device)[:, None]
                       + torch.arange(block, device=ctx.device)[None]).reshape(1, -1)
    positions = torch.cat((ctx_positions, query_positions), dim=1)
    mask = sft_blocks.build_anchor_mask(anchors, block, ctx.shape[1], ctx.device, ctx.dtype)
    mode = mode_of(head)
    module = getattr(head, 'evidence_readout', None)
    if mode == 'none':
        return head(target_hidden=ctx, noise_embedding=block_embed, position_ids=positions,
                    attention_mask=mask, past_key_values=None, use_cache=False, is_causal=False)
    if module.request is None or len(module.request['prompt_visual_mask']) != prompt_len:
        raise ValueError('packed evidence needs the active full-prompt request')
    if any(name in module.__dict__ for name in ('prepare', 'forward', '_visual')):
        raise RuntimeError('nested/custom packed evidence substitution is unsupported')
    request = module.request
    original_prepare, original_forward = module.prepare, module.forward
    calls = {'prepare': 0, 'forward': 0}

    def prepare(source, actual_positions, cache, use_cache):
        calls['prepare'] += 1
        if calls['prepare'] != 1 or cache is not None or use_cache:
            raise ValueError('one uncached packed evidence preparation required')
        if source.shape[:2] != ctx.shape[:2] or not torch.equal(actual_positions, positions):
            raise ValueError('actual head changed the validated packed layout')
        result = []
        for anchor in anchors:
            scope = (known_visual_prefix(head, prompt_len, request)
                     if mode == 'visual' and anchor == prompt_len - 1 else nullcontext())
            with scope:
                result.append(original_prepare(source[:, :anchor],
                              torch.arange(anchor + block, device=source.device)[None], None, False))
        return result

    def readout(hidden, memories):
        calls['forward'] += 1
        if calls['forward'] != 1 or hidden.shape[1] != len(anchors) * block or len(memories) != len(anchors):
            raise ValueError('unexpected packed evidence readout geometry')
        return torch.cat([original_forward(hidden[:, i * block:(i + 1) * block], memory)
                          for i, memory in enumerate(memories)], dim=1)

    module.prepare, module.forward = prepare, readout
    try:
        result = head(target_hidden=ctx, noise_embedding=block_embed, position_ids=positions,
                      attention_mask=mask, past_key_values=None, use_cache=False, is_causal=False)
        if calls != {'prepare': 1, 'forward': 1}:
            raise RuntimeError('actual head did not execute its evidence path exactly once')
        return result
    finally:
        del module.prepare
        del module.forward


def packed_logits(target, head, trace, anchors):
    """Same recorded root + B-1 MASK tokens and raw target LM projection."""
    block = int(head.block_size)
    ctx = trace['fused'][:, :max(anchors)].to(dtype=head.fc.weight.dtype)
    blocks = torch.full((len(anchors), block), int(head.mask_token_id),
                        dtype=torch.long, device=trace['ids'].device)
    blocks[:, 0] = trace['ids'][torch.tensor(anchors, device=blocks.device)]
    noise = target.get_input_embeddings()(blocks).to(head.fc.weight.dtype).reshape(1, -1, head.config.hidden_size)
    hidden = forward_anchors(head, ctx, noise, anchors, int(trace['prompt_len']))
    predictions = hidden.reshape(len(anchors), block, -1)[:, 1:, :]
    logits = target.lm_head(predictions.to(target.lm_head.weight.dtype))
    return logits


def score(target, head, trace, *, pack_size=4, e2e_tv_weight=0.0, diagnostics=None):
    """Yield grouped original OPD pools; caller must backward each group once."""
    if e2e_tv_weight != 0.0:
        raise ValueError('experimental packed OPD currently supports only --e2e-tv-weight 0')
    if isinstance(pack_size, bool) or not isinstance(pack_size, int) or pack_size not in (1, 2, 4):
        raise ValueError('pack_size must be 1, 2 or 4')
    if head.training or target.training or any(p.requires_grad for p in target.parameters()):
        raise ValueError('packed OPD requires eval heads and a frozen eval target')
    plan = validate_trace(head, trace)
    mode = mode_of(head)
    request_scope = (evidence.bound_request(head, trace['evidence_request'])
                     if mode != 'none' else nullcontext())
    roots = trace['ids'].detach().cpu().tolist()
    cache = {'cursor': 0, 'start': None, 'logits': None}

    def draft(actual_head, actual_target, fused, *, anchor_pos, root_token):
        index = cache['cursor']
        if actual_head is not head or actual_target is not target or fused is not trace['fused']:
            raise ValueError('packed logits escaped the bound trace/head')
        if index >= len(plan) or anchor_pos != plan[index][0] or root_token != roots[anchor_pos]:
            raise ValueError('original score changed recorded anchor order or known root')
        start = index // pack_size * pack_size
        if start != cache['start']:
            anchors = [pos for pos, _ in plan[start:start + pack_size]]
            cache['logits'] = packed_logits(target, head, trace, anchors)
            cache['start'] = start
        cache['cursor'] += 1
        return cache['logits'][index - start]

    with request_scope:
        original = runtime.score(target, head, trace, draft_fn=draft, e2e_tv_weight=0.0)
        try:
            for start in range(0, len(plan), pack_size):
                members = tuple(next(original) for _ in plan[start:start + pack_size])
                numerator, denominator = members[0].numerator, members[0].denominator
                for member in members[1:]:
                    numerator = numerator + member.numerator
                    denominator = denominator + member.denominator
                yield PackedPool(numerator, denominator, tuple(plan[start:start + pack_size]),
                                 members, cache['logits'])
            if next(original, None) is not None or cache['cursor'] != len(plan):
                raise RuntimeError('original scorer did not consume the exact complete plan')
            if diagnostics is not None and mode != 'none':
                diagnostics.append(head.evidence_diagnostics())
        finally:
            original.close()
            cache['logits'] = None


@contextmanager
def packed_runtime(trainer, pack_size=4):
    """Future opt-in scoring replacement; rollout/backward/optimizer untouched."""
    if isinstance(pack_size, bool) or not isinstance(pack_size, int) or pack_size not in (1, 2, 4):
        raise ValueError('pack_size must be 1, 2 or 4')
    original_score, original_protocol = trainer.score_training_trace, trainer.make_protocol

    def scoring(target, head, trace, *, e2e_tv_weight, diagnostics=None):
        yield from score(target, head, trace, pack_size=pack_size,
                         e2e_tv_weight=e2e_tv_weight, diagnostics=diagnostics)

    def protocol(args, *rest, **kwargs):
        if args.e2e_tv_weight != 0 or args.loss_normalization != 'global_pooled':
            raise ValueError('packed OPD requires lambda=0 and unchanged global_pooled normalization')
        result = original_protocol(args, *rest, **kwargs)
        result['packed_opd_score'] = dict(
            experimental=True, pack_size=pack_size, actual_head_forward=True,
            teacher_rollout_and_actions='unchanged original native trace and runtime.score',
            objective='sum original anchor numerators / original global QA denominator; no chunk average',
            backward='one backward per shared packed graph; same gradient sum formula, different floating-point grouping',
            evidence_prefix='each anchor known prefix; L-1 uses audited known_visual_prefix',
            sft_labels_used=False, cpu_cuda_bitwise_equivalence_claimed=False,
            training_numerics='not numerically equivalent training: bf16 gradient accumulation is regrouped; near-zero Adam gradients may change sign',
            memory_policy='pack4 may increase activation/logit peak; same-weight same-data CUDA memory/numerics gate required before adoption',
            production_speed_or_cuda_validation_claimed=False)
        result['hashes']['packed_opd_score'] = training_resume.file_hash(__file__)
        result['hashes']['packed_anchor_mask'] = training_resume.file_hash(sft_blocks.__file__)
        result['hashes']['packed_evidence_policy'] = training_resume.file_hash(evidence.__file__)
        result['hashes']['packed_visual_boundary'] = training_resume.file_hash(
            Path(__file__).with_name('evidence_opd_prompt_boundary.py'))
        result.pop('protocol_sha256', None)
        result['protocol_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result

    trainer.score_training_trace, trainer.make_protocol = scoring, protocol
    try:
        yield
    finally:
        trainer.score_training_trace, trainer.make_protocol = original_score, original_protocol
