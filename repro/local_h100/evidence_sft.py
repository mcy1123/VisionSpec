"""Local, request-scoped packed SFT through the actual DFlash forward.

The backbone runs once with the original anchor mask. Each evidence readout
receives only that anchor's committed prefix. No shared implementation is
changed, and no draft KV cache is created. This is an existing readout adapter,
not an implementation of the official ViSpec architecture.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import torch

from shared import sft_blocks


TIME_SETTING = 'off'
OTHER_BRANCHES = ('use_xattn', 'use_layer_gate', 'use_visual_fovea',
                  'use_visual_film', 'use_visual_xattn', 'use_visual_perceiver')


def minimal_config(value):
    """Require explicit architecture identity, not an accidentally zero tensor."""
    if isinstance(value, (str, Path)):
        value = json.loads((Path(value) / 'config.json').read_text())
    if hasattr(value, 'config'):
        value = value.config
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    draft = value.get('dflash_config', {})
    get = lambda key, default=None: draft.get(key, value.get(key, default))
    if get('evidence_mode') not in ('text', 'visual'):
        raise ValueError('local evidence entry requires text or visual evidence')
    if get('evidence_time_encoding') != TIME_SETTING:
        raise ValueError('local minimal evidence requires explicit evidence_time_encoding=off')
    if any(get(name, False) for name in OTHER_BRANCHES):
        raise ValueError('minimal evidence cannot be combined with legacy branches')
    return dict(mode=get('evidence_mode'), time_encoding=TIME_SETTING,
                dim=int(get('evidence_dim', 256)), max_tokens=int(get('evidence_max_tokens', 256)),
                spatial_bins=int(get('evidence_spatial_bins', 2)))


def assert_time_off(head, *, require_frozen=True):
    minimal_config(head)
    module = head.evidence_readout
    if any(p.dtype != torch.float32 for p in module.parameters()):
        raise ValueError('all evidence parameters must remain fp32')
    if bool(torch.count_nonzero(module.time.weight)):
        raise ValueError('time-off checkpoint contains nonzero time weights')
    if require_frozen and module.time.weight.requires_grad:
        raise ValueError('time weights were unfrozen after time-off setup')
    if module.time.weight.grad is not None:
        raise ValueError('time-off parameter unexpectedly has a gradient')


def freeze_time_after_unfreeze(head):
    """Call after blanket head.requires_grad_(True), before optimizer creation.

    Never silently zero a learned tensor: invalid checkpoint weights fail.
    Both SFT and OPD local entries must use this same initialization function.
    """
    assert_time_off(head, require_frozen=False)
    head.evidence_readout.time.weight.requires_grad_(False)
    return head


@contextmanager
def bound_request(head, request):
    """Bind exactly one QA; release references even if forward/loss fails."""
    module = head.evidence_readout
    if module.request is not None:
        raise RuntimeError('an evidence request is already active')
    head.set_evidence_request(request)
    try:
        yield
    finally:
        # set_request(None) is intentionally not part of the shared API.
        module.request = None
        module.generation += 1


def forward_anchors(head, ctx, block_embed, ctx_positions, q_positions, attn_mask):
    """Use actual head.forward once; evidence is isolated by anchor prefix.

    Temporary instance methods adapt only the packed layout. Original prepare
    and forward perform all memory validation, pooling and attention. A nested
    call is refused; methods are restored before returning the autograd graph.
    """
    module = getattr(head, 'evidence_readout', None)
    if module is None or module.request is None:
        raise ValueError('packed evidence SFT requires an active request')
    if 'prepare' in module.__dict__ or 'forward' in module.__dict__:
        raise RuntimeError('nested/custom evidence method substitution is unsupported')
    block = int(head.block_size)
    if ctx.ndim != 3 or ctx.shape[0] != 1 or block_embed.ndim != 3 or block_embed.shape[0] != 1:
        raise ValueError('packed SFT requires exactly one sequence')
    length = block_embed.shape[1]
    if length == 0 or length % block or q_positions.shape != (1, length):
        raise ValueError('packed query positions must contain complete draft blocks')
    if not torch.equal(ctx_positions, torch.arange(ctx.shape[1], device=ctx.device)[None]):
        raise ValueError('context positions must be the complete zero-based stream')
    starts = q_positions.reshape(-1, block)[:, 0]
    expected_q = starts[:, None] + torch.arange(block, device=starts.device)
    if not torch.equal(q_positions.reshape(-1, block), expected_q):
        raise ValueError('query positions must be contiguous inside each anchor')
    anchors = starts.tolist()
    prompt_len = len(module.request['prompt_visual_mask'])
    if any(a < prompt_len or a > ctx.shape[1] for a in anchors):
        raise ValueError('every anchor must include the whole prompt and lie within context')
    expected_mask = sft_blocks.build_anchor_mask(anchors, block, ctx.shape[1], ctx.device, ctx.dtype)
    if attn_mask.shape != expected_mask.shape or not torch.equal(attn_mask, expected_mask):
        raise ValueError('attention mask differs from the original isolated anchor mask')
    packed_positions = torch.cat((ctx_positions, q_positions), dim=1)
    original_prepare, original_forward = module.prepare, module.forward
    calls = {'prepare': 0, 'forward': 0}

    def prepare(source, positions, cache, use_cache):
        calls['prepare'] += 1
        if calls['prepare'] != 1 or cache is not None or use_cache:
            raise ValueError('packed SFT permits one uncached evidence preparation')
        if not torch.equal(positions, packed_positions) or source.shape[:2] != ctx.shape[:2]:
            raise ValueError('head changed the validated packed layout')
        # Full-context text memory would leak teacher features beyond an early
        # anchor. Prepare independently, even though visual prompt memory is static.
        return [original_prepare(source[:, :a], torch.arange(a + block, device=source.device)[None],
                                 None, False) for a in anchors]

    def readout(hidden, prepared):
        calls['forward'] += 1
        if calls['forward'] != 1 or hidden.shape[1] != length or len(prepared) != len(anchors):
            raise ValueError('unexpected packed readout shape or repeated invocation')
        return torch.cat([original_forward(hidden[:, i * block:(i + 1) * block], memory)
                          for i, memory in enumerate(prepared)], dim=1)

    module.prepare, module.forward = prepare, readout
    try:
        result = head(target_hidden=ctx, noise_embedding=block_embed,
                      position_ids=packed_positions, attention_mask=attn_mask,
                      past_key_values=None, use_cache=False, is_causal=False)
        if calls != {'prepare': 1, 'forward': 1}:
            raise RuntimeError('actual head.forward did not execute its evidence branch')
        return result
    finally:
        del module.prepare
        del module.forward


def hidden_loss(base_loss, head, target, ctx, full, anchors, reveals, mask_id, request):
    """Preserve the exact baseline labels/weights/LM head/CE normalization."""
    assert_time_off(head)
    original = sft_blocks.forward_anchors
    if original is forward_anchors:
        raise RuntimeError('nested evidence SFT loss is unsupported')
    with bound_request(head, request):
        sft_blocks.forward_anchors = forward_anchors
        try:
            loss, stats = base_loss(head, target, ctx, full, anchors, reveals, mask_id)
            stats = dict(stats, evidence=head.evidence_diagnostics(), evidence_time_encoding=TIME_SETTING)
            return loss, stats
        finally:
            sft_blocks.forward_anchors = original
