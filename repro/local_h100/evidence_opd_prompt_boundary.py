"""Local visual-memory adaptation for OPD's prompt_len-1 scoring anchor.

The first response anchor uses the final prompt text token as its known root.
Its backbone context therefore stops one token before the complete prompt.
When every visual position is already in that prefix, the visual readout can
pool those exact known features. The full request mask and identity remain
unchanged. No future hidden state, padding or alternate anchor plan is used.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from shared import training_resume


@contextmanager
def known_visual_prefix(head, prompt_len, request):
    """Allow only the complete-visual, final-prompt-text scoring boundary."""
    module = head.evidence_readout
    if module.mode != 'visual' or module.request is not request:
        raise ValueError('boundary visual adapter requires the active trace request')
    if '_visual' in module.__dict__:
        raise RuntimeError('nested/custom visual-memory override is unsupported')
    mask = request['prompt_visual_mask']
    if len(mask) != prompt_len or prompt_len < 2:
        raise ValueError('trace prompt length differs from the complete request mask')
    positions = mask.nonzero(as_tuple=True)[0]
    if not len(positions) or int(positions[-1]) >= prompt_len - 1:
        raise ValueError('first OPD scoring anchor does not contain every visual position')
    if request.get('intervention', 'normal') != 'normal':
        raise ValueError('boundary training adapter accepts only normal visual evidence')
    if getattr(module, 'paper_method', None):
        if 'prepared_visual' not in request:
            raise ValueError('missing captured vision projector output for scoring')
        yield
        return
    def visual(source, prefix_len):
        # This override is installed only around the known first anchor call.
        if prefix_len != 0 or source.shape[1] != prompt_len - 1:
            raise ValueError('boundary adaptation requires a fresh cache and exactly prompt_len-1 known rows')
        if module.request is not request or source.ndim != 3 or source.shape[0] != 1 or source.shape[-1] != module.hidden_size:
            raise ValueError('boundary source/request changed')
        old = request.get('prepared_visual')
        if old is not None and (old.get('identity') != request['identity'] or old.get('config') != module._config_key()):
            raise ValueError('prepared visual evidence identity/config mismatch')
        # Integer positions reference the unmodified full prompt mask. All are
        # proven to be inside source; the missing final text feature is unused.
        t, h, w = request['grid_thw']
        merge = request['spatial_merge_size']
        h, w = h // merge, w // merge
        by, bx = min(module.spatial_bins, h), min(module.spatial_bins, w)
        if by * bx > module.max_tokens:
            raise ValueError('evidence_max_tokens is smaller than one spatial group')
        groups = min(t, module.max_tokens // (by * bx))
        visual_features = source[0, positions.to(source.device)].detach().float().reshape(t, h, w, module.hidden_size)
        features = F.adaptive_avg_pool3d(visual_features.permute(3, 0, 1, 2)[None],
                                        (groups, by, bx))[0].permute(1, 2, 3, 0).reshape(-1, module.hidden_size)
        times = F.adaptive_avg_pool1d(request['time_values'].to(source.device).float()[None, None], groups)[0, 0]
        times = times.repeat_interleave(by * bx)
        with torch.inference_mode(False), torch.no_grad():
            features, times = features.detach().clone(), times.detach().clone()
        request['prepared_visual'] = dict(identity=request['identity'], config=module._config_key(),
                                           features=features, times=times, groups=groups, spatial_bins=by * bx)
        module._stats['known_visual_prefix_prepares'] = module._stats.get('known_visual_prefix_prepares', 0) + 1
        return features, times

    module._visual = visual
    try:
        yield
    finally:
        del module._visual


@contextmanager
def prompt_boundary_runtime(trainer):
    """Compose inside the existing local evidence OPD runtime, in one process."""
    original_score, original_protocol = trainer.score_training_trace, trainer.make_protocol

    def score(target, head, trace, *args, **kwargs):
        prompt_len = int(trace['prompt_len'])
        request = trace.get('evidence_request')
        if not isinstance(request, dict) or len(request['prompt_visual_mask']) != prompt_len:
            raise ValueError('scoring trace/request prompt length mismatch')
        if any(pos < prompt_len - 1 for pos, _ in trace['plan']):
            raise ValueError('OPD scoring anchor precedes the final known prompt root')
        original_draft = trainer.adapters.draft

        def draft(actual_head, actual_target, fused, *, anchor_pos, root_token):
            if actual_head is not head or actual_target is not target or fused is not trace['fused']:
                raise ValueError('scoring draft is not bound to the active trace')
            if int(root_token) != int(trace['ids'][anchor_pos]):
                raise ValueError('scoring root differs from the recorded known token')
            if anchor_pos == prompt_len - 1 and head.evidence_readout.mode == 'visual':
                with known_visual_prefix(head, prompt_len, request):
                    return original_draft(head, target, fused, anchor_pos=anchor_pos, root_token=root_token)
            return original_draft(head, target, fused, anchor_pos=anchor_pos, root_token=root_token)

        trainer.adapters.draft = draft
        try:
            yield from original_score(target, head, trace, *args, **kwargs)
        finally:
            trainer.adapters.draft = original_draft

    def protocol(*args, **kwargs):
        value = original_protocol(*args, **kwargs)
        value['evidence_prompt_boundary'] = dict(
            implementation='local known-visual-prefix pooling for prompt_len-1 response scoring only',
            backbone_and_opd_plan='unchanged', request_mask_and_identity='unchanged complete prompt',
            causal_guard='fresh cache; exactly prompt_len-1 source rows; every visual index strictly before anchor',
            source_policy='only actual known source rows; no padding or future hidden features',
            text_memory='unchanged original prefix rule', helper_sha256=training_resume.file_hash(__file__))
        value['local_evidence_boundary_entry'] = 'local_h100/train_video_evidence_opd_boundary.py'
        value['hashes']['local_evidence_boundary_entry'] = training_resume.file_hash(
            Path(__file__).with_name('train_video_evidence_opd_boundary.py'))
        value.pop('protocol_sha256', None)
        value['protocol_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return value

    trainer.score_training_trace, trainer.make_protocol = score, protocol
    try:
        yield
    finally:
        trainer.score_training_trace, trainer.make_protocol = original_score, original_protocol
