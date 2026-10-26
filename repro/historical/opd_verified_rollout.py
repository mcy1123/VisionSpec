"""Current-target greedy chain rollout; no stored continuation is consumed.

Reference implementation, prioritising reproducible labels over throughput:
target verification is token-serial, including the rejected draft suffix, and
the student uses a fresh context cache just like the differentiable loss pass.
Only verified prefix KV/hidden rows survive; the correction/bonus becomes the
next root. This is an online full-KL variant, not the official sampled loss.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()


import torch
from transformers import DynamicCache

from shared import common
from historical import opd_fullkl
from glance.head import context_feature


def eos_ids(target):
    for config in (getattr(target, "generation_config", None), target.config,
                   getattr(target.config, "text_config", None)):
        value = getattr(config, "eos_token_id", None)
        if value is not None:
            return {int(x) for x in (value if isinstance(value, (list, tuple)) else [value])}
    raise ValueError("target has no EOS token IDs")


def decision(prop, teacher_ids, *, eos, remaining):
    """Return accepted length, committed suffix and supervision validity.

teacher_ids has K+1 rows, including the all-accepted bonus distribution.
EOS ends a draft-induced sequence even when it is a rejected proposal.
"""
    if len(teacher_ids) != len(prop) + 1 or remaining < 0:
        raise ValueError("verification rows/budget do not match proposal")
    valid, alive = [], True
    for i, token in enumerate(prop):
        valid.append(alive and i < remaining)
        alive = alive and token not in eos
    accepted, committed = 0, []
    for i, token in enumerate(prop):
        if len(committed) >= remaining:
            break
        if token != teacher_ids[i]:
            committed.append(teacher_ids[i])
            break
        committed.append(token)
        accepted += 1
        if token in eos:
            break
    else:
        if len(committed) < remaining:
            committed.append(teacher_ids[-1])
    return accepted, committed, valid


def assert_replay(plan, p_logits, q_logits):
    """Fail closed if scoring does not reproduce the policy/teacher labels."""
    if plan.teacher_ids is None:
        raise RuntimeError("online plan lacks current-target verification evidence")
    q_ids = q_logits.detach().argmax(-1).tolist()
    if q_ids != plan.prop:
        raise RuntimeError(f"online student replay drift at anchor {plan.pos}")
    got = p_logits.argmax(-1).tolist()
    if got != plan.teacher_ids:
        raise RuntimeError(f"online teacher replay drift at anchor {plan.pos}")
    accepted = 0
    for token, teacher, valid in zip(plan.prop, got, plan.valid):
        if not valid or token != teacher:
            break
        accepted += 1
    if accepted != plan.n_acc:
        raise RuntimeError(f"online acceptance label drift at anchor {plan.pos}")


@torch.no_grad()
def rollout(target, head, prompt_ids, *, pixel_values_videos, video_grid_thw,
            max_seq_len, max_new_tokens, max_rounds, head_dtype, draft_fn):
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("expected one encoded prompt")
    if getattr(head, "training", False) or getattr(target, "training", False):
        raise ValueError("online rollout requires eval mode (autograd may remain enabled)")
    if head_dtype is None and hasattr(head, "parameters"):
        head_dtype = next(head.parameters()).dtype
    prompt_len = int(prompt_ids.shape[1])
    if max_new_tokens < 1 or max_rounds < 0 or prompt_len >= max_seq_len:
        raise ValueError("invalid online rollout bounds")
    k = int(head.block_size) - 1
    eos = eos_ids(target)
    vispos = opd_fullkl.prefix_vision_positions(
        target, prompt_ids, pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw)
    out = target.model(prompt_ids, past_key_values=DynamicCache(), use_cache=True,
                       output_hidden_states=True,
                       **vispos.prefill_inputs(prompt_ids, None))
    cache = out.past_key_values
    fused = context_feature(out.hidden_states, common.KEPT_LAYERS).detach()
    hidden = out.last_hidden_state.detach()
    root = int(target.lm_head(hidden[:, -1:])[0, 0].argmax())
    ids = torch.cat((prompt_ids[0], prompt_ids.new_tensor([root])))
    del out
    anchors = []
    stop = "length"

    def append(token):
        pos = int(cache.get_seq_length())
        index = torch.tensor([[pos]], device=ids.device)
        out = target.model(ids.new_tensor([[token]]), past_key_values=cache,
                           use_cache=True, output_hidden_states=True,
                           position_ids=vispos(index), cache_position=index[0])
        h = out.last_hidden_state.detach()
        f = context_feature(out.hidden_states, common.KEPT_LAYERS).detach()
        greedy = int(target.lm_head(h)[0, 0].argmax())
        return f, h, greedy

    while True:
        pos = int(ids.numel()) - 1
        if root in eos:
            stop = "eos"
            break
        if max_rounds and len(anchors) >= max_rounds:
            stop = "round_limit"
            break
        remaining = min(max_new_tokens - (ids.numel() - prompt_len),
                        max_seq_len - ids.numel())
        if remaining <= 0 or pos + k >= max_seq_len:
            break
        if cache.get_seq_length() != pos or fused.shape[1] != pos:
            raise RuntimeError("online root/cache/context positions disagree")
        logits = draft_fn(head, target, fused, anchor_pos=pos, root_token=root,
                          prompt_len=prompt_len, head_dtype=head_dtype)
        prop = logits.argmax(-1).tolist()
        del logits
        fs, hs, teacher_ids = [], [], []
        # Evaluate root + full proposal. Rejected suffix rows are discarded
        # from the live state, but their distributions define the KL labels.
        for token in [root] + prop:
            f, h, greedy = append(token)
            fs.append(f)
            hs.append(h)
            teacher_ids.append(greedy)
        n_acc, committed, valid = decision(prop, teacher_ids, eos=eos,
                                          remaining=int(remaining))
        anchors.append(dict(pos=pos, prop=prop, n_acc=n_acc, valid=valid,
                            teacher_ids=teacher_ids[:-1],
                            committed=committed, source="current_target"))
        ids = torch.cat((ids, ids.new_tensor(committed)))
        # Keep all committed rows EXCEPT the last token, the next root. This
        # also handles accepted EOS and length truncation without a fake bonus.
        keep = len(committed)
        cache.crop(pos + keep)
        fused = torch.cat((fused, *fs[:keep]), dim=1)
        hidden = torch.cat((hidden, *hs[:keep]), dim=1)
        root = int(ids[-1])
        del fs, hs
    # Materialise the final root for a self-contained, crop-able training trace.
    f, h, _ = append(root)
    fused = torch.cat((fused, f), dim=1)
    hidden = torch.cat((hidden, h), dim=1)
    if cache.get_seq_length() != ids.numel():
        raise RuntimeError("final online cache length differs from committed IDs")
    return dict(full_ids=ids, prompt_len=prompt_len, valid_end=ids.numel() - 1,
                fused=fused, prefix_kv=cache, prefix_hidden=hidden,
                vispos=vispos, pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw, stop_reason=stop), anchors
