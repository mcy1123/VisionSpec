"""Thin image/video input adapters; all policy and loss logic is shared."""

from functools import partial


def encode(processor, row, modality, *, max_pixels=768, time_mode="real"):
    if modality == "image":
        from scienceqa_image.imageencode import encode_image_inputs

        return encode_image_inputs(processor, row, max_pixels=max_pixels)
    if modality == "video":
        from qa_opd_fair.videoencode import encode_inputs

        return encode_inputs(processor, row, time_mode)
    raise ValueError(modality)


def collect(target, head, inputs, *, max_new_tokens, max_seq_len,
            teacher_verify_impl="pure_block_native", verification_observer=None,
            batch_teacher_ids=False, preallocate_context=False):
    from transformers import DynamicCache
    from glance.vl import VisionPositions
    from glance.head import context_feature
    from shared.draft_geometry import eos_ids
    from shared import common
    from .runtime import rollout

    device = next(head.parameters()).device
    x = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    if "attention_mask" in x and not x["attention_mask"].bool().all():
        raise ValueError("adapter accepts one unpadded prompt; remove padding first")
    pos = VisionPositions(
        **{
            k: x[k]
            for k in [
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
            ]
            if k in x
        }
    )
    return rollout(
        target,
        head,
        x["input_ids"],
        pos,
        draft_fn=draft,
        feature_fn=partial(context_feature, layer_ids=common.KEPT_LAYERS),
        cache_factory=DynamicCache,
        eos=eos_ids(target),
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        teacher_verify_impl=teacher_verify_impl,
        verification_observer=verification_observer,
        batch_teacher_ids=batch_teacher_ids,
        preallocate_context=preallocate_context,
    )


def draft(head, target, fused, *, anchor_pos, root_token):
    # Geometry-only helper; no legacy objective or fixed-anchor planner used.
    from shared.draft_geometry import draft_block_logits_mixed

    return draft_block_logits_mixed(
        head,
        target,
        fused[:, :anchor_pos],
        anchor_pos=anchor_pos,
        root_token=root_token,
        head_dtype=next(head.parameters()).dtype,
    )
