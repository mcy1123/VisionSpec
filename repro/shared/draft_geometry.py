"""Shared block geometry and EOS handling; no OPD loss or rollout policy."""
import torch
from transformers import DynamicCache

def draft_block_logits(head, target, fused, *, anchor_pos, root_token,
                       block_size=None, prompt_len=None, cache=None):
    """Return the K=B-1 draft prediction rows for a root-plus-mask block.

    Context contains exactly the rows from cache length through anchor_pos-1.
    Absolute positions and fresh/cropped KV semantics match GLANCE decoding.
    This differentiable helper computes logits only; it has no training loss.
    """
    block = int(head.block_size) if block_size is None else int(block_size)
    if block < 2:
        raise ValueError(f"block_size must be >= 2, got {block}")
    if prompt_len is not None and int(anchor_pos) < int(prompt_len):
        raise ValueError(f"anchor_pos {anchor_pos} is before prompt_len {prompt_len}")

    cache = DynamicCache() if cache is None else cache
    cache_len = int(cache.get_seq_length())
    want = int(anchor_pos) - cache_len
    if fused.ndim != 3 or fused.shape[0] != 1:
        raise ValueError(f"fused must be (1, n, L*H), got {tuple(fused.shape)}")
    if int(fused.shape[1]) != want:
        raise ValueError(
            f"fused has {int(fused.shape[1])} context rows but the head needs "
            f"{want} (anchor_pos {anchor_pos} - cache {cache_len}); pass the "
            "rows for stream positions cache_len..anchor_pos-1")

    ids = torch.full((1, block), int(head.mask_token_id), dtype=torch.long,
                     device=fused.device)
    ids[0, 0] = int(root_token)
    position_ids = torch.arange(cache_len, int(anchor_pos) + block,
                                device=fused.device).unsqueeze(0)
    hidden = head(target_hidden=fused,
                  noise_embedding=target.get_input_embeddings()(ids),
                  position_ids=position_ids, past_key_values=cache,
                  use_cache=True, is_causal=False)
    return target.lm_head(hidden[:, 1 - block:, :])[0]

def eos_ids(target):
    for config in (getattr(target, "generation_config", None), target.config,
                   getattr(target.config, "text_config", None)):
        value = getattr(config, "eos_token_id", None)
        if value is not None:
            return {int(x) for x in (value if isinstance(value, (list, tuple)) else [value])}
    raise ValueError("target has no EOS token IDs")

def draft_block_logits_mixed(head, target, fused, *, anchor_pos, root_token,
                             head_dtype, block_size=None):
    """Run shared block geometry with head-dtype embeddings and target-dtype logits."""

    fused = fused if fused.dtype == head_dtype else fused.to(head_dtype)
    embed = target.get_input_embeddings()

    class _HeadDtypeTarget:
        def __init__(self, real, embedding):
            self._lm = real.lm_head
            self._lm_dtype = real.lm_head.weight.dtype
            self._embedding = embedding

        def lm_head(self, hidden):
            if hidden.dtype != self._lm_dtype:
                hidden = hidden.to(self._lm_dtype)
            return self._lm(hidden)

        def get_input_embeddings(self):
            return self._embedding

    def _embed(ids):
        out = embed(ids)
        return out if out.dtype == head_dtype else out.to(head_dtype)

    return draft_block_logits(
        head, _HeadDtypeTarget(target, _embed), fused, anchor_pos=anchor_pos,
        root_token=int(root_token), block_size=block_size)
