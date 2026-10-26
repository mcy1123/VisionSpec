"""Shared variable-reveal SFT blocks and differentiable head forward."""
import torch
import torch.nn.functional as F
from shared import common, markov as markov_mod

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def rope_tables(head, positions, dtype):
    """cos/sin for arbitrary positions, shaped (1, 1, L, head_dim)."""
    dummy = torch.zeros(1, positions.shape[1], head.config.hidden_size,
                        device=positions.device, dtype=dtype)
    cos, sin = head.rotary_emb(dummy, positions)
    return cos.unsqueeze(1), sin.unsqueeze(1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

def build_anchor_mask(anchors, block_size, ctx_len, device, dtype):
    """Additive mask (1, 1, L, ctx_len + L) for one sequence's anchors.

    Block `i` may read the context positions before its anchor and its own
    block, and nothing else -- the exact context it would have had at decode
    time. Blocks are bidirectional inside themselves, as the block-diffusion
    head is (`is_causal=False` in `glance/decode.py`).
    """
    n = len(anchors)
    length = n * block_size
    keys = ctx_len + length
    mask = torch.full((1, 1, length, keys), float("-inf"), device=device, dtype=dtype)
    rows = torch.arange(length, device=device)
    for i, anchor in enumerate(anchors):
        lo, hi = i * block_size, (i + 1) * block_size
        if anchor > 0:
            mask[0, 0, lo:hi, :anchor] = 0.0
        mask[0, 0, lo:hi, ctx_len + lo: ctx_len + hi] = 0.0
    del rows
    return mask

def forward_anchors(head, ctx, block_embed, ctx_positions, q_positions, attn_mask):
    """Run every anchor of one sequence through the head in one pass.

    `ctx` is the raw concatenated target hidden state (pre-`fc`), so the
    expensive 20480->4096 projection happens once instead of once per anchor.
    """
    cfg = head.config
    ctx_hidden = head.hidden_norm(head.fc(ctx))          # (1, T, H)
    cos_c, sin_c = rope_tables(head, ctx_positions, block_embed.dtype)
    cos_q, sin_q = rope_tables(head, q_positions, block_embed.dtype)

    x = block_embed
    for layer in head.layers:
        attn = layer.self_attn
        residual = x
        xn = layer.input_layernorm(x)
        bsz, length, _ = xn.shape

        q = attn.q_proj(xn).view(bsz, length,
                                 cfg.num_key_value_heads * attn.num_key_value_groups,
                                 attn.head_dim)
        q = attn.q_norm(q).transpose(1, 2)               # (B, H, L, D)

        kc = attn.k_proj(ctx_hidden).view(
            bsz, -1, cfg.num_key_value_heads, attn.head_dim)
        kn = attn.k_proj(xn).view(bsz, length, cfg.num_key_value_heads, attn.head_dim)
        kc = attn.k_norm(kc).transpose(1, 2)             # (B, Kv, T, D)
        kn = attn.k_norm(kn).transpose(1, 2)             # (B, Kv, L, D)
        vc = attn.v_proj(ctx_hidden).view(
            bsz, -1, cfg.num_key_value_heads, attn.head_dim).transpose(1, 2)
        vn = attn.v_proj(xn).view(
            bsz, length, cfg.num_key_value_heads, attn.head_dim).transpose(1, 2)

        q = apply_rope(q, cos_q, sin_q)
        kc = apply_rope(kc, cos_c, sin_c)
        kn = apply_rope(kn, cos_q, sin_q)

        k = torch.cat([kc, kn], dim=2)
        v = torch.cat([vc, vn], dim=2)
        # transformers' sdpa path repeats the KV heads rather than asking
        # PyTorch for GQA; matching it keeps the bf16 drift against the
        # released forward to rounding noise instead of a kernel difference.
        if attn.num_key_value_groups > 1:
            k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
            v = v.repeat_interleave(attn.num_key_value_groups, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=attn.scaling)
        out = out.transpose(1, 2).reshape(bsz, length, -1)
        out = attn.o_proj(out)
        x = residual + out

        residual = x
        x = residual + layer.mlp(layer.post_attention_layernorm(x))
    return head.norm(x)

def sample_reveals(n, block_size, generator, device, base=None):
    """Reveal lengths from the truncated-geometric prior (CONFIG.md §5).

    `base` is the one free constant the paper does not pin down (D2SD gives only
    `beta in (0,1)`), so it is a flag rather than a literal: the prior controls
    how much of the corpus trains the shortest reveal, and the shortest reveal
    is the only one decode ever uses.
    """
    lengths, probs = common.prefix_weights(block_size, base)
    idx = torch.multinomial(probs.to(torch.float32), n, replacement=True, generator=generator)
    return lengths.to(device)[idx]

def live_anchors(anchors, reveals, total, block_size):
    """Drop anchors whose block has nothing left to supervise.

    A block supervises offsets `>= reveal` that are still inside the row, so a
    short generation's last blocks can be empty. Removing an empty block does
    not change the objective -- the loss is a weighted mean over supervised
    positions -- it only stops the run paying for a forward with no gradient.
    """
    if not anchors:
        return anchors, reveals
    # A mismatch here would index a short tensor with a long index and, on CUDA,
    # raise a device-side assert that surfaces somewhere else entirely. Fail in
    # Python instead.
    if len(anchors) != len(reveals):
        raise ValueError(
            f"live_anchors got {len(anchors)} anchors and {len(reveals)} reveals")
    offsets = torch.arange(block_size, device=reveals.device)
    index = (torch.tensor(anchors, device=reveals.device).unsqueeze(1)
             + offsets.unsqueeze(0))
    in_bounds = index <= (total - 1)
    visible = offsets.unsqueeze(0) < reveals.unsqueeze(1)
    keep = (~visible & in_bounds).any(dim=1)
    idx = keep.nonzero(as_tuple=True)[0]
    return [anchors[i] for i in idx.tolist()], reveals[idx]

def build_blocks(seq_ids, anchors, reveals, block_size, mask_id, device):
    """Block token ids and the supervised-position plan.

    Block position `k` predicts the sequence token at `anchor + k`
    (`label_start = 0`, matching `glance/decode.py`). Positions `< reveal` are
    real tokens, the rest are MASK; only `k >= reveal` is supervised.

    Short generations (a TextVQA answer is a few tokens) still train: offsets
    that run past the end of the row are masked out rather than dropping the
    row, which is what the reference block trainers do with a short sequence's
    partial block. Past the end the decode-time stream is MASK, so the block
    sees MASK there too.
    """
    n = len(anchors)
    seq_len = seq_ids.shape[0]
    offsets = torch.arange(block_size, device=device)
    index = torch.tensor(anchors, device=device).unsqueeze(1) + offsets.unsqueeze(0)
    in_bounds = index <= (seq_len - 1)
    targets = seq_ids[index.clamp(max=seq_len - 1)]           # (N, B) ints

    visible = offsets.unsqueeze(0) < reveals.unsqueeze(1)     # (N, B)
    real = visible & in_bounds
    blocks = torch.where(real, targets, torch.full_like(targets, mask_id))
    super_mask = (~visible & in_bounds).to(torch.float32)

    # [Table 5] "coverage CE, 0.8^offset weight" -- decay anchored at the
    # reveal boundary, w_k = 0.8 ** (k - reveal). CONFIG.md §5.
    rel = offsets.unsqueeze(0) - reveals.unsqueeze(1)
    weights = torch.where(rel >= 0,
                          common.DECAY ** rel.clamp_min(0).to(torch.float32),
                          torch.zeros_like(rel, dtype=torch.float32))
    return blocks, targets, super_mask, weights * super_mask

def self_conditioned_source(blocks, super_mask, own_full):
    """The previous-token condition for self-conditioned Markov training.

    Slot ``k`` is conditioned on slot ``k-1``: the model's *own* proposal for
    that slot where it was itself a prediction (``super_mask``), and the block's
    given token (a true token, or the committed root at slot 0) where it was
    not. That is what `markov.steer_block_greedy` feeds the bias at decode, so
    training on it removes the train/inference conditioning mismatch that makes
    ``--markov-input gt`` actively harmful at decode.

    Pulled out as a pure function because the failure mode here is silent: if
    the shift or the mask were wrong, every condition would fall back to the
    block and the arm would quietly become ``--markov-input block``.
    """
    prev_sup = torch.zeros_like(super_mask, dtype=torch.bool)
    prev_sup[:, 1:] = super_mask[:, :-1].bool()
    shifted_own = torch.cat([blocks[:, :1], own_full[:, :-1]], dim=1)
    return torch.where(prev_sup, shifted_own,
                       markov_mod.block_prev_tokens(blocks))

def sequence_loss(head, target, ctx, seq_ids, anchors, reveals, embed_weight,
                  mask_id, block_size, device, verify_logits=None,
                  markov=None, markov_input="gt"):
    n = len(anchors)
    length = n * block_size
    blocks, tgt, super_mask, weights = build_blocks(
        seq_ids, anchors, reveals, block_size, mask_id, device)

    embed = F.embedding(blocks, embed_weight).reshape(1, n * block_size, -1)
    ctx_len = ctx.shape[1]
    ctx_positions = torch.arange(ctx_len, device=device).unsqueeze(0)
    q_positions = (torch.tensor(anchors, device=device).unsqueeze(1)
                   + torch.arange(block_size, device=device).unsqueeze(0)).reshape(1, length)
    attn_mask = build_anchor_mask(anchors, block_size, ctx_len, device, embed.dtype)

    hidden = forward_anchors(head, ctx, embed, ctx_positions, q_positions, attn_mask)
    hidden = hidden.reshape(1, length, -1)[0]                 # (L, H)

    flat_super = super_mask.reshape(-1).bool()
    sel_hidden = hidden[flat_super]
    if verify_logits is not None:
        return sel_hidden
    logits = head.compute_logits(sel_hidden[None], target.lm_head)[0]

    # DSpark's sequential bias, applied only where the loss actually looks -- the
    # full [N, B, V] bias would be 255*16*151936 floats per row for nothing.
    # Slot k's condition is the block's slot k-1 (DSpark's sample_from_anchor=False,
    # which is GLANCE's convention).
    #   markov_input == "gt"    -> the target's own token (DSpark's training recipe)
    #   markov_input == "block" -> what the block actually holds, i.e. MASK past
    #                              the reveal boundary; the cheap stand-in for
    #                              "condition on your own proposal", which at
    #                              decode time is what the semi-AR loop feeds it.
    if markov is not None:
        if markov_input == "self":
            # Self-conditioned training: feed the bias the model's OWN proposal
            # for the previous slot, which is what `steer_block_greedy` feeds it
            # at decode. Conditioning on the target's true tokens instead
            # (`gt`) teaches a mapping that is never queried at inference: the
            # bias then sees an out-of-distribution prefix and *lowers* decode
            # accuracy even while it lifts the teacher-forced metric (measured:
            # training-time deep offsets +0.123, decode-time off8 -0.060).
            #
            # Pass A conditions on the target's own tokens to obtain the
            # model's proposal at every supervised slot; it only supplies an
            # argmax, so it is under no_grad and never carries gradient.
            prev_gt = markov_mod.block_prev_tokens(tgt).reshape(-1)[flat_super]
            with torch.no_grad():
                own = (logits + markov.bias(prev_gt).to(logits.dtype)).argmax(dim=-1)
                own_full = torch.zeros_like(blocks)
                own_full.reshape(-1)[flat_super] = own
            # Slot k's condition is the previous slot's token, taken from the
            # model's own proposal where that previous slot was itself a
            # prediction, and from the block (a true token, or the root) where
            # it was given. This mirrors decode, where the root is committed and
            # every later condition is the spine token just chosen.
            src = self_conditioned_source(blocks, super_mask, own_full)
            sel_prev = src.reshape(-1)[flat_super]
        else:
            source = tgt if markov_input == "gt" else blocks
            sel_prev = markov_mod.block_prev_tokens(source).reshape(-1)[flat_super]
        logits = logits + markov.bias(sel_prev).to(logits.dtype)

    labels = tgt.reshape(-1)[flat_super]
    flat_w = weights.reshape(-1)[flat_super]

    nll = F.cross_entropy(logits.float(), labels, reduction="none")
    loss = (nll * flat_w).sum() / flat_w.sum().clamp_min(1.0)
    with torch.no_grad():
        correct = (logits.argmax(-1) == labels).float()
        # [App.D] "The final head is selected by held-out top-1 accuracy at
        # each offset", so offset-wise accuracy is the metric that matters.
        offsets = (torch.arange(block_size, device=device).unsqueeze(0)
                   .expand(n, block_size))[super_mask.bool()]
        by_offset = torch.zeros(block_size, 2, device=device)
        by_offset[:, 0].index_add_(0, offsets, correct)
        by_offset[:, 1].index_add_(0, offsets, torch.ones_like(correct))
    return loss, correct, flat_w, by_offset

def verify(head, target, ctx, seq_ids, mask_id, block_size, device, anchor=None):
    """One anchor, released forward vs. this file's forward, must agree.

    The released forward is handed only the prefix, with positions running
    0..anchor+block_size-1, which is exactly the decode-time layout. This
    file's forward gets the whole sequence and a mask that hides everything
    at or past the anchor, so the two must produce the same block hidden
    states.
    """
    total = ctx.shape[1]
    if anchor is None:
        anchor = max(1, total // 2 - block_size)
    anchor = max(1, min(anchor, total - block_size - 1))

    offsets = torch.arange(block_size, device=device)
    block = seq_ids[anchor + offsets].clone()
    block[1:] = mask_id                          # decode-time reveal: l = 1
    embed_w = target.get_input_embeddings().weight
    noise = F.embedding(block[None], embed_w)

    position_ids = torch.cat([torch.arange(anchor, device=device),
                              torch.arange(anchor, anchor + block_size, device=device)]
                             ).unsqueeze(0)
    with torch.no_grad():
        stock = head(target_hidden=ctx[:, :anchor], noise_embedding=noise,
                     position_ids=position_ids)[0]

        ctx_positions = torch.arange(total, device=device).unsqueeze(0)
        q_positions = (anchor + offsets).unsqueeze(0)
        attn_mask = build_anchor_mask([anchor], block_size, total, device, noise.dtype)
        mine = forward_anchors(head, ctx, noise, ctx_positions, q_positions, attn_mask)[0]

    diff = (stock.float() - mine.float()).abs().max().item()
    scale = stock.float().abs().max().item()
    cosine = F.cosine_similarity(stock.float().flatten(1),
                                 mine.float().flatten(1)).mean().item()
    return diff, scale, cosine
