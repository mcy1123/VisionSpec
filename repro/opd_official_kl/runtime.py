"""Pure block-native OPD rollout and unchanged Bernoulli/k3 scoring.

Approximate teacher method. Legacy verification implementations are archived.
Deployment decoding/verifying is independent and unchanged.
"""

import torch
from .plan import verify, response_plan
from .loss import pool


@torch.no_grad()
def rollout(
    target,
    head,
    prompt_ids,
    positions,
    *,
    draft_fn,
    feature_fn,
    cache_factory,
    eos,
    max_new_tokens,
    max_seq_len,
    teacher_verify_impl="pure_block_native",
    verification_observer=None,
    batch_teacher_ids=False,
    preallocate_context=False,
):
    if teacher_verify_impl != "pure_block_native":
        raise ValueError("active OPD supports pure_block_native only; restore archive for legacy methods")
    if target.training or head.training:
        raise ValueError("rollout requires eval mode")
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or max_new_tokens < 1:
        raise ValueError("one unpadded prompt and positive budget required")
    prompt_length = prompt_ids.shape[1]
    if prompt_length < 1 or prompt_length >= max_seq_len:
        raise ValueError("prompt outside context budget")
    positions.bind(target, prompt_ids)
    out = target.model(
        prompt_ids,
        past_key_values=cache_factory(),
        use_cache=True,
        output_hidden_states=True,
        **positions.ar_prefill_inputs(prompt_ids)
    )
    cache = out.past_key_values
    fused = feature_fn(out.hidden_states).detach()
    hidden = out.last_hidden_state.detach()
    if preallocate_context:
        fused_buffer = fused.new_empty((1, max_seq_len, fused.shape[-1]))
        hidden_buffer = hidden.new_empty((1, max_seq_len, hidden.shape[-1]))
        fused_buffer[:, :prompt_length].copy_(fused)
        hidden_buffer[:, :prompt_length].copy_(hidden)
        fused = fused_buffer[:, :prompt_length]
        hidden = hidden_buffer[:, :prompt_length]
    root = int(target.lm_head(hidden[:, -1:])[0, 0].argmax())
    ids = torch.cat((prompt_ids[0], prompt_ids.new_tensor([root])))
    del out
    rounds = []
    boundaries = []
    draft_length = head.block_size - 1

    def append(token):
        pos = cache.get_seq_length()
        index = ids.new_tensor([[pos]])
        output = target.model(
            ids.new_tensor([[token]]),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            position_ids=positions(index),
            cache_position=index[0],
        )
        token_hidden = output.last_hidden_state.detach()
        return (
            feature_fn(output.hidden_states).detach(),
            token_hidden,
            target.lm_head(token_hidden)[0, 0].float(),
        )

    while root not in eos and ids.numel() - prompt_length < max_new_tokens:
        pos = ids.numel() - 1
        if pos + draft_length >= max_seq_len:
            break
        if cache.get_seq_length() != pos or fused.shape[1] != pos:
            raise RuntimeError("root/KV/fused prefix mismatch")
        logits = draft_fn(head, target, fused, anchor_pos=pos, root_token=root)
        prop = logits.argmax(-1).tolist()
        if len(prop) != draft_length:
            raise RuntimeError("draft block size mismatch")
        draft_features = []
        draft_hidden = []
        teacher_ids = []
        teacher_logp = []
        # KV ends before root. Causal row i consumes [root, d1, ..., di]
        # and predicts d(i+1); the last row supplies the all-accepted bonus.
        # No pixels or explicit mask here: the populated cache holds vision,
        # and Qwen's native causal mask uses these absolute cache positions.
        index = torch.arange(pos, pos + draft_length + 1, device=ids.device)[None]
        output = target.model(
            ids.new_tensor([[root] + prop]),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            position_ids=positions(index),
            cache_position=index[0],
        )
        block_features = feature_fn(output.hidden_states).detach()
        block_hidden = output.last_hidden_state.detach()
        block_logits = target.lm_head(block_hidden)[0].float()
        draft_features = list(block_features.split(1, dim=1))
        draft_hidden = list(block_hidden.split(1, dim=1))
        teacher_ids = block_logits.argmax(-1).tolist()
        proposal_ids = ids.new_tensor(prop)
        teacher_logp = list(
            block_logits[:-1].log_softmax(-1).gather(1, proposal_ids[:, None]).unbind(0)
        )
        teacher_logp = [value.squeeze(0) for value in teacher_logp]
        if verification_observer is not None:
            verification_observer(pos, [root] + prop, block_logits)
        del output, block_logits
        decision = verify(
            prop,
            teacher_ids,
            eos,
            min(
                max_new_tokens - (ids.numel() - prompt_length),
                max_seq_len - ids.numel(),
            ),
        )
        rounds.append(
            dict(
                pos=pos,
                prop=prop,
                teacher_ids=teacher_ids,
                teacher_logp=torch.stack(teacher_logp),
                **decision
            )
        )
        ids = torch.cat((ids, ids.new_tensor(decision["committed"])))
        boundaries.append(ids.numel() - prompt_length - 1)
        keep = len(decision["committed"])
        cache.crop(pos + keep)
        if preallocate_context:
            for offset, (f, h) in enumerate(zip(draft_features[:keep], draft_hidden[:keep])):
                fused_buffer[:, pos+offset:pos+offset+1].copy_(f)
                hidden_buffer[:, pos+offset:pos+offset+1].copy_(h)
            fused = fused_buffer[:, :pos+keep]
            hidden = hidden_buffer[:, :pos+keep]
        else:
            fused = torch.cat((fused, *draft_features[:keep]), dim=1)
            hidden = torch.cat((hidden, *draft_hidden[:keep]), dim=1)
        root = int(ids[-1])
    # Last root is not in cache yet. Materialize for correct-trajectory scoring.
    token_features, token_hidden, _ = append(root)
    if preallocate_context:
        fused_buffer[:, ids.numel()-1:ids.numel()].copy_(token_features)
        hidden_buffer[:, ids.numel()-1:ids.numel()].copy_(token_hidden)
        fused = fused_buffer[:, :ids.numel()]
        hidden = hidden_buffer[:, :ids.numel()]
    else:
        fused = torch.cat((fused, token_features), dim=1)
        hidden = torch.cat((hidden, token_hidden), dim=1)
    if cache.get_seq_length() != ids.numel():
        raise RuntimeError("final KV mismatch")
    rejected = [
        (round_data["pos"], output)
        for round_data in rounds
        for output in round_data["rejected_offsets"]
    ]
    plan = response_plan(
        prompt_length,
        ids.numel() - prompt_length,
        boundaries,
        head.block_size,
        rejected,
    )
    trace = dict(
        ids=ids,
        prompt_len=prompt_length,
        fused=fused,
        hidden=hidden,
        rounds=rounds,
        plan=plan,
        boundaries=boundaries,
        final_cache_length=cache.get_seq_length(),
    )
    trace['teacher_verify_impl'] = teacher_verify_impl
    return trace


def score(target, head, trace, *, draft_fn, deferred_checks=False):
    """Score recorded proposal tokens with current logits, then yield pool stats.

    Replay argmax need not match the collected proposal: its recorded tokens are
    the sampled actions being scored. ``deferred_checks`` is retained for older
    diagnostic callers; it no longer enables argmax-equality gates.
    """
    by_pos = {round_data["pos"]: round_data for round_data in trace["rounds"]}
    uncovered = (
        trace["ids"].numel() - trace["prompt_len"] - sum(n for _, n in trace["plan"])
    )
    for index, (pos, count) in enumerate(trace["plan"]):
        logits = draft_fn(
            head,
            target,
            trace["fused"],
            anchor_pos=pos,
            root_token=int(trace["ids"][pos]),
        )
        student_logp = logits.float().log_softmax(-1)
        empty = student_logp.reshape(-1)[:0]
        round_data = by_pos.get(pos)
        if count:
            tokens = trace["ids"][pos + 1 : pos + count + 1]
            response_student_logp = (
                student_logp[:count].gather(1, tokens[:, None]).squeeze(1)
            )
            with torch.no_grad():
                teacher_logp = (
                    target.lm_head(trace["hidden"][0, pos : pos + count])
                    .float()
                    .log_softmax(-1)
                )
                # Keep the native approximate-teacher scoring used by the completed run.
                response_teacher_logp = teacher_logp.gather(1, tokens[:, None]).squeeze(
                    1
                )
        else:
            response_student_logp = response_teacher_logp = empty
        offsets = student_logp.new_tensor(
            round_data["rejected_offsets"] if round_data else [], dtype=torch.long
        )
        if offsets.numel():
            token = trace["ids"].new_tensor(round_data["prop"])[offsets - 1]
            rejected_student_logp = (
                student_logp[offsets - 1].gather(1, token[:, None]).squeeze(1)
            )
            rejected_teacher_logp = round_data["teacher_logp"][offsets - 1]
        else:
            rejected_student_logp = rejected_teacher_logp = empty
        yield pool(
            response_student_logp,
            response_teacher_logp,
            rejected_student_logp,
            rejected_teacher_logp,
            offsets,
            response_count=count + (uncovered if index == 0 else 0),
        )


def denominator(trace):
    return (
        float(trace["ids"].numel() - trace["prompt_len"])
        + 2
        * sum(
            0.9 ** (offset - 1)
            for round_data in trace["rounds"]
            for offset in round_data["rejected_offsets"]
        )
    )
