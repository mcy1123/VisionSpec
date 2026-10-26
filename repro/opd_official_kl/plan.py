"""Single-sample stride=1 adapter of DFlashStudent._build_opd_anchor_plan.

The response stream includes target correction/bonus tokens. It is NOT just
leading accepted draft tokens. Padding never creates response boundaries.
"""


def response_plan(prompt_len, response_len, boundaries, block_size, rejected=()):
    rejects = sorted({int(x) for x in boundaries if 0 <= int(x) < response_len})
    plan = []
    if rejects:
        if rejects[-1] < response_len - 1:
            rejects.append(response_len - 1)
        for start, end in zip([-1] + rejects, rejects):
            pos = prompt_len + start
            count = min(end - start, block_size - 1)
            if count > 0 and 0 <= pos < prompt_len + response_len - 1:
                plan.append((pos, count))
    existing = {p for p, n in plan}
    for pos, offset in rejected:
        if (
            0 < offset < block_size
            and 0 <= pos < prompt_len + response_len
            and pos not in existing
        ):
            plan.append((pos, 0))
            existing.add(pos)
    return plan


def verify(proposal, teacher_ids, eos, remaining):
    """SGLang greedy verify semantics, including termination before rejection.

    Only if correction was reached is the ENTIRE rejected suffix recorded;
    suffix metadata may extend past output EOS/cap, as in frozen dflash_info.py.
    """
    if len(teacher_ids) != len(proposal) + 1 or remaining <= 0:
        raise ValueError("invalid verification rows or remaining budget")
    n = 0
    for q, p in zip(proposal, teacher_ids):
        if q != p:
            break
        n += 1
    proposed = proposal[:n] + [teacher_ids[n]]
    committed = []
    for token in proposed[:remaining]:
        committed.append(token)
        if token in eos:
            break
    offsets = (
        list(range(n + 1, len(proposal) + 1))
        if len(committed) >= n + 1 and n < len(proposal)
        else []
    )
    return dict(n_acc=n, committed=committed, rejected_offsets=offsets)
