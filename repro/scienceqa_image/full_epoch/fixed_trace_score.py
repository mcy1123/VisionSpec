"""Diagnostic-only reference scoring of a frozen trajectory under rebuilt context.

A changed reference representation may change greedy proposals. We score the
original sampled tokens with their actual reference probabilities; no logits are
edited to force the native policy's argmax. Production runtime.score is unchanged.
"""
import torch
from opd_official_kl.loss import pool


def score(target, head, trace, *, draft_fn, diagnostics):
    rounds = {r['pos']: r for r in trace['rounds']}
    uncovered = trace['ids'].numel() - trace['prompt_len'] - sum(n for _, n in trace['plan'])
    for index, (pos, count) in enumerate(trace['plan']):
        logits = draft_fn(head, target, trace['fused'], anchor_pos=pos, root_token=int(trace['ids'][pos]))
        student = logits.float().log_softmax(-1)
        empty = student.reshape(-1)[:0]
        round_data = rounds.get(pos)
        if round_data is not None:
            actual = student.detach().argmax(-1).tolist()
            differences = [i+1 for i,(a,b) in enumerate(zip(actual,round_data['prop'])) if a != b]
            if differences:
                diagnostics.append(dict(anchor=pos,changed_proposal_offsets=differences,
                    original=round_data['prop'],reference_argmax=actual,
                    cause='independent reference fused context; fixed native proposal remains the scored token'))
        if count:
            tokens = trace['ids'][pos+1:pos+count+1]
            response_student = student[:count].gather(1,tokens[:,None]).squeeze(1)
            with torch.no_grad():
                response_teacher = target.lm_head(trace['hidden'][0,pos:pos+count]).float().log_softmax(-1).gather(1,tokens[:,None]).squeeze(1)
        else:
            response_student = response_teacher = empty
        offsets = student.new_tensor(round_data['rejected_offsets'] if round_data else [],dtype=torch.long)
        if offsets.numel():
            token = trace['ids'].new_tensor(round_data['prop'])[offsets-1]
            rejected_student = student[offsets-1].gather(1,token[:,None]).squeeze(1)
            rejected_teacher = round_data['teacher_logp'][offsets-1]
        else:
            rejected_student = rejected_teacher = empty
        yield pool(response_student,response_teacher,rejected_student,rejected_teacher,offsets,
                   response_count=count+(uncovered if index==0 else 0))
