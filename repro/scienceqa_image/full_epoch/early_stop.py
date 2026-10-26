"""Paired early stopping at common evaluated checkpoints, persisted in the run."""
import math


def initial(scores, patience=3, min_delta=0.01):
    return dict(patience=patience, min_delta=min_delta, last_update=0, stop=False,
                arms={arm:dict(best=score,best_update=0,bad_evaluations=0)
                      for arm,score in scores.items()}, history=[])


def observe(state, update, scores):
    """Return an updated copy; a checkpoint already recorded is never counted twice."""
    import copy
    result=copy.deepcopy(state)
    if update<=result['last_update']:
        return result
    stalled=[]
    for arm,score in scores.items():
        record=result['arms'][arm]
        # Invalid/non-lossless metrics are recorded, never counted as improvement
        # or evidence of a plateau. Training can continue to the planned epoch.
        if score is None or not math.isfinite(score):
            record['bad_evaluations']=0
            continue
        if record['best'] is None or score>record['best']+result['min_delta']:
            record.update(best=score,best_update=update,bad_evaluations=0)
        else:
            record['bad_evaluations']+=1
        if record['bad_evaluations']>=result['patience']:stalled.append(arm)
    result.update(last_update=update,stop=bool(stalled),trigger_arms=stalled)
    result['history'].append(dict(update=update,scores=scores,trigger_arms=stalled))
    return result
