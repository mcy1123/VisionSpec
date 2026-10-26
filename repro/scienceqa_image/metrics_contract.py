"""Boundary-aware local diagnostics; never convert these into online tau."""

def classify_lengths(l_raw, l_tree, horizon):
    if not 0 <= l_tree <= l_raw <= horizon:
        raise ValueError("expected 0 <= L_tree <= L_raw <= horizon")
    if l_tree < l_raw:
        return "in_topk_not_in_tree", l_tree + 1
    if l_raw < horizon:
        return "not_in_topk", l_raw + 1
    return "horizon_end", horizon + 1


def correct_saved_anchor(row):
    """Recover only quantities evidenced by stored trace; do not invent a path."""
    out = dict(row)
    cause, depth = classify_lengths(row['L_raw'], row['L_tree'], row['horizon'])
    out['historical_first_blocker'] = row['first_blocker']
    out['historical_L_actual'] = row['L_actual']
    out['first_blocker'], out['first_blocker_depth'] = cause, depth
    # Stored correct-path intersection excludes post-EOS/horizon continuation.
    actual = row.get('L_correct_in_actual_path')
    if actual is None:
        out['L_actual'] = None
        out['actual_status'] = 'unavailable: missing stored verified correct path'
    else:
        if not 0 <= actual <= row['L_tree']:
            raise ValueError('verified correct path exceeds candidate tree path')
        out['L_actual'] = actual
        out['actual_status'] = 'recovered_from_stored_verified_correct_path'
    out['online_anchor_distribution'] = 'unavailable'
    return out


def answer_metrics(rows):
    n = len(rows)
    parsed = sum(bool(x.get('answer_parse_ok', x.get('pred_letter') is not None)) for x in rows)
    correct = sum(bool(x['answer_correct']) for x in rows)
    return dict(correct=correct, parsed=parsed, total=n,
                parsed_only=correct / parsed if parsed else None,
                all_questions=correct / n if n else None,
                parse_rate=parsed / n if n else None)
