#!/usr/bin/env python3
"""Read-only summary of the eight public-head train8 diagnostic evaluations.

No confidence intervals or quality scores: the same eight training videos are
reused across budgets, timing was concurrent, and repetition is only heuristic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
import uuid

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
FRAMES = (64, 128)
BUDGETS = (15, 31, 63, 127)
WINDOWS = ((0, 256, '0_256'), (256, 512, '256_512'), (512, None, '512_plus'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = [value for value in values if value is not None]
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)} if values else {'n': 0, 'mean': None, 'median': None,
                                                               'min': None, 'max': None}


def ngram_duplicates(sequence, n):
    grams = [tuple(sequence[i:i+n]) for i in range(max(0, len(sequence)-n+1))]
    return {'ngram_occurrences': len(grams), 'unique_ngrams': len(set(grams)),
            'duplicate_occurrence_fraction': (len(grams)-len(set(grams)))/len(grams) if grams else None}


def repetition(text, ids):
    # Strip only a leading timestamp range from each paragraph. This avoids
    # treating a changed timestamp as new semantic text. It does not evaluate
    # video grounding, correctness, event coverage, or answer quality.
    timestamp = re.compile(r'^\s*\d+:\d+(?::\d+)?(?:\.\d+)?\s*[-–—]\s*'
                           r'\d+:\d+(?::\d+)?(?:\.\d+)?\s*:?\s*')
    paragraphs = [timestamp.sub('', p.strip()) for p in re.split(r'\n\s*\n', text) if p.strip()]
    normalized = [' '.join(re.findall(r'\w+', p.lower())) for p in paragraphs]
    retained = [p for p in normalized if len(p.split()) >= 8]
    counts = Counter(retained)
    repeated = [{'count': count, 'normalized_preview': p[:240]} for p, count in counts.most_common()
                if count > 1][:3]
    words = re.findall(r'\w+', ' '.join(paragraphs).lower())
    token_windows = {}
    for lo, hi, name in WINDOWS:
        hi = min(hi or len(ids), len(ids))
        segment = ids[lo:hi]
        before = {tuple(ids[i:i+8]) for i in range(max(0, lo-7))}
        grams = [tuple(segment[i:i+8]) for i in range(max(0, len(segment)-7))]
        token_windows[name] = dict(ngram_duplicates(segment, 8),
                                  fraction_already_seen_before_window=(sum(g in before for g in grams)/len(grams)
                                                                     if grams else None))
    return {'heuristic_only_not_quality_score': True,
            'token_8gram': ngram_duplicates(ids, 8), 'word_4gram_after_timestamp_removal': ngram_duplicates(words, 4),
            'paragraphs_at_least_8_words': len(retained), 'unique_normalized_paragraphs': len(counts),
            'duplicate_normalized_paragraph_fraction': (len(retained)-len(counts))/len(retained) if retained else None,
            'repeated_paragraph_examples': repeated, 'token_windows': token_windows}


def acceptance_windows(row):
    total = row['gen_tokens']
    result = {}
    for lo, hi, name in WINDOWS:
        hi = min(hi or total, total)
        selected = [(p, a) for p, a in zip(row['anchor_rel'], row['accepted']) if lo <= p < hi]
        raw = [a+1 for _, a in selected]
        useful = [min(a+1, max(0, total-p)) for p, a in selected]
        result[name] = {'rounds_started': len(selected), 'output_tokens_in_window': max(0, hi-lo),
                        'raw_accepted_sum': sum(a for _, a in selected), 'raw_committed_sum': sum(raw),
                        'tau_by_round_start': statistics.mean(raw) if raw else None,
                        'eos_cap_clipped_tokens_from_rounds_starting_here': sum(useful),
                        'clipped_tokens_per_start_round': statistics.mean(useful) if useful else None}
    return result


def validate_row(row):
    if row.get('lossless') is not True or row.get('spec_ids') != row.get('ar_ids'):
        raise ValueError(f'{row.get("video")}: lossless token check failed')
    if len(row['spec_ids']) != row['gen_tokens'] or row['ar_tokens'] != row['gen_tokens']:
        raise ValueError('output token count mismatch')
    if row['rounds'] <= 0 or len(row['anchor_rel']) != row['rounds'] or len(row['accepted']) != row['rounds']:
        raise ValueError('round trace count mismatch')
    pos, accepted = row['anchor_rel'], row['accepted']
    if pos[0] != 0 or any(p < 0 or a < 0 for p, a in zip(pos, accepted)):
        raise ValueError('invalid round trace')
    if any(pos[i+1] != pos[i]+accepted[i]+1 for i in range(len(pos)-1)):
        raise ValueError('round trace is not contiguous')
    if sum(min(a+1, row['gen_tokens']-p) for p, a in zip(pos, accepted)) != row['gen_tokens']:
        raise ValueError('clipped round trace does not cover generated output exactly')
    if not math.isclose(row['tau'], statistics.mean(a+1 for a in accepted), rel_tol=1e-9):
        raise ValueError('tau does not match round trace')
    if not math.isclose(row['effective_tokens_per_round'], row['gen_tokens']/row['rounds'], rel_tol=1e-9):
        raise ValueError('effective tokens/round mismatch')
    for field in ('spec_wall_s', 'ar_wall_s', 'spec_end_to_end_s', 'ar_end_to_end_s'):
        if not math.isfinite(row[field]) or row[field] <= 0:
            raise ValueError('invalid wall time')
    if not row.get('input_identity'):
        raise ValueError('actual input identity missing')


def per_video(row):
    validate_row(row)
    fields = ('video', 'n_frames', 'prompt_len', 'n_visual_tokens', 'video_grid_thw', 'gen_tokens',
              'rounds', 'tau', 'effective_tokens_per_round', 'hit_eos', 'truncated', 'spec_wall_s',
              'ar_wall_s', 'spec_end_to_end_s', 'ar_end_to_end_s', 'spec_prefill_ms', 'ar_prefill_ms',
              'spec_decode_ms_per_tok', 'ar_decode_ms_per_tok', 'tree_size', 'warmup')
    result = {key: row[key] for key in fields}
    result.update(ar_over_spec_wall=row['ar_wall_s']/row['spec_wall_s'],
                  ar_over_spec_end_to_end=row['ar_end_to_end_s']/row['spec_end_to_end_s'],
                  spec_ms_per_round=1000*row['spec_wall_s']/row['rounds'],
                  ar_ids_sha256=digest(row['ar_ids']), input_identity=row['input_identity'],
                  acceptance_windows=acceptance_windows(row),
                  repetition=repetition(row.get('ar_text', ''), row['ar_ids']))
    return result


def summarize_run(path):
    blob = json.loads(path.read_text())
    exit_info = json.loads((path.parent / 'EXIT.json').read_text())
    if exit_info.get('exit_code') != 0 or blob.get('status') != 'LOSSLESS_PASSED':
        raise ValueError(f'{path}: wrapper/evaluation failed')
    source = blob['rows']
    if not source or len({r['video'] for r in source}) != len(source):
        raise ValueError('rows must be nonempty and unique by video')
    rows = [per_video(row) for row in source]
    frames = {row['n_frames'] for row in rows}
    if len(frames) != 1:
        raise ValueError('mixed frame counts in one evaluation')
    measured = [r for r in rows if not r['warmup']]
    if not measured:
        raise ValueError('evaluation has no measured rows after warmup exclusion')
    window_summary = {}
    for _, _, name in WINDOWS:
        parts = [r['acceptance_windows'][name] for r in rows]
        rounds = sum(r['rounds_started'] for r in parts)
        window_summary[name] = {'rounds_started': rounds,
            'macro_video_tau': stats(r['tau_by_round_start'] for r in parts if r['tau_by_round_start'] is not None),
            'pooled_tau': sum(r['raw_committed_sum'] for r in parts)/rounds if rounds else None,
            'pooled_clipped_tokens_per_start_round': sum(r['eos_cap_clipped_tokens_from_rounds_starting_here']
                                                        for r in parts)/rounds if rounds else None}
    stats_fields = ('prompt_len', 'n_visual_tokens', 'gen_tokens', 'rounds', 'tau',
                    'effective_tokens_per_round', 'spec_wall_s', 'ar_wall_s', 'spec_prefill_ms',
                    'ar_prefill_ms', 'spec_ms_per_round')
    summary = {key: stats(r[key] for r in rows) for key in stats_fields}
    n_rounds = sum(r['rounds'] for r in rows)
    summary.update(n=len(rows), lossless=len(rows), hit_eos=sum(r['hit_eos'] for r in rows),
                   truncated=sum(r['truncated'] for r in rows),
                   truncated_fraction=sum(r['truncated'] for r in rows)/len(rows),
                   pooled_tau=sum(sum(a+1 for a in r['accepted']) for r in source)/n_rounds,
                   pooled_effective_tokens_per_round=sum(r['gen_tokens'] for r in rows)/n_rounds,
                   acceptance_windows=window_summary,
                   repetition_token8_duplicate_fraction=stats(r['repetition']['token_8gram']['duplicate_occurrence_fraction']
                                                              for r in rows),
                   repetition_word4_duplicate_fraction=stats(r['repetition']['word_4gram_after_timestamp_removal']['duplicate_occurrence_fraction']
                                                             for r in rows),
                   timing={'n': len(measured), 'ar_over_spec_total_wall': sum(r['ar_wall_s'] for r in measured)/sum(r['spec_wall_s'] for r in measured),
                           'ar_over_spec_total_end_to_end': sum(r['ar_end_to_end_s'] for r in measured)/sum(r['spec_end_to_end_s'] for r in measured),
                           'per_video_wall_speedup': stats(r['ar_over_spec_wall'] for r in measured),
                           'total_spec_wall_s': sum(r['spec_wall_s'] for r in measured),
                           'total_ar_wall_s': sum(r['ar_wall_s'] for r in measured)})
    launch_path = path.parent / 'LAUNCH.json'
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
    return {'path': str(path), 'sha256': file_sha(path), 'exit_sha256': file_sha(path.parent/'EXIT.json'),
            'launch': {key: launch.get(key) for key in ('gpu_uuids', 'started_utc', 'source_commit', 'spec_sha256')},
            'frames': next(iter(frames)), 'budget': blob['config']['budget'], 'config': blob['config'],
            'protocol': blob['protocol'], 'summary': summary, 'rows': rows}, {r['video']: r for r in source}


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def build_summary(paths):
    summaries, originals = {}, {}
    for path in paths:
        summary, source = summarize_run(path)
        key = f'{summary["frames"]}f_b{summary["budget"]}'
        if key in summaries:raise ValueError('duplicate frame/budget condition')
        summaries[key], originals[key] = summary, source
    matching, pairs = {}, []
    for frames in sorted({s['frames'] for s in summaries.values()}):
        keys = sorted((k for k, s in summaries.items() if s['frames'] == frames), key=lambda k: summaries[k]['budget'])
        reference = keys[0]
        video_ids = sorted(originals[reference])
        comparison = []
        for key in keys[1:]:
            same_video_set = sorted(originals[key]) == video_ids
            protocol_equal = all(summaries[key]['config'].get(field) == summaries[reference]['config'].get(field)
                                 for field in ('max_new', 'branch', 'dtype', 'time_mode', 'eos_stop', 'head', 'target'))
            artifacts_equal = all(summaries[key]['protocol']['artifacts'].get(field) ==
                                  summaries[reference]['protocol']['artifacts'].get(field)
                                  for field in ('head_weights_sha256', 'head_config_sha256', 'target_config_sha256', 'source_sha256'))
            rows = []
            for video in sorted(set(video_ids) & set(originals[key])):
                a, b = originals[reference][video], originals[key][video]
                rows.append({'video': video, 'same_actual_input': a['input_identity'] == b['input_identity'],
                             'ar_ids_identical': a['ar_ids'] == b['ar_ids'],
                             'first_ar_mismatch': first_difference(a['ar_ids'], b['ar_ids'])})
            comparison.append({'reference': reference, 'candidate': key, 'same_video_set': same_video_set,
                               'same_protocol_except_budget': protocol_equal, 'same_artifact_identity': artifacts_equal,
                               'all_same_input_and_ar': same_video_set and protocol_equal and artifacts_equal
                                  and all(r['same_actual_input'] and r['ar_ids_identical'] for r in rows), 'rows': rows})
        matching[str(frames)] = comparison
        for base, candidate in itertools.combinations(keys, 2):
            rows = []
            for video in sorted(set(originals[base]) & set(originals[candidate])):
                a, b = originals[base][video], originals[candidate][video]
                valid = a['input_identity'] == b['input_identity'] and a['ar_ids'] == b['ar_ids']
                rows.append({'video': video, 'matched_input_and_ar': valid,
                             'delta_tau': b['tau']-a['tau'],
                             'delta_effective_tokens_per_round': b['effective_tokens_per_round']-a['effective_tokens_per_round'],
                             'base_spec_wall_over_candidate': a['spec_wall_s']/b['spec_wall_s'],
                             'base_spec_end_to_end_over_candidate': a['spec_end_to_end_s']/b['spec_end_to_end_s'],
                             'base_ar_wall_over_candidate': a['ar_wall_s']/b['ar_wall_s']})
            pairs.append({'base': base, 'candidate': candidate, 'n': len(rows),
                          'valid_matching': all(r['matched_input_and_ar'] for r in rows),
                          'delta_tau': stats(r['delta_tau'] for r in rows),
                          'delta_effective_tokens_per_round': stats(r['delta_effective_tokens_per_round'] for r in rows),
                          'per_video_spec_wall_ratio': stats(r['base_spec_wall_over_candidate'] for r in rows), 'rows': rows})
    return {'schema': 'public_baseline_train8_diagnostic_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'diagnostic_only': True, 'split': 'train8', 'n_unique_videos': len(set().union(*(set(x) for x in originals.values()))),
            'config_count': len(summaries), 'limitations': [
                'Eight training videos, reused across budgets; no independent confirmation or generalization claim.',
                'No bootstrap, confidence interval or significance claim is produced.',
                'Jobs ran concurrently on different GPUs; timing is exploratory and not balanced same-GPU benchmarking.',
                'Raw tau includes accepted+1 before EOS/cap; effective tokens/round uses final returned output.',
                'Acceptance windows assign a whole verification round by its starting output position; boundary-crossing tokens stay with that round.',
                'No per-round timings exist, so acceptance windows do not imply per-window wall-clock speed.',
                'Repetition counts exact token/word ngrams and timestamp-normalized paragraphs only; these are not grounding, correctness or quality scores.',
                'SPEC prefill is a startup window including its first draft; it is not directly comparable to pure AR target prefill.'
            ], 'same_frame_cross_budget_ar_checks': matching, 'paired_budget_comparisons': pairs, 'runs': summaries}


def write_exclusive(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    try:os.link(tmp, path)
    finally:tmp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args()
    root = args.run_root.resolve()
    out = (args.out or root/'analysis/public_baseline_summary.json').resolve()
    if not out.is_relative_to(root/'analysis') or out.exists():
        parser.error('out must be a new file under this run/analysis')
    if not 0.1 <= args.poll_seconds <= 30:
        parser.error('poll-seconds must be within 0.1..30')
    expected = [root/'jobs'/f'public_baseline_{f}f_b{b}_train8'/'result.json' for f in FRAMES for b in BUDGETS]
    while True:
        missing = [p for p in expected if not p.exists() or not (p.parent/'EXIT.json').exists()]
        if not missing:break
        print(json.dumps({'status': 'WAITING', 'completed': len(expected)-len(missing),
                          'pending': [p.parent.name for p in missing]}), flush=True)
        if not args.wait:raise SystemExit(2)
        time.sleep(args.poll_seconds)
    result = build_summary(expected)
    write_exclusive(out, result)
    print(json.dumps({'status': 'COMPLETE', 'out': str(out), 'config_count': result['config_count'],
                      'n_unique_videos': result['n_unique_videos']}), flush=True)
    for key, run in result['runs'].items():
        s = run['summary']
        print(json.dumps({'condition': key, 'tau': s['tau']['mean'], 'effective': s['effective_tokens_per_round']['mean'],
                          'speedup_wall': s['timing']['ar_over_spec_total_wall'], 'truncated': s['truncated'], 'n': s['n'],
                          'window_tau': {k: v['macro_video_tau']['mean'] for k,v in s['acceptance_windows'].items()}}), flush=True)


if __name__ == '__main__':main()
