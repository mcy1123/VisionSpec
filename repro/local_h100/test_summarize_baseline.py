import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('summary', Path(__file__).with_name('summarize_baseline.py'))
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_windows_assign_round_by_start_and_clip_cap():
    row = dict(gen_tokens=530, anchor_rel=[0, 250, 260, 510, 520], accepted=[249, 9, 249, 9, 15])
    result = summary.acceptance_windows(row)
    assert result['0_256']['rounds_started'] == 2
    assert result['0_256']['raw_committed_sum'] == 260
    assert result['256_512']['raw_committed_sum'] == 260
    assert result['512_plus']['tau_by_round_start'] == 16
    assert result['512_plus']['clipped_tokens_per_start_round'] == 10
    assert sum(r['eos_cap_clipped_tokens_from_rounds_starting_here'] for r in result.values()) == 530


def test_timestamp_changes_do_not_hide_repeated_paragraphs():
    content = 'The person looks at the camera and smiles for a moment.'
    text = f'0:00-0:03: {content}\n\n0:04-0:07: {content}'
    result = summary.repetition(text, list(range(12))*2)
    assert result['heuristic_only_not_quality_score'] is True
    assert result['duplicate_normalized_paragraph_fraction'] == .5
    assert result['repeated_paragraph_examples'][0]['count'] == 2
    assert result['token_8gram']['duplicate_occurrence_fraction'] > 0


def test_short_outputs_have_no_undefined_ngram_score():
    result = summary.repetition('yes', [1])
    assert result['token_8gram']['duplicate_occurrence_fraction'] is None
    assert summary.stats([None])['n'] == 0


def test_cross_window_repetition_uses_only_prior_prefix():
    ids = list(range(256)) * 3
    result = summary.repetition('example', ids)
    assert result['token_windows']['0_256']['fraction_already_seen_before_window'] == 0
    assert result['token_windows']['256_512']['fraction_already_seen_before_window'] == 1


def test_validation_distinguishes_raw_tau_and_effective():
    row = dict(video='a', lossless=True, spec_ids=[1,2,3,4,5,6], ar_ids=[1,2,3,4,5,6],
               gen_tokens=6, ar_tokens=6, rounds=3, anchor_rel=[0,2,4], accepted=[1,1,2],
               tau=7/3, effective_tokens_per_round=2, spec_wall_s=1, ar_wall_s=2,
               spec_end_to_end_s=1, ar_end_to_end_s=2, input_identity={'image':'known'})
    summary.validate_row(row)
    row['tau'] = 2
    with pytest.raises(ValueError, match='tau does not match'):
        summary.validate_row(row)


def test_atomic_summary_does_not_overwrite(tmp_path):
    path = tmp_path/'summary.json'
    summary.write_exclusive(path, {'first': True})
    with pytest.raises(FileExistsError):summary.write_exclusive(path, {'first': False})
    assert json.loads(path.read_text()) == {'first': True}


def test_first_difference_reports_length_mismatch():
    assert summary.first_difference([1,2], [1,2,3]) == 2
    assert summary.first_difference([1,2], [1,3]) == 1
    assert summary.first_difference([1,2], [1,2]) is None
