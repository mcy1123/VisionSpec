import copy
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import summarize_warm_dev as summary


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def make_pool(root, frames):
    path = root / f'data/bounded_summary_v2/dev_{frames}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(video_name=f'video{i}', source_sha256=sha(i), split='dev', n_frames=frames,
                       cache_hash=sha((i, frames))) for i in range(40)]).to_parquet(path)
    return summary.pool_identity(path, frames)


def row(i, frames):
    return dict(video=f'video{i}', lossless=True, spec_ids=[1, 2, 3, 4], ar_ids=[1, 2, 3, 4],
                gen_tokens=4, ar_tokens=4, rounds=2, anchor_rel=[0, 2], accepted=[1, 1],
                tau=2., effective_tokens_per_round=2., spec_wall_s=1., ar_wall_s=2.,
                spec_end_to_end_s=1.1, ar_end_to_end_s=2.1,
                input_identity={key: sha((i, frames, key)) for key in ('input_ids', 'pixel_values_videos', 'video_grid_thw')},
                timeline={'cache_sha256': sha((i, frames))}, hit_eos=True, ar_hit_eos=True,
                truncated=False, n_frames=frames, warmup=False, prompt_len=100, n_visual_tokens=80)


def evaluation(path, pool, frames, indices, shard=0, shards=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = dict(status='LOSSLESS_PASSED', config=dict(rows=pool['path'], ar=True, warmup=0,
                    eos_stop='immediate', max_new=1024, shards=shards, shard=shard,
                    target='target', head='head', dtype='float32', budget=63, branch=8),
                protocol=dict(rows_sha256=pool['sha256'], lossless_checked=True,
                    artifacts={'head_weights_sha256': {'weight': sha('public')},
                               'head_config_sha256': sha('config'), 'target_config_sha256': sha('target'),
                               'source_sha256': {'code': sha('code')}}, model_contract={'hidden_size': 32}),
                rows=[row(i, frames) for i in indices])
    path.write_text(json.dumps(blob))
    (path.parent / 'EXIT.json').write_text(json.dumps({'exit_code': 0}))
    return blob


def baselines(root):
    pools = {}
    for frames in (64, 128):
        pool = make_pool(root, frames); pools[frames] = pool
        for shard in range(4):
            evaluation(root / 'jobs' / f'bounded_v2_public_dev40_{frames}f_shard{shard}' / 'result.json',
                       pool, frames, range(shard, 40, 4), shard=shard)
    return pools


def test_partial_includes_only_complete_lossless_matched_arms(tmp_path):
    pools = baselines(tmp_path)
    name = 'bounded_v2_warm20_64f_lr1e-05_seed20261112_dev40'
    path = tmp_path / 'jobs' / name / 'result.json'
    evaluation(path, pools[64], 64, range(40), shards=1)
    result = summary.build_summary(tmp_path, resamples=1000)
    assert result['status'] == 'PARTIAL' and result['complete_arms'] == 1
    assert len(result['pending']) == 7 and not result['invalid']
    arm = result['arms'][name]
    assert arm['paired_vs_public']['delta_tau']['ci95_percentile'] == [0., 0.]
    assert arm['summary']['teacher_natural_eos_count'] == 40
    assert arm['paired_vs_public']['all_input_tensor_sha_and_ar_ids_exact']


@pytest.mark.parametrize('change', ['input', 'ar', 'missing', 'spec', 'pool_sha', 'exit'])
def test_invalid_completed_arm_is_not_compared(tmp_path, change):
    pools = baselines(tmp_path)
    path = tmp_path / 'jobs/bounded_v2_warm20_64f_lr1e-05_seed20261112_dev40/result.json'
    blob = evaluation(path, pools[64], 64, range(40), shards=1)
    if change == 'input': blob['rows'][0]['input_identity']['input_ids'] = sha('different')
    if change == 'ar':
        blob['rows'][0]['ar_ids'][0] = 99
        blob['rows'][0]['spec_ids'][0] = 99
    if change == 'missing': blob['rows'].pop()
    if change == 'spec': blob['rows'][0]['spec_ids'][0] = 99
    if change == 'pool_sha': blob['protocol']['rows_sha256'] = sha('wrong')
    if change == 'exit': (path.parent / 'EXIT.json').write_text('{"exit_code":2}')
    path.write_text(json.dumps(blob))
    result = summary.build_summary(tmp_path, resamples=1000)
    assert result['status'] == 'VALIDATION_FAILED'
    assert result['complete_arms'] == 0 and len(result['invalid']) == 1


def test_result_without_exit_remains_pending(tmp_path):
    pools = baselines(tmp_path)
    path = tmp_path / 'jobs/bounded_v2_warm20_64f_lr1e-05_seed20261112_dev40/result.json'
    evaluation(path, pools[64], 64, range(40), shards=1)
    (path.parent / 'EXIT.json').unlink()
    result = summary.build_summary(tmp_path, resamples=1000)
    assert result['status'] == 'PARTIAL' and result['complete_arms'] == 0
    assert len(result['pending']) == 8


def test_duplicate_baseline_shards_and_source_pool_fail(tmp_path):
    pools = baselines(tmp_path)
    path = tmp_path / 'jobs/bounded_v2_public_dev40_64f_shard1/result.json'
    evaluation(path, pools[64], 64, range(0, 40, 4), shard=1)
    result = summary.build_summary(tmp_path, resamples=1000)
    assert any('overlap' in row.get('error', '') for row in result['invalid'])


def test_bootstrap_resamples_paired_video_deltas_reproducibly():
    values = [-1, 0, 1, 2] * 10
    result = summary.paired_bootstrap(values, resamples=1000)
    assert result == summary.paired_bootstrap(values, resamples=1000)
    assert result['mean'] == .5 and result['unit'] == 'paired source video'
    assert result['wins'] == 20 and result['ties'] == 10 and result['losses'] == 10
    assert result['ci95_percentile'][0] < .5 < result['ci95_percentile'][1]


def test_natural_bonus_eos_includes_the_initial_prefill_root():
    value = row(0, 64)
    value.update(spec_ids=[1, 2, 3, 4, 5], ar_ids=[1, 2, 3, 4, 5],
                 gen_tokens=5, ar_tokens=5, effective_tokens_per_round=2.5)
    summary.validate_row(value)
    assert value['effective_tokens_per_round'] == value['tau'] + 1 / value['rounds']
    value.update(spec_ids=[1, 2, 3, 4, 5, 6], ar_ids=[1, 2, 3, 4, 5, 6],
                 gen_tokens=6, ar_tokens=6, effective_tokens_per_round=3.)
    with pytest.raises(ValueError, match='prefill root'):
        summary.validate_row(value)


def test_only_derived_summary_can_be_atomically_replaced(tmp_path):
    path = tmp_path / 'analysis/warm_dev_summary.json'
    summary.write_summary(path, {'schema': summary.SCHEMA, 'complete_arms': 1}, tmp_path)
    before = summary.file_sha(path)
    summary.write_summary(path, {'schema': summary.SCHEMA, 'complete_arms': 2}, tmp_path)
    saved = json.loads(path.read_text())
    assert saved['previous_summary_sha256'] == before and saved['complete_arms'] == 2
    with pytest.raises(ValueError, match='under'):
        summary.write_summary(tmp_path / 'jobs/result.json', {'schema': summary.SCHEMA}, tmp_path)
    path.write_text('{"original_artifact":true}')
    with pytest.raises(ValueError, match='non-summary'):
        summary.write_summary(path, {'schema': summary.SCHEMA}, tmp_path)
