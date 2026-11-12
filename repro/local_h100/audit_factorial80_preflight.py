#!/usr/bin/env python3
"""Snapshot audit of finite factorial specifications and observed runtime facts.

CPU only. Does not dispatch jobs, edit the queue, or modify original results.
"""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import audit_evidence_engineering as engineering
import pandas as pd
import torch
from safetensors import safe_open


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cli(argv):
    values = {}
    for index, value in enumerate(argv):
        if value.startswith('--'):
            if value in values:
                raise ValueError('duplicate CLI option: ' + value)
            values[value] = argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith('--') else True
    return values


def audit(root):
    errors, checks = [], []
    def check(name, actual, expected=True):
        passed = actual == expected
        checks.append(dict(name=name, passed=passed, actual=actual, expected=expected))
        if not passed:
            errors.append(name)
            print('HARD_MISMATCH: ' + name, flush=True)
        return passed
    plan = read(root / 'FACTORIAL80_PROTOCOL.json')
    queue_path = root / 'control/scheduler/queue.json'
    queue_bytes = queue_path.read_bytes()
    queue = {job['id']: job for job in json.loads(queue_bytes)['jobs']}
    selected = {name: job for name, job in queue.items()
                if re.fullmatch(r'factorial80_128f_(A_plain|B_text|C_visual)_(sft|opd)_seed\d+', name)}
    expected_names = {f'factorial80_128f_{arm}_{objective}_seed{seed}'
                      for arm in plan['arms'] for objective in ('sft', 'opd') for seed in plan['training_seeds']}
    check('exact_twelve_factorial_arms', sorted(selected), sorted(expected_names))
    check('plan_budget', plan['budget']['updates_per_arm'] * plan['budget']['QA_per_update'], 320)
    base = Path(plan['common_base'])
    init = Path(plan['evidence_initialization'])
    manifest = read(init / 'INIT_MANIFEST.json')
    old = read(root / 'matched_evidence_engineering_init/INIT_MANIFEST.json')
    check('initialization_source', manifest['source'], str(base))
    check('initialization_seed', manifest['seed'], plan['added_readout_initialization_seed'])
    identities = {'A_plain': engineering.sha(base / 'model.safetensors')}
    check('actual_base_weight_sha', identities['A_plain'], plan['common_base_sha256']['model.safetensors'])
    for arm in ('B_text', 'C_visual'):
        identities[arm] = engineering.sha(init / arm / 'model.safetensors')
        check(arm + '_actual_weight_sha', identities[arm], manifest['arm_weights_sha256'][arm]['model.safetensors'])
        check(arm + '_matches_actually_gated_initial_weights', identities[arm],
              engineering.sha(root / 'matched_evidence_engineering_init' / arm / 'model.safetensors'))
        cfg = read(init / arm / 'config.json')['dflash_config']
        for key, expected in [('evidence_dim', 256), ('evidence_max_tokens', 128),
                              ('evidence_spatial_bins', 2), ('evidence_time_encoding', 'off'),
                              ('evidence_mode', 'text' if arm == 'B_text' else 'visual')]:
            check(arm + '_' + key, cfg[key], expected)
        tensors = engineering.evidence_weights(init / arm)
        check(arm + '_all_evidence_fp32', all(v.dtype == torch.float32 for v in tensors.values()))
        check(arm + '_output_zero', int(torch.count_nonzero(tensors['evidence_readout.output.weight'])), 0)
        check(arm + '_time_zero', int(torch.count_nonzero(tensors['evidence_readout.time.weight'])), 0)
    check('B_C_file_weights_identical', identities['B_text'], identities['C_visual'])
    for key in ('source_weights_sha256', 'source_config_sha256', 'added_tensor_sha256',
                'arm_weights_sha256', 'evidence_time_encoding', 'evidence_max_tokens', 'evidence_dim'):
        check('formal_init_matches_gates_' + key, manifest[key], old[key])
    with safe_open(str(base / 'model.safetensors'), framework='pt', device='cpu') as af, \
            safe_open(str(init / 'B_text/model.safetensors'), framework='pt', device='cpu') as bf:
        unchanged = all(af.get_tensor(name).dtype == bf.get_tensor(name).dtype and
                        torch.equal(af.get_tensor(name), bf.get_tensor(name)) for name in af.keys())
        check('all_shared_base_tensors_exact_in_matched_head', unchanged)
        base_tensor_count = len(af.keys())
    pool = pd.read_parquet(plan['pool'])
    check('pool_sha', engineering.sha(plan['pool']), plan['pool_sha256'])
    check('pool_count', len(pool), 320)
    check('pool_unique_tasks', pool['task'].nunique(), 320)
    check('pool_unique_sources', pool['source_sha256'].nunique(), 320)
    check('pool_all_train', sorted(pool['split'].unique().tolist()), ['train'])
    check('pool_all_frames128', sorted(pool['n_frames'].unique().tolist()), [128])
    encoded = read(root / 'ENCODE_EXPANDED_VALIDATION.json')
    encoded_train = [row for row in encoded['rows'] if row['n_frames'] == 128 and row['split'] == 'train']
    check('encoded_train_sources_match_pool', sorted(row['source_sha256'] for row in encoded_train),
          sorted(pool.source_sha256.tolist()))
    longest_encoded_input = max(row['input_ids_len'] for row in encoded_train)
    dev = pd.read_parquet(root / 'data/bounded_summary_v2/dev_128.parquet')
    hold = lines(root / 'data/expanded320/future_holdout.jsonl')
    check('train_dev_disjoint_source_sha', len(set(pool.source_sha256) & set(dev.source_sha256)), 0)
    check('train_holdout_disjoint_source_sha', len(set(pool.source_sha256) & {x['source_sha256'] for x in hold}), 0)
    check('dev_holdout_disjoint_source_sha', len(set(dev.source_sha256) & {x['source_sha256'] for x in hold}), 0)
    check('dev_count', len(dev), 40)
    check('holdout_count', len(hold), 80)
    schedules = {}
    for seed in plan['training_seeds']:
        path = root / f'control/schedule_expanded320_128_seed{seed}.jsonl'
        schedule = lines(path)
        check(f'schedule_{seed}_count', len(schedule), 320)
        check(f'schedule_{seed}_each_source_once', Counter(x['task'] for x in schedule), Counter(pool.task))
        check(f'schedule_{seed}_80_updates4slots', sorted((x['update'], x['slot']) for x in schedule),
              [(u, slot) for u in range(80) for slot in range(4)])
        schedules[seed] = dict(path=str(path), sha256=engineering.sha(path), rows=schedule)
    # Keep check records JSON-friendly; mappings were used for exact multiset comparison.
    for item in checks:
        for key in ('actual', 'expected'):
            if isinstance(item[key], Counter):
                item[key] = dict(item[key])
    gate_names = {
        ('A_plain', 'sft'): 'gpu_master_sft_fullstate_peak128_smoke2_dev2',
        ('B_text', 'sft'): 'gpu_master_evidence128_B_text_fullstate_peak2_dev2',
        ('C_visual', 'sft'): 'gpu_master_evidence128_C_visual_fullstate_peak2_dev2',
        ('A_plain', 'opd'): 'gpu_master_opd128_plain_fullstate_peak2_dev2',
        ('B_text', 'opd'): 'gpu_master_opd128_text_fullstate_peak2_dev2',
        ('C_visual', 'opd'): 'gpu_master_opd128_visual_boundary_fullstate_peak2_dev2'}
    gate_records = {}
    for gate in gate_names.values():
        definition = queue[gate]
        check(gate + '_required_status', definition['result']['status'], 'LOSSLESS_PASSED')
        exit_path, result_path = Path(definition['job_dir']) / 'EXIT.json', Path(definition['result']['path'])
        gate_records[gate] = dict(definition=definition, actual_exit=read(exit_path) if exit_path.exists() else None,
                                  actual_result_status=read(result_path)['status'] if result_path.exists() else None)
    observed = []
    for name, job in selected.items():
        arm = next(a for a in plan['arms'] if name.startswith('factorial80_128f_' + a + '_'))
        objective = 'sft' if '_sft_' in name else 'opd'
        seed = int(name.rsplit('seed', 1)[1])
        args = cli(job['argv'])
        entry = ('train_video_sft_gpu_master.py' if arm == 'A_plain' else 'train_video_evidence_sft_gpu_master.py') if objective == 'sft' else (
            'train_video_opd_gpu_memory.py' if arm == 'A_plain' else 'train_video_evidence_opd_boundary_gpu_memory.py')
        check(name + '_entry', job['argv'][2], 'repro/local_h100/' + entry)
        expected = {'--pool': plan['pool'], '--schedule': schedules[seed]['path'], '--seed': str(seed),
                    '--lr': '1e-05', '--updates': '80', '--qa-per-update': '4', '--save-every': '20',
                    '--max-new-tokens': '1024', '--max-seq-len': '16384',
                    '--init-from': str(base if arm == 'A_plain' else init / arm),
                    '--target': '/czsun/mcy/models/Qwen3-VL-8B-Instruct',
                    '--out': str(Path(job['job_dir']) / 'run')}
        if objective == 'opd':
            expected.update({'--master-weights': 'gpu', '--e2e-tv-weight': '0', '--loss-normalization': 'global_pooled',
                             '--time-mode': 'real', '--evidence-mode': {'A_plain':'plain','B_text':'text','C_visual':'visual'}[arm]})
        else:
            expected['--anchors-per-qa'] = '32'
        check(name + '_CLI', {key: args.get(key) for key in expected}, expected)
        check(name + '_sole_UUID', len(job['gpu_uuids']) == 1 and job['gpu_uuids'][0].startswith('GPU-'))
        gate = gate_names[(arm, objective)]
        check(name + '_direct_architecture_objective_gate', gate in job['depends_on'])
        check(name + '_no_resume_flag', not any('resume' in value for value in args))
        launch_path, pro_path = Path(job['job_dir']) / 'LAUNCH.json', Path(job['job_dir']) / 'run/PROTOCOL.json'
        record = dict(job=name, architecture=arm, objective=objective, seed=seed,
                      specification=job, required_gate=gate,
                      scope='SPECIFICATION_ONLY_NOT_STARTED', runtime=None)
        if launch_path.exists():
            launch = read(launch_path)
            record['scope'] = 'STARTED_WITH_RUNTIME_PROTOCOL' if pro_path.exists() else 'STARTED_PROTOCOL_PENDING'
            check(name + '_launched_argv_equals_queue', launch['argv'], job['argv'])
            check(name + '_launched_UUID_equals_queue', launch['gpu_uuids'], job['gpu_uuids'])
            check(name + '_gate_EXIT0_before_start', gate_records[gate]['actual_exit']['exit_code'], 0)
            check(name + '_gate_lossless_before_start', gate_records[gate]['actual_result_status'], 'LOSSLESS_PASSED')
            check(name + '_gate_completed_before_start', gate_records[gate]['actual_exit']['completed_utc'] <= launch['started_utc'])
        if pro_path.exists():
            pro = read(pro_path)
            check(name + '_runtime_shared_config',
                  {k:pro['config'].get(k) for k in ('updates','qa_per_update','lr','seed','init_from','schedule','pool','save_every')},
                  dict(updates=80,qa_per_update=4,lr=1e-5,seed=seed,init_from=expected['--init-from'],
                       schedule=schedules[seed]['path'],pool=plan['pool'],save_every=20))
            check(name + '_runtime_precision', [pro[k] for k in ('target_dtype','head_dtype','master_dtype')],
                  ['float32','bfloat16','float32'])
            check(name + '_runtime_optimizer', pro['optimizer'],
                  dict(name='AdamW',betas=[.9,.95],weight_decay=0.,where='gpu',**({'clip':1.} if objective=='sft' else {})))
            check(name + '_runtime_clip', pro['config']['clip'], 1.)
            check(name + '_fresh_optimizer', pro['initialization'], 'weights_only_fresh_optimizer')
            check(name + '_runtime_schedule_SHA', pro['hashes']['schedule'], schedules[seed]['sha256'])
            check(name + '_runtime_pool_SHA', pro['hashes']['pool'], plan['pool_sha256'])
            check(name + '_runtime_init_SHA', pro['hashes']['initial_head_weights']['model.safetensors'], identities[arm])
            if objective == 'opd':
                check(name + '_runtime_lambda0', pro['config']['e2e_tv_weight'], 0.)
                check(name + '_runtime_lossnorm', pro['config']['loss_normalization'], 'global_pooled')
            if arm != 'A_plain':
                check(name + '_runtime_timeoff', pro['evidence_architecture']['time_encoding'], 'off')
                check(name + '_runtime_evidence_FP32', pro['evidence_architecture']['parameter_dtype'], 'float32')
                if objective == 'opd':
                    check(name + '_boundary_source', pro['evidence_prompt_boundary']['helper_sha256'],
                          engineering.sha(Path(__file__).with_name('evidence_opd_prompt_boundary.py')))
                    check(name + '_combined_entry_source', pro['hashes']['combined_evidence_opd_entry'],
                          engineering.sha(Path(__file__).with_name(entry)))
            logpath = pro_path.with_name('log.jsonl')
            log = lines(logpath) if logpath.exists() else []
            scheduled = schedules[seed]['rows']
            for row in log:
                tasks = [item['task'] for item in row['qa']] if objective == 'sft' else row['tasks']
                check(name + '_observed_schedule_update' + str(row['update']), tasks,
                      [item['task'] for item in scheduled if item['update'] == row['update'] - 1])
            checkpoints = sorted(pro_path.parent.glob('step*/CHECKPOINT_IDENTITY.json'))
            record['runtime'] = dict(protocol_sha256=pro['protocol_sha256'], protocol_file_sha256=engineering.sha(pro_path),
                observed_completed_update_logs=len(log), last_logged_update=log[-1]['update'] if log else None,
                checkpoint_identities={str(path):read(path) for path in checkpoints},
                remaining_budget_not_yet_verified=True)
        observed.append(record)
    probe = 'gpu_master_opd128_visual_boundary_fullstate_peak2'
    probe_audit = None
    if (root / 'jobs' / probe / 'EXIT.json').exists():
        # This CPU call loads only the draft head, compares FP32 evidence exactly,
        # and preserves the failed original C_visual run as a separate identity.
        probe_audit = engineering.audit_job(root, probe)
        for update in probe_audit['updates']:
            check('visual_probe_update' + str(update['update']) + '_boundary_once_per_QA',
                  [value.get('known_visual_prefix_prepares') for value in update['evidence']['scoring']], [1]*4)
            check('visual_probe_update' + str(update['update']) + '_memory128',
                  [value['memory_tokens'] for value in update['evidence']['scoring']], [128]*4)
            check('visual_probe_update' + str(update['update']) + '_longest_input', max(update['input_tokens_per_qa']), longest_encoded_input)
        for key in ('query.weight','key.weight','value.weight','output.weight'):
            check('visual_probe_second_update_nonzero_gradient_' + key,
                  probe_audit['updates'][1]['evidence']['gradient_norms'][key] > 0)
    return dict(schema='factorial80_preflight_readonly_audit_v1',
        snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='HARD_MISMATCH_FOUND' if errors else 'SPECIFICATIONS_MATCH_RUNTIME_PARTIAL',
        hard_errors=errors, gpu_used=False, original_results_or_queue_modified=False,
        protocol_file_sha256=engineering.sha(root/'FACTORIAL80_PROTOCOL.json'),
        queue_snapshot_sha256=__import__('hashlib').sha256(queue_bytes).hexdigest(),
        shared_base_tensor_count_verified=base_tensor_count, initial_weight_sha256=identities,
        schedules={seed:{k:v for k,v in value.items() if k!='rows'} for seed,value in schedules.items()},
        checks=checks,jobs=observed,gates=gate_records,corrected_visual_probe=probe_audit,
        limitations=['Unstarted arms are specification checks, not runtime validation.',
                     'Running arms have only their logged prefix checked; no 80-step completion is implied.',
                     'GPU-master placement is enforced by verified entry and protocol; actual saved allocator/device metrics are available only at checkpoints.',
                     'SFT saves weights only, OPD saves full state; neither entry supports resume. Continuous80 does not mean restarting from saved20.',
                     'Gate and queue are snapshots; pending corrected C_visual dev2 is not a pass until EXIT0 plus LOSSLESS_PASSED.',
                     'No holdout model execution, new training, final quality, or formal timing was performed.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        parser.error("requires CUDA_VISIBLE_DEVICES='' for CPU-only audit")
    torch.set_num_threads(2)
    root=args.run_root.resolve()
    out=(args.out or root/'analysis/factorial80_preflight_audit.json').resolve()
    if out.parent!=root/'analysis' or out.exists():
        parser.error('output must be a new file directly in this run/analysis')
    result=audit(root)
    with out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(out=str(out),status=result['status'],hard_errors=result['hard_errors'])))


if __name__=='__main__':
    main()
