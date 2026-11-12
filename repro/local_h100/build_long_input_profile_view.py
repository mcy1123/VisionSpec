#!/usr/bin/env python3
"""CPU-only fixed-spatial text/processor view of existing long-input pixels."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from local_h100.prepare_data import atomic_json, require_local
from local_h100.build_prompt_view import exclusive_json
from local_h100.processor_profile_runtime import (sha256, load_profile, apply_profile,
                                                  validate_encoded)

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
PROCESSOR = PROFILE = None


def initialize(model, profile_path):
    global PROCESSOR, PROFILE
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    PROFILE = load_profile(profile_path)
    PROCESSOR = AutoProcessor.from_pretrained(model, local_files_only=True)
    apply_profile(PROCESSOR, PROFILE)


def encode(row):
    from qa_opd_fair.videoencode import encode_inputs
    from evaluation.eval_video_opd import encoded_identity
    inputs = encode_inputs(PROCESSOR, row, 'real')
    metric = validate_encoded(PROCESSOR, row, inputs, PROFILE)
    metric.update(split=row['split'], input_identity=encoded_identity(inputs),
                  input_plus_1024=metric['input_ids_len'] + 1024,
                  cache_sha256=row['cache_hash'], source_sha256=row['source_sha256'])
    row.update(input_ids_len=metric['input_ids_len'],
               visual_token_count=metric['visual_token_count'])
    return row, metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT/'data/long_input_probe'))
    parser.add_argument('--out', default=str(ROOT/'data/long_input_probe_fixed_spatial'))
    parser.add_argument('--model', default='/czsun/mcy/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('CPU workers must be 1..4')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    source, out = map(require_local, (args.source, args.out))
    if out.exists():
        raise FileExistsError(out)
    parent = json.loads((source/'PROTOCOL.json').read_text())
    if not parent['all_saved_nested_pixels_verified'] or parent['source_count'] != 10:
        raise ValueError('requires completed ten-source nested-pixel engineering dataset')
    original_config = Path(args.model).resolve()/'video_preprocessor_config.json'
    original = json.loads(original_config.read_text())
    if original['size'] != {'shortest_edge': 4096, 'longest_edge': 25165824}:
        raise ValueError('original processor size differs from the audited starting config')
    out.mkdir(parents=True)
    profile_path = out/'PROCESSOR_PROFILE.json'
    exclusive_json(profile_path, {
        'schema': 'qwen3_vl_fixed_spatial_engineering_v1',
        'video_processor_class': 'Qwen3VLVideoProcessor',
        'video_size': {'shortest_edge': 4096, 'longest_edge': 50331648},
        'patch_size': 16, 'merge_size': 2, 'temporal_patch_size': 2,
        'expected_spatial_grid_hw': [14, 24], 'frame_counts': [256, 512],
        'do_sample_frames': False,
        'original_video_preprocessor_config': str(original_config),
        'original_video_preprocessor_config_sha256': sha256(original_config),
        'size_semantics': 'Qwen3VL smart_resize uses longest_edge as total T*H*W pixel cap',
        'purpose': 'keep identical 14x24 patch grid at 256/512 frames of the audited 384x216 videos',
        'runtime_entry': 'repro/local_h100/eval_long_input.py --processor-profile THIS_FILE',
        'default_encode_inputs_does_not_apply_profile': True,
        'engineering_only': True, 'dev_reused_not_independent': True,
        'output_task': 'bounded three-part summary; no thousand-token output claim',
    })
    profile = load_profile(profile_path)
    inputs = []
    for n in (256, 512):
        old = [json.loads(line) for line in (source/f'manifest_{n}.jsonl').read_text().splitlines()]
        if len(old) != 10 or sum(r['split'] == 'train' for r in old) != 8:
            raise ValueError('source view must have eight train and two reused dev')
        for row in old:
            inputs.append(dict(row, processor_profile=str(profile_path),
                               processor_profile_sha256=profile['sha256'],
                               input_ids_len=None, visual_token_count=None))
    started = time.time()
    state = {'phase': 'cpu_encoding', 'conditions': 20, 'completed': 0,
             'new_pixel_cache_bytes': 0, 'workers': args.workers}
    atomic_json(out/'STATUS.json', state)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(args.model, str(profile_path))) as pool:
            for future in as_completed([pool.submit(encode, row) for row in inputs]):
                results.append(future.result())
                state.update(completed=len(results), elapsed_s=time.time()-started)
                atomic_json(out/'STATUS.json', state)
                print(json.dumps(state), flush=True)
        import pandas as pd
        metrics = sorted([metric for _, metric in results], key=lambda r:(r['n_frames'], r['video']))
        hashes = {}
        for n in (256, 512):
            records = sorted([row for row, _ in results if row['n_frames'] == n], key=lambda r:r['video_id'])
            for name, subset in ((f'manifest_{n}', records),
                                 (f'train_{n}', [r for r in records if r['split'] == 'train']),
                                 (f'dev_{n}', [r for r in records if r['split'] == 'dev'])):
                path = out/f'{name}.jsonl'
                with path.open('x') as stream:
                    for row in subset:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                pd.DataFrame(subset).to_parquet(out/f'{name}.parquet', index=False)
                hashes[path.name] = sha256(path)
                hashes[f'{name}.parquet'] = sha256(out/f'{name}.parquet')
        maximum = max(row['input_plus_1024'] for row in metrics)
        if maximum > 32768:
            raise ValueError('measured input + 1024 exceeds the planned 32768 engineering envelope')
        protocol = dict(parent, schema='engineering_long_input_fixed_spatial_v1',
                        parent_data=str(source), parent_protocol_sha256=sha256(source/'PROTOCOL.json'),
                        processor_profile=profile, view_files_sha256=hashes,
                        profile_view_source_sha256=sha256(__file__), new_pixel_cache_bytes=0,
                        parent_metadata_and_pixels_unchanged=True,
                        cpu_actual_encoding_conditions=20, maximum_input_plus_1024=maximum,
                        gpu_memory_or_losslessness_tested=False,
                        runtime_requires_explicit_processor_profile=True,
                        elapsed_s=time.time()-started)
        protocol['limitations'] = [
            'Engineering input-length probe with reused development sources, not an independent test.',
            'GPU peak memory and exact SPEC=AR must be validated separately, 256 before 512.',
            '512-frame inputs exceed 16384; no default max-sequence length is assumed.',
            'All output is bounded summary with natural EOS and max-new 1024; no long-output claim.',
            'Use the explicit profile runtime; generic videoencode does not read row profile metadata.']
        exclusive_json(out/'PROTOCOL.json', protocol)
        exclusive_json(out/'ENCODE_VALIDATION.json', {
            'schema': 'engineering_long_input_fixed_spatial_encoding_v1', 'cpu_only': True,
            'processor_profile': profile, 'n_conditions': 20, 'rows': metrics,
            'maximum_input_plus_1024': maximum, 'all_fit_32768_with_1024': True,
            'actual_processor_size': profile['profile']['video_size'],
            'frames_and_splits_unchanged': True, 'new_pixel_cache_bytes': 0})
        state.update(phase='completed', elapsed_s=time.time()-started,
                     maximum_input_plus_1024=maximum)
        atomic_json(out/'STATUS.json', state)
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc), elapsed_s=time.time()-started)
        atomic_json(out/'STATUS.json', state)
        raise


if __name__ == '__main__':
    main()
