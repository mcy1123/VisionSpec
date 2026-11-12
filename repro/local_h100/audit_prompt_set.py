#!/usr/bin/env python3
"""Native-greedy prompt audit on one train video; never trains or changes pools."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()
from local_h100 import audit_native_generation as native
from local_h100.summarize_baseline import write_exclusive


def validate_prompts(value):
    if not isinstance(value, list) or not value:
        raise ValueError('prompts JSON must be a nonempty list of {label,prompt}')
    seen, result = set(), []
    for row in value:
        if not isinstance(row, dict) or set(row) != {'label', 'prompt'}:
            raise ValueError('each prompt entry must contain exactly label and prompt')
        label, prompt = row['label'], row['prompt']
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', label):
            raise ValueError('labels must be short filename-safe strings')
        if label in seen:
            raise ValueError('prompt labels must be unique')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt text must be nonempty')
        seen.add(label)
        result.append({'label': label, 'prompt': prompt})
    return result


def validate_visual_identity(reference, candidate):
    for key in ('pixel_values_videos', 'video_grid_thw'):
        if reference['encoded_sha256'][key] != candidate['encoded_sha256'][key]:
            raise ValueError(f'prompt variation changed visual input: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--row-index', type=int, required=True)
    parser.add_argument('--prompts-json', type=Path, required=True)
    parser.add_argument('--max-new', type=int, default=1536)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true', help='CPU processor validation, no target weights or CUDA generation')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or not out.is_relative_to(native.RUN_ROOT.resolve()):
        parser.error('out must be a new path beneath the authorized local_h100 run')
    if args.row_index < 0 or args.max_new <= 0:
        parser.error('row-index must be nonnegative and max-new must be positive')
    gpu = native.single_uuid()
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    pool_sha, prompts_sha = native.sha(args.pool), native.sha(args.prompts_json)
    cases = validate_prompts(json.loads(args.prompts_json.read_text()))
    import pandas as pd
    import torch
    import transformers
    from shared import common
    from evaluation.eval_video_opd import validate_rows

    frame = pd.read_parquet(args.pool)
    if args.row_index >= len(frame):
        parser.error('row-index exceeds pool length')
    selected = frame.iloc[[args.row_index]]
    row = selected.iloc[0].to_dict()
    native.prompts(row)  # Reuse the explicit train-only/original-prompt guard.
    timeline = validate_rows(selected, 'real', require_exact_time=True)[0]
    target = None
    if args.preflight_only:
        processor = transformers.AutoProcessor.from_pretrained(str(args.target), local_files_only=True)
    else:
        torch.cuda.set_device(0)
        target, processor = common.load_target(path=str(args.target), device='cuda:0', dtype=torch.float32)
    eos = processor.tokenizer.eos_token_id
    if eos is None:raise ValueError('tokenizer EOS must be explicit')
    pad = processor.tokenizer.pad_token_id
    pad = eos if pad is None else pad
    config = native.generation_config(eos, pad, args.max_new)
    result = {'schema': 'native_prompt_set_audit_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'pool': str(args.pool.resolve()), 'pool_sha256': pool_sha,
              'prompts_json': str(args.prompts_json.resolve()), 'prompts_sha256': prompts_sha,
              'row_index': args.row_index, 'video': str(row['video_name']), 'split': row['split'],
              'original_pool_prompt': row.get('question', row.get('prompt')),
              'n_frames': int(row['n_frames']), 'gpu_uuid': gpu, 'timeline': timeline,
              'protocol': {'target': str(args.target.resolve()), 'target_dtype': 'float32',
                           'target_config_sha256': native.sha(args.target/'config.json'),
                           'target_verified_manifest_sha256': native.sha(args.target/'download_manifest.json') if (args.target/'download_manifest.json').exists() else None,
                           'production_fast_patch_embed': True, 'independent_vision_encoder_validation': False,
                           'native_decoder_only_no_repo_comparison_in_this_job': True,
                           'generation_config': config.to_dict(), 'use_model_defaults': False,
                           'sampling': False, 'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0,
                           'minimum_output_length_added': False, 'rope_deltas_reset_before_every_generation': True,
                           'max_new_tokens': args.max_new, 'torch_version': torch.__version__,
                           'transformers_version': transformers.__version__,
                           'source_sha256': {str(p.relative_to(REPRO.parent)): native.sha(p) for p in
                                             (Path(__file__).resolve(), Path(native.__file__), Path(common.__file__),
                                              REPRO/'local_h100/summarize_baseline.py', REPRO/'qa_opd_fair/videoencode.py')},
                           'limitations': ['Train-video exploratory prompt diagnostic, not an independent quality evaluation.',
                                           'Decoder adds no minimum output length or repetition penalty; prompt text is recorded verbatim.',
                                           'Repeated ngrams/paragraphs are heuristic observations, not grounding or quality scores.',
                                           'Same production fast-patched vision encoder; no independent encoder verification.',
                                           'Timings are diagnostic and not balanced benchmarking.']},
              'prompt_cases': []}
    reference = None
    for case in cases:
        cpu_inputs, identity = native.encode_case(processor, row, case['prompt'])
        if reference is None:reference = identity
        else:validate_visual_identity(reference, identity)
        record = dict(case, input_identity=identity)
        if not args.preflight_only:
            inputs = {k: v.to('cuda:0') if torch.is_tensor(v) else v for k, v in cpu_inputs.items()}
            torch.cuda.synchronize();start = time.perf_counter()
            with torch.inference_mode():
                output = native.native_generate(target, inputs, native.generation_config(eos, pad, args.max_new))
            torch.cuda.synchronize()
            record['generation'] = native.record(output, inputs['input_ids'], processor.tokenizer,
                                                  eos, args.max_new, time.perf_counter()-start)
            print(json.dumps({'phase': 'prompt', 'label': case['label'],
                              'gen_tokens': record['generation']['gen_tokens'],
                              'hit_eos': record['generation']['hit_eos'],
                              'truncated': record['generation']['truncated']}), flush=True)
            del inputs, output
        result['prompt_cases'].append(record)
    if native.sha(args.pool) != pool_sha or native.sha(args.prompts_json) != prompts_sha:
        raise ValueError('pool or prompts JSON changed during audit')
    result.update(status='PREFLIGHT_PASSED' if args.preflight_only else 'PROMPT_SET_COMPLETED',
                  gpu_generation_performed=not args.preflight_only, same_visual_inputs_across_prompts=True)
    write_exclusive(out, result)
    print(json.dumps({'status': result['status'], 'out': str(out)}), flush=True)


if __name__ == '__main__':main()
