#!/usr/bin/env python3
"""Compare production AR with native greedy generation; inspect three prompts.

Uses the same production fp32 target including common.use_fast_patch_embed.
This checks decoding paths, not an independently implemented vision encoder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

from local_h100.summarize_baseline import repetition, first_difference, write_exclusive

RUN_ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
EVENT_PROMPT = ('Describe at most eight distinct main events visible in this video, in chronological order. '
                'For each event, give an approximate time range and one concise sentence about the visible action. '
                'Combine repeated views of the same event. Stop after covering the distinct events; '
                'use fewer than eight when appropriate. Do not infer dialogue or unseen events.')
SUMMARY_PROMPT = ('Summarize what is visibly happening at the beginning, middle, and end of this video. '
                  'Use one concise paragraph for each part, focusing on the main actions and visible changes. '
                  'Avoid repeating the same event and do not infer dialogue or unseen events.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def single_uuid(value=None):
    value = os.environ.get('CUDA_VISIBLE_DEVICES', '') if value is None else value
    if not value.startswith('GPU-') or ',' in value or any(c.isspace() for c in value):
        raise ValueError('exactly one explicit GPU UUID must be set in CUDA_VISIBLE_DEVICES')
    return value


def prompts(row):
    if row.get('split') != 'train':
        raise ValueError('native generation diagnostic must select a train video')
    original = row.get('question', row.get('prompt'))
    if not isinstance(original, str) or not original.strip():
        raise ValueError('selected row has no original prompt')
    return [('original', original), ('at_most_8_events', EVENT_PROMPT), ('beginning_middle_end', SUMMARY_PROMPT)]


def generation_config(eos, pad, maximum):
    from transformers import GenerationConfig
    return GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                            repetition_penalty=1.0, no_repeat_ngram_size=0,
                            eos_token_id=int(eos), pad_token_id=int(pad),
                            max_new_tokens=int(maximum), use_cache=True)


def reset_rope(target):
    if not hasattr(target, 'model') or not hasattr(target.model, 'rope_deltas'):
        raise ValueError('expected production Qwen-VL target.model.rope_deltas')
    target.model.rope_deltas = None


def native_generate(target, inputs, config):
    reset_rope(target)
    # False is essential in Transformers >=4.50: otherwise default-valued
    # fields in this fresh config can be replaced by the model's sampling config.
    return target.generate(**inputs, generation_config=config, use_model_defaults=False)


def encode_case(processor, row, prompt):
    from qa_opd_fair.videoencode import build_video_text, encode_inputs
    from evaluation.eval_video_opd import encoded_identity
    copy = dict(row)
    copy.update(question=prompt, prompt=prompt)
    encoded = encode_inputs(processor, copy, 'real', text=build_video_text(processor, prompt))
    identity = encoded_identity(encoded)
    layouts = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in encoded.items() if hasattr(v, 'shape')}
    return encoded, {'encoded_sha256': identity, 'encoded_layouts': layouts,
                     'prompt_tokens': int(encoded['input_ids'].shape[1]),
                     'visual_tokens': int((encoded['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')).sum()),
                     'video_grid_thw': encoded['video_grid_thw'].tolist()}


def record(output, input_ids, tokenizer, eos, maximum, elapsed):
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError('expected one generated sequence')
    prefix = input_ids[0].tolist()
    if output[0, :len(prefix)].tolist() != prefix:
        raise ValueError('generation did not preserve encoded input prefix')
    ids = output[0, len(prefix):].tolist()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return {'gen_ids': ids, 'text': text, 'gen_tokens': len(ids), 'wall_s': elapsed,
            'max_new_tokens': maximum, 'hit_eos': bool(ids and ids[-1] == eos),
            'eos_positions': [i for i, token in enumerate(ids) if token == eos],
            'truncated': len(ids) >= maximum and not (ids and ids[-1] == eos),
            'repetition': repetition(text, ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--row-index', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true', help='validate/encode all prompts on CPU; no target weights loaded')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or not out.is_relative_to(RUN_ROOT.resolve()):
        parser.error('out must be new and beneath the authorized local_h100 run')
    if args.row_index < 0:
        parser.error('row-index must be nonnegative')
    gpu = single_uuid()
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    import pandas as pd
    import torch
    import transformers
    from shared import common
    from evaluation.eval_video_opd import validate_rows
    from glance.decode import greedy_generate
    from glance.vl import VisionPositions

    pool_hash = sha(args.pool)
    frame = pd.read_parquet(args.pool)
    if args.row_index >= len(frame):
        parser.error('row-index exceeds pool length')
    selected = frame.iloc[[args.row_index]]
    row = selected.iloc[0].to_dict()
    cases = prompts(row)
    timeline = validate_rows(selected, 'real', require_exact_time=True)[0]
    target = None
    if args.preflight_only:
        processor = transformers.AutoProcessor.from_pretrained(str(args.target), local_files_only=True)
    else:
        torch.cuda.set_device(0)
        target, processor = common.load_target(path=str(args.target), device='cuda:0', dtype=torch.float32)
    eos = processor.tokenizer.eos_token_id
    if eos is None:raise ValueError('tokenizer EOS token must be explicit')
    pad = processor.tokenizer.pad_token_id
    pad = eos if pad is None else pad
    encoded_cases = []
    for label, prompt in cases:
        inputs, identity = encode_case(processor, row, prompt)
        if encoded_cases:
            for key in ('pixel_values_videos', 'video_grid_thw'):
                if identity['encoded_sha256'][key] != encoded_cases[0][3]['encoded_sha256'][key]:
                    raise ValueError('prompt variation changed visual input')
        encoded_cases.append((label, prompt, inputs, identity))
    protocol = {'target': str(args.target.resolve()), 'target_dtype': 'float32',
                'target_config_sha256': sha(args.target/'config.json'),
                'target_weight_identity': json.loads((args.target/'download_manifest.json').read_text()) if (args.target/'download_manifest.json').exists() else None,
                'production_fast_patch_embed': True, 'independent_vision_encoder_validation': False,
                'same_production_target_for_both_decoders': True, 'sampling': False, 'use_model_defaults': False,
                'repetition_penalty': 1.0, 'minimum_output_length_added': False,
                'rope_deltas_reset_before_every_generation': True,
                'comparison_max_new_tokens': 512, 'prompt_diagnostic_max_new_tokens': 1024,
                'eos_token_id': int(eos), 'pad_token_id': int(pad),
                'native_generation_config_512': generation_config(eos, pad, 512).to_dict(),
                'native_generation_config_1024': generation_config(eos, pad, 1024).to_dict(),
                'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                'source_sha256': {str(p.relative_to(REPRO.parent)): sha(p) for p in
                                  (Path(__file__).resolve(), Path(common.__file__), REPRO/'qa_opd_fair/videoencode.py',
                                   REPRO.parent/'GLANCE_repo/glance/decode.py', REPRO.parent/'GLANCE_repo/glance/vl.py')},
                'limitations': ['Train-video diagnostic; not a quality score or independent test set.',
                                'Same patched production vision encoder is shared by both paths.',
                                'Repetition heuristics do not determine factuality or video grounding.',
                                'Timings are diagnostic, not a balanced speed benchmark.']}
    result = {'schema': 'native_generation_audit_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'pool': str(args.pool.resolve()), 'pool_sha256': pool_hash, 'row_index': args.row_index,
              'video': str(row['video_name']), 'split': row['split'], 'n_frames': int(row['n_frames']),
              'gpu_uuid': gpu, 'timeline': timeline, 'protocol': protocol, 'prompt_cases': []}
    if args.preflight_only:
        result.update(status='PREFLIGHT_PASSED', gpu_generation_performed=False,
                      prompt_cases=[dict(label=label, prompt=prompt, input_identity=identity)
                                    for label, prompt, _, identity in encoded_cases])
    else:
        def timed(fn, inputs, maximum):
            torch.cuda.synchronize();start = time.perf_counter()
            with torch.inference_mode():output = fn()
            torch.cuda.synchronize()
            return record(output, inputs['input_ids'], processor.tokenizer, eos, maximum, time.perf_counter()-start)

        original_inputs = {k: v.to('cuda:0') if torch.is_tensor(v) else v for k, v in encoded_cases[0][2].items()}
        reset_rope(target)
        repo = timed(lambda: greedy_generate(target, original_inputs['input_ids'], 512, [eos],
                     positions=VisionPositions(pixel_values_videos=original_inputs['pixel_values_videos'],
                                               video_grid_thw=original_inputs['video_grid_thw'])), original_inputs, 512)
        native = timed(lambda: native_generate(target, original_inputs, generation_config(eos, pad, 512)), original_inputs, 512)
        mismatch = first_difference(repo['gen_ids'], native['gen_ids'])
        result['decoder_comparison'] = {'same_encoded_input': True, 'input_identity': encoded_cases[0][3],
                                        'identical_tokens': mismatch is None, 'first_mismatch': mismatch,
                                        'repo': repo, 'native': native}
        print(json.dumps({'phase': 'decoder_comparison', 'video': result['video'],
                          'identical_tokens': mismatch is None, 'first_mismatch': mismatch}), flush=True)
        del original_inputs
        for label, prompt, cpu_inputs, identity in encoded_cases:
            inputs = {k: v.to('cuda:0') if torch.is_tensor(v) else v for k, v in cpu_inputs.items()}
            generated = timed(lambda: native_generate(target, inputs, generation_config(eos, pad, 1024)), inputs, 1024)
            result['prompt_cases'].append(dict(label=label, prompt=prompt, input_identity=identity, generation=generated))
            print(json.dumps({'phase': 'prompt_case', 'label': label, 'gen_tokens': generated['gen_tokens'],
                              'hit_eos': generated['hit_eos'], 'truncated': generated['truncated']}), flush=True)
            del inputs
        result.update(status='NATIVE_AGREEMENT_PASSED' if mismatch is None else 'NATIVE_AGREEMENT_FAILED', gpu_generation_performed=True)
    if sha(args.pool) != pool_hash:raise ValueError('pool changed during diagnostic')
    write_exclusive(out, result)
    print(json.dumps({'status': result['status'], 'out': str(out)}), flush=True)
    if result['status'] == 'NATIVE_AGREEMENT_FAILED':raise SystemExit(2)


if __name__ == '__main__':main()
