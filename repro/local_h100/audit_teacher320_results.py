#!/usr/bin/env python3
"""Read-only audit of the selected train320 teacher caches and long-input gates.

Only explicitly referenced train caches and completed gate artifacts are read;
no model execution, holdout reads, queue updates, or cache modifications occur.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from local_h100.build_prompt_view import summary_prompt, exclusive_json
from local_h100.summarize_baseline import repetition

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(value, message):
    if not value:
        raise ValueError(message)


def file_identity(path):
    # Same metadata/content-sample guard as the collector, without touching locks.
    path = Path(path).resolve()
    stat = path.stat()
    samples = hashlib.sha256()
    with path.open('rb') as stream:
        for offset in sorted({0, max(0, stat.st_size//2 - 32768), max(0, stat.st_size - 65536)}):
            stream.seek(offset)
            samples.update(str(offset).encode() + b':' + stream.read(65536))
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                content_samples_sha256=samples.hexdigest())


def stats(values):
    import numpy as np
    return dict(n=len(values), mean=statistics.mean(values), min=min(values), max=max(values),
                quantiles={str(q):float(np.quantile(values, q)) for q in (0.25,0.5,0.75,0.9,0.95,0.99)})


def group_summary(records):
    def repvalue(row):
        return row['repetition']['token_8gram']['duplicate_occurrence_fraction'] or 0.0
    return {
        'n_conditions': len(records), 'unique_sources': len({r['video_id'] for r in records}),
        'generated_tokens_including_eos': stats([r['generated_tokens'] for r in records]),
        'natural_eos': sum(r['hit_eos'] for r in records),
        'truncated': sum(r['truncated'] for r in records),
        'token_count_at_least_512': sum(r['generated_tokens'] >= 512 for r in records),
        'input_tokens': stats([r['input_tokens'] for r in records]),
        'visual_tokens': stats([r['visual_tokens'] for r in records]),
        'repetition': {
            'heuristic_only_not_grounding_or_quality': True,
            'token_8gram_duplicate_fraction': stats([repvalue(r) for r in records]),
            'rows_with_any_repeated_token_8gram': sum(repvalue(r)>0 for r in records),
            'rows_with_token_8gram_duplicate_fraction_at_least_0_25': sum(repvalue(r)>=0.25 for r in records),
            'rows_with_token_8gram_duplicate_fraction_at_least_0_5': sum(repvalue(r)>=0.5 for r in records),
            'rows_with_duplicate_normalized_paragraph': sum(bool(r['repetition']['repeated_paragraph_examples']) for r in records),
            'top_10_token_8gram_rows': [dict(video_id=r['video_id'], n_frames=r['n_frames'],
                                              fraction=repvalue(r), generated_tokens=r['generated_tokens'])
                                       for r in sorted(records,key=repvalue,reverse=True)[:10]],
        },
    }


def audit_teachers(root, tokenizer, target):
    import numpy as np
    data = root/'data/expanded320'
    prompt = summary_prompt(REPRO/'local_h100/audit_native_generation.py')
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    expected = {}
    for n in (64,128):
        for row in lines(data/f'train_{n}.jsonl'):
            require(row['split']=='train' and row['prompt']==prompt and row['question']==prompt,
                    'current train view does not use bounded v2 prompt')
            key = (row['video_id'],n)
            require(key not in expected, 'duplicate source condition in train view')
            expected[key] = row
    require(len(expected)==640, 'expected 320 train sources in each frame tier')
    source_sets = [{r['source_sha256'] for (video,frames),r in expected.items() if frames==n} for n in (64,128)]
    require(source_sets[0]==source_sets[1] and len(source_sets[0])==320, 'source SHA aliases or frame-tier source mismatch')
    encoding = read(root/'ENCODE_EXPANDED_VALIDATION.json')
    encoded = {(r['video_id'],r['n_frames']):r for r in encoding['rows'] if r['split']=='train'}
    require(encoding['prompt_sha256']==prompt_sha, 'processor audit has different prompt')
    eos = sorted(read(target/'generation_config.json')['eos_token_id'])
    target_config_sha = sha(target/'config.json')
    decoder_sha = sha(REPRO.parent/'GLANCE_repo/glance/decode.py')
    protocols, records, seen = [], [], set()
    verified_targets = {}
    pool_hashes = {str(data/f'train_{n}.parquet'):sha(data/f'train_{n}.parquet') for n in (64,128)}
    runs = sorted((root/'jobs').glob('teacher_train320*/run/PROTOCOL.json'))
    require(len(runs)==8, 'expected eight explicitly selected train320 collector shards')
    for protocol_path in runs:
        p = read(protocol_path)
        run = protocol_path.parent
        job = run.parent
        require(read(job/'EXIT.json')['exit_code']==0 and read(run/'RUN_STATUS.json')['status']=='COMPLETED',
                f'{job.name}: collector did not complete')
        require(p['protocol_sha256']==digest({k:v for k,v in p.items() if k!='protocol_sha256'}), 'collector protocol digest mismatch')
        require(p['split']=='train' and p['time_mode']=='real' and p['target_dtype']=='float32', 'teacher execution mode differs')
        require(p['pool_sha256']==pool_hashes[p['config']['pool']], 'current train parquet differs from collector input')
        require(Path(p['config']['target']).resolve()==target.resolve(), 'different target path')
        verification = p['target_weight_verification']
        identity_path = Path(verification['path'])
        require(sha(identity_path)==verification['sha256'], 'verified target identity manifest changed')
        if str(identity_path) not in verified_targets:
            identity = read(identity_path)
            actual = {name:file_identity(target/name) for name in identity['file_identity']}
            require(actual==identity['file_identity']==verification['file_identity'], 'current target weight identity changed')
            verified_targets[str(identity_path)] = identity
        identity = verified_targets[str(identity_path)]
        hashes = p['teacher_protocol']['hashes']
        require(hashes['target_config']==target_config_sha and hashes['target_weights']==identity['weight_sha256']
                and hashes['decoder']==decoder_sha, 'teacher target/decoder hash mismatch')
        rows = lines(run/'rows.jsonl')
        require(len(rows)==80 and {r['task'] for r in rows}==set(p['selected_tasks']), 'shard rows differ from selected tasks')
        for result in rows:
            key = (result['video_id'],result['n_frames'])
            require(key in expected and key not in seen, f'duplicate or unexpected train condition {key}')
            seen.add(key)
            row, enc = expected[key], encoded[key]
            require(p['input_validation']['cache_sha256'][row['cache_path']]==row['cache_hash']==enc['cache_sha256'], 'cached pixel identity differs from audited train manifest')
            path = root/'teacher_cache'/f"{result['teacher_cache_key']}.json"
            cached = read(path)
            ident = cached['identity']
            require(digest(ident)==result['teacher_cache_key'], 'teacher cache key mismatch')
            require(ident['encoded_inputs']==result['encoded_input_identity'], 'collector and cache actual tensor identity mismatch')
            require(ident['target_config']==target_config_sha and ident['target_weights']==identity['weight_sha256']
                    and ident['decoder_sha256']==decoder_sha and ident['target_dtype']=='float32'
                    and ident['max_new_tokens']==1024 and ident['eos']==eos
                    and ident['generation']=='greedy_generate_trim_first_eos_v1', 'teacher generation contract mismatch')
            require(ident['encoded_inputs']['input_ids']['sha256']==enc['input_ids_sha256'], 'cached input IDs differ from independent CPU processor audit')
            input_ids, gen = cached['prompt_ids'], cached['gen_ids']
            require(hashlib.sha256(np.asarray(input_ids,dtype=np.int64).tobytes()).hexdigest()==enc['input_ids_sha256'], 'saved prompt IDs do not match tensor SHA')
            require(digest(gen)==cached['gen_ids_sha256'], 'saved generation token digest mismatch')
            require(len(input_ids)==result['input_tokens']==enc['input_ids_len'], 'prompt token count mismatch')
            require(result['visual_tokens']==enc['visual_token_count'], 'visual token count mismatch')
            suffix = tokenizer.decode(input_ids[-128:], skip_special_tokens=False)
            require(suffix.endswith(prompt+'<|im_end|>\n<|im_start|>assistant\n'), 'actual cached prompt tail is not the current bounded summary prompt')
            require(gen and len(gen)<=1024 and all(t not in eos for t in gen[:-1]), 'teacher has invalid length or tokens beyond first EOS')
            hit = gen[-1] in eos
            truncated = len(gen)==1024 and not hit
            require((hit or truncated) and result['hit_eos']==hit and result['truncated']==truncated
                    and result['generated_tokens']==len(gen), 'EOS/truncation/count summary mismatch')
            text = tokenizer.decode(gen,skip_special_tokens=True)
            records.append(dict(video_id=key[0],n_frames=key[1],source_sha256=row['source_sha256'],
                                teacher_cache_key=result['teacher_cache_key'],cache_file_sha256=sha(path),
                                gen_ids_sha256=cached['gen_ids_sha256'],input_tokens=len(input_ids),
                                visual_tokens=result['visual_tokens'],generated_tokens=len(gen),
                                hit_eos=hit,truncated=truncated,repetition=repetition(text,gen),
                                collector_job=job.name,collector_cache_hit=result['teacher_cache_hit'],
                                actual_prompt_and_cpu_encoded_identity_verified=True))
        protocols.append(dict(job=job.name,path=str(protocol_path),sha256=sha(protocol_path),
                              rows_sha256=sha(run/'rows.jsonl'),run_status=read(run/'RUN_STATUS.json')))
    require(seen==set(expected), 'missing train source conditions')
    require(len({r['teacher_cache_key'] for r in records})==640, 'teacher cache input aliases among source conditions')
    records.sort(key=lambda r:(r['n_frames'],r['video_id']))
    return dict(schema='teacher320_readonly_audit_v1',status='PASSED',utc=datetime.now(timezone.utc).isoformat(),
                root=str(root),n_sources=320,n_conditions=640,frame_counts=[64,128],
                all_expected_source_conditions_exactly_once=True,unique_source_byte_hashes=320,
                unique_teacher_cache_keys=640,prompt_sha256=prompt_sha,prompt=prompt,
                target=str(target),target_config_sha256=target_config_sha,decoder_sha256=decoder_sha,
                target_weight_identity_manifests={path:dict(sha256=sha(path),weight_sha256=item['weight_sha256']) for path,item in verified_targets.items()},
                target_weight_verification='existing local full-SHA manifest plus freshly matched metadata and first/middle/last content samples; no full weight rehash',
                raw_frame_cache_verification='collector validation hashes equal current manifest and CPU audit; no raw cache reread',
                actual_cached_prompt_ids_match_independent_processor_audit=True,
                current_parquet_sha256=pool_hashes,overall=group_summary(records),
                by_frames={str(n):group_summary([r for r in records if r['n_frames']==n]) for n in (64,128)},
                collectors=protocols,rows=records,holdout_read_or_evaluated=False,gpu_execution=False,
                limitations=['Training-only descriptive cache audit; no held-out quality claim.',
                             'Repetition statistics are lexical heuristics, not video grounding or correctness scores.',
                             'Bounded summary output is roughly two hundred tokens, not a thousand-token long-output benchmark.'],
                audit_script_sha256=sha(__file__))


def audit_long_input(root, tokenizer):
    view = root/'data/long_input_probe_fixed_spatial'
    profile_path = view/'PROCESSOR_PROFILE.json'
    profile_sha = sha(profile_path)
    profile = read(profile_path)
    cpu = {(r['video'],r['n_frames']):r for r in read(view/'ENCODE_VALIDATION.json')['rows']}
    records, pending, outputs = [], [], {}
    for frames in (256,512):
        for arm in ('public','warm20'):
            name = f'long_input_fixed{frames}_{arm}_peaktrain1'
            job = root/'jobs'/name
            path = job/'result.json'
            if not path.is_file() or not (job/'EXIT.json').is_file():
                pending.append(name)
                continue
            result = read(path)
            require(read(job/'EXIT.json')['exit_code']==0 and result['status']=='LOSSLESS_PASSED',f'{name}: gate failed')
            require(len(result['rows'])==1, f'{name}: expected one engineering source')
            row = result['rows'][0]
            require(row['video']=='2ka5AleJ48I' and row['n_frames']==frames, 'unexpected engineering source/frame count')
            expected = cpu[(row['video'],frames)]
            require(row['input_identity']==expected['input_identity'] and row['L']==expected['input_ids_len']
                    and row['n_visual_tokens']==expected['visual_token_count'] and row['video_grid_thw']==expected['video_grid_thw'], 'actual long-input tensors/grid/tokens differ from CPU profile audit')
            cfg, prot = result['config'],result['protocol']
            require(cfg['dtype']=='float32' and cfg['ar'] and cfg['max_new']==1024 and cfg['time_mode']=='real'
                    and cfg['eos_stop']=='immediate', 'long-input execution mode differs')
            require(prot['rows_sha256']==sha(view/f'train_{frames}.parquet'), 'long-input source view SHA mismatch')
            pp = prot['processor_profile']
            require(pp['sha256']==profile_sha==row['processor_profile_sha256']==cfg['processor_profile_sha256']
                    and pp['profile']==profile and row['processor_size']==profile['video_size'], 'actual profile identity/size mismatch')
            require(pp['actual_processor_instantiated'] and len(pp['actual_encodings'])==1
                    and pp['actual_encodings'][0]['processor_size']==profile['video_size'], 'actual processor audit missing')
            require(row['lossless'] and row['spec_ids']==row['ar_ids'] and row['first_mismatch'] is None
                    and len(row['spec_ids'])==row['gen_tokens']==row['ar_tokens'], 'direct SPEC/AR token equality failed')
            require(row['hit_eos'] and row['ar_hit_eos'] and not row['truncated'] and row['spec_ids'][-1]==151645,
                    'natural EOS gate failed')
            memory = prot['cuda_process_peak_memory']
            require(memory['status']=='COLLECTED' and memory['counters_reset_by_wrapper'] is False
                    and 0<memory['max_memory_allocated_bytes']<=memory['max_memory_reserved_bytes'], 'invalid process peak memory record')
            outputs[(frames,arm)] = row
            records.append(dict(job=name,frames=frames,arm=arm,video=row['video'],head=cfg['head'],
                                result=str(path),result_sha256=sha(path),status=result['status'],
                                input_tokens=row['L'],visual_tokens=row['n_visual_tokens'],grid=row['video_grid_thw'],
                                profile_sha256=profile_sha,processor_size=row['processor_size'],
                                actual_input_tensor_hashes_match_cpu_audit=True,output_tokens=row['gen_tokens'],
                                spec_equals_ar_direct_token_comparison=True,hit_natural_eos=True,truncated=False,
                                tau=row['tau'],effective_tokens_per_round=row['effective_tokens_per_round'],rounds=row['rounds'],
                                process_peak_allocated_gib=memory['max_memory_allocated_bytes']/2**30,
                                process_peak_reserved_gib=memory['max_memory_reserved_bytes']/2**30,
                                process_peak_scope=memory['scope'],spec_wall_s=row['spec_wall_s'],ar_wall_s=row['ar_wall_s'],
                                observed_ar_over_spec_wall_ratio=row['ar_wall_s']/row['spec_wall_s'],
                                observed_ar_over_spec_decode_ratio=row['ar_decode_ms_per_tok']/row['spec_decode_ms_per_tok'],
                                repetition=repetition(tokenizer.decode(row['spec_ids'],skip_special_tokens=True),row['spec_ids'])))
    paired = []
    for frames in (256,512):
        if all((frames,arm) in outputs for arm in ('public','warm20')):
            a,b = outputs[(frames,'public')],outputs[(frames,'warm20')]
            require(a['input_identity']==b['input_identity'] and a['ar_ids']==b['ar_ids'], 'cross-head target input or AR output differs')
            paired.append(dict(frames=frames,identical_actual_inputs_and_ar_output=True,
                               public_tau=a['tau'],warm20_tau=b['tau'],tau_delta=b['tau']-a['tau']))
    return dict(schema='long_input_peaktrain1_readonly_audit_v1',utc=datetime.now(timezone.utc).isoformat(),
                status='COMPLETED_ALL_PASS' if len(records)==4 else 'PARTIAL_COMPLETED_PASS',
                n_completed=len(records),pending=pending,rows=records,paired_heads=paired,
                source_count=1,train_source_reused=True,holdout_read_or_evaluated=False,
                gpu_execution_by_this_audit=False,profile_sha256=profile_sha,
                limitations=['One reused training source is an engineering memory/losslessness probe, not a performance estimate.',
                             'Timing is one observation per head/frame and may be under concurrent load; no clean speedup claim.',
                             'Peak memory includes process model loading plus both AR and SPEC; not SPEC-only memory.',
                             'Bounded summary output naturally ends below 1024 tokens; no forced or thousand-token output claim.'],
                audit_script_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(ROOT))
    parser.add_argument('--target',default='/czsun/mcy/models/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    root,target = Path(args.root).resolve(),Path(args.target).resolve()
    require(root==ROOT.resolve(),'audit output is restricted to the authorized local run root')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(target,local_files_only=True)
    teachers = audit_teachers(root,tokenizer,target)
    exclusive_json(root/'analysis/teacher320_summary.json',teachers)
    gates = audit_long_input(root,tokenizer)
    exclusive_json(root/'analysis/long_input_peaktrain1_summary.json',gates)
    print(json.dumps({'teacher_status':teachers['status'],'teacher_conditions':teachers['n_conditions'],
                      'eos':teachers['overall']['natural_eos'],'teacher_repetition':teachers['overall']['repetition'],
                      'long_input_status':gates['status'],'completed_gates':gates['n_completed']},indent=2))


if __name__=='__main__':
    main()
