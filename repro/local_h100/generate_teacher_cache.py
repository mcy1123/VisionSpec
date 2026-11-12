#!/usr/bin/env python3
"""Generate finite, sharded train-only AR caches identical to local video SFT.

Each process uses exactly one explicitly selected GPU UUID. Cache keys and token
publication are delegated to train_video_sft; this script never trains a head.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import time

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

import pandas as pd
import torch

from local_h100 import train_video_sft as sft
from shared import common, training_resume
from shared.draft_geometry import eos_ids
from shared.model_contract import validate_model_contract
from video_caption.train_video_9k import parse_deadline, stop_reason, validate_rows


def partition_train(rows, shards, shard):
    if isinstance(shards, bool) or not isinstance(shards, int) or shards < 1:
        raise ValueError('shards must be a positive integer')
    if isinstance(shard, bool) or not isinstance(shard, int) or not 0 <= shard < shards:
        raise ValueError('shard must be in [0, shards)')
    if not rows:
        raise ValueError('training pool is empty')
    tasks = [row.get('task') for row in rows]
    if any(not isinstance(task, str) or not task for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('each training task must be a unique nonempty string')
    if any(row.get('split') != 'train' for row in rows):
        raise ValueError('teacher collector accepts only an explicitly train-only pool; dev/test cannot be mixed in')
    if any(not isinstance(row.get('prompt'), str) or not row['prompt'].strip() for row in rows):
        raise ValueError('each training task requires a nonempty prompt')
    return sorted(rows, key=lambda row: row['task'])[shard::shards]


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    # Some mounted filesystems expose coarse mtime/ctime despite *_ns names.
    # Bind content samples too; for small files this hashes all bytes.
    samples = hashlib.sha256()
    with path.open('rb') as stream:
        for offset in sorted({0, max(0, stat.st_size // 2 - 32768), max(0, stat.st_size - 65536)}):
            stream.seek(offset)
            samples.update(str(offset).encode() + b':' + stream.read(65536))
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                content_samples_sha256=samples.hexdigest())


def target_weight_hashes(target, identity_dir):
    """Hash bytes once; reuse only with identical device/inode/size/mtime/ctime and content samples.

    The cache is local to this run and generated here, not an unverified model
    path or an imported manifest. A fingerprint change yields a different key.
    The lock prevents eight simultaneous workers re-reading all target weights.
    """
    target, identity_dir = Path(target), Path(identity_dir)
    paths = sorted(target.glob('*.safetensors')) + sorted(target.glob('pytorch_model*.bin'))
    if not paths:
        raise ValueError('target has no local weight files')
    before = {path.name: file_identity(path) for path in paths}
    identity_dir.mkdir(parents=True, exist_ok=True)
    stem = sft.digest(before)
    result_path, lock_path = identity_dir / f'{stem}.json', identity_dir / f'{stem}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reused = result_path.exists()
        if reused:
            result = json.loads(result_path.read_text())
            hashes = result.get('weight_sha256', {})
            if (result.get('schema') != 1 or result.get('file_identity') != before or
                    set(hashes) != set(before) or
                    any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in hashes.values())):
                raise ValueError('invalid locally verified target identity cache')
        else:
            hashes = {path.name: training_resume.file_hash(path) for path in paths}
            result = dict(schema=1, file_identity=before, weight_sha256=hashes,
                          verified_utc=datetime.now(timezone.utc).isoformat(),
                          method='full_SHA256_then_metadata_and_content_sample_identity')
        after = {path.name: file_identity(path) for path in paths}
        if after != before:
            raise RuntimeError('target weights changed while their identity was being checked')
        if not reused:
            sft.write_json_exclusive(result_path, result)
    return hashes, dict(path=str(result_path), sha256=training_resume.file_hash(result_path),
                        verified_identity_reused=reused, file_identity=before)


def teacher_protocol(target, weights, max_new_tokens):
    """Exactly the fields consumed by SFT.teacher_key, with identical values."""
    return dict(hashes=dict(target_config=training_resume.file_hash(Path(target) / 'config.json'),
                            target_weights=weights,
                            decoder=training_resume.file_hash(REPRO.parent / 'GLANCE_repo/glance/decode.py')),
                config=dict(max_new_tokens=max_new_tokens))


def output_summary(tokens, eos, max_new_tokens):
    if not tokens or len(tokens) > max_new_tokens or tokens != sft.trim_at_eos(tokens, eos):
        raise ValueError('invalid teacher tokens or tokens after the first EOS')
    hit_eos = tokens[-1] in eos
    if len(tokens) < max_new_tokens and not hit_eos:
        raise ValueError('short teacher trajectory must end at EOS')
    return dict(generated_tokens=len(tokens), hit_eos=hit_eos,
                eos_token_id=tokens[-1] if hit_eos else None,
                truncated=not hit_eos and len(tokens) == max_new_tokens,
                usable_for_sft=len(tokens) >= 2)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'target', 'init-from', 'teacher-cache', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--shards', type=int, required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--max-seq-len', type=int, default=16384)
    p.add_argument('--deadline-utc')
    p.add_argument('--stop-file')
    p.add_argument('--dry-run', action='store_true')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.max_new_tokens < 1 or a.max_seq_len <= a.max_new_tokens:
        raise ValueError('positive output budget and additional prompt capacity are required')
    selected_gpu = sft.visible_gpu_ids()
    if len(selected_gpu) != 1:
        raise ValueError('teacher collection requires exactly one GPU UUID')
    deadline = parse_deadline(a.deadline_utc)
    rows = pd.read_parquet(a.pool).to_dict('records')
    selected = partition_train(rows, a.shards, a.shard)
    if not selected:
        raise ValueError('empty shard; reduce --shards')
    out, cache = sft.safe_output(a.out), sft.safe_output(a.teacher_cache)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('collector output must be new or empty')
    indexed = pd.DataFrame(selected).set_index('task', verify_integrity=True)
    validation = validate_rows(indexed, {0: [{'task': row['task']} for row in selected]}, 'real')
    model_contract = validate_model_contract(a.target, a.init_from)
    preflight = dict(schema=1, entry='local_h100/generate_teacher_cache.py',
                     target_dtype='float32', split='train', time_mode='real',
                     total_pool_rows=len(rows), selected_tasks=[row['task'] for row in selected],
                     config=vars(a), gpu_uuids=selected_gpu,
                     pool_sha256=training_resume.file_hash(a.pool), model_contract=model_contract,
                     input_validation=validation, source_sha256=training_resume.file_hash(__file__),
                     sft_cache_source_sha256=training_resume.file_hash(sft.__file__))
    if a.dry_run:
        print(json.dumps(preflight, indent=2))
        return preflight
    reason = stop_reason(deadline, a.stop_file)
    if reason:
        raise RuntimeError(f'not starting model load: {reason}')
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    weights, verification = target_weight_hashes(a.target, cache / '_target_identity')
    protocol = teacher_protocol(a.target, weights, a.max_new_tokens)
    preflight['target_weight_verification'] = verification
    preflight['teacher_protocol'] = protocol
    preflight['protocol_sha256'] = sft.digest(preflight)
    sft.write_json_exclusive(out / 'PROTOCOL.json', preflight)
    completed = []
    started = time.perf_counter()
    try:
        torch.cuda.set_device(0)
        target, processor = common.load_target(a.target, device='cuda:0', dtype=torch.float32)
        target.eval().requires_grad_(False)
        eos = eos_ids(target)
        video_id = processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')
        for row in selected:
            reason = stop_reason(deadline, a.stop_file)
            if reason:
                break
            from qa_opd_fair.videoencode import encode_inputs
            qa_started = time.perf_counter()
            cpu_started = time.perf_counter()
            inputs = encode_inputs(processor, row, 'real')
            preproc_s = time.perf_counter() - cpu_started
            prompt_tokens = int(inputs['input_ids'].shape[1])
            if prompt_tokens + a.max_new_tokens > a.max_seq_len:
                raise ValueError(f"{row['task']}: prompt plus output budget exceeds max-seq-len")
            key, encoded_identity = sft.teacher_key(inputs, protocol, eos)
            existed = (cache / f'{key}.json').exists()
            print(f"[teacher] shard={a.shard}/{a.shards} task={row['task']} "
                  f"prompt={prompt_tokens} cache_exists={existed}", flush=True)
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize()
            ar_started = time.perf_counter()
            tokens, cached, actual_key = sft.get_teacher(target, inputs, protocol, cache, eos)
            torch.cuda.synchronize()
            teacher_s = time.perf_counter() - ar_started
            if actual_key != key:
                raise RuntimeError('SFT teacher cache identity changed during collection')
            row_result = dict(task=row['task'], video_id=row.get('video_id'),
                              n_frames=int(row['n_frames']), input_tokens=prompt_tokens,
                              visual_tokens=int((inputs['input_ids'] == video_id).sum()),
                              teacher_cache_key=key, teacher_cache_hit=cached,
                              encoded_input_identity=encoded_identity['encoded_inputs'],
                              cpu_preprocessing_seconds=preproc_s,
                              teacher_call_seconds=teacher_s,
                              ar_seconds=None if cached else teacher_s,
                              ar_timing='independent_AR_plus_input_transfer_hash_and_cache_write; no speed-comparison claim',
                              peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved(0) / 2**30,
                              total_seconds=time.perf_counter() - qa_started,
                              **output_summary(tokens, eos, a.max_new_tokens))
            with (out / 'rows.jsonl').open('a') as stream:
                stream.write(json.dumps(row_result) + '\n')
            completed.append(row_result)
            print(json.dumps(row_result), flush=True)
        result = dict(status='COMPLETED' if len(completed) == len(selected) else 'STOPPED',
                      completed_rows=len(completed), planned_rows=len(selected),
                      stop_reason=reason, shard=a.shard, shards=a.shards,
                      generated_tokens=sum(row['generated_tokens'] for row in completed),
                      eos_rows=sum(row['hit_eos'] for row in completed),
                      truncated_rows=sum(row['truncated'] for row in completed),
                      cache_hits=sum(row['teacher_cache_hit'] for row in completed),
                      rows_not_usable_for_sft=[row['task'] for row in completed if not row['usable_for_sft']],
                      elapsed_seconds=time.perf_counter() - started,
                      protocol_sha256=preflight['protocol_sha256'])
        sft.write_json_exclusive(out / 'RUN_STATUS.json', result)
        print(json.dumps(result), flush=True)
        return result
    except Exception as exc:
        sft.write_json_exclusive(out / 'FAILED.json', dict(completed_rows=len(completed),
                                planned_rows=len(selected), error_type=type(exc).__name__,
                                error=str(exc), protocol_sha256=preflight['protocol_sha256']))
        raise


if __name__ == '__main__':
    main()
