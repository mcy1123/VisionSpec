#!/usr/bin/env python3
"""CPU-only local H100 video preparation, with strict PTS and early smoke data.

This explicit local entry accepts only the user-authorized /czsun/mcy data root.
It reuses the real-video builder's sampling, pixel decode and metadata checks;
it does not relax the shared builder's historical storage policy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from video_caption import build_real_video_pilot as b

LOCAL_ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
COUNTS = (64, 128)


def require_local(path):
    path = Path(path).expanduser().resolve()
    root = LOCAL_ROOT.resolve()
    if not root.exists() or not path.is_relative_to(root) or path == root:
        raise ValueError(f'output must be a child of existing authorized root {root}')
    return path


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def probe(source):
    return dict(source, **b.probe_video(source['source_path']))


def process(source, cache_root, long_side):
    import numpy as np
    path = source['source_path']
    timeline, fps, origin = b.decode_timeline(path)
    samples = b.nested_samples(timeline, COUNTS)
    digest = b.sha256_file(path)
    width, height = b.resized_dimensions(source['source_width'], source['source_height'], long_side)
    records = []
    # Validate each exact timeline before creating pixels or cache files.
    for n in COUNTS:
        ids = samples[n]
        row = dict(source, task=source['video_id'], video_name=source['video_id'],
                   dataset='Video-MME-derived', prompt=b.PROMPT, question=b.PROMPT, answer='',
                   hash=digest, source_sha256=digest, source_fps=fps, fps=fps,
                   source_frame_count=len(timeline), time_origin='first_source_frame_pts',
                   source_pts_origin_s=origin, sample_frame_ids=ids,
                   sample_timestamps=[timeline[i] for i in ids],
                   sample_source_pts=[timeline[i] + origin for i in ids],
                   n_frames=n, n_unique_frames=n, width=width, height=height,
                   cache_path='', cache_hash='', materialized=True,
                   input_ids_len=None, visual_token_count=None,
                   processing_config={'sampling':'uniform_pts_nested_v1','long_side':long_side,
                                      'resize':'PIL_BILINEAR_no_upscale','color':'RGB',
                                      'cache_layout':'uint8_TCHW','cache_format':'npz_lossless',
                                      'processor_resamples_frames':False,
                                      'requested_frame_counts':list(COUNTS)})
        row.update(b.validate_real_metadata(row, require_cache=False, require_exact=True))
        records.append(row)
    pixels = b.materialize_frames(path, samples[max(COUNTS)], width, height)
    for row in records:
        destination = Path(cache_root) / f"{digest[:24]}_{row['n_frames']}f.npz"
        temporary = destination.with_suffix('.npz.partial')
        with temporary.open('xb') as file:
            np.savez_compressed(file, frames=np.stack([pixels[i] for i in row['sample_frame_ids']]))
        # Exclusive final publication: never replace historical bytes.
        os.link(temporary, destination)
        temporary.unlink()
        row['cache_path'] = str(destination)
        row['cache_hash'] = b.sha256_file(destination)
        b.validate_real_metadata(row, require_cache=True, require_exact=True)
    return records


def write_manifests(out, records, protocol):
    import pandas as pd
    out.mkdir(parents=True, exist_ok=False)
    for n in COUNTS:
        subset = sorted([r for r in records if r['n_frames'] == n], key=lambda r:r['video_id'])
        b.write_jsonl(out / f'manifest_{n}.jsonl', subset)
        for split in ('train','dev'):
            split_rows = [r for r in subset if r['split'] == split]
            b.write_jsonl(out / f'{split}_{n}.jsonl', split_rows)
            pd.DataFrame(split_rows).to_parquet(out / f'{split}_{n}.parquet', index=False)
    atomic_json(out / 'PROTOCOL.json', protocol)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-jsonl', default='/czsun/mcy/video_opd_pilot/Video-MME/full/videomme_900videos_2700qa.jsonl')
    ap.add_argument('--source-root', default='/czsun/mcy/video_opd_pilot/Video-MME/full/videos')
    ap.add_argument('--out-dir', default=str(LOCAL_ROOT / 'data'))
    ap.add_argument('--cache-root', default=str(LOCAL_ROOT / 'frames'))
    ap.add_argument('--n-train',type=int,default=80)
    ap.add_argument('--n-dev',type=int,default=40)
    ap.add_argument('--seed',type=int,default=20261112)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--long-side',type=int,default=384)
    a = ap.parse_args()
    if not 1 <= a.workers <= 8 or a.long_side < 1:
        raise ValueError('workers must be 1..8 and long-side positive')
    out, cache = require_local(a.out_dir), require_local(a.cache_root)
    if out.exists() or cache.exists():
        raise FileExistsError('new output/cache directories required; no silent restart or overwrite')
    out.mkdir(parents=True)
    cache.mkdir(parents=True)
    started = time.time()
    status = {'phase':'probing','completed_videos':0,'target_videos':a.n_train+a.n_dev,
              'started_unix':started,'workers':a.workers}
    atomic_json(out/'STATUS.json',status)
    try:
        qa = [json.loads(line) for line in Path(a.source_jsonl).read_text().splitlines() if line.strip()]
        sources = b.unique_sources(qa, set(), a.source_root)
        medium = [s for s in sources if s['duration_category']=='medium']
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            probed = list(pool.map(probe, medium))
        eligible = [s for s in probed if 240.0 <= s['video_duration'] <= 600.0]
        selected = b.assign_splits(eligible,a.n_train,a.n_dev,a.seed)
        ids = {s['video_id'] for s in selected}
        reserve = [dict(s, split='reserved_unused') for s in sources if s['video_id'] not in ids]
        b.write_jsonl(out/'selection.jsonl',selected)
        b.write_jsonl(out/'reserved_unused_sources.jsonl',reserve)
        protocol = {'schema_version':'real_video_pilot_v1','dataset':'Video-MME-derived',
                    'official_benchmark':False,'n_source_qa_rows':len(qa),'n_unique_sources':len(sources),
                    'n_eligible':len(eligible),'n_train':a.n_train,'n_dev':a.n_dev,'seed':a.seed,
                    'n_reserved_unused_sources':len(reserve),'frame_counts':list(COUNTS),
                    'source_jsonl_sha256':b.sha256_file(a.source_jsonl),
                    'prompt_sha256':hashlib.sha256(b.PROMPT.encode()).hexdigest(),
                    'selection_sha256':b.sha256_file(out/'selection.jsonl'),
                    'storage_policy':'explicit_user_authorized_local_h100_root',
                    'cache_root':str(cache),'long_side':a.long_side,'trainable':False,
                    'preprocessing':'strict_real_PTS_nested_64_128',
                    'historical_exclusions':[],
                    'limitations':['Local train/dev are disjoint by source id and verified source bytes.',
                       'Reserved sources are not decoded or byte-deduplicated yet; not a final test protocol.',
                       'No claim of independence from historical experiments on this public dataset.',
                       'No model run: visual token count and output length unmeasured.']}
        atomic_json(out/'PROTOCOL.json',protocol)
        smoke = ([s for s in selected if s['split']=='train'][:min(8,a.n_train)]
                 +[s for s in selected if s['split']=='dev'][:min(2,a.n_dev)])
        smoke_ids = {s['video_id'] for s in smoke}
        stages = [smoke,[s for s in selected if s['video_id'] not in smoke_ids]]
        records, hashes = [],set()
        for stage_index, stage in enumerate(stages):
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                futures = {pool.submit(process,s,str(cache),a.long_side):s for s in stage}
                for future in as_completed(futures):
                    result = future.result()
                    digest = result[0]['source_sha256']
                    if digest in hashes:
                        raise ValueError('duplicate source bytes; refuse cross-source alias leakage')
                    hashes.add(digest)
                    records.extend(result)
                    status.update(phase='decoding',completed_videos=len(hashes),elapsed_s=time.time()-started)
                    atomic_json(out/'STATUS.json',status)
                    print(json.dumps(status),flush=True)
            if stage_index == 0:
                smoke_protocol = dict(protocol,trainable=True,n_train=min(8,a.n_train),n_dev=min(2,a.n_dev),
                                      purpose='smoke_only_subset_of_full_training_and_development')
                write_manifests(out/'smoke10',records,smoke_protocol)
                print('SMOKE_READY '+str(out/'smoke10'),flush=True)
        protocol.update(trainable=True,all_timestamps_exact=True,source_byte_dedup_verified=True,
                        completed_unix=time.time(),elapsed_s=time.time()-started)
        # Publish final files only when every selected source has passed.
        final = out/'full'
        write_manifests(final,records,protocol)
        atomic_json(out/'PROTOCOL.json',protocol)
        status.update(phase='completed',output=str(final),elapsed_s=time.time()-started)
        atomic_json(out/'STATUS.json',status)
        print('DATA_READY '+str(final),flush=True)
    except BaseException as exc:
        status.update(phase='failed',error=repr(exc),elapsed_s=time.time()-started)
        atomic_json(out/'STATUS.json',status)
        raise


if __name__=='__main__':
    main()
