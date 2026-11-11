#!/usr/bin/env python3
"""Parallel per-source Video-MME frame-cache builder (same schema as build_real_video_pilot)."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))
from video_caption import build_real_video_pilot as b

ROOT = os.environ['RV_ROOT']
CACHE_ROOT = Path(os.environ['RV_CACHE'])
COUNTS = [64, 128]
LONG_SIDE = 384


def process(source):
    path = source['source_path']
    timeline, fps, origin = b.decode_timeline(path)
    samples = b.nested_samples(timeline, COUNTS)
    digest = b.sha256_file(path)
    width, height = b.resized_dimensions(source['source_width'], source['source_height'], LONG_SIDE)
    all_ids = sorted(set(i for ids in samples.values() for i in ids))
    pixels = b.materialize_frames(path, all_ids, width, height)
    rows = []
    import numpy as np
    for n, ids in samples.items():
        cache_path = CACHE_ROOT / f"{digest[:24]}_{n}f.npz"
        with cache_path.open('xb') as f:
            np.savez_compressed(f, frames=np.stack([pixels[i] for i in ids]))
        row = dict(source, task=source['video_id'], video_name=source['video_id'],
                   dataset='Video-MME-derived', prompt=b.PROMPT, question=b.PROMPT, answer='',
                   hash=digest, source_sha256=digest, source_fps=fps, fps=fps,
                   source_frame_count=len(timeline), time_origin='first_source_frame_pts',
                   source_pts_origin_s=origin, sample_frame_ids=ids,
                   sample_timestamps=[timeline[i] for i in ids],
                   sample_source_pts=[timeline[i] + origin for i in ids],
                   n_frames=n, n_unique_frames=n, width=width, height=height,
                   cache_path=str(cache_path), cache_hash=b.sha256_file(cache_path),
                   materialized=True, input_ids_len=None, visual_token_count=None,
                   processing_config={'sampling': 'uniform_pts_nested_v1', 'long_side': LONG_SIDE,
                                      'resize': 'PIL_BILINEAR_no_upscale', 'color': 'RGB',
                                      'cache_layout': 'uint8_TCHW', 'cache_format': 'npz_lossless',
                                      'processor_resamples_frames': False,
                                      'requested_frame_counts': list(COUNTS)})
        row.update(b.validate_real_metadata(row, require_cache=True, require_exact=True))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-jsonl', required=True)
    ap.add_argument('--exclude-video-list', required=True)
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-train', type=int, default=80)
    ap.add_argument('--n-dev', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20261111)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--write-parquet', action='store_true')
    a = ap.parse_args()
    ex = b.load_exclusions(a.exclude_video_list)
    rows = [json.loads(l) for l in Path(a.source_jsonl).read_text().splitlines() if l.strip()]
    sources = b.unique_sources(rows, ex, a.source_root)
    eligible = []
    for s in sources:
        if s['duration_category'] != 'medium':
            continue
        info = b.probe_video(s['source_path'])
        if 240.0 <= info['video_duration'] <= 600.0:
            eligible.append(dict(s, **info))
    selected = b.assign_splits(eligible, a.n_train, a.n_dev, a.seed)
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = {n: [] for n in COUNTS}
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(process, s): s for s in selected}
        for fut in as_completed(futures):
            rows2 = fut.result()
            done += 1
            for r in rows2:
                manifests[r['n_frames']].append(r)
            if done % 5 == 0 or done == len(selected):
                print(f'[build] {done}/{len(selected)}', flush=True)
    for n, records in manifests.items():
        records.sort(key=lambda r: r['video_id'])
        b.write_jsonl(out / f'manifest_{n}.jsonl', records)
        for split in ('train', 'dev'):
            subset = [r for r in records if r['split'] == split]
            b.write_jsonl(out / f'{split}_{n}.jsonl', subset)
            if a.write_parquet:
                import pandas as pd
                pd.DataFrame(subset).to_parquet(out / f'{split}_{n}.parquet', index=False)
    summary = {'schema_version':'real_video_pilot_v1','dataset':'Video-MME-derived','official_benchmark':False,
               'n_source_qa_rows':len(rows),'n_unique_after_exclusions':len(sources),
               'n_eligible':len(eligible),'n_train':a.n_train,'n_dev':a.n_dev,'seed':a.seed,
               'frame_counts':COUNTS,'prompt_sha256':hashlib.sha256(b.PROMPT.encode()).hexdigest(),
               'cache_root':str(CACHE_ROOT),'trainable':True,'parallel_workers':a.workers,
               'selection':[{'video_id':s['video_id'],'split':s['split'],'video_duration':s['video_duration']} for s in selected]}
    (out/'PROTOCOL.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('n_eligible','n_train','n_dev','trainable')},indent=2), flush=True)


if __name__ == '__main__':
    main()
