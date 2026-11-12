#!/usr/bin/env python3
"""CPU-only 128-subset-preserving 256/512-frame engineering video inputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPRO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPRO))
from video_caption import build_real_video_pilot as b
from local_h100.prepare_data import atomic_json,require_local
from local_h100.build_prompt_view import summary_prompt

ROOT=Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
_PROCESSOR=None


def nested_indices(base,source_count,counts=(256,512)):
    """Split the largest source-index gap; ties choose the smaller left index."""
    base=list(base)
    if len(base)!=128 or len(set(base))!=128 or base!=sorted(base):
        raise ValueError('base must contain exactly 128 distinct ordered source indices')
    if base[0]!=0 or base[-1]!=source_count-1:
        raise ValueError('base must preserve the source first and last frames')
    if source_count<max(counts):
        raise ValueError(f'ineligible source: only {source_count} original frames, need {max(counts)}; repetition is forbidden')
    selected=set(base)
    heap=[(-(right-left),left,right) for left,right in zip(base,base[1:]) if right-left>1]
    heapq.heapify(heap)
    snapshots={}
    for count in sorted(counts):
        while len(selected)<count:
            if not heap:raise ValueError('not enough distinct original frames')
            _,left,right=heapq.heappop(heap)
            middle=(left+right)//2
            if middle in selected:raise AssertionError('midpoint already selected')
            selected.add(middle)
            for a,z in ((left,middle),(middle,right)):
                if z-a>1:heapq.heappush(heap,(-(z-a),a,z))
        snapshots[count]=sorted(selected)
    return snapshots


def worker_init(model):
    global _PROCESSOR
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    _PROCESSOR=AutoProcessor.from_pretrained(model,local_files_only=True)


def prepare(source,cache_root,prompt):
    import numpy as np
    from qa_opd_fair.videoencode import encode_inputs,validate_real_metadata
    if b.sha256_file(source['source_path'])!=source['source_sha256']:
        raise ValueError('source SHA changed from original 128-frame manifest')
    if b.sha256_file(source['cache_path'])!=source['cache_hash']:
        raise ValueError('original 128-frame cache SHA changed')
    timeline,fps,origin=b.decode_timeline(source['source_path'])
    if len(timeline)!=source['source_frame_count']:
        raise ValueError('actual source frame count changed')
    if abs(origin-source['source_pts_origin_s'])>1e-6:
        raise ValueError('source PTS origin changed')
    samples=nested_indices(source['sample_frame_ids'],len(timeline))
    width,height=int(source['width']),int(source['height'])
    pixels=b.materialize_frames(source['source_path'],samples[512],width,height)
    with np.load(source['cache_path'],allow_pickle=False) as loaded:
        original=loaded['frames']
    if any(not np.array_equal(original[i],pixels[source_id]) for i,source_id in enumerate(source['sample_frame_ids'])):
        raise ValueError('fresh MP4 source pixels differ from the original 128-frame cache')
    rows=[];metrics=[]
    video_token=_PROCESSOR.tokenizer.convert_tokens_to_ids('<|video_pad|>')
    for n in (256,512):
        ids=samples[n]
        path=Path(cache_root)/f"{source['source_sha256'][:24]}_{n}f.npz"
        temporary=path.with_suffix('.npz.partial')
        frames=np.stack([pixels[i] for i in ids])
        with temporary.open('xb') as file:np.savez_compressed(file,frames=frames)
        os.link(temporary,path);temporary.unlink()
        row=dict(source,prompt=prompt,question=prompt,n_frames=n,n_unique_frames=n,
                 fps=fps,source_fps=fps,source_frame_count=len(timeline),source_pts_origin_s=origin,
                 sample_frame_ids=ids,sample_timestamps=[timeline[i] for i in ids],
                 sample_source_pts=[timeline[i]+origin for i in ids],cache_path=str(path),
                 cache_hash=b.sha256_file(path),materialized=True,input_ids_len=None,visual_token_count=None,
                 engineering_long_input_probe=True,parent_frame_count=128,
                 parent_cache_path=source['cache_path'],parent_cache_hash=source['cache_hash'],
                 processing_config={'sampling':'largest_index_gap_midpoint_nested_from128_v1',
                                    'gap_definition':'right_source_frame_index - left_source_frame_index',
                                    'tie_break':'smallest_left_source_index','midpoint':'floor((left+right)/2)',
                                    'long_side':384,'resize':'PIL_BILINEAR_no_upscale','color':'RGB',
                                    'cache_layout':'uint8_TCHW','cache_format':'npz_lossless',
                                    'processor_resamples_frames':False,'requested_frame_counts':[128,256,512]})
        row.update(validate_real_metadata(row,require_cache=True,require_exact=True))
        encoded=encode_inputs(_PROCESSOR,row,'real')
        row['input_ids_len']=int(encoded['input_ids'].shape[-1])
        row['visual_token_count']=int((encoded['input_ids']==video_token).sum())
        rows.append(row)
        metrics.append({'video_id':row['video_id'],'split':row['split'],'n_frames':n,
                        'input_ids_len':row['input_ids_len'],'visual_token_count':row['visual_token_count'],
                        'video_grid_thw':encoded['video_grid_thw'].tolist(),
                        'pixel_values_shape':list(encoded['pixel_values_videos'].shape),
                        'input_plus_1024':row['input_ids_len']+1024,
                        'fits_16384_with_1024':row['input_ids_len']+1024<=16384,
                        'fits_32768_with_1024':row['input_ids_len']+1024<=32768,
                        'first_sample_s':row['sample_timestamps'][0],'last_sample_s':row['sample_timestamps'][-1],
                        'duration_s':row['video_duration'],'source_sha256':row['source_sha256'],
                        'cache_sha256':row['cache_hash']})
        del frames,encoded
    # Read final files and verify the actual saved pixels, including old 128.
    with np.load(rows[0]['cache_path'],allow_pickle=False) as a,np.load(rows[1]['cache_path'],allow_pickle=False) as z:
        map256={frame:i for i,frame in enumerate(samples[256])}
        map512={frame:i for i,frame in enumerate(samples[512])}
        if not np.array_equal(original,a['frames'][[map256[i] for i in source['sample_frame_ids']]]):
            raise ValueError('saved 256 cache lost original 128 pixels')
        if not np.array_equal(a['frames'],z['frames'][[map512[i] for i in samples[256]]]):
            raise ValueError('saved 512 cache lost 256 pixel subset')
    return {'video_id':source['video_id'],'rows':rows,'encoding':metrics,
            'source_sha_verified':True,'old_cache_sha_verified':True,'actual_nested_pixels_exact':True,
            'source_frame_count':len(timeline),'sampling_nested_indices_exact':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',default=str(ROOT/'data/smoke10/manifest_128.jsonl'))
    parser.add_argument('--out',default=str(ROOT/'data/long_input_probe'))
    parser.add_argument('--cache',default=str(ROOT/'frames_long_probe'))
    parser.add_argument('--model',default='/czsun/mcy/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:raise ValueError('workers must be 1..8')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    source_manifest=require_local(args.source_manifest)
    out,cache=map(require_local,(args.out,args.cache))
    sources=[json.loads(line) for line in source_manifest.read_text().splitlines()]
    if len(sources)!=10 or sum(r['split']=='train' for r in sources)!=8 or sum(r['split']=='dev' for r in sources)!=2:
        raise ValueError('expected the existing smoke10 split: eight train and two reused dev')
    if any(row['source_frame_count']<512 for row in sources):
        raise ValueError('ineligible source below 512 original frames; repeated frames are forbidden')
    raw_bytes=sum((256+512)*3*row['height']*row['width'] for row in sources)
    conservative=int(raw_bytes*1.02)+64*1024**2
    if conservative>10*1024**3 or conservative>shutil.disk_usage(ROOT).free:
        raise ValueError('storage preflight exceeds the 10 GiB cap or current free space')
    if out.exists() or cache.exists():raise FileExistsError('new output/cache directories required')
    out.mkdir(parents=True);cache.mkdir(parents=True)
    started=time.time();state={'phase':'preparing_and_encoding','completed_sources':0,'target_sources':10,
                             'workers':args.workers,'raw_uint8_bytes':raw_bytes,'conservative_disk_bound_bytes':conservative}
    atomic_json(out/'STATUS.json',state)
    try:
        prompt=summary_prompt(REPRO/'local_h100/audit_native_generation.py')
        prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()
        results=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(args.model,)) as pool:
            futures={pool.submit(prepare,row,str(cache),prompt):row for row in sources}
            for future in as_completed(futures):
                result=future.result();results.append(result)
                state.update(completed_sources=len(results),elapsed_s=time.time()-started)
                atomic_json(out/'STATUS.json',state);print(json.dumps(state),flush=True)
        import pandas as pd
        metrics=sorted([m for result in results for m in result['encoding']],key=lambda m:(m['n_frames'],m['video_id']))
        hashes={}
        for n in (256,512):
            records=sorted([row for result in results for row in result['rows'] if row['n_frames']==n],key=lambda r:r['video_id'])
            for name,subset in ((f'manifest_{n}',records),(f'train_{n}',[r for r in records if r['split']=='train']),
                                (f'dev_{n}',[r for r in records if r['split']=='dev'])):
                path=out/f'{name}.jsonl'
                with path.open('x') as file:
                    for row in subset:file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                pd.DataFrame(subset).to_parquet(out/f'{name}.parquet',index=False)
                hashes[path.name]=b.sha256_file(path);hashes[f'{name}.parquet']=b.sha256_file(out/f'{name}.parquet')
        cache_bytes=sum(p.stat().st_size for p in cache.glob('*.npz'))
        if cache_bytes>10*1024**3:raise ValueError('actual frame cache exceeded 10 GiB cap')
        protocol={'schema':'engineering_long_input_probe_v1','engineering_long_input_probe':True,
                  'dataset':'Video-MME-derived','n_train':8,'n_dev':2,'source_count':10,
                  'source_manifest':str(source_manifest),'source_manifest_sha256':b.sha256_file(source_manifest),
                  'source_script_sha256':b.sha256_file(__file__),'prompt':prompt,'prompt_sha256':prompt_sha,
                  'frame_counts':[256,512],'base_frame_count':128,'source_indices_nesting':'128 subset 256 subset 512',
                  'sampling':'largest source-index gap; floor midpoint; smallest-left-index tie break',
                  'all_nested_indices_verified':True,'all_saved_nested_pixels_verified':True,
                  'all_source_and_original_cache_sha_verified':True,'all_timestamps_exact':True,
                  'materialized':True,'valid_for_real_input_encoding':True,'cache_root':str(cache),
                  'raw_uint8_bytes':raw_bytes,'conservative_disk_bound_bytes':conservative,
                  'actual_cache_bytes':cache_bytes,'within_10gib':True,'view_files_sha256':hashes,
                  'holdout_used':False,'dev_reused_not_independent':True,'original_metadata_and_cache_modified':False,
                  'task_scope':'bounded-summary output; input-length engineering probe, not thousand-token long-output evaluation',
                  'elapsed_s':time.time()-started,
                  'limitations':['The two development sources were already used; no independent-test claim.',
                                 'Token counts are processor measurements, not GPU memory or speed measurements.',
                                 '512-frame inputs can exceed the existing 16384-token training cap; inspect measured lengths before GPU execution.']}
        atomic_json(out/'PROTOCOL.json',protocol)
        atomic_json(out/'ENCODE_VALIDATION.json',{'schema':'engineering_long_input_actual_encoding_v1',
                                                'cpu_only':True,'model':args.model,'prompt_sha256':prompt_sha,
                                                'n_conditions':20,'rows':metrics,
                                                'maximum_input_plus_1024':max(r['input_plus_1024'] for r in metrics),
                                                'all_fit_32768_with_1024':all(r['fits_32768_with_1024'] for r in metrics)})
        state.update(phase='completed',actual_cache_bytes=cache_bytes,elapsed_s=time.time()-started)
        atomic_json(out/'STATUS.json',state)
        print(json.dumps({'out':str(out),'actual_cache_gib':cache_bytes/1024**3,
                          'token_ranges':{str(n):{'input':[min(r['input_ids_len'] for r in metrics if r['n_frames']==n),
                                                          max(r['input_ids_len'] for r in metrics if r['n_frames']==n)],
                                                  'visual':sorted({r['visual_token_count'] for r in metrics if r['n_frames']==n})}
                                          for n in (256,512)}},indent=2),flush=True)
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),elapsed_s=time.time()-started)
        atomic_json(out/'STATUS.json',state)
        raise


if __name__=='__main__':
    main()
