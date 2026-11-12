#!/usr/bin/env python3
"""CPU-only real nested 256/512-frame view of the frozen forty development videos.

Reuses four existing real dev2 caches; creates only the other 76. No teacher,
training, GPU work, holdout access, or modification of an existing data file.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

REPRO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPRO))
from local_h100.build_long_input_probe import nested_indices, b
from local_h100.build_prompt_view import exclusive_json, summary_prompt
from local_h100.prepare_data import atomic_json, require_local
from local_h100.processor_profile_runtime import load_profile, apply_profile, validate_encoded

ROOT=Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
PROCESSOR=PROFILE=None


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def initialize(model,profile_path,cpu_ids):
    global PROCESSOR,PROFILE
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    # Restrict only this new CPU worker. PyAV's auto threading sees two CPUs;
    # the four workers collectively use at most eight logical CPUs.
    if cpu_ids and hasattr(os,'sched_setaffinity'):
        worker=(multiprocessing.current_process()._identity[-1]-1)%4
        own=cpu_ids[worker*2:worker*2+2] or cpu_ids[:2]
        os.sched_setaffinity(0,set(own))
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    PROFILE=load_profile(profile_path)
    PROCESSOR=AutoProcessor.from_pretrained(model,local_files_only=True)
    apply_profile(PROCESSOR,PROFILE)


def save_frames_exclusive(path,frames):
    import numpy as np
    temporary=path.with_suffix('.npz.partial')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream,frames=frames)
    os.link(temporary,path)
    temporary.unlink()


def prepare(source,reused,cache_root,predictions):
    import numpy as np
    from qa_opd_fair.videoencode import encode_inputs,validate_real_metadata
    from evaluation.eval_video_opd import encoded_identity
    if b.sha256_file(source['source_path'])!=source['source_sha256']:
        raise ValueError('source SHA differs from the frozen original dev128 manifest')
    if b.sha256_file(source['cache_path'])!=source['cache_hash']:
        raise ValueError('original dev128 frame cache SHA differs')
    timeline,fps,origin=b.decode_timeline(source['source_path'])
    if len(timeline)!=source['source_frame_count'] or abs(origin-source['source_pts_origin_s'])>1e-6:
        raise ValueError('source frame count or PTS origin changed')
    if abs(fps-source['source_fps'])>1e-9:
        raise ValueError('source FPS changed')
    samples=nested_indices(source['sample_frame_ids'],len(timeline))
    with np.load(source['cache_path'],allow_pickle=False) as data:
        original=data['frames']
    if original.dtype!=np.uint8 or original.shape!=(128,3,216,384):
        raise ValueError('the fixed profile requires original uint8 (128,3,216,384)')
    rows=[]
    if reused:
        if set(reused)!={256,512}:
            raise ValueError('reuse requires both existing real frame tiers')
        for n in (256,512):
            row=dict(reused[n])
            if (row['source_sha256']!=source['source_sha256']
                    or row['parent_cache_hash']!=source['cache_hash']
                    or row['sample_frame_ids']!=samples[n]
                    or b.sha256_file(row['cache_path'])!=row['cache_hash']):
                raise ValueError('reused cache identity/nested sample differs')
            if any(abs(t-timeline[i])>1e-9 for t,i in zip(row['sample_timestamps'],samples[n])):
                raise ValueError('reused cache timestamps differ from fresh source PTS')
            rows.append(row)
    else:
        pixels=b.materialize_frames(source['source_path'],samples[512],source['width'],source['height'])
        for index,source_id in enumerate(source['sample_frame_ids']):
            if not np.array_equal(original[index],pixels[source_id]):
                raise ValueError('fresh source pixels differ from an original128 frame')
        for n in (256,512):
            path=Path(cache_root)/f"{source['source_sha256'][:24]}_{n}f.npz"
            frames=np.stack([pixels[i] for i in samples[n]])
            save_frames_exclusive(path,frames)
            row=dict(source,n_frames=n,n_unique_frames=n,source_fps=fps,fps=fps,
                     source_frame_count=len(timeline),source_pts_origin_s=origin,
                     sample_frame_ids=samples[n],sample_timestamps=[timeline[i] for i in samples[n]],
                     sample_source_pts=[timeline[i]+origin for i in samples[n]],
                     cache_path=str(path),cache_hash=b.sha256_file(path),materialized=True,
                     parent_frame_count=128,parent_cache_path=source['cache_path'],
                     parent_cache_hash=source['cache_hash'],engineering_long_input_probe=True,
                     processing_config=dict(sampling='largest_index_gap_midpoint_nested_from128_v1',
                         gap_definition='right_source_frame_index - left_source_frame_index',
                         tie_break='smallest_left_source_index',midpoint='floor((left+right)/2)',
                         long_side=384,resize='PIL_BILINEAR_no_upscale',color='RGB',
                         cache_layout='uint8_TCHW',cache_format='npz_lossless',
                         processor_resamples_frames=False,requested_frame_counts=[128,256,512]))
            rows.append(row)
            del frames
        del pixels
    # Verify actual saved arrays, including all original128 pixels and 256⊂512.
    with np.load(rows[0]['cache_path'],allow_pickle=False) as data:
        frames256=data['frames']
    with np.load(rows[1]['cache_path'],allow_pickle=False) as data:
        frames512=data['frames']
    map256={source_id:i for i,source_id in enumerate(samples[256])}
    map512={source_id:i for i,source_id in enumerate(samples[512])}
    if (frames256.dtype!=np.uint8 or frames256.shape!=(256,3,216,384)
            or frames512.dtype!=np.uint8 or frames512.shape!=(512,3,216,384)):
        raise ValueError('saved frame cache has wrong dtype/shape')
    if not np.array_equal(original,frames256[[map256[i] for i in source['sample_frame_ids']]]):
        raise ValueError('saved256 did not preserve original128 pixels')
    if not np.array_equal(frames256,frames512[[map512[i] for i in samples[256]]]):
        raise ValueError('saved512 did not preserve256 pixels')
    unique_pixels={n:len({hashlib.sha256(frame.tobytes()).digest() for frame in frames})
                   for n,frames in ((256,frames256),(512,frames512))}
    del original,frames256,frames512
    metrics=[]
    for row in rows:
        n=row['n_frames']
        row.update(split='dev',prompt=source['prompt'],question=source['question'],
                   processor_profile=PROFILE['path'],processor_profile_sha256=PROFILE['sha256'],
                   engineering_long_input_probe=True,development_long_input_transfer=True,
                   evaluation_only=True,trainable=False,cache_reused_from_dev2=bool(reused),
                   input_ids_len=None,visual_token_count=None)
        row.update(validate_real_metadata(row,require_cache=True,require_exact=True))
        inputs=encode_inputs(PROCESSOR,row,'real')
        measured=validate_encoded(PROCESSOR,row,inputs,PROFILE)
        identity=encoded_identity(inputs)
        prediction=predictions[n]
        if (measured['input_ids_len']!=prediction['input_tokens']
                or identity['input_ids']!=prediction['input_ids_sha256']):
            raise ValueError('actual processor input differs from the reviewed formula prediction')
        row['input_ids_len']=measured['input_ids_len']
        row['visual_token_count']=measured['visual_token_count']
        metrics.append(dict(**measured,video_id=row['video_id'],split='dev',
                            input_identity=identity,pixel_values_shape=list(inputs['pixel_values_videos'].shape),
                            input_plus_1024=row['input_ids_len']+1024,
                            source_sha256=row['source_sha256'],cache_sha256=row['cache_hash'],
                            actual_cpu_processor_execution=True,formula_prediction_sha_matches=True,
                            cache_reused=bool(reused),source_indices_unique=len(set(samples[n]))==n,
                            unique_pixel_frames=unique_pixels[n],
                            pixel_duplicate_policy='natural equal source pixels allowed; no source-frame index repeated or synthesized',
                            source_pts_first=row['sample_timestamps'][0],source_pts_last=row['sample_timestamps'][-1],
                            real_time_exact=row['real_time_exact']))
        del inputs
    return dict(video_id=source['video_id'],rows=rows,encoding=metrics,
                source_sha256=source['source_sha256'],parent_cache_sha256=source['cache_hash'],
                actual_frame_provenance={str(n):dict(
                    source_indices=ids,source_indices_sha256=canonical_sha(ids),
                    relative_pts_s=[timeline[i] for i in ids],
                    relative_pts_sha256=canonical_sha([timeline[i] for i in ids]),
                    source_pts_s=[timeline[i]+origin for i in ids],
                    source_pts_sha256=canonical_sha([timeline[i]+origin for i in ids]))
                    for n,ids in ((128,source['sample_frame_ids']),(256,samples[256]),(512,samples[512]))},
                source_sha_verified=True,old_cache_sha_verified=True,
                actual_nested_pixels_exact=True,source_frame_count=len(timeline),
                sampling_nested_indices_exact=True,source_pts_redecoded=True,
                cache_reused=bool(reused),cpu_affinity=sorted(os.sched_getaffinity(0)))


def preflight(source,profile_path,model,reused):
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    profile=load_profile(profile_path)
    rows=read_rows(source)
    if len(rows)!=40 or any(r['split']!='dev' or r['n_frames']!=128 for r in rows):
        raise ValueError('requires exactly the frozen40 development128 rows')
    if len({r['source_sha256'] for r in rows})!=40:
        raise ValueError('source byte aliases in dev40')
    proc=AutoProcessor.from_pretrained(model,local_files_only=True)
    applied=apply_profile(proc,profile)
    for row in rows:
        samples=nested_indices(row['sample_frame_ids'],row['source_frame_count'])
        if not set(row['sample_frame_ids'])<set(samples[256])<set(samples[512]):
            raise AssertionError('strict frame index nesting failed')
        for n in (256,512):
            if len(set(samples[n]))!=n or samples[n][0]!=0 or samples[n][-1]!=row['source_frame_count']-1:
                raise AssertionError('frame uniqueness/endpoints failed')
            if smart_resize(n,row['height'],row['width'],factor=32,min_pixels=4096,max_pixels=50331648)!=(224,384):
                raise ValueError('fixed profile spatial size does not match frozen14x24 grid')
            old=reused.get(row['video_id'],{}).get(n)
            if old and old['sample_frame_ids']!=samples[n]:
                raise ValueError('reused sample plan changed')
    try:
        nested_indices(list(range(128)),128)
    except ValueError:
        pass
    else:
        raise AssertionError('insufficient original frames did not fail')
    return rows,profile,dict(status='PASSED_CPU_NESTED_AND_ACTUAL_PROCESSOR_PROFILE_CHECK',
                             n_sources=40,n_conditions=80,profile_sha256=profile['sha256'],
                             actual_processor=applied,insufficient_source_fails=True,
                             no_pixels_written=True,no_gpu=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(ROOT/'data/bounded_summary_v2/dev_128.jsonl'))
    parser.add_argument('--profile',default=str(ROOT/'data/long_input_probe_fixed_spatial/PROCESSOR_PROFILE.json'))
    parser.add_argument('--out',default=str(ROOT/'data/dev40_long_input_fixed_spatial'))
    parser.add_argument('--cache',default=str(ROOT/'frames_long_dev40'))
    parser.add_argument('--model',default='/czsun/mcy/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    if not 1<=args.workers<=4:
        raise ValueError('CPU workers must be1..4')
    source,out,cache=map(require_local,(args.source,args.out,args.cache))
    reused={}
    for n in (256,512):
        for row in read_rows(ROOT/f'data/long_input_probe_fixed_spatial/dev_{n}.jsonl'):
            reused.setdefault(row['video_id'],{})[n]=row
    rows,profile,checks=preflight(source,args.profile,args.model,reused)
    if args.self_test:
        print(json.dumps(checks,indent=2))
        return
    if out.exists() or cache.exists():
        raise FileExistsError('new output and cache directories required')
    source_parquet=source.with_suffix('.parquet')
    source_bindings={str(path):b.sha256_file(path) for path in (source,source_parquet)}
    import pandas as pd
    parquet_rows=pd.read_parquet(source_parquet).to_dict('records')
    if {r['video_id']:r['source_sha256'] for r in parquet_rows}!={r['video_id']:r['source_sha256'] for r in rows}:
        raise ValueError('original dev128 parquet/source identity differs from JSONL')
    prompt=summary_prompt(REPRO/'local_h100/audit_native_generation.py')
    if any(r['prompt']!=prompt or r['question']!=prompt for r in rows):
        raise ValueError('dev40 must retain the current bounded v2 prompt')
    train_path=ROOT/'data/expanded320/train_128.jsonl'
    train_sha=b.sha256_file(train_path)
    if {r['source_sha256'] for r in rows}&{r['source_sha256'] for r in read_rows(train_path)}:
        raise ValueError('dev source SHA intersects training pool')
    feasibility_path=ROOT/'analysis/dev40_long_input_feasibility.json'
    feasibility=json.loads(feasibility_path.read_text())
    if feasibility['source_manifest_sha256']!=b.sha256_file(source) or feasibility['profile_sha256']!=profile['sha256']:
        raise ValueError('reviewed feasibility input/profile changed')
    predictions={r['video_id']:{c['n_frames']:c for c in r['conditions']} for r in feasibility['rows']}
    raw=sum((256+512)*3*r['height']*r['width'] for r in rows if r['video_id'] not in reused)
    conservative=int(raw*1.02)+64*1024**2
    if conservative>10*1024**3 or conservative>shutil.disk_usage(ROOT).free:
        raise ValueError('new cache budget exceeds10GiB or current free space')
    out.mkdir(parents=True);cache.mkdir(parents=True)
    profile_copy=out/'PROCESSOR_PROFILE.json'
    with profile_copy.open('xb') as stream:
        stream.write(Path(args.profile).read_bytes())
    profile=load_profile(profile_copy)
    started=time.time()
    state=dict(phase='cpu_source_decode_and_actual_encoding',target_sources=40,target_conditions=80,
               completed_sources=0,completed_actual_encodings=0,workers=args.workers,
               new_cache_files_expected=76,reused_cache_files_expected=4,conservative_disk_bound_bytes=conservative)
    atomic_json(out/'STATUS.json',state)
    cpu_ids=sorted(os.sched_getaffinity(0))[-8:]
    try:
        results=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                                 initargs=(args.model,str(profile_copy),cpu_ids)) as pool:
            futures=[pool.submit(prepare,row,reused.get(row['video_id']),str(cache),predictions[row['video_id']]) for row in rows]
            for future in as_completed(futures):
                results.append(future.result())
                state.update(completed_sources=len(results),completed_actual_encodings=2*len(results),
                             current_new_cache_bytes=sum(p.stat().st_size for p in cache.glob('*.npz')),
                             elapsed_s=time.time()-started)
                atomic_json(out/'STATUS.json',state)
                print(json.dumps(state),flush=True)
        import pandas as pd
        metrics=sorted([m for result in results for m in result['encoding']],key=lambda r:(r['n_frames'],r['video']))
        if len(metrics)!=80 or len({(m['video'],m['n_frames']) for m in metrics})!=80:
            raise ValueError('missing/duplicate actual condition encodings')
        if any(m['input_plus_1024']>32768 for m in metrics):
            raise ValueError('actual input+1024 exceeds reviewed engineering capacity')
        hashes={};peaks={}
        for n in (256,512):
            records=sorted([row for result in results for row in result['rows'] if row['n_frames']==n],key=lambda r:r['video_id'])
            peak=min(records,key=lambda r:(-r['input_ids_len'],r['video_id']))
            for name,subset in ((f'manifest_{n}',records),(f'dev_{n}',records),(f'peak_{n}',[peak])):
                path=out/f'{name}.jsonl'
                with path.open('x') as stream:
                    for row in subset:
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                pd.DataFrame(subset).to_parquet(out/f'{name}.parquet',index=False)
                hashes[path.name]=b.sha256_file(path)
                hashes[f'{name}.parquet']=b.sha256_file(out/f'{name}.parquet')
            peaks[str(n)]=dict(source=peak['video_id'],source_sha256=peak['source_sha256'],
                              input_tokens=peak['input_ids_len'],visual_tokens=peak['visual_token_count'],
                              parquet=str(out/f'peak_{n}.parquet'),parquet_sha256=hashes[f'peak_{n}.parquet'])
        if any(b.sha256_file(path)!=digest for path,digest in source_bindings.items()) or b.sha256_file(train_path)!=train_sha:
            raise ValueError('original development view or training pool changed during build')
        files=list(cache.glob('*.npz'))
        if len(files)!=76 or sum(bool(r['cache_reused']) for r in results)!=2:
            raise ValueError('expected76 new caches and2 reused sources')
        cache_bytes=sum(p.stat().st_size for p in files)
        if cache_bytes>10*1024**3:
            raise ValueError('actual new cache exceeds10GiB')
        source_provenance=sorted([{k:v for k,v in r.items() if k not in ('rows','encoding')} for r in results],
                                 key=lambda r:r['video_id'])
        source_verifications=[{k:v for k,v in r.items() if k not in ('actual_frame_provenance','cpu_affinity')}
                              | {'frame_index_and_pts_sha256':{n:{k:v for k,v in frame.items() if k.endswith('_sha256')}
                                                              for n,frame in r['actual_frame_provenance'].items()}}
                              for r in source_provenance]
        exclusive_json(out/'SOURCE_PROVENANCE.json',dict(original_dev128_files_sha256=source_bindings,
                       source_count=40,rows=source_provenance))
        provenance_sha=b.sha256_file(out/'SOURCE_PROVENANCE.json')
        protocol=dict(schema='dev40_long_input_fixed_spatial_v1',n_train=0,n_dev=40,
                      evaluation_only=True,trainable=False,development_long_input_transfer=True,
                      source_manifest=str(source),source_manifest_sha256=b.sha256_file(source),
                      original_dev128_files_sha256=source_bindings,
                      source_provenance_sha256=provenance_sha,source_verifications=source_verifications,
                      training_source_sha_intersection=0,training_pool_file_unchanged=True,
                      profile=profile,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      frame_counts=[256,512],base_frame_count=128,spatial_grid_hw=[14,24],
                      sampling='largest source-index gap; floor midpoint; smallest-left-index tie break',
                      source_indices_nested=True,saved128_subset256_subset512_pixels_exact=True,
                      all_source_and_old_cache_sha_verified=True,all_source_pts_redecoded=True,
                      all_actual_frame_indices_unique=True,no_synthetic_frame_repeats=True,
                      naturally_identical_source_pixels_allowed=True,all_timestamps_exact=True,
                      actual_cpu_encodings=80,new_cache_files=76,reused_cache_files=4,
                      new_cache_root=str(cache),new_cache_bytes=cache_bytes,raw_new_uint8_bytes=raw,
                      conservative_disk_bound_bytes=conservative,view_files_sha256=hashes,peak_sources=peaks,
                      source_script_sha256=b.sha256_file(__file__),
                      sampling_helper_sha256=b.sha256_file(REPRO/'local_h100/build_long_input_probe.py'),
                      decode_helper_sha256=b.sha256_file(REPRO/'video_caption/build_real_video_pilot.py'),
                      original_files_modified=False,holdout_used=False,teacher_generated=False,gpu_used=False,
                      task_scope='frozen128-frame-trained heads evaluated on new long inputs of the same40dev; bounded summary natural EOS',
                      limitations=['Existing40dev sources were repeatedly used; this is not an independent test set.',
                                   'Actual CPU tensors do not prove GPU memory safety, losslessness or acceleration.',
                                   '512-frame inputs exceed16384; longest-source gate must precede full40 per intended head.',
                                   'Longest prompt does not upper-bound output length or peak GPU memory.'],
                      elapsed_s=time.time()-started)
        exclusive_json(out/'PROTOCOL.json',protocol)
        exclusive_json(out/'ENCODE_VALIDATION.json',dict(schema='dev40_long_input_actual_processor_v1',
                       cpu_only=True,n_conditions=80,processor_profile=profile,rows=metrics,
                       longest_sources=peaks,maximum_input_plus_1024=max(m['input_plus_1024'] for m in metrics),
                       all_formula_predictions_confirmed_by_actual_encoding=True))
        ready=dict(status='READY_ACTUAL80_ENCODINGS_VALIDATED',n_dev=40,n_conditions=80,
                   frame_counts=[256,512],processor_profile=str(profile_copy),processor_profile_sha256=profile['sha256'],
                   original_dev128_files_sha256=source_bindings,
                   source_provenance=str(out/'SOURCE_PROVENANCE.json'),source_provenance_sha256=provenance_sha,
                   source_verifications=source_verifications,
                   new_cache_files=76,reused_cache_files=4,new_cache_bytes=cache_bytes,
                   peak_sources=peaks,protocol_sha256=b.sha256_file(out/'PROTOCOL.json'),
                   encoding_validation_sha256=b.sha256_file(out/'ENCODE_VALIDATION.json'),
                   view_files_sha256=hashes,training_allowed=False,teacher_generated=False,gpu_used=False,
                   elapsed_s=time.time()-started)
        exclusive_json(out/'READY.json',ready)
        state.update(phase='completed',new_cache_bytes=cache_bytes,elapsed_s=time.time()-started)
        atomic_json(out/'STATUS.json',state)
        print(json.dumps(ready,indent=2),flush=True)
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),elapsed_s=time.time()-started)
        atomic_json(out/'STATUS.json',state)
        raise


if __name__=='__main__':
    main()
