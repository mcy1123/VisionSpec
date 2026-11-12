#!/usr/bin/env python3
"""CPU-only four-source/sixteen-window validation; never a full-pool builder.

Windows retain source-clock frame IDs and timestamps. Existing encode_inputs,
training, evaluation and frame sampling code are not patched. Source-level
identity protects splits even though window-level task IDs are unique.
"""
from __future__ import annotations
import argparse
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time

REPRO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPRO))
from local_h100.prepare_data import b, require_local, atomic_json
from local_h100.build_prompt_view import exclusive_json, summary_prompt

ROOT=Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
PROCESSOR=None


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_windows(timeline,fps):
    """Conservative visible PTS point span; half-open bounds exclude last PTS."""
    if len(timeline)<512 or fps<=0 or timeline[0]!=0 or any(a>=z for a,z in zip(timeline,timeline[1:])):
        raise ValueError('source needs >=512 strictly increasing real PTS and positive FPS')
    error=max(abs(i/fps-t) for i,t in enumerate(timeline))
    if error>1e-3:
        raise ValueError(f'full source PTS does not fit unchanged source-index/FPS contract: {error}s')
    duration=timeline[-1]
    if duration<=600:
        raise ValueError('fixed long-source probe needs a true PTS point span >600s')
    quarter=duration/4
    width=min(300.0,quarter)
    windows=[]
    for i in range(4):
        start=i*quarter+(quarter-width)/2
        end=start+width
        lower,upper=bisect.bisect_left(timeline,start),bisect.bisect_left(timeline,end)
        candidates=list(range(lower,upper))
        ids=b.sample_unique_timestamps(timeline,128,candidates)
        if not all(start<=timeline[j]<end for j in ids):
            raise AssertionError('half-open timestamp bounds violated')
        windows.append(dict(ordinal=i,start_s=start,end_s=end,nominal_duration_s=width,
                            eligible_source_frames=len(candidates),source_frame_ids=ids))
    if len({i for w in windows for i in w['source_frame_ids']})!=512:
        raise AssertionError('same source frame selected in multiple windows')
    return windows,error


def initialize(model,cpu_ids):
    global PROCESSOR
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    if cpu_ids and hasattr(os,'sched_setaffinity'):
        worker=(multiprocessing.current_process()._identity[-1]-1)%4
        os.sched_setaffinity(0,set(cpu_ids[worker*2:worker*2+2] or cpu_ids[:2]))
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    PROCESSOR=AutoProcessor.from_pretrained(model,local_files_only=True)


def materialize(source,cache_root,prompt,forbidden_sha):
    import numpy as np
    from qa_opd_fair import videoencode
    from evaluation import eval_video_opd
    path=Path(source['source_path'])
    before=path.stat()
    digest=b.sha256_file(path)
    if digest in forbidden_sha:
        raise ValueError('new source is a byte alias of an existing train/dev/holdout source')
    source_info=b.probe_video(path)
    timeline,fps,origin=b.decode_timeline(path)
    windows,error=source_windows(timeline,fps)
    if source_info['video_duration']<timeline[-1]-1e-3:
        raise ValueError('container duration is shorter than decoded visual PTS span')
    width,height=b.resized_dimensions(source_info['source_width'],source_info['source_height'],384)
    all_ids=sorted(i for w in windows for i in w['source_frame_ids'])
    pixels=b.materialize_frames(path,all_ids,width,height)
    rows=[];metrics=[]
    for window in windows:
        ordinal=window['ordinal'];ids=window['source_frame_ids']
        window_id=f"{source['video_id']}__window{ordinal:02d}"
        cache_identity=dict(schema='source_clock_four_windows_v1',source_sha256=digest,
                            source_video_id=source['video_id'],window_ordinal=ordinal,
                            source_pts_origin_s=origin,start_s=window['start_s'],end_s=window['end_s'],
                            sample_source_frame_ids=ids,width=width,height=height,
                            resize='PIL_BILINEAR_no_upscale')
        cache=Path(cache_root)/f'{canonical_sha(cache_identity)}_128f.npz'
        frames=np.stack([pixels[i] for i in ids])
        raw_pixel_sha=hashlib.sha256(frames.tobytes()).hexdigest()
        pixel_frame_sha=[hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
        temporary=cache.with_suffix('.npz.partial')
        with temporary.open('xb') as stream:
            np.savez_compressed(stream,frames=frames)
        os.link(temporary,cache);temporary.unlink()
        with np.load(cache,allow_pickle=False) as saved:
            if not np.array_equal(saved['frames'],frames):
                raise AssertionError('lossless saved pixels differ from decoded source pixels')
        row=dict(source,**source_info)
        row.update(video_id=window_id,window_id=window_id,
                 source_video_id=source['video_id'],task=window_id,video_name=window_id,
                 split='train',dataset='Video-MME-derived-window-engineering-probe',
                 prompt=prompt,question=prompt,answer='',hash=digest,source_sha256=digest,
                 source_fps=fps,fps=fps,source_frame_count=len(timeline),
                 n_frames=128,n_unique_frames=128,width=width,height=height,
                 source_pts_origin_s=origin,time_origin='first_source_frame_pts',
                 source_visual_pts_span_s=timeline[-1],window_clock='source_clock_no_reset',
                 source_video_duration_s=source_info['video_duration'],
                 window_layout_duration_s=timeline[-1],window_ordinal=ordinal,
                 window_start_s=window['start_s'],window_end_s=window['end_s'],
                 window_duration_s=window['nominal_duration_s'],window_interval='[start,end)',
                 window_start_absolute_pts_s=origin+window['start_s'],
                 window_end_absolute_pts_s=origin+window['end_s'],
                 window_eligible_source_frames=window['eligible_source_frames'],
                 sample_frame_ids=ids,source_frame_ids=ids,
                 sample_timestamps=[timeline[i] for i in ids],
                 sample_source_pts=[timeline[i]+origin for i in ids],
                 sample_window_relative_timestamps=[timeline[i]-window['start_s'] for i in ids],
                 cache_path=str(cache),cache_hash=b.sha256_file(cache),
                 cache_identity_sha256=canonical_sha(cache_identity),raw_pixel_sha256=raw_pixel_sha,
                 pixel_frame_sha256=pixel_frame_sha,source_file_bytes=before.st_size,
                 materialized=True,input_ids_len=None,visual_token_count=None,
                 engineering_window_probe=True,training_authorized=False,
                 independent_unit='source_video_id_and_source_sha256_not_window_id',
                 processing_config=dict(sampling='uniform_real_pts_within_source_clock_half_open_window_v1',
                     source_layout='four_quarter_centers_width_min_300_D_over4',
                     layout_D='last_real_relative_source_PTS; last source frame excluded by half-open bounds',
                     long_side=384,resize='PIL_BILINEAR_no_upscale',color='RGB',
                     cache_layout='uint8_TCHW',cache_format='npz_lossless',processor_resamples_frames=False))
        row.update(videoencode.validate_real_metadata(row,require_cache=True,require_exact=True))
        # These are the same function used by local SFT/video teacher and eval;
        # exercise both consumer references with actual pixels, not mock input.
        import ast
        training_ast=ast.parse((REPRO/'local_h100/train_video_sft.py').read_text())
        if not any(isinstance(node,ast.ImportFrom) and node.module=='qa_opd_fair.videoencode'
                   and any(alias.name=='encode_inputs' and alias.asname is None for alias in node.names)
                   for node in ast.walk(training_ast)):
            raise AssertionError('training no longer imports the expected real input encoder')
        if eval_video_opd.encode_inputs is not videoencode.encode_inputs:
            raise AssertionError('eval no longer imports the expected real input encoder')
        # This is the direct import inside local train_video_sft.main; no training
        # loop or model loading is invoked by this CPU data-contract check.
        from qa_opd_fair.videoencode import encode_inputs as training_encode_inputs
        encoded_train=training_encode_inputs(PROCESSOR,row,'real')
        identity_train=eval_video_opd.encoded_identity(encoded_train)
        encoded_eval=eval_video_opd.encode_inputs(PROCESSOR,row,'real')
        identity_eval=eval_video_opd.encoded_identity(encoded_eval)
        if identity_train!=identity_eval:
            raise AssertionError('actual train/eval encoded tensors differ')
        decoded=PROCESSOR.tokenizer.decode(encoded_train['input_ids'][0].tolist())
        markers=re.findall(r'<(\d+\.\d) seconds>',decoded)
        expected=[f'{(ids[i]/fps+ids[i+1]/fps)/2:.1f}' for i in range(0,128,2)]
        if markers!=expected:
            raise AssertionError('actual model time markers differ from source-clock pair means')
        length=int(encoded_train['input_ids'].shape[-1])
        video_token=PROCESSOR.tokenizer.convert_tokens_to_ids('<|video_pad|>')
        visual=int((encoded_train['input_ids']==video_token).sum())
        row.update(input_ids_len=length,visual_token_count=visual)
        metrics.append(dict(window_id=window_id,source_video_id=source['video_id'],source_sha256=digest,
                       window_ordinal=ordinal,source_frame_indices_unique=128,
                       half_open_source_pts_verified=True,train_eval_actual_tensor_sha_equal=True,
                       actual_processor_calls=2,input_identity=identity_train,input_ids_len=length,
                       visual_token_count=visual,video_grid_thw=encoded_train['video_grid_thw'].tolist(),
                       pixel_values_shape=list(encoded_train['pixel_values_videos'].shape),
                       uint8_cache_shape=list(frames.shape),raw_pixel_sha256=raw_pixel_sha,
                       unique_pixel_frames=len(set(pixel_frame_sha)),cache_sha256=row['cache_hash'],
                       source_clock_markers_verified=True,first_time_marker=markers[0],last_time_marker=markers[-1],
                       first_real_sample_pts=timeline[ids[0]],last_real_sample_pts=timeline[ids[-1]],
                       source_pts_origin_s=origin,full_source_index_fps_max_error_s=error,
                       window_start_s=window['start_s'],window_end_s=window['end_s'],
                       input_plus_1024=length+1024,fits_16384_with_1024=length+1024<=16384))
        rows.append(row)
        del frames,encoded_train,encoded_eval
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('source file changed while decoding')
    return dict(status='PASSED',source_video_id=source['video_id'],source_sha256=digest,
                source_bytes=before.st_size,source_duration_s=source_info['video_duration'],
                source_visual_pts_span_s=timeline[-1],source_frame_count=len(timeline),
                source_fps=fps,source_pts_origin_s=origin,full_source_index_fps_max_error_s=error,
                actual_source_indices_across_four_windows=512,rows=rows,encoding=metrics)


def attempt(*args):
    try:
        return materialize(*args)
    except Exception as exc:
        return dict(status='FAILED_FIXED_SOURCE_NO_REPLACEMENT',source_video_id=args[0]['video_id'],error=repr(exc))


def self_test():
    from qa_opd_fair.videoencode import validate_real_metadata
    timeline=[i/2 for i in range(1601)]
    windows,error=source_windows(timeline,2)
    assert len(windows)==4 and all(w['nominal_duration_s']==200 for w in windows)
    assert windows[-1]['end_s']==timeline[-1] and 1600 not in windows[-1]['source_frame_ids']
    row=dict(n_frames=128,source_fps=2,video_duration=800.5,sample_frame_ids=windows[3]['source_frame_ids'],
             sample_timestamps=[timeline[i] for i in windows[3]['source_frame_ids']],width=384,height=216)
    assert validate_real_metadata(row,require_cache=False,require_exact=True)['real_time_exact']
    # Retaining source indices while claiming reset timestamps must be rejected.
    wrong=dict(row,sample_timestamps=[t-windows[3]['start_s'] for t in row['sample_timestamps']])
    try:validate_real_metadata(wrong,require_cache=False,require_exact=True)
    except ValueError:pass
    else:raise AssertionError('mixed source indices/window clock was accepted')
    irregular=timeline.copy();irregular[50]+=.02
    try:source_windows(irregular,2)
    except ValueError:pass
    else:raise AssertionError('irregular PTS was accepted')
    return dict(status='PASSED',half_open_bounds=True,unique_real_source_indices=True,
                full_source_clock_contract=True,mixed_window_clock_rejected=True,irregular_pts_rejected=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',default=str(ROOT/'data/multiwindow_probe16'))
    ap.add_argument('--cache',default=str(ROOT/'frames_multiwindow_probe16'))
    ap.add_argument('--model',default='/czsun/mcy/models/Qwen3-VL-8B-Instruct')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    if not 1<=args.workers<=4:raise ValueError('at most four CPU workers')
    checks=self_test()
    if args.self_test:
        print(json.dumps(checks,indent=2));return
    out,cache=map(require_local,(args.out,args.cache))
    if out.exists() or cache.exists():raise FileExistsError('new isolated output/cache paths required')
    audit_path=ROOT/'analysis/scale_window_expansion_feasibility.json'
    audit=json.loads(audit_path.read_text());candidates=audit['candidate_source_window_metadata']
    if len(candidates)!=399:raise ValueError('expected the reviewed399 unused source candidates')
    catalog_path=Path('/czsun/mcy/video_opd_pilot/Video-MME/full/videomme_900videos_2700qa.jsonl')
    catalog=b.unique_sources(read_rows(catalog_path),set(),'/czsun/mcy/video_opd_pilot/Video-MME/full/videos')
    by_id={r['video_id']:r for r in catalog}
    if len(by_id)!=900:raise ValueError('expected the complete900-source identity catalog')
    # Reserve whole-source candidates also belong to the future whole-video
    # pool; a byte alias must not reappear as four new window sources.
    protected_paths=[ROOT/f'data/expanded320/{name}.jsonl' for name in
                     ('train_128','dev_128','future_holdout','unused_train_candidate_reserve')]
    protected=[r for path in protected_paths for r in read_rows(path)]
    forbidden_ids={r['video_id'] for r in protected};forbidden_sha={r['source_sha256'] for r in protected}
    if len(forbidden_ids)!=467 or len(forbidden_sha)!=467:
        raise ValueError('expected320train+40dev+80holdout+27reserved whole-source identities')
    if len({r['video_id'] for r in candidates})!=399 or {r['video_id'] for r in candidates}&forbidden_ids:
        raise ValueError('candidate source IDs duplicate or intersect the protected source pool')
    for row in candidates:
        if row['video_id'] not in by_id or Path(row['source_path']).resolve()!=Path(by_id[row['video_id']]['source_path']):
            raise ValueError('candidate source alias/path differs from the full900 catalog')
    ordered=sorted(candidates,key=lambda r:(r['source_duration_s'],r['video_id']))
    selections=[]
    for quartile in range(4):
        index=int((quartile+.5)*len(ordered)/4)
        record=ordered[index]
        selections.append(dict(by_id[record['video_id']],source_selection_quartile=quartile,
                               source_selection_sorted_index=index,
                               selection_container_duration_s=record['source_duration_s']))
    bindings={str(p):b.sha256_file(p) for p in [audit_path,catalog_path,*protected_paths]}
    out.mkdir(parents=True);cache.mkdir(parents=True)
    selection=dict(schema='fixed_four_source_window_cpu_probe_v1',n_sources=4,n_windows=16,
                   selection='sort399 by container duration,video_id; quartile midpoint indices49,149,249,349',
                   full_source_catalog_count=900,protected_source_count=467,
                   selected=selections,input_files_sha256=bindings,source_ids_reserved_to_split='train',
                   failure_policy='report fixed source failure; never replace or choose by model output',
                   self_test=checks)
    exclusive_json(out/'SELECTION.json',selection)
    prompt=summary_prompt(REPRO/'local_h100/audit_native_generation.py')
    started=time.time();results=[]
    state=dict(phase='cpu_real_window_materialization',target_sources=4,target_windows=16,completed_sources=0,
               completed_windows=0,actual_processor_calls=0,workers=args.workers)
    atomic_json(out/'STATUS.json',state)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             initargs=(args.model,sorted(os.sched_getaffinity(0))[-8:])) as pool:
        futures=[pool.submit(attempt,s,str(cache),prompt,forbidden_sha) for s in selections]
        for future in as_completed(futures):
            results.append(future.result())
            state.update(completed_sources=len(results),completed_windows=sum(len(r.get('rows',[])) for r in results),
                         actual_processor_calls=2*sum(len(r.get('rows',[])) for r in results),elapsed_s=time.time()-started)
            atomic_json(out/'STATUS.json',state);print(json.dumps(state),flush=True)
    rows=sorted([r for result in results for r in result.get('rows',[])],key=lambda r:r['video_id'])
    metrics=sorted([r for result in results for r in result.get('encoding',[])],key=lambda r:r['window_id'])
    if len({r['source_sha256'] for r in results if r['status']=='PASSED'})!=sum(r['status']=='PASSED' for r in results):
        raise ValueError('selected sources are byte aliases of each other')
    if any(b.sha256_file(p)!=digest for p,digest in bindings.items()):
        raise ValueError('source catalog or protected split files changed during this probe')
    import pandas as pd
    files={}
    if rows:
        for name in ('manifest_128','train_128'):
            path=out/f'{name}.jsonl'
            with path.open('x') as stream:
                for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            pd.DataFrame(rows).to_parquet(out/f'{name}.parquet',index=False)
            for extension in ('jsonl','parquet'):files[f'{name}.{extension}']=b.sha256_file(out/f'{name}.{extension}')
    complete=len(rows)==16 and len(metrics)==16 and all(r['status']=='PASSED' for r in results)
    report=dict(schema='source_clock_window_cpu_validation_v1',status='PASSED_16_WINDOW_CPU_PROBE' if complete else 'INCOMPLETE_FIXED_SOURCE_FAILURE',
                n_source_attempts=4,n_valid_sources=sum(r['status']=='PASSED' for r in results),
                n_windows=len(rows),actual_processor_calls=2*len(metrics),full_pool_ready=False,
                sources=[{k:v for k,v in r.items() if k not in ('rows','encoding')} for r in results],rows=metrics,
                window_time_semantics='source-clock: global source frame IDs and source-first-relative PTS; never reset window time',
                video_duration_semantics='full source container duration retained for the unchanged train/eval encoder',
                layout_duration_semantics='D=last real PTS minus first real PTS (point span); half-open upper bound excludes last source frame',
                window_rule='q=D/4,W=min(300,q); [i*q+(q-W)/2,i*q+(q-W)/2+W),i=0..3',
                actual_train_eval_cpu_tensor_identity_equal=complete,
                source_grouping_required='source_video_id+source_sha256; never count16 windows as16 independent sources',
                source_sha_aliases_against_protected_sources=0,protected_source_count=467,
                full_catalog_source_id_protection=True,
                source_or_old_data_modified=False,holdout_model_outputs_used=False,
                cache_bytes=sum(p.stat().st_size for p in cache.glob('*.npz')),
                view_files_sha256=files,selection_sha256=b.sha256_file(out/'SELECTION.json'),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                builder_sha256=b.sha256_file(__file__),encoder_sha256=b.sha256_file(REPRO/'qa_opd_fair/videoencode.py'),
                sampler_sha256=b.sha256_file(REPRO/'video_caption/build_real_video_pilot.py'),
                gpu_used=False,teacher_generated=False,training_started=False,training_authorized=False,
                limitations=['Nonzero source-clock positions are a new window-data convention and must match across all future arms.',
                             'This four-source engineering check does not validate all399 sources, training quality, GPU memory or speed.',
                             'Natural identical source pixels are allowed; selected source-frame indices are distinct.',
                             'Any future dataset split and uncertainty estimate must group by original source, not window ID.'],
                elapsed_s=time.time()-started)
    exclusive_json(out/'VALIDATION.json',report)
    if complete:
        exclusive_json(out/'READY_CPU_PROBE.json',dict(status='READY_16_WINDOW_CPU_PROBE_ONLY',n_sources=4,n_windows=16,
                       validation_sha256=b.sha256_file(out/'VALIDATION.json'),view_files_sha256=files,
                       all399_or1943_ready=False,training_authorized=False,gpu_used=False))
    state.update(phase='completed' if complete else 'incomplete_fixed_source_failure',elapsed_s=time.time()-started)
    atomic_json(out/'STATUS.json',state)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','sources')},indent=2),flush=True)
    if not complete:raise SystemExit(2)


if __name__=='__main__':
    main()
