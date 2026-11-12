#!/usr/bin/env python3
"""CPU read-only comparison of same-seed mature and factorial plain SFT runs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPRO))
from local_h100.build_prompt_view import exclusive_json

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [item for k in sorted(set(a)|set(b)) for item in diff(a.get(k),b.get(k),path+'.'+k)]
    return [] if a==b else [dict(path=path,mature=a,factorial_A=b)]


def argv_without_out(argv):
    i=argv.index('--out')
    return argv[:i]+argv[i+2:]


def header(path):
    with path.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        value=json.loads(stream.read(length))
    return length+8,{k:v for k,v in value.items() if k!='__metadata__'}


def compare_weights(first,second):
    import numpy as np
    base_a,head_a=header(first)
    base_b,head_b=header(second)
    if head_a.keys()!=head_b.keys():
        raise ValueError('tensor names differ between repeated configurations')
    tensors=[]
    for name in sorted(head_a):
        a,b=head_a[name],head_b[name]
        if a['shape']!=b['shape'] or a['dtype']!=b['dtype'] or a['dtype']!='BF16':
            raise ValueError('expected matching all-BF16 head tensors')
        count=math.prod(a['shape'])
        aa=np.memmap(first,mode='r',dtype='<u2',offset=base_a+a['data_offsets'][0],shape=(count,))
        bb=np.memmap(second,mode='r',dtype='<u2',offset=base_b+b['data_offsets'][0],shape=(count,))
        changed,maximum,squares,norm_a=0,0.0,0.0,0.0
        for start in range(0,count,1048576):
            x,y=aa[start:start+1048576],bb[start:start+1048576]
            changed+=int(np.count_nonzero(x!=y))
            xf=(x.astype(np.uint32)<<16).view(np.float32)
            yf=(y.astype(np.uint32)<<16).view(np.float32)
            delta=xf-yf
            maximum=max(maximum,float(np.max(np.abs(delta))))
            squares+=float(np.sum(delta*delta,dtype=np.float64))
            norm_a+=float(np.sum(xf*xf,dtype=np.float64))
        tensors.append(dict(name=name,shape=a['shape'],elements=count,changed_elements=changed,
                            changed_fraction=changed/count,max_abs_difference=maximum,
                            difference_l2=math.sqrt(squares),mature_l2=math.sqrt(norm_a)))
        del aa,bb
    count=sum(x['elements'] for x in tensors)
    changed=sum(x['changed_elements'] for x in tensors)
    squares=sum(x['difference_l2']**2 for x in tensors)
    norm=sum(x['mature_l2']**2 for x in tensors)
    return dict(elements=count,changed_elements=changed,changed_fraction=changed/count,
                tensors=len(tensors),tensors_with_changes=sum(x['changed_elements']>0 for x in tensors),
                max_abs_difference=max(x['max_abs_difference'] for x in tensors),
                difference_l2=math.sqrt(squares),relative_l2_difference=math.sqrt(squares/norm),
                rms_difference=math.sqrt(squares/count),per_tensor=tensors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(ROOT/'analysis/duplicate_plain_sft_diagnosis.json'))
    args=parser.parse_args()
    out=Path(args.out).resolve()
    if not out.is_relative_to(ROOT/'analysis') or out.exists():
        raise ValueError('new analysis output required')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    started=time.time()
    pairs=[]
    for seed in (20261112,20261113):
        jobs=[ROOT/'jobs'/f'mature_warm80_128f_seed{seed}',
              ROOT/'jobs'/f'factorial80_128f_A_plain_sft_seed{seed}']
        launches=[read(p/'LAUNCH.json') for p in jobs]
        protocols=[read(p/'run/PROTOCOL.json') for p in jobs]
        logs=[[json.loads(line) for line in (p/'run/log.jsonl').read_text().splitlines()] for p in jobs]
        if len(logs[0])!=80 or len(logs[1])!=80:
            raise ValueError('both repeated training runs must have completed 80 updates')
        updates=[]
        for a,b in zip(*logs):
            qa=[]
            for x,y in zip(a['qa'],b['qa']):
                if x['task']!=y['task'] or x['teacher_cache_key']!=y['teacher_cache_key']:
                    raise ValueError('repeated run received different scheduled task or teacher identity')
                qa.append(dict(task=x['task'],same_cache_key=True,
                               mature_cache_hit=x['teacher_cache_hit'],factorial_cache_hit=y['teacher_cache_hit'],
                               loss_difference=x['loss']-y['loss'],
                               other_field_differences=diff({k:v for k,v in x.items() if k not in ('loss','teacher_cache_hit')},
                                                           {k:v for k,v in y.items() if k not in ('loss','teacher_cache_hit')})))
            updates.append(dict(update=a['update'],loss_mature=a['loss'],loss_factorial_A=b['loss'],
                                loss_difference=a['loss']-b['loss'],
                                gradient_norm_mature=a['gradient_norm'],gradient_norm_factorial_A=b['gradient_norm'],
                                gradient_norm_difference=a['gradient_norm']-b['gradient_norm'],
                                written_update_norm_mature=a['written_update_norm'],
                                written_update_norm_factorial_A=b['written_update_norm'],qa=qa))
        commit_diff=subprocess.check_output(['git','diff','--name-status',launches[0]['source_commit'],
                                              launches[1]['source_commit']],cwd=REPRO.parent,text=True).splitlines()
        checkpoints=[]
        for step in (20,80):
            paths=[j/'run'/f'step{step}' for j in jobs]
            manifests=[read(p/'SHA256.json') for p in paths]
            checks=[]
            for p,manifest in zip(paths,manifests):
                actual=sha(p/'model.safetensors')
                if actual!=manifest['model.safetensors']:
                    raise ValueError('actual checkpoint weight SHA differs from published manifest')
                checks.append(dict(path=str(p),model_safetensors_sha256=actual,
                                   config_sha256=sha(p/'config.json'),
                                   identity=read(p/'CHECKPOINT_IDENTITY.json'),
                                   training_protocol_sha256=sha(p/'TRAINING_PROTOCOL.json'),
                                   optimizer_or_rng_state_saved=(p/'training_state.pt').exists()))
            comparison=compare_weights(paths[0]/'model.safetensors',paths[1]/'model.safetensors')
            checkpoints.append(dict(step=step,artifacts=checks,weight_comparison=comparison))
            print(json.dumps(dict(seed=seed,step=step,changed_fraction=comparison['changed_fraction'],
                                  relative_l2_difference=comparison['relative_l2_difference'],
                                  elapsed_s=time.time()-started)),flush=True)
        pairs.append(dict(seed=seed,jobs=[p.name for p in jobs],
                          launch_sha256=[sha(p/'LAUNCH.json') for p in jobs],
                          protocol_sha256=[sha(p/'run/PROTOCOL.json') for p in jobs],
                          log_sha256=[sha(p/'run/log.jsonl') for p in jobs],
                          argv_identical_except_out=argv_without_out(launches[0]['argv'])==argv_without_out(launches[1]['argv']),
                          protocol_differences=diff(*protocols),
                          recorded_data_model_and_code_hashes_identical=protocols[0]['hashes']==protocols[1]['hashes'],
                          commits=[l['source_commit'] for l in launches],commit_name_status_diff=commit_diff,
                          gpu_uuids=[p['visible_gpu_uuids'] for p in protocols],
                          all_320_qa_have_same_teacher_cache_key=True,
                          teacher_cache_misses=[sum(not q['teacher_cache_hit'] for step in log for q in step['qa']) for log in logs],
                          last_mature_cache_miss_update=max(step['update'] for step in logs[0] if any(not q['teacher_cache_hit'] for q in step['qa'])),
                          first_loss_difference_update=next((u['update'] for u in updates if u['loss_difference']!=0),None),
                          first_gradient_norm_difference_update=next((u['update'] for u in updates if u['gradient_norm_difference']!=0),None),
                          maximum_absolute_update_loss_difference=max(abs(u['loss_difference']) for u in updates),
                          update1_all_qa_losses_exactly_equal=all(q['loss_difference']==0 for q in updates[0]['qa']),
                          update1=updates[0],checkpoints=checkpoints,updates=updates))
    report=dict(schema='same_seed_duplicate_plain_sft_readonly_audit_v1',utc=datetime.now(timezone.utc).isoformat(),
                status='COMPLETED_OBSERVED_NUMERICAL_DIVERGENCE_CAUSE_NOT_EXPERIMENTALLY_ISOLATED',
                pairs=pairs,gpu_executed=False,training_executed=False,holdout_read_or_evaluated=False,
                conclusions=[
                    'These are two executions of the same configuration per seed, not independent method/model arms.',
                    'All recorded training/source/data/initialization hashes match; launch commit differences only add unrelated files.',
                    'All teacher keys match at every scheduled QA; cached and online collection did not change teacher trajectories.',
                    'All first-update QA losses match exactly, but first-update gradient norms and written-update norms already differ.',
                    'The earliest observable divergence is in backward/gradient reduction/update statistics, before update2 forward losses diverge.',
                    'Uncontrolled CUDA numeric nondeterminism in BF16 SDPA/backward is the leading explanation; actual kernel/backend was not recorded.',
                    'Cache misses also change allocation and kernel execution history, so their numerical contribution cannot be isolated from these runs.'],
                static_code_evidence={
                    'greedy_generation':'greedy_generate is scoped torch.inference_mode, uses argmax, has no random sampling or persistent grad-mode/precision mutation.',
                    'rng_anchor_plan':'Each QA uses a fresh CPU torch.Generator seeded by seed + cursor*100003 + slot; global RNG advances cannot change reveal plan.',
                    'dropout':'Both models are eval; differentiable sft_blocks calls SDPA with dropout_p=0.0.',
                    'target_state':'Before every training context replay target.model.rope_deltas is reset; replay has fresh DynamicCache and explicit position_ids.',
                    'determinism':'The entry does not enable deterministic algorithms or pin the SDPA backend; launcher inherits unrecorded environment.',
                    'precision':'Recorded target fp32, head bf16, GPU master/Adam fp32 are identical; no autocast/precision changes in cache generation.',
                    'incidental_global_seed_reset':'common.use_fast_patch_embed verify calls torch.manual_seed(0) on both paths during loading; unrelated to same-seed cache hit/miss divergence, but global seed alone is not a complete RNG provenance.',
                    'checkpoint_limit':'Weights-only pilot checkpoints contain no saved per-step gradients, optimizer moments or RNG state.'},
                possible_future_disambiguation_not_executed=[
                    'Same GPU, same completed teacher cache, two update1 replays with per-QA gradients and runtime backend/determinism flags recorded.',
                    'Controlled cache-hit versus forced-online teacher generation from identical RNG/model state, recording RNG/grad-mode before and after and actual target-context hashes.',
                    'A separate explicit deterministic SDPA/math-backend diagnostic if supported and memory-safe; do not silently change an active experiment.'],
                source_sha256={str(p.relative_to(REPRO.parent)):sha(p) for p in [Path(__file__),
                    REPRO/'local_h100/train_video_sft.py',REPRO/'local_h100/train_video_sft_gpu_master.py',
                    REPRO/'shared/sft_blocks.py',REPRO/'shared/master_optim.py',REPRO/'shared/common.py',
                    REPRO.parent/'GLANCE_repo/glance/decode.py',REPRO/'local_h100/run_job.py']},
                elapsed_s=time.time()-started)
    exclusive_json(out,report)
    print(out,flush=True)


if __name__=='__main__':
    main()
