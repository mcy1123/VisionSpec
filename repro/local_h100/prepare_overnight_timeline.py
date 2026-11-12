#!/usr/bin/env python3
"""Queue the user-authorized first night of old-task architecture exploration."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_h100.prepare_lambda10 import ROOT, PY, make_job, eval_job, passed, replace
from local_h100.supervise import atomic_json, validate_queue
from video_caption.build_pilot_schedule import build_schedule


def main():
    import pandas as pd
    folder=ROOT/'overnight_timeline_20261113'
    if folder.exists():raise FileExistsError(folder)
    queue_path=ROOT/'control/scheduler/queue.json'
    before=queue_path.read_bytes();queue=json.loads(before);old={j['id']:j for j in queue['jobs']}
    if shutil.disk_usage(ROOT).free<250*2**30:raise RuntimeError('need 250 GiB reserve')
    folder.mkdir()
    data=ROOT/'data/full';train=pd.read_parquet(data/'train_128.parquet');dev=pd.read_parquet(data/'dev_128.parquet')
    assert len(train)==80 and len(dev)==40
    assert set(train.video_id).isdisjoint(dev.video_id)
    from video_caption.build_real_video_pilot import PROMPT
    assert train['prompt'].eq(PROMPT).all() and dev['prompt'].eq(PROMPT).all()
    schedule=folder/'schedule.jsonl'
    rows=build_schedule(train.to_dict('records'),epochs=4,qa_per_update=4,seed=20261113)
    schedule.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    # Fixed longest-input gate, no outcome-based example selection.
    peak=train.sort_values(['input_ids_len','task'],ascending=[False,True]).iloc[:4]
    peak_path=folder/'gate_train.parquet';peak.to_parquet(peak_path,index=False)
    gate_schedule=folder/'gate_schedule.jsonl'
    gate_schedule.write_text(''.join(json.dumps(x)+'\n' for x in build_schedule(peak.to_dict('records'),epochs=2,qa_per_update=4,seed=20261113)))
    gpus={int(i):u.strip() for i,u in (line.split(',') for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    jobs=[];teachers=[];public=Path('/czsun/mcy/models/Qwen3-8B-DFlash-b16-public')
    for i in range(8):
        name=f'night13_teacher_{i}'
        argv=replace(old['teacher_train320_128f_shard0']['argv'],pool=data/'train_128.parquet',out=ROOT/'jobs'/name/'run',shards=8,shard=i)
        job=make_job(name,argv,gpus[i],inputs=[data/'train_128.parquet'])
        jobs.append(job);teachers.append(job)
    def sft(name,updates,init,gpu,deps,gate=False):
        argv=replace(old['factorial80_128f_A_plain_sft_seed20261112']['argv'],pool=peak_path if gate else data/'train_128.parquet',schedule=gate_schedule if gate else schedule,init_from=init,out=ROOT/'jobs'/name/'run',updates=updates,seed=20261113,save_every=updates)
        return make_job(name,argv,gpus[gpu],deps=[j['id'] for j in deps],requires=[passed(j) for j in deps],inputs=[init/'model.safetensors'])
    smoke=sft('night13_warm_gate2',2,public,4,teachers,True);jobs.append(smoke)
    smoke_eval=eval_job('night13_warm_gate2_dev2',ROOT/'jobs'/smoke['id']/'run/step2',gpus[4],data/'dev_128.parquet',deps=[smoke['id']],requires=[passed(smoke)],n=2);jobs.append(smoke_eval)
    warm=sft('night13_common_warm80',80,public,4,[smoke_eval]);jobs.append(warm)
    warmhead=ROOT/'jobs'/warm['id']/'run/step80'
    warm_eval=eval_job('night13_common_warm80_dev40',warmhead,gpus[4],data/'dev_128.parquet',deps=[warm['id']],requires=[passed(warm)]);jobs.append(warm_eval)
    initializers={}
    for bins in (1,2):
        name=f'night13_init_bins{bins}';out=folder/f'init_bins{bins}'
        j=make_job(name,[PY,'-u','repro/local_h100/init_matched_evidence.py','--source',str(warmhead),'--out',str(out),'--seed','20261113','--spatial-bins',str(bins)],gpus[5 if bins==1 else 4],deps=[warm['id']],requires=[passed(warm)],inputs=[warmhead/'model.safetensors'])
        j.pop('result');jobs.append(j);initializers[bins]=j
    records=[];final_evals=[]
    for index,(arm,lam) in enumerate((a,l) for a in ('A_plain','B_text','C_visual','D_visual_bins1') for l in (0,10)):
        bins=1 if arm.startswith('D_') else 2
        sourcearm='C_visual' if arm.startswith('D_') else arm
        init=warmhead if arm=='A_plain' else folder/f'init_bins{bins}'/sourcearm
        base=old[f'factorial80_128f_{sourcearm}_opd_seed20261112']
        gate_name=f'night13_{arm}_lambda{lam}_gate2';name=f'night13_{arm}_lambda{lam}_u80'
        dependencies=[warm_eval['id']]+([] if arm=='A_plain' else [initializers[bins]['id']])
        argv=replace(base['argv'],pool=peak_path,schedule=gate_schedule,init_from=init,out=ROOT/'jobs'/gate_name/'run',updates=2,seed=20261113,save_every=2,e2e_tv_weight=lam)
        gate=make_job(gate_name,argv,gpus[index],deps=dependencies,requires=[passed(warm_eval)],inputs=[init/'model.safetensors']);jobs.append(gate)
        eg=eval_job(gate_name+'_dev2',ROOT/'jobs'/gate_name/'run/step2',gpus[index],data/'dev_128.parquet',deps=[gate_name],requires=[passed(gate)],n=2);jobs.append(eg)
        argv=replace(argv,pool=data/'train_128.parquet',schedule=schedule,out=ROOT/'jobs'/name/'run',updates=80,save_every=80)
        formal=make_job(name,argv,gpus[index],deps=[eg['id']],requires=[passed(eg)],inputs=[init/'model.safetensors']);jobs.append(formal)
        final=eval_job(name+'_dev40',ROOT/'jobs'/name/'run/step80',gpus[index],data/'dev_128.parquet',deps=[name],requires=[passed(formal)]);jobs.append(final);final_evals.append(final)
        # Same trained head at64 frames, scheduled after its primary128 result.
        e64=eval_job(name+'_64f_dev40',ROOT/'jobs'/name/'run/step80',gpus[index],data/'dev_64.parquet',deps=[final['id']],requires=[passed(final)]);jobs.append(e64)
        records.append(dict(arm=arm,lambda_weight=lam,training=name,primary=final['id'],transfer64=e64['id'],gate=gate_name,gate_eval=eg['id'],init=str(init),updates=80,seed=20261113,gpu=gpus[index]))
    # All existing definitions preserved; readiness and GPU exclusivity handled by live scheduler.
    merged=dict(version=1,jobs=queue['jobs']+jobs)
    validate_queue(merged,ROOT)
    assert all(next(j for j in merged['jobs'] if j['id']==k)==v for k,v in old.items())
    for job in jobs:
        if '--save-every' in job['argv']:
            a=job['argv'];assert a[a.index('--save-every')+1]==a[a.index('--updates')+1]
    protocol=dict(schema='overnight_timeline_architecture_v1',created_utc=datetime.now(timezone.utc).isoformat(),user_authorized=True,status='QUEUED',jobs=[j['id'] for j in jobs],arms=records,
        warm_training=warm['id'],warm_evaluation=warm_eval['id'],warm_gate=smoke['id'],warm_gate_eval=smoke_eval['id'],
        budget=dict(warm_updates=80,formal_updates=640,gate_updates=18,total_optimizer_updates=738,qa_per_update=4,total_QA_exposures=2952),
        task=dict(prompt=PROMPT,train_sources=80,dev_sources=40,frames=128,max_new=1024,budget=63,branch=8,seed=20261113,source_split_seed=20261112),
        alignment='Old task/prompt/real-frame recipe and decoding; new local split and public initialization, NOT exact old checkpoint replication.',
        architectures=dict(A_plain='original',B_text='matched text readout',C_visual='existing visual readout spatial_bins2 time_encoding off',D_visual_bins1='same readout spatial_bins1 time_encoding off; tests pooling geometry, not temporal positional encoding'),
        training='common SFT80 from public head; each OPD arm fresh Adam from same common weights; gate updates discarded',
        checkpoint_policy='one terminal checkpoint per training job; save_every=updates; no periodic checkpoints; gate binaries retired after their paired evaluation by night reporter',
        limitations=['single-seed architecture screen','dev40 is reused development, not independent test','only existing lambda0/10 objectives tonight; new visual weighting is a later stage','SFT pilot and OPD pilot do not promise exact resume','no wallclock speed claims under concurrent load'],
        data_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (data/'train_128.parquet',data/'dev_128.parquet',data/'dev_64.parquet',schedule,peak_path,gate_schedule)},
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),no_automatic_retry=True,no_continuous_GPU_monitoring=True)
    atomic_json(folder/'PROTOCOL.json',protocol)
    atomic_json(folder/'queue.additions.json',dict(version=1,jobs=jobs))
    if queue_path.read_bytes()!=before:raise RuntimeError('queue changed; do not adopt stale snapshot')
    atomic_json(folder/'queue.before.json',queue)
    atomic_json(queue_path,merged)
    print(json.dumps(dict(added_jobs=len(jobs),optimizer_updates=738,folder=str(folder),status='QUEUED')))

if __name__=='__main__':main()
