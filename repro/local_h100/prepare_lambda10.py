#!/usr/bin/env python3
"""Prepare/adopt a finite matched lambda10 supplement without changing pilots."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_h100 import summarize_factorial80 as factorial
from local_h100.summarize_lambda10 import validate_training_pair
from local_h100.supervise import validate_queue, atomic_json

ROOT=factorial.ROOT
PY='/czsun/mcy/envs/visionspec/bin/python'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(argv,**values):
    result=list(argv)
    for key,value in values.items():
        option='--'+key.replace('_','-')
        result[result.index(option)+1]=str(value)
    return result


def make_job(name,argv,gpu,*,deps=(),requires=(),inputs=(),status='COMPLETED',result='run/RUN_STATUS.json'):
    directory=ROOT/'jobs'/name
    if directory.exists():raise FileExistsError(directory)
    return dict(id=name,gpu_uuids=[gpu],cwd=str(Path(__file__).resolve().parents[2]),
        job_dir=str(directory),argv=argv,depends_on=list(deps),requires=list(requires),
        inputs=list(map(str,inputs)),result=dict(path=str(directory/result),status=status))


def passed(job):
    return dict(exit=str(Path(job['job_dir'])/'EXIT.json'),
        result=job['result']['path'],status=job['result']['status'])


def eval_job(name,head,gpu,rows,*,deps,requires,n=0,profile=None):
    argv=[PY,'-u','repro/local_h100/eval_long_input.py' if profile else 'repro/evaluation/eval_video_opd.py',
        '--rows',str(rows),'--head',str(head),'--target','/czsun/mcy/models/Qwen3-VL-8B-Instruct',
        '--dtype','float32','--max-new','1024','--budget','63','--branch','8','--ar','--warmup','0',
        '--time-mode','real','--require-exact-time','--eos-stop','immediate','--tag',name,
        '--out',str(ROOT/'jobs'/name/'result.json'),'--n',str(n),'--evidence-intervention','normal']
    if profile:argv+=['--processor-profile',str(profile)]
    return make_job(name,argv,gpu,deps=deps,requires=requires,
        inputs=[rows,head/'model.safetensors',head/'config.json',head/'SHA256.json'],
        status='LOSSLESS_PASSED',result='result.json')


def preflight(record):
    argv=record['argv'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=record['gpu'],
        CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    runner="""import json,runpy,sys,torch
args=json.load(sys.stdin)
def forbidden(*a,**k):raise RuntimeError('CUDA initialization forbidden in lambda10 preflight')
torch.cuda._lazy_init=forbidden
sys.argv=args
runpy.run_path(args[0],run_name='__main__')
assert not torch.cuda.is_initialized()
"""
    result=subprocess.run([PY,'-c',runner],input=json.dumps(argv[2:]+['--dry-run']),
        text=True,capture_output=True,env=env,cwd=record['cwd'])
    if result.returncode:raise RuntimeError(record['id']+' preflight failed: '+result.stderr[-3000:])
    protocol=json.loads(result.stdout)
    baseline=factorial.read(ROOT/'jobs'/record['baseline_training']/'run/PROTOCOL.json')
    validate_training_pair(baseline,protocol)
    return dict(id=record['id'],status='EXACT_PROTOCOL_EXCEPT_LAMBDA',cuda_initialized=False,
        protocol=protocol,stderr_tail=result.stderr[-1500:])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt',action='store_true')
    args=parser.parse_args()
    plan_path=ROOT/'LAMBDA10_PROTOCOL.json'
    if plan_path.exists():raise FileExistsError('supplement already prepared; no automatic relaunch')
    if (ROOT/'control/STOP_TRAIN').exists():raise RuntimeError('STOP_TRAIN is set; refusing new training')
    queue_path=ROOT/'control/scheduler/queue.json';before=queue_path.read_bytes()
    previous=json.loads(before);old={j['id']:j for j in previous['jobs']}
    state=factorial.read(ROOT/'control/scheduler/STATE.json')['jobs']
    ready=ROOT/'data/dev40_long_input_fixed_spatial/READY.json'
    ready_data=factorial.read(ready)
    if ready_data['status']!='READY_ACTUAL80_ENCODINGS_VALIDATED':raise ValueError('long data not ready')
    data=ready.parent;profile=data/'PROCESSOR_PROFILE.json';dev=ROOT/'data/bounded_summary_v2/dev_128.parquet'
    uuid_by_index={int(i):u.strip() for i,u in (line.split(',') for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    gates=[];gates_dev=[];formal=[];evaluations=[];long_peaks=[];long_full512=[];long_full256=[];records=[]
    for arm,index in [('A_plain',2),('B_text',5),('C_visual',4)]:
        base_id=f'factorial80_128f_{arm}_opd_seed20261112'
        base=old[base_id];name=f'lambda10_{arm}_peak2';gpu=uuid_by_index[index]
        argv=replace(base['argv'],out=ROOT/'jobs'/name/'run',updates=2,
            schedule=ROOT/'control/schedule_expanded_peak_memory_128.jsonl',save_every=2,e2e_tv_weight=10)
        gate=make_job(name,argv,gpu,deps=[base_id],requires=[passed(base)],
            inputs=[plan_path,ROOT/'control/schedule_expanded_peak_memory_128.jsonl'])
        deployed=eval_job(name+'_dev2',ROOT/'jobs'/name/'run/step2',gpu,dev,
            deps=[name],requires=[passed(gate)],n=2)
        gates.append(gate);gates_dev.append(deployed)
        for seed in (20261112,20261113):
            base_id=f'factorial80_128f_{arm}_opd_seed{seed}';base=old[base_id]
            if state[base_id]['status']!='SUCCEEDED':raise ValueError('lambda0 training must be completed')
            name=f'factorial80_tv10_128f_{arm}_opd_seed{seed}';gpu=base['gpu_uuids'][0]
            argv=replace(base['argv'],out=ROOT/'jobs'/name/'run',e2e_tv_weight=10)
            job=make_job(name,argv,gpu,deps=[deployed['id']],requires=[passed(deployed)],
                inputs=[plan_path,ROOT/'FACTORIAL80_PROTOCOL.json'])
            formal.append(job)
            final=eval_job(name+'_eval80_dev40',ROOT/'jobs'/name/'run/step80',gpu,dev,
                deps=[name],requires=[passed(job)])
            evaluations.append(final)
            record=dict(arm=arm,seed=seed,training=name,baseline_training=base_id,
                evaluation80=final['id'],gpu=gpu,argv=argv,cwd=job['cwd'])
            records.append(record)
            prefix=f'long_tv10_{arm}_opd_seed{seed}'
            peak=eval_job(prefix+'_512f_peak1',ROOT/'jobs'/name/'run/step80',gpu,data/'peak_512.parquet',
                deps=[final['id']],requires=[passed(final)],profile=profile)
            long_peaks.append(peak);record['peak512']=peak['id']
            for frames,destination in [(512,long_full512),(256,long_full256)]:
                full=eval_job(prefix+f'_{frames}f_full40',ROOT/'jobs'/name/'run/step80',gpu,data/f'dev_{frames}.parquet',
                    deps=[peak['id']],requires=[passed(final),passed(peak)],profile=profile)
                destination.append(full);record[f'full{frames}']=full['id']
    # Read all actual protocols before adopting any new training. CPU only.
    with ThreadPoolExecutor(max_workers=2) as pool:preflights=list(pool.map(preflight,[dict(id=x['training'],**x) for x in records]))
    expected_storage=27*max((ROOT/'jobs'/r['baseline_training']/'run/step80/training_state.pt').stat().st_size
        +(ROOT/'jobs'/r['baseline_training']/'run/step80/model.safetensors').stat().st_size for r in records)
    free=shutil.disk_usage(ROOT).free
    if free<expected_storage+150*2**30:raise ValueError('insufficient checkpoint reserve')
    frozen={str(p):sha(p) for p in [ROOT/'FACTORIAL80_PROTOCOL.json',ready,
        ROOT/'LONG_INPUT_DEV40_PROTOCOL.json',ROOT/'LONG_INPUT_DEV40_ADOPTION_REQUIREMENTS.json',
        data/'SOURCE_PROVENANCE.json',data/'ENCODE_VALIDATION.json',data/'PROTOCOL.json',profile]}
    for relative,digest in ready_data['view_files_sha256'].items():
        if sha(data/relative)!=digest:raise ValueError('long view changed')
        frozen[str(data/relative)]=digest
    document=dict(schema='local_matched_lambda10_supplement_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        status='ADOPTED' if args.adopt else 'PREPARED_NOT_DISPATCHED',
        source_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        purpose='Complete the missing lambda0/10 comparison on the existing bounded task, not reproduce old-server checkpoint or1024-token task',
        changed_training_parameter_only='e2e_tv_weight:0->10',
        formal_arms=records,budget=dict(formal_arms=6,updates_per_arm=80,formal_updates=480,
            engineering_arms=3,engineering_updates_each=2,total_actual_updates=486,
            qa_per_update=4,total_actual_qa_exposures=1944,
            engineering_deployment_pairs=6,primary_dev40_pairs=240,long_input_pairs=486,total_AR_SPEC_pairs=732),
        initialization='same exact initial weights as each existing lambda0 arm; fresh Adam; no continuation from lambda0 step80 or engineering probes',
        precision='unchanged fp32 Target, bf16 original head, fp32 readout/master/Adam',
        data=dict(train_sources=320,dev_sources=40,frames_train=128,task='bounded three-part summary',
            long_frame_counts=[256,512],holdout_accessed=False,frozen_sha256=frozen),
        checkpoints='same20/40/60/80; pilot fullstates lack RNG, no resume claim; finite single-process runs',
        primary_analysis='per-seed lambda10-minus-lambda0; C-B withinlambda10; (C10-B10)-(C0-B0); lambda10-minus-existingSFT',
        statistics='complete40 paired sources and exact input/independentAR; exploratory reused dev, no seed pooling',
        memory='each architecture must pass two longest-input updates with full Adam states and independent AR dev2; no automatic retries',
        checkpoint_storage_estimate_bytes=expected_storage,free_bytes_at_prepare=free,
        monitoring_reenabled=False,preflights=preflights)
    summary=make_job('lambda10_final128_summary',[PY,'-u','repro/local_h100/summarize_lambda10.py'],
        uuid_by_index[0],deps=[j['id'] for j in evaluations],requires=[passed(j) for j in evaluations],
        inputs=[plan_path],result='unused.json',status='COMPLETE')
    summary['result']['path']=str(ROOT/'analysis/lambda10_summary.json')
    new=gates+gates_dev+formal+evaluations+[summary]+long_peaks+long_full512+long_full256
    # Existing critical C/B lambda0 long-input comparisons retain priority.
    critical=[];remainder=[]
    for j in previous['jobs']:
        if (j['id'].startswith('long_dev40_') and ('C_visual' in j['id'] or 'B_text' in j['id'])
                and '_opd_' in j['id'] and '_512f_' in j['id']):critical.append(j)
        else:remainder.append(j)
    # Dispatched definitions and all old IDs are retained verbatim. New lambda10
    # formal work comes before old secondary curves/256 conditions.
    dispatched=[j for j in remainder if state[j['id']].get('dispatched')]
    pending=[j for j in remainder if not state[j['id']].get('dispatched')]
    merged=dict(version=1,jobs=dispatched+critical+new+pending)
    validate_queue(merged,ROOT)
    oldids={j['id'] for j in previous['jobs']}
    if oldids&{j['id'] for j in new}:raise ValueError('duplicate supplement IDs')
    if not all(j==next(x for x in merged['jobs'] if x['id']==j['id']) for j in previous['jobs']):
        raise ValueError('old definition changed')
    atomic_json(ROOT/'control/lambda10_queue.proposed.json',dict(version=1,jobs=new))
    atomic_json(plan_path,document)
    if args.adopt:
        if queue_path.read_bytes()!=before:raise RuntimeError('queue changed during preflight; not adopted')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup=queue_path.with_name('queue.before_lambda10.'+stamp+'.json');backup.write_bytes(before)
        atomic_json(queue_path,merged)
        event=dict(utc=document['created_utc'],decision='user_authorized_matched_lambda10_supplement',
            new_jobs=len(new),total_actual_updates=486,protocol=str(plan_path),
            old_definitions_unchanged=True,old_queue_sha256=hashlib.sha256(before).hexdigest(),new_queue_sha256=sha(queue_path),
            backup=str(backup),monitoring_reenabled=False)
        with (ROOT/'DECISIONS.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
    print(json.dumps(dict(status=document['status'],jobs=len(new),budget=document['budget'],
        protocol=str(plan_path),preflights=[dict(id=x['id'],status=x['status']) for x in preflights])))


if __name__=='__main__':main()
