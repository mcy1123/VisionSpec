#!/usr/bin/env python3
"""Fixed, balanced paired inference timing; never trains or changes deployment code."""
import argparse, csv, hashlib, itertools, json, os, random, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
import numpy as np
import torch
from PIL import Image
from shared import common
from scienceqa_image.eval_sqa import load_split
import glance
from glance.vl import VisionPositions


def write(path, data):
    Path(path).write_text(json.dumps(data,indent=2)+'\n')


def snapshot():
    fields={}
    for key, query in [('gpu',['--query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw']),('processes',['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'])]:
        fields[key]=subprocess.check_output(['nvidia-smi',*query,'--format=csv,noheader,nounits'],text=True).strip()
    return fields


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--n',type=int,default=24)
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--uuid',required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'measurements.jsonl').exists(),'Refusing to overwrite timings'
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.cuda.set_device(0)
    rows=load_split(a.run,'test');random.Random(20261020).shuffle(rows);rows=rows[:a.n]
    orders=list(itertools.permutations(('ar','sft','opd')))
    plan=[{'pid':r['pid'],'image_group_id':r['image_group_id'],'orders':[list(orders[i%6] if rep%2==0 else orders[i%6][::-1]) for rep in range(a.repeats)]} for i,r in enumerate(rows)]
    write(a.out/'PLAN.json',dict(run=str(a.run),n=len(rows),repeats=a.repeats,seed=20261020,uuid=a.uuid,target_dtype='float32',head_dtype='float32',budget=63,branch_k=8,max_new=256,max_pixels=768,primary='synchronized external wall time for full model generation including vision/text prefill, excluding IO/processor/model loading/head transfer',secondary='existing decoder decode_only window; SPEC excludes first draft preparation, AR excludes prefill',plan=plan))
    write(a.out/'LOAD_START.json',snapshot())
    target,proc=common.load_target(device='cuda:0',dtype=torch.float32)
    heads={arm:common.load_head(a.run/'train'/arm/'to_1182/step1182',dtype=torch.float32,device='cpu').eval() for arm in ('sft','opd')}
    oracle={r['pid']:r['tokens'] for r in torch.load(a.run/'eval/ar_test.pt',map_location='cpu',weights_only=False)['records']}
    eos=int(proc.tokenizer.eos_token_id);active=None
    def encode(row):
        img=Image.open(row['image_path']).convert('RGB')
        if max(img.size)>768:
            s=768/max(img.size);img=img.resize((max(1,int(img.width*s)),max(1,int(img.height*s))))
        ids,pix,grid=common.encode(proc,row['prompt'],img)
        return ids.cuda(),pix.cuda() if pix is not None else None,grid.cuda() if grid is not None else None
    def call(arm,inputs,max_new):
        nonlocal active
        if arm!='ar' and active!=arm:
            if active:heads[active].cpu()
            heads[arm].cuda();active=arm
        ids,pix,grid=inputs
        # Fresh KV / position adapter on every call. No cached AR timing used.
        pos=VisionPositions(pix,grid)
        torch.cuda.synchronize();start=time.perf_counter()
        if arm=='ar':out,st=glance.greedy_generate(target,ids,max_new,[eos],positions=pos,return_stats=True)
        else:out,st=glance.generate(heads[arm],target,ids,max_new,[eos],positions=pos,return_stats=True,budget=63,branch_k=8,temperature=0.,timing='decode_only',eos_stop='immediate')
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        tokens=out[0,ids.shape[1]:].tolist()
        return dict(arm=arm,wall_ms=elapsed*1000,decode_ms=st.ms_per_token*st.n_tokens,n_tokens=len(tokens),rounds=st.rounds,tokens=tokens)
    # Two unmeasured short calls per arm to warm model kernels and allocation.
    warm=encode(rows[0])
    for _ in range(2):
        for arm in ('ar','sft','opd'):call(arm,warm,64)
    del warm
    measurements=[];loads=[]
    for i,row in enumerate(rows):
        inputs=encode(row)
        for rep,order in enumerate(plan[i]['orders']):
            for arm in order:
                result=call(arm,inputs,256)
                result.update(pid=row['pid'],image_group_id=row['image_group_id'],repeat=rep,order=order,token_exact=result['tokens']==list(oracle[row['pid']]))
                measurements.append(result)
                with (a.out/'measurements.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                if not result['token_exact']:raise RuntimeError(f"token mismatch {row['pid']} {arm}; timings saved")
        loads.append(snapshot())
        write(a.out/'LOAD_SAMPLES.json',loads)
        print(f"{i+1}/{len(rows)} all token exact; latest order {order}",flush=True)
        del inputs
    arms={}
    means={arm:np.array([np.mean([m['wall_ms'] for m in measurements if m['pid']==row['pid'] and m['arm']==arm]) for row in rows]) for arm in ('ar','sft','opd')}
    rng=np.random.default_rng(20261020);indices=rng.integers(0,len(rows),size=(10000,len(rows)))
    for arm in ('ar','sft','opd'):
        mm=[m for m in measurements if m['arm']==arm]
        arms[arm]=dict(total_wall_s=sum(m['wall_ms'] for m in mm)/1000,mean_generation_ms=float(means[arm].mean()),wall_tokens_per_second=sum(m['n_tokens'] for m in mm)/(sum(m['wall_ms'] for m in mm)/1000),total_decode_s=sum(m['decode_ms'] for m in mm)/1000,total_tokens=sum(m['n_tokens'] for m in mm),total_rounds=sum(m['rounds'] for m in mm))
    ratios={}
    for ref,arm in [('ar','sft'),('ar','opd'),('sft','opd')]:
        boot=means[ref][indices].sum(axis=1)/means[arm][indices].sum(axis=1)
        ratios[f'{ref}_over_{arm}']=dict(speedup=float(means[ref].sum()/means[arm].sum()),image_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist())
    foreign=[]
    for load in loads:
        for p in csv.reader(load['processes'].splitlines()):
            if len(p)>1 and p[0].strip()==a.uuid and int(p[1].strip())!=os.getpid():foreign.append(p)
    summary=dict(status='complete',n_images=len(rows),repeats=a.repeats,arms=arms,ratios=ratios,all_token_exact=all(m['token_exact'] for m in measurements),foreign_process_samples_on_selected_gpu=foreign,timing_contaminated=bool(foreign),scope='24-image fixed seeded subset of existing confirm200, batch1, fp32 Target and Head, A6000; other GPUs shared; no training, decoder/coverage instrumentation or tuning',primary='ratio of total synchronized generation wall times, includes image/text prefill; excludes image IO/tokenization/transfers/model loading')
    write(a.out/'RESULT.json',summary);write(a.out/'DONE.json',{'status':'complete'})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
