#!/usr/bin/env python3
"""Paired SFT/native-OPD training, one-epoch maximum, development-set early stop."""
import argparse
import csv
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repro'))
from _bootstrap import configure
configure()
from scienceqa_image.full_epoch.data import WARM,WARM_SHA,sha,mounted,write_json
PYTHON='/home/mcy/.conda/envs/fin_research/bin/python'
BASE=ROOT/'repro/scienceqa_image'
WORKER=BASE/'full_epoch/worker.py'
DEFAULT='/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_20261019'
DIAGNOSTIC='/mnt/local3/mcy/GLANCE/results/scienceqa_full_epoch_20261019_recovery'


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default=DEFAULT,type=Path)
    p.add_argument('--diagnostic-root',default=DIAGNOSTIC,type=Path)
    p.add_argument('--gpus',default='1,2,6,7',help='four physical GPU indices; shared use allowed when memory fits')
    p.add_argument('--execute',action='store_true',help='actually load models and train; otherwise print plan')
    p.add_argument('--resume',action='store_true',help='resume completed stages only; never retry consumed updates')
    return p.parse_args()


class Controller:
    def __init__(self,a):
        self.a=a;self.run=mounted(a.run);self.diag=mounted(a.diagnostic_root)
        self.gpus=a.gpus.split(',')
        if len(set(self.gpus))!=4: raise ValueError('exactly four distinct GPUs required')
        self.audit=json.loads((self.run/'DATA_AUDIT.json').read_text())
        self.n=self.audit['updates'];self.grid=list(range(200,self.n,200))+[self.n]
        self.p=self.run/'dataset/pool.parquet';self.s=self.run/'dataset/schedule.jsonl'
        self.budget=self.run/'BUDGET.json';self.latest={};self.cost=[]
        self.env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                      TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(ROOT/'repro'),CUDA_DEVICE_ORDER='PCI_BUS_ID')
        self.stage_count=0

    def resources(self,devices,full=False):
        # Fit this fp32 Target/bf16 Head path; other GPU processes are allowed.
        # Prefer our observed peaks plus CUDA context overhead, with the known
        # ~40-GiB model path as the cold-start estimate. This is admission only:
        # a concurrent allocation can still cause a later CUDA OOM.
        required_mib = 40 * 1024
        for log in (self.run/'train').glob('*/*/log.jsonl'):
            with log.open() as stream:
                for line in stream:
                    if not line.strip():continue
                    record=json.loads(line)
                    peak_mib=max(record.get('peak_allocated',0)/2**20,
                                 record.get('peak_gib',0)*1024)
                    required_mib=max(required_mib,int(peak_mib+1024))
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
            rows={r[0].strip():r for r in csv.reader(raw.splitlines())}
            insufficient=[gpu for gpu in devices if float(rows[gpu][2])<required_mib]
            write_json(self.run/'RESOURCES.json',dict(time=time.time(),gpu_csv=raw,processes=procs,
                policy='shared_gpu_memory_fit',required_free_mib=required_mib,waiting_for=insufficient))
            if not insufficient:return
            if (self.run/'STOP').exists():raise RuntimeError('user requested STOP while waiting for GPUs')
            print(f'Waiting for {required_mib} MiB free on GPUs:',','.join(insufficient),flush=True)
            time.sleep(30)

    def command(self,arm,out,stop,resume=None,case=None):
        case=case or arm+'_main'
        shared=['--pool',str(self.p),'--schedule',str(self.s),'--init-from',WARM,'--out',str(out),
            '--updates',str(self.n),'--qa-per-update','4','--seed','20261013','--lr','1e-5',
            '--clip','1','--target-dtype','float32','--master-weights','cpu','--max-pixels','768',
            '--max-seq-len','4096','--save-every','200','--allow-partial-final',
            '--stop-file',str(self.run/'STOP'),'--stop-after',str(stop),'--update-budget',str(self.budget),'--update-case',case]
        if resume: shared+=['--resume-from',str(resume)]
        if arm=='opd':
            args=shared+['--lr-schedule','constant','--max-new-tokens','256']
            return [PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','--',str(BASE/'train_sqa_opd.py'),*args]
        return [PYTHON,str(BASE/'train_sqa_sft.py'),*shared,'--run',str(self.run),'--gpu','0',
                '--records',str(self.run/'dataset/labels.pt'),'--anchors-per-qa','6']

    def budget_init(self):
        from opd_official_kl.cli import parser
        from opd_official_kl.trainer import identity
        cases={}
        for arm in ['sft','opd']:
            for kind in ['main','audit']:
                name=f'{arm}_{kind}'
                spec=dict(updates=self.n if kind=='main' else 2,seed=20261013,
                          entry='scienceqa_pure_block' if arm=='opd' else 'scienceqa_sft')
                if arm=='opd':
                    cmd=self.command(arm,self.run/'unused',2,case=name)
                    args=parser().parse_args(cmd[cmd.index(str(BASE/'train_sqa_opd.py'))+1:])
                    spec['config_sha256']=identity(args)['config_sha256']
                cases[name]=spec
        if self.budget.exists():
            if json.loads(self.budget.read_text())['cases']!=cases: raise ValueError('budget identity changed')
        else: write_json(self.budget,dict(cases=cases,max_attempts=2*self.n+4,attempts=[],completed=[]))

    def stage(self,name,cmd,devices):
        done=self.run/'stages'/f'{name}.json'
        if done.exists():
            state=json.loads(done.read_text())
            if state['status']!='done': raise RuntimeError(f'{name} failed previously; automatic retry disabled, inspect {done}')
            print('skip completed',name,flush=True); return
        if (self.run/'STOP').exists(): raise RuntimeError('STOP requested before next stage')
        if devices: self.resources(devices)
        self.stage_count+=1
        log=self.run/'logs'/f'{name}.log'; log.parent.mkdir(exist_ok=True)
        env=dict(self.env,CUDA_VISIBLE_DEVICES=','.join(devices))
        write_json(self.run/'CURRENT.json',dict(stage=name,command=cmd,devices=devices,log=str(log)))
        write_json(self.run/'commands'/f'{name}.json',dict(command=cmd,CUDA_VISIBLE_DEVICES=','.join(devices)))
        print('starting',name,'log:',log,flush=True)
        start=time.time()
        with log.open('x') as f:
            result=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            try:
                result.wait()
            except BaseException:
                (self.run/'STOP').write_text('controller interrupted; requested update-boundary stop\n')
                # Do not terminate another job or auto-retry this one. Wait for our child.
                result.wait()
                raise

        record=dict(status='done' if result.returncode==0 else 'failed',returncode=result.returncode,
                    seconds=time.time()-start,gpus=len(devices),stage=name,command=cmd)
        record['gpu_hours']=record['seconds']*len(devices)/3600
        write_json(done,record)
        if (self.run/'STOP').exists(): raise RuntimeError('STOP requested; inspect saved checkpoint before resuming')
        if result.returncode: raise RuntimeError(f'{name} failed; see {log}; no automatic retry')

    def checkpoint(self,arm,phase,start,end,resume=None,case=None):
        root=self.diag if phase.startswith('audit') else self.run/'train'
        out=root/arm/phase
        candidates=[p for directory in (root/arm).glob(phase+'*') for p in directory.glob('step*')
                    if (p/'SHA256.json').exists() and start <= int(p.name[4:]) <= end]
        existing=max(candidates,key=lambda p:int(p.name[4:])) if candidates else None
        if existing and int(existing.name[4:]) == end:
            ck=existing
        else:
            if existing:
                start=int(existing.name[4:]);resume=existing
                out=root/arm/(phase+f'_resume_{start}')
                ledger=json.loads(self.budget.read_text())
                active_case=case or arm+'_main'
                if any(x['case']==active_case and start <= x['index'] < end for x in ledger['attempts']):
                    raise RuntimeError('uncheckpointed optimizer attempts exist; refusing automatic replay or budget expansion')
            cmd=self.command(arm,out,end-start,resume,case)
            self.stage(f'{arm}_{out.name}',cmd,self.gpus if arm=='opd' else [self.gpus[0]])
            ck=out/f'step{end}'
        hashes=json.loads((ck/'SHA256.json').read_text())
        # Trainer atomically published all files. Verify complete inventory and sizes here;
        # resume worker verifies full state identities and tensor contents.
        if any(not (ck/n).is_file() for n in hashes): raise RuntimeError('incomplete checkpoint')
        if not phase.startswith('audit'):
            self.latest[arm]=str(ck);write_json(self.run/'LATEST.json',self.latest)
            recovery=self.command(arm,self.run/'train'/arm/f'manual_from_{end}',max(1,self.n-end),ck)
            write_json(self.run/f'RECOVER_{arm}.json',dict(checkpoint=str(ck),step=end,command=recovery,
                CUDA_VISIBLE_DEVICES=','.join(self.gpus if arm=='opd' else [self.gpus[0]]),
                rule='same ledger; already consumed optimizer attempts cannot be retried; checkpoint is complete, uncheckpointed updates are not claimed recoverable'))
        return ck

    def evaluate(self,head,tag,split):
        out=self.run/'eval'/f'{tag}_{split}.json'
        if split=='dev':
            self.stage(f'eval_{tag}_{split}',[PYTHON,str(BASE/'full_epoch/dev_eval.py'),
                '--run',str(self.run),'--head',str(head),'--tag',tag,'--out',str(out)],[self.gpus[0]])
            return json.loads(out.read_text())
        self.stage(f'eval_{tag}_{split}',[PYTHON,str(BASE/'full_epoch/evaluate.py'),'--run',str(self.run),
            '--head',str(head),'--tag',tag,'--split',split,'--gpu','0','--head-dtype','float32',
            '--ar-cache',str(self.run/'eval'/f'ar_{split}.pt'),'--repeat-spec','0','--out',str(out)], [self.gpus[0]])
        gate=json.loads(out.with_name(out.stem+'_spec_equals_ar.json').read_text())
        if not gate['all_token_exact']:
            print(f'Observed SPEC != AR at {tag}/{split}; recorded, no lossless claim; continuing per requested policy',flush=True)
        return json.loads(out.read_text())

    def execute(self):
        for p,h in json.loads((self.run/'INPUTS.json').read_text()).items():
            if sha(p)!=h: raise ValueError(f'frozen input changed: {p}')
        for p,h in json.loads((self.run/'SOURCE.json').read_text()).items():
            if sha(p)!=h: raise ValueError(f'prepared source changed: {p}')
        if sha(Path(WARM)/'model.safetensors')!=WARM_SHA: raise ValueError('Warm changed')
        self.resources(self.gpus,full=True)
        self.budget_init()
        # User superseded the diagnostic gates: keep their old results, do not execute them.
        # AR label generation is actual SFT data preparation, not repeated testing.
        # Four independent deterministic shards; one frozen target per card.
        label_done=self.run/'stages/labels.json'
        if not label_done.exists():
            print('starting SFT label preparation (not SPEC=AR checking); logs/labels_0.log ... labels_3.log',flush=True)
            write_json(self.run/'CURRENT.json',dict(stage='labels',note='frozen Target AR training labels, four shards; no SPEC comparison'))
            self.resources(self.gpus)
            processes=[];opened=[];start=time.time()
            for i,gpu in enumerate(self.gpus):
                log=self.run/'logs'/f'labels_{i}.log';log.parent.mkdir(exist_ok=True)
                f=log.open('a');opened.append(f)
                cmd=[PYTHON,str(WORKER),'labels','--run',str(self.run),'--shard',str(i),'--shards','4']
                processes.append(subprocess.Popen(cmd,env=dict(self.env,CUDA_VISIBLE_DEVICES=gpu),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT))
            try:
                while any(p.poll() is None for p in processes):
                    if any(p.poll() not in [None,0] for p in processes):
                        raise RuntimeError('label shard failed')
                    time.sleep(1)
                if any(p.returncode for p in processes): raise RuntimeError('label shard failed')
            finally:
                for p in processes:
                    if p.poll() is None: p.terminate()
                for p in processes: p.wait()
                for f in opened:f.close()
            write_json(label_done,dict(status='done',stage='labels',seconds=time.time()-start,gpus=4,gpu_hours=(time.time()-start)*4/3600))
        import torch
        import pandas as pd
        labels=self.run/'dataset/labels.pt'
        if not labels.exists():
            records=[torch.load(self.run/'dataset/labels'/(task.replace(':','_')+'.pt'),map_location='cpu',weights_only=False)
                     for task in pd.read_parquet(self.p).task]
            temp=labels.with_suffix('.pending');torch.save(dict(records=records),temp);temp.replace(labels)
        warm=self.evaluate(WARM,'warm','dev')
        from scienceqa_image.full_epoch.early_stop import initial,observe
        def metric(result):
            # Development uses SPEC-only metrics by request; equality is tested only at the final endpoint.
            groups={}
            for row in result['rows']:
                groups.setdefault(row['image_group_id'],[]).append(row['tokens_per_round'])
            return sum(sum(v)/len(v) for v in groups.values())/len(groups)
        early_path=self.run/'EARLY_STOP.json'
        early=(json.loads(early_path.read_text()) if early_path.exists()
               else initial(dict(sft=metric(warm),opd=metric(warm))))
        write_json(early_path,early)
        states=dict(sft=None,opd=None)
        endpoint=self.n
        for end in self.grid:
            start=0 if end==self.grid[0] else self.grid[self.grid.index(end)-1]
            scores={}
            for arm in ['sft','opd']:
                states[arm]=self.checkpoint(arm,f'to_{end}',start,end,states[arm])
                scores[arm]=metric(self.evaluate(states[arm],f'{arm}_{end}','dev'))
            early=observe(early,end,scores)
            write_json(early_path,early)
            print('early stopping:',json.dumps(early['arms']),flush=True)
            if early['stop'] and end>=early['last_update']:
                endpoint=end
                print('Paired early stop at',end,'trigger:',early['trigger_arms'],flush=True)
                break
        schedule=[json.loads(line) for line in self.s.read_text().splitlines() if line.strip()]
        exposures=sum(row['update']<endpoint for row in schedule)
        write_json(self.run/'ENDPOINT.json',dict(update=endpoint,qa_per_arm=exposures,
            planned_updates=self.n,early_stopped=endpoint<self.n,
            reason='dev plateau/degradation' if endpoint<self.n else 'completed one epoch',
            checkpoints={arm:str(path) for arm,path in states.items()},
            selection='last common evaluated checkpoint; not different best steps per arm'))
        # The final shared endpoint, whether early-stopped or full epoch, sees test once.
        self.stage('oracle_test',[PYTHON,str(BASE/'build_eval_ar.py'),'--run',str(self.run),'--split','test',
            '--gpu','0','--out',str(self.run/'eval/ar_test.pt')],[self.gpus[0]])
        for arm in ['sft','opd']:self.evaluate(states[arm],f'{arm}_{endpoint}','test')
        self.stage('report',[PYTHON,str(BASE/'full_epoch/report.py'),'--run',str(self.run)],[])
        write_json(self.run/'DONE.json',dict(status='early_stopped' if endpoint<self.n else 'complete',
            updates_per_arm=endpoint,qa_per_arm=exposures,planned_updates=self.n))
        (self.run/'DONE.txt').write_text(f'Finished both arms at update {endpoint}. See FINAL_REPORT.md\n')


def main():
    a=arguments();c=Controller(a)
    plan=dict(run=str(c.run),gpus=c.gpus,max_qa_per_arm=c.audit['train_qa'],
        max_updates_per_arm=c.n,checkpoints=c.grid,lr=1e-5,lr_schedule='constant',seed=20261013,
        precision='Target fp32 / Head bf16 / CPU fp32 master AdamW',
        quick_checks='disabled by user; no teacher/gradient or 2-vs-1+1 gates',
        early_stopping=dict(metric='dev32 image-mean effective tokens/verification round',patience=3,min_delta=.01,rule='either arm plateaus/degrades -> stop both at same update'),
        spec_ar_policy='no training/dev AR check; final confirm200 only; mismatches invalidate lossless claims',
        primary_endpoint='last common checkpoint at paired early stop or one epoch; new confirm200 only after selection',
        no_early_success=True,execute=a.execute)
    print(json.dumps(plan,indent=2,ensure_ascii=False),flush=True)
    if not a.execute:return
    c.run.mkdir(parents=True,exist_ok=True)
    with (c.run/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (c.run/'STARTED.json').exists() and not a.resume:
            raise RuntimeError('already started; inspect logs then use --resume (no failed-stage retries)')
        write_json(c.run/'STARTED.json',dict(time=time.time(),plan=plan))
        write_json(c.run/'PROTOCOL.json',plan)
        try:c.execute()
        except BaseException as exc:
            c.latest.update({arm:str(max(paths,key=lambda p:int(p.name[4:]))) for arm in ['sft','opd'] if (paths:=[p for p in (c.run/'train'/arm).glob('*/step*') if (p/'SHA256.json').exists()])})
            write_json(c.run/'LATEST.json',c.latest)
            write_json(c.run/'STOPPED.json',dict(error=repr(exc),traceback=__import__('traceback').format_exc(),time=time.time(),latest=c.latest,
                rule='no automatic retries; only atomically completed checkpoints are recoverable'))
            (c.run/'DONE.txt').write_text('STOPPED: '+repr(exc)+'\nSee STOPPED.json, CURRENT.json and logs.\n')
            raise

if __name__=='__main__':main()
