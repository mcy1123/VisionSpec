#!/usr/bin/env python3
"""Authorized fixed two-epoch extension, three cards / four logical QA slots."""
import argparse, csv, fcntl, json, os, shlex, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repro'))
from _bootstrap import configure
configure()
from scienceqa_image.full_epoch.data import WARM, write_json, sha
PYTHON='/home/mcy/.conda/envs/fin_research/bin/python'
BASE=ROOT/'repro/scienceqa_image'
DEFAULT=Path('/mnt/local/mcy/GLANCE/results/scienceqa_continue_ep23_3gpu_20261020')
PREVIOUS=Path('/mnt/local/mcy/GLANCE/results/scienceqa_full_epoch_restart_20261019_180942')
GPUS=['GPU-8af87d4f-b2b2-404f-e7b2-6b39cbedcd55','GPU-e1a07214-4092-280d-4e13-365bd4f4a439','GPU-be4999dd-0619-aeb4-c83d-97d249fd5476']
GRID=[1183,*range(1382,2364,200),2364,*range(2564,3546,200),3546]


def command(run,arm,out,stop,resume):
    shared=['--pool',str(run/'dataset/pool.parquet'),'--schedule',str(run/'dataset/schedule.jsonl'),
        '--init-from',WARM,'--out',str(out),'--updates','3546','--qa-per-update','4','--epoch-updates','1182',
        '--seed','20261013','--lr','1e-5','--clip','1','--target-dtype','float32','--master-weights','cpu',
        '--max-pixels','768','--max-seq-len','4096','--save-every','100000','--allow-partial-final',
        '--stop-file',str(run/'STOP'),'--stop-after',str(stop),'--resume-from',str(resume),
        '--update-budget',str(run/'BUDGET.json'),'--update-case',arm+'_ep23']
    if arm=='opd':
        return [PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=3','-m',
            'opd_official_kl.three_card_trainer',*shared,'--parallel-rollout','3','--lr-schedule','constant','--max-new-tokens','256']
    return [PYTHON,str(BASE/'train_sqa_sft_continuation.py'),*shared,'--run',str(run),
            '--gpu','0','--records',str(run/'dataset/labels.pt'),'--anchors-per-qa','6']


def opd_args(cmd):
    from opd_official_kl.cli import parser
    ap=parser()
    for action in ap._actions:
        if action.dest=='parallel_rollout':action.choices=[3];action.default=3
    ap.add_argument('--epoch-updates',type=int,default=1182)
    return ap.parse_args(cmd[cmd.index('--pool'):])


class Controller:
    def __init__(self,run):
        self.run=run.resolve();self.latest=json.loads((run/'LATEST.json').read_text())
        self.env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                      TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(ROOT/'repro'))
    def resources(self,devices):
        while devices:
            rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()))
            free={r[0].strip():float(r[1]) for r in rows}
            wait=[g for g in devices if free.get(g,0)<40*1024]
            write_json(self.run/'RESOURCES.json',dict(time=time.time(),free_mib=free,waiting_for=wait,policy='shared use allowed when 40 GiB free'))
            if not wait:return
            if (self.run/'STOP').exists():raise RuntimeError('STOP requested')
            print('waiting for enough free GPU memory',wait,flush=True);time.sleep(30)
    def stage(self,name,cmd,devices):
        done=self.run/'stages'/f'{name}.json'
        if done.exists():
            record=json.loads(done.read_text())
            if record['status']!='done':raise RuntimeError(f'failed stage requires inspection: {name}')
            return
        if (self.run/'STOP').exists():raise RuntimeError('STOP requested')
        self.resources(devices)
        log=self.run/'logs'/f'{name}.log'
        write_json(self.run/'CURRENT.json',dict(stage=name,command=cmd,gpus=devices,log=str(log)))
        write_json(self.run/'commands'/f'{name}.json',dict(command=cmd,CUDA_VISIBLE_DEVICES=','.join(devices)))
        print('starting',name,'log',log,flush=True)
        start=time.time()
        with log.open('x') as stream:
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(self.env,CUDA_VISIBLE_DEVICES=','.join(devices)),stdout=stream,stderr=subprocess.STDOUT)
            try:code=child.wait()
            except BaseException:
                (self.run/'STOP').write_text('controller interrupted; stop after completed update\n');child.wait();raise
        seconds=time.time()-start
        write_json(done,dict(stage=name,status='done' if code==0 else 'failed',returncode=code,seconds=seconds,gpus=len(devices),gpu_hours=seconds*len(devices)/3600))
        if code:raise RuntimeError(f'{name} failed; inspect {log}')
        if (self.run/'STOP').exists():raise RuntimeError('STOP requested; checkpoint saved by worker')
    def train(self,arm,end):
        current=Path(self.latest[arm]);start=int(current.name.removeprefix('step'))
        if start>=end:return
        ledger=json.loads((self.run/'BUDGET.json').read_text())
        if any(x['case']==arm+'_ep23' and start<=x['index']<end for x in ledger['attempts']):
            raise RuntimeError('uncheckpointed attempts exist; no silent replay')
        out=self.run/'train'/arm/f'to_{end}'
        self.stage(f'{arm}_{end}',command(self.run,arm,out,end-start,current),GPUS if arm=='opd' else GPUS[:1])
        ck=out/f'step{end}'
        hashes=json.loads((ck/'SHA256.json').read_text())
        assert all((ck/n).is_file() for n in hashes),'incomplete checkpoint'
        self.latest[arm]=str(ck);write_json(self.run/'LATEST.json',self.latest)
        recover=command(self.run,arm,self.run/'train'/arm/f'manual_from_{end}',3546-end,ck)
        write_json(self.run/f'RECOVER_{arm}.json',dict(checkpoint=str(ck),update=end,command=recover,
            CUDA_VISIBLE_DEVICES=','.join(GPUS if arm=='opd' else GPUS[:1]),
            note='Use only if no uncheckpointed attempts exist. Final 3546 means complete; do not extend budget.'))
    def evaluate(self,arm,end):
        self.stage(f'eval_{arm}_{end}',[PYTHON,str(BASE/'full_epoch/dev_eval.py'),
            '--run',str(self.run),'--head',self.latest[arm],'--tag',f'{arm}_{end}',
            '--out',str(self.run/'eval'/f'{arm}_{end}_dev.json')],GPUS[:1])
    def execute(self):
        for path,digest in json.loads((self.run/'SOURCE.json').read_text()).items():
            if sha(path)!=digest:raise RuntimeError(f'source changed: {path}')
        for path,digest in json.loads((self.run/'INPUTS.json').read_text()).items():
            if sha(path)!=digest:raise RuntimeError(f'input changed: {path}')
        for end in GRID:
            # The first real update of EACH arm is startup verification and counts
            # toward the fixed 2364-update extension. No separate smoke budget.
            for arm in ('opd','sft') if end==1183 else ('sft','opd'):
                self.train(arm,end)
                if end!=1183:self.evaluate(arm,end)
            write_json(self.run/'PROGRESS.json',dict(common_update=end,added_updates_per_arm=end-1182,
                planned_final_update=3546,checkpoints=self.latest,early_stop=False))
            if end==1183:write_json(self.run/'STARTUP_OK.json',dict(real_updates_per_arm=1,checkpoints=self.latest))
            self.stage(f'curves_{end}',[PYTHON,str(BASE/'full_epoch/continue_report.py'),'--run',str(self.run)],[])
        for arm in ('sft','opd'):
            self.stage(f'final_{arm}',[PYTHON,str(BASE/'full_epoch/evaluate.py'),'--run',str(self.run),
                '--head',self.latest[arm],'--tag',f'{arm}_3546','--split','test','--gpu','0','--head-dtype','float32',
                '--ar-cache',str(self.run/'eval/ar_test.pt'),'--repeat-spec','0','--out',str(self.run/'eval'/f'{arm}_3546_test.json')],GPUS[:1])
        self.stage('final_report',[PYTHON,str(BASE/'full_epoch/continue_report.py'),'--run',str(self.run),'--final'],[])
        write_json(self.run/'DONE.json',dict(status='complete',total_updates_per_arm=3546,additional_updates_per_arm=2364,
            additional_qa_per_arm=9452,checkpoints=self.latest,confirm200_status='reused descriptive set, not independent final confirmation'))
        (self.run/'DONE.txt').write_text('Completed 2 additional epochs per arm. See FINAL_REPORT.md\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=DEFAULT);a=ap.parse_args()
    c=Controller(a.run)
    with (a.run/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:c.execute()
        except BaseException as exc:
            write_json(a.run/'STOPPED.json',dict(error=repr(exc),traceback=traceback.format_exc(),latest=c.latest,time=time.time()))
            (a.run/'DONE.txt').write_text('STOPPED: '+repr(exc)+'\nSee STOPPED.json and logs.\n');raise
if __name__=='__main__':main()
