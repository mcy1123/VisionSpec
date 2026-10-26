"""CPU-only progress curves and final descriptive report for ep2/ep3 continuation."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);ap.add_argument('--final',action='store_true');a=ap.parse_args()
    p=a.run;protocol=json.loads((p/'PROTOCOL.json').read_text());old=Path(protocol['parent_run'])
    with (old/'CURVES.csv').open() as f:curves=[{'arm':r['arm'],'update':int(r['update']),'effective_tokens_per_round':float(r['effective_tokens_per_round'])} for r in csv.DictReader(f)]
    for path in (p/'eval').glob('*_dev.json'):
        arm,step,_=path.stem.split('_');b=json.loads(path.read_text());groups={}
        for row in b['rows']:groups.setdefault(row['image_group_id'],[]).append(row['tokens_per_round'])
        curves.append(dict(arm=arm,update=int(step),effective_tokens_per_round=float(np.mean([np.mean(v) for v in groups.values()]))))
    curves.sort(key=lambda r:(r['arm'],r['update']))
    with (p/'CURVES.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','update','effective_tokens_per_round']);w.writeheader();w.writerows(curves)
    logs=[]
    for origin in (old,p):
        for arm in ('sft','opd'):
            for path in (origin/'train'/arm).glob('*/log.jsonl'):
                for line in path.read_text().splitlines():
                    r=json.loads(line);logs.append(dict(arm=arm,update=r['update'],loss=r.get('loss',r.get('loss_weighted_mean')),source=str(path)))
    with (p/'TRAINING_CURVES.jsonl').open('w') as f:
        for r in sorted(logs,key=lambda r:(r['arm'],r['update'])):f.write(json.dumps(r)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm in ('sft','opd'):
        z=[r for r in curves if r['arm']==arm];axes[0].plot([r['update'] for r in z],[r['effective_tokens_per_round'] for r in z],'-o',label=arm.upper())
        z=sorted([r for r in logs if r['arm']==arm],key=lambda r:r['update']);loss=np.array([r['loss'] for r in z]);x=[r['update'] for r in z]
        smooth=[float(np.mean(loss[max(0,i-99):i+1])) for i in range(len(loss))];axes[1].plot(x,smooth,label=arm.upper())
    axes[0].set_title('Fixed dev32 effective tokens / round');axes[1].set_title('Training loss (100-update trailing mean; different objectives)')
    for ax in axes:
        ax.set_xlabel('Cumulative optimizer updates');ax.legend();ax.grid(alpha=.25)
        for b in (1182,2364,3546):ax.axvline(b,color='gray',linestyle=':',alpha=.5)
    fig.tight_layout();fig.savefig(p/'CURVES.png',dpi=160);plt.close(fig)
    stages=[json.loads(f.read_text()) for f in (p/'stages').glob('*.json')]
    costs={}
    for s in stages:
        group='opd' if s['stage'].startswith('opd_') else 'sft' if s['stage'].startswith('sft_') else 'evaluation_or_reporting'
        costs[group]=costs.get(group,0)+s.get('gpu_hours',0)
    result=dict(status='complete' if a.final else 'running',protocol=protocol,cost_gpu_hours=costs,curves=curves)
    if a.final:
        per={};gates={}
        for arm in ('sft','opd'):
            b=json.loads((p/'eval'/f'{arm}_3546_test.json').read_text());gates[arm]=dict(exact=b['n_token_exact'],n=b['n_records'])
            groups={}
            for r in b['rows']:groups.setdefault(r['image_group_id'],[]).append(r['tokens_per_round'])
            per[arm]={k:float(np.mean(v)) for k,v in groups.items()}
        keys=sorted(set(per['sft']) & set(per['opd']));delta=np.array([per['opd'][k]-per['sft'][k] for k in keys]);rng=np.random.default_rng(20261020)
        result.update(final_output_checks=gates,reused_confirm200=dict(n=len(keys),sft_mean=np.mean(list(per['sft'].values())),opd_mean=np.mean(list(per['opd'].values())),opd_minus_sft=float(delta.mean()),ci95=np.quantile(delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1),[.025,.975]).tolist(),independent_confirmation=False))
    (p/('FINAL_RESULT.json' if a.final else 'PROGRESS_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n')
    if a.final:
        (p/'FINAL_REPORT.md').write_text('# ScienceQA 三卡续训两个 epoch\n\n两臂从各自 step1182 完整状态继续，到 step3546；各追加2364次更新、9452条QA曝光。固定预算，无指标早停。\n\n拓扑由4 rank迁移到3 rank/4逻辑slot；保留全部4条逻辑RNG流、Head、master和Adam，按slot0–3有序计算梯度。不声明跨拓扑数值逐位等价。\n\n曲线见 CURVES.png / CURVES.csv，原始训练日志及进度均保留。旧confirm200已参与研究决策，以下仅为复用集描述性结果，不能称为新的独立最终确认。\n\n```json\n'+json.dumps({k:v for k,v in result.items() if k not in ('protocol','curves')},indent=2)+'\n```\n')
if __name__=='__main__':main()
