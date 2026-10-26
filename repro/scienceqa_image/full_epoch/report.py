#!/usr/bin/env python3
"""Compile complete learning curves, paired final endpoint, and measured cost."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main(run):
    audit=json.loads((run/'DATA_AUDIT.json').read_text())
    endpoint=json.loads((run/'ENDPOINT.json').read_text())
    n=endpoint['update'];qa_count=endpoint['qa_per_arm']
    curves=[];train=[]
    for arm in ['sft','opd']:
        logs=[]
        for p in (run/'train'/arm).glob('*/log.jsonl'):
            logs.extend(dict(json.loads(l),arm=arm) for l in p.read_text().splitlines() if l)
        logs.sort(key=lambda r:r['update'])
        if [r['update'] for r in logs]!=list(range(1,n+1)):raise ValueError('missing or repeated formal updates')
        train.extend(logs)
        for p in (run/'eval').glob(f'{arm}_*_dev.json'):
            obj=json.loads(p.read_text());step=int(p.stem.split('_')[1])
            cov=json.loads(p.with_suffix('.coverage.json').read_text())
            rounds=[z for row in cov['rows'] for z in row['rounds']]
            curves.append(dict(arm=arm,update=step,images=obj['n_image_groups'],
                effective_tokens_per_round=float(np.mean([r['tokens_per_round'] for r in obj['rows']])),
                total_rounds=obj['total_rounds'],total_tokens=obj['total_tokens'],lossless=obj['lossless'],
                mean_topk_prefix_coverage=float(np.mean([r['topk_prefix_coverage'] for r in rounds])),
                mean_tree_prefix_coverage=float(np.mean([r['tree_prefix_coverage'] for r in rounds]))))
    curves.sort(key=lambda r:(r['arm'],r['update']))
    for name,rows in [('CURVES.csv',curves)]:
        with (run/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (run/'TRAINING_CURVES.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in train))
    finals={a:json.loads((run/'eval'/f'{a}_{n}_test.json').read_text()) for a in ['sft','opd']}
    valid_lossless=all(o['lossless'] for o in finals.values())
    paired={a:{r['image_group_id']:r for r in o['rows']} for a,o in finals.items()}
    if paired['sft'].keys()!=paired['opd'].keys() or len(paired['sft'])!=200:raise ValueError('confirmation pair mismatch')
    keys=sorted(paired['sft']);delta=np.array([paired['opd'][k]['tokens_per_round']-paired['sft'][k]['tokens_per_round'] for k in keys])
    rng=np.random.default_rng(20261019);boots=np.mean(rng.choice(delta,(10000,len(delta)),replace=True),axis=1)
    cost=[json.loads(p.read_text()) for p in (run/'stages').glob('*.json') if p.stem!='report']
    target={a:dict(forward_calls=sum(r.get('target_forward_calls',r.get('context_forwards',0)) for r in train if r['arm']==a),
        input_tokens=sum(r.get('target_input_tokens',r.get('context_tokens',0)) for r in train if r['arm']==a)) for a in ['sft','opd']}
    label_rows=[json.loads(l) for p in run.glob('labels_cost_*.jsonl') for l in p.read_text().splitlines() if l]
    target['sft_label_preparation']=dict(forward_calls=sum(r['target_forward_calls'] for r in label_rows),input_tokens=sum(r['target_input_tokens'] for r in label_rows))
    result=dict(updates_per_arm=n,qa_per_arm=qa_count,paired_images=200,endpoint=endpoint,
        lossless=valid_lossless,primary_comparison_valid=valid_lossless,
        opd_minus_sft_effective_tokens_per_round=float(delta.mean()) if valid_lossless else None,ci95=np.quantile(boots,[.025,.975]).tolist() if valid_lossless else None,
        final={a:dict(mean_effective_tokens_per_round=float(np.mean([r['tokens_per_round'] for r in o['rows']])),
                     exact=o['n_token_exact'],correct=o['answer_correct'],rounds=o['total_rounds']) for a,o in finals.items()},
        target_compute=target,cost=cost,total_gpu_hours=sum(r.get('gpu_hours',0) for r in cost),
        limitations=['single seed','native approximate teacher','teacher/gradient and recovery startup gates disabled by explicit user amendment; old diagnostic failure retained',
                    'dev metrics are SPEC-only; dev coverage follows emitted prefixes, not AR-verified correct prefixes',
                    'confirmation unseen within discovered project inventory, not foundation-model pretraining guarantee',
                    'instrumented/shared-host timing is not clean paired speedup evidence'])
    (run/'FINAL_RESULT.json').write_text(json.dumps(result,indent=2))
    text=f'''# ScienceQA paired experiment with development-set early stopping

Completed {n} updates and {qa_count} QA per arm, same Warm-2000, lr=1e-5 constant, fp32 Target / bf16 Head / fp32 CPU master AdamW.
Endpoint: {'paired early stopping' if endpoint['early_stopped'] else 'completed one epoch'}. Both arms stop at the same update; dev32 patience=3, min_delta=.01. No independent best-step selection per arm.

Final new image-level confirmation: OPD − SFT effective tokens/verification round = {result['opd_minus_sft_effective_tokens_per_round']}, paired bootstrap 95% CI {result['ci95']}.
Primary comparison valid only if both outputs are lossless: {valid_lossless}. SFT/OPD SPEC=AR: {result['final']['sft']['exact']}/200 and {result['final']['opd']['exact']}/200.
Measured allocated GPU-hours including preparation, checks, training, and evaluation: {result['total_gpu_hours']:.3f}.

See CURVES.csv (fixed SPEC-only dev32, emitted-prefix candidate coverage and effective throughput), TRAINING_CURVES.jsonl (every update), FINAL_RESULT.json (full cost, target computation and final metrics).

Limits: one seed; native approximate method, not serial gradient equivalence; startup diagnostic gates disabled by user; prior gradient gate failure retained; historical-image inventory and exact-pixel dedup do not prove semantic-near-duplicate or pretraining isolation; instrumented/shared-host timings are not clean paired speedup evidence.
'''
    (run/'FINAL_REPORT.md').write_text(text)
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for arm in ['sft','opd']:
            subset=[r for r in train if r['arm']==arm]
            ax=axes[0 if arm=='sft' else 1]
            ax.plot([r['update'] for r in subset],[r.get('loss',r.get('loss_weighted_mean')) for r in subset]);ax.set_title(arm+' objective (not cross-comparable)');ax.set_xlabel('update')
            subset=[r for r in curves if r['arm']==arm]
            axes[2].plot([r['update'] for r in subset],[r['effective_tokens_per_round'] for r in subset],label=arm)
        axes[2].legend();axes[2].set_title('dev32 effective tokens / round');axes[2].set_xlabel('update')
        fig.tight_layout();fig.savefig(run/'CURVES.png',dpi=160)
    except ImportError:pass

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
