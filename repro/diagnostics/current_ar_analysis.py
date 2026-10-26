#!/usr/bin/env python3
"""CPU analyses kept on two separate axes: A relabelling vs B current AR."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from diagnostics.diagnostic_acl_metrics import native_objective, part_of, paired, local_oracles
from diagnostics.current_ar_closed_loop import OLD, read, write, leading


def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def csvwrite(p,rows):
    with open(p,'x') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def relabel(out):
    out.mkdir(exist_ok=False)
    d=OLD/'snapshots/phaseC5'
    aa={x['task']:x for x in lines(d/'d1_anchors.jsonl')}
    data={arm:{x['task']:x for x in lines(d/f'd1_{arm}.jsonl')} for arm in ['C0','C1','C2']}
    pervideo=[];changed=[]
    for task,a in aa.items():
        ns=[]
        for i,x in enumerate(a['anchors']):
            p=data['C0'][task]['anchors'][i]['p_arg']
            new=leading(ok and t==r for ok,t,r in zip(x['valid'],x['prop'],p))
            ns.append(new)
            if new!=x['n_acc']:changed.append({'video':task,'anchor_index':i,'pos':x['pos'],'old':x['n_acc'],'relabel':new})
        for mode in ['A_old','A_relabel']:
            for arm in data:
                anchors=[];cells=defaultdict(list)
                for i,x in enumerate(data[arm][task]['anchors']):
                    assert x['p_arg']==data['C0'][task]['anchors'][i]['p_arg']
                    n=x['n_acc_anchor'] if mode=='A_old' else ns[i]
                    anchors.append({**x,'n_acc_anchor':n})
                    for j,ok in enumerate(x['valid']):
                        part=part_of(j+1,n,ok)
                        if part:cells[part].append({k:x[k][j] for k in ['fkl','rkl','rank','hit']})
                native=native_objective(anchors)
                for part,values in cells.items():
                    pervideo.append({'video':task,'group':a['group'],'arm':arm,'reference_mode':mode,'part':part,
                        'n_positions':len(values),'native_replay_loss':native['loss'],
                        'fkl':float(np.mean([x['fkl'] for x in values])),
                        'rkl':float(np.mean([x['rkl'] for x in values])),
                        'rank':float(np.mean([x['rank'] for x in values])),
                        'top1':float(np.mean([x['hit'] for x in values])),
                        'rank_le8_proxy':float(np.mean([x['rank']<=8 for x in values])),
                        'rank_le32_proxy':float(np.mean([x['rank']<=32 for x in values]))})
    csvwrite(out/'per_video_A.csv',pervideo)
    stats={}
    warm=read(OLD/'manifest.json')['warmup_ids']
    for mode in ['A_old','A_relabel']:
        for group in ['g1','g2','g3']:
            for part in ['accepted','first_rejected','strict_suffix']:
                for left,right in [('C0','C2'),('C0','C1'),('C1','C2')]:
                    subset=[r for r in pervideo if r['group']==group and r['reference_mode']==mode and r['part']==part and r['video'] not in warm]
                    for metric in ['native_replay_loss','fkl','rkl','rank','top1','rank_le8_proxy','rank_le32_proxy']:
                        a={r['video']:r[metric] for r in subset if r['arm']==left}
                        b={r['video']:r[metric] for r in subset if r['arm']==right}
                        stats[f'{mode}/{group}/{part}/{right}-{left}/{metric}']=paired(a,b)
    write(out/'summary_A.json',{'changed_anchors':changed,'n_changed':len(changed),'n_total':sum(len(x['anchors']) for x in aa.values()),
        'pairs':stats,'reference_condition':'same legacy prefix, frozen C0 proposal, same teacher; only grouping changes',
        'not_available':['actual top8 IDs','target-token logprob','entropy','historical training error rate'],
        'no_A_to_B_context_pairing':True})


def analyse(run):
    records=lines(run/'trace_current_ar.jsonl')
    manifest=read(run/'manifest_final.json');assert manifest['status']=='completed'
    assert len(records)==manifest['formal_head_scores']==144
    c0={(r['video'],r['round_index']):r for r in records if r['arm']=='C0'}
    rows=[];part_rows=[];members=[]
    for arm in ['C0','C1','C2']:
        for v in manifest['videos']:
            task=v['video'];rs=[r for r in records if r['video']==task and r['arm']==arm]
            assert len(rs)==len(v['round_indices'])
            counts=Counter(r['oracle']['reason'] for r in rs)
            row={'video':task,'arm':arm,'n_anchor':len(rs),'reference_mode':'B_current_fixed_C0_tree_anchors',
                 'n_predictive_anchor':sum(r['H_valid']>0 for r in rs),
                 'n_valid_offsets':sum(r['H_valid'] for r in rs),
                 'L_raw':float(np.mean([r['oracle']['L_raw'] for r in rs])),
                 'L_tree':float(np.mean([r['oracle']['L_tree_oracle'] for r in rs])),
                 'L_actual':float(np.mean([r['oracle']['L_actual'] for r in rs])),
                 'tree_gap':float(np.mean([r['tree_gap'] for r in rs])),
                 'commit_gap':float(np.mean([r['commit_gap'] for r in rs])),
                 'H_valid':float(np.mean([r['H_valid'] for r in rs])),
                 **{k:counts[k] for k in ['raw_candidate_missing','tree_selection_loss','normal_termination','verification_commit_inconsistency_pending']},
                 'gate':'PASS','global_tau':'N/A','formal_speed':'N/A'}
            tokens=[m for r in rs for m in r['metrics']]
            for k in ['top1','top8','top32','fkl','rkl','js','rank','target_logp_in_draft']:
                row[k]=float(np.mean([t[k] for t in tokens])) if tokens else None
            rows.append(row)
            for part in ['tree_covered','next_uncovered','later_offsets']:
                ts=[t for t in tokens if t['C0_tree_group']==part]
                part_rows.append({'video':task,'arm':arm,'group':part,'n_positions':len(ts),
                     **{k:float(np.mean([t[k] for t in ts])) if ts else None for k in ['top1','top8','top32','rank','fkl','rkl','js','target_logp_in_draft']}})
            for r in rs:
                base=c0[(task,r['round_index'])]
                h=r['H_valid'];assert h==base['H_valid'] and r['context_sha256']==base['context_sha256']
                assert base['top1_leading_agreement']==leading(base['top8_ids'][i][0]==base['reference_tokens'][i] for i in range(h))
                # C0 top1 leading boundary is a separate label from C0 tree boundary.
                for i,t in enumerate(r['metrics']):
                    t['C0_top1_group']=part_of(i+1,base['top1_leading_agreement'])
                if arm!='C0':
                    setsame=[set(r['top8_ids'][i])==set(base['top8_ids'][i]) for i in range(h)]
                    ordersame=[r['top8_ids'][i]==base['top8_ids'][i] for i in range(h)]
                    members.append({'video':task,'arm':arm,'round_index':r['round_index'],'H_valid':h,
                        'same_top8_set_offsets':sum(setsame),'same_top8_order_offsets':sum(ordersame),
                        'all_valid_top8_sets_same':all(setsame),'delta_L_tree':r['oracle']['L_tree_oracle']-base['oracle']['L_tree_oracle'],
                        'delta_L_raw':r['oracle']['L_raw']-base['oracle']['L_raw'],
                        'C0_next_uncovered_top8':base['metrics'][base['oracle']['L_actual']]['top8'] if base['oracle']['L_actual']<h else None})
    pairs={};table=[]
    for left,right in [('C0','C2'),('C0','C1'),('C1','C2')]:
        for group in ['all','tree_covered','next_uncovered','later_offsets']:
            keys=['L_raw','L_tree','L_actual','tree_gap','commit_gap','top1','top8','top32','rank','fkl','js'] if group=='all' else ['top1','top8','top32','rank','fkl','rkl','js','target_logp_in_draft']
            rr=rows if group=='all' else [r for r in part_rows if r['group']==group]
            for metric in keys:
                a={r['video']:r[metric] for r in rr if r['arm']==left and r[metric] is not None}
                b={r['video']:r[metric] for r in rr if r['arm']==right and r[metric] is not None}
                result=paired(a,b);pairs[f'{right}-{left}/{group}/{metric}']=result
                table.append({'comparison':f'{right}-{left}','reference_mode':'B_current_fixed_C0_tree_anchors','group':group,
                              'metric':metric,'delta':result['mean'],'ci95':json.dumps(result['ci95']),
                              'n_video':result['n'],'n_anchor_per_video':8,'positive_videos':result.get('positive'),
                              'changed_videos':result.get('changed'),'gate':'PASS','not_global_tau':True})
    csvwrite(run/'per_video_current_ar.csv',rows);csvwrite(run/'diagnosis_current_ar.csv',table)
    csvwrite(run/'per_video_C0_tree_groups.csv',part_rows);csvwrite(run/'candidate_set_changes.csv',members)
    write(run/'summary_current_ar.json',{'pairs':pairs,'per_video':rows,'candidate_set_changes':members,
         'classification_counts':{arm:dict(Counter(r['oracle']['reason'] for r in records if r['arm']==arm)) for arm in ['C0','C1','C2']},
         'n_videos':6,'n_anchors':48,'formal_head_scores':144,'unit':'video, anchors/tokens nested',
         'all_commit_gaps_zero':all(r['commit_gap']==0 for r in records),
         'max_packed_vs_AR_abs_logit':max(r['packed_vs_AR_max_abs_logit'] or 0 for r in records),
         'no_A_B_causal_pairing':True,'no_local_oracle_to_global_tau_conversion':True})
    with open(run/'token_metrics_current_ar.jsonl','x') as f:
        for r in records:
            for t in r['metrics']:f.write(json.dumps({'video':r['video'],'arm':r['arm'],'round_index':r['round_index'],**t})+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['relabel','analyse']);ap.add_argument('--out',required=True)
    a=ap.parse_args();relabel(Path(a.out)) if a.mode=='relabel' else analyse(Path(a.out))
