#!/usr/bin/env python3
"""Read-only CPU reanalysis of phaseC4/C5. Writes only to a NEW output folder.

No GPU model loading, optimisation, downloads, test-set reads, or job launches.
Existing metrics are explicitly labelled A (legacy trajectory teacher replay),
never relabelled B (correct incremental AR). Missing traces remain null/N/A.
"""
from __future__ import annotations

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from shared import common
from diagnostics.diagnostic_acl_metrics import native_objective, paired, part_of


def read(p):
    return json.loads(Path(p).read_text())


def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def capture(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return {"command": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'snapshots').mkdir()
    root = Path(common.ROOT)
    c4, c5 = root/'results/phaseC4', root/'results/phaseC5'
    sources = {}

    def snap(p):
        p = Path(p)
        target = out/'snapshots'/p.parent.name/p.name
        target.parent.mkdir(exist_ok=True)
        before = digest(p)
        shutil.copyfile(p, target)
        if before != digest(target) or before != digest(p):
            raise RuntimeError(f'input changed during snapshot: {p}')
        sources[str(p)] = {"sha256": before, "snapshot": str(target)}
        return target

    split = read(snap(c5/'split.json'))
    anchors = {r['task']: r for r in lines(snap(c5/'d1_anchors.jsonl'))}
    metrics = {arm: {r['task']: r for r in lines(snap(c5/f'd1_{arm}.jsonl'))}
               for arm in ('C0','C1','C2')}
    logs = {arm: lines(snap(c4/f'{arm}.jsonl')) for arm in ('C1','C2')}
    for p in [c4/'C1.log',c4/'C2.log',c5/'d1_anchors.log']:
        snap(p)
    cfg2 = read(snap(root/'checkpoints/phaseC4_C2/opd_fullkl_config.json'))
    paths = {'C0':root/'results/phaseA4/sft439_step1000_fresh30_tree_float32.json',
             'C1':c4/'phaseC4_C1_update10_fresh30.json',
             'C2':c4/'phaseC4_C2_update10_fresh30.json'}
    evals = {k: read(snap(p)) for k,p in paths.items()}
    heads = {k:Path(v['config']['head']) for k,v in evals.items()}
    assert heads['C0'] == Path(cfg2['init_from'])
    assert heads['C2'] == Path(cfg2['out'])/'step10'
    launch = _source_path('run_phaseC4_10upd.sh').read_text()
    assert '--out "$R/checkpoints/phaseC4_C1"' in launch and heads['C1'].name == 'step10'
    assert str(heads['C1']) == str(root/'checkpoints/phaseC4_C1/step10')
    head_before = {k:{'path':str(p), 'model_sha256':digest(p/'model.safetensors'),
                       'config_sha256':digest(p/'config.json'), 'config':read(p/'config.json'),
                       'source':str(paths[k]), 'optimizer_master_exists':(p/'optimizer_master.pt').exists()}
                   for k,p in heads.items()}
    for k in heads:
        assert all(evals[k]['config'][x] == evals['C0']['config'][x]
                   for x in ['rows','max_new','budget','branch','dtype','ar','warmup'])
    pool_path = root/'data/pool_499.parquet'
    fresh_path = root/'data/video_fresh30.parquet'
    pool = pd.read_parquet(snap(pool_path)).set_index('video_name')
    fresh = pd.read_parquet(snap(fresh_path))
    records_path = Path(cfg2['records'])
    records = torch.load(records_path, weights_only=False, map_location='cpu')['records']
    recs = {str(r['task']):r for r in records}
    train = [r for r in records if pool.loc[r['task'],'split']=='train']
    rng = np.random.default_rng(0)
    order1 = rng.permutation(len(train))
    order2 = list(range(len(train))); random.Random(cfg2['seed']).shuffle(order2)
    c1 = [train[int(i)]['task'] for i in order1[:10]]
    c2 = [train[i]['task'] for i in order2[:10]]
    assert c1 == split['groups']['g1_phaseC4_trained']['C1_order']
    assert c2 == split['groups']['g1_phaseC4_trained']['C2_order']
    assert [int(recs[t]['input_ids'].numel()) for t in c1] == [x['L'] for x in logs['C1']]
    assert len(logs['C1']) == len(logs['C2']) == 10
    assert all(x['videos']==1 and x['anchors']==24 and x['anchorless_videos']==0
               and x['no_anchor_videos']==0 for x in logs['C2'])
    # Reproduce C1 supervision planning on CPU using the actual pure functions.
    from historical import train_head as th
    supervision = []
    for step, task in enumerate(c1):
        r = recs[task]; full = r['input_ids'][:2048].long()
        aa = list(range(int(r['prompt_len']),len(full)-1))
        reveals = th.sample_reveals(len(aa),16,torch.Generator().manual_seed(step),'cpu')
        aa, reveals = th.live_anchors(aa,reveals,len(full),16)
        if len(aa)>24:
            keep=rng.choice(len(aa),24,replace=False);keep.sort()
            aa=[aa[i] for i in keep];reveals=reveals[torch.as_tensor(keep)]
        _,_,mask,w=th.build_blocks(full,aa,reveals,16,common.MASK_ID,'cpu')
        supervision.append({'step':step+1,'video':task,'anchors':len(aa),
                            'positions':int(mask.sum()),'weight_sum':float(w.double().sum()),
                            'anchor_positions':aa,'reveals':reveals.tolist()})
    exposure = {'C1_order':c1,'C2_order':c2,'intersection':sorted(set(c1)&set(c2)),
                'C1_only':sorted(set(c1)-set(c2)), 'C2_only':sorted(set(c2)-set(c1)),
                'identity_evidence':'reconstructed from config/code + C1 ten logged lengths; C2 logged one video/update, zero skips and existing update-1 replay. Neither log stores video IDs; not direct-ID log evidence.',
                'C1_supervision_replay':supervision,
                'C1_positions':sum(r['positions'] for r in supervision),
                'C1_weight_sum':sum(r['weight_sum'] for r in supervision),
                'C2_accepted':sum(r['n_acc_offsets'] for r in logs['C2']),
                'C2_rejected':sum(r['n_rej_offsets'] for r in logs['C2']),
                'C2_rejected_weight_sum':sum(r['Z_total'] for r in logs['C2']),
                'C2_updates':logs['C2']}
    warmups={r['video'] for r in evals['C0']['rows'] if r['warmup']}
    assert len(warmups)==1
    val=[v for v in fresh.video_name if v not in warmups]
    g2=split['groups']['g2_train_not_sampled_by_phaseC4']['videos']
    groups={'C1_only':exposure['C1_only'],'C2_only':exposure['C2_only'],
            'unexposed_this_round_union_matched20':g2,
            'unexposed_this_round_arm_matched10':g2[:10], 'validation29':val}
    exposure['groups']=groups
    exposure['group_intersections']={f'{x}/{y}':sorted(set(groups[x])&set(groups[y]))
                                    for i,x in enumerate(groups) for y in list(groups)[i+1:]}
    exposure['control_rule']='Reuse pre-score phaseC5-g2 hash order. First ten provide equal-size per-arm comparison; all twenty retained. Not matched on length; not never-trained.'
    data_manifest=[]
    for task in sorted(anchors):
        r=pool.loc[task]; video=Path(r.cache_path)
        data_manifest.append({'video':task,'source_id':task.removeprefix('v_'),
            'cache_path':str(video),'video_sha256':digest(video),
            'question':str(r.question),'prompt_sha256':hashlib.sha256(str(r.question).encode()).hexdigest(),
            'fps':float(r.fps),'n_frames_declared':int(r.n_frames),'grid_declared':str(r.grid),
            'prompt_len_record':int(recs[task]['prompt_len']),
            'input_ids_sha256':hashlib.sha256(recs[task]['input_ids'].numpy().tobytes()).hexdigest(),
            'split':str(r.split),'exposed_C1':task in c1,'exposed_C2':task in c2,
            'frame_indices':None,'time_encoding':None,
            'alignment_status':'historical records; exact processor input replay not run'})
    content=Counter(x['video_sha256'] for x in data_manifest)
    exposure['duplicate_video_contents']=[x for x in data_manifest if content[x['video_sha256']]>1]
    exposure['same_source_limit']='source identity uses pool video_name and cached-byte hash; original URL/segment genealogy absent'
    write(out/'d3_exposure_manifest.json',exposure)
    write(out/'data_manifest.json',data_manifest)
    # Cross-check stored trajectories against independently saved incremental AR.
    armap={r['video']:r for r in evals['C0']['rows']}
    for arm in ['C0','C1','C2']:
        p=c5/f'phaseC5_tree_{arm}_g1g2.json'
        if p.exists():
            data=read(snap(p))
            evals[arm+'_g12']=data
            if arm=='C0':
                armap.update({r['video']:r for r in data['rows']})
    alignment=[]
    for task, row in armap.items():
        old=recs[task]['input_ids'].tolist()[int(recs[task]['prompt_len']):]
        ar=row['ar_ids']; m=next((i for i,(x,y) in enumerate(zip(old,ar)) if x!=y),min(len(old),len(ar)))
        full_match=old==ar
        prefix_ok=[a['pos']-int(recs[task]['prompt_len']) < m for a in anchors[task]['anchors']]
        alignment.append({'video':task,'old_generated_length':len(old),'ar_length':len(ar),
                          'equal':full_match,'common_prefix_length':m,
                          'old_token_at_divergence':old[m] if m<len(old) else None,
                          'ar_token_at_divergence':ar[m] if m<len(ar) else None,
                          'anchors':len(prefix_ok),'anchors_in_shared_AR_prefix':sum(prefix_ok),
                          'prompt_lengths_equal':row['prompt_len']==int(recs[task]['prompt_len'])})
    write(out/'trajectory_ar_alignment.json',alignment)
    # Per-video A-reference metrics, fixed anchor part assignment. No B claims.
    pairing_errors=[]; mismatch=[]; token_file=open(out/'d1_tokens_A.jsonl','w')
    pervideo={}; teacher_checks=0
    for task, ref in metrics['C0'].items():
        base=anchors[task]
        for arm in ['C0','C1','C2']:
            rr=metrics[arm][task]
            assert len(rr['anchors'])==len(base['anchors'])
            cells={p:[] for p in ['accepted','first_rejected','strict_suffix']}
            replay=native_objective(rr['anchors'])
            bad_acc=bad_first=teacher_rollout_disagree=0
            for j,(aa,bb) in enumerate(zip(rr['anchors'],base['anchors'])):
                a0=ref['anchors'][j]
                assert aa['pos']==bb['pos'] and aa['valid']==bb['valid'] and aa['n_acc_anchor']==bb['n_acc']
                if aa['p_arg']!=a0['p_arg'] or aa['teacher_bits']!=a0['teacher_bits']:
                    pairing_errors.append([arm,task,j])
                if arm=='C0':
                    teacher_checks+=1
                # n_acc is against stored token IDs, not necessarily teacher greedy.
                native_n=0
                for i,(q,p) in enumerate(zip(bb['prop'],aa['p_arg'])):
                    if not aa['valid'][i] or q!=p:break
                    native_n+=1
                teacher_rollout_disagree+=int(native_n!=bb['n_acc'])
                for i,valid in enumerate(aa['valid']):
                    part=part_of(i+1,bb['n_acc'],valid)
                    if part is None:continue
                    rank=aa['rank'][i]
                    cell={'fkl':aa['fkl'][i],'rkl':aa['rkl'][i],'rank':rank,
                          'top1':float(aa['hit'][i]),'rank_le8':float(rank<=8),'rank_le32':float(rank<=32)}
                    cells[part].append(cell)
                    if arm=='C0':
                        bad_acc+=int(part=='accepted' and bb['prop'][i]!=aa['p_arg'][i])
                        bad_first+=int(part=='first_rejected' and bb['prop'][i]==aa['p_arg'][i])
                    token={'arm':arm,'video':task,'anchor_index':j,'pos':aa['pos'],'offset':i+1,
                           'part':part,'reference_mode':'A_legacy_trajectory_draft_induced',
                           'teacher_prefix_id':hashlib.sha256(json.dumps([task,aa['pos'],bb['prop'][:i]]).encode()).hexdigest(),
                           'c0_proposal_token':bb['prop'][i],'teacher_token':aa['p_arg'][i],**cell}
                    token_file.write(json.dumps(token)+'\n')
            vv={'native_replay':replay}
            for part,cs in cells.items():
                vv[part]={'positions':len(cs),**({k:float(np.mean([c[k] for c in cs])) for k in cs[0]} if cs else {})}
            pervideo.setdefault(arm,{})[task]=vv
            if arm=='C0':
                mismatch.append({'video':task,'accepted_but_teacher_disagrees':bad_acc,
                                 'first_rejected_but_teacher_agrees':bad_first,
                                 'anchors_nacc_differs_from_teacher':teacher_rollout_disagree})
    token_file.close()
    assert not pairing_errors, pairing_errors[:3]
    write(out/'d1_per_video_A.json',pervideo)
    write(out/'d1_reference_mismatch.json',mismatch)
    summary={'pairs':{},'d2':{},'reference_alignment':alignment,'teacher_checks':teacher_checks,
             'teacher_pairing_errors':pairing_errors,'native_split_mismatch':mismatch}
    table=[]
    for group,ids in groups.items():
        for left,right in [('C0','C1'),('C0','C2'),('C1','C2')]:
            key=f'{right}-{left}/{group}'
            pair={}
            for part in ['native_replay','accepted','first_rejected','strict_suffix']:
                keys=['loss'] if part=='native_replay' else ['fkl','rkl','rank','top1','rank_le8','rank_le32']
                for metric in keys:
                    aa={t:pervideo[left][t][part][metric] for t in ids if metric in pervideo[left][t][part]}
                    bb={t:pervideo[right][t][part][metric] for t in ids if metric in pervideo[right][t][part]}
                    pair[f'{part}/{metric}']=paired(aa,bb)
            summary['pairs'][key]=pair
            for part in ['accepted','first_rejected','strict_suffix']:
                table.append({'comparison':f'{right}-{left}','data_group':group,
                  'reference_mode':f'A_legacy/{part}', 'loss_delta':pair['native_replay/loss']['mean'],
                  'part_fkl_delta':pair[part+'/fkl']['mean'],'part_rkl_delta':pair[part+'/rkl']['mean'],
                  'rank_delta':pair[part+'/rank']['mean'],'top1_delta':pair[part+'/top1']['mean'],
                  'top8_delta':pair[part+'/rank_le8']['mean'],'top32_delta':pair[part+'/rank_le32']['mean'],
                  'first_rejection_or_oracle_delta':'N/A','tau_delta':'N/A','elapsed_s':'N/A',
                  'n':pair[part+'/rank']['n'], 'gate':'A teacher checksum+argmax paired; B not verified',
                  'limitation':'top-k estimated by strict rank; original top-k IDs/ties unavailable. Loss is full-KL replay with fp32 head, not historical bf16 update or C1 CE.'})
    # Exact AR checks, legacy tau, terminal-clipped lengths and timing, all per video.
    rowmetrics={}; eval_gates={}
    def measure(name,blob):
        measured={}; failures=[]
        for r in blob['rows']:
            ok=r['spec_ids']==r['ar_ids'] and r['lossless'] is True
            if not ok:failures.append(r['video'])
            clipped=[max(0,min(n,r['gen_tokens']-p-1)) for p,n in zip(r['anchor_rel'],r['accepted'])]
            measured[r['video']]={**r,'clipped_draft_accepted':sum(clipped),
                'raw_draft_accepted':sum(r['accepted']),
                'terminal_overcount':sum(r['accepted'])-sum(clipped),
                'round_progress_tau':r['gen_tokens']/r['rounds'],
                'spec_decode_s':r['spec_decode_ms_per_tok']*r['gen_tokens']/1000}
        eval_gates[name]={'n':len(measured),'exact_token_pass':len(measured)-len(failures),
                          'failed':failures,'config':blob['config']}
        return measured
    for arm in ['C0','C1','C2']:
        rowmetrics[arm]=measure(arm,evals[arm])
        if arm+'_g12' in evals:
            rowmetrics[arm].update(measure(arm+'_g12',evals[arm+'_g12']))
        p=c5/f'phaseC5_chain_{arm}_fresh30.json'
        blob=read(snap(p)); rowmetrics[arm+'_b1']=measure(arm+'_b1',blob)
    for group,ids in groups.items():
        for suffix in ['', '_b1']:
            if suffix and group!='validation29':continue
            for arm in ['C0','C1','C2']:
                rr=[rowmetrics[arm+suffix][t] for t in ids if t in rowmetrics[arm+suffix]]
                if len(rr)!=len(ids):continue
                summary['d2'][f'{arm}{suffix}/{group}']={
                    'n':len(rr),'mean_tau':float(np.mean([r['tau'] for r in rr])),
                    'pooled_tau':sum(r['raw_draft_accepted']+r['rounds'] for r in rr)/sum(r['rounds'] for r in rr),
                    'draft_accepted_raw':sum(r['raw_draft_accepted'] for r in rr),
                    'draft_accepted_clipped':sum(r['clipped_draft_accepted'] for r in rr),
                    'terminal_overcount':sum(r['terminal_overcount'] for r in rr),
                    'rounds':sum(r['rounds'] for r in rr),'tokens':sum(r['gen_tokens'] for r in rr),
                    'spec_decode_s':sum(r['spec_decode_s'] for r in rr),
                    'spec_prefill_decode_wall_s':sum(r['spec_wall_s'] for r in rr),
                    'verify_nodes':sum(r['tree_size']*r['rounds'] for r in rr),
                    'lossless':all(r['lossless'] for r in rr),
                    'timing_limit':'historical hardware/process contention not controlled; no speed claim'}
            for left,right in [('C0','C1'),('C0','C2'),('C1','C2')]:
                aa={t:rowmetrics[left+suffix][t]['tau'] for t in ids if t in rowmetrics[left+suffix]}
                bb={t:rowmetrics[right+suffix][t]['tau'] for t in ids if t in rowmetrics[right+suffix]}
                d=paired(aa,bb);summary['pairs'][f'{right}-{left}{suffix}/{group}/tau']=d
                table.append({'comparison':f'{right}-{left}','data_group':group,
                    'reference_mode':'online_b1k1_first_offset' if suffix else 'online_b63k8',
                    'loss_delta':'N/A','rank_delta':'N/A','top1_delta':'N/A','top8_delta':'N/A',
                    'first_rejection_or_oracle_delta':'N/A','tau_delta':d['mean'],
                    'elapsed_s':sum(rowmetrics[right+suffix][t]['spec_decode_s'] for t in bb) if bb else 'N/A',
                    'n':d['n'],'gate':'see audit exact AR tokens; final output truncates EOS/length',
                    'limitation':'not block16 chain' if suffix else 'oracle trace missing; timing not causal; legacy tau before terminal clipping'})
    for stage,why in [('D1_B','correct-prefix distribution/top-k not stored; legacy trajectory divergence'),
                      ('D2_oracles','raw candidates and final tree not stored'),
                      ('D2_block16_chain','existing b1k1 is only a first-offset probe'),
                      ('D4','blocked by reference/rollout inconsistency and no qualified idle GPU')]:
        table.append({'comparison':'C0/C1/C2','data_group':'not_run','reference_mode':stage,
                      'loss_delta':'N/A','rank_delta':'N/A','top8_delta':'N/A',
                      'first_rejection_or_oracle_delta':'N/A','tau_delta':'N/A','elapsed_s':'N/A',
                      'n':'N/A','gate':'blocked','limitation':why})
    write(out/'summary.json',summary)
    write(out/'d2_per_video.json',rowmetrics)
    columns=list(dict.fromkeys(k for r in table for k in r))
    with open(out/'diagnosis_table.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        w.writerows({k:('N/A' if r.get(k) is None else r.get(k,'N/A')) for k in columns} for r in table)
    resources=capture(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,utilization.gpu','--format=csv'])
    processes=capture(['ps','-eo','pid,user,etime,args'])
    processes['stdout']='\n'.join(x for x in processes['stdout'].splitlines() if any(s in x for s in ['phaseC5','eval_video_opd','diag_gain']))
    head_after={k:digest(p/'model.safetensors') for k,p in heads.items()}
    assert all(head_after[k]==v['model_sha256'] for k,v in head_before.items())
    audit={'read_only':True,'checkpoint_hash_unchanged':True,'eval_gates':eval_gates,
           'reference_mismatch_videos':sum(not x['equal'] for x in alignment),
           'reference_compared_videos':len(alignment),'teacher_pairing_errors':pairing_errors,
           'issues':[
             'Rollout accepts against stored sequence IDs, not current fp32 incremental target. Old trajectories diverge from saved AR.',
             'A teacher prefix is frozen, but first-rejected labels are against the old sequence; C0 accepted labels sometimes disagree with current teacher.',
             'Existing chain is b1k1 (one draft candidate), not block16 chain; width and depth both change.',
             'Legacy tau/accepted counters precede EOS and max-length clipping; keep separate from output-effective lengths.',
             'C1/C2 exposure IDs are not written into training logs; identities reconstructed with independent count/length checks.',
             'C1 checkpoint has no optimizer_master.pt despite the phaseC4 report claiming both arms have it; no impact on inference, exact optimiser resume not supported for C1.',
             'D1 p_arg/rank suffix is draft-induced A reference, not correct AR B; n_acc_own is not actual online acceptance.',
             'D1 fp32 head evaluates historical bf16 weights in deployment precision, not exact training-dtype replay.'],
           'resources':resources,'existing_processes':processes,
           'blocked':{'D1_B':'need corrected actual C0 anchors and independently verified AR distribution replay',
                      'D2_oracles':'no raw top8/tree nodes in historical outputs; need actual-anchor trace',
                      'D2_block16_chain':'needs b15k1 AR-gated minimal pilot',
                      'D4':'stop at implementation/reference gate; no temporal inference'},
           'not_claimed':['suffix gradients useless','prediction versus selection bottleneck identified','generalisation proved','temporal understanding mechanism']}
    write(out/'audit.json',audit)
    scripts=[Path(__file__),_source_path('diagnostic_acl_metrics.py'),
             _source_path('test_diagnostic_acl.py')]
    scripts += [_source_path(x) for x in ['common.py','aligned_forward.py','opd_fullkl.py',
                 'opd_train_video_fullkl.py','opd_online_rollout.py','diag_gain_stuck.py','eval_video_opd.py',
                 'run_phaseC4_10upd.sh','run_phaseC5_d2.sh','train_head_video.py','train_head.py']]
    scripts += [Path(common.REPO)/'glance'/x for x in ['tree.py','decode.py','vl.py']]
    manifest={'created':datetime.now().astimezone().isoformat(),'root':str(root.resolve()),
              'workspace':common.WORKSPACE,'python':sys.executable,'python_version':sys.version,
              'torch':torch.__version__,'transformers':transformers.__version__,'pandas':pd.__version__,
              'command':[sys.executable,*sys.argv],'seed':20261012,'bootstrap_samples':2000,
              'unit':'video','warmup_ids':sorted(warmups),'groups':groups,
              'target':common.TARGET_PATH,'attention_backend':'sdpa (common.load_target default)',
              'dtype':'existing D1/eval target fp32 head fp32; historical training head bf16',
              'feature_layers':list(common.KEPT_LAYERS),'feature_path':'cached',
              'head_checkpoints':head_before,'head_hash_after':head_after,
              'records':{'path':str(records_path),'sha256':digest(records_path)},
              'script_hashes':{str(p):digest(p) for p in scripts if p.exists()},
              'input_snapshots':sources,'tree_semantics':{'budget_excludes_root':True,'packed_cost':64,
              'draft_offsets':15,'bonus_excluded_from_oracles':True,'legacy_tau':'mean(raw_accepted+1) before terminal clipping'},
              'scope':'CPU reanalysis only; no model forward; existing 24 anchors/video reused, no new selection',
              'missing':['historical original command lines for D1 (script defaults/logs only)',
                         'original frame indices/timestamps','AR-reference logits','actual top-k/tree trace']}
    write(out/'manifest.json',manifest)
    print(json.dumps({'output':str(out),'exposure_positions':exposure['C1_positions'],
                      'C1_weight_sum':exposure['C1_weight_sum'],
                      'alignment_videos':len(alignment),'diverged':audit['reference_mismatch_videos'],
                      'shared_prefix_anchors':sum(x['anchors_in_shared_AR_prefix'] for x in alignment),
                      'compared_anchors':sum(x['anchors'] for x in alignment),
                      'teacher_split_mismatches':sum(x['anchors_nacc_differs_from_teacher'] for x in mismatch),
                      'd2':summary['d2']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
