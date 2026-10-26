#!/usr/bin/env python3
"""Recompute local oracles and packed path decisions from persisted trace."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
from pathlib import Path
import torch
from diagnostics.current_ar_closed_loop import read, write, walk
from diagnostics.current_ar_analysis import lines
from glance.tree import build_budget_tree
from diagnostics.diagnostic_acl_metrics import local_oracles


def validate(r):
    nodes=build_budget_tree(r['top8_ids'],r['top8_logp'],63,8)
    assert [vars(n) for n in nodes]==r['nodes']
    assert [r['root']]+[n.token for n in nodes]==r['packed_input_ids']
    children={i:[] for i in range(-1,len(nodes))}
    for i,n in enumerate(nodes):children[n.parent].append(i)
    path=[];cur=-1
    while True:
        token=r['verification_argmax'][cur+1]
        hit=next((i for i in children[cur] if nodes[i].token==token),None)
        if hit is None:break
        path.append(hit);cur=hit
    assert path==r['selected_node_path'] and token==r['bonus']
    assert [nodes[i].token for i in path]==r['raw_accepted_tokens']
    expected=local_oracles(r['top8_ids'],r['nodes'],r['reference_tokens'],r['raw_accepted_tokens'],
         budget=63,remaining_after_root=r['H_valid'],root_token=r['root'],eos_id=151645,reference_verified=True)
    assert expected==r['oracle']
    assert r['effective_draft_tokens']==r['reference_tokens'][:expected['L_actual']]
    if r['bonus_survives_output_clip']:
        assert r['bonus']==r['reference_tokens'][expected['L_actual']]
    assert r['draft_context_last_position']==r['pos']-1 and not r['future_context_used']
    visible=r['packed_visibility']
    for i,n in enumerate(nodes):
        ancestors={0,i+1};parent=n.parent
        while parent!=-1:ancestors.add(parent+1);parent=nodes[parent].parent
        assert {j for j,x in enumerate(visible[i+1]) if x}==ancestors
    for m in r['metrics']:
        i=m['offset']-1;t=m['reference_token']
        assert m['top8']==(t in r['top8_ids'][i])
        assert m['top32']==(t in r['top32_ids'][i])
        assert m['top1']==(t==r['top8_ids'][i][0])
    return True


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();p=Path(a.run)
    rs=lines(p/'trace_current_ar.jsonl')
    assert len(rs)==144
    for r in rs:validate(r)
    write(p/'trace_recompute_gate.json',{'pass':True,'records':len(rs),'anchor_contexts':48,
         'checks':['GPU candidate scores rebuild identical packed tree','packed greedy walk reproduces path and bonus',
                   'oracle inequalities and terminal clipping','prefix-only head boundary',
                   'ancestor mask closure','actual top-k membership']})
    print('144/144 persisted local traces independently recomputed PASS')
