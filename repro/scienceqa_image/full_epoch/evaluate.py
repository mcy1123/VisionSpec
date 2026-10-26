#!/usr/bin/env python3
"""Read-only coverage instrumentation around the unchanged deployment decoder."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[1])]
from _bootstrap import configure
configure()
import torch
import glance
import glance.decode as decode
from scienceqa_image import eval_sqa
from scienceqa_image.build_eval_ar import load_split
from scienceqa_image.full_epoch.data import write_json


def main():
    def arg(name):return sys.argv[sys.argv.index(name)+1]
    rows=load_split(arg('--run'),arg('--split'))
    oracle={r['pid']:r for r in torch.load(arg('--ar-cache'),map_location='cpu',weights_only=False)['records']}
    original_generate=glance.generate; original_tree=decode.build_budget_tree
    results=[]
    def generate(head,target,ids,*args,**kwargs):
        row=rows[len(results)]; tokens=oracle[row['pid']]['tokens'];rounds=[];candidate={}
        def tree(*a,**k):
            nodes=original_tree(*a,**k)
            candidate.update(top=a[0].tolist(),nodes=nodes)
            return nodes
        def observe(record):
            begin=record['pos']-ids.shape[1]+1
            suffix=tokens[begin:begin+len(candidate['top'])]
            raw=0
            for i,t in enumerate(suffix):
                if t not in candidate['top'][i]:break
                raw+=1
            tree_len=0;parent=-1
            for t in suffix:
                hits=[i for i,n in enumerate(candidate['nodes']) if n.parent==parent and n.token==t]
                if not hits:break
                parent=hits[0];tree_len+=1
            rounds.append(dict(round=record['round'],pos=record['pos'],horizon=len(suffix),
                topk_prefix_coverage=raw,tree_prefix_coverage=tree_len,accepted_before_eos_clip=record['accepted']))
        kwargs['on_round']=observe
        with patch.object(decode,'build_budget_tree',tree):
            result=original_generate(head,target,ids,*args,**kwargs)
        results.append(dict(pid=row['pid'],image_group_id=row['image_group_id'],rounds=rounds))
        return result
    with patch.object(glance,'generate',generate):
        code=eval_sqa.main()
    write_json(Path(arg('--out')).with_suffix('.coverage.json'),dict(rows=results,
        definition='AR correct-prefix length in marginal top-k / in actual budget tree at each online anchor; EOS/cap clipped horizon',
        timing='instrumented single pass; do not claim clean wall-clock speedup'))
    return code

if __name__=='__main__':sys.exit(main())
