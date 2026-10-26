#!/usr/bin/env python3
"""Fast development evaluation: SPEC only, no AR oracle or lossless check."""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from _bootstrap import configure
configure()
import torch
import glance
import glance.decode as decode
from shared import common
from scienceqa_image.build_eval_ar import load_split,encode
from scienceqa_image.full_epoch.data import write_json


def main():
    p=argparse.ArgumentParser()
    for name in ['run','head','tag','out']:p.add_argument('--'+name,required=True)
    a=p.parse_args()
    torch.set_grad_enabled(False);torch.cuda.set_device(0)
    target,proc=common.load_target(device='cuda:0',dtype=torch.float32)
    head=common.load_head(a.head,device='cuda:0',dtype=torch.float32).eval()
    eos=int(proc.tokenizer.eos_token_id)
    rows=load_split(a.run,'dev');results=[];coverage=[];start=time.time()
    original_tree=decode.build_budget_tree
    for row in rows:
        ids,pos,_=encode(proc,row,768,'cuda:0')
        trees=[];rounds=[]
        def tree(*args,**kwargs):
            nodes=original_tree(*args,**kwargs)
            trees.append((args[0].tolist(),nodes));return nodes
        with patch.object(decode,'build_budget_tree',tree):
            final,stats=glance.generate(head,target,ids,256,[eos],budget=63,branch_k=8,
                temperature=0.,positions=pos,return_stats=True,timing='decode_only',
                eos_stop='immediate',on_round=rounds.append)
        tokens=final[0,ids.shape[1]:].tolist()
        text=proc.tokenizer.decode(tokens,skip_special_tokens=True)
        parsed=re.findall(r'Answer:\s*\(?\s*([A-Ea-e])\s*\)?',text)
        answer=parsed[-1].upper() if parsed else None
        results.append(dict(pid=row['pid'],image_group_id=row['image_group_id'],
            n_tokens=len(tokens),spec_rounds=stats.rounds,tokens_per_round=len(tokens)/max(stats.rounds,1),
            answer_correct=answer==row['gold_letter'],token_exact=None,
            lossless_checked=False,ms_per_token=stats.ms_per_token))
        item=[]
        for record,(top,nodes) in zip(rounds,trees):
            begin=record['pos']-ids.shape[1]+1;suffix=tokens[begin:begin+len(top)]
            raw=0
            for i,token in enumerate(suffix):
                if token not in top[i]:break
                raw+=1
            parent=-1;depth=0
            for token in suffix:
                hit=next((i for i,node in enumerate(nodes) if node.parent==parent and node.token==token),None)
                if hit is None:break
                parent=hit;depth+=1
            item.append(dict(round=record['round'],pos=record['pos'],horizon=len(suffix),
                topk_prefix_coverage=raw,tree_prefix_coverage=depth))
        coverage.append(dict(pid=row['pid'],image_group_id=row['image_group_id'],rounds=item))
        print(f"dev {len(results)}/{len(rows)} {row['pid']} effective={results[-1]['tokens_per_round']:.4f}",flush=True)
    write_json(a.out,dict(tag=a.tag,split='dev',n_records=len(results),
        n_image_groups=len({r['image_group_id'] for r in results}),rows=results,
        total_rounds=sum(r['spec_rounds'] for r in results),total_tokens=sum(r['n_tokens'] for r in results),
        lossless=None,lossless_checked=False,ar_generated=False,wall_seconds=time.time()-start,
        metric_note='SPEC-only development metric; independent AR equality checked only at final confirmation'))
    write_json(Path(a.out).with_suffix('.coverage.json'),dict(rows=coverage,
        definition='Coverage of emitted SPEC prefixes at online anchors, not independently verified AR-correct prefixes',
        lossless_checked=False))

if __name__=='__main__':main()
