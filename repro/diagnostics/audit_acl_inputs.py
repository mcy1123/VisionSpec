#!/usr/bin/env python3
"""CPU processor replay: exact stored prompt IDs, frame metadata and repeatability.

Only reads the existing phaseC5 frozen split (not any untouched test set).
Writes a new file, without modifying video caches or any model parameter.
"""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor
from shared import common


def safe(x):
    if isinstance(x,torch.Tensor):return x.tolist()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if dataclasses.is_dataclass(x):return safe(dataclasses.asdict(x))
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    return x


def sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--split',required=True)
    a=ap.parse_args()
    if Path(a.out).exists():raise FileExistsError(a.out)
    torch.set_num_threads(4)
    split=json.load(open(a.split))
    records=torch.load(split['records'],map_location='cpu',weights_only=False)['records']
    recs={r['task']:r for r in records}
    pool=pd.read_parquet(split['pool']).set_index('video_name')
    processor=AutoProcessor.from_pretrained(common.TARGET_PATH,local_files_only=True)
    # Explicit return_metadata=True makes this installed processor try to tensorise
    # VideoMetadata. Observe its existing internal metadata before it is popped;
    # keep the evaluator's exact public call and tensor outputs unchanged.
    last_metadata={}
    original=processor.video_processor.preprocess
    def observe(*args,**kwargs):
        result=original(*args,**kwargs)
        last_metadata['value']=safe(result.get('video_metadata'))
        return result
    processor.video_processor.preprocess=observe
    tasks=[t for g in split['groups'].values() for t in g['videos']]
    start=time.perf_counter()
    with open(a.out,'x') as f:
        for i,task in enumerate(tasks):
            r=pool.loc[task]
            text=processor.apply_chat_template([{'role':'user','content':[
                {'type':'video'},{'type':'text','text':str(r.question)}]}],
                add_generation_prompt=True,tokenize=False)
            def process():
                return processor(text=[text],videos=[r.cache_path],return_tensors='pt',
                                 fps=float(r.fps))
            ins=process(); ids=ins['input_ids'][0]
            stored=recs[task]['input_ids'][:int(recs[task]['prompt_len'])].long()
            row={'video':task,'keys':list(ins.keys()),'input_len':len(ids),
                 'input_ids_sha256':sha(ids),'stored_prefix_ids_sha256':sha(stored),
                 'prompt_exact':torch.equal(ids,stored),'pixel_sha256':sha(ins['pixel_values_videos']),
                 'pixel_shape':list(ins['pixel_values_videos'].shape),
                 'video_grid_thw':safe(ins['video_grid_thw']),
                 'video_metadata':last_metadata.get('value'),
                 'elapsed_s':time.perf_counter()-start,'repeat_exact':None}
            if i==0:
                repeat=process()
                row['repeat_exact']={k:torch.equal(ins[k],repeat[k]) for k in
                                     ['input_ids','pixel_values_videos','video_grid_thw']}
            f.write(json.dumps(row)+'\n');f.flush()
            print(f'{i+1}/{len(tasks)} {task} prompt_exact={row["prompt_exact"]}',flush=True)


if __name__=='__main__':main()
