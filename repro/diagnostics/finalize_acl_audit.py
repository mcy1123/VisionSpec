#!/usr/bin/env python3
"""Supplement a completed CPU audit with processor replay and group contrasts.

Does not mutate the frozen manifest/audit. Writes supplemental evidence only.
"""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from diagnostics.audit_diagnostic_acl import read, lines, write, digest, capture
from shared import common


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True)
    a=ap.parse_args();root=Path(a.run)
    if (root/'supplement.json').exists():raise FileExistsError(root/'supplement.json')
    replay=lines(root/'input_replay_v2.jsonl')
    assert len(replay)==70 and all(x['prompt_exact'] for x in replay)
    assert all(replay[0]['repeat_exact'].values())
    groups=read(root/'d3_exposure_manifest.json')['groups']
    pv=read(root/'d1_per_video_A.json')
    results={}
    for metric in ['loss','top1','rank_le8']:
        part='native_replay' if metric=='loss' else 'first_rejected'
        delta={t:pv['C2'][t][part][metric]-pv['C0'][t][part][metric] for t in pv['C0']}
        exposed=groups['C2_only'];other=groups['unexposed_this_round_arm_matched10']
        ea=np.array([delta[t] for t in exposed]);oa=np.array([delta[t] for t in other])
        rng=np.random.default_rng(20261012)
        boot=ea[rng.integers(0,len(ea),(2000,len(ea)))].mean(1)-oa[rng.integers(0,len(oa),(2000,len(oa)))].mean(1)
        results[metric]={'contrast':'mean(C2-C0 exposed10) - mean(C2-C0 unexposed10)',
                         'reference_mode':'A_legacy_trajectory_draft_induced',
                         'mean':float(ea.mean()-oa.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
                         'n_exposed':len(ea),'n_unexposed':len(oa),
                         'resampling':'independent video-cluster bootstrap of within-video model differences',
                         'causal_limit':'not random exposure assignment; control not matched on input length/task difficulty'}
    manifest=read(root/'manifest.json')
    target=Path(common.TARGET_PATH)
    configs={str(p):digest(p) for p in target.iterdir() if p.suffix in ['.json','.jinja']}
    scripts={str(_source_path(n)):digest(_source_path(n)) for n in
             ['audit_acl_inputs.py','finalize_acl_audit.py','test_diagnostic_acl.py']}
    changed=[p for p,h in manifest['script_hashes'].items() if digest(p)!=h]
    # Absolute timestamps are those of the cached 16-frame clip. Original source
    # timeline is not recoverable merely from its cache encoding fps.
    result={'created':datetime.now().astimezone().isoformat(),'processor_prompt_exact':'70/70',
            'processor_repeat':replay[0]['repeat_exact'],'processor_repeat_n':1,
            'model_forward_repeatability':'not_run',
            'metadata_scope':'frame indices/fps/duration refer to cached video, not original source timeline',
            'prior_sources_unchanged':not changed,'source_changes':changed,
            'target_processor_tokenizer_config_hashes':configs,'script_hashes':scripts,
            'exposure_contrasts':results,'resources_final':capture(['nvidia-smi',
            '--query-gpu=index,name,memory.total,memory.free,utilization.gpu','--format=csv']),
            'failures':[{'command':'audit_acl_inputs.py (initial version; preserved as audit_acl_inputs.failed.py)',
                         'status':'failed before first sample',
                         'cause':'explicit return_metadata=True + return_tensors=pt attempts to tensorise VideoMetadata',
                         'resolution':'local observer of processor internal metadata; original evaluator public call retained; no dependency edits',
                         'log':'input_replay.log','successful_rerun':'input_replay_v2.log'}]}
    write(root/'supplement.json',result)
    print(json.dumps(result['exposure_contrasts'],indent=2))


if __name__=='__main__':main()
