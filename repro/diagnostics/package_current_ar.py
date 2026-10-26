#!/usr/bin/env python3
"""Finalize provenance and honest model-call accounting, including failed pilot."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import torch
import transformers
from shared import common
from diagnostics.current_ar_closed_loop import read, write, sha


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--prior',required=True)
    a=ap.parse_args();p=Path(a.run);prior=Path(a.prior)
    assert not (p/'delivery_manifest.json').exists()
    final=read(p/'manifest_final.json');failed=read(prior/'failure.json')
    assert read(prior/'manifest.json')['videos']==final['videos']
    assert read(p/'trace_recompute_gate.json')['pass']
    calls=read(p/'calls.json')['calls'];old=failed['calls']
    def count(c):
        return {'head_forwards':sum(c.get(k,0) for k in ['C0_decode_head_forwards','fixed_context_other_head_forwards','gate_extra_head_forwards']),
                'target_forwards':sum(c.get(k,0) for k in ['C0_decode_target_forwards','independent_AR_target_forwards','local_tree_verification_target_forwards'])}
    current=count(calls);overhead=count(old)
    accounting={'successful_run':calls,'failed_pilot_extra':old,
         'successful_actual_forwards':current,'failed_pilot_actual_forwards':overhead,
         'all_actual_forwards':{k:current[k]+overhead[k] for k in current},
         'formal_unique_scores':144,'discarded_partial_score_attempts':failed['formal_head_scores'],
         'discarded_partial_records':sum(1 for _ in open(prior/'trace_current_ar.jsonl')),
         'note':'C0 formal scores reused from replay; gate_verification_subset is included in local_tree_verification_target_forwards, never added twice. Failed pilot is separate overhead, not hidden by the 144 unique-score limit.'}
    write(p/'all_call_accounting.json',accounting)
    shutil.copytree(prior/'A_relabel',p/'A_relabel')
    shutil.copyfile(prior/'A_relabel.log',p/'A_relabel.log')
    sources=[_source_path(n) for n in ['current_ar_closed_loop.py','current_ar_analysis.py',
                'verify_current_ar_trace.py','test_current_ar_closed_loop.py','diagnostic_acl_metrics.py','package_current_ar.py',
                'opd_fullkl.py','common.py']]
    target=Path(common.TARGET_PATH)
    config_files=[f for f in target.iterdir() if f.suffix in ['.json','.jinja']]
    manifest={'created':datetime.now().astimezone().isoformat(),'status':'completed',
        'frozen_manifest':str(p/'manifest.json'),'model_run_manifest':str(p/'manifest_final.json'),
        'reference_modes':['A_old (legacy trajectory/draft-induced teacher)','A_relabel (same inputs; teacher leading-match grouping)',
                           'B_current (current deployment trajectory/correct AR teacher)'],
        'sample_limitations':'6 validation videos, 48 nested anchors; no A/B same-context causal comparison',
        'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,
        'scripts':{str(f.resolve()):sha(f) for f in sources},
        'target_tokenizer_processor_config_hashes':{str(f):sha(f) for f in config_files},
        'call_accounting':accounting,'formal_score_budget_pass':final['formal_head_scores']==144,
        'checkpoint_hash_gate':all(final['checkpoint_hashes_after'][arm]==c['model_sha256'] for arm,c in final['checkpoint_identity'].items()),
        'pilot_gate':read(p/'pilot_gate.json'),'trace_recompute_gate':read(p/'trace_recompute_gate.json'),
        'failed_run_preserved':str(prior),'minimal_repair':'capture GPU topk scores/tree directly; avoid CPU log_softmax rebuilding purported actual tree',
        'raw_trace':str(p/'trace_current_ar.jsonl'),'per_video':str(p/'per_video_current_ar.csv'),
        'diagnosis':str(p/'diagnosis_current_ar.csv'),'A_relabel':str(p/'A_relabel/summary_A.json'),
        'no_training':True,'no_D4':True,'no_new_test_set':True,'no_chain_eval':True,
        'gpu_at_finish':subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True).stdout}
    write(p/'delivery_manifest.json',manifest)
    print(accounting)
