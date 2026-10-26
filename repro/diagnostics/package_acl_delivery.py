#!/usr/bin/env python3
"""Package the final table without altering either frozen audit snapshot."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
import csv
import difflib
from pathlib import Path
from diagnostics.audit_diagnostic_acl import read, write, digest


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True)
    a=ap.parse_args();root=Path(a.run);done=root/'completed_tree_audit'
    dest=root/'diagnosis_table_final.csv'
    if dest.exists():raise FileExistsError(dest)
    checks=read(done/'completed_eval_checks.json')
    with open(done/'diagnosis_table.csv') as f:rows=list(csv.DictReader(f))
    for r in rows:
        r['timed_n']='N/A';r['timing_warmup_rule']='N/A'
        if r['reference_mode']=='online_b63k8':
            arm=r['comparison'].split('-')[0]
            timing=checks['timing'][arm][r['data_group']]
            r['elapsed_s']=timing['total_decode_s_excluding_job_warmup']
            r['timed_n']=timing['timed_n']
            r['timing_warmup_rule']='exclude job warmup from timing only; tau fixed group retained'
            r['gate']='own SPEC/AR and cross-arm AR exact, 70/70 overall'
        if r['reference_mode']=='online_b1k1_first_offset':
            r['timed_n']=r['n'];r['timing_warmup_rule']='exclude original validation warmup'
    with open(dest,'x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    workspace=_REPRO_ROOT.parent
    before=(root/'AGENTS.before.md').read_text().splitlines(True)
    after=(workspace/'AGENTS.md').read_text().splitlines(True)
    (root/'AGENTS.diff').write_text(''.join(difflib.unified_diff(before,after,fromfile='AGENTS.before.md',tofile='AGENTS.md')))
    scripts=[workspace/'repro'/n for n in ['audit_diagnostic_acl.py','diagnostic_acl_metrics.py',
        'audit_acl_inputs.py','finalize_acl_audit.py','check_acl_completed_eval.py',
        'package_acl_delivery.py','test_diagnostic_acl.py']]
    docs=[workspace/'AGENTS.md',workspace/'docs/reports/2026-10-12-gain-stuck.md']
    write(root/'delivery_manifest.json',{
        'primary_table':str(dest),'primary_summary':str(done/'summary.json'),
        'primary_manifest':str(done/'manifest.json'),'primary_audit':str(done/'audit.json'),
        'additional_input_replay':str(root/'input_replay_v2.jsonl'),
        'additional_evidence':[str(root/'supplement.json'),str(done/'completed_eval_checks.json')],
        'tests':{'passed':16,'total':16,'log':str(root/'tests.log'),
                 'initial_failure':'JS exact-zero assertion replaced by explicit fp32 tolerance'},
        'status':{'D0':'completed; implementation/reference gate failed',
                  'D1_A':'completed reanalysis on legacy reference; not exact historical bf16 training replay',
                  'D1_B':'blocked; correct AR distributions absent',
                  'D2_online_tree_and_b1':'completed from existing evidence',
                  'D2_block16_chain':'not_run','D2_local_oracles':'not_run; CPU operator tested only',
                  'D3':'completed exposure reconstruction and existing decoding reanalysis; source genealogy limited',
                  'D4':'not_run; implementation/reference gate'},
        'final_resource_limit':'GPU0-2 became free; not a final resource block. Stop at confirmed reference inconsistency gate.',
        'model_updates':False,'new_model_forward_runs':0,'new_test_set_used':False,
        'script_hashes':{str(p):digest(p) for p in scripts},
        'document_hashes':{str(p):digest(p) for p in docs},
        'table_sha256':digest(dest),
        'checkpoint_hash_unchanged_in_both_audits':all(read(p/'audit.json')['checkpoint_hash_unchanged'] for p in [root,done])})
    print(dest)


if __name__=='__main__':main()
