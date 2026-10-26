#!/usr/bin/env python3
"""Final checks after existing g1/g2 jobs finish; consumes frozen CPU audit only."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
from pathlib import Path
import numpy as np
from diagnostics.audit_diagnostic_acl import read, write


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True)
    a=ap.parse_args();p=Path(a.run)
    if (p/'completed_eval_checks.json').exists():raise FileExistsError(p/'completed_eval_checks.json')
    rows=read(p/'d2_per_video.json');groups=read(p/'d3_exposure_manifest.json')['groups']
    ids=set(rows['C0'])
    checks={arm:{'n':len(ids),'same_AR_as_C0':sum(rows[arm][t]['ar_ids']==rows['C0'][t]['ar_ids'] for t in ids),
                 'own_AR_exact':sum(rows[arm][t]['ar_ids']==rows[arm][t]['spec_ids'] for t in ids)} for arm in ['C0','C1','C2']}
    assert all(x['n']==x['same_AR_as_C0']==x['own_AR_exact'] for x in checks.values())
    contrasts={}
    for control in ['unexposed_this_round_arm_matched10','unexposed_this_round_union_matched20']:
        e=np.array([rows['C2'][t]['tau']-rows['C0'][t]['tau'] for t in groups['C2_only']])
        c=np.array([rows['C2'][t]['tau']-rows['C0'][t]['tau'] for t in groups[control]])
        rng=np.random.default_rng(20261012)
        b=e[rng.integers(len(e),size=(2000,len(e)))].mean(1)-c[rng.integers(len(c),size=(2000,len(c)))].mean(1)
        contrasts[control]={'n_exposed':len(e),'n_control':len(c),'difference_of_paired_deltas':float(e.mean()-c.mean()),
                            'ci95':np.quantile(b,[.025,.975]).tolist()}
    timing={}
    for arm in ['C0','C1','C2']:
        timing[arm]={g:{'total_decode_s_excluding_job_warmup':sum(rows[arm][t]['spec_decode_s'] for t in ts if not rows[arm][t]['warmup']),
                        'timed_n':sum(not rows[arm][t]['warmup'] for t in ts),'tau_n':len(ts),
                        'job_warmup_ids':[t for t in ts if rows[arm][t]['warmup']]} for g,ts in groups.items()}
    result={'cross_arm_AR':checks,'C2_exposure_tau_contrasts':contrasts,
            'timing':timing,'warmup_policy':'tau retains all fixed g1/g2 identities; excludes original validation warmup only. For timing, also exclude the g12 job first row; do not drop that video from tau.',
            'timing_limit':'raw observed timings only; no controlled speed comparison'}
    write(p/'completed_eval_checks.json',result)
    print(result)


if __name__=='__main__':main()
