#!/usr/bin/env python3
"""Gate newly initialized B/C heads against their exact saved plain parent."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed(path):
    if json.loads((path.parent / 'EXIT.json').read_text())['exit_code'] != 0:
        raise ValueError(f'{path}: unsuccessful process')
    value = json.loads(path.read_text())
    if value['status'] != 'LOSSLESS_PASSED':
        raise ValueError(f'{path}: failed lossless gate')
    return value


def main():
    marker = ROOT / 'control/EVIDENCE_ZERO_GATE_PASSED.json'
    if marker.exists():
        raise FileExistsError(marker)
    init = ROOT / 'matched_evidence_engineering_init/INIT_MANIFEST.json'
    identity = json.loads(init.read_text())
    reference_path = ROOT / 'jobs/bounded_v2_warm20_128f_lr1e-05_seed20261112_dev40/result.json'
    reference = completed(reference_path)
    if reference['config']['head'] != identity['source']:
        raise ValueError('reference is not the exact initialization parent')
    if reference['protocol']['artifacts']['head_weights_sha256'] != identity['source_weights_sha256']:
        raise ValueError('parent weight SHA differs')
    refs = {row['video']: row for row in reference['rows']}
    if len(refs) != 40:
        raise ValueError('parent reference is incomplete')
    checked = []
    for arm in ('B_text', 'C_visual'):
        path = ROOT / f'jobs/evidence_zero128_{arm}_dev2/result.json'
        value = completed(path)
        if value['config']['head'] != identity['arms'][arm]:
            raise ValueError('evaluated head differs from matched initialization')
        if value['protocol']['artifacts']['head_weights_sha256'] != identity['arm_weights_sha256'][arm]:
            raise ValueError('evaluated head weights differ from initialization')
        for key in ('target', 'dtype', 'budget', 'branch', 'max_new', 'time_mode', 'eos_stop'):
            if value['config'][key] != reference['config'][key]:
                raise ValueError(f'evaluation protocol differs: {key}')
        if len(value['rows']) != 2 or len({row['video'] for row in value['rows']}) != 2:
            raise ValueError('expected two distinct completed gate videos')
        for row in value['rows']:
            base = refs[row['video']]
            if not row['lossless'] or row['spec_ids'] != row['ar_ids']:
                raise ValueError('SPEC differs from independent AR')
            for key in ('input_identity', 'spec_ids', 'ar_ids', 'anchor_rel', 'accepted', 'rounds'):
                if row[key] != base[key]:
                    raise ValueError(f'{arm}/{row["video"]}: zero residual changed {key}')
        checked.append(dict(arm=arm, result=str(path), result_sha256=sha(path),
                            videos=[row['video'] for row in value['rows']]))
    result = dict(status='ZERO_RESIDUAL_GATE_PASSED', init_sha256=sha(init),
                  reference=str(reference_path), reference_sha256=sha(reference_path),
                  checked=checked, fields_exact=['input_identity', 'spec_ids', 'ar_ids',
                                               'anchor_rel', 'accepted', 'rounds'],
                  scope='two dev videos; this gate does not establish a learned visual benefit')
    with marker.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
