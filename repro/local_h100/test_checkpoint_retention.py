import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_h100 import checkpoint_retention as retention


def checkpoint(root,job,step):
    p=root/'jobs'/job/'run'/f'step{step}';p.mkdir(parents=True)
    for name in retention.LARGE:(p/name).write_bytes((name+str(step)).encode())
    (p/'config.json').write_text('{"hidden_size":4}')
    manifest={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in (*retention.LARGE,'config.json')}
    (p/'SHA256.json').write_text(json.dumps(manifest))
    return p


def setup(root,jobs):
    (root/'control/scheduler').mkdir(parents=True)
    q=dict(version=1,jobs=[dict(id=name,argv=argv) for name,status,argv in jobs])
    (root/'control/scheduler/queue.json').write_text(json.dumps(q))
    (root/'control/scheduler/STATE.json').write_text(json.dumps(dict(jobs={name:dict(status=status) for name,status,argv in jobs})))
    (root/'FACTORIAL80_PROTOCOL.json').write_text(json.dumps(dict(common_base=str(root/'required_init'))))


def test_pending_head_and_formal_final_are_preserved(tmp_path):
    job='factorial80_128f_C_visual_opd_seed1'
    p=checkpoint(tmp_path,job,20);checkpoint(tmp_path,job,60);checkpoint(tmp_path,job,80)
    setup(tmp_path,[(job,'SUCCEEDED',[]),('eval','WAITING',['python','--head',str(p)])])
    plan=retention.plan(tmp_path)
    assert [x['step'] for x in plan['retire']]==[60]


def test_active_lambda10_retains_latest_and_then_final(tmp_path):
    job='factorial80_tv10_128f_C_visual_opd_seed1'
    checkpoint(tmp_path,job,20);checkpoint(tmp_path,job,40)
    setup(tmp_path,[(job,'RUNNING',[])])
    assert [x['step'] for x in retention.plan(tmp_path)['retire']]==[20]
    checkpoint(tmp_path,job,80)
    assert [x['step'] for x in retention.plan(tmp_path)['retire']]==[20,40]


def test_apply_removes_only_large_files_after_metadata_archive(tmp_path):
    job='gpu_master_completed';p=checkpoint(tmp_path,job,2)
    setup(tmp_path,[(job,'SUCCEEDED',[])])
    manifest=(p/'SHA256.json').read_bytes()
    preview=retention.run_once(tmp_path)
    assert preview['planned_checkpoints']==1 and all((p/f).exists() for f in retention.LARGE)
    result=retention.run_once(tmp_path,apply=True)
    assert result['deleted_checkpoints']==1
    assert not any((p/f).exists() for f in retention.LARGE)
    assert (p/'SHA256.json').read_bytes()==manifest
    assert (tmp_path/retention.AUDIT_REL/'metadata'/job/'step2/SHA256.json').read_bytes()==manifest
    assert json.loads((p/'RETIRED.json').read_text())['original_metadata_preserved'] is True
    assert retention.run_once(tmp_path,apply=True)['deleted_checkpoints']==0


def test_queue_change_aborts_before_deletion(tmp_path):
    job='gpu_master_completed';p=checkpoint(tmp_path,job,2)
    setup(tmp_path,[(job,'SUCCEEDED',[])])
    value=retention.plan(tmp_path)
    with (tmp_path/'control/scheduler/queue.json').open('a') as f:f.write(' ')
    with pytest.raises(RuntimeError,match='queue changed'):
        retention.apply_one(tmp_path,value['retire'][0],value['queue_bytes'])
    assert all((p/f).exists() for f in retention.LARGE)


def test_file_change_aborts_before_deletion(tmp_path):
    job='gpu_master_completed';p=checkpoint(tmp_path,job,2)
    setup(tmp_path,[(job,'SUCCEEDED',[])])
    value=retention.plan(tmp_path);(p/'training_state.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='file changed'):
        retention.apply_one(tmp_path,value['retire'][0],value['queue_bytes'])
    assert all((p/f).exists() for f in retention.LARGE)


def test_unknown_family_unfinished_owner_and_symlink_are_preserved(tmp_path):
    checkpoint(tmp_path,'outside_experiment',2)
    checkpoint(tmp_path,'gpu_master_unfinished',2)
    p=checkpoint(tmp_path,'gpu_master_symlink',2)
    (p/'model.safetensors').unlink();(p/'model.safetensors').symlink_to(p/'training_state.pt')
    setup(tmp_path,[('outside_experiment','SUCCEEDED',[]),('gpu_master_unfinished','RUNNING',[]),('gpu_master_symlink','SUCCEEDED',[])])
    assert not retention.plan(tmp_path)['retire']
