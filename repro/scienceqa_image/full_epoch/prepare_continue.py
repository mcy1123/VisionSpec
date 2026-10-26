"""CPU-only append-only schedule freeze and explicit four-to-three checkpoint migration."""
import argparse,copy,json,os,random,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repro'))
from _bootstrap import configure
configure()
import numpy as np
import torch
from shared import training_resume
from scienceqa_image.full_epoch.data import sha,write_json,WARM
from scienceqa_image.full_epoch.continue_three_card import DEFAULT,PREVIOUS,GPUS,GRID,command,opd_args
from opd_official_kl.three_card_trainer import identity,validate_checkpoint,SLOT_MAPPING


def exact(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def main():
    run=DEFAULT;old=PREVIOUS
    assert str(run.resolve()).startswith('/mnt/local/')
    assert not (run/'PREPARED.json').exists()
    for name in ('dataset','logs','commands','stages','eval','migration'): (run/name).mkdir(exist_ok=True)
    for item in (old/'dataset').iterdir():
        if item.name!='schedule.jsonl':(run/'dataset'/item.name).symlink_to(item.resolve(),target_is_directory=item.is_dir())
    (run/'eval/ar_test.pt').symlink_to((old/'eval/ar_test.pt').resolve())
    original=(old/'dataset/schedule.jsonl').read_text()
    first=[json.loads(x) for x in original.splitlines() if x.strip()]
    assert len(first)==4726 and len({x['task'] for x in first})==4726
    continuation=[]
    for epoch in (1,2):
        rows=copy.deepcopy(first);random.Random(20261013+epoch).shuffle(rows)
        for index,row in enumerate(rows):
            row.update(order=epoch*4726+index,update=epoch*1182+index//4,slot=index%4)
            continuation.append(row)
        assert {r['task'] for r in rows}=={r['task'] for r in first}
    schedule=run/'dataset/schedule.jsonl'
    schedule.write_text(original+''.join(json.dumps(x)+'\n' for x in continuation))
    assert schedule.read_text().startswith(original)
    counts={}
    for r in first+continuation:counts[r['update']]=counts.get(r['update'],0)+1
    assert sorted(counts)==list(range(3546))
    assert {i:n for i,n in counts.items() if n!=4}=={1181:2,2363:2,3545:2}
    write_json(run/'PROTOCOL.json',dict(parent_run=str(old),start_update=1182,final_update=3546,additional_epochs=2,
        additional_updates_per_arm=2364,qa_per_epoch=4726,additional_qa_per_arm=9452,batch=4,partial_epoch_final_batch=2,
        gpus=GPUS,physical_gpu_indices=[0,1,2],world_size=3,slot_mapping=SLOT_MAPPING,lr=1e-5,lr_schedule='constant',
        target_dtype='float32',head_dtype='bfloat16',optimizer='CPU fp32 master AdamW; moments/step retained',
        seed=20261013,epoch_shuffle_seeds=[20261014,20261015],checkpoint_grid=GRID,early_stop=False,
        dev='fixed dev32 SPEC-only after each ~200 updates',final_spec_ar='existing independent AR cache, only at final step3546',
        confirm200='reused descriptive evaluation; no fresh independent test claim',
        native_equivalence='approximate native method; no cross-topology bitwise equivalence claim',
        startup='one real continuation update per arm, included in the two-epoch budget'))
    latest={};cases={};migration={}
    for arm in ('opd','sft'):
        src=old/'train'/arm/'to_1182/step1182';dest=run/'migration'/arm/'step1182'
        assert not dest.exists();dest.parent.mkdir(exist_ok=True)
        print('migrating',arm,src,flush=True)
        cmd=command(run,arm,run/'unused',1,dest)
        if arm=='opd':
            args=opd_args(cmd);metadata=identity(args)
        else:
            args=SimpleNamespace(seed=20261013,qa_per_update=4,anchors_per_qa=6,lr=1e-5,clip=1.,master_weights='cpu',
                target_dtype='float32',time_mode='legacy',max_pixels=768,max_seq_len=4096,allow_partial_final=True,
                init_from=WARM,pool=str(run/'dataset/pool.parquet'),records=str(run/'dataset/labels.pt'))
            metadata=training_resume.identity(args,schedule);metadata['epoch_updates']=1182
        oldstate=torch.load(src/'training_state.pt',map_location='cpu',weights_only=False,mmap=True)
        assert oldstate['update_step']==oldstate['data_cursor']==oldstate['update']==1182
        assert {float(v['step']) for v in oldstate['master_optimizer']['optimizer']['state'].values()}=={1182.}
        assert oldstate['resume_identity']['schedule_sha256']==sha(old/'dataset/schedule.jsonl')
        new=dict(oldstate);new['resume_identity']=metadata
        if arm=='opd':
            pp=oldstate['parallel_state'];assert pp['world_size']==4 and len(pp['per_rank_rng'])==4
            assert oldstate['resume_identity']['pool_sha256']==metadata['pool_sha256']
            p=dict(pp,world_size=3,slot_mapping=SLOT_MAPPING,implementation='pure_block_three_card_four_slot_v1',per_slot_rng=pp['per_rank_rng'])
            del p['per_rank_rng'];new['parallel_state']=p
            new['training_identity']=dict(metadata,head_version=oldstate['training_identity']['head_version'])
            validate_checkpoint(new,metadata,requested_step=1182)
        else:
            assert oldstate['resume_identity']['data_sha256']==metadata['data_sha256']
            assert oldstate['resume_identity']['config_sha256']==metadata['config_sha256']
            new['schedule']=str(schedule)
        source_sha=json.loads((src/'SHA256.json').read_text())
        new['continuation_provenance']=dict(source=str(src),source_sha256=source_sha,parent_update=1182,
            migration='append two epoch schedules; preserve all tensors; OPD four logical RNG streams on three physical ranks')
        tmp=dest.with_name('.step1182.pending');tmp.mkdir()
        for name in ('model.safetensors','config.json'):shutil.copy2(src/name,tmp/name)
        torch.save(new,tmp/'training_state.pt')
        hashes={name:sha(tmp/name) for name in ('model.safetensors','config.json','training_state.pt')}
        assert hashes['model.safetensors']==source_sha['model.safetensors']
        write_json(tmp/'SHA256.json',hashes);tmp.rename(dest)
        saved=torch.load(dest/'training_state.pt',map_location='cpu',weights_only=False,mmap=True)
        assert exact(oldstate['head'],saved['head']) and exact(oldstate['master_optimizer'],saved['master_optimizer'])
        assert exact(oldstate['rng'],saved['rng'])
        if arm=='opd':assert exact(oldstate['parallel_state']['per_rank_rng'],saved['parallel_state']['per_slot_rng'])
        migration[arm]=dict(source=str(src),destination=str(dest),head_master_adam_rng_exact=True,optimizer_step=1182,
            before_identity=oldstate['resume_identity'],after_identity=metadata,source_sha256=source_sha,migrated_sha256=hashes)
        latest[arm]=str(dest)
        case=dict(updates=3546,start_update=1182,entry='scienceqa_pure_block' if arm=='opd' else 'scienceqa_sft',seed=20261013)
        if arm=='opd':case['config_sha256']=metadata['config_sha256']
        cases[arm+'_ep23']=case
        del saved,new,oldstate
        print('verified exact Head/master/Adam/RNG migration',arm,flush=True)
    write_json(run/'MIGRATION.json',migration);write_json(run/'LATEST.json',latest)
    write_json(run/'BUDGET.json',dict(cases=cases,max_attempts=4728,attempts=[],completed=[]))
    inputs=[run/'dataset/pool.parquet',run/'dataset/labels.pt',schedule,run/'dataset/records_manifest.csv']
    write_json(run/'INPUTS.json',{str(p):sha(p) for p in inputs})
    paths=list((ROOT/'repro/opd_official_kl').glob('*.py'))+list((ROOT/'repro/scienceqa_image/full_epoch').glob('*.py'))+[ROOT/'repro/scienceqa_image/train_sqa_sft_continuation.py',ROOT/'repro/shared/training_resume.py']
    write_json(run/'SOURCE.json',{str(p):sha(p) for p in paths})
    write_json(run/'PREPARED.json',dict(time=time.time(),schedule_checks='three exact full QA permutations, epoch final batch2, original prefix unchanged',migration_checks='exact tensor equality for both Head/master/Adam and RNG, step1182 retained'))
    print('PREPARED',run,flush=True)
if __name__=='__main__':main()
