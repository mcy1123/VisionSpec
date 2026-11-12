"""CPU counterexamples for matched factorial evaluation and video bootstrap."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from local_h100 import summarize_factorial80 as summary
from test_warm_dev_summary import make_pool, evaluation, sha, row


def fixture(root):
    pool=make_pool(root,128)
    plan=dict(arms=list(summary.MODES),frame_count=128,evaluation={'development_sources':40},
              training_seeds=[12,13],budget={'evaluate_development_steps':[20,40,80]})
    (root/'FACTORIAL80_PROTOCOL.json').write_text(json.dumps(plan))
    (root/'control/scheduler').mkdir(parents=True)
    (root/'control/scheduler/queue.json').write_text('{"jobs":[]}')
    return pool


def write_arm(root,pool,arm='C_visual',objective='opd',seed=12,step=20,pace='normal'):
    name=summary.eval_id(arm,objective,seed,step)
    path=root/'jobs'/name/'result.json'
    head=root/'jobs'/name.split('_eval')[0]/'run'/f'step{step}'
    head.mkdir(parents=True,exist_ok=True)
    protocol_sha=sha((arm,objective,seed))
    identity=dict(optimizer_step=step,qa_exposures=step*4,protocol_sha256=protocol_sha)
    (head/'CHECKPOINT_IDENTITY.json').write_text(json.dumps(identity))
    (head/'config.json').write_text('{"tiny":true}')
    (head/'SHA256.json').write_text(json.dumps({'model.safetensors':sha(name)}))
    (head.parent/'PROTOCOL.json').write_text(json.dumps(dict(protocol_sha256=protocol_sha,
        config={'seed':seed},evidence_architecture={'time_encoding':'off'})))
    value=evaluation(path,pool,128,range(40),shards=1)
    value['config'].update(head=str(head),n=0,time_mode='real',require_exact_time=True,
        zero_visual=False,evidence_intervention='normal',evidence_mode=summary.MODES[arm])
    value['protocol']['artifacts'].update(checkpoint_identity=identity,
        checkpoint_identity_sha256=summary.warm.file_sha(head/'CHECKPOINT_IDENTITY.json'),
        head_config_sha256=summary.warm.file_sha(head/'config.json'),
        head_weights_sha256={'model.safetensors':sha(name)})
    value['protocol']['evidence']=dict(mode=summary.MODES[arm],dim=256,max_tokens=128,spatial_bins=2)
    for item in value['rows']:
        item.update(first_mismatch=None,evidence=dict(metadata={'encoded_sha256':copy.deepcopy(item['input_identity'])},
            diagnostics={'mode':summary.MODES[arm]}))
        if pace=='slow':
            item.update(rounds=3,anchor_rel=[0,1,2],accepted=[0,0,0],tau=1.,effective_tokens_per_round=4/3)
        elif pace=='fast':
            item.update(rounds=1,anchor_rel=[0],accepted=[2],tau=3.,effective_tokens_per_round=4.)
    path.write_text(json.dumps(value))
    queue=summary.read(root/'control/scheduler/queue.json')
    job=dict(id=name,job_dir=str(path.parent),result={'path':str(path),'status':'LOSSLESS_PASSED'},
             argv=['python','-u','repro/evaluation/eval_video_opd.py','--head',str(head),'--rows',pool['path'],
                   '--target','target','--max-new','1024'])
    queue['jobs'].append(job);(root/'control/scheduler/queue.json').write_text(json.dumps(queue))
    return path,value,job


def test_interaction_computed_per_video_and_two_seeds_never_pooled(tmp_path):
    pool=fixture(tmp_path)
    for seed in (12,13):
        for arm,obj,pace in [('B_text','sft','slow'),('C_visual','sft','normal'),
                             ('B_text','opd','normal'),('C_visual','opd','fast' if seed==12 else 'normal')]:
            write_arm(tmp_path,pool,arm,obj,seed,pace=pace)
    result=summary.build_summary(tmp_path,resamples=1000)
    assert result['status']=='PARTIAL' and not result['invalid'] and not result['invalid_contrasts']
    assert result['complete_evaluations']==8 and len(result['pending'])==28
    assert result['complete_contrasts']==6 and result['planned_contrasts']==42
    for seed,expected in ((12,4/3),(13,-2/3)):
        value=result['by_seed'][str(seed)]['by_step']['20']['interaction_CminusB_OPD_minus_SFT']
        assert value['n']==40 and len(value['paired_videos'])==40
        assert value['delta_effective_tokens_per_round']['mean']==pytest.approx(expected)
        assert value['delta_effective_tokens_per_round']['ci95_percentile']==pytest.approx([expected,expected])
    assert result['holdout_accessed'] is False


@pytest.mark.parametrize('change',['input','ar'])
def test_lossless_individually_but_unmatched_input_or_AR_cannot_compare(tmp_path,change):
    pool=fixture(tmp_path)
    write_arm(tmp_path,pool,'B_text')
    path,value,_=write_arm(tmp_path,pool,'C_visual')
    if change=='input':
        value['rows'][0]['input_identity']['input_ids']=sha('different')
        value['rows'][0]['evidence']['metadata']['encoded_sha256']=copy.deepcopy(value['rows'][0]['input_identity'])
    else:
        value['rows'][0]['ar_ids'][0]=99;value['rows'][0]['spec_ids'][0]=99
    path.write_text(json.dumps(value))
    result=summary.build_summary(tmp_path,resamples=1000)
    comparison=result['by_seed']['12']['by_step']['20']['within_objective']['opd']['C_minus_B']
    assert comparison['status']=='VALIDATION_FAILED'
    assert result['complete_contrasts']==0 and result['status']=='VALIDATION_FAILED'


@pytest.mark.parametrize('change',['missing_row','missing_exit'])
def test_incomplete_is_pending_without_inferred_training_failure(tmp_path,change):
    pool=fixture(tmp_path);path,value,_=write_arm(tmp_path,pool)
    if change=='missing_row':
        value['rows'].pop();path.write_text(json.dumps(value))
    else:(path.parent/'EXIT.json').unlink()
    result=summary.build_summary(tmp_path,resamples=1000)
    assert result['status']=='PARTIAL' and result['complete_evaluations']==0 and not result['invalid']
    assert all(item['training_failure_inferred'] is False for item in result['pending'])


@pytest.mark.parametrize('change',['spec','wrong_step'])
def test_completed_invalid_SPEC_or_checkpoint_step_is_rejected(tmp_path,change):
    pool=fixture(tmp_path);path,value,_=write_arm(tmp_path,pool)
    if change=='spec':value['rows'][0]['spec_ids'][0]=99
    else:value['protocol']['artifacts']['checkpoint_identity']['optimizer_step']=40
    path.write_text(json.dumps(value))
    result=summary.build_summary(tmp_path,resamples=1000)
    assert result['complete_evaluations']==0 and len(result['invalid'])==1


def test_natural_bonus_EOS_prefill_root_is_not_a_coverage_error():
    value=row(0,128)
    value.update(spec_ids=[1,2,3,4,5],ar_ids=[1,2,3,4,5],gen_tokens=5,ar_tokens=5,
                 effective_tokens_per_round=2.5)
    summary.warm.validate_row(value)
    assert value['effective_tokens_per_round']==value['tau']+1/value['rounds']


def test_atomic_repeat_only_replaces_own_summary(tmp_path):
    path=tmp_path/'analysis/factorial80_summary.json'
    summary.write_summary(path,{'schema':summary.SCHEMA,'complete_evaluations':0},tmp_path)
    previous=summary.warm.file_sha(path)
    summary.write_summary(path,{'schema':summary.SCHEMA,'complete_evaluations':1},tmp_path)
    assert summary.read(path)['previous_summary_sha256']==previous
    path.write_text('{"original_result":true}')
    with pytest.raises(ValueError,match='non-factorial'):
        summary.write_summary(path,{'schema':summary.SCHEMA},tmp_path)
    with pytest.raises(ValueError,match='under'):
        summary.write_summary(tmp_path/'jobs/result.json',{},tmp_path)
