"""Fixed-trace real-DFlash CPU integration for serial OPD long-run lifecycle."""
import copy
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import configure
configure()
from local_h100 import train_video_opd_long as entry
from local_h100 import long_training_state as state
from shared import training_resume
from shared.master_optim import MasterWeightAdamW
from opd_official_kl import runtime
from test_local_packed_opd_score import head as _head, trace, serial, Target
from test_local_long_training_state import equal_tree


def head(mode):
    model = _head(mode)
    # Fixed teacher features contain two retained layers; label the last as
    # the production layer33 so the unmodified request contract is exercised.
    model.config.dflash_config['target_layer_ids'] = [1, 33]
    model.config.num_target_layers = 34
    model.target_layer_ids = [1, 33]
    return model


@contextmanager
def evidence_scope(mode):
    with ExitStack() as stack:
        if mode != 'none':
            stack.enter_context(entry.evidence.evidence_runtime(entry.trainer))
            stack.enter_context(entry.boundary.prompt_boundary_runtime(entry.trainer))
        yield


def args():
    return SimpleNamespace(max_pixels=768, time_mode='real', max_new_tokens=16,
        max_seq_len=64, e2e_tv_weight=0., loss_normalization='global_pooled',
        qa_per_update=2, total_updates=2, save_every=1, deadline_utc=None, stop_file=None)


def prep(path, mode):
    path.mkdir()
    protocol=dict(config=dict(total_updates=2, qa_per_update=2), evidence=dict(mode=mode))
    protocol['protocol_sha256']=state._protocol_hash(protocol)
    return dict(out=path, protocol=protocol, start_step=0, end_step=2,
        parent_checkpoint=None, parent_manifest_sha256=None, gpus=[], _test_allow_cpu=True,
        resume_state=None, rows=pd.DataFrame([dict(task='a'),dict(task='b')]).set_index('task'),
        schedule={0:[dict(task='a',slot=0),dict(task='b',slot=1)],
                  1:[dict(task='b',slot=0),dict(task='a',slot=1)]})


def opt(model):
    return MasterWeightAdamW(model, lr=1e-5, betas=(.9,.95), weight_decay=0., clip=1., where='cpu')


def seed():
    random.seed(811); np.random.seed(811); torch.manual_seed(811)


def fixtures(monkeypatch):
    # Frozen teacher trace stand-in only. Head, original serial runtime.score,
    # pooling, global denominator, gradients and optimizer remain real.
    monkeypatch.setattr(entry.trainer.adapters, 'encode', lambda *a,**k:{'input_ids':torch.arange(1,6)[None]})
    monkeypatch.setattr(entry, 'build_evidence_request', lambda *a,**k:copy.deepcopy(trace()['evidence_request']))
    def collect(*a,**k):
        number=int(random.random()*100)+int(np.random.randint(100))+int(torch.randint(100,(1,)))
        return trace(number)
    monkeypatch.setattr(entry.trainer.adapters, 'collect', collect)
    return collect


@pytest.mark.parametrize('mode', ['none','text','visual'])
def test_two_step_real_serial_opd_state_restore_exact_cpu(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(state,'ROOT',tmp_path)
    fixtures(monkeypatch)
    model=head(mode)
    for name,p in model.named_parameters():
        if not name.startswith('evidence_readout.'):p.data=p.data.to(torch.bfloat16)
    target=Target().eval(); processor=SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _:2))
    optimizer=opt(model); prepared=prep(tmp_path/'baseline',mode)
    with evidence_scope(mode):
        seed()
        first=entry.run_update(args(),prepared,0,target,processor,model,optimizer)
        checkpoint=state.save_checkpoint(prepared,1,model,optimizer)
        second=entry.run_update(args(),prepared,1,target,processor,model,optimizer)
        expected_rng=training_resume.capture_rng()
    restored=head(mode)
    for name,p in restored.named_parameters():
        if not name.startswith('evidence_readout.'):p.data=p.data.to(torch.bfloat16)
    restored_opt=opt(restored)
    loaded,protocol,parent=state.read_checkpoint(checkpoint,allow_cpu_test=True)
    resumed=prep(tmp_path/'resumed',mode)
    resumed.update(start_step=1,resume_state=loaded,parent_checkpoint=str(checkpoint),parent_manifest_sha256=parent)
    assert state.restore_if_needed(resumed,restored,restored_opt)==1
    with evidence_scope(mode):
        replay=entry.run_update(args(),resumed,1,target,processor,restored,restored_opt)
    for key in ('loss','pooled_loss','numerator','denominator','gradient_norm','tasks'):
        assert second[key]==replay[key]
    equal_tree(model.state_dict(),restored.state_dict())
    equal_tree(optimizer.state_dict(),restored_opt.state_dict())
    equal_tree(expected_rng,training_resume.capture_rng())
    for (name,p),(_,q) in zip(model.named_parameters(),restored.named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:assert torch.equal(p.grad,q.grad),name
    if mode!='none':
        assert restored.evidence_readout.request is None
        assert restored.evidence_readout.time.weight.grad is None
    assert second['tasks']==['b','a'] and all(p.grad is None for p in target.parameters())


@pytest.mark.parametrize('mode',['none','text','visual'])
def test_update_matches_independent_original_serial_pool_reference(tmp_path,monkeypatch,mode):
    collect=fixtures(monkeypatch)
    a=head(mode); b=copy.deepcopy(a); oa=opt(a); ob=opt(b); target=Target().eval()
    prepared=prep(tmp_path/'run',mode)
    processor=SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _:2))
    with evidence_scope(mode):
        seed(); actual=entry.run_update(args(),prepared,0,target,processor,a,oa)
    seed(); records=[collect(),collect()]
    denominator=sum(runtime.denominator(t) for t in records)
    ob.zero_grad(set_to_none=True); loss=0.
    for record in records:
        for pool in serial(target,b,record,[]):
            loss+=float(pool.numerator.detach())/denominator
            (pool.numerator/denominator).backward()
    ob.sync_grads(); ob.step()
    assert abs(actual['loss']-loss)<1e-12
    equal_tree(a.state_dict(),b.state_dict());equal_tree(oa.state_dict(),ob.state_dict())


def test_owner_loop_absolute_cursor_and_finite_save_budget(tmp_path,monkeypatch):
    prepared=prep(tmp_path/'segment','none'); prepared.update(start_step=1,end_step=2)
    observed=[]; saves=[]
    monkeypatch.setattr(state,'restore_if_needed',lambda *a:1)
    monkeypatch.setattr(entry,'run_update',lambda a,p,c,*rest:observed.append(c) or {'update':c+1})
    monkeypatch.setattr(state,'save_checkpoint',lambda p,s,*a:saves.append(s) or p['out']/f'step{s}')
    monkeypatch.setattr(state,'finish_segment',lambda p,c,r,l:dict(completed=c,reason=r))
    result=entry.run_segment(args(),prepared,None,None,None,None)
    assert observed==[1] and saves==[2] and result['completed']==2


def test_cli_dryrun_no_cuda_or_output(monkeypatch,tmp_path):
    p=prep(tmp_path/'fixture','none')
    called=[]
    monkeypatch.setattr(state,'prepare',lambda a,*rest:called.append(a) or p)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:(_ for _ in ()).throw(AssertionError('CUDA forbidden')))
    result=entry.main(['--pool','p','--schedule','s','--target','t','--init-from','h',
        '--out','o','--total-updates','2','--run-updates','1','--seed','1','--dry-run'])
    assert result==p['protocol'] and called[0].run_updates==1
