"""CPU tests of explicit no-time matched evidence initialization and reload."""
import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen3Config

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_h100 import init_matched_evidence as local
from video_caption import init_evidence_heads as base
from shared import common
from dflash.model import DFlashDraftModel


def source(path):
    cfg=Qwen3Config(hidden_size=16,intermediate_size=24,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=32,
        num_target_layers=36,block_size=4,dflash_config={
            'target_layer_ids':[1,9,17,25,33],'mask_token_id':31,'evidence_mode':'none'})
    cfg._attn_implementation='sdpa'
    head=DFlashDraftModel(cfg).bfloat16()
    head.save_pretrained(path)
    return {n:v.clone() for n,v in head.state_dict().items()}


def test_matched_saved_tensors_timeoff_hash_and_fp32_reload(tmp_path,monkeypatch):
    monkeypatch.setattr(local,'LOCAL_ROOT',tmp_path)
    old_roots,old_make=base.ALLOWED_ROOTS,base.make_matched_head
    original=source(tmp_path/'source')
    result=local.initialize(tmp_path/'source',tmp_path/'arms',dim=8,max_tokens=8,seed=9)
    assert base.ALLOWED_ROOTS is old_roots and base.make_matched_head is old_make
    assert result['evidence_time_encoding']=='off'
    assert result['official_vispec_reproduction'] is False
    assert result['matched_saved_weights_verified'] is True
    assert result['total_parameter_count']-result['trainable_parameter_count']==6*8
    assert result['evidence_total_parameter_count']-result['evidence_trainable_parameter_count']==6*8
    states=[]
    for arm,mode in (('B_text','text'),('C_visual','visual')):
        path=tmp_path/'arms'/arm
        state=base.read_weights(base.weight_files(path));states.append(state)
        manifest=json.loads((path/'INIT_MANIFEST.json').read_text())
        assert manifest['evidence_time_encoding']=='off'
        assert manifest['added_tensor_sha256']==base.state_digest({n:v for n,v in state.items() if n.startswith(base.PREFIX)})
        assert all(torch.equal(state[n],v) and state[n].dtype==v.dtype for n,v in original.items())
        head=common.load_head(path,device='cpu',dtype=torch.bfloat16)
        assert head.config.dflash_config['evidence_mode']==mode
        assert head.config.dflash_config['evidence_time_encoding']=='off'
        for n,p in head.evidence_readout.named_parameters():
            assert p.dtype==torch.float32
            assert torch.equal(p,state['evidence_readout.'+n])
        head.requires_grad_(True)
        local.freeze_time_off(head)
        assert head.evidence_readout.time.weight.requires_grad is False
        assert all(p.requires_grad for n,p in head.named_parameters() if n!='evidence_readout.time.weight')
        assert not torch.count_nonzero(head.evidence_readout.output.weight)
    assert states[0].keys()==states[1].keys()
    assert all(torch.equal(states[0][n],states[1][n]) for n in states[0])
    after=base.read_weights(base.weight_files(tmp_path/'source'))
    assert all(torch.equal(after[n],v) for n,v in original.items())


def test_frozen_time_contract_refuses_implicit_or_corrupted_time(tmp_path,monkeypatch):
    monkeypatch.setattr(local,'LOCAL_ROOT',tmp_path)
    source(tmp_path/'source')
    local.initialize(tmp_path/'source',tmp_path/'arms',dim=8,max_tokens=8)
    head=common.load_head(tmp_path/'arms'/'C_visual',device='cpu',dtype=torch.bfloat16)
    with torch.no_grad():head.evidence_readout.time.weight[0,0]=1.
    with pytest.raises(ValueError,match='nonzero'):
        local.freeze_time_off(head)
    head.config.dflash_config.pop('evidence_time_encoding')
    with pytest.raises(ValueError,match='explicit'):
        local.freeze_time_off(head)


def test_dryrun_no_writes_no_overwrite_and_patches_restore(tmp_path,monkeypatch):
    monkeypatch.setattr(local,'LOCAL_ROOT',tmp_path)
    source(tmp_path/'source')
    result=local.initialize(tmp_path/'source',tmp_path/'dry',dry_run=True)
    assert result['evidence_max_tokens']==128 and result['evidence_dim']==256
    assert result['evidence_time_encoding']=='off' and not (tmp_path/'dry').exists()
    with pytest.raises(FileExistsError):
        local.initialize(tmp_path/'source',tmp_path/'source',dry_run=True)
    before=(base.ALLOWED_ROOTS,base.make_matched_head)
    with pytest.raises(RuntimeError):
        with local.local_initializer({}):raise RuntimeError('test')
    assert (base.ALLOWED_ROOTS,base.make_matched_head)==before
