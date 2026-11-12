"""Strict new resume contract: real serialization, Adam/RNG and fail-before-mutate."""
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()
from local_h100 import long_training_state as long
from shared.master_optim import MasterWeightAdamW
from shared import training_resume


class Head(torch.nn.Linear):
    def save_pretrained(self, path):
        save_file({k: v.detach().contiguous() for k, v in self.state_dict().items()}, str(path / 'model.safetensors'))
        (path / 'config.json').write_text('{}')


def fixture(tmp_path, monkeypatch, dtype=torch.bfloat16):
    monkeypatch.setattr(long, 'ROOT', tmp_path)
    head = Head(3, 2).to(dtype)
    opt = MasterWeightAdamW(head, lr=1e-5, betas=(.9, .95), weight_decay=0., clip=1., where='cpu')
    protocol = dict(schema=long.SCHEMA, config=dict(total_updates=2, qa_per_update=1), objective='fixture')
    protocol['protocol_sha256'] = long._protocol_hash(protocol)
    out = tmp_path / 'baseline'; out.mkdir()
    prepared = dict(protocol=protocol, out=out, start_step=0, end_step=2, gpus=[],
                    parent_checkpoint=None, parent_manifest_sha256=None, resume_state=None, _test_allow_cpu=True)
    return head, opt, prepared


def equal_tree(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_tree(x, y)
    else:
        assert a == b


def update(head, optimizer):
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(2, 3).to(head.weight.dtype)
    scalar = random.random() + float(np.random.rand())
    loss = (head(x).float() - scalar).square().mean()
    loss.backward(); optimizer.sync_grads(); optimizer.step()
    return float(loss.detach()), {n: p.grad.clone() for n, p in head.named_parameters()}


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_full_serialized_resume_matches_all_parameters_adam_gradients_and_rng(tmp_path, monkeypatch, dtype):
    a, oa, prepared = fixture(tmp_path, monkeypatch, dtype)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    update(a, oa)
    rng = training_resume.capture_rng()
    checkpoint = long.save_checkpoint(prepared, 1, a, oa)
    equal_tree(rng, training_resume.capture_rng())
    tail = update(a, oa); final_rng = training_resume.capture_rng()
    b = Head(3, 2).to(dtype)  # intentionally consumes RNG before restore
    ob = MasterWeightAdamW(b, lr=1e-5, betas=(.9, .95), weight_decay=0., clip=1., where='cpu')
    loaded, protocol, sha = long.read_checkpoint(checkpoint, allow_cpu_test=True)
    next_prepared = dict(prepared, start_step=1, resume_state=loaded, parent_manifest_sha256=sha)
    assert long.restore_if_needed(next_prepared, b, ob) == 1
    audit = json.loads((prepared['out'] / 'RESTORE_AUDIT.json').read_text())
    assert audit['status'] == 'EXACT_RESTORE_VERIFIED' and audit['complete_state_exact'] is True
    assert audit['next_update_started'] is False and audit['restored_step'] == 1
    assert audit['parent_manifest_sha256'] == sha and set(audit['checks']) == {'head', 'master_optimizer', 'rng'}
    equal_tree(rng, training_resume.capture_rng())
    resumed = update(b, ob)
    equal_tree(tail, resumed); equal_tree(a.state_dict(), b.state_dict())
    equal_tree(oa.state_dict(), ob.state_dict()); equal_tree(final_rng, training_resume.capture_rng())
    with pytest.raises(ValueError, match='CPU fixture'):
        long.read_checkpoint(checkpoint)
    with pytest.raises(FileExistsError):
        long.save_checkpoint(prepared, 1, a, oa)


@pytest.mark.parametrize('mutation', ['clip', 'where', 'order', 'shape', 'head', 'lr', 'rng', 'cursor', 'identity'])
def test_bad_resume_is_rejected_before_any_parameter_mutation(tmp_path, monkeypatch, mutation):
    a, oa, prepared = fixture(tmp_path, monkeypatch)
    update(a, oa); ckpt = long.save_checkpoint(prepared, 1, a, oa)
    blob, _, _ = long.read_checkpoint(ckpt, allow_cpu_test=True)
    blob = copy.deepcopy(blob)
    if mutation == 'clip': blob['master_optimizer']['clip'] = 9.
    if mutation == 'where': blob['master_optimizer']['where'] = 'gpu'
    if mutation == 'order': blob['parameter_layout'].reverse()
    if mutation == 'shape': blob['master_optimizer']['master']['weight'] = torch.ones(1, 3)
    if mutation == 'head': blob['head']['weight'][0, 0] += 1
    if mutation == 'lr': blob['master_optimizer']['optimizer']['param_groups'][0]['lr'] = .3
    if mutation == 'rng': blob['rng'].pop('numpy')
    if mutation == 'cursor': blob['data_cursor'] = 0
    if mutation == 'identity': blob['resume_identity'] = {'bad': True}
    b = Head(3, 2).to(torch.bfloat16)
    ob = MasterWeightAdamW(b, lr=1e-5, betas=(.9, .95), weight_decay=0., clip=1., where='cpu')
    before = copy.deepcopy(b.state_dict()); masters = copy.deepcopy(ob.state_dict())
    with pytest.raises(ValueError):
        long.restore_if_needed(dict(prepared, start_step=1, resume_state=blob), b, ob)
    equal_tree(before, b.state_dict()); equal_tree(masters, ob.state_dict())


def test_postload_audit_detects_incomplete_optimizer_loading_before_next_update(tmp_path, monkeypatch):
    a, oa, prepared = fixture(tmp_path, monkeypatch)
    update(a, oa); ckpt = long.save_checkpoint(prepared, 1, a, oa)
    blob, _, _ = long.read_checkpoint(ckpt, allow_cpu_test=True)
    b = Head(3, 2).to(torch.bfloat16)
    ob = MasterWeightAdamW(b, lr=1e-5, betas=(.9, .95), weight_decay=0., clip=1., where='cpu')
    original = ob.load_state_dict
    def defective_load(value):
        original(value)
        ob.master['weight'][0, 0] += 1
    monkeypatch.setattr(ob, 'load_state_dict', defective_load)
    with pytest.raises(ValueError, match='restored tensor is not exact: master_optimizer.master.weight'):
        long.restore_if_needed(dict(prepared, start_step=1, resume_state=blob), b, ob)
    assert not (prepared['out'] / 'RESTORE_AUDIT.json').exists()


def test_corruption_missing_rng_old_schema_and_incomplete_files_fail(tmp_path, monkeypatch):
    a, oa, prepared = fixture(tmp_path, monkeypatch)
    update(a, oa); ckpt = long.save_checkpoint(prepared, 1, a, oa)
    manifest = json.loads((ckpt/'SHA256.json').read_text())
    blob = torch.load(ckpt/'training_state.pt', weights_only=False)
    for mutation in ('rng', 'schema'):
        changed = copy.deepcopy(blob)
        if mutation == 'rng': changed.pop('rng')
        else: changed['long_state_schema'] = 'old'
        torch.save(changed, ckpt/'training_state.pt')
        manifest['training_state.pt'] = training_resume.file_hash(ckpt/'training_state.pt')
        (ckpt/'SHA256.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError): long.read_checkpoint(ckpt, allow_cpu_test=True)
    (ckpt/'config.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='checksum'):
        long.read_checkpoint(ckpt, allow_cpu_test=True)
    (ckpt/'SHA256.json').write_text('{}')
    with pytest.raises(ValueError, match='complete long-run'):
        long.read_checkpoint(ckpt, allow_cpu_test=True)


def test_save_failure_never_publishes_a_checkpoint_or_overwrites_history(tmp_path, monkeypatch):
    a, oa, prepared = fixture(tmp_path, monkeypatch)
    update(a, oa)
    def fail(*args, **kwargs): raise RuntimeError('injected save failure')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(RuntimeError, match='injected'):
        long.save_checkpoint(prepared, 1, a, oa)
    assert not (prepared['out']/'step1').exists()


def test_path_symlink_escape_and_budget_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(long, 'ROOT', tmp_path/'allowed'); long.ROOT.mkdir()
    outside=tmp_path/'outside'; outside.mkdir()
    (long.ROOT/'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError): long.safe_path(long.ROOT/'escape'/'x')
    for objective in ('sft', 'opd'):
        parser = long.argument_parser('test', objective)
        with pytest.raises(SystemExit): parser.parse_args([])


def test_cpu_dryrun_preflight_protocol_is_stable_across_accepted_resume(tmp_path, monkeypatch):
    # Pure CPU fake inputs; the real preflight builds full identity and checks
    # that operational segment switches do not change the experiment hash.
    import pandas as pd
    from types import SimpleNamespace
    monkeypatch.setattr(long, 'ROOT', tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-fixture')
    target=tmp_path/'target'; target.mkdir(); init=tmp_path/'init'; init.mkdir()
    for p in (target,init):
        (p/'config.json').write_text(json.dumps({'hidden_size':3,'vocab_size':8,'mask_token_id':1,'block_size':4}))
        (p/'model.safetensors').write_bytes(b'fixture')
    pool=tmp_path/'pool.parquet'; pd.DataFrame([dict(task='a',split='train',prompt='x')]).to_parquet(pool)
    schedule=tmp_path/'schedule.jsonl'; schedule.write_text(''.join(json.dumps(dict(update=i,slot=0,task='a'))+'\n' for i in range(2)))
    monkeypatch.setattr(long, 'validate_rows', lambda *a: [])
    monkeypatch.setattr(long, 'validate_model_contract', lambda *a: {'hidden_size':3})
    args=long.argument_parser('test','opd').parse_args(['--pool',str(pool),'--schedule',str(schedule),'--target',str(target),'--out',str(tmp_path/'segment1'),'--init-from',str(init),'--total-updates','2','--run-updates','1','--qa-per-update','1','--seed','3','--dry-run'])
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: (_ for _ in ()).throw(AssertionError('CUDA forbidden')))
    monkeypatch.setattr(torch.backends.cudnn, 'version', lambda *a: (_ for _ in ()).throw(AssertionError('cuDNN version probes CUDA')))
    first=long.prepare(args,'opd',[])
    assert not first['out'].exists()
    previous=copy.deepcopy(first['protocol'])
    fake=dict(update_step=1,execution={'cuda_uuids':['GPU-fixture']})
    monkeypatch.setattr(long,'read_checkpoint',lambda *a:(fake,previous,'parent'))
    args.init_from=None; args.resume_from=str(init); args.out=str(tmp_path/'segment2')
    second=long.prepare(args,'opd',[])
    assert first['protocol']==second['protocol'] and second['start_step']==1
    args.max_new_tokens=512
    with pytest.raises(ValueError,match='identity mismatch'):long.prepare(args,'opd',[])
