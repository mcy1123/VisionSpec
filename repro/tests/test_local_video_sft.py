"""CPU tests of the local bounded video SFT entry; no GPU model load."""
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_h100 import train_video_sft as train
from dflash.model import DFlashDraftModel


def test_finite_budget_and_explicit_uuid_requirement():
    with pytest.raises(SystemExit):
        train.parser().parse_args(['--pool', 'p', '--schedule', 's', '--target', 't',
                                   '--init-from', 'i', '--out', 'o', '--seed', '0'])
    assert train.visible_gpu_ids('GPU-first,GPU-second') == ['GPU-first', 'GPU-second']
    assert train.visible_gpu_ids('GPU-first') == ['GPU-first']
    for bad in ('', '0', '0,1', 'GPU-a,GPU-a', 'GPU-a, GPU-b', 'GPU-a,GPU-b,GPU-c'):
        with pytest.raises(ValueError):
            train.visible_gpu_ids(bad)


def test_storage_uses_realpath_and_atomic_json_cannot_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'ALLOWED_ROOTS', (tmp_path / 'allowed',))
    (tmp_path / 'allowed').mkdir()
    assert train.safe_output(tmp_path / 'allowed' / 'run') == tmp_path / 'allowed' / 'run'
    (tmp_path / 'allowed' / 'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        train.safe_output(tmp_path / 'allowed' / 'escape' / 'outside')
    p = tmp_path / 'artifact.json'
    train.write_json_exclusive(p, {'original': True})
    with pytest.raises(FileExistsError):
        train.write_json_exclusive(p, {'original': False})
    assert json.loads(p.read_text()) == {'original': True}
    assert not list(tmp_path.glob('*.tmp-*'))


@pytest.mark.parametrize('field', ['use_xattn', 'use_layer_gate', 'use_visual_fovea',
                                   'use_visual_film', 'use_visual_xattn', 'use_visual_perceiver',
                                   'evidence_mode'])
def test_rejects_adapter_forward_that_sft_helper_would_bypass(tmp_path, field):
    draft = {field: 'visual' if field == 'evidence_mode' else True}
    (tmp_path / 'config.json').write_text(json.dumps({'vocab_size': 200000, 'dflash_config': draft}))
    with pytest.raises(ValueError, match='adapter branches'):
        train.validate_head(tmp_path)


def test_first_eos_and_short_answer_labels_are_safe():
    assert train.trim_at_eos([9, 3, 4], {9}) == [9]
    assert train.trim_at_eos([3, 9, 4], {9}) == [3, 9]
    assert train.trim_at_eos([3, 4], {9}) == [3, 4]
    for seed in range(20):
        anchors, reveals = train.anchor_plan(10, 12, 16, 16, seed)
        assert anchors == [10] and reveals.tolist() == [1]
        full = torch.arange(12)
        blocks, labels, supervised, weights = train.sft_blocks.build_blocks(
            full, anchors, reveals, 16, 100, 'cpu')
        assert labels[supervised.bool()].tolist() == [11]
        assert blocks[0, 0].item() == 10
    anchors, reveals = train.anchor_plan(10, 11, 16, 16, 0)
    assert anchors == [] and len(reveals) == 0


def test_teacher_cache_identity_binds_pixels_weights_output_budget():
    inputs = {'input_ids': torch.tensor([[1, 2]]), 'pixel_values_videos': torch.zeros(2, 4),
              'video_grid_thw': torch.tensor([[1, 2, 2]])}
    protocol = {'hashes': {'target_config': 'cfg', 'target_weights': {'model': 'a'}},
                'config': {'max_new_tokens': 256}}
    key, identity = train.teacher_key(inputs, protocol, {9})
    assert train.teacher_key(inputs, protocol, {9})[0] == key
    inputs['pixel_values_videos'][0, 0] = 1
    assert train.teacher_key(inputs, protocol, {9})[0] != key
    inputs['pixel_values_videos'].zero_()
    protocol['hashes']['target_weights']['model'] = 'b'
    assert train.teacher_key(inputs, protocol, {9})[0] != key
    protocol['hashes']['target_weights']['model'] = 'a'
    protocol['config']['max_new_tokens'] = 255
    assert train.teacher_key(inputs, protocol, {9})[0] != key


def test_actual_plain_head_sft_has_gradients_and_frozen_teacher():
    torch.manual_seed(4)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      vocab_size=64, eos_token_id=63)
    cfg._attn_implementation = 'sdpa'
    target = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    hcfg = Qwen3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       vocab_size=64, num_target_layers=4, block_size=4,
                       dflash_config={'target_layer_ids': [0, 2], 'mask_token_id': 62})
    hcfg._attn_implementation = 'sdpa'
    head = DFlashDraftModel(hcfg).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    with torch.no_grad():
        hidden = target(ids, output_hidden_states=True).hidden_states
        ctx = torch.cat((hidden[1], hidden[3]), -1)
    anchors, reveals = [3, 5], torch.tensor([1, 2])
    loss, stats = train.hidden_loss(head, target, ctx, ids[0], anchors, reveals, 62)
    loss.backward()
    assert torch.isfinite(loss)
    assert stats['positions'] == 5
    assert head.fc.weight.grad is not None and head.fc.weight.grad.norm() > 0
    assert all(p.grad is None for p in target.parameters())
    # The custom block path must reproduce stock head with only its allowed
    # context prefix. Full-sequence fused features cannot leak past the anchor.
    anchor = 3
    blocks, *_ = train.sft_blocks.build_blocks(ids[0], [anchor], torch.tensor([1]), 4, 62, 'cpu')
    noise = target.get_input_embeddings()(blocks)
    stock = head(target_hidden=ctx[:, :anchor], noise_embedding=noise,
                 position_ids=torch.arange(anchor + 4)[None])
    batched = train.sft_blocks.forward_anchors(
        head, ctx, noise, torch.arange(ctx.shape[1])[None], torch.arange(anchor, anchor + 4)[None],
        train.sft_blocks.build_anchor_mask([anchor], 4, ctx.shape[1], 'cpu', noise.dtype))
    torch.testing.assert_close(stock, batched, atol=3e-6, rtol=3e-5)


def test_checkpoint_manifest_is_accepted_by_evaluator_and_never_overwrites(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from evaluation.eval_video_opd import evaluation_artifacts
    monkeypatch.setattr(train, 'ALLOWED_ROOTS', (tmp_path,))
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
        def save_pretrained(self, path):
            (path / 'config.json').write_text('{"test":true}')
            save_file(self.state_dict(), str(path / 'model.safetensors'))
    class Optimizer:
        def state_dict(self):
            return {'master': {'weight': torch.ones(2)}, 'optimizer': {}}
    protocol = {'protocol_sha256': 'protocol', 'config': {'qa_per_update': 4},
                'hashes': {'initial_head_weights': {'model': 'head'}, 'target_weights': {'model': 'target'},
                           'pool': 'pool', 'schedule': 'schedule'}}
    root = tmp_path / 'run'
    root.mkdir()
    head = Head()
    ckpt = train.checkpoint(root, 3, head, Optimizer(), protocol)
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'config.json').write_text('{}')
    artifacts = evaluation_artifacts(target, ckpt)
    assert artifacts['checkpoint_identity']['optimizer_step'] == 3
    assert artifacts['checkpoint_identity']['qa_exposures'] == 12
    with pytest.raises(FileExistsError):
        train.checkpoint(root, 3, head, Optimizer(), protocol)
    (Path(ckpt) / 'config.json').write_text('{"altered":true}')
    with pytest.raises(ValueError, match='config/weights'):
        evaluation_artifacts(target, ckpt)
