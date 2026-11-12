#!/usr/bin/env python3
"""Finite local video warm/SFT from a public DFlash initialization.

This entry deliberately trains only the unmodified DFlash head. It reuses the
repository variable-reveal labels/masks, cached target context path and fp32
master optimizer. This is the repository coverage-CE warm baseline, not a
reproduction of a paper KD recipe. Frozen greedy teacher continuations are content-addressed;
the optimizer always starts afresh. No shared runtime or decoder is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
from _bootstrap import configure
configure()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from shared import common, sft_blocks, training_resume
from shared.master_optim import MasterWeightAdamW
from shared.model_contract import validate_model_contract
from video_caption.train_video_9k import (load_schedule, parse_deadline, stop_reason,
                                         validate_rows, wrap_target)

ALLOWED_ROOTS = (Path('/czsun/mcy/video_opd_pilot/local_h100_20261112'),
                 Path('/mnt/local'), Path('/mnt/local2'), Path('/mnt/local3'))


def safe_output(path):
    resolved = Path(path).resolve()
    if not any(root.exists() and root.resolve() != resolved and
               resolved.is_relative_to(root.resolve()) for root in ALLOWED_ROOTS):
        raise ValueError('output must resolve beneath the existing local_h100_20261112 run root or /mnt/local{,2,3}')
    return resolved


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json_exclusive(path, value):
    """Atomically publish new JSON; never replace another writer's artifact."""
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def validate_head(path):
    cfg = json.loads((Path(path) / 'config.json').read_text())
    draft = cfg.get('dflash_config', {})
    flags = ('use_xattn', 'use_layer_gate', 'use_visual_fovea', 'use_visual_film',
             'use_visual_xattn', 'use_visual_perceiver')
    active = [name for name in flags if draft.get(name, cfg.get(name, False))]
    if draft.get('evidence_mode', cfg.get('evidence_mode', 'none')) != 'none':
        active.append('evidence_mode')
    if active:
        raise ValueError('plain SFT helper does not implement adapter branches: ' + ', '.join(active))
    block = int(draft.get('block_size', cfg.get('block_size', 16)))
    mask = int(draft.get('mask_token_id', cfg.get('mask_token_id', common.MASK_ID)))
    if block < 2 or not 0 <= mask < cfg['vocab_size']:
        raise ValueError('invalid head block size or mask id')
    return block, mask


def visible_gpu_ids(value=None):
    value = os.environ.get('CUDA_VISIBLE_DEVICES', '') if value is None else value
    ids = value.split(',')
    if len(ids) not in (1, 2) or len(set(ids)) != len(ids) or any(
            not item.startswith('GPU-') or item.strip() != item for item in ids):
        raise ValueError('CUDA_VISIBLE_DEVICES must explicitly contain one or two distinct GPU UUIDs')
    return ids


def trim_at_eos(tokens, eos):
    for index, token in enumerate(tokens):
        if token in eos:
            return tokens[:index + 1]
    return tokens


def anchor_plan(prompt_len, total_len, limit, block_size, seed):
    # Root at prompt_len is the first target-generated token. The first EOS is
    # retained in total_len, and no block label can cross that end of sequence.
    candidates = list(range(prompt_len, total_len - 1))
    if not candidates:
        return [], torch.empty(0, dtype=torch.long)
    if len(candidates) > limit:
        index = np.linspace(0, len(candidates) - 1, limit).round().astype(int)
        candidates = [candidates[int(i)] for i in index]
    reveals = sft_blocks.sample_reveals(len(candidates), block_size,
                                       torch.Generator().manual_seed(seed), 'cpu')
    anchors, reveals = sft_blocks.live_anchors(candidates, reveals, total_len, block_size)
    if not anchors:
        # Short outputs can lose every anchor to a long random reveal. Keep a
        # valid root+next-token example with the actual decode reveal of one.
        anchors = [prompt_len]
        reveals = torch.ones(1, dtype=torch.long)
    return anchors, reveals


def checkpoint(out, step, head, optimizer, protocol):
    dest = safe_output(Path(out) / f'step{step}')
    if dest.exists():
        raise FileExistsError(dest)
    tmp = dest.with_name('.' + dest.name + '.tmp-' + uuid.uuid4().hex)
    tmp.mkdir()
    head.save_pretrained(tmp)
    state = dict(head={n: t.detach().cpu() for n, t in head.state_dict().items()},
                 master_optimizer=optimizer.state_dict(), rng=training_resume.capture_rng(),
                 update=step, data_cursor=step, training_protocol=protocol,
                 protocol_sha256=protocol['protocol_sha256'],
                 initialization='weights_only_fresh_optimizer', resume_supported=False,
                 arm='LOCAL_VIDEO_VARIABLE_REVEAL_SFT')
    torch.save(state, tmp / 'training_state.pt')
    identity = dict(schema='local_video_sft_checkpoint_v1', optimizer_step=step,
                    qa_exposures=step * protocol['config']['qa_per_update'],
                    protocol_sha256=protocol['protocol_sha256'],
                    initialization_weights_sha256=protocol['hashes']['initial_head_weights'],
                    pool_sha256=protocol['hashes']['pool'], schedule_sha256=protocol['hashes']['schedule'],
                    target_weights_sha256=protocol['hashes']['target_weights'],
                    target_dtype='float32', head_dtype='bfloat16', master_dtype='float32')
    write_json_exclusive(tmp / 'CHECKPOINT_IDENTITY.json', identity)
    write_json_exclusive(tmp / 'SHA256.json', {
        p.name: training_resume.file_hash(p) for p in sorted(tmp.iterdir()) if p.is_file()})
    os.rename(tmp, dest)
    return str(dest)


def build_protocol(a, inputs, dimensions, gpus):
    init = Path(a.init_from)
    target = Path(a.target)
    def weights(path):
        files = sorted(path.glob('*.safetensors')) + sorted(path.glob('pytorch_model*.bin'))
        if not files:
            raise ValueError(f'model weights absent: {path}')
        return {p.name: training_resume.file_hash(p) for p in files}
    config = {k: v for k, v in vars(a).items() if k not in ('out', 'dry_run', 'teacher_cache')}
    protocol = dict(schema=1, entry='local_h100/train_video_sft.py', config=config,
                    initialization='weights_only_fresh_optimizer', resume_supported=False,
                    world_size=1, visible_gpu_uuids=gpus, target_dtype='float32',
                    head_dtype='bfloat16', master_dtype='float32',
                    objective='mean_QA weighted variable-reveal CE on frozen greedy target tokens',
                    eos_policy='trim immediately after first target EOS, including first generated token',
                    context_path='use_cache=True full frozen replay',
                    teacher_generation='independent greedy AR; same content-addressed continuation across arms',
                    model_dimensions=dimensions, inputs=inputs,
                    optimizer=dict(name='AdamW', betas=list(common.BETAS),
                                   weight_decay=common.WEIGHT_DECAY, where='cpu', clip=a.clip),
                    hashes=dict(pool=training_resume.file_hash(a.pool),
                                schedule=training_resume.file_hash(a.schedule),
                                target_config=training_resume.file_hash(target / 'config.json'),
                                target_weights=weights(target),
                                initial_head_config=training_resume.file_hash(init / 'config.json'),
                                initial_head_weights=weights(init),
                                trainer=training_resume.file_hash(__file__),
                                sft_blocks=training_resume.file_hash(sft_blocks.__file__),
                                common=training_resume.file_hash(common.__file__),
                                decoder=training_resume.file_hash(REPRO.parent / 'GLANCE_repo/glance/decode.py'),
                                videoencode=training_resume.file_hash(REPRO / 'qa_opd_fair/videoencode.py'),
                                master_optim=training_resume.file_hash(sys.modules[MasterWeightAdamW.__module__].__file__)))
    protocol['protocol_sha256'] = digest(protocol)
    return protocol


def teacher_key(inputs, protocol, eos):
    signatures = {}
    for name, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor):
            t = tensor.detach().cpu().contiguous()
            signatures[name] = dict(shape=list(t.shape), dtype=str(t.dtype),
                                    sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())
    identity = dict(schema=1, encoded_inputs=signatures,
                    target_config=protocol['hashes']['target_config'],
                    target_weights=protocol['hashes']['target_weights'],
                    target_dtype='float32', max_new_tokens=protocol['config']['max_new_tokens'],
                    eos=sorted(eos), decoder_sha256=protocol['hashes'].get('decoder'),
                    generation='greedy_generate_trim_first_eos_v1')
    return digest(identity), identity


def get_teacher(target, inputs, protocol, cache_root, eos):
    from glance.decode import greedy_generate
    from glance.vl import VisionPositions
    key, identity = teacher_key(inputs, protocol, eos)
    path = cache_root / f'{key}.json'
    prompt = inputs['input_ids'][0].tolist()
    if path.exists():
        record = json.loads(path.read_text())
        if record.get('identity') != identity or record.get('prompt_ids') != prompt:
            raise ValueError('teacher cache identity mismatch')
        tokens = record['gen_ids']
        if not tokens or tokens != trim_at_eos(tokens, eos):
            raise ValueError('teacher cache has empty output or tokens beyond first EOS')
        if record.get('gen_ids_sha256') != digest(tokens):
            raise ValueError('teacher cache token hash mismatch')
        return tokens, True, key
    dev = 'cuda:0'
    x = {k: v.to(dev) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    pos = VisionPositions(**{k: x[k] for k in ('pixel_values_videos', 'video_grid_thw') if k in x})
    generated = greedy_generate(target, x['input_ids'], protocol['config']['max_new_tokens'],
                                stop_token_ids=sorted(eos), positions=pos)
    tokens = trim_at_eos(generated[0, len(prompt):].tolist(), eos)
    if not tokens:
        raise ValueError('empty teacher continuation')
    record = dict(identity=identity, prompt_ids=prompt, gen_ids=tokens,
                  gen_ids_sha256=digest(tokens))
    try:
        write_json_exclusive(path, record)
    except FileExistsError:
        if json.loads(path.read_text()) != record:
            raise RuntimeError('two workers produced different teacher trajectories for the same cache key')
    return tokens, False, key


def hidden_loss(head, target, ctx, full, anchors, reveals, mask_id):
    block = head.block_size
    dev = ctx.device
    blocks, labels, supervised, weights = sft_blocks.build_blocks(
        full, anchors, reveals.to(dev), block, mask_id, dev)
    with torch.no_grad():
        emb_layer = target.get_input_embeddings()
        emb = emb_layer(blocks.to(emb_layer.weight.device)).to(device=dev, dtype=ctx.dtype)
    positions = (torch.tensor(anchors, device=dev)[:, None] + torch.arange(block, device=dev)).reshape(1, -1)
    hidden = sft_blocks.forward_anchors(
        head, ctx, emb.reshape(1, -1, emb.shape[-1]),
        torch.arange(ctx.shape[1], device=dev)[None], positions,
        sft_blocks.build_anchor_mask(anchors, block, ctx.shape[1], dev, ctx.dtype))
    keep = supervised.reshape(-1).bool()
    selected = hidden[0][keep]
    # Match deployment's raw target LM head; do not use optional head logits scaling.
    lm = target.lm_head
    logits = lm(selected.to(device=lm.weight.device, dtype=lm.weight.dtype)).to(dev)
    y, w = labels.reshape(-1)[keep], weights.reshape(-1)[keep]
    nll = F.cross_entropy(logits.float(), y, reduction='none')
    loss = (nll * w).sum() / w.sum().clamp_min(1.0)
    return loss, dict(positions=int(keep.sum()), weight_sum=float(w.sum()),
                      correct=int((logits.argmax(-1) == y).sum()), anchors=len(anchors))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'schedule', 'target', 'init-from', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--updates', type=int, required=True, help='explicit finite optimizer update budget')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--qa-per-update', type=int, default=4)
    p.add_argument('--anchors-per-qa', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-seq-len', type=int, default=16384)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--teacher-cache', help='optional shared content-addressed AR cache; writes only allowed storage roots')
    p.add_argument('--deadline-utc')
    p.add_argument('--stop-file')
    p.add_argument('--device-map', choices=('single', 'balanced'), default='single')
    p.add_argument('--dry-run', action='store_true')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    for name in ('updates', 'qa_per_update', 'anchors_per_qa', 'save_every', 'max_new_tokens', 'max_seq_len'):
        if getattr(a, name) < 1:
            raise ValueError(f'{name} must be positive')
    if a.max_seq_len <= a.max_new_tokens:
        raise ValueError('max-seq-len must leave room for the prompt')
    if not all(math.isfinite(x) and x > 0 for x in (a.lr, a.clip)):
        raise ValueError('lr and clip must be finite and positive')
    gpus = visible_gpu_ids()
    if (len(gpus) == 1) != (a.device_map == 'single'):
        raise ValueError('one UUID requires single; two UUIDs require balanced')
    deadline = parse_deadline(a.deadline_utc)
    out = safe_output(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('output must be a new or empty directory')
    teacher_cache = safe_output(a.teacher_cache or out / 'teacher_cache')
    rows = pd.read_parquet(a.pool).set_index('task', verify_integrity=True)
    schedule = load_schedule(a.schedule, rows, a.updates, a.qa_per_update)
    inputs_info = validate_rows(rows, schedule, 'real')
    dimensions = validate_model_contract(a.target, a.init_from)
    block, mask_id = validate_head(a.init_from)
    protocol = build_protocol(a, inputs_info, dimensions, gpus)
    if a.dry_run:
        print(json.dumps(protocol, indent=2))
        return protocol
    if not hasattr(torch.nn.utils, 'get_total_norm'):
        raise RuntimeError('this environment lacks torch.nn.utils.get_total_norm; select a supported PyTorch environment')
    if stop_reason(deadline, a.stop_file):
        raise RuntimeError('stop requested before model load')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    teacher_cache.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(out / 'PROTOCOL.json', protocol)
    dev = 'cuda:0'
    torch.cuda.set_device(0)
    target, processor = common.load_target(a.target, device=dev, dtype=torch.float32,
                                           device_map='balanced' if len(gpus) == 2 else None)
    if len(gpus) == 2:
        wrap_target(target, dev)
    target.eval().requires_grad_(False)
    head = common.load_head(a.init_from, dtype=torch.bfloat16, device=dev).eval()
    head.requires_grad_(True)
    if int(head.block_size) != block:
        raise ValueError('loaded block size differs from the preflight config')
    optimizer = MasterWeightAdamW(head, lr=a.lr, betas=common.BETAS,
                                  weight_decay=common.WEIGHT_DECAY, where='cpu', clip=a.clip)
    from shared.draft_geometry import eos_ids
    eos = eos_ids(target)
    completed, last_checkpoint, reason = 0, None, None
    run_started = time.time()
    try:
        for cursor in range(a.updates):
            reason = stop_reason(deadline, a.stop_file)
            if reason:
                break
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            records = []
            for item in schedule[cursor]:
                row = rows.loc[item['task']].to_dict()
                from qa_opd_fair.videoencode import encode_inputs
                inputs = encode_inputs(processor, row, 'real')
                prompt_len = int(inputs['input_ids'].shape[1])
                if prompt_len + a.max_new_tokens > a.max_seq_len:
                    raise ValueError(f"{item['task']}: prompt plus output budget exceeds max-seq-len")
                print(f"[teacher] update={cursor + 1} slot={item['slot']} task={item['task']} prompt_tokens={prompt_len}", flush=True)
                tokens, cached, cache_key = get_teacher(target, inputs, protocol, teacher_cache, eos)
                if len(tokens) < 2:
                    raise ValueError(f"{item['task']}: fewer than two teacher tokens; cannot train a committed-root block")
                full = torch.tensor(inputs['input_ids'][0].tolist() + tokens, device=dev)
                anchors, reveals = anchor_plan(prompt_len, len(full), a.anchors_per_qa, block,
                                                a.seed + cursor * 100003 + int(item['slot']))
                target.model.rope_deltas = None
                ctx = common.target_context_cached(
                    target, full[None], pixel_values_videos=inputs['pixel_values_videos'].to(dev),
                    video_grid_thw=inputs['video_grid_thw'].to(dev)).to(dtype=torch.bfloat16, device=dev)
                loss, stats = hidden_loss(head, target, ctx, full, anchors, reveals, mask_id)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('nonfinite SFT loss')
                (loss / a.qa_per_update).backward()
                if any(p.grad is not None for p in target.parameters()):
                    raise RuntimeError('frozen target received a gradient')
                records.append(dict(task=item['task'], loss=float(loss.detach()), prompt_tokens=prompt_len,
                                    generated_tokens=len(tokens), teacher_cache_hit=cached,
                                    teacher_cache_key=cache_key, **stats))
                del ctx, full, loss
            optimizer.sync_grads()
            norm = optimizer.grad_norm()
            if not math.isfinite(norm):
                raise RuntimeError('nonfinite gradient')
            update_norms = optimizer.step()
            completed = cursor + 1
            record = dict(update=completed, planned_updates=a.updates,
                          loss=sum(r['loss'] for r in records) / a.qa_per_update,
                          gradient_norm=norm, written_update_norm=math.sqrt(sum(v*v for v in update_norms.values())),
                          seconds=time.time() - start, elapsed_seconds=time.time() - run_started,
                          protocol_sha256=protocol['protocol_sha256'], qa=records)
            with (out / 'log.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            if completed % a.save_every == 0 or completed == a.updates:
                last_checkpoint = checkpoint(out, completed, head, optimizer, protocol)
        if completed and last_checkpoint != str(out / f'step{completed}'):
            last_checkpoint = checkpoint(out, completed, head, optimizer, protocol)
        result = dict(status='COMPLETED' if completed == a.updates else 'STOPPED',
                      completed_updates=completed, planned_updates=a.updates,
                      stop_reason=reason, last_checkpoint=last_checkpoint, resume_supported=False,
                      protocol_sha256=protocol['protocol_sha256'])
        write_json_exclusive(out / 'RUN_STATUS.json', result)
        print(json.dumps(result), flush=True)
        return result
    except Exception as exc:
        write_json_exclusive(out / 'FAILED.json', dict(completed_updates=completed,
                             planned_updates=a.updates, last_checkpoint=last_checkpoint,
                             error_type=type(exc).__name__, error=str(exc),
                             protocol_sha256=protocol['protocol_sha256']))
        raise


if __name__ == '__main__':
    main()
