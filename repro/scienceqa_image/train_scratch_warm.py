#!/usr/bin/env python3
"""Resumable image warm training with the historical GLANCE coverage objective.

The inputs are the original COCO/TextVQA target-greedy rows. The text-only
DFlash checkpoint is an initialization, never an image-warm checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure
configure()

import argparse
import hashlib
import json
import math
import os
import random
import time
import uuid

import numpy as np
import torch

from shared import common, sft_blocks, training_resume
from shared.master_optim import MasterWeightAdamW, master_change


def sha(path):
    return training_resume.file_hash(path)


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def save_checkpoint(out, step, head, opt, identity, rows_seen):
    root = Path(out)
    final = root / f'step{step}'
    if final.exists():
        raise FileExistsError(f'checkpoint already exists: {final}')
    tmp = root / f'.step{step}.tmp-{uuid.uuid4().hex}'
    tmp.mkdir(parents=True)
    try:
        head.save_pretrained(tmp)
        state = {
            **training_resume.extras(step, identity),
            'head': {n: p.detach().cpu() for n, p in head.state_dict().items()},
            'master_optimizer': opt.state_dict(),
            'update': step, 'data_cursor': step,
            'rows_seen': rows_seen, 'phase': 'SCRATCH_IMAGE_WARM',
        }
        torch.save(state, tmp / 'training_state.pt')
        manifest = {p.name: sha(p) for p in tmp.iterdir() if p.is_file()}
        atomic_json(tmp / 'SHA256.json', manifest)
        os.rename(tmp, final)
        return final, manifest
    except Exception:
        # Leave the incomplete temporary directory for diagnosis. It is never
        # interpreted as a checkpoint and no existing checkpoint is replaced.
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--data', required=True, help='frozen combined JSONL')
    ap.add_argument('--init-from', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--updates', type=int, default=2000)
    ap.add_argument('--stop-at', type=int, default=2000)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=common.LR)
    ap.add_argument('--save-every', type=int, default=250)
    ap.add_argument('--resume-from')
    ap.add_argument('--deadline-epoch', type=float, default=0)
    ap.add_argument('--master-weights', choices=['cpu', 'gpu'], default='cpu')
    a = ap.parse_args()
    training_resume.validate_output(a.out)
    if a.stop_at > a.updates or a.stop_at < 1:
        raise ValueError('invalid stop-at')
    if not Path(a.data).is_file() or not Path(a.init_from, 'model.safetensors').is_file():
        raise FileNotFoundError('data or initialization is absent')
    Path(a.out).mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.cuda.set_device(0)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    dev = 'cuda:0'
    rows = common.read_jsonl(a.data)
    if len(rows) != 16000 or any(len(r['gen_ids']) < 2 for r in rows):
        raise ValueError('expected 16000 valid original warm rows')
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(a.seed)).tolist()
    identity = {
        'data_sha256': sha(a.data),
        'init_sha256': sha(Path(a.init_from) / 'model.safetensors'),
        'target_config_sha256': sha(Path(common.TARGET_PATH) / 'config.json'),
        'warm_source_sha256': sha(__file__),
        'sft_blocks_sha256': sha(sft_blocks.__file__),
        'config': {'seed': a.seed, 'accum': a.accum, 'lr': a.lr,
                   'updates': a.updates, 'master_weights': a.master_weights,
                   'block_size': common.BLOCK_SIZE, 'max_seq_len': common.MAX_SEQ_LEN,
                   'long_side': common.IMAGE_LONG_SIDE,
                   'objective': 'coverage_ce_variable_reveal'},
    }
    target, proc = common.load_target(device=dev, dtype=torch.bfloat16)
    head = common.load_head(a.init_from, device=dev, dtype=torch.bfloat16)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    embed = target.get_input_embeddings().weight
    opt = MasterWeightAdamW(head, lr=a.lr, betas=common.BETAS,
                            weight_decay=common.WEIGHT_DECAY,
                            where=a.master_weights, clip=common.GRAD_CLIP)
    start = 0
    if a.resume_from:
        ck = Path(a.resume_from)
        manifest = json.loads((ck / 'SHA256.json').read_text())
        if any(sha(ck / name) != digest for name, digest in manifest.items()):
            raise ValueError('checkpoint SHA mismatch')
        state = torch.load(ck / 'training_state.pt', weights_only=False,
                           map_location='cpu')
        start = training_resume.restore(state, head, opt, identity)
        if state['rows_seen'] != start * a.accum:
            raise ValueError('warm cursor inconsistent with accumulation')
        print(f'[resume] verified step {start}: {ck}', flush=True)
    else:
        if any(Path(a.out).iterdir()):
            raise ValueError('new warm output is not empty')
        with torch.no_grad():
            row = rows[order[0]]
            image = common.load_image(row['image'])
            inp, pixel, grid = common.encode(proc, row['prompt'], image)
            full = torch.cat((inp[0], torch.tensor(row['gen_ids'])))[:common.MAX_SEQ_LEN].to(dev)
            if inp[0].tolist() != row['input_ids']:
                raise ValueError('frozen target prompt IDs differ from current processor')
            ctx = common.target_context(target, full[None], pixel.to(dev), grid.to(dev))
            diff, scale, cosine = sft_blocks.verify(head, target, ctx, full,
                                                      common.MASK_ID, common.BLOCK_SIZE, dev)
            if cosine < 0.999:
                raise RuntimeError(f'head geometry verification failed: {cosine}')
            print(f'[verify] image fused ctx={tuple(ctx.shape)} cosine={cosine:.8f} '
                  f'diff={diff:.4g} scale={scale:.4g}', flush=True)
        del ctx, full, pixel, grid
    log = open(Path(a.run) / 'warm/train_logs/updates.jsonl', 'a', buffering=1)
    run_start = time.time()
    last_completed = start
    last_checkpoint = str(Path(a.resume_from)) if a.resume_from else None
    for step in range(start, a.stop_at):
        if a.deadline_epoch and time.time() > a.deadline_epoch - 900:
            print('[deadline] leaving 15 minutes to save a complete checkpoint', flush=True)
            break
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        n_anchors = 0
        n_positions = 0
        t0 = time.time()
        for j in range(a.accum):
            position = order[step * a.accum + j]
            row = rows[position]
            image = common.load_image(row['image'])
            inp, pixel, grid = common.encode(proc, row['prompt'], image)
            if inp[0].tolist() != row['input_ids']:
                raise ValueError(f'processor drift at row {position}')
            full = torch.cat((inp[0], torch.tensor(row['gen_ids'])))[:common.MAX_SEQ_LEN].to(dev)
            prompt_len = int(inp.shape[1])
            ctx = common.target_context(target, full[None], pixel.to(dev), grid.to(dev))
            anchors = list(range(prompt_len, len(full) - 1))
            reveals = sft_blocks.sample_reveals(
                len(anchors), common.BLOCK_SIZE,
                torch.Generator().manual_seed(a.seed + position), dev)
            anchors, reveals = sft_blocks.live_anchors(
                anchors, reveals, len(full), common.BLOCK_SIZE)
            if not anchors:
                anchors = list(range(prompt_len, len(full) - 1))
                reveals = torch.ones(len(anchors), dtype=torch.long, device=dev)
                anchors, reveals = sft_blocks.live_anchors(
                    anchors, reveals, len(full), common.BLOCK_SIZE)
            if not anchors:
                raise ValueError(f'no supervision at row {position}')
            loss, correct, weights, by_offset = sft_blocks.sequence_loss(
                head, target, ctx, full, anchors, reveals, embed,
                common.MASK_ID, common.BLOCK_SIZE, dev)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite warm loss at row {position}')
            (loss / a.accum).backward()
            if any(p.grad is not None for p in target.parameters()):
                raise RuntimeError('frozen target received a gradient')
            loss_sum += float(loss.detach())
            n_anchors += len(anchors)
            n_positions += int(by_offset[:, 1].sum())
            del ctx, loss, correct, weights, by_offset
        opt.sync_grads()
        gn = opt.grad_norm()
        if not math.isfinite(gn):
            raise RuntimeError(f'nonfinite gradient at update {step + 1}')
        before = {n: p.detach().cpu().clone() for n, p in opt.master.items()} if step == 0 else None
        opt.step()
        if before is not None:
            delta = master_change(before, opt.master)
            if delta <= 0:
                raise RuntimeError('first update did not change fp32 master parameters')
            del before
        else:
            delta = None
        record = {'update': step + 1, 'loss': loss_sum / a.accum,
                  'grad_norm_preclip': gn, 'anchors': n_anchors,
                  'supervised_positions': n_positions, 'rows_seen': (step + 1) * a.accum,
                  'seconds': time.time() - t0, 'elapsed_seconds': time.time() - run_start,
                  'master_delta_first': delta}
        last_completed = step + 1
        log.write(json.dumps(record) + '\n')
        print(f"[warm] {step+1}/{a.updates} loss={record['loss']:.4f} "
              f"anchors={n_anchors} positions={n_positions} grad={gn:.3f} "
              f"sec={record['seconds']:.1f}", flush=True)
        atomic_json(Path(a.run) / 'RUN_STATUS.json',
                    {'stage': 'WARM_TRAINING', 'status': 'RUNNING',
                     'warm_updates': step + 1, 'sft_updates': 0, 'opd_updates': 0,
                     'last_heartbeat_epoch': time.time(), 'gpu': a.gpu,
                     'latest_checkpoint': last_checkpoint})
        if (step + 1) % a.save_every == 0 or step + 1 == a.stop_at:
            ck, hashes = save_checkpoint(a.out, step + 1, head, opt, identity,
                                         (step + 1) * a.accum)
            last_checkpoint = str(ck)
            print(f'[ckpt] {ck} verified {len(hashes)} files', flush=True)
            atomic_json(Path(a.run) / 'RUN_STATUS.json',
                        {'stage': 'WARM_TRAINING', 'status': 'RUNNING',
                         'warm_updates': step + 1, 'sft_updates': 0,
                         'opd_updates': 0, 'last_heartbeat_epoch': time.time(),
                         'gpu': a.gpu, 'latest_checkpoint': last_checkpoint})
    if last_completed > start and last_checkpoint != str(Path(a.out) / f'step{last_completed}'):
        ck, hashes = save_checkpoint(a.out, last_completed, head, opt, identity,
                                     last_completed * a.accum)
        last_checkpoint = str(ck)
        print(f'[ckpt] final partial-window {ck}', flush=True)
    atomic_json(Path(a.run) / 'RUN_STATUS.json',
                {'stage': 'WARM_TRAINING' if last_completed < a.updates else 'WARM_READY',
                 'status': 'PAUSED_WINDOW' if last_completed < a.updates else 'COMPLETE',
                 'warm_updates': last_completed, 'sft_updates': 0, 'opd_updates': 0,
                 'last_heartbeat_epoch': time.time(), 'gpu': a.gpu,
                 'latest_checkpoint': last_checkpoint})
    log.close()


if __name__ == '__main__':
    main()
