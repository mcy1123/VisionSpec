"""Real-video strict-online gate. Backward only; never constructs an optimizer."""

# Direct script execution and historical workspace-relative path calculations.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_REPRO_ROOT = _BootstrapPath(__file__).resolve().parents[1]
_bootstrap_sys.path.insert(0, str(_REPRO_ROOT))
from _bootstrap import configure as _configure_paths, source_path as _source_path
_configure_paths()

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
import pandas as pd
from transformers import AutoProcessor, DynamicCache
from shared import common
from historical import opd_train_video_fullkl as T
from historical import opd_verified_rollout as V


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


@torch.no_grad()
def independent_ar(target, trace, count):
    ids = trace.full_ids[:trace.prompt_len][None]
    cache = DynamicCache()
    target.model.rope_deltas = None
    out = target(ids, pixel_values_videos=trace.pixel_values_videos,
                 video_grid_thw=trace.video_grid_thw, past_key_values=cache,
                 use_cache=True, logits_to_keep=1,
                 cache_position=torch.arange(ids.shape[1], device=ids.device))
    tokens = [int(out.logits[0, -1].argmax())]
    while len(tokens) < count and tokens[-1] not in V.eos_ids(target):
        out = target(ids.new_tensor([[tokens[-1]]]), past_key_values=cache,
                     use_cache=True, logits_to_keep=1,
                     cache_position=ids.new_tensor([cache.get_seq_length()]))
        tokens.append(int(out.logits[0, -1].argmax()))
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--head', default=f'{common.CKPT}/head_video439_aligned/step1000')
    a = ap.parse_args()
    dest = Path(a.out).resolve()
    if not any(dest.is_relative_to(Path(p)) for p in ('/mnt/local','/mnt/local2','/mnt/local3')):
        raise ValueError('gate artifacts must be under /mnt/local{,2,3}')
    dest.mkdir(parents=True, exist_ok=False)
    pool = pd.read_parquet(f'{common.DATA}/pool_499.parquet').set_index('video_name')
    records = torch.load(f'{common.DATA}/video_greedy_499.pt', weights_only=False, map_location='cpu')['records']
    chosen = sorted((r for r in records if pool.loc[str(r['task']), 'split'] == 'train'),
                    key=lambda r: str(r['task']))[:2]
    paths = list(Path(a.head).glob('*.safetensors')) + list(Path(a.head).glob('*.bin'))
    before = {str(p): sha(p) for p in paths}
    manifest = dict(tasks=[str(r['task']) for r in chosen], seed=0, gpu=a.gpu,
                    rounds=3, max_new_tokens=48, max_seq_len=2048,
                    target_dtype='float32', head_dtype='bfloat16', head_sha=before,
                    optimizer_steps=0, source_sha={str(p): sha(p) for p in
                    (Path(__file__), Path(T.__file__), Path(V.__file__))})
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2))
    free = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
        '--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'], text=True)
    (dest/'gpu_before.txt').write_text(free)
    used, available = map(int, free.strip().split(','))
    if used > 100 or available < 47000:
        raise RuntimeError(f'GPU {a.gpu} is not idle: {free}')
    torch.manual_seed(0)
    device = f'cuda:{a.gpu}'
    torch.cuda.set_device(a.gpu)
    target, _ = common.load_target(device=device, dtype=torch.float32)
    proc = AutoProcessor.from_pretrained(common.TARGET_PATH)
    head = common.load_head(a.head, device=device, dtype=torch.bfloat16).eval()
    for p in head.parameters():
        p.requires_grad_(True)
    versions = [p._version for p in head.parameters()]
    calls = dict(target=0, draft=0)
    def count_target(*unused):
        calls['target'] += 1
    def count_head(*unused):
        calls['draft'] += 1
    target.model.register_forward_pre_hook(count_target)
    head.register_forward_pre_hook(count_head)
    results = []
    for index, rec in enumerate(chosen):
        start = time.time()
        trace, anchors = T.build_video_trace(target, head, proc, rec, pool.loc[str(rec['task'])],
            device=device, max_seq_len=2048, max_rounds=3,
            max_new_tokens=48, head_dtype=torch.bfloat16)
        generated = trace.full_ids[trace.prompt_len:].tolist()
        ar = independent_ar(target, trace, len(generated))
        row = dict(task=trace.task, tokens=generated, independent_ar=ar,
                   lossless=generated == ar, anchors=anchors,
                   stop_reason=trace.stop_reason)
        (dest/f'video{index}_trace.json').write_text(json.dumps(row, indent=2))
        assert generated == ar, 'online committed sequence differs from independent AR'
        assert anchors, 'frozen smoke video produced no anchors'
        plans = [T.build_anchor_plan(trace.task, anchor, total=trace.full_ids.numel(),
                 valid_end=trace.valid_end, prompt_len=trace.prompt_len, device=device)
                 for anchor in anchors]
        denoms = T.plan_update_denominators(plans)
        row['losses'] = []
        for j, plan in enumerate(plans):
            loss, stats = T.score_microbatch(head, target, trace, plan, gamma=.8,
                denoms=denoms, head_dtype=torch.bfloat16, teacher_dtype=torch.float32)
            assert torch.isfinite(loss)
            row['losses'].append(float(loss.detach()))
            if j == 0:
                loss.backward()
                grads = [p.grad for p in head.parameters() if p.grad is not None]
                assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
                row['gradient_norm'] = sum(float(g.float().square().sum()) for g in grads)**.5
                assert row['gradient_norm'] > 0
                assert all(p.grad is None for p in target.parameters())
                head.zero_grad(set_to_none=True)
            del loss
        row['seconds'] = time.time() - start
        row['peak_gib'] = torch.cuda.max_memory_allocated(a.gpu)/2**30
        results.append(row)
        (dest/f'video{index}_result.json').write_text(json.dumps(row, indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('anchors','tokens','independent_ar')}), flush=True)
        del trace, plans
    assert versions == [p._version for p in head.parameters()], 'head parameters mutated'
    after = {str(p): sha(p) for p in paths}
    assert before == after
    summary = dict(status='passed', videos=len(results), anchors=sum(len(r['anchors']) for r in results),
                   lossless=sum(r['lossless'] for r in results), calls=calls,
                   peak_gib=torch.cuda.max_memory_allocated(a.gpu)/2**30,
                   optimizer_steps=0, checkpoint_sha_after=after)
    (dest/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
