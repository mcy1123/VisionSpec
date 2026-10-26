"""CPU-only, deterministic full eligible-QA freeze. No model outputs used."""
import collections
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from scienceqa_image.build_dataset import OFFICIAL_DIR, OFFICIAL_REV, build_prompt

WARM = '/mnt/local3/mcy/GLANCE/results/sqa_scratch_warm_officialkl_20261016_232843/warm/checkpoints/step2000'
WARM_SHA = 'bd16ebcff1c513d42b14d9b8d89d428d93973ff8620bb78eafa9689509616735'
OLD = Path('/mnt/local3/mcy/GLANCE/results/scienceqa_image_reasoning_20261015T1955')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''): h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def mounted(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(Path(m)) for m in ['/mnt/local', '/mnt/local2', '/mnt/local3']):
        raise ValueError(f'large outputs must be on /mnt/local*: {path}')
    return path


def freeze(run, seed=20261013):
    run = mounted(run)
    if run.exists(): raise FileExistsError(run)
    run.mkdir(parents=True)
    assert sha(Path(WARM) / 'model.safetensors') == WARM_SHA, 'Warm identity changed'
    data = run / 'dataset'; data.mkdir()
    rows, splits, sources = [], collections.defaultdict(set), {}
    for split in ['train', 'validation', 'test']:
        path = Path(OFFICIAL_DIR) / 'ScienceQA-IMG' / f'{split}-00000-of-00001.parquet'
        sources[str(path)] = sha(path)
        idx = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for raw in batch.to_pylist():
                blob = raw.pop('image')['bytes']
                h = hashlib.sha256(blob).hexdigest()
                with Image.open(io.BytesIO(blob)) as im:
                    im = im.convert('RGB'); im.load()
                    pixel = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                canonical = 'dev' if split == 'validation' else split
                rows.append(dict(raw, pid=f'sqa-img:{canonical}:{idx}', official_row_index=idx,
                                 official_split=split, canonical_split=canonical,
                                 image_sha256=h, pixel_sha256=pixel, image_bytes=blob))
                splits[pixel].add(split); idx += 1
    cross = {h for h, s in splits.items() if len(s) > 1}
    clean = [r for r in rows if r['pixel_sha256'] not in cross]
    train, duplicates, seen = [], [], set()
    for r in clean:
        if r['official_split'] != 'train': continue
        key = (r['pixel_sha256'], r['question'].strip(), tuple(r['choices']), r['answer'])
        if key in seen: duplicates.append(r['pid']); continue
        seen.add(key); train.append(r)
    # Inventory all discoverable prior ScienceQA manifests/evaluation CSVs on the three volumes.
    roots = [f'/mnt/{m}/mcy/GLANCE/results' for m in ['local','local2','local3']]
    found = subprocess.run(['rg', '--files', '--hidden', '-g', '*manifest*.csv',
                            '-g', '*per_question.csv', *roots], capture_output=True, text=True)
    if found.returncode not in [0, 1]: raise RuntimeError(found.stderr)
    history, used_groups, used_hashes = {}, set(), set()
    for name in sorted(set(found.stdout.splitlines())):
        p = Path(name).resolve()
        if str(p) in history or p.is_relative_to(run): continue
        with p.open() as f:
            records = list(csv.DictReader(f))
        relevant = [r for r in records if str(r.get('pid', r.get('task', ''))).startswith('sqa-img:')]
        if relevant:
            history[str(p)] = sha(p)
            used_groups.update(r.get('image_group_id', '') for r in relevant)
            used_hashes.update(r.get('image_sha256', '') for r in relevant)
    if not history: raise RuntimeError('no historical inventory discovered')
    used_pixels = {r['pixel_sha256'] for r in rows if r['image_sha256'] in used_hashes
                   or f"img:{r['image_sha256'][:16]}" in used_groups}
    old_dev = list(csv.DictReader((OLD/'dataset/records_manifest.csv').open()))
    dev_pids = {r['pid'] for r in old_dev if r['canonical_split'] == 'dev'}
    dev, dev_seen = [], set()
    for r in clean:
        if r['pid'] in dev_pids and r['pixel_sha256'] not in dev_seen:
            dev.append(r); dev_seen.add(r['pixel_sha256'])
    dev = sorted(dev, key=lambda r: r['official_row_index'])[:32]
    candidates = sorted([r for r in clean if r['official_split'] == 'test'
                         and r['pixel_sha256'] not in used_pixels],
                        key=lambda r: hashlib.sha256(f"confirm-20261019:{r['pid']}".encode()).hexdigest())
    confirm, seen = [], set()
    for r in candidates:
        if r['pixel_sha256'] not in seen:
            confirm.append(r); seen.add(r['pixel_sha256'])
        if len(confirm) == 200: break
    if len(dev) != 32 or len(confirm) != 200:
        raise ValueError(f'insufficient dev/unused confirm images: {len(dev)}/{len(confirm)}')
    train.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['pid']}".encode()).hexdigest())
    images = data / 'images'; images.mkdir()
    final = []
    for r in train + dev + confirm:
        path = images / (r['image_sha256'] + '.img')
        if not path.exists(): path.write_bytes(r['image_bytes'])
        prompt = build_prompt(r['question'], r['choices'])
        final.append(dict(pid=r['pid'], task=r['pid'], video_id=r['pid'], question_id=r['pid'],
            official_row_index=r['official_row_index'], official_split=r['official_split'],
            canonical_split=r['canonical_split'], image_path=str(path), image_sha256=r['image_sha256'],
            pixel_sha256=r['pixel_sha256'], image_group_id=f"img:{r['pixel_sha256']}",
            prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            subject=r['subject'], grade=r['grade'], n_choices=len(r['choices']),
            question=r['question'], choices=json.dumps(r['choices']), gold_letter='ABCDE'[r['answer']]))
    with (data/'records_manifest.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(final[0])); w.writeheader(); w.writerows(final)
    pd.DataFrame(final[:len(train)]).to_parquet(data/'pool.parquet', index=False)
    with (data/'schedule.jsonl').open('w') as f:
        for i,r in enumerate(final[:len(train)]):
            f.write(json.dumps(dict(order=i, update=i//4, slot=i%4, task=r['pid'],
                                   video_id=r['pid'], question_id=r['pid'],
                                   prompt_sha256=r['prompt_sha256']))+'\n')
    audit = dict(source_revision=OFFICIAL_REV, source_hashes=sources,
        official_counts=dict(collections.Counter(r['official_split'] for r in rows)),
        cross_split_decoded_image_groups=len(cross), excluded_exact_duplicate_qa=duplicates,
        train_qa=len(train), train_images=len({r['pixel_sha256'] for r in train}),
        updates=(len(train)+3)//4, last_batch=len(train)%4 or 4,
        dev_images=32, confirm_images=200, historical_inventory=history,
        confirm_policy='excluded every discovered historical image, selected without model outputs',
        limits='exact bytes and decoded-pixel dedup; not a semantic near-image or foundation-model pretraining audit',
        forbidden_prompt_fields=['answer','lecture','solution','hint'])
    write_json(run/'DATA_AUDIT.json', audit)
    write_json(run/'INPUTS.json', {str(p): sha(p) for p in [data/'records_manifest.csv',data/'pool.parquet',data/'schedule.jsonl']})
    return audit
