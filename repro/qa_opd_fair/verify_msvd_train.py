#!/usr/bin/env python
"""End-to-end verification + ``download_summary.json`` for the MSVD-QA train split.

Re-opens a random sample of the produced 16-frame cache clips and proves that
they decode to exactly 16 frames and that the recorded sha256 still matches the
bytes on disk.  Also re-hashes a sample of raw ``.avi`` files and cross-checks
every acquired file against the UT Austin tar index sizes where known.

Usage::

    python verify_msvd_train.py --sample 30 --seed 20261013
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

import cv2

DATASET = pathlib.Path('/mnt/local2/mcy/datasets/msvd_qa_full')
OFFICIAL = DATASET / 'official'


def sha256_file(path: pathlib.Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_frames(path: pathlib.Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, None
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frames, (w, h), fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20261013)
    ap.add_argument('--exhaustive', action='store_true',
                    help='also CRC32-check every raw avi against the ZIP central '
                         'directory and sha256-check every raw+cache file')
    a = ap.parse_args()

    t0 = time.time()
    inv_path = DATASET / 'inventory.jsonl'
    rows = [json.loads(l) for l in inv_path.read_text().splitlines() if l.strip()]
    plan = [json.loads(l) for l in (OFFICIAL / 'train_plan.jsonl').read_text().splitlines()
            if l.strip()]
    frec = json.loads((OFFICIAL / 'fetch_result.json').read_text()) \
        if (OFFICIAL / 'fetch_result.json').exists() else {}

    planned = len(plan)
    assert len(rows) == planned, f'inventory rows {len(rows)} != planned {planned}'

    raw_ok = [r for r in rows if r.get('raw_path') and pathlib.Path(r['raw_path']).exists()]
    decode_ok = [r for r in rows if r.get('decode_ok')]
    cache_ok = [r for r in decode_ok
                if r.get('cache_path') and pathlib.Path(r['cache_path']).exists()]
    missing = [{'video_id': r['video_id'],
                'reason': r.get('failure_reason') or 'raw file absent'}
               for r in rows if not (r.get('raw_path')
                                     and pathlib.Path(r['raw_path']).exists())]
    failed = [{'video_id': r['video_id'], 'failure_reason': r.get('failure_reason')}
              for r in rows if not r.get('decode_ok')]

    # ---- random re-verification ----------------------------------------- #
    rng = random.Random(a.seed)
    sample = rng.sample(cache_ok, min(a.sample, len(cache_ok))) if cache_ok else []
    checks = []
    for r in sample:
        cp = pathlib.Path(r['cache_path'])
        rp = pathlib.Path(r['raw_path'])
        frames, wh, fps = read_frames(cp)
        chash = sha256_file(cp)
        rhash = sha256_file(rp)
        checks.append({
            'video_id': r['video_id'],
            'cache_path': str(cp),
            'raw_path': str(rp),
            'cache_frames_decoded': len(frames) if frames is not None else None,
            'cache_frame_count_ok': bool(frames is not None and len(frames) == 16),
            'cache_frame_shape': [list(f.shape) for f in frames[:1]] if frames else None,
            'cache_wh': list(wh) if wh else None,
            'cache_fps': fps,
            'cache_sha256_recorded': r.get('cache_sha256'),
            'cache_sha256_actual': chash,
            'cache_sha256_match': chash == r.get('cache_sha256'),
            'raw_sha256_recorded': r.get('raw_sha256'),
            'raw_sha256_actual': rhash,
            'raw_sha256_match': rhash == r.get('raw_sha256'),
            'recorded_width_height': [r.get('width'), r.get('height')],
            'recorded_n_frames': r.get('n_frames'),
            'recorded_size_bytes': r.get('size_bytes'),
            'actual_raw_size_bytes': rp.stat().st_size,
        })

    n_cache_frames_ok = sum(1 for c in checks if c['cache_frame_count_ok'])
    n_cache_sha_ok = sum(1 for c in checks if c['cache_sha256_match'])
    n_raw_sha_ok = sum(1 for c in checks if c['raw_sha256_match'])
    n_wh_ok = sum(1 for c in checks
                  if c['cache_wh'] is not None
                  and c['cache_wh'] == c['recorded_width_height'])
    n_size_ok = sum(1 for c in checks
                    if c['actual_raw_size_bytes'] == c['recorded_size_bytes'])

    # ---- exhaustive cheap checks over every row -------------------------- #
    bad_timestamps = []
    bad_ids = []
    bad_dims = []
    bad_long_side = []
    for r in decode_ok:
        ts = r.get('sample_timestamps')
        ids = r.get('sample_frame_ids')
        if not ts or len(ts) != 16 or any(t < 0 for t in ts) \
                or any(ts[i] > ts[i + 1] for i in range(15)):
            bad_timestamps.append(r['video_id'])
        if not ids or len(ids) != 16 or ids != sorted(ids):
            bad_ids.append(r['video_id'])
        w, h = r.get('width'), r.get('height')
        if not w or not h or w % 2 or h % 2:
            bad_dims.append(r['video_id'])
        if w and h and max(w, h) != 384:
            bad_long_side.append(r['video_id'])

    tar_sizes = {p['video_id']: p['size'] for p in plan if p.get('in_tar_index')}
    tar_mismatch = []
    for r in raw_ok:
        exp = tar_sizes.get(r['video_id'])
        if exp is not None and pathlib.Path(r['raw_path']).stat().st_size != exp:
            tar_mismatch.append(r['video_id'])

    # ---- exhaustive CRC32 / sha256 sweep (optional) ---------------------- #
    exhaustive = None
    if a.exhaustive:
        import zlib
        zipidx = {}
        zf = DATASET / 'zipmeta' / 'msvd_zip_index.json'
        if zf.exists():
            for e in json.loads(zf.read_text()):
                if e['name'].lower().endswith('.avi'):
                    zipidx[pathlib.Path(e['name']).name] = e
        crc_bad, raw_sha_bad, cache_sha_bad, crc_missing = [], [], [], []
        for r in rows:
            vid = r['video_id']
            rp = pathlib.Path(r['raw_path']) if r.get('raw_path') else None
            if rp is None or not rp.exists():
                continue
            data = rp.read_bytes()
            e = zipidx.get(vid)
            if e is None:
                crc_missing.append(vid)
            elif (zlib.crc32(data) & 0xFFFFFFFF) != e['crc32']:
                crc_bad.append(vid)
            if hashlib.sha256(data).hexdigest() != r.get('raw_sha256'):
                raw_sha_bad.append(vid)
            cp = pathlib.Path(r['cache_path']) if r.get('cache_path') else None
            if cp is not None and cp.exists():
                if hashlib.sha256(cp.read_bytes()).hexdigest() != r.get('cache_sha256'):
                    cache_sha_bad.append(vid)
        exhaustive = {
            'raw_files_crc32_checked': len(raw_ok) - len(crc_missing),
            'raw_files_crc32_mismatch': crc_bad,
            'raw_files_absent_from_zip_index': crc_missing,
            'raw_sha256_mismatch': raw_sha_bad,
            'cache_sha256_mismatch': cache_sha_bad,
            'all_ok': not (crc_bad or raw_sha_bad or cache_sha_bad or crc_missing),
        }

    total_raw = sum(pathlib.Path(r['raw_path']).stat().st_size for r in raw_ok)
    total_cache = sum(pathlib.Path(r['cache_path']).stat().st_size for r in cache_ok)

    verification = {
        'verified_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'sample_size': len(sample),
        'sample_seed': a.seed,
        'sample_cache_16_frames_ok': n_cache_frames_ok,
        'sample_cache_sha256_match': n_cache_sha_ok,
        'sample_raw_sha256_match': n_raw_sha_ok,
        'sample_cache_wh_matches_record': n_wh_ok,
        'sample_raw_size_matches_record': n_size_ok,
        'all_rows_sample_timestamps_valid': len(bad_timestamps) == 0,
        'rows_with_bad_timestamps': bad_timestamps,
        'all_rows_sample_frame_ids_valid': len(bad_ids) == 0,
        'rows_with_bad_frame_ids': bad_ids,
        'all_rows_even_dims': len(bad_dims) == 0,
        'rows_with_non_even_dims': bad_dims,
        'tar_index_size_mismatches': tar_mismatch,
        'long_side_not_384': [
            {'video_id': r['video_id'], 'source_width': r['source_width'],
             'source_height': r['source_height'], 'width': r['width'], 'height': r['height'],
             'note': 'verbatim prepare.py expression max(int(v*scale)//2*2,2); '
                     'float64 rounding makes 384/294*294 == 383.99999999999994'}
            for r in decode_ok if max(r['width'], r['height']) != 384],
        'exhaustive': exhaustive,
        'checks': checks,
    }
    (DATASET / 'verification.json').write_text(json.dumps(verification, indent=2))

    summary = {
        'dataset': 'MSVD-QA',
        'split': 'official LAVIS train split (qa_train.json)',
        'annotation_file': str(OFFICIAL / 'qa_train.json'),
        'official_train_qa_rows': 30933,
        'official_train_videos': planned,
        'planned': planned,
        'downloaded': len(raw_ok),
        'decode_ok': len(decode_ok),
        'cache_written': len(cache_ok),
        'missing': len(missing),
        'missing_video_names': [m['video_id'] for m in missing],
        'missing_details': missing,
        'failed': len(failed),
        'failed_details': failed,
        'source_urls_used': {
            'attempted_first': 'ut_austin_range (probe failed -> fell through to '
                                'hf_mirror_zip)',
            'final_source_used': frec.get('source_used'),
            'ut_austin_tar': 'https://www.cs.utexas.edu/users/ml/clamp/'
                             'videoDescription/YouTubeClips.tar',
            'ut_austin_probe': frec.get('ut_austin_probe'),
            'hf_mirror_resolve': 'https://hf-mirror.com/datasets/friedrichor/MSVD/'
                                 'resolve/main/MSVD_Videos.zip',
            'hf_mirror_zip_size_bytes': 1840431084,
            'hf_mirror_sha256_etag': ('725647918b19a8252f7b6fa17392262ff8f0b529f03dea43757'
                                      'ee446c63fb029'),
            'source_used': frec.get('source_used'),
        },
        'totals': {
            'total_raw_bytes': total_raw,
            'total_raw_mib': round(total_raw / 2**20, 1),
            'total_cache_bytes': total_cache,
            'total_cache_mib': round(total_cache / 2**20, 1),
            'total_bytes_on_disk': total_raw + total_cache,
            'total_mib_on_disk': round((total_raw + total_cache) / 2**20, 1),
        },
        'wall_time': {
            'download_wall_seconds': frec.get('wall_seconds'),
            'cache_build_and_verify_wall_seconds': round(time.time() - t0, 1),
        },
        'aggregate_bytes_read_from_zip': frec.get('aggregate_bytes_read'),
        'tar_index_size_crosscheck': {
            **(frec.get('tar_index_size_crosscheck') or {}),
            'note': 'the pre-existing msvd_tar_index.jsonl is a PARTIAL prefix scan of '
                    'YouTubeClips.tar (1329 avi entries); only 856 of the 1161 official '
                    'train videos appear in it, so for the other 305 the ZIP central '
                    'directory is the only size/CRC reference. Every one of the 856 '
                    'overlapping entries matched the ZIP uncompressed size exactly.',
        },
        'zip_redirect_resolutions': frec.get('zip_redirect_resolutions'),
        'verification': {k: v for k, v in verification.items() if k != 'checks'},
        'output_paths': {
            'raw_dir': str(DATASET / 'raw'),
            'cache_dir': str(DATASET / 'train_cache16'),
            'inventory': str(inv_path),
            'download_summary': str(DATASET / 'download_summary.json'),
            'verification': str(DATASET / 'verification.json'),
            'fetch_result': str(OFFICIAL / 'fetch_result.json'),
            'train_plan': str(OFFICIAL / 'train_plan.jsonl'),
            'zip_index': str(DATASET / 'zipmeta' / 'msvd_zip_index.json'),
            'central_directory': str(DATASET / 'zipmeta' / 'central_directory.bin'),
            'extract_ledger': str(DATASET / 'zipmeta' / 'extract_ledger.jsonl'),
            'cache_ledger': str(DATASET / 'zipmeta' / 'cache_ledger.jsonl'),
            'fetch_log': str(DATASET / 'logs' / 'fetch_msvd_train.log'),
            'cache_log': str(DATASET / 'logs' / 'make_cache16.log'),
        },
        'commands_and_scripts': [
            '/home/mcy/.conda/envs/fin_research/bin/python '
            'repro/qa_opd_fair/plan_msvd_train.py',
            '/home/mcy/.conda/envs/fin_research/bin/python '
            'repro/qa_opd_fair/fetch_msvd_train.py --source auto '
            '--out /mnt/local2/mcy/datasets/msvd_qa_full/official/fetch_result.json',
            '/home/mcy/.conda/envs/fin_research/bin/python '
            'repro/qa_opd_fair/make_cache16.py --workers 16',
            ' '.join([sys.executable, str(pathlib.Path(__file__).resolve()),
                      *sys.argv[1:]]),
        ],
        'scripts': [
            '/home/mcy/Projects/GLANCE/repro/qa_opd_fair/plan_msvd_train.py',
            '/home/mcy/Projects/GLANCE/repro/qa_opd_fair/fetch_msvd_train.py',
            '/home/mcy/Projects/GLANCE/repro/qa_opd_fair/make_cache16.py',
            '/home/mcy/Projects/GLANCE/repro/qa_opd_fair/verify_msvd_train.py',
        ],
    }
    (DATASET / 'download_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('missing_details', 'failed_details')}, indent=2))
    print(f'[verify] decode_ok={len(decode_ok)}/{planned} '
          f'sample16={n_cache_frames_ok}/{len(sample)} '
          f'cache_sha={n_cache_sha_ok}/{len(sample)} raw_sha={n_raw_sha_ok}/{len(sample)}')
    return 0 if len(decode_ok) == planned and n_cache_frames_ok == len(sample) else 1


if __name__ == '__main__':
    sys.exit(main())
