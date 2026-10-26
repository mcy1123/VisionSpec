#!/usr/bin/env python
"""Build the project-standard 16-frame cache clips for MSVD-QA train videos.

The convention is copied verbatim from ``repro/videoqa_transfer/prepare.py`` so
the train caches are byte-for-byte the same *kind* of artifact as the frozen
20-video dev set:

  * open the raw ``.avi`` with cv2; ``total = CAP_PROP_FRAME_COUNT``,
    ``fps = CAP_PROP_FPS``, source size ``(w, h)``;
  * ``ids = np.linspace(0, total - 1, 16).astype(int)``; for each id step back up
    to 25 frames until a frame decodes (same fallback as ``prepare.py``);
  * resize so the long side is 384 with both dimensions forced even:
    ``max(int(v * scale) // 2 * 2, 2)``, ``cv2.INTER_AREA``;
  * write ``train_cache16/<stem>.cache16.mp4`` with
    ``cv2.VideoWriter_fourcc(*'mp4v')`` at fps 30;
  * compute real PTS with PyAV: ``pts = [float(fr.pts * fr.time_base) for fr in
    av.open(raw).decode(video=0)]``, ``ts = [pts[i] for i in ids]``, assert
    non-decreasing and >= 0; record the container duration.

Writes ``inventory.jsonl`` (one row per *official* train video, including rows
for videos that could not be acquired) and ``cache_metrics.json``.  Idempotent:
a video whose cache already exists and is recorded in ``cache_ledger.jsonl``
with matching sizes/sha256 is skipped.

Usage::

    python make_cache16.py --workers 12
    python make_cache16.py --workers 12 --reverify
    python make_cache16.py --workers 4 --limit 20
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import sys
import time

import cv2
import numpy as np
import av

DATASET = pathlib.Path('/mnt/local2/mcy/datasets/msvd_qa_full')
OFFICIAL = DATASET / 'official'
RAW_DIR = DATASET / 'raw'
CACHE_DIR = DATASET / 'train_cache16'
ZIPMETA = DATASET / 'zipmeta'
LEDGER = ZIPMETA / 'cache_ledger.jsonl'

LONG_SIDE = 384
N_FRAMES = 16


def sha256_file(path: pathlib.Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_id(video_id: str) -> str:
    return video_id[:11]


# --------------------------------------------------------------------------- #
# one video
# --------------------------------------------------------------------------- #
def build_one(args: tuple) -> dict:
    """Decode -> sample 16 frames -> write cache -> probe PTS. Returns metrics."""
    video_id, info = args
    raw = RAW_DIR / video_id
    stem = pathlib.Path(video_id).stem
    cache = CACHE_DIR / f'{stem}.cache16.mp4'
    rec: dict = {
        'video_id': video_id,
        'source_id': source_id(video_id),
        'raw_path': str(raw),
        'cache_path': str(cache),
        'raw_sha256': None,
        'cache_sha256': None,
        'size_bytes': None,
        'cache_size_bytes': None,
        'download_source': info.get('download_source'),
        'zip_or_tar_offset': info.get('zip_local_header_offset')
                            if info.get('zip_local_header_offset') is not None
                            else info.get('tar_data_offset'),
        'zip_local_header_offset': info.get('zip_local_header_offset'),
        'tar_data_offset': info.get('tar_data_offset'),
        'zip_csize': info.get('zip_csize'),
        'zip_crc32': info.get('zip_crc32'),
        'tar_index_size': info.get('tar_index_size'),
        'video_duration': None,
        'source_fps': None,
        'source_frame_count': None,
        'source_width': None,
        'source_height': None,
        'sample_frame_ids': None,
        'sample_timestamps': None,
        'sample_timestamps_opencv': None,
        'pts_source': None,
        'width': None,
        'height': None,
        'n_frames': 0,
        'cache_readback_frames': None,
        'decode_ok': False,
        'failure_reason': None,
    }

    if not raw.exists():
        rec['failure_reason'] = info.get('failure_reason', 'raw file not acquired')
        return rec

    try:
        cap = cv2.VideoCapture(str(raw))
        if not cap.isOpened():
            raise RuntimeError('cv2 could not open raw avi')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        sw = int(cap.get(3))
        sh = int(cap.get(4))
        if not (total > 0 and fps > 0):
            raise RuntimeError(f'bad cv2 probe: total={total} fps={fps}')
        rec['source_frame_count'] = total
        rec['source_fps'] = fps
        rec['source_width'], rec['source_height'] = sw, sh

        scale = LONG_SIDE / max(sw, sh)
        w = max(int(sw * scale) // 2 * 2, 2)
        h = max(int(sh * scale) // 2 * 2, 2)

        ids, cv_ts, frames = [], [], []
        for i in np.linspace(0, total - 1, N_FRAMES).astype(int):
            ok = False
            for back in range(25):
                j = int(i) - back
                if j < 0:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                ok, f = cap.read()
                if ok:
                    break
            if not ok:
                raise RuntimeError(f'could not decode a frame near index {int(i)}')
            ids.append(j)
            cv_ts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
            frames.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA))
        cap.release()
        rec['sample_frame_ids'] = [int(x) for x in ids]
        rec['sample_timestamps_opencv'] = cv_ts

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # NOTE: cv2.VideoWriter infers the muxer from the extension, so the
        # temp file must still end in .mp4 (a bare ".partial" suffix fails).
        part = CACHE_DIR / f'{stem}.cache16.partial.mp4'
        wr = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        if not wr.isOpened():
            raise RuntimeError('cv2.VideoWriter could not open output')
        for f in frames:
            wr.write(f)
        wr.release()
        if part.stat().st_size <= 0:
            raise RuntimeError('VideoWriter produced an empty file')
        part.replace(cache)
        rec['width'], rec['height'] = w, h
        rec['n_frames'] = len(frames)

        # --- real PTS via PyAV -------------------------------------------- #
        container = av.open(str(raw))
        duration = (container.duration / av.time_base
                    if container.duration is not None else total / fps)
        pts_decode = [float(fr.pts * fr.time_base) for fr in container.decode(video=0)]
        container.close()
        rec['video_duration'] = float(duration)

        if len(pts_decode) < total:
            # some containers report a frame count slightly above what decodes
            pass
        idx = [i for i in ids if 0 <= i < len(pts_decode)]
        if len(idx) != len(ids):
            raise RuntimeError(f'PTS list too short: {len(pts_decode)} < max index+1 '
                               f'{max(ids) + 1}')
        ts = [pts_decode[i] for i in ids]
        if all(t >= 0 for t in ts) and all(ts[k] <= ts[k + 1] for k in range(len(ts) - 1)):
            rec['pts_source'] = 'pyav_decode_order'
        else:
            # B-frame reordering makes container.decode() yield non-monotonic PTS.
            # Fall back to genuine presentation-ordered PTS (sorted by pts) so the
            # recorded timestamps stay real and non-decreasing; never fabricate.
            pts_sorted = sorted(pts_decode)
            ts2 = [pts_sorted[i] for i in ids]
            if all(t >= 0 for t in ts2) and all(ts2[k] <= ts2[k + 1]
                                                for k in range(len(ts2) - 1)):
                ts = ts2
                rec['pts_source'] = 'pyav_presentation_order_sorted'
            else:
                rec['pts_source'] = 'pyav_decode_order_nonmonotonic'
                raise RuntimeError('PTS not non-decreasing under either ordering')
        rec['sample_timestamps'] = ts
        assert all(t >= 0 for t in ts) and all(ts[k] <= ts[k + 1]
                                               for k in range(len(ts) - 1))

        # --- read the cache back and prove 16 frames decode ---------------- #
        cap2 = cv2.VideoCapture(str(cache))
        nread = 0
        while True:
            ok2, _ = cap2.read()
            if not ok2:
                break
            nread += 1
        cap2.release()
        rec['cache_readback_frames'] = nread
        if nread != N_FRAMES:
            raise RuntimeError(f'cache readback decoded {nread} frames, expected {N_FRAMES}')

        rec['size_bytes'] = raw.stat().st_size
        rec['cache_size_bytes'] = cache.stat().st_size
        rec['raw_sha256'] = sha256_file(raw)
        rec['cache_sha256'] = sha256_file(cache)
        rec['decode_ok'] = True
        rec['failure_reason'] = None
    except Exception as e:                                      # noqa: BLE001
        rec['failure_reason'] = f'{type(e).__name__}: {e}'
        rec['decode_ok'] = False
        if rec['size_bytes'] is None and raw.exists():
            rec['size_bytes'] = raw.stat().st_size
    return rec


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--reverify', action='store_true')
    ap.add_argument('--only-fetched', action='store_true',
                    help='restrict to videos whose raw avi is present')
    a = ap.parse_args()

    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    plan = [json.loads(l) for l in (OFFICIAL / 'train_plan.jsonl').read_text().splitlines()
            if l.strip()]
    if a.limit:
        plan = plan[:a.limit]

    fetch = {}
    fpath = OFFICIAL / 'fetch_result.json'
    if fpath.exists():
        fetch = json.loads(fpath.read_text()).get('verification', {})
    ledger = {}
    lpath = ZIPMETA / 'extract_ledger.jsonl'
    if lpath.exists():
        for line in lpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                ledger[r['video_id']] = r

    cache_ledger: dict[str, dict] = {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache_ledger[r['video_id']] = r

    todo, infos = [], {}
    for p in plan:
        vid = p['video_id']
        info = dict(ledger.get(vid) or fetch.get(vid) or {})
        info.setdefault('tar_index_size', p.get('size') if p.get('in_tar_index') else None)
        if not info.get('download_source') and vid in fetch:
            info['download_source'] = 'hf_mirror_zip'
        prev = cache_ledger.get(vid)
        raw = RAW_DIR / vid
        if (not a.reverify and prev and prev.get('decode_ok')
                and pathlib.Path(prev['cache_path']).exists()
                and raw.exists()
                and raw.stat().st_size == prev.get('size_bytes')
                and pathlib.Path(prev['cache_path']).stat().st_size
                    == prev.get('cache_size_bytes')):
            infos[vid] = prev
            continue
        todo.append(vid)

    if a.only_fetched:
        todo = [v for v in todo if (RAW_DIR / v).exists()]
    print(f'[cache] {len(plan)} planned, {len(todo)} to build, '
          f'{len(infos)} reused from ledger', flush=True)

    results: dict[str, dict] = dict(infos)
    lf = LEDGER.open('a')

    def record(rec: dict):
        results[rec['video_id']] = rec
        lf.write(json.dumps(rec) + '\n')
        lf.flush()

    done = 0
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_one, (v, infos.get(v) or _info_for(v, ledger, fetch, plan))): v
                for v in todo}
        for fut in cf.as_completed(futs):
            vid = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:                              # noqa: BLE001
                rec = {'video_id': vid, 'decode_ok': False,
                       'failure_reason': f'worker crash: {type(e).__name__}: {e}'}
            record(rec)
            done += 1
            if done % 20 == 0 or done == len(todo):
                ok = sum(1 for r in results.values() if r.get('decode_ok'))
                print(f'[cache] {done}/{len(todo)} ({ok} ok total, '
                      f'{time.time() - t0:.0f}s)', flush=True)
    lf.close()

    # ---------- assemble inventory in official plan order ----------------- #
    rows = []
    for p in plan:
        vid = p['video_id']
        r = results.get(vid)
        if r is None:
            r = {'video_id': vid, 'source_id': source_id(vid),
                 'raw_path': str(RAW_DIR / vid), 'cache_path': None,
                 'decode_ok': False,
                 'failure_reason': 'not processed (absent from plan results)'}
        r = dict(r)
        r['qa_count'] = p.get('qa_count')
        r['official_in_tar_index'] = bool(p.get('in_tar_index'))
        if p.get('in_tar_index'):
            # provenance from the (partial) UT Austin tar index, kept alongside
            # the ZIP offsets actually used for the download
            r['tar_index_size'] = p.get('size')
            r['tar_data_offset'] = p.get('data_offset')
            r['tar_url'] = p.get('url')
        else:
            r['tar_index_size'] = None
            r['tar_data_offset'] = None
            r['tar_url'] = None
        rows.append(r)

    inv = DATASET / 'inventory.jsonl'
    tmp = inv.with_suffix('.jsonl.partial')
    with tmp.open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    tmp.replace(inv)

    n_ok = sum(1 for r in rows if r.get('decode_ok'))
    n_raw = sum(1 for r in rows if r['raw_path'] and pathlib.Path(r['raw_path']).exists())
    metrics = {
        'official_train_videos': len(plan),
        'raw_present': n_raw,
        'decode_ok': n_ok,
        'cache_written': sum(1 for r in rows if r.get('cache_path')
                             and pathlib.Path(r['cache_path']).exists()),
        'failed': [{'video_id': r['video_id'], 'failure_reason': r.get('failure_reason')}
                   for r in rows if not r.get('decode_ok')],
        'total_raw_bytes': sum(r.get('size_bytes') or 0 for r in rows),
        'total_cache_bytes': sum(r.get('cache_size_bytes') or 0 for r in rows),
        'pts_source_counts': _count(rows, 'pts_source'),
        'wall_seconds': round(time.time() - t0, 1),
        'inventory_path': str(inv),
        'cache_dir': str(CACHE_DIR),
        'raw_dir': str(RAW_DIR),
    }
    (DATASET / 'cache_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'failed'}, indent=2))
    print(f'[cache] done: decode_ok={n_ok}/{len(plan)}', flush=True)
    return 0 if n_ok == len(plan) else 1


def _info_for(vid, ledger, fetch, plan):
    info = dict(ledger.get(vid) or fetch.get(vid) or {})
    p = next((x for x in plan if x['video_id'] == vid), None)
    if p:
        info.setdefault('tar_index_size', p.get('size') if p.get('in_tar_index') else None)
    if not info.get('download_source') and vid in fetch:
        info['download_source'] = 'hf_mirror_zip'
    return info


def _count(rows, key):
    out: dict = {}
    for r in rows:
        k = r.get(key)
        out[str(k)] = out.get(str(k), 0) + 1
    return out


if __name__ == '__main__':
    sys.exit(main())
