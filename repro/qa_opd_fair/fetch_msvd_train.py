#!/usr/bin/env python
"""Fetch the official MSVD-QA *train* split videos (1161 unique clips).

Source strategy (recorded per file in the inventory):

  a) ``ut_austin_range``  -- HTTP Range reads against the UT Austin
     ``YouTubeClips.tar`` using a 512-byte-range tar index.  Tried first, with a
     short bounded retry (3 attempts x 10 s).  In this environment
     ``www.cs.utexas.edu`` is unreachable (connection timeout), so this source
     is abandoned gracefully and the run falls through to (b).
  b) ``hf_mirror_zip``    -- the HuggingFace mirror dataset
     ``friedrichor/MSVD`` -> ``MSVD_Videos.zip`` (1,840,431,084 bytes).  Rather
     than downloading the whole archive we fetch the ZIP End-Of-Central-Directory
     record and the central directory with a suffix Range request, then pull only
     the members we need with precise Range reads.  Every extracted member is
     checked for exact byte size, ZIP CRC32, and cv2 openability later on.

The run is idempotent: a member already extracted and verified (recorded in the
ledger) is skipped.  Aggregate throughput is capped with a global token bucket
and at most 8 concurrent connections are used.

Nothing outside ``/mnt/local2/mcy/datasets/msvd_qa_full/`` is written.

Usage::

    python fetch_msvd_train.py --source auto          # probe (a) then use (b)
    python fetch_msvd_train.py --source hf_mirror_zip
    python fetch_msvd_train.py --source ut_austin_range
    python fetch_msvd_train.py --limit 5              # smoke subset
    python fetch_msvd_train.py --reverify             # ignore ledger, re-check all
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import struct
import sys
import threading
import time
import zlib

import requests

# --------------------------------------------------------------------------- #
# constants / paths
# --------------------------------------------------------------------------- #
DATASET = pathlib.Path('/mnt/local2/mcy/datasets/msvd_qa_full')
OFFICIAL = DATASET / 'official'
RAW_DIR = DATASET / 'raw'
ZIPMETA = DATASET / 'zipmeta'
LOGS = DATASET / 'logs'

TAR_INDEX = pathlib.Path(
    '/mnt/local/mcy/GLANCE/results/videoqa_transfer/20261013T134000/sources/msvd_tar_index.jsonl')
TAR_URL = 'https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar'

HF_RESOLVE = ('https://hf-mirror.com/datasets/friedrichor/MSVD/'
              'resolve/main/MSVD_Videos.zip')
HF_ZIP_SIZE = 1840431084

MAX_WORKERS = 8
AGG_BYTES_PER_SEC = 3.8 * 1024 * 1024   # stay under the ~4 MB/s aggregate cap
CHUNK = 1 << 16
LOCAL_HDR_OVERREAD = 4096               # slack for the local extra field
HTTP_ATTEMPTS = 4

_user_agent = {'User-Agent': 'GLANCE-msvd-fetch/1.0'}


# --------------------------------------------------------------------------- #
# global bandwidth limiter
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Global token bucket; ``consume(n)`` blocks until ``n`` bytes may be read."""

    def __init__(self, bytes_per_sec: float, burst: float | None = None):
        self.rate = float(bytes_per_sec)
        self.cap = float(burst if burst is not None else bytes_per_sec)
        self.lock = threading.Lock()
        self.allowance = self.cap
        self.last = time.monotonic()
        self.total = 0

    def consume(self, n: int) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.allowance = min(self.cap,
                                     self.allowance + (now - self.last) * self.rate)
                self.last = now
                if self.allowance >= n:
                    self.allowance -= n
                    self.total += n
                    return
                need = (n - self.allowance) / self.rate
            time.sleep(min(max(need, 0.001), 0.25))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: pathlib.Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_plan() -> list[dict]:
    p = OFFICIAL / 'train_plan.jsonl'
    if not p.exists():
        raise SystemExit(f'missing {p}; run plan_msvd_train.py first')
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_tar_index() -> dict:
    idx = {}
    for line in TAR_INDEX.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n = pathlib.Path(r['name']).name
        if r.get('type') == '5' or not n:
            continue
        idx[n] = r
    return idx


# --------------------------------------------------------------------------- #
# (a) UT Austin tar via range requests -- probed, expected to fail here
# --------------------------------------------------------------------------- #
def ut_austin_reachable(attempts: int = 3, timeout: float = 10.0) -> tuple[bool, str]:
    last = ''
    for i in range(attempts):
        try:
            r = requests.get(TAR_URL, headers={**_user_agent, 'Range': 'bytes=0-1'},
                             timeout=timeout, stream=True)
            ok = r.status_code in (200, 206)
            last = f'attempt {i + 1}: HTTP {r.status_code}'
            r.close()
            if ok:
                return True, last
        except Exception as e:                                  # noqa: BLE001
            last = f'attempt {i + 1}: {type(e).__name__}: {e}'
    return False, last


# --------------------------------------------------------------------------- #
# (b) HF mirror ZIP: central directory + selective member extraction
# --------------------------------------------------------------------------- #
class ZipSource:
    """Range reader for the HF-mirror ZIP, with a cached signed redirect URL."""

    def __init__(self, limiter: RateLimiter, workers: int = MAX_WORKERS):
        self.limiter = limiter
        self.workers = workers
        self._direct: str | None = None
        self._direct_at = 0.0
        self._lock = threading.Lock()
        self.session = requests.Session()
        self.resolutions = 0

    # -- redirect handling ------------------------------------------------ #
    def direct_url(self, force: bool = False) -> str:
        with self._lock:
            fresh = (self._direct is not None and time.monotonic() - self._direct_at < 1200)
            if fresh and not force:
                return self._direct
        last: Exception | None = None
        for attempt in range(6):
            try:
                # The signed redirect target is valid for 1 h; caching it for
                # 20 min avoids hammering the (flaky) mirror front end.
                r = requests.get(HF_RESOLVE,
                                 headers={**_user_agent, 'Range': 'bytes=0-0'},
                                 timeout=45, allow_redirects=True)
                r.raise_for_status()
                if 'xethub' not in r.url and 'hf.co' not in r.url:
                    # not a CAS redirect: make sure we can still range-read it
                    pass
                with self._lock:
                    self._direct = r.url
                    self._direct_at = time.monotonic()
                    self.resolutions += 1
                return self._direct
            except Exception as e:                              # noqa: BLE001
                last = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f'could not resolve HF mirror redirect: {last}')

    # -- single range request --------------------------------------------- #
    def get_range(self, start: int, end: int, want_url: bool = False):
        """Return ``(bytes, url)`` for the inclusive byte range ``start..end``."""
        assert 0 <= start <= end < HF_ZIP_SIZE, (start, end)
        last_exc: Exception | None = None
        for attempt in range(HTTP_ATTEMPTS):
            url = self.direct_url(force=(attempt > 0 and last_exc is not None
                                         and '403' in str(last_exc)))
            try:
                r = self.session.get(url, headers={**_user_agent,
                                                   'Range': f'bytes={start}-{end}'},
                                     timeout=(20, 180), stream=True)
                if r.status_code == 403 and attempt == 0:
                    r.close()
                    self.direct_url(force=True)
                    last_exc = Exception('403 stale signature')
                    continue
                if r.status_code == 429:
                    r.close()
                    time.sleep(3 * (attempt + 1))
                    last_exc = Exception('429 rate limited')
                    continue
                r.raise_for_status()
                if r.status_code != 206:
                    r.close()
                    raise RuntimeError(f'expected 206, got {r.status_code}')
                cr = r.headers.get('Content-Range', '')
                if not cr.startswith(f'bytes {start}-{end}/'):
                    r.close()
                    raise RuntimeError(f'bad Content-Range {cr!r}')
                buf = bytearray()
                for chunk in r.iter_content(CHUNK):
                    if not chunk:
                        continue
                    self.limiter.consume(len(chunk))
                    buf += chunk
                r.close()
                if len(buf) != end - start + 1:
                    raise RuntimeError(f'short read {len(buf)} != {end - start + 1}')
                return bytes(buf), url
            except Exception as e:                              # noqa: BLE001
                last_exc = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f'range {start}-{end} failed after {HTTP_ATTEMPTS} '
                           f'attempts: {last_exc}')

    # -- central directory ------------------------------------------------ #
    def central_directory(self, cache_dir: pathlib.Path) -> list[dict]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cd_bin = cache_dir / 'central_directory.bin'
        idx_json = cache_dir / 'msvd_zip_index.json'
        if cd_bin.exists() and idx_json.exists():
            print(f'[zip] reusing cached central directory {cd_bin}', flush=True)
        else:
            tail_len = 2 << 20
            start = HF_ZIP_SIZE - tail_len
            tail, _ = self.get_range(start, HF_ZIP_SIZE - 1)
            i = tail.rfind(b'PK\x05\x06')
            if i < 0:
                raise RuntimeError('EOCD not found in zip tail')
            (_sig, _dn, _ds, n_entries, _ns, cd_size, cd_off, _cl) = \
                struct.unpack('<IHHHHIIH', tail[i:i + 22])
            print(f'[zip] EOCD: entries={n_entries} cd_size={cd_size} cd_off={cd_off}',
                  flush=True)
            # ZIP64 sanity check
            if cd_off == 0xFFFFFFFF or n_entries == 0xFFFF:
                raise RuntimeError('ZIP64 end-of-central-directory not supported here')
            if cd_off + cd_size <= HF_ZIP_SIZE - tail_len:
                # central directory entirely inside the tail we already have
                cd = tail[cd_off - start:cd_off - start + cd_size]
            else:
                cd_raw, _ = self.get_range(cd_off, cd_off + cd_size - 1)
                cd = cd_raw
                tail = tail + cd   # keep for nothing; explicit
            if len(cd) != cd_size:
                raise RuntimeError(f'central directory size {len(cd)} != {cd_size}')
            cd_bin.write_bytes(cd)

        entries = parse_central_directory(cd_bin.read_bytes())
        idx_json.write_text(json.dumps(entries, indent=1))
        print(f'[zip] parsed {len(entries)} central-directory entries', flush=True)
        return entries


def parse_central_directory(cd: bytes) -> list[dict]:
    off, out = 0, []
    while off + 4 <= len(cd) and cd[off:off + 4] == b'PK\x01\x02':
        (_sig, _vmb, _vne, flag, method, _mt, _md, crc, csize, usize,
         nlen, elen, clen, disk, _ia, _ea, lho) = struct.unpack(
            '<IHHHHHHIIIHHHHHII', cd[off:off + 46])
        name = cd[off + 46:off + 46 + nlen].decode('utf-8', 'replace')
        out.append({'name': name, 'method': method, 'flag': flag, 'crc32': crc,
                    'csize': csize, 'usize': usize, 'nlen': nlen, 'elen': elen,
                    'clen': clen, 'local_header_offset': lho, 'disk': disk})
        off += 46 + nlen + elen + clen
    if off != len(cd):
        raise RuntimeError(f'central directory parse stopped at {off}/{len(cd)}')
    return out


def fetch_member(src: ZipSource, entry: dict, dest: pathlib.Path) -> dict:
    """Range-fetch, decompress and CRC-verify one zip member; write atomically."""
    lho = entry['local_header_offset']
    nlen_cd = entry['nlen']
    want = 30 + nlen_cd + LOCAL_HDR_OVERREAD + entry['csize']
    end = min(lho + want - 1, HF_ZIP_SIZE - 1)
    blob, _ = src.get_range(lho, end)
    if blob[:4] != b'PK\x03\x04':
        raise RuntimeError(f'bad local header signature at {lho}')
    # local file header: sig(4) ver(2) flag(2) method(2) mtime(2) mdate(2)
    #                    crc(4) csize(4) usize(4) nlen(2) elen(2) -> 30 bytes
    (_ver, _flag, method, _time, _date, _crc, _cs, _us, nlen, elen) = \
        struct.unpack('<HHHHHIIIHH', blob[4:30])
    data_start = 30 + nlen + elen
    if data_start + entry['csize'] > len(blob):
        # over-read slack was not enough (huge local extra field) -- exact refetch
        blob, _ = src.get_range(lho, lho + data_start + entry['csize'] - 1)
        if data_start + entry['csize'] > len(blob):
            raise RuntimeError('truncated member after exact refetch')
    comp = blob[data_start:data_start + entry['csize']]
    if method != entry['method']:
        raise RuntimeError(f'method mismatch local={method} cd={entry["method"]}')
    raw = zlib.decompress(comp, -15) if method == 8 else comp
    if len(raw) != entry['usize']:
        raise RuntimeError(f'size mismatch {len(raw)} != {entry["usize"]}')
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    if crc != entry['crc32']:
        raise RuntimeError(f'crc32 mismatch {crc:#010x} != {entry["crc32"]:#010x}')

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.partial')
    part.write_bytes(raw)
    if part.stat().st_size != entry['usize']:
        part.unlink(missing_ok=True)
        raise RuntimeError('partial size mismatch')
    part.replace(dest)
    return {'video_id': dest.name, 'zip_member': entry['name'],
            'zip_local_header_offset': lho, 'zip_csize': entry['csize'],
            'zip_usize': entry['usize'], 'zip_crc32': f'{entry["crc32"]:#010x}',
            'size_bytes': len(raw), 'crc_ok': True}


# --------------------------------------------------------------------------- #
# ledger (idempotency)
# --------------------------------------------------------------------------- #
class Ledger:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r['video_id']] = r
        self.lock = threading.Lock()
        self.fh = None

    def __contains__(self, vid: str) -> bool:
        return vid in self.rows

    def get(self, vid: str):
        return self.rows.get(vid)

    def add(self, rec: dict) -> None:
        with self.lock:
            self.rows[rec['video_id']] = rec
            if self.fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.fh = self.path.open('a')
            self.fh.write(json.dumps(rec) + '\n')
            self.fh.flush()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['auto', 'ut_austin_range', 'hf_mirror_zip'],
                    default='auto')
    ap.add_argument('--limit', type=int, default=0, help='smoke subset size')
    ap.add_argument('--workers', type=int, default=MAX_WORKERS)
    ap.add_argument('--rate-mbps', type=float, default=AGG_BYTES_PER_SEC / 1024 / 1024)
    ap.add_argument('--reverify', action='store_true',
                    help='re-extract/verify every planned video, ignoring the ledger')
    ap.add_argument('--out', type=pathlib.Path, default=OFFICIAL / 'fetch_result.json')
    a = ap.parse_args()

    for d in (RAW_DIR, ZIPMETA, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    plan = load_plan()
    if a.limit:
        plan = plan[:a.limit]
    names = [p['video_id'] for p in plan]
    print(f'[plan] {len(names)} official train videos to acquire', flush=True)

    source = a.source
    notes: list[str] = []
    reachable, detail = (False, 'not probed')
    if source == 'auto':
        reachable, detail = ut_austin_reachable()
        notes.append(f'ut_austin probe: reachable={reachable} ({detail})')
        print(f'[probe] UT Austin reachable={reachable} ({detail})', flush=True)
        source = 'ut_austin_range' if reachable else 'hf_mirror_zip'

    ledger = Ledger(ZIPMETA / 'extract_ledger.jsonl')
    limiter = RateLimiter(a.rate_mbps * 1024 * 1024)
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}

    if source == 'ut_austin_range':
        tar = load_tar_index()
        missing = [n for n in names if n not in tar]
        if missing:
            failures.update({n: 'not present in tar index' for n in missing})
        todo = [n for n in names if n in tar]
        session = requests.Session()

        def one(video: str):
            rec = tar[video]
            dest = RAW_DIR / video
            if dest.exists() and not a.reverify:
                if dest.stat().st_size == rec['size']:
                    return video, {'size_bytes': rec['size'], 'reused': True,
                                   'crc_ok': None}
            st, sz = rec['data_offset'], rec['size']
            last = None
            for attempt in range(3):
                try:
                    r = session.get(rec['url'],
                                    headers={**_user_agent,
                                             'Range': f'bytes={st}-{st + sz - 1}'},
                                    timeout=60, stream=True)
                    if r.status_code != 206:
                        raise RuntimeError(f'HTTP {r.status_code}')
                    buf = bytearray()
                    for chunk in r.iter_content(CHUNK):
                        limiter.consume(len(chunk))
                        buf += chunk
                    r.close()
                    if len(buf) != sz:
                        raise RuntimeError(f'short read {len(buf)} != {sz}')
                    part = dest.with_suffix('.avi.partial')
                    part.write_bytes(bytes(buf))
                    part.replace(dest)
                    return video, {'size_bytes': sz, 'crc_ok': None,
                                   'tar_data_offset': st, 'reused': False}
                except Exception as e:                          # noqa: BLE001
                    last = e
                    time.sleep(2 ** attempt)
            return video, {'error': f'{type(last).__name__}: {last}'}

        with cf.ThreadPoolExecutor(a.workers) as ex:
            for video, res in ex.map(one, todo):
                if 'error' in res:
                    failures[video] = res['error']
                else:
                    results[video] = res
    else:
        src = ZipSource(limiter, a.workers)
        entries = src.central_directory(ZIPMETA)
        by_base = {os.path.basename(e['name']): e
                   for e in entries if e['name'].lower().endswith('.avi')}
        missing = [n for n in names if n not in by_base]
        if missing:
            failures.update({n: 'not present in HF mirror zip' for n in missing})
        todo = [n for n in names if n in by_base and n not in failures]
        print(f'[zip] {len(todo)} members to extract, {len(missing)} absent',
              flush=True)

        done = 0
        done_lock = threading.Lock()

        def one(video: str):
            nonlocal done
            entry = by_base[video]
            dest = RAW_DIR / video
            if not a.reverify and video in ledger and dest.exists() \
                    and dest.stat().st_size == entry['usize']:
                with done_lock:
                    done += 1
                return video, dict(ledger.get(video), reused=True)
            last = None
            for attempt in range(3):
                try:
                    rec = fetch_member(src, entry, dest)
                    rec['download_source'] = 'hf_mirror_zip'
                    rec['reused'] = False
                    ledger.add(rec)
                    with done_lock:
                        done += 1
                        if done % 25 == 0:
                            print(f'[zip] {done}/{len(todo)} extracted '
                                  f'({time.time() - t0:.0f}s, '
                                  f'{limiter.total / 2**20:.0f} MiB read)', flush=True)
                    return video, rec
                except Exception as e:                          # noqa: BLE001
                    last = e
                    time.sleep(1.5 * (attempt + 1))
            return video, {'error': f'{type(last).__name__}: {last}'}

        with cf.ThreadPoolExecutor(a.workers) as ex:
            for video, res in ex.map(one, todo):
                if 'error' in res:
                    failures[video] = res['error']
                else:
                    results[video] = res
        ledger.close()

    # post-verify every downloaded file (size + sha256 present, crc from ledger)
    verified = {}
    for video, res in results.items():
        dest = RAW_DIR / video
        if not dest.exists():
            failures[video] = 'file missing after download'
            continue
        sz = dest.stat().st_size
        exp = res.get('size_bytes')
        if exp is not None and sz != exp:
            failures[video] = f'size mismatch on disk {sz} != {exp}'
            continue
        verified[video] = {**res, 'size_bytes': sz, 'sha256': sha256_file(dest)}

    # cross-check against the UT Austin tar index where available
    tar = load_tar_index()
    size_crosscheck = {'compared': 0, 'mismatch': 0, 'details': []}
    for video in verified:
        if video in tar:
            size_crosscheck['compared'] += 1
            if tar[video]['size'] != verified[video]['size_bytes']:
                size_crosscheck['mismatch'] += 1
                size_crosscheck['details'].append(
                    {'video_id': video, 'tar_size': tar[video]['size'],
                     'on_disk': verified[video]['size_bytes']})

    out = {
        'source_used': source,
        'ut_austin_probe': {'reachable': reachable, 'detail': detail},
        'notes': notes,
        'planned': len(names),
        'acquired': len(verified),
        'failed': len(failures),
        'failures': failures,
        'tar_index_size_crosscheck': size_crosscheck,
        'raw_dir': str(RAW_DIR),
        'total_raw_bytes': sum(v['size_bytes'] for v in verified.values()),
        'average_throughput_bytes_per_sec': round(limiter.total / max(time.time() - t0, 1e-9), 1),
        'aggregate_bytes_read': limiter.total,
        'zip_redirect_resolutions': None,
        'wall_seconds': round(time.time() - t0, 1),
        'verification': verified,
    }
    if source == 'hf_mirror_zip':
        out['zip_redirect_resolutions'] = src.resolutions
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('verification', 'failures')}, indent=2))
    print(f'[done] acquired={len(verified)} failed={len(failures)} '
          f'wall={out["wall_seconds"]}s', flush=True)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
