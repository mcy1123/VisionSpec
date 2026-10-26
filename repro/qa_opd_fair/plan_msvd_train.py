#!/usr/bin/env python
"""Build the MSVD-QA official-train download plan.

Reads the official LAVIS qa_train.json and the 512-byte-range tar index of the
UT Austin YouTubeClips.tar, and emits a machine-readable plan listing every
official train video with its tar offset/size (or a missing marker).

Idempotent: only writes under /mnt/local2/mcy/datasets/msvd_qa_full/.
"""
import argparse, json, pathlib, collections, sys

DATASET = pathlib.Path('/mnt/local2/mcy/datasets/msvd_qa_full')
OFFICIAL = DATASET / 'official'
DEFAULT_TAR_INDEX = pathlib.Path(
    '/mnt/local/mcy/GLANCE/results/videoqa_transfer/20261013T134000/sources/msvd_tar_index.jsonl')


def source_id(video_id: str) -> str:
    """First 11 chars of the YouTube id, per the project convention."""
    return video_id[:11]


def load_official_train_names(path: pathlib.Path):
    rows = json.loads(path.read_text())
    # preserve official file order of first appearance for stable planning
    seen, order = set(), []
    per_video = collections.Counter()
    for r in rows:
        v = r['video']
        per_video[v] += 1
        if v not in seen:
            seen.add(v)
            order.append(v)
    return order, per_video, len(rows)


def load_tar_index(path: pathlib.Path):
    idx = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        name = pathlib.Path(rec['name']).name
        if rec.get('type') == '5' or not name:
            continue
        rec = dict(rec)
        rec['tar_name'] = name
        idx[name] = rec
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar-index', type=pathlib.Path, default=DEFAULT_TAR_INDEX)
    ap.add_argument('--out', type=pathlib.Path, default=OFFICIAL / 'train_plan.json')
    a = ap.parse_args()

    names, per_video, n_qa = load_official_train_names(OFFICIAL / 'qa_train.json')
    tar = load_tar_index(a.tar_index)

    entries, missing = [], []
    total_planned_bytes = 0
    for v in names:
        rec = tar.get(v)
        if rec is None:
            missing.append(v)
            entries.append({
                'video_id': v, 'source_id': source_id(v), 'qa_count': per_video[v],
                'tar_name': None, 'in_tar_index': False,
                'data_offset': None, 'size': None, 'url': None,
            })
        else:
            total_planned_bytes += rec['size']
            entries.append({
                'video_id': v, 'source_id': source_id(v), 'qa_count': per_video[v],
                'tar_name': rec['tar_name'], 'in_tar_index': True,
                'data_offset': rec['data_offset'], 'size': rec['size'],
                'url': rec['url'],
            })

    sizes = [e['size'] for e in entries if e['in_tar_index']]
    plan = {
        'official_train_qa_rows': n_qa,
        'official_train_videos': len(names),
        'tar_index_video_entries': len(tar),
        'in_tar_index': len(names) - len(missing),
        'missing_from_tar_index': len(missing),
        'missing_video_names': missing,
        'total_planned_raw_bytes': total_planned_bytes,
        'total_planned_raw_mib': round(total_planned_bytes / 2**20, 2),
        'size_min': min(sizes) if sizes else None,
        'size_max': max(sizes) if sizes else None,
        'size_mean': round(sum(sizes) / len(sizes), 1) if sizes else None,
        'tar_index_path': str(a.tar_index),
        'qa_train_path': str(OFFICIAL / 'qa_train.json'),
        'tar_url': entries[0]['url'] if entries and entries[0]['url'] else None,
    }
    a.out.write_text(json.dumps(plan, indent=2))
    with (OFFICIAL / 'train_plan.jsonl').open('w') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
