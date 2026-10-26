"""Read-only historical replay into a NEW audit directory; never rewrites source."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from metrics_contract import correct_saved_anchor, answer_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    src, out = Path(a.source).resolve(), Path(a.out).resolve()
    if src == out or src in out.parents:
        raise SystemExit('output must be outside historical source directory')
    if not any(str(out).startswith(x + '/') for x in ['/mnt/local', '/mnt/local2', '/mnt/local3']):
        raise SystemExit('audit output must resolve under allowed /mnt storage')
    out.mkdir(parents=True, exist_ok=True)
    summary, hashes = {}, {}
    for p in sorted((src / 'eval').glob('coverage_*.json')):
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        d = json.loads(p.read_text())
        rows = [correct_saved_anchor(x) for x in d['anchors']]
        summary[d['tag']] = dict(n=len(rows), horizon_histogram=dict(collections.Counter(x['horizon'] for x in rows)),
            blockers=dict(collections.Counter(x['first_blocker'] for x in rows)),
            changed_blocker=sum(x['first_blocker'] != x['historical_first_blocker'] for x in rows),
            L_actual=float(np.mean([x['L_actual'] for x in rows])),
            L_tree=float(np.mean([x['L_tree'] for x in rows])),
            online_anchor_distribution='unavailable')
        (out / p.name).write_text(json.dumps(dict(source=str(p), source_sha256=hashes[str(p)],
            summary=summary[d['tag']], anchors=rows), indent=2))
    pairs, accuracy = [], {}
    for split in ['dev', 'test']:
        for seed in [20261013, 20261014]:
            groups = []
            for arm in ['opd', 'sft']:
                p = src / 'eval' / f'{arm}s{seed}u120_{split}.json'
                hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
                rows = json.loads(p.read_text())['rows']
                if any(x["split"] != split for x in rows):
                    raise ValueError("cross-split row contamination")
                accuracy[p.stem] = answer_metrics(rows)
                g = collections.defaultdict(list)
                for x in rows:
                    g[x['image_group_id']].append(x['n_tokens'] / x['spec_rounds'])
                groups.append({k: float(np.mean(v)) for k, v in g.items()})
            if groups[0].keys() != groups[1].keys():
                raise ValueError('unmatched paired groups')
            delta = np.array([groups[0][k] - groups[1][k] for k in sorted(groups[0])])
            rng = np.random.default_rng(20261015)
            draws = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
            pairs.append(dict(split=split, seed=seed, metric='actual_output_tokens_per_round',
                              delta=float(delta.mean()), ci95=np.percentile(draws,[2.5,97.5]).tolist()))
    result = dict(coverage=summary, paired_deltas=pairs, accuracy=accuracy, source_hashes=hashes)
    (out / 'corrected_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(coverage=summary, paired_deltas=pairs), indent=2))


if __name__ == '__main__':
    main()
