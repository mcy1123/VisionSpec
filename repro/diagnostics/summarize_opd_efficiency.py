"""Summarize clean no-update timings; raw traces use the bounded stream parser."""
import argparse
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');args=ap.parse_args()
    root=Path(args.run).resolve()
    if not any(str(root).startswith(p+'/') for p in ('/mnt/local','/mnt/local2','/mnt/local3')):
        raise ValueError('mounted output required')
    timings=[json.loads(p.read_text()) for p in sorted(root.glob('timing_*.json'))]
    measured=[r for r in timings if not r['warmup']]
    result=dict(kind='no_update_rollout_scoring_backward',optimizer_measured=False,
        repeats=len(measured),seconds=[r['seconds'] for r in measured],
        median_seconds=statistics.median(r['seconds'] for r in measured) if measured else None,
        losses=[r['loss'] for r in measured],
        peak_allocated_bytes=max((r['peak_allocated'] for r in measured),default=None),
        peak_reserved_bytes=max((r['peak_reserved'] for r in measured),default=None))
    if (root/'profile.json').exists():
        from diagnostics.summarize_opd_profile_stream import summarize
        result['profiler']=summarize(root/'profile.json')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
