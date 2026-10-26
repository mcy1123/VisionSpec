"""Streaming Chrome/CUPTI trace summary; bounded memory, no event-tree build."""
import argparse
import bisect
from collections import Counter,defaultdict
import heapq
import json
from pathlib import Path
import re


def events(path):
    decoder=json.JSONDecoder();skip=re.compile(r'[\s,]*')
    with path.open() as f:
        buf=''
        while '"traceEvents"' not in buf:
            chunk=f.read(1048576)
            if not chunk:raise ValueError('missing traceEvents')
            buf+=chunk
        pos=buf.index('[',buf.index('"traceEvents"'))+1
        while True:
            pos=skip.match(buf,pos).end()
            if pos<len(buf) and buf[pos]==']':return
            try: e,end=decoder.raw_decode(buf,pos)
            except json.JSONDecodeError:
                chunk=f.read(1048576)
                if not chunk:raise ValueError('incomplete trace')
                buf=buf[pos:]+chunk;pos=0;continue
            yield e;pos=end
            if pos>1048576:
                buf=buf[pos:];pos=0


def summarize(path):
    api=Counter();copies=Counter();cpu=defaultdict(float);counts=Counter()
    scopes=defaultdict(list);launches=defaultdict(list);kernels=[]
    for e in events(path):
        cat=e.get('cat');name=e.get('name','');args=e.get('args',{});thread=(e.get('pid'),e.get('tid'))
        if cat=='user_annotation' and e.get('ph')=='X':
            scopes[thread].append((e['ts'],e['ts']+e.get('dur',0),name))
            cpu[name]+=e.get('dur',0)/1e6;counts[name]+=1
        elif cat=='cuda_runtime':
            api[name]+=1
            if 'correlation' in args:launches[thread].append((e['ts'],args['correlation']))
        elif cat=='kernel':kernels.append((args.get('correlation'),e.get('dur',0)/1e6))
        elif cat and 'gpu_memcpy' in cat:copies[name]+=1
    backwards=defaultdict(list)
    for (pid,tid),ranges in scopes.items():
        backwards[pid].extend((a,b) for a,b,name in ranges if name=='anchor_backward')
    for pid in backwards: backwards[pid].sort()
    back_starts={pid:[a for a,b in ranges] for pid,ranges in backwards.items()}
    correlations={}
    for thread,calls in launches.items():
        ranges=sorted(scopes[thread]);heap=[];idx=0
        for ts,corr in sorted(calls):
            while idx<len(ranges) and ranges[idx][0]<=ts:
                start,end,name=ranges[idx];heapq.heappush(heap,(-start,end,name));idx+=1
            while heap and heap[0][1]<ts:heapq.heappop(heap)
            label=heap[0][2] if heap else 'unattributed'
            if label=='unattributed':
                intervals=backwards[thread[0]]
                i=bisect.bisect_right(back_starts.get(thread[0],[]),ts)-1
                if i>=0 and ts<=intervals[i][1]:label='anchor_backward_cross_thread'
            correlations[corr]=label
    gpu=defaultdict(float)
    for corr,dur in kernels:gpu[correlations.get(corr,'unattributed')]+=dur
    return dict(raw_trace=str(path),actual_range_calls=dict(counts),cpu_inclusive_seconds=dict(cpu),
                gpu_kernel_seconds_by_innermost_launch_scope=dict(gpu),kernel_count=len(kernels),
                cuda_api_counts=dict(api),explicit_sync_api_counts={k:v for k,v in api.items() if 'Synchronize' in k},
                gpu_copy_counts=dict(copies),notes='Profile-only. CPU ranges overlap. Synchronous copies can also block. Cross-thread backward attribution uses launch timestamp inside the synchronous main-thread backward range. Optimizer not executed.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('trace');args=ap.parse_args();p=Path(args.trace).resolve()
    if not any(str(p).startswith(m+'/') for m in ('/mnt/local','/mnt/local2','/mnt/local3')):raise ValueError('mounted output required')
    result=summarize(p);p.with_name('stream_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('raw_trace','actual_range_calls','gpu_kernel_seconds_by_innermost_launch_scope','explicit_sync_api_counts','kernel_count')},indent=2))

if __name__=='__main__':main()
