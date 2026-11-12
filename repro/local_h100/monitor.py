#!/usr/bin/env python3
"""Read-only GPU telemetry; low utilization is reported, never masked by filler work."""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def query(fields, kind='gpu'):
    output = subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [[v.strip() for v in line.split(',')] for line in output.splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control', required=True)
    ap.add_argument('--interval', type=float, default=5)
    a = ap.parse_args()
    root = Path(a.control).resolve()
    root.mkdir(parents=True, exist_ok=True)
    history = {}
    with (root / 'gpu_telemetry.jsonl').open('a', buffering=1) as log:
        while not (root / 'STOP_MONITOR').exists():
            start = time.time()
            record = {'utc': datetime.now(timezone.utc).isoformat(), 'unix': start}
            try:
                rows = query('index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw')
                processes = query('gpu_uuid,pid,process_name,used_gpu_memory', 'compute-apps')
                record['processes'] = processes
                record['gpus'] = []
                for index, uid, util, used, total, temp, power in rows:
                    samples = history.setdefault(uid, deque())
                    samples.append((start, float(util)))
                    while samples and samples[0][0] < start - 60:
                        samples.popleft()
                    avg = sum(v for _, v in samples) / len(samples)
                    record['gpus'].append(dict(index=int(index), uuid=uid,
                        utilization=float(util), mean_60s=round(avg, 2),
                        window_s=round(start-samples[0][0], 1), below_target=avg < 50,
                        memory_used_mib=float(used), memory_total_mib=float(total),
                        temperature_c=float(temp), power_w=float(power)))
                state = root / 'RUN_STATE.json'
                record['run_state'] = json.loads(state.read_text()) if state.exists() else {'phase':'preparing'}
            except Exception as exc:
                record['error'] = repr(exc)
            log.write(json.dumps(record) + '\n')
            temporary = root / 'GPU_STATUS.json.tmp'
            temporary.write_text(json.dumps(record, indent=2) + '\n')
            temporary.replace(root / 'GPU_STATUS.json')
            time.sleep(max(.1, a.interval - (time.time() - start)))


if __name__ == '__main__':
    main()
