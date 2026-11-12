#!/usr/bin/env python3
"""Run one explicit finite GPU job with UUID locks and immutable launch identity."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', required=True)
    a = ap.parse_args()
    spec_path = Path(a.spec).resolve()
    job = json.loads(spec_path.read_text())
    out = Path(job['job_dir']).resolve()
    out.mkdir(parents=True, exist_ok=False)
    uuids = job['gpu_uuids']
    if not uuids or len(set(uuids)) != len(uuids) or any(not u.startswith('GPU-') for u in uuids):
        raise ValueError('GPU UUIDs must be explicit and unique')
    locks = []
    for uid in sorted(uuids):
        handle = open(f'/tmp/xtuner-{uid}.lock', 'a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
        '--format=csv,noheader,nounits'], text=True)
    occupied = [line for line in processes.splitlines() if line.split(',')[0].strip() in uuids]
    if occupied:
        raise RuntimeError(f'assigned GPUs have external processes: {occupied}')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(uuids), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    cwd = job['cwd']
    start = time.time()
    identity = dict(job, spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                    started_utc=datetime.now(timezone.utc).isoformat(), wrapper_pid=os.getpid(),
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip(),
                    source_status=subprocess.check_output(['git','status','--porcelain'],cwd=cwd,text=True),
                    automatic_retry=False)
    with (out/'LAUNCH.json').open('x') as stream:
        json.dump(identity,stream,indent=2)
    with (out/'stdout.log').open('x') as log:
        child = subprocess.Popen(job['argv'], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        def terminate(sig, frame):
            os.killpg(child.pid, sig)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        (out/'PID.json').write_text(json.dumps({'wrapper':os.getpid(),'child':child.pid}))
        code = child.wait()
    (out/'EXIT.json').write_text(json.dumps(dict(exit_code=code,elapsed_s=time.time()-start,
        completed_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
