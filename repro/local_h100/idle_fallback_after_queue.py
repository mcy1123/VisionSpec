#!/usr/bin/env python3
"""Run the user's existing idle burner only after the finite queue is terminal."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
ROOT=Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
CONTROL=ROOT/'control/idle_fallback'
TERMINAL={'SUCCEEDED','FAILED','BLOCKED'}


def status(value):
    value['updated_utc']=datetime.now(timezone.utc).isoformat()
    tmp=CONTROL/'STATUS.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(CONTROL/'STATUS.json')


def main():
    CONTROL.mkdir(exist_ok=True)
    while not (CONTROL/'STOP').exists():
        try:
            queue=json.loads((ROOT/'control/scheduler/queue.json').read_text())
            state=json.loads((ROOT/'control/scheduler/STATE.json').read_text())
            pending=[j['id'] for j in queue['jobs'] if state['jobs'].get(j['id'],{}).get('status') not in TERMINAL]
            if state.get('queue_valid') and queue['jobs'] and not pending:
                break
            status(dict(status='WAITING_FOR_ALL_QUEUED_WORK',pending_jobs=len(pending),queue_valid=state.get('queue_valid')))
        except Exception as exc:status(dict(status='WAITING_ON_STATE',error=str(exc)))
        time.sleep(15)
    else:
        status(dict(status='STOPPED'));return
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    devices=sorted((int(x.split(',')[0]),x.split(',')[1].strip()) for x in raw.splitlines())
    # gpu.py uses NVML physical indices and CUDA ordinals: align explicitly.
    assert [i for i,u in devices]==list(range(len(devices)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(u for i,u in devices),CUDA_DEVICE_ORDER='PCI_BUS_ID')
    cmd=['/opt/conda/bin/python','-u','/czsun/mcy/code-l/gpu.py','--gpus','all','--wait-idle','--idle-seconds','60','--no-fill-vram','--check-interval','1']
    child=subprocess.Popen(cmd,env=env,start_new_session=True)
    status(dict(status='IDLE_FALLBACK_ACTIVE',pid=child.pid,argv=cmd,gpu_uuid_order=[u for i,u in devices]))
    read_errors = 0
    try:
        while child.poll() is None:
            if (CONTROL/'STOP').exists():break
            # New queue additions revoke fallback. No automatic relaunch/retry.
            try:
                q=json.loads((ROOT/'control/scheduler/queue.json').read_text())
                s=json.loads((ROOT/'control/scheduler/STATE.json').read_text())
                read_errors = 0
            except (OSError, ValueError) as exc:
                read_errors += 1
                status(dict(status='STATE_READ_RETRY', pid=child.pid, error=str(exc), consecutive_errors=read_errors))
                if read_errors >= 5:
                    break  # Release our load if scheduler state stays unknown.
                time.sleep(1)
                continue
            if not s.get('queue_valid') or any(s['jobs'].get(j['id'],{}).get('status') not in TERMINAL for j in q['jobs']):break
            time.sleep(1)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
        status(dict(status='FALLBACK_EXITED',returncode=child.returncode))

if __name__=='__main__':main()
