#!/usr/bin/env python3
"""Dispatch only explicit finite jobs, with dependencies and GPU UUID exclusion.

Queue v1 contains jobs with id, job_dir, cwd, gpu_uuids and argv (run_job.py's
fields), plus optional depends_on IDs, inputs paths, requires checks and result.
Each requires check has exit (EXIT.json), optionally result and status. A job's
own result is {"path": "...", "status": "LOSSLESS_PASSED"}. No shell expansion,
GPU filler, automatic retry, training-budget invention or process killing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ALLOWED_RUN_ROOT = Path('/czsun/mcy/video_opd_pilot/local_h100_20261112')
ACTIVE = {'STARTING', 'RUNNING', 'RECOVERY_REQUIRED'}
TERMINAL = {'SUCCEEDED', 'FAILED', 'BLOCKED'}
JOB_KEYS = {'id', 'job_dir', 'cwd', 'gpu_uuids', 'argv', 'depends_on', 'inputs',
            'requires', 'result'}


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def confined(path, root):
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if p == root or not p.is_relative_to(root):
        raise ValueError(f'path must resolve beneath this run: {path}')
    return p


def process_identity(pid):
    """PID plus Linux start ticks prevents attaching to a reused PID."""
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(fields[19])}
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def process_alive(identity):
    return bool(identity and process_identity(identity['pid']) == identity)


def validate_queue(document, root):
    if not isinstance(document, dict) or set(document) != {'version', 'jobs'}:
        raise ValueError('queue must contain exactly version and jobs')
    if document['version'] != 1 or not isinstance(document['jobs'], list):
        raise ValueError('expected queue version 1 and a jobs list')
    jobs, output_dirs = {}, set()
    for raw in document['jobs']:
        if not isinstance(raw, dict) or set(raw) - JOB_KEYS:
            raise ValueError('unknown job fields')
        name = raw.get('id')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', name):
            raise ValueError('job id must be a short filename-safe string')
        if name in jobs:
            raise ValueError(f'duplicate job: {name}')
        job = dict(raw)
        job['job_dir'] = str(confined(job['job_dir'], root))
        if job['job_dir'] in output_dirs:
            raise ValueError('jobs cannot share job_dir')
        output_dirs.add(job['job_dir'])
        cwd = Path(job['cwd'])
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError('cwd must be an existing absolute directory')
        job['cwd'] = str(cwd.resolve())
        args = job['argv']
        if not isinstance(args, list) or not args or any(not isinstance(s, str) or not s or '\0' in s for s in args):
            raise ValueError('argv must be a nonempty string list; shell commands are not supported')
        gpus = job['gpu_uuids']
        if (not isinstance(gpus, list) or not gpus or len(set(gpus)) != len(gpus)
                or any(not isinstance(g, str) or not re.fullmatch(r'GPU-[A-Za-z0-9-]+', g) for g in gpus)):
            raise ValueError('GPU UUIDs must be explicit, nonempty and unique')
        deps = job.setdefault('depends_on', [])
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps) or len(set(deps)) != len(deps):
            raise ValueError('depends_on must contain unique job ids')
        inputs = job.setdefault('inputs', [])
        if not isinstance(inputs, list) or any(not isinstance(p, str) or not p for p in inputs):
            raise ValueError('inputs must be a list of paths')
        # Inputs are read-only and may include public models outside this run.
        job['inputs'] = [str((root / p).resolve()) for p in inputs]
        checks = job.setdefault('requires', [])
        if not isinstance(checks, list):
            raise ValueError('requires must be a list')
        job['requires'] = []
        for check in checks:
            if (not isinstance(check, dict) or 'exit' not in check
                    or set(check) - {'exit', 'result', 'status'}
                    or (('result' in check) != ('status' in check))):
                raise ValueError('requires check needs exit and optional result/status together')
            item = {'exit': str(confined(check['exit'], root))}
            if 'result' in check:
                if not isinstance(check['status'], str) or not check['status']:
                    raise ValueError('required status must be a nonempty string')
                item.update(result=str(confined(check['result'], root)), status=check['status'])
            job['requires'].append(item)
        result = job.get('result')
        if result is not None:
            if (not isinstance(result, dict) or set(result) != {'path', 'status'}
                    or not isinstance(result['status'], str) or not result['status']):
                raise ValueError('result must contain path and nonempty status')
            job['result'] = {'path': str(confined(result['path'], root)), 'status': result['status']}
        jobs[name] = job
    for name, job in jobs.items():
        if any(dep not in jobs for dep in job['depends_on']):
            raise ValueError(f'unknown dependency for {name}')
    visiting, complete = set(), set()
    def visit(name):
        if name in visiting:
            raise ValueError('dependency cycle')
        if name in complete:
            return
        visiting.add(name)
        for dep in jobs[name]['depends_on']:
            visit(dep)
        visiting.remove(name)
        complete.add(name)
    for name in jobs:
        visit(name)
    return jobs


class Supervisor:
    def __init__(self, run_root, control=None, queue=None, *, runner=None,
                 allowed_root=ALLOWED_RUN_ROOT):
        self.root = Path(run_root).resolve()
        if self.root != Path(allowed_root).resolve() or not self.root.is_dir():
            raise ValueError('run-root must be this authorized run and already exist')
        self.control = confined(control or self.root / 'control/scheduler', self.root)
        self.control.mkdir(parents=True, exist_ok=True)
        self.queue_path = confined(queue or self.control / 'queue.json', self.root)
        self.runner = Path(runner or Path(__file__).with_name('run_job.py')).resolve()
        if not self.runner.is_file():
            raise ValueError('runner is missing')
        self.state_path = self.control / 'STATE.json'
        self.children = {}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get('run_root') != str(self.root) or self.state.get('version') != 1:
                raise ValueError('existing scheduler state belongs to another run/schema')
        else:
            self.state = {'version': 1, 'run_root': str(self.root), 'jobs': {}, 'queue_valid': False}
        self.last_queue_error = None
        self.jobs = {}

    def event(self, kind, **fields):
        with (self.control / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'utc': now(), 'event': kind, **fields}, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def save(self):
        self.state['updated_utc'] = now()
        atomic_json(self.state_path, self.state)

    def transition(self, name, status, reason=None, **fields):
        row = self.state['jobs'][name]
        changed = row.get('status') != status or row.get('reason') != reason
        row.update(status=status, reason=reason, **fields)
        if changed:
            row['updated_utc'] = now()
            self.event('job_state', id=name, status=status, reason=reason)

    def load_queue(self):
        try:
            jobs = validate_queue(json.loads(self.queue_path.read_text()), self.root)
            old = self.state['jobs']
            if set(old) - set(jobs):
                raise ValueError('queue updates may not remove previously accepted job ids')
            for name, row in old.items():
                if row.get('dispatched') and fingerprint(jobs[name]) != row['definition_sha256']:
                    raise ValueError(f'cannot modify dispatched job definition: {name}')
            for name, job in jobs.items():
                if name not in old:
                    old[name] = {'status': 'PENDING', 'definition': job,
                                 'definition_sha256': fingerprint(job), 'dispatched': False}
                    self.event('job_registered', id=name, definition_sha256=fingerprint(job))
                elif not old[name].get('dispatched'):
                    if old[name]['definition_sha256'] != fingerprint(job):
                        self.event('pending_job_updated', id=name, definition_sha256=fingerprint(job))
                        old[name].update(definition=job, definition_sha256=fingerprint(job))
            self.jobs = jobs
            self.state.update(queue_valid=True, queue_error=None)
            self.last_queue_error = None
            return True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            error = f'{type(exc).__name__}: {exc}'
            self.state.update(queue_valid=False, queue_error=error)
            if error != self.last_queue_error:
                self.event('queue_rejected', error=error)
                self.last_queue_error = error
            return False

    def collect(self):
        for name, row in self.state['jobs'].items():
            if row['status'] not in ACTIVE:
                continue
            child = self.children.get(name)
            alive = child.poll() is None if child is not None else process_alive(row.get('process'))
            if alive:
                if row['status'] != 'RUNNING':
                    self.transition(name, 'RUNNING')
                continue
            job = row['definition']
            exit_path = Path(job['job_dir']) / 'EXIT.json'
            if exit_path.exists():
                try:
                    result = json.loads(exit_path.read_text())
                    code = result['exit_code']
                    if type(code) is not int:
                        raise ValueError('exit_code must be an integer')
                    if code != 0:
                        self.transition(name, 'FAILED', f'EXIT={code}', exit_code=code)
                        continue
                    expected = job.get('result')
                    if expected:
                        actual = json.loads(Path(expected['path']).read_text()).get('status')
                        if actual != expected['status']:
                            raise ValueError(f'result status {actual!r} != {expected["status"]!r}')
                    self.transition(name, 'SUCCEEDED', exit_code=0, completed_utc=now())
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    self.transition(name, 'FAILED', f'invalid final result: {exc}')
            elif child is not None:
                # run_job may fail before creating EXIT (e.g. an occupied GPU).
                code = child.returncode
                self.transition(name, 'FAILED', f'wrapper exited {code} without EXIT.json', exit_code=code)
            elif row.get('process'):
                # A crashed wrapper might leave its training process alive. Keep
                # these GPUs reserved for explicit operator inspection.
                self.transition(name, 'RECOVERY_REQUIRED', 'wrapper missing without final EXIT; no automatic relaunch')
            else:
                self.transition(name, 'RECOVERY_REQUIRED', 'dispatch interrupted before PID was recorded; no automatic relaunch')

    def readiness(self, job):
        for dep in job['depends_on']:
            state = self.state['jobs'][dep]['status']
            if state in {'FAILED', 'BLOCKED', 'RECOVERY_REQUIRED'}:
                return 'BLOCKED', f'dependency {dep} is {state}'
        for dep in job['depends_on']:
            state = self.state['jobs'][dep]['status']
            if state != 'SUCCEEDED':
                return 'WAITING', f'dependency {dep} is {state}'
        for check in job['requires']:
            try:
                result = json.loads(Path(check['exit']).read_text())
                code = result['exit_code']
                if type(code) is not int:
                    return 'BLOCKED', f'invalid exit_code at {check["exit"]}'
                if code != 0:
                    return 'BLOCKED', f'prerequisite EXIT={code}: {check["exit"]}'
                if 'result' in check:
                    actual = json.loads(Path(check['result']).read_text()).get('status')
                    if actual != check['status']:
                        return 'BLOCKED', f'prerequisite status {actual!r} != {check["status"]!r}'
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                return 'WAITING', 'prerequisite exit/result is missing or not ready'
        missing = [p for p in job['inputs'] if not Path(p).exists()]
        if missing:
            return 'WAITING', f'inputs not ready: {missing}'
        return 'READY', None

    def dispatch(self, name):
        row = self.state['jobs'][name]
        job = row['definition']
        if Path(job['job_dir']).exists():
            self.transition(name, 'FAILED', 'job_dir already exists; refusing overwrite or automatic adoption')
            # Freeze this launch definition too: the failure must not be hidden
            # by changing its output directory and retrying the same job id.
            row['dispatched'] = True
            return
        spec_dir = self.control / 'specs'
        spec_dir.mkdir(exist_ok=True)
        spec = spec_dir / f'{name}.json'
        if spec.exists():
            row['dispatched'] = True
            self.transition(name, 'RECOVERY_REQUIRED', 'immutable launch spec exists without dispatch state; no automatic relaunch')
            return
        with spec.open('x') as stream:
            json.dump(job, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        row.update(dispatched=True, spec=str(spec), started_utc=now())
        self.transition(name, 'STARTING')
        self.save()  # Durable intent BEFORE Popen: crashes never auto-resubmit.
        try:
            with (self.control / f'{name}.wrapper.log').open('x') as log:
                child = subprocess.Popen([sys.executable, str(self.runner), '--spec', str(spec)],
                                         cwd=job['cwd'], stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            self.children[name] = child
            identity = process_identity(child.pid)
            row['process'] = identity
            self.transition(name, 'RUNNING', wrapper_pid=child.pid)
            self.event('job_dispatched', id=name, gpu_uuids=job['gpu_uuids'], process=identity,
                       definition_sha256=row['definition_sha256'])
        except OSError as exc:
            self.transition(name, 'FAILED', f'cannot spawn wrapper: {exc}')
        self.save()

    def tick(self):
        valid = self.load_queue()
        self.collect()  # Always observe active jobs even if queue was rejected.
        stopped = (self.control / 'STOP_SCHEDULER').exists()
        self.state['dispatch_stopped'] = stopped
        if valid:
            used = {g for row in self.state['jobs'].values() if row['status'] in ACTIVE
                    for g in row['definition']['gpu_uuids']}
            for name, job in self.jobs.items():
                row = self.state['jobs'][name]
                if row.get('dispatched') or row['status'] in TERMINAL:
                    continue
                status, reason = self.readiness(job)
                if status != 'READY':
                    self.transition(name, status, reason)
                elif stopped:
                    self.transition(name, 'WAITING', 'STOP_SCHEDULER prevents new dispatch')
                elif used.intersection(job['gpu_uuids']):
                    self.transition(name, 'WAITING', 'assigned GPU UUIDs are reserved by another job')
                else:
                    self.dispatch(name)
                    if row['status'] in ACTIVE:
                        used.update(job['gpu_uuids'])
        self.save()
        return self.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ALLOWED_RUN_ROOT)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--queue', type=Path)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true', help='one scan; active jobs continue independently')
    args = parser.parse_args()
    if args.interval < 0.1:
        parser.error('interval must be at least 0.1 seconds')
    supervisor = Supervisor(args.run_root, args.control, args.queue)
    with (supervisor.control / 'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor.event('scheduler_started', pid=os.getpid(), interval=args.interval)
        try:
            while True:
                started = time.monotonic()
                supervisor.tick()
                if args.once:
                    break
                time.sleep(max(0.1, args.interval - (time.monotonic() - started)))
        except KeyboardInterrupt:
            supervisor.event('scheduler_interrupted', running_jobs_left_untouched=True)


if __name__ == '__main__':
    main()
