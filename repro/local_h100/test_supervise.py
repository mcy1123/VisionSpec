"""Finite CPU subprocesses test orchestration; no CUDA or nvidia-smi is used."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

spec = importlib.util.spec_from_file_location('supervise', Path(__file__).with_name('supervise.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def environment(tmp_path):
    runner = tmp_path / 'cpu_runner.py'
    runner.write_text('''import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--spec');a=p.parse_args()
j=json.loads(Path(a.spec).read_text());d=Path(j['job_dir']);d.mkdir(parents=True)
(d/'LAUNCH.json').write_text(json.dumps(j))
code=subprocess.run(j['argv'],cwd=j['cwd']).returncode
(d/'EXIT.json').write_text(json.dumps({'exit_code':code}))
raise SystemExit(code)
''')
    supervisor = mod.Supervisor(tmp_path, runner=runner, allowed_root=tmp_path)
    return tmp_path, runner, supervisor


def job(root, name, gpu='GPU-a', code=0, delay=0.04, **extra):
    return dict(id=name, job_dir=str(root / 'jobs' / name), cwd=str(root),
                gpu_uuids=[gpu], argv=[sys.executable, '-c', f'import time; time.sleep({delay}); raise SystemExit({code})'],
                **extra)


def queue(s, jobs):
    mod.atomic_json(s.queue_path, {'version': 1, 'jobs': jobs})


def run_until(s, condition, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = s.tick()
        if condition(state):
            return state
        time.sleep(.02)
    raise AssertionError(s.state)


def cleanup(s):
    for child in s.children.values():
        child.wait(timeout=5)


def test_gpu_exclusion_and_dependency_handoff(environment):
    root, _, s = environment
    queue(s, [job(root, 'first', delay=.15), job(root, 'second', depends_on=['first']),
              job(root, 'parallel', gpu='GPU-b', delay=.15)])
    state = s.tick()
    assert state['jobs']['first']['status'] == 'RUNNING'
    assert state['jobs']['parallel']['status'] == 'RUNNING'
    assert state['jobs']['second']['status'] == 'WAITING'
    run_until(s, lambda st: all(j['status'] == 'SUCCEEDED' for j in st['jobs'].values()))
    events = [json.loads(x) for x in (s.control / 'events.jsonl').read_text().splitlines()]
    success = next(i for i, e in enumerate(events) if e.get('id') == 'first' and e.get('status') == 'SUCCEEDED')
    launch = next(i for i, e in enumerate(events) if e.get('id') == 'second' and e['event'] == 'job_dispatched')
    assert success < launch
    cleanup(s)


def test_same_gpu_without_dependency_waits(environment):
    root, _, s = environment
    queue(s, [job(root, 'first', delay=.15), job(root, 'second')])
    s.tick()
    assert s.state['jobs']['second']['status'] == 'WAITING'
    assert 'reserved' in s.state['jobs']['second']['reason']
    run_until(s, lambda st: st['jobs']['second']['status'] == 'SUCCEEDED')
    cleanup(s)


def test_inputs_and_external_lossless_gate(environment):
    root, _, s = environment
    ready = root / 'checkpoint/config.json'
    exit_file, result = root / 'gate_EXIT.json', root / 'gate.json'
    queue(s, [job(root, 'train', inputs=[str(ready)], requires=[
        dict(exit=str(exit_file), result=str(result), status='LOSSLESS_PASSED')])])
    s.tick()
    assert not s.state['jobs']['train']['dispatched']
    exit_file.write_text('{"exit_code":0}')
    result.write_text('{"status":"LOSSLESS_PASSED"}')
    s.tick()
    assert 'inputs not ready' in s.state['jobs']['train']['reason']
    ready.parent.mkdir();ready.write_text('{}')
    run_until(s, lambda st: st['jobs']['train']['status'] == 'SUCCEEDED')
    cleanup(s)


@pytest.mark.parametrize('exit_code,status', [(1, 'LOSSLESS_PASSED'), (0, 'LOSSLESS_FAILED')])
def test_external_gate_failure_blocks_only_dependents(environment, exit_code, status):
    root, _, s = environment
    exit_file, result = root / 'gate_EXIT.json', root / 'gate.json'
    exit_file.write_text(json.dumps({'exit_code': exit_code}))
    result.write_text(json.dumps({'status': status}))
    queue(s, [job(root, 'blocked', requires=[dict(exit=str(exit_file), result=str(result), status='LOSSLESS_PASSED')]),
              job(root, 'other', gpu='GPU-b')])
    run_until(s, lambda st: st['jobs']['other']['status'] == 'SUCCEEDED')
    assert s.state['jobs']['blocked']['status'] == 'BLOCKED'
    assert not (root / 'jobs/blocked').exists()
    cleanup(s)


def test_failed_job_is_not_retried_and_dependents_freeze(environment):
    root, _, s = environment
    queue(s, [job(root, 'fail', code=3), job(root, 'blocked', depends_on=['fail']),
              job(root, 'independent', gpu='GPU-b')])
    run_until(s, lambda st: st['jobs']['independent']['status'] == 'SUCCEEDED'
              and st['jobs']['blocked']['status'] == 'BLOCKED')
    assert s.state['jobs']['fail']['status'] == 'FAILED'
    for _ in range(3):s.tick()
    dispatched = [json.loads(x)['id'] for x in (s.control / 'events.jsonl').read_text().splitlines()
                  if json.loads(x)['event'] == 'job_dispatched']
    assert dispatched.count('fail') == 1
    cleanup(s)


def test_own_result_status_checked_before_dependency(environment):
    root, _, s = environment
    result = root / 'result.json'
    result.write_text('{"status":"LOSSLESS_FAILED"}')
    queue(s, [job(root, 'gate', result=dict(path=str(result), status='LOSSLESS_PASSED')),
              job(root, 'train', depends_on=['gate'])])
    run_until(s, lambda st: st['jobs']['train']['status'] == 'BLOCKED')
    assert s.state['jobs']['gate']['status'] == 'FAILED'
    cleanup(s)


def test_stop_does_not_terminate_running_and_append_is_supported(environment):
    root, _, s = environment
    first = job(root, 'first', delay=.1)
    queue(s, [first]);s.tick()
    (s.control / 'STOP_SCHEDULER').touch()
    second = job(root, 'second')
    queue(s, [first, second])
    run_until(s, lambda st: st['jobs']['first']['status'] == 'SUCCEEDED')
    assert s.state['dispatch_stopped'] is True
    assert not s.state['jobs']['second']['dispatched']
    (s.control / 'STOP_SCHEDULER').unlink()
    run_until(s, lambda st: st['jobs']['second']['status'] == 'SUCCEEDED')
    cleanup(s)


def test_dispatched_definition_mutation_rejected_but_jobs_observed(environment):
    root, _, s = environment
    first = job(root, 'first', delay=.1)
    queue(s, [first]);s.tick()
    changed = copy.deepcopy(first);changed['argv'][-1] = 'raise SystemExit(7)'
    queue(s, [changed, job(root, 'second', gpu='GPU-b')])
    run_until(s, lambda st: st['jobs']['first']['status'] == 'SUCCEEDED')
    assert s.state['queue_valid'] is False
    assert 'cannot modify dispatched' in s.state['queue_error']
    assert 'second' not in s.state['jobs']
    cleanup(s)


def test_restart_observes_live_process_without_relaunch(environment):
    root, runner, s = environment
    queue(s, [job(root, 'first', delay=.2)]);s.tick()
    restarted = mod.Supervisor(root, runner=runner, allowed_root=root)
    restarted.tick()
    assert restarted.state['jobs']['first']['status'] == 'RUNNING'
    assert restarted.children == {}
    run_until(restarted, lambda st: st['jobs']['first']['status'] == 'SUCCEEDED')
    cleanup(s)


def test_ambiguous_restart_reserves_gpu_and_never_retries(environment):
    root, runner, s = environment
    first, second = job(root, 'first'), job(root, 'second')
    queue(s, [first, second]);assert s.load_queue()
    row = s.state['jobs']['first'];row.update(status='STARTING', dispatched=True)
    s.save()
    restarted = mod.Supervisor(root, runner=runner, allowed_root=root)
    restarted.tick()
    assert restarted.state['jobs']['first']['status'] == 'RECOVERY_REQUIRED'
    assert restarted.state['jobs']['second']['status'] == 'WAITING'
    assert restarted.children == {}


@pytest.mark.parametrize('mutation', ['escape', 'cycle', 'duplicate_gpu', 'unknown_dependency', 'shell_argv'])
def test_invalid_queue_rejected(environment, mutation):
    root, _, s = environment
    first = job(root, 'first')
    if mutation == 'escape':first['job_dir'] = str(root.parent / 'outside')
    if mutation == 'cycle':first['depends_on'] = ['first']
    if mutation == 'duplicate_gpu':first['gpu_uuids'] *= 2
    if mutation == 'unknown_dependency':first['depends_on'] = ['missing']
    if mutation == 'shell_argv':first['argv'] = 'echo unsafe'
    queue(s, [first]);s.tick()
    assert s.state['queue_valid'] is False
    assert not s.children


def test_cli_root_cannot_escape_authorized_run(tmp_path):
    with pytest.raises(ValueError, match='authorized run'):
        mod.Supervisor(tmp_path)


def test_queue_cannot_remove_accepted_job(environment):
    root, _, s = environment
    queue(s, [job(root, 'future', inputs=[str(root / 'missing')])]);s.tick()
    queue(s, []);s.tick()
    assert s.state['queue_valid'] is False
    assert 'may not remove' in s.state['queue_error']


def test_wrapper_failure_before_exit_file_is_terminal(environment):
    root, runner, s = environment
    runner.write_text('raise SystemExit(9)\n')
    queue(s, [job(root, 'occupied_gpu')])
    run_until(s, lambda st: st['jobs']['occupied_gpu']['status'] == 'FAILED')
    assert 'without EXIT.json' in s.state['jobs']['occupied_gpu']['reason']
    s.tick()
    assert len(s.children) == 1
    cleanup(s)


def test_symlink_output_escape_rejected(environment):
    root, _, s = environment
    (root / 'outside_link').symlink_to(root.parent, target_is_directory=True)
    first = job(root, 'first');first['job_dir'] = str(root / 'outside_link/escape')
    queue(s, [first]);s.tick()
    assert s.state['queue_valid'] is False
    assert not s.children


def test_spec_without_recorded_dispatch_requires_manual_recovery(environment):
    root, _, s = environment
    queue(s, [job(root, 'first')])
    (s.control / 'specs').mkdir()
    (s.control / 'specs/first.json').write_text('{}')
    s.tick()
    assert s.state['jobs']['first']['status'] == 'RECOVERY_REQUIRED'
    assert not s.children
