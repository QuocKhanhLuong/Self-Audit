#!/usr/bin/env python3
"""Explicit, single-writer recovery and ordered exploratory Q3D campaign.

Run from /root/Self-Audit with --source-sha HEAD --plan-sha SHA256 --reason
REASON. This wrapper never changes a recipe or relaxes the campaign runner's
no-implicit-resume guard. Quality measurements remain failed when waived.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import q3d_campaign as campaign
from self_audit.candidate_q3d import campaign_runtime as runtime
from self_audit.candidate_q3d import engine
from self_audit.candidate_q3d.config import load_config

DEFAULT_BASE = Path('/root/mount_data/runs/q3d-resenc-pixdim-20261112')
ORDER = (('v1', 8), ('v1', 5), ('v1', 10), ('v1u', 8), ('c1', 8), ('c0', 8))
WAIVER = '--q3d-exploratory-quality-waiver'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def queue_lock(base):
    """The inode is never unlinked; a second queue cannot acquire a new lock."""
    with (Path(base) / 'exploratory-queue.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another exploratory queue holds the lock') from exc
        # Children inherit the open-file-description lock until they exit even
        # if the wrapper is killed. Popen receives this fd explicitly.
        yield handle.fileno()


def assert_children_absent(base, output=None, process_reader=None):
    """A successful, complete ps snapshot is required; errors are not absence."""
    reader = process_reader or (lambda: subprocess.check_output(
        ['ps', '-ww', '-eo', 'pid=,args='], text=True))
    snapshot = reader()
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise RuntimeError('cannot positively establish child absence from ps')
    needles = [str(Path(base).resolve())]
    if output is not None:
        needles.append(str(Path(output).resolve()))
    for line in snapshot.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise RuntimeError('malformed ps snapshot; child liveness unknown')
        if int(parts[0]) == os.getpid():
            continue
        command = parts[1]
        try:
            scripts = {Path(token).name for token in shlex.split(command)}
        except ValueError as exc:
            raise RuntimeError('unparseable ps argv; child liveness unknown') from exc
        # Legacy supervisors have no flock and some use a hard-coded base,
        # hence their argv need not contain the campaign directory at all.
        queue_writers = {'run_ordered.py', 'resume_queue.py', 'q3d_resume_queue.py', 'q3d_resume_epoch_queue.py', 'q3d_campaign.py'}
        if scripts & queue_writers or (scripts & {'train.py', 'q3d_profile_gpu.py'} and any(
                needle in command for needle in needles)):
            raise RuntimeError(f'training/profile/queue process still present: {line.strip()}')
    return {'observed_utc': now(), 'ps_sha256': hashlib.sha256(snapshot.encode()).hexdigest()}


def with_waiver(command, reason):
    command = list(command)
    if WAIVER in command:
        index = command.index(WAIVER)
        if command.count(WAIVER) != 1 or index + 1 >= len(command) or command[index + 1] != reason:
            raise ValueError('stored command requires the same explicit waiver reason')
        return command
    return command + [WAIVER, reason]


def without_waiver(command):
    command = list(command)
    if WAIVER in command:
        index = command.index(WAIVER)
        if command.count(WAIVER) != 1 or index + 1 >= len(command):
            raise ValueError('malformed stored waiver arguments')
        del command[index:index + 2]
    return command


def align_metrics(path, checkpoint, audit):
    """Keep exactly the committed JSONL prefix, preserving all discarded bytes."""
    path = Path(path)
    expected = checkpoint.get('metrics')
    if not isinstance(expected, list):
        raise ValueError('checkpoint has no committed metrics list')
    raw = path.read_bytes() if path.exists() else b''
    lines = raw.splitlines(keepends=True)
    if len(lines) < len(expected):
        raise ValueError('metrics stream is shorter than checkpoint metrics')
    for line, metric in zip(lines, expected):
        if json.loads(line) != metric:
            raise ValueError('metrics prefix differs from checkpoint metrics')
    prefix = b''.join(lines[:len(expected)])
    suffix = raw[len(prefix):]
    if suffix:
        (audit / 'metrics.uncommitted.jsonl').write_bytes(suffix)
        temporary = path.with_name('.metrics.recovery.' + uuid.uuid4().hex)
        temporary.write_bytes(prefix)
        os.replace(temporary, path)
    return {'committed_rows': len(expected), 'uncommitted_bytes': len(suffix)}


class ResumeQueue:
    def __init__(self, base, source_sha, plan_sha, reason, *, root=ROOT,
                 process_reader=None, child_runner=None):
        self.base, self.root = Path(base).resolve(), Path(root).resolve()
        self.plan_path = self.base / 'campaign-v2/PLAN.json'
        self.state_path = self.plan_path.parent / 'RUN_STATE.json'
        self.source_sha, self.plan_sha = source_sha, plan_sha
        self.policy = runtime.exploratory_waiver_policy(reason)
        if self.policy is None:
            raise ValueError('an explicit nonempty --reason is required')
        self.reason = self.policy['reason']
        self.process_reader, self.child_runner = process_reader, child_runner
        self.lock_fd = None
        self.log_lock = threading.Lock()
        self.stop = threading.Event()
        self.offsets = {}
        self.current = {}

    def log(self, message):
        line = now() + ' ' + message
        with self.log_lock:
            print(line, flush=True)
            with (self.base / 'launch.log').open('a') as stream:
                stream.write(line + '\n')

    def status(self, status, **fields):
        self.current.update(fields)
        atomic_json(self.base / 'QUEUE_STATUS.json', {
            **self.current, 'status': status, 'updated_utc': now(),
            'code_commit': self.source_sha, 'plan_sha256': self.plan_sha,
            'exploratory': True, 'exploratory_quality_waiver': self.policy,
        })

    def audit_dir(self, category):
        path = self.base / 'exploratory' / category / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid.uuid4().hex[:8])
        path.mkdir(parents=True, exist_ok=False)
        return path

    def assert_pins(self):
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip()
        if actual != self.source_sha or sha(self.plan_path) != self.plan_sha:
            raise ValueError('source HEAD or immutable plan SHA mismatch')

    def state(self):
        return campaign._read_state(self.state_path, self.plan, self.plan_sha)

    def absence(self, output=None):
        return assert_children_absent(self.base, output, self.process_reader)

    def runner(self, command, **kwargs):
        """Retain stdout/stderr while the child runs, including wrapper crashes."""
        self.assert_pins()
        self.absence()
        if '--out' in command:
            output = command[command.index('--out') + 1]
            job = next((j for j in self.plan['stage_jobs'] if j['output'] == output), None)
            if job is not None:
                self.status('RUNNING', current_stage=job['stage'], current_job=job['id'])
        audit = self.audit_dir('attempts')
        atomic_json(audit / 'COMMAND.json', {'command': list(command), 'cwd': str(self.root),
                    'source_sha256': self.source_sha, 'plan_sha256': self.plan_sha,
                    'started_utc': now(), 'exploratory': True, 'exploratory_quality_waiver': self.policy})
        self.log('TRAIN_COMMAND ' + shlex.join(command) + ' transcript=' + str(audit))
        if self.child_runner is not None:
            result = self.child_runner(command, **kwargs)
            (audit / 'stdout.txt').write_text(result.stdout or '')
            (audit / 'stderr.txt').write_text(result.stderr or '')
        else:
            with (audit / 'stdout.txt').open('w') as stdout, (audit / 'stderr.txt').open('w') as stderr:
                child = subprocess.Popen(command, cwd=self.root, stdout=stdout, stderr=stderr,
                                         text=True, pass_fds=(self.lock_fd,) if self.lock_fd is not None else ())
                atomic_json(audit / 'CHILD.json', {'pid': child.pid, 'started_utc': now()})
                code = child.wait()
            result = subprocess.CompletedProcess(command, code, (audit / 'stdout.txt').read_text(),
                                                 (audit / 'stderr.txt').read_text())
        atomic_json(audit / 'EXIT.json', {'returncode': result.returncode, 'exited_utc': now()})
        return result

    def verify_stored_command(self, job, record, state):
        command = record.get('command')
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError('interrupted job has no stored argv')
        experiment = dict(next(e for e in self.plan['experiments'] if e['id'] == job['experiment_id']))
        experiment['output'] = job['output']
        inputs, pins = {}, {}
        for stage, reference in job.get('stage_artifact_inputs', {}).items():
            producer = self.jobs[reference['producer_job_id']]
            previous = state['jobs'][producer['id']]
            if previous['status'] != 'COMPLETED':
                raise ValueError('recovery dependency is not completed')
            artifact = runtime._verified_output(producer, pinned_sha=previous['artifact']['sha256'])
            runtime._require_waiver_for(runtime.exploratory_marker(artifact), self.policy, producer['id'])
            if Path(reference['path']).resolve() != Path(artifact['path']).resolve():
                raise ValueError('recovery dependency path differs from immutable plan')
            inputs[stage], pins[stage] = artifact['path'], artifact['sha256']
        expected = campaign._job_command(experiment, config_path=Path(job['config_path']),
                    train_ref=job['train_manifest'], dev_ref=job['development_manifest'],
                    device='cuda', synthetic=False, stage=job['stage'], stage_artifacts=inputs)
        for stage, digest in sorted(pins.items()):
            expected += ['--q3d-stage-artifact-sha', f'{stage}={digest}']
        # Preserve the original Python executable, not the wrapper interpreter.
        expected[0] = command[0]
        if without_waiver(command) != expected:
            raise ValueError('stored command differs from immutable planned command')
        if record.get('input_manifest_sha256', {}) != pins:
            raise ValueError('stored input artifact pins differ from verified dependencies')
        return with_waiver(command, self.reason)

    def checkpoint(self, job):
        config = load_config(job['config_path'], real_run=True)
        if config['runtime']['synthetic']:
            raise ValueError('recovery is real-only')
        identity = engine._data_identity_without_labels(engine._load_data(config, None))
        # This actual engine loader checks sidecar, weights_only, schema, config
        # and live cohort identity. The child subsequently restores every RNG,
        # optimizer, sampler and model through the actual engine resume path.
        state = engine._load_checkpoint(Path(job['output']) / 'latest.pt',
                   config_sha256=engine._object_sha256(config), data_identity=identity)
        runtime._require_waiver_for(runtime.exploratory_marker(state), self.policy, job['id'])
        if state.get('synthetic_gate_override_requested') or state.get('synthetic_gate_override_used'):
            raise ValueError('real recovery rejected a synthetic gate override')
        return state

    def verify_receipt(self, job, receipt, checkpoint):
        latest = Path(job['output']) / 'latest.pt'
        if (not isinstance(receipt, dict) or receipt.get('status') != 'COMPLETED'
                or runtime._receipt_is_synthetic(receipt)
                or receipt.get('runtime_synthetic') is not False
                or receipt.get('evidence_status') != 'TRAIN_EXECUTION'
                or receipt.get('active_stages') != [job['stage']]
                or job['stage'] not in receipt.get('completed_stages', [])
                or receipt.get('completed_stages') != checkpoint.get('completed_stages')
                or receipt.get('config_sha256') != checkpoint['config_sha256']
                or receipt.get('data_identity') != checkpoint['data_identity']
                or receipt.get('completed_updates') != checkpoint['completed_updates_total']
                or checkpoint.get('stage') is not None or checkpoint.get('stage_cursor') != 0
                or Path(receipt.get('checkpoint', '')).resolve() != latest.resolve()
                or receipt.get('checkpoint_sha256') != sha(latest)
                or receipt.get('synthetic_gate_override_used')):
            raise ValueError('incompatible real completed receipt/checkpoint')
        runtime._require_waiver_for(runtime.exploratory_marker(receipt), self.policy, job['id'])
        if runtime.exploratory_marker(checkpoint) and not runtime.exploratory_marker(receipt):
            raise ValueError('exploratory checkpoint cannot be relabeled as strict')
        if receipt.get('tokenizer_gate') != checkpoint.get('gate'):
            raise ValueError('receipt changed the measured tokenizer gate')
        artifact = runtime._verified_output(job) if job.get('artifact_output') else None
        if artifact is not None:
            runtime._require_waiver_for(runtime.exploratory_marker(artifact), self.policy, job['id'])
        return artifact

    def existing_receipts(self, job, record):
        if isinstance(record.get('receipt'), dict):
            yield record['receipt'], 'RUN_STATE.receipt'
        # A child can exit after writing stdout but before EXIT.json or RUN_STATE
        # is written. Positive ps absence plus full receipt verification suffices.
        attempts = self.base / 'exploratory/attempts'
        for directory in sorted(attempts.glob('*'), reverse=True):
            metadata = directory / 'COMMAND.json'
            stdout = directory / 'stdout.txt'
            if not metadata.exists() or not stdout.exists():
                continue
            command = read(metadata).get('command', [])
            if without_waiver(command) != without_waiver(record['command']):
                continue
            try:
                receipt = json.loads(stdout.read_text())
            except (ValueError, UnicodeError):
                continue
            yield receipt, str(stdout)

    def recover(self, job):
        state = self.state()
        record = state['jobs'][job['id']]
        if record['status'] not in {'FAILED', 'RUNNING'}:
            return
        self.assert_pins()
        absence = self.absence(job['output'])
        command = self.verify_stored_command(job, record, state)
        latest = Path(job['output']) / 'latest.pt'
        if not latest.is_file():
            raise ValueError('interrupted output has no latest.pt; refusing a from-scratch replay')
        audit = self.audit_dir('recovery')
        for path in (self.state_path, self.base / 'QUEUE_STATUS.json', latest.with_suffix('.pt.sha256'),
                     latest.parent / 'metrics.jsonl'):
            if path.is_file():
                shutil.copy2(path, audit / path.name)
        os.link(latest, audit / 'latest.pt')
        original_sha = sha(latest)
        checkpoint = self.checkpoint(job)
        metrics = align_metrics(latest.parent / 'metrics.jsonl', checkpoint, audit)
        self.offsets[latest.parent / 'metrics.jsonl'] = (latest.parent / 'metrics.jsonl').stat().st_size if (latest.parent / 'metrics.jsonl').exists() else 0
        prepared = {'job_id': job['id'], 'original_command': record['command'], 'resume_command': command,
                    'checkpoint_sha256': original_sha, 'stage_cursor': checkpoint.get('stage_cursor'),
                    'absence': absence, 'metrics': metrics, 'exploratory': True,
                    'exploratory_quality_waiver': self.policy}
        atomic_json(audit / 'PREPARED.json', prepared)
        receipt, artifact, reused = None, None, None
        for candidate, source in self.existing_receipts(job, record):
            try:
                artifact = self.verify_receipt(job, candidate, checkpoint)
            except (ValueError, FileNotFoundError):
                continue
            receipt, reused = candidate, source
            break
        if receipt is None:
            self.status('RUNNING', current_stage=job['stage'], current_job=job['id'], recovery_audit=str(audit))
            result = self.runner(command, cwd=self.root, check=False, capture_output=True, text=True)
            if result.returncode:
                atomic_json(audit / 'FAILED.json', {'returncode': result.returncode, 'exploratory': True})
                raise RuntimeError(f'resumed stage failed ({result.returncode}): {job["id"]}')
            receipt = json.loads(result.stdout)
            checkpoint = self.checkpoint(job)
            artifact = self.verify_receipt(job, receipt, checkpoint)
        event = {**prepared, 'audit_dir': str(audit), 'completed_utc': now(), 'reused_receipt': reused,
                 'completed_checkpoint_sha256': receipt['checkpoint_sha256']}
        # Only this job changes; the campaign runner computes experiment status.
        record.update(status='COMPLETED', exit_code=0, receipt=copy.deepcopy(receipt),
                      checkpoint={'path': str(latest), 'sha256': receipt['checkpoint_sha256']})
        record.setdefault('recovery_history', []).append(event)
        if artifact is not None:
            record['artifact'] = artifact
        if runtime.exploratory_marker(receipt):
            record.update(exploratory=True, exploratory_quality_waiver=self.policy)
        atomic_json(audit / 'COMPLETED.json', {'receipt': receipt, 'artifact': artifact, **event})
        atomic_json(self.state_path, state)
        self.log('RECOVERED ' + job['id'] + ' exploratory=' + str(receipt.get('exploratory', False)))

    def poll_metrics(self):
        for job in self.plan['stage_jobs']:
            path = Path(job['output']) / 'metrics.jsonl'
            if not path.is_file():
                continue
            with path.open('rb') as stream:
                offset = self.offsets.get(path, 0)
                if path.stat().st_size < offset:
                    # A recovery truncated an archived uncommitted suffix.
                    # Never replay old rows just because the file shrank.
                    self.offsets[path] = path.stat().st_size
                    continue
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b'\n'):
                        break
                    self.log('METRIC ' + job['id'] + ' ' + line.decode().strip())
                    self.offsets[path] = stream.tell()

    def stream_metrics(self):
        while not self.stop.wait(2):
            self.poll_metrics()

    def experiment(self, setting, k):
        matches = [e for e in self.plan['experiments'] if e['setting'] == setting and e['k'] == k]
        if len(matches) != 1:
            raise ValueError('ordered queue requires exactly one experiment per setting/K')
        return matches[0]

    def validate_completed(self, experiment):
        state = self.state()
        producer_ids = set(experiment['stage_job_ids']) | set(
            experiment.get('stage_artifact_producers', {}).values())
        for job_id in producer_ids:
            job, record = self.jobs[job_id], state['jobs'][job_id]
            if record['status'] != 'COMPLETED':
                raise ValueError('completed experiment has an incomplete producer')
            runtime._require_waiver_for(runtime._job_exploratory_policy(job_id, state),
                                        self.policy, job_id)
            if job.get('artifact_output'):
                artifact = runtime._verified_output(job, pinned_sha=record['artifact']['sha256'])
                runtime._require_waiver_for(runtime.exploratory_marker(artifact), self.policy, job_id)
            if job['stage'] == 'J':
                self.verify_receipt(job, record.get('receipt'), self.checkpoint(job))

    def train(self, setting, k, through='J'):
        experiment = self.experiment(setting, k)
        if self.state()['experiments'][experiment['id']]['status'] == 'COMPLETED':
            self.validate_completed(experiment)
            self.log(f'SKIP_COMPLETED setting={setting} k={k}')
            return
        self.status('RUNNING', setting=setting, k=k, through_stage=through, current_stage=None, current_job=None)
        visited = set()
        def recover_dependencies(job_id):
            if job_id in visited:
                return
            visited.add(job_id)
            job = self.jobs[job_id]
            for upstream in job['dependencies']:
                recover_dependencies(upstream)
            if 'TQPJ'.index(job['stage']) <= 'TQPJ'.index(through):
                self.recover(job)
        for job_id in experiment['stage_job_ids']:
            recover_dependencies(job_id)
        result = campaign.execute_campaign(self.plan_path, execute=True, setting=setting, k=(k,),
                    through_stage=through, device='cuda', runner=self.runner,
                    exploratory_quality_waiver=self.reason)
        if result['status'] not in ({'PARTIAL', 'COMPLETED'} if through == 'P' else {'COMPLETED'}):
            raise RuntimeError('unexpected campaign completion status')
        self.log('CAMPAIGN_RESULT ' + json.dumps(result))

    def validate_profile(self, path, stage, experiment, artifacts=None):
        report = read(path)
        if (report.get('status') != 'COMPLETED' or report.get('evidence_status') != 'GPU_PROFILE'
                or report.get('execution_mode') != 'REAL_CUDA_PROFILE'
                or report.get('stage_results', {}).get(stage, {}).get('measured_updates') != 20
                or report.get('config_sha256') != experiment['config_sha256']):
            raise ValueError(f'{stage} profile does not verify 20 real updates of the pinned recipe')
        runtime._require_waiver_for(runtime.exploratory_marker(report), self.policy, f'{stage} profile')
        if artifacts is not None:
            if not runtime.exploratory_marker(report):
                raise ValueError('J profile must be explicitly exploratory')
            bindings = report.get('stage_artifact_inputs', {})
            for key, artifact in artifacts.items():
                binding = bindings.get(key, {})
                if (binding.get('manifest_sha256') != artifact['sha256'] or
                        Path(binding.get('manifest_path', '')).resolve() != Path(artifact['path']).resolve()):
                    raise ValueError('J profile uses different production artifacts')
        return report

    def profile_j(self, experiment):
        state = self.state()
        artifacts = {}
        for stage in ('T', 'Q', 'P'):
            producer = experiment['stage_artifact_producers'][stage]
            previous = state['jobs'][producer]
            if previous['status'] != 'COMPLETED':
                raise ValueError('J profile requires completed T/Q/P')
            artifacts[stage] = runtime._verified_output(self.jobs[producer], pinned_sha=previous['artifact']['sha256'])
        target = self.base / 'exploratory/GPU_PROFILE_J.json'
        if target.exists():
            # A failed bounded profile may be retried, preserving its original
            # bytes first. A different waiver reason is never an implicit retry.
            try:
                existing = read(target)
            except ValueError:
                existing = {}
            runtime._require_waiver_for(runtime.exploratory_marker(existing), self.policy, 'existing J profile')
            try:
                self.validate_profile(target, 'J', experiment, artifacts)
            except (ValueError, KeyError) as exc:
                audit = self.audit_dir('profile-history')
                os.link(target, audit / target.name)
                atomic_json(audit / 'RETRY.json', {'reason': str(exc), 'original_sha256': sha(target),
                            'exploratory': True, 'exploratory_quality_waiver': self.policy})
                target.unlink()
            else:
                self.log('J_PROFILE_REUSE ' + str(target))
                return
        args = read(self.base / 'profile_args.json')
        if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
            raise ValueError('profile_args.json must contain the original argv')
        for flag, value in (('--stage', 'J'), ('--config', experiment['config_path']), ('--report', str(target))):
            if args.count(flag) != 1:
                raise ValueError('profile argv missing or repeating ' + flag)
            args[args.index(flag) + 1] = value
        # Preserve the original T profile argv; do not accumulate stale inputs.
        for flag in ('--stage-artifact', '--stage-artifact-sha'):
            while flag in args:
                index = args.index(flag)
                del args[index:index + 2]
        for stage, artifact in artifacts.items():
            args += ['--stage-artifact', stage + '=' + artifact['path'],
                     '--stage-artifact-sha', stage + '=' + artifact['sha256']]
        args = with_waiver(args, self.reason)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.status('RUNNING', setting='v1', k=8, current_stage='J_PROFILE', current_job=None)
        result = self.runner(args, cwd=self.root, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('J profile failed; immutable report retained')
        self.validate_profile(target, 'J', experiment, artifacts)

    def run(self):
        with queue_lock(self.base) as self.lock_fd:
            # Do not rewrite QUEUE_STATUS if source identity or live-child checks
            # fail: another (older) supervisor may still own the active job.
            self.assert_pins()
            self.absence()
            self.plan = campaign.validate_campaign_plan(self.plan_path)
            if self.plan['selection']['synthetic'] or any(j['readiness'] != 'READY' for j in self.plan['stage_jobs']):
                raise ValueError('ordered exploratory queue requires a READY real plan')
            self.jobs = {j['id']: j for j in self.plan['stage_jobs']}
            profile_args = read(self.base / 'profile_args.json')
            t_report = Path(profile_args[profile_args.index('--report') + 1])
            self.validate_profile(t_report, 'T', self.experiment('v1', 8))
            # Start at current EOF, never replay the whole historical stream.
            for job in self.plan['stage_jobs']:
                path = Path(job['output']) / 'metrics.jsonl'
                self.offsets[path] = path.stat().st_size if path.exists() else 0
            thread = threading.Thread(target=self.stream_metrics, daemon=True)
            thread.start()
            try:
                self.train('v1', 8, 'P')
                self.profile_j(self.experiment('v1', 8))
                for setting, k in ORDER:
                    self.train(setting, k)
                self.status('COMPLETED', current_stage=None, current_job=None)
                self.log('QUEUE_COMPLETED EXPLORATORY')
            except Exception as exc:
                self.status('FAILED', error=str(exc))
                self.log('QUEUE_FAILED ' + str(exc))
                raise
            finally:
                self.stop.set()
                thread.join()
                self.poll_metrics()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args(argv)
    ResumeQueue(args.base, args.source_sha, args.plan_sha, args.reason).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
