import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

source = Path('/workspace/Self-Audit-cptu-launch-20261031')
out = Path('/workspace/cptu-launch-20261031')
commit = '786511b47ce0a9aa6da8e166bafdfe52126b8838'
sys.path.insert(0, str(source))
from scripts.continue_b8_transfer_pilot import build_backup_deferred_receipt
from self_audit.compact_teacher import validate_reproduction_receipt, verify_reproduction_source
from self_audit.transfer_preflight import validate_locked_protocol

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip() == commit
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
assert not (out/'LOCKED_PLAN.json').exists(), 'Existing launch must be inspected; do not duplicate arms'
old_path = Path('/workspace/b8-repro-setup-20261031/LOCKED_CPTU_CONTINUATION_3fe1bb6.json')
old = json.loads(old_path.read_text())
for pin in old['pinned_files']:
    assert sha(pin['path']) == pin['sha256'], pin['path']
plan = dict(old)
plan.update(out=str(out), source_commit=commit,
            base_protocol=str(source/'configs/transfer_utility/protocol.json'),
            matrix=str(source/'configs/transfer_utility/matrix.json'),
            supersedes_unstarted_continuation_plan_sha256=sha(old_path))
plan.pop('backup_retry', None)
assert sha(plan['base_protocol']) == sha(old['base_protocol'])
assert sha(plan['matrix']) == sha(old['matrix'])
auth = out/'EXPLORATORY_TEACHER_AUTHORIZATION.json'
assert sha(auth) == 'dbcea6c02557c39ef7e411b8b6bd54aff6185f0808e03753c407b5a7237a53bd'
plan['execution_mode'] = 'EXPLORATORY_TEACHER_NO_GO_WITH_EXPLICIT_AUTHORIZATION'
plan['exploratory_authorization'] = {'path':str(auth),'sha256':sha(auth)}
paths = [plan['data'], plan['val'], plan['base_protocol'], plan['matrix'],
         '/workspace/b8-repro-setup-20261031/LOCKED_B8_PLAN.json',
         str(source/'scripts/continue_b8_transfer_pilot.py'),
         plan['verification_completed_before_launch'],str(auth),
         str(out/'BACKUP_DEFERRED_AUTHORIZATION.json'),str(out/'EVAL_STATUS.json')]
plan['pinned_files'] = [{'path':p,'sha256':sha(p)} for p in paths]
plan_path = Path('/workspace/b8-repro-setup-20261031/LOCKED_CPTU_EXPLORATORY_786511b.json')
if plan_path.exists():
    assert json.loads(plan_path.read_text()) == plan
else:
    write(plan_path,plan)
receipt = build_backup_deferred_receipt(plan,out,exploratory_authorization_path=auth,
                                      exploratory_authorization_sha256=sha(auth))
identity = validate_reproduction_receipt(receipt)
verify_reproduction_source(plan['teacher_source'], expected_commit=identity['source_commit'],
                           expected_closure=identity['source_sha256'])
assert identity['quality_gate'] == 'NO_GO'
assert identity['exploratory_authorized'] is True
assert identity['remote_drive_verified'] is False
assert identity['verified'] is True
assert not validate_locked_protocol(json.loads(Path(plan['base_protocol']).read_text()))
argv = [plan['python'],str(source/'scripts/continue_b8_transfer_pilot.py'),
        '--plan',str(plan_path),'--plan-sha256',sha(plan_path),
        '--allow-backup-deferred','--allow-exploratory-teacher',
        '--exploratory-authorization',str(auth),'--exploratory-authorization-sha256',sha(auth)]
write(out/'LAUNCH_COMMAND.json',{'argv':argv,'cwd':str(source),'source_commit':commit,
                              'plan_sha256':sha(plan_path),'tmux':'cptu_exploratory_20261031'})
write(out/'REAL_TEACHER_PREFLIGHT.json',{'status':'PASS','time':time.time(),'source_commit':commit,
      'quality_gate':'NO_GO','exploratory_authorized':True,'backup_status':'UPLOAD_PENDING',
      'checkpoint_payload_sha256':identity['checkpoint_payload_sha256'],
      'raw_student_sha256':identity['raw_student_sha256'],'plan_sha256':sha(plan_path),
      'checks':['locked historical plan pins','actual checkpoint and raw state tensor equality',
                'label lineage','teacher source closure','evaluation recipe/frozen predictions/cohort',
                'explicit exploratory authorization','unchanged base protocol and matrix']})
print(json.dumps({'status':'REAL_TEACHER_PREFLIGHT_PASS','plan':str(plan_path),
                  'plan_sha256':sha(plan_path),'argv':argv}),flush=True)
