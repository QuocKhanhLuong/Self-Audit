"""Verify and copy the byte-preserved reference into a NEW isolated directory.

No model weights, patient data, repository changes, or downloads are required.
--run-checks runs manufactured CPU tests and a random full-size model smoke.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / 'reports/layer_level_upgrade_20261015/reference'
MANIFEST_SHA256 = '3f442aa35872bea5db00eb8880c002b1049a1e53da2a88accfbf8e6e5cec2576'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_payloads() -> dict[str, bytes]:
    manifest_bytes = (REFERENCE / 'MANIFEST.json').read_bytes()
    if sha(manifest_bytes) != MANIFEST_SHA256:
        raise ValueError('reference manifest hash mismatch')
    manifest = json.loads(manifest_bytes)
    payloads = {'MANIFEST.json': manifest_bytes}
    stored = {'MANIFEST.json'}
    for name, entry in manifest['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('unsafe reference path')
        archive_name = name + '.txt' if name == 'tests/test_operators.py' else name
        p = REFERENCE / archive_name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(REFERENCE.resolve()):
            raise ValueError('invalid reference file: ' + name)
        data = p.read_bytes()
        if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
            raise ValueError('reference payload hash mismatch: ' + name)
        payloads[name] = data
        stored.add(archive_name)
    actual = {str(p.relative_to(REFERENCE)) for p in REFERENCE.rglob('*') if p.is_file()}
    if actual != stored:
        raise ValueError('unexpected or missing reference payload')
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='NEW directory outside this Git checkout')
    parser.add_argument('--run-checks', action='store_true')
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists() or out.is_relative_to(REPO.resolve()) or REPO.resolve().is_relative_to(out):
        raise ValueError('destination must be fresh and outside the repository')
    payloads = verified_payloads()  # Verify every byte before creating outputs.
    source, results = out/'source', out/'results'
    source.mkdir(parents=True, exist_ok=False)
    results.mkdir()
    for name, data in payloads.items():
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    receipt = dict(schema='self_audit.layer_reference_copy.v1',
                   status='REFERENCE_BYTES_VERIFIED_AND_COPIED',
                   reference_manifest_sha256=MANIFEST_SHA256, files=len(payloads),
                   tests_executed=False, patient_data_read=False,
                   pretrained_weights_loaded=False, commands=[])
    receipt_path = out/'RECEIPT.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    if args.run_checks:
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(source),
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        commands = [
            ('pytest', [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                        '-c', str(source/'pytest.ini'), '--rootdir='+str(source),
                        '--confcutdir='+str(source), str(source/'tests'),
                        '--junitxml='+str(results/'TEST_RESULTS.xml')]),
            ('full_model', [sys.executable, str(source/'verify_full_model.py')]),
            ('probes', [sys.executable, str(source/'probe_report.py')]),
        ]
        for label, command in commands:
            with (results/(label+'.log')).open('w') as log:
                run = subprocess.run(command, cwd=results, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            receipt['commands'].append(dict(name=label, argv=command, exit_code=run.returncode))
            receipt['status'] = 'CHECK_FAILED' if run.returncode else 'CHECKS_RUNNING'
            receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
            if run.returncode:
                print('CHECK_FAILED: '+str(results/(label+'.log')))
                return run.returncode
        receipt.update(status='PASS_REFERENCE_CPU_CHECKS_NOT_PRETRAINED_NOT_PATIENT_TRAINING', tests_executed=True)
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
