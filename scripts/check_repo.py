#!/usr/bin/env python3
"""Reject ambiguous runtime layouts; never delete user data or active jobs."""
from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[1]
active=json.loads((root/'ACTIVE_PIPELINE.json').read_text())
assert active['schema']=='self_audit.active_pipeline.v2' and active['entrypoint']=='train.py'
for rel in ['src/self_audit_guidance','src/self_audit_pseudolabel','experiments/layer_guided_student_v1']:
    if (root/rel).exists():
        raise SystemExit(f'LEGACY_LAYOUT_PRESENT: {rel}. Use a fresh worktree; do not git-clean datasets/runs.')
for rel in ['self_audit/engine.py','self_audit/control.py','self_audit/decoder_control.py','docs/METHOD.md']:
    if not (root/rel).is_file():raise SystemExit(f'MISSING_CANONICAL_FILE: {rel}')
print(json.dumps({'status':'PASS','architecture':active['architecture'],'supervision':active['supervision']}))
