"""The runbook's synthetic reload smoke, pinned by a separately saved pilot gate."""
from pathlib import Path
import json
import sys
import torch
from integration.artifacts import sha256
from training_v1.checkpoint import load_checkpoint

run, approval_path, expected, gate_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
assert sha256(approval_path) == expected
report = json.loads((run / 'REPORT.json').read_text())
gate = json.loads(gate_path.read_text())
assert report['receipt_status'] == 'SUCCESS'
assert report['optimizer_steps'] == 32 and report['semantic_updates'] > 0
assert report['final_manifest_sha256'] == gate['final_manifest_sha256']
model, metadata = load_checkpoint(run / 'final', gate['final_manifest_sha256'],
                                  json.loads(approval_path.read_text()), device='cuda')
assert metadata['status'] == 'PSEUDO_SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED'
with torch.no_grad():
    outputs = model(torch.zeros(1, 3, 65, 97, device='cuda'))
assert tuple(outputs['logits'].shape) == (1, 4, 65, 97)
assert torch.isfinite(outputs['logits']).all()
receipt = dict(status='PASS_RELOAD_SOFTWARE_ONLY', checkpoint_sha256=gate['final_manifest_sha256'],
               architecture=metadata['config']['architecture'],
               semantic_updates=metadata['lineage']['semantic_updates'],
               patient_arrays_read=False, gt_read=False)
(gate_path.parent / 'RELOAD.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
