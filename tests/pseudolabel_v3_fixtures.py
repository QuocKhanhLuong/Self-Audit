"""Explicitly synthetic artifacts for consumer contract tests; no accuracy evidence."""
import json
from pathlib import Path
import numpy as np
from self_audit_pseudolabel.freeze import export_pseudo_npz, write_freeze_manifest


def seal_test_teacher(root,entries,cfg):
    from self_audit_pseudolabel.checkpoint import source_identity
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/pseudolabel_v3.json').read_text())
    config['deployment'].update(cfg.get('resolved_config',{}).get('deployment',{}))
    cfg.update(artifact_role='teacher_freeze',manual_mask_input=False,scribble_input=False,
               resolved_config=config,source_sha256=source_identity(),synthetic_fixture=True)
    (root/'teacher.pt').write_bytes(b'SYNTHETIC CONSUMER CONTRACT FIXTURE, NOT A TRAINED TEACHER')
    (root/'train_metrics.json').write_text('[]')
    (root/'split.json').write_text(json.dumps(cfg['split_patients']))
    records={r['patient_id']:r for r in cfg['image_records']}
    for entry in entries:
        path=root/entry['path']; record=records[entry['patient_id']]
        with np.load(path,allow_pickle=False) as data:
            label=data['pseudo_label'].copy();valid=data['valid'].copy();soft=data['soft_label'].copy()
        path.unlink()
        metadata={k:record[k] for k in ('patient_id','dataset','split','image_sha256','affine','axis_order','spatial_unit')}
        metadata.update(t=entry['t'],z=entry['z'])
        entry['sha256']=export_pseudo_npz(path,pseudo_label=label,valid=valid,soft_label=soft,metadata=metadata)
        counts=[int(((label==c)&valid.astype(bool)).sum()) for c in range(4)]
        entry.update(class_pixels=counts,valid_foreground=sum(counts[1:]),valid_fraction=float(valid.mean()))
    return write_freeze_manifest(root,entries,cfg)
