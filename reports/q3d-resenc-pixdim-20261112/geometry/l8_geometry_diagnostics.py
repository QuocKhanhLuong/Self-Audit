"""L8-only ground-truth geometry diagnostics; no teacher prediction or training."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.q3d_preflight import _ground_truth_stage4_coverage
from self_audit.candidate_a3d.data import CandidateA3DData
from self_audit.candidate_q3d.alignment import student_grid_phase
from self_audit.candidate_q3d.config import load_config
from self_audit.candidate_q3d.physiology import _physio_options
from self_audit.candidate_q3d.student import CandidateQ3DStudent


def main():
    torch.set_num_threads(4)
    path = Path('/root/mount_data/prepared/ACDC/k8/train.q3d-pixdim.json')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(path.read_text())
    config = load_config('configs/candidate_q3d/v1.json')
    options = _physio_options(config)
    student = CandidateQ3DStudent(config=config)
    phase = student_grid_phase(student)
    coverage = _ground_truth_stage4_coverage(path, sha, phase)
    data = CandidateA3DData(path, sha, synthetic=False)
    grouped = defaultdict(list)
    for key in data.labeled_frame_keys():
        _image, labels, _geometry, metadata = data.load_labeled_frame(key)
        pid = str(metadata['patient'])
        spacing = manifest['geometry']['by_patient'][pid]['spacing_zyx_mm']
        counts = np.bincount(np.asarray(labels).reshape(-1), minlength=4)
        grouped[pid].append({'frame_key': str(key), 'spacing_zyx_mm': spacing,
                             'volumes_ml': (counts * np.prod(spacing) / 1000.0).tolist()})
    rows = []
    for pid, frames in sorted(grouped.items()):
        lv = [frame['volumes_ml'][3] for frame in frames]
        assert len(frames) == 2, 'This diagnostic requires exactly the declared L ED/ES pair'
        ed, es = max(lv), min(lv)
        ef = (ed - es) / ed * 100 if ed > 0 else None
        reasons = []
        if ed < options['minimum_lv_ed_ml']:
            reasons.append('lv_ed_volume_below_10_ml')
        if ed - es <= 0:
            reasons.append('lv_stroke_volume_nonpositive')
        if ef is None or not options['ef_range_percent'][0] <= ef <= options['ef_range_percent'][1]:
            reasons.append('lv_ef_outside_5_to_90_percent')
        rows.append({'patient_id': pid, 'frames': frames, 'lv_ed_ml': ed, 'lv_es_ml': es,
                     'lv_ef_percent': ef, 'volume_ef_guard_reasons': reasons})
    output = {'status': 'PASS' if all(not r['volume_ef_guard_reasons'] for r in rows) else 'GUARD_FAILURE',
              'evidence_status': 'REAL_LABELED_DATA_GEOMETRY_DIAGNOSTIC_NO_TRAINING',
              'scope': 'Only the two supplied GT frames of each L8 patient; no full-cine GT, no U/DEV/TEST masks',
              'manifest': str(path), 'manifest_sha256': sha, 'student_phase': phase,
              'stage4_class_coverage': coverage, 'physiology_options': options, 'labeled_patients': rows,
              'nonclaims': ['No teacher physiology weight estimated', 'No clinical normality or efficacy claim', 'No threshold modified']}
    out = Path('/root/mount_data/receipts/q3d-pixdim-20261112/L8_GEOMETRY_DIAGNOSTICS.json')
    with out.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
