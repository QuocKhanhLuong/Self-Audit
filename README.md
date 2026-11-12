# Self-Audit — Candidate Q3D

Candidate Q3D is the active research track. Its authoritative design and evidence limits are in [the Q3D specification](docs/CANDIDATE_Q3D_SPEC.md). Candidate A and Candidate A-3D remain explicit, unchanged comparison tracks.

```bash
python scripts/check_repo.py
python -m pytest -q
python scripts/smoke_q3d.py --out /tmp/self-audit-q3d-smoke
```

After the Q3D TRAIN/development manifest hashes and pretrained teacher provenance are verified in the config, use:

```bash
python train.py --config configs/candidate_q3d/v1.json --out "$Q3D_RUN_DIR" --allow-full-run
python evaluate.py --checkpoint "$Q3D_STUDENT_EXPORT" --checkpoint-sha256 "$Q3D_STUDENT_EXPORT_SHA256" --data "$Q3D_EVALUATION_MANIFEST" --data-sha256 "$Q3D_EVALUATION_SHA256" --train-data "$Q3D_TRAIN_MANIFEST" --train-data-sha256 "$Q3D_TRAIN_SHA256" --out "$Q3D_EVALUATION_DIR"
```

The smoke uses manufactured data only; patient training, GPU validation, and pretrained-weight provenance are separate evidence requirements.
