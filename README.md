# Self-Audit — Candidate Q3D

Candidate Q3D is the active research track. Its authoritative design and evidence limits are in [the Q3D specification](docs/CANDIDATE_Q3D_SPEC.md). Candidate A and Candidate A-3D remain explicit, unchanged comparison tracks.

```bash
python scripts/check_repo.py
python -m pytest -q
python scripts/smoke_q3d.py --out /tmp/self-audit-q3d-smoke
```

After the Q3D train/val manifest hashes and pretrained teacher provenance are verified in the config, use:

```bash
python train.py --config configs/candidate_q3d/v1.json --out "$Q3D_RUN_DIR" --allow-full-run
python evaluate.py --checkpoint "$Q3D_STUDENT_EXPORT" --checkpoint-sha256 "$Q3D_STUDENT_EXPORT_SHA256" --data "$Q3D_EVALUATION_MANIFEST" --data-sha256 "$Q3D_EVALUATION_SHA256" --train-data "$Q3D_TRAIN_MANIFEST" --train-data-sha256 "$Q3D_TRAIN_SHA256" --out "$Q3D_EVALUATION_DIR"
```

For a read-only per-class metrics view, run `python scripts/q3d_watch_metrics.py --plan reports/q3d-resenc-plan-v2-20261112/PLAN.json --once`. It accepts legacy `development_*` event names and prints `VAL` labels; internal keys such as `development`, `teacher_dev_fg_dice`, and `development_epoch` remain unchanged for compatibility.

The smoke uses manufactured data only; patient training, GPU validation, and pretrained-weight provenance are separate evidence requirements.
