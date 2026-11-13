# Self-Audit — Candidate FUSE

Candidate FUSE is the active research track: a frozen ResEnc teacher supplies gated features to a native-volume UNETR++-lite student. The [FUSE specification](docs/CANDIDATE_FUSE_SPEC.md) defines the data, losses, epochs and deployment rule. It uses the labels of **all TRAIN80 patients, not a few-label protocol**. Candidate A, A-3D and Q3D remain unchanged historical comparison tracks.

```bash
python scripts/check_repo.py
python -m pytest -q tests/test_fuse_*.py
python scripts/smoke_fuse.py --out /tmp/self-audit-fuse-smoke
```

After creating the FUSE train/val/test manifests and pinning their hashes, use:

```bash
python train.py --config "$FUSE_CONFIG" --out "$FUSE_RUN_DIR" --device cuda --fuse-stage all
python scripts/fuse_campaign.py --config "$FUSE_CONFIG" --out "$FUSE_RUN_DIR" --execute
```

Choose the campaign command for automatic T → J → validation/deployment/export → one TEST evaluation. Use a fresh output directory for a new run. T uses 50 epochs of 160 labeled frames; J uses 100 epochs of 80 patient pairs. Validation reports foreground and RV/MYO/LV Dice after each epoch. The locked VAL rule selects either pure-student or fusion deployment.

The historical Q3D entrypoints remain available:

```bash
python train.py --config configs/candidate_q3d/v1.json --out "$Q3D_RUN_DIR" --allow-full-run
python evaluate.py --checkpoint "$Q3D_STUDENT_EXPORT" --checkpoint-sha256 "$Q3D_STUDENT_EXPORT_SHA256" --data "$Q3D_EVALUATION_MANIFEST" --data-sha256 "$Q3D_EVALUATION_SHA256" --train-data "$Q3D_TRAIN_MANIFEST" --train-data-sha256 "$Q3D_TRAIN_SHA256" --out "$Q3D_EVALUATION_DIR"
```

For a read-only per-class metrics view, run `python scripts/q3d_watch_metrics.py --plan reports/q3d-resenc-plan-v2-20261112/PLAN.json --once`. It accepts legacy `development_*` event names and prints `VAL` labels; internal keys such as `development`, `teacher_dev_fg_dice`, and `development_epoch` remain unchanged for compatibility.

The smoke uses manufactured data only; patient training, GPU validation, and pretrained-weight provenance are separate evidence requirements.
