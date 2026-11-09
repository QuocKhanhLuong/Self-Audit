# Self-Audit — Candidate A

The active implementation is the typed REGION/BOUNDARY measurement-code model.
A shared-encoder, two-branch teacher guides query proposals. The student predicts
quantized measurement values and selects queries; its decoder receives only the
code, coordinates and public native geometry. Inference needs the student only.

Training uses dense budgeted TRAIN labels. An epoch visits every annotated
frame/slice once. U images are unused. No no-GT, semi-supervised improvement,
SOTA or 96-Dice claim follows from this software implementation.

- [Method and tensor/geometry contracts](docs/CANDIDATE_A.md)
- [ACDC and original M&Ms data preparation](docs/DATA_PREPARATION.md)
- [Train, epoch logs, export and evaluation](docs/RUNBOOK.md)
- [Component ablations](docs/EXPERIMENTS.md)
- [Removed code and historical reproduction](docs/MIGRATION.md)

```bash
python scripts/check_repo.py
python -m pytest -q
python scripts/smoke_candidate_a.py --out /tmp/self-audit-a-smoke
```

The smoke test uses manufactured data. Real training requires licensed datasets,
locked patient splits and local hash-pinned backbone weights; nothing is downloaded
or trained on patients by these checks.
