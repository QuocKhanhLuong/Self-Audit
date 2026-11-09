# Self-Audit — Candidate A and A-3D

The active code contains two versioned implementations of the same research
question: can a student build a finite typed anatomical measurement code and
decode a mask from that code alone?

- [Active methods and protocol boundary](docs/METHOD.md)
- [Candidate A, native 2D/2.5D](docs/CANDIDATE_A.md)
- [Candidate A-3D, native full-volume](docs/CANDIDATE_A3D.md)

Training uses dense budgeted TRAIN labels. An epoch visits every annotated
frame/slice once. U images are unused. No no-GT, semi-supervised improvement,
SOTA or 96-Dice claim follows from this software implementation.

- [ACDC and original M&Ms data preparation](docs/DATA_PREPARATION.md)
- [Train, epoch logs, export and evaluation](docs/RUNBOOK.md)
- [A/A-3D ablation plans](docs/EXPERIMENTS.md)
- [Removed code and historical reproduction](docs/MIGRATION.md)
- [Portable design sources](docs/design/README.md)

```bash
python scripts/check_repo.py
python -m pytest -q
python scripts/smoke_candidate_a.py --out /tmp/self-audit-a-smoke
python scripts/smoke_candidate_a3d.py --out /tmp/self-audit-a3d-smoke
```

Both smoke tests use manufactured data. The A-3D command is available once its
implementation is present. Real training requires licensed datasets, locked
patient splits and local hash-pinned backbone weights; nothing is downloaded or
trained on patients by these checks.
