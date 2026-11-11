# Self-Audit — Candidate A and A-3D

The active code contains two versioned implementations of the same research
question: can a student build a finite typed anatomical measurement code and
decode a mask from that code alone?

- [Active methods and protocol boundary](docs/METHOD.md)
- [Candidate A, native 2D/2.5D](docs/CANDIDATE_A.md)
- [Candidate A-3D, native full-volume](docs/CANDIDATE_A3D.md)

The default recipes use dense budgeted TRAIN labels and visit every annotated
frame/slice once per epoch; U images are unused. A-3D has a separate opt-in
semi-supervised profile at
[`configs/candidate_a3d/semi_supervised.json`](configs/candidate_a3d/semi_supervised.json).
That profile uses image-only TRAIN U for detached EMA consistency only; it never
opens U masks or creates U GT-derived SDF/K1 targets; detached EMA soft SDF
consistency is allowed. No efficacy, SOTA, 96-Dice or
semi-supervised improvement claim follows from this software implementation.

- [ACDC and original M&Ms data preparation](docs/DATA_PREPARATION.md)
- [Train, epoch logs, export and evaluation](docs/RUNBOOK.md)
- [A/A-3D ablation plans](docs/EXPERIMENTS.md)
- [A-3D few-shot label-budget sweep](docs/FEWSHOT.md)
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
