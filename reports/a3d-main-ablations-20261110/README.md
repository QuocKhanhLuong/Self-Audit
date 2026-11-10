# Compact A-3D main ablation

The default A-3D plan now uses seed 42 and four student variants: `full`,
`teacher_free`, `fixed_queries`, `global_only_queries`. Each dataset schedules
six jobs including the two shared teacher preparation stages. Existing plans
are not rewritten. The 21-arm catalogue remains available through explicit
`--arms`; it is no longer the default A-3D campaign.

`teacher_free` isolates teacher proposals while retaining TRAIN-label utility
learning. `fixed_queries` tests adaptive K1 selection as a package.
`global_only_queries` tests local query feature extraction. These are not
U-Net architecture or conventional-KD comparisons, and do not establish all
original novelty claims. A single seed does not measure across-seed variance.

Validation: `python scripts/check_repo.py` passed; `python -m pytest -q` reported
**119 passed**; focused Ruff passed. Both ACDC and M&Ms dataset profiles completed
the six-job default plan on tiny synthetic CPU fixtures (**12 jobs total**).
No patient training, GPU validation or efficacy measurement was run.

See [RECEIPT.json](RECEIPT.json) for changed-file hashes and
[SMOKE_RECEIPT.json](SMOKE_RECEIPT.json) for source, config and synthetic weight
hashes. Model runtime source is unchanged from `2011c21`; only campaign defaults,
planner metadata, focused tests and documentation changed. No datasets or
weights are included in this report.
