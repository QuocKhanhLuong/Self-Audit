# Self-Audit — current student

**Canonical model: `LayerGuidedStudent` + approved DINOv3 ConvNeXt-Tiny.**
Start here, not in an old report or historical launcher.

```bash
python train.py --describe
python scripts/check_repo.py
python train.py --help
```

The only supported training phase is currently **coarse guidance**: frozen
CineMA full-Q/reliability targets, stride-16 feature cosine, ordinary AdamW.
The exact coarse/detail architecture is present, but the native-detail head
has no learning objective and stays zero. Guard, refiner, masked learning and
EMA semantic self-learning are **not enabled**. This checkout must not be
reported as a completed native-detail/self-learning method.

## Run the new model

Install the matching CUDA PyTorch/torchvision pair for your host first; do not
replace a working GPU environment with CPU packages. Then install the small
[dependency set](requirements.txt) and dependencies required by your reviewed
local DINOv3 source revision. Record the resolved environment for every run.

```bash
python train.py --phase coarse \
  --bank "$BANK" --bank-manifest-sha256 "$BANK_SHA" \
  --approval "$APPROVAL" --approval-sha256 "$APPROVAL_SHA" \
  --out "$NEW_RUN_DIR" --device cuda
```

Those variables must point to real host artifacts. Default budget: **32 updates**,
not 32 epochs. A complete run additionally requires an explicit full recipe and
`--allow-full-run`. Missing pretrained approval, mismatched bank/checkpoint,
legacy recipe or an existing output directory produces an error, never a
fallback to the old CNN or random weights.

Read [CURRENT](docs/CURRENT.md), [RUNBOOK](docs/RUNBOOK.md), and
[MIGRATION](docs/MIGRATION.md), in that order. The current research handoff is
[ASTRA_NEXT](docs/prompts/ASTRA_NEXT.md).

## Repository map

- `train.py`: sole canonical dispatcher; the Astra-named script is an alias.
- `configs/`: bounded pilot and explicit full-budget recipe.
- `experiments/layer_guided_student_v1/`: current model, verified constructor,
  geometry/bank adapter, training loop, checkpoints and tests. The directory
  name is retained to avoid changing source/checkpoint identity during cleanup.
- `src/self_audit_guidance/io.py`: artifact/NIfTI compatibility only; no old model.
- `tests/`: routing and anti-regression checks.
- `reports/layer_level_upgrade_20261015/`: immutable reference oracle needed by
  parity tests. Its historical instructions are not the active runbook.
- `splits/`, `checkpoints/`: retained, not modified by this cleanup. Training
  uses the split pinned in the original bank, not an arbitrary split file.

## Historical results

Old baseline implementations, runbooks, prompt chains and logs are removed
from the active tree, **not erased from Git history**. They are recoverable at
`archive/pre-cleanup-20261017` (`92a60744...`). The imported training integration
is pinned at `c4973553...`. Use a separate worktree for historical reproduction.
Legacy training command names now fail with `LEGACY_ENTRYPOINT_DISABLED` rather
than silently translating arguments or running another model.

This repository cleanup does not touch any local dataset, run directory,
checkpoint payload, server process or credentials. Do not run `git clean -fdx`.
