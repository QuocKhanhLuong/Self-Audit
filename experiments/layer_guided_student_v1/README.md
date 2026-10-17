# Canonical layer-guided student implementation

This directory is the current implementation, not an alternate legacy trainer.
Its path is retained to preserve source/checkpoint identities during cleanup.

Use the repository-root `train.py`. Read [CURRENT](../../docs/CURRENT.md),
[RUNBOOK](../../docs/RUNBOOK.md), and [ASTRA_NEXT](../../docs/prompts/ASTRA_NEXT.md).

The six layer modules and the training/integration implementation are retained
from c4973553a743648a04c603c6b4634240933b8b80. Historical review receipts and
superseded handoffs remain in that commit, rather than competing with current docs.

Active policy: full frozen Q/reliability and feature cosine, coarse-only AdamW.
Native detail, gradient guard, refiner and semantic self-learning are not enabled.
`integration/` provides internal constructors and verification utilities, not a
second patient trainer. `training_v1/` contains the actual optimizer loop.

Tests include the unchanged reference operators and differential parity checks.
Use `python scripts/verify_current.py --out /new/path/outside/checkout` from the
repository root. Known reference limitations remain documented in
[REFERENCE_DEVIATIONS](REFERENCE_DEVIATIONS.md); they are not safety guarantees.
