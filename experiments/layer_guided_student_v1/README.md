# Layer-guided student v1

**Current training handoff (2026-10-16):**
[RTX 4070 runbook](../../docs/prompts/ASTRA_TRAIN_4070_LATEST.md) and
[training_v1 policy/code](training_v1/README.md). This adds an explicit training
entrypoint with a new artifact schema. The original reproduction below remains
the record of the exact-reference/scaffolding delivery.

Opt-in reproduction of the published layer-level prototype, plus integration
scaffolding. The historical student, baseline ablations, configs, splits and
frozen evidence are unchanged. This directory is not imported by the existing
training entrypoints.

Start with [RUNBOOK.md](RUNBOOK.md) for executable commands and contracts, or
[DECISION_BRIEF.md](DECISION_BRIEF.md) for the Vietnamese delivery decision.

| Artifact | Purpose |
| --- | --- |
| [FUNCTION_MAP.md](FUNCTION_MAP.md) | Exact source-to-port symbol map and integration boundaries |
| [PARITY_REPORT.json](PARITY_REPORT.json) | Final CPU test, hash and random full-model receipt |
| [REFERENCE_DEVIATIONS.md](REFERENCE_DEVIATIONS.md) | Preserved counterexamples and limits of claims |
| [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md) | Separate reviewer findings and final disposition |
| [recipe.json](recipe.json) | Explicit split-head architecture, guard off, admission off |
| [integration/](integration/) | Local approved-backbone loader, geometry, bank and isolated steps |
| [receipts/](receipts/) | Raw verification logs and machine-readable outputs |

The six files in `layer_upgrade/`, `tests/test_operators.py`, the two probe
scripts and `pytest.ini` intentionally preserve reference bytes, including
formatting. Cleanup applies to the integration code and documentation; changing
the reference operators requires another version.

No official DINOv3 weights or patient arrays were used. The complete random
torchvision model smoke is software evidence only. No anatomical performance
claim is made.
