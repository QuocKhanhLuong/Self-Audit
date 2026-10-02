# Historical supervised reproduction

The obsolete triple-config Python runner and the Bash/PowerShell multi-process
runners were removed from the current tree on 2026-10-02. They are preserved in
Git at `c31825a7f90df47f9f9382a9c9595e43f7f71236`. These recipes use mask GT for
training; they cannot substantiate a fully no-GT claim.

To inspect or reproduce that historical code, create a separate checkout:

```bash
git worktree add --detach ../Self-Audit-historical c31825a7f90df47f9f9382a9c9595e43f7f71236
cd ../Self-Audit-historical
```

Only in that historical checkout, the triple-config runner is available:

```bash
python scripts/train_self_audit_legacy.py --config_a configs/self_audit_annotation.yaml --config_b configs/self_audit_auditor.yaml --config_c configs/self_audit_joint.yaml --output_dir weights/self_audit_legacy
```

The same revision contains `scripts/run_full_pipeline_legacy.sh` and
`scripts/run_full_pipeline_legacy.ps1`. Reproduction also requires its specified
environment and datasets; checking out code alone does not reproduce a result.

The historical single-process Python runner carried live weights between phases
while resetting optimizers. The historical shell workflow invoked disconnected
phase processes and reloaded saved stage-best checkpoints. These recipes are
not interchangeable. Phase A trained annotation, Phase B trained the auditor,
and Phase C jointly fine-tuned them.

The current supervised reference uses one configuration and `UnifiedTrainer`:

```bash
python scripts/train_self_audit.py --config configs/self_audit_full.yaml
```

It retains live weights across intervals, selects `best.pt` only inside the
profile's gated interval, and refuses missing-best calibration. Compatibility
helpers for evaluating historical phase artifacts now live in
`scripts/self_audit_post_training.py`; their historical fallback semantics do
not replace the stricter `UnifiedTrainer` path. Phase YAMLs and shared training
modules remain because current tests, artifacts and benchmarks depend on them.

For the current fully no-GT research flow, use the v3 description in
[README](../README.md) and the [main audit](../reports/main_baseline_20261002/00_DECISION.md).
