# Self-Audit v2 — joint audited pseudo-updates

**Current setting: semi-supervised native cardiac segmentation. This is NOT the
historical no-target-mask method.** Every dense labeled TRAIN patient used by the
supervised step or the audit is counted in the same annotation budget.

One student, one EMA copy, one canonical entrypoint. DINOv3 initializes a native
multiscale decoder; dense CE/Dice trains its encoder and detail path. Pseudo
updates are constructed from EMA predictions and selected jointly against a
labeled TRAIN audit slice. Functional trial evaluation checks the actual output
before an update is committed. EMA moves once, after the final decision.

- `python train.py --describe`: actual architecture and supervision contract.
- `docs/METHOD.md`: equations, pipeline interfaces, proofs and boundaries.
- `docs/NOVELTY.md`: prior-art collisions; no established-novelty claim.
- `docs/EXPERIMENTS.md`: controls, measurements and falsification plan.
- `docs/RUNBOOK.md`: prepare data, verify weights, pilot, evaluate, plan runs.
- `docs/MIGRATION.md`: historical recovery and cleanup scope.

The head fast path uses exact affine-head logit effects. The decoder path uses
parameter-to-probability JVPs and independently checks nonlinear trial outputs.
Neither path certifies Dice, clinical quality or generalization. Audit labels are
TRAIN optimization data, not a clean unseen validation set.

## Reproduce software checks

```bash
python scripts/check_repo.py
python -m pytest -q
python experiments/run_mechanisms.py --out /tmp/self-audit-mechanisms
python experiments/run_synthetic.py --out /tmp/self-audit-synthetic --steps 24
```

These commands use manufactured arrays and an explicitly named tiny CPU model.
They do not train on patients, load pretrained DINOv3 weights, or measure medical
accuracy. See `reports/semi_v2_20261020/` for the recorded scope/results.

`configs/pilot.json` is a 32-update head pilot; `configs/full.json` describes a
10,000-update decoder experiment, not an automatically authorized job. Patient
runs require an explicit label-budgeted manifest and local hash-pinned DINOv3
source/weights. There is no random, legacy-model, dataset or checkpoint fallback.

Historical code, results and exact layer reference are recoverable at
`archive/pre-semisupervised-20261020` (`924265f7df604d3dca84dda49578b1a0cf28d821`).
They were not rewritten. Use a separate worktree for reproduction.
