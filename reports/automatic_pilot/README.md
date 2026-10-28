# Coordinator delivery evidence

Base: `bdb1c399d29c63ae49865af8db78954f79f9ad9d`; branch: `feat/fully-automatic-head-pilot`.

Training source digest: `0364bf9c85c5b90430b6fccd83aa51b74c910e7e17ed0cfa0aad612e14a5595b`.

All results below are manufactured CPU software evidence. No patient efficacy, pretrained DINO execution, PRO4000 profiling, or novelty conclusion was established. The CPU host was shared with validation processes; these are smoke timings, not a performance benchmark.

## Executed checks

- Repository contract check: PASS.
- Full pytest: 529 passed, 0 failed, 1 existing torch.jit.script deprecation warning (60.87s).
- Ten arms completed in equal-update and bounded equal-time mode.
- Twenty raw-student exports covered all six native slices per fixture; all prediction hashes and offline receipt chains verified.
- Fresh-run cold phase cost completed for every exported arm.
- One-second deadline completion passed; actual times did not meet the fixed 5% compute-matching rule. Research decision remains NO_GO_SOFTWARE_ONLY.

## Equal-update smoke (two outer steps per arm)

| Arm | Nonzero extra steps | Acceptance | p50 ms | p95 ms | Cold phase seconds |
|---|---:|---:|---:|---:|---:|
| scalar_brier | 0 | 0.000 | 165.978 | 176.705 | 1.3008 |
| joint_brier | 1 | 0.500 | 132.781 | 137.430 | 1.2286 |
| scalar_dice | 0 | 0.000 | 131.756 | 136.786 | 1.2062 |
| joint_dice | 1 | 0.500 | 162.528 | 170.215 | 1.2798 |
| frozen | 0 | NOT_APPLICABLE | 70.356 | 72.653 | 1.0942 |
| support_only | 0 | NOT_APPLICABLE | 76.083 | 82.816 | 1.1292 |
| plain_head_u | 2 | 1.000 | 148.075 | 152.174 | 1.2580 |
| extra_supervised | 2 | NOT_APPLICABLE | 63.194 | 68.113 | 1.1019 |
| smaller_lr | 2 | 1.000 | 130.770 | 132.818 | 1.2173 |
| full_ema | 2 | 1.000 | 286.696 | 291.276 | 1.5532 |

Cold phase clocks exclude interpreter/import startup; matrix subprocess wall times are separate. Nonzero counts for extra_supervised denote its extra labeled step. Frozen/support-only have no extra step. See [VALIDATION.json](VALIDATION.json) for both regimes, margins, throughput, weight/source hashes and exact artifact paths.

## Review and limits

Agy/Antigravity workers implemented controller, engine, cache, evaluation and initial runner. Coordinator review rejected premature completion reports, repaired final integration, and executed the final checks. A final agy runner launch stopped at sign-in (not a quota-exhaustion claim); coordinator completed that integration. All worker resources were released.

The findings in [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md) were corrected, with one explicitly bounded timing limitation: numerical resume state is tested exactly on CPU, but prior checkpoint-write cost is absent from its cumulative clock. Resumed cold costs remain INCOMPLETE and cannot certify equal compute.

No U/query/TEST masks entered learner or guard. Synthetic query references were opened only after all raw-student predictions were frozen. Guards certify only the fixed TRAIN-L extra-U transaction against theta_sup; they do not certify EMA or query patients.

Main and its pre-existing untracked work were preserved. Generated worker smoke artifacts under the isolated worktree datasets/test_run were left unstaged and were not deleted. No datasets or model weights are committed in this delivery.

Use [the runbook](../../docs/AUTOMATIC_PILOT_RUNBOOK.md) for bounded PRO4000 preparation. Required external evidence remains: actual target hardware profile, pinned pretrained weights execution, and independent matched-compute real-query contrasts (joint over scalar, Dice guard over smaller LR, U over extra-supervised).
