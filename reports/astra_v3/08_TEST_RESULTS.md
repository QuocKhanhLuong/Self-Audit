# Software and bounded experiment results

Environment: `/Users/alvinluong/miniforge3/bin/python`, Python 3.11.16, PyTorch 2.14.0, CPU with 4 Torch threads for scoped tests. Commands run in the isolated audit worktree using `PYTHONPATH=src`.

## Regression sequence

| Test state | Result | Meaning |
|---|---|---|
| Original requested `pytest -q tests/test_pseudolabel_system_v3.py` at base | 3 passed | Original narrow suite; did not exercise target 255 |
| Root audit suite before corrections | 9 passed, 4 failed | Conflicting-region acceptance plus three UNKNOWN cases reproduce defects |
| Original + root audit suite after corrections | 16 passed | Corrected contracts and required student/teacher checks |
| Final combined tree after all workers completed |16 passed in 0.90s | Same corrected source, independently rerun final gate |

Evidence: [baseline](evidence/baseline_pytest.txt), [before fix](evidence/audit_tests_before_fix.txt), [after fix](evidence/audit_tests_after_fix.txt), [final combined tree](evidence/final_scoped_pytest.txt). Run:

```bash
rtk proxy env PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /Users/alvinluong/miniforge3/bin/python -m pytest -q tests/test_pseudolabel_system_v3.py tests/test_pseudolabel_v3_audit.py
```

Teacher checks cover finite forward/backward, odd-size tensors, region/semantic/dense probability simplexes, four trainable classes only, no semantic CE or semantic gradient from seed-free reconstruction, abstention, zero/strong evidence, and conflicting confident regions. The backward objective is a synthetic diagnostic, not a proposed semantic objective.

Student checks cover compact exact A0 tensor identity, 1/2 outer refinement turns, 1/3 internal DW calls, encoder reuse with tensor-identity hooks, no Auditor, audit-invariant but annotation-sensitive feature-only behavior, target-255 exclusion even under stale validity, all-UNKNOWN differentiable zero, and gradients to encoder/A0/DW from final logits. CPU profile determinism passes; no cross-device numerical/deterministic equivalence claim is made.

## Intervention results

Seed17, default teacher, random weights, synthetic `B=2,3,64,64`. Soft changes are average absolute probability differences from the unperturbed run.

| Intervention | Soft change | Base accepted fraction | Interpretation |
|---|---:|---:|---|
| Zero motion descriptor | .005040 | 0 | Computational influence, no anatomical validation |
| Shuffle motion across cases | .000913 | 0 | Computational influence only |
| Zero temporal differences | .005351 | 0 | A zero raw difference is not necessarily a zero learned descriptor |
| Swap temporal neighbors | .001018 | 0 | Direction sensitivity; reversal invariance is not an imposed contract |
| Shuffle z neighbors, keep center | .000694 | 0 | 2.5D context affects representation |
| Zero evidence | 0 | 0 | Exact neutral intervention |
| Random evidence | substantial | .978516 | Arbitrary semantic evidence can manufacture acceptance |
| Strong LV evidence | substantial | 1 | Interface overrides names; no anatomy guarantee |

After the software fix, random-evidence acceptance becomes .089355 and the four-way conflicting-region counterexample goes from 1 to 0. Strong consistent evidence still works. Neither change constitutes calibration.

## Post-freeze evaluation verification

Worker H evaluated seven frozen seed arms on 16 ACDC volumes after checking the pinned manifest and every source/config/script/split/prediction hash. Root independently inspected the evaluator, reran it in an isolated evaluator process, and recomputed hierarchical Dice from per-volume confusion counts. All record and split metrics reproduce exactly after JSON key normalization. No absent GT foreground class occurs in this cohort; every confusion matrix accounts for every voxel.

Worker H's original synthetic tamper check was tautological. Root added six actual verifier tests simulating bad manifest/config/script/split/image/prediction hashes, each rejected before any NIfTI load. Root also verifies all hashes after evaluation, including config/script/split omitted from H's post-check. UNKNOWN false negatives, null precision, swaps and phase-equal aggregation were checked. Evidence: [verification](evidence/root_evaluator_verification.json), [log](evidence/root_evaluator_verification.log), [evaluator](workers/H_evidence/evaluator.py).

## Evidence boundaries

No long training, 150-epoch run, learned adaptive selector, matched-compute semantic trial, real full-cine motion trial, M&Ms run, CUDA benchmark, 4-core/8GB constrained-host test, clinical UI test or .91 evaluation was run. A small synthetic optimization exercise by C is explicitly not clinical evidence. Round-1/2 and long student training are **STOPPED BY FAILED ROUND-0 GATE**, not silently omitted.
