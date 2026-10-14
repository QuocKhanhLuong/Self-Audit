# Code review closure and evidence boundary

**Final local software verdict: PASS_SOFTWARE_ONLY, coordinator verified.** The independent Astra reviewer completed substantive defect finding and seven standalone mathematical check groups, then its terminal reported **“Selected model is at capacity. Please try a different model.”** before a final post-fix signoff. This document is written by the coordinator from the actual messages, source changes and final test receipts; it is not an invented completed reviewer report.

Independent review dispatch: `ctx_b9300583b44e`, task `task_b8f39ee35c4e`, original reviewer terminal `term_d6db4cc5-f67d-4a1c-860c-041ce6fdcd2f`. The initial scientific adversarial review had already completed; this later code-review follow-up is the interrupted task. It was fenced with `worker-abandon`, retained without process termination because Orca marked the terminal user-owned, and recorded failed with the capacity limitation. No additional agent or model result is claimed.

## Findings and local closure

| Finding raised by independent reviewer | Implemented correction | Verification |
|---|---|---|
| Absolute excluded-weight allowance can discard order-one loss at the denominator floor | Bound excluded weight divided by `max(total_weight,1e-8)`; check every scheduled batch before optimizer | Tiny fully excluded1e-8 and late-batch regressions |
| A plan self-hash does not detect edits made before execution | Require independently recorded canonical plan SHA at run entry | Allowed-range step/seed/patient mutations rejected before loaders/optimizer |
| Self-pinning an edited recipe can change learning rate | Independently pin published ORIGINAL recipe and bank; preserve signal→recipe→plan bindings | Recipe-lr mutation and arbitrary-bank identity tests |
| Fresh paths can still be children of frozen artifacts | Reject frozen/locked ancestors, source overlap and guarded plan paths | Sentinel/ancestor tests; originals retained |
| Modified count records could be attributed to the publication | Verify fixed publication manifest and count bytes | Count tamper regression |
| Clamped logs produce a finite proxy instead of true KL | Stable common-grid log-softmax; explicit undefined/infinite status where needed | Disjoint-support and KL1000 softmax-underflow checks |
| RV above-uniform probability is not winning class evidence | Separate above-uniform and max-competitor contrast; report Q-argmax strata separately from CE channels | Target(.1,.3,.5,.1) and empty-RV-stratum tests |
| float16 cache serialization was being treated as proven precision loss | Confirm exact teacher already returns float16; require actual replay instead | Source inspection by root, evidence worker and reviewer |

The root also added independently pinned original student/checkpoint identities, source/teacher/split/permission checks, and tests for those contracts. Pilot deterministic settings are explicit and fail on unsupported operations; CUDA behavior is still untested.

Final integrated command:

```bash
rtk proxy python -B -m experiments.guidance_signal_20261014.preflight --output reports/guidance_research_continuation_20261014/preflight_local_02
```

Exit0; **85 passed in2.61s**, no failures/skips. [Receipt](preflight_local_02/PREFLIGHT.json), [test log](preflight_local_02/pytest.log), [JUnit](preflight_local_02/junit.xml). The earlier integration pass overlapped a changing pilot API and failed12 tests because callers had not yet supplied the newly required plan SHA; its receipt remains in `preflight_local_01/`.

The independent [math checks](CODE_REVIEW_CHECKS.py) passed and were rerun by the coordinator: twelve dense inverse comparisons, absent/new-name counterexamples, R0 non-identifiability, exact target-gradient difference, denominator/activation semantics, plan-tamper ordering, and stable-logprob KL. These are mathematical/software checks, not real segmentation evidence.

## Remaining limits

- No original private bank/checkpoint or CUDA workload was run here after the user's local-only steering. Host identity, replay and stage scoring are still mandatory.
- Tests use explicit manufactured arrays and test-only identity injection. No fake teacher fallback exists in the CLI.
- The runner pins the new world-left-v2 source and checkpoint. It does not implement an old-source replay or recover paired old/new spatial error regions from marginal counts.
- Pilot final weights are private technical artifacts, not a drop-in original student export. Anatomical comparison of trained arms needs a separately frozen evaluation adapter/protocol.
- Solver residuals estimate iterative truncation in exact arithmetic, not verified floating-point intervals, anatomical uncertainty or label-free risk.
- No independent final post-fix reviewer signoff was completed because of the provider capacity error. The coordinator owns final integration acceptance; all raised defects have local regression coverage.

Production files and frozen publication bytes remain unchanged (`git diff --exit-code HEAD`, exit0). All new code and reports remain local/uncommitted/unpublished. No full-run, push or merge follows from this verdict.
