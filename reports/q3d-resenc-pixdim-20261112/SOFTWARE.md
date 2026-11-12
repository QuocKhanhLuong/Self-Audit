# Pixdim amendment software verification

User approved the explicit ACDC pixdim amendment on 2026-11-12 and authorized training after checks. This receipt records SOFTWARE_ONLY verification of the opt-in geometry policy, K and stage selectors, and separately bounded T/J profiling. Strict geometry remains the default; baseline sources/configs are unchanged.

- check_repo: PASS.
- Full pytest with default OMP environment: 605 passed, 2 skipped in 189.75s (0:03:09). The skips require external nnssl assets; the pinned-asset integration was separately checked on the server (2 passed, 10 deselected).
- All-arm smoke: PASS; synthetic override is reported and does not authorize a real gate bypass.
- Independent geometry review found no blocker for the actual 148 header forms (all qform_code=0); dual-coded forms with scale-only disagreement remain conservatively rejected.
- Independent campaign/profile review found a T-only baseline root regression. It was fixed before final verification; the original baseline and Q3D config/campaign tests pass.
- Three Luna Orca tasks in run_4ed992683d4c completed and were released: geometry policy (53 focused tests), stage/K selection (9 focused tests), T/J profile selection (18 focused tests); coordinator integration fixed one missing synthetic unit field and the T-only baseline root fallback.

Patient training and GPU profiling are NOT_RUN at this software publication. Original Drive integrity audit is in progress and is recorded separately. The historical blocked audit and historical phase receipts remain unchanged.
