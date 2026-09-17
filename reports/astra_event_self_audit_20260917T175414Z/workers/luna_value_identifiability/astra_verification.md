# Astra verification of narrow Luna derivation task

Worker: GPT-5.6 Luna subagent `/root/luna_value_identifiability`; no nested delegation. Exact remit: (1) an observational non-identifiability counterexample, (2) a sufficient RF-to-true-utility bound, (3) exploration/replay identifiability, (4) what detach does and does not prove. Allowed material: supplied equations/definitions; no code changes, overall plan or architecture decisions. Stop after the four derivation checks.

Completed report returned to Astra. This file is Astra's verification/synthesis, not a verbatim worker transcript.

Astra checked the two-pixel construction directly: fixed observed audit={1}, skip={2}, hidden truth {1} yields +1 Dice contrast, hidden truth {2} yields −1. Thus identical observables do not identify truth. A uniform assumed contrast-error bound ε implies the same expected-error bound and positive net true utility only beyond that margin; without a semantic calibration assumption this is vacuous. RF regression alone does not establish it.

Astra narrows the worker's propensity statement: actual twin execution observes both outputs at the same frozen state, so no action-level importance weight is needed within that pair. Random exploration samples states; prioritized or stale replay can change the state/version estimand. No amount of unlabeled exploration identifies true Dice directly.

Detach blocks direct gradient paths, not statistical dependence, data selection effects, or actor-version drift. In the selected disjoint-parameter baseline, direct cross-loss gradient conflict is already absent for always-audit too, so selective audit cannot claim that mechanism without additional evidence.

All four results are mathematical checks, not measured segmentation performance.
