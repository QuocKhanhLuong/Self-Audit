# Event-triggered reference-free Self-Audit: research result

**Decision C — REFERENCE-FREE AUDIT SIGNAL IS TOO WEAK for the implemented baseline.** Preserve the supervised Auditor as a control and study event-triggered intervention separately. This is a rejection of the tested region/edge compatibility proxy as a new core, not a proof that all reference-free audit is impossible.

Astra fetched and inspected main `874c0772c9b00a0fbc5e781ea8e457794160f654`, then implemented an isolated baseline on `feat/event-triggered-reference-free-audit` in `/Users/alvinluong/Self-Audit-event`. Canonical Self-Audit, REW, Candidate C, mask-free code and the active training checkout were preserved.

The method is: supervised Annotation with a learned dynamic read → cheap learned audit-value trigger → optional GT-free spatial compatibility Auditor → redirected support → one shared-reader/writer correction. Skip is exact identity. Auditor/Trigger learn only from image/prediction compatibility and detached counterfactual experiences; GT is used for Annotation loss and offline evaluation. There are no supervised FIX/REGRESS targets, hidden GT error maps or best-of-M candidate selection in this baseline.

**Executed:** 16 passing unit tests; synthetic mechanism checks; 21 bounded CPU runs on real ACDC (seven controls x three seeds); complete 20-patient validation; frozen source/guidance interventions; an evaluation-only semantic/extent red team. The existing 80/20 patient split was retained. Each real run used 120 actor updates, 960 supervised examples, B8, 64x64 and a compact actor. These deliberately underfit runs are not clinical or full-convergence evidence.

| Key result | Observation |
|---|---|
| Mean retained Dice, learned / entropy / always / unguided extra read | 0.1822 / 0.2386 / 0.2607 / 0.2669 |
| Learned final audit calls, seeds 0/1/2 | 0 / 4 / 0 out of 376 slices |
| RF-vs-true improvement correlation, 20 patients per seed | Spearman +0.250 / +0.433 / -0.332 |
| Remove sampled source | Dice falls; reading uses evidence |
| Remove audit guidance | Negligible or favorable mean effect |
| Repair value-head tracking at a frozen state | Actions return; benefit per call remains near matched random |
| Wrong myocardium dilation preferred to evaluation oracle | 67.3% of 364 nonempty slices |
| Time to prespecified Dice 0.5/0.7/0.8 | All right-censored; no convergence success |

The novelty review covers active computation, active perception, QC, correction, test adaptation and 2025–2026 work, with method-level reading for the nearest threats and explicit unknowns. Dynamic geometry, a critic, replay and sparse updates are established components. A potentially novel result would require useful RF action contrast and causal audit-guided reading that improve learning efficiency; that result was not obtained.

The GPU on `vast-gpu` was occupied by REW when inspected; local CPU execution avoided interference. The last SSH recheck returned connection refused, so current GPU/job state is unknown. New CUDA/4090, strong pretrained A0, full-resolution ACDC and M&Ms results are NOT RUN.

Read [final decision](13_NEXT_DECISION.md), [bounded evidence](10_BOUNDED_EXPERIMENTS.md), [baseline comparison](11_BASELINE_COMPARISON.md), [convergence](12_CONVERGENCE_ANALYSIS.md), [mathematics](04_MATHEMATICAL_MODEL.md), [prior art](02_PRIOR_ART_AND_NOVELTY.md), and [orchestration failures and verification](00_ORCHESTRATION.md). Machine-readable raw metrics and hashes are in `evidence/` and `evidence_manifest.json`. Reproduction instructions and exact publication scope are in 08.
