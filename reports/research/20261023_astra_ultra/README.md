# Astra ultra research continuation — 23 October 2026

This council continues from the actual Git state, not from the superseded no-GT design. The current canonical source is semi-supervised v2 at `637f38ced217fe1db2347a87924695533056d8f0`; the latest experiment evidence is on `8960b6baac4c6600553d900c303f192c2fcfa4f0`. The historical six-worker theory report at `f170dc1aeb030bf60abbae439c778815decc8266` informs caution about surrogate objectives and matched controls, but does not establish the current supervision or implementation state.

Six workers were launched with explicit `gpt-6-astra` and `ultra` settings. The resolved backend identity is not independently exposed. Each worker owns its own report and probes; reciprocal reviews preserve disagreements and narrowed claims. Parallel reviewers do not create independent empirical samples.

This is a report and bounded CPU-probe delivery. No patient training, GPU job, weight download, external job change, or production architecture change is part of this session. Historical checkpoints, datasets, baseline evidence and the original dirty checkout are preserved. The experimental branch reports prior external GPU runs; their saved scalar tables can be checked locally, but raw predictions and checkpoints are not thereby replayed.

Evidence labels used throughout:

- **VERIFIED_SOURCE / VERIFIED_LOCAL:** inspected immutable source, or checks actually executed in this session.
- **RECOMPUTED_FROM_TRACKED:** arithmetic rerun from Git-tracked results; this verifies aggregation, not the underlying predictions.
- **REPORTED_EXTERNAL:** a historical run/receipt recorded in Git; raw-run replay not performed here.
- **DERIVED:** mathematical consequence or explicit manufactured counterexample.
- **PROPOSED / NOT_RUN:** a new control, repair qualification or experiment that has not been implemented/run as a patient study.

Entry points:

- [Decision brief in Vietnamese](DECISION_BRIEF.md): result, limitations and research decision.
- [Next study](NEXT_STUDY.md): ordered gates and the minimum P/S/J contrast.
- [Review summary](REVIEW_SUMMARY.md): six actual challenge/response reviews.
- Worker dossiers: [A source](workers/A_source.md), [B theory](workers/B_theory.md), [C empirical](workers/C_empirical.md), [D prior art](workers/D_prior_art.md), [E design](workers/E_design.md), [F adversarial](workers/F_adversarial.md).
- [Local canonical checks](evidence/LOCAL_VALIDATION.json): check_repo PASS, 43 tests passed with one FutureWarning, mechanisms PASS.
- [Coordinator replay](evidence/coordinator_replay/RECEIPT.json): five research scripts rerun successfully in an isolated source copy.
- [Preservation receipt](evidence/PRESERVATION.json): all 236 pre-existing original-checkout files retain hashes; original HEAD unchanged.

Run the bounded coordinator replay from the repository root with `rtk proxy python reports/research/20261023_astra_ultra/probes/coordinator_replay.py`. It uses the current Python environment, reads immutable Git objects, and creates temporary source copies; it launches no patient jobs. Dependencies are those of the canonical software checks. `evidence/SESSION.json` pins all source refs. `evidence/GIT_SOURCES.json` records hashes and immutable source URLs. `evidence/LOCAL_VALIDATION.json` records actual canonical checks and environment.
