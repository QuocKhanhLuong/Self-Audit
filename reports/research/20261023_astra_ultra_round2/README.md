# Astra ultra — second research round, 23 October 2026

Read the [Vietnamese decision brief](DECISION_BRIEF.md) and [independent review summary](REVIEW_SUMMARY.md). This round extends the [first report](../20261023_astra_ultra/README.md) at `ccfde9bb99b6050e00ad94fbd6d57fcf9afc8b3f`; every first-round artifact remains unchanged.

Canonical semi-supervised v2 source stays `637f38ced217fe1db2347a87924695533056d8f0`; recorded external experimental evidence stays `8960b6baac4c6600553d900c303f192c2fcfa4f0`. The six existing workers were resumed with requested configuration **gpt-6-astra / ultra** and separate ownership. Backend execution identity is not independently exposed by the collaboration interface. No production solver, model, config or baseline was changed; no patient/GPU work or downloads occurred.

## New evidence

| Dossier | Question and new result |
|---|---|
| [A — numerical qualification](workers/A_numerics.md) | 114 cases: normalization fixes scalar false stalls but retains a multivariate failure; exact rational checks qualify feasibility and rounding. |
| [B — geometry and finite guarding](workers/B_geometry.md) | Exact local J can reject every configured finite step while matched S accepts; equality and actual J-benefit cases retained. |
| [C — endpoint headroom](workers/C_headroom.md) | Whole-patient hindsight oracles and marginal coupling bounds; final aggregates cannot identify transaction histories. |
| [D — claim limits](workers/D_claims.md) | Local/actual/selection/population claims separated; four narrow primary references, no new-method claim. |
| [E — sequential identification](workers/E_identification.md) | Manufactured transductive two-step policy reversal with EMA; 12 shadow-isolation checks. |
| [F — generated-direction counterexample](workers/F_realizable.md) | Real candidate generator and guard lower every Brier group but lower RV Dice; full bounded search and selected-fixture replay retained. |

Evidence labels are scoped, not interchangeable: VERIFIED_SOURCE means inspected immutable code; EXECUTED_MANUFACTURED / VERIFIED_SYNTHETIC_CPU means CPU fixtures; RECOMPUTED_TRACKED means saved-count arithmetic; DERIVED means explicit proofs; REPORTED_EXTERNAL means prior remote runs; PROPOSED/NOT_RUN means future controls and patient execution. Selected counterexamples and retrospective oracles are not frequency estimates or operational policies.

## Verification and reproduction

[Canonical checks](evidence/canonical_checks.json): `scripts/check_repo.py` PASS; `pytest -q` 43 passed with one torch.jit FutureWarning. [Initial coordinator replay](evidence/coordinator_replay/RECEIPT.json): six primary probes passed in an isolated source/report copy. [Post-review replay](evidence/coordinator_replay_final/RECEIPT.json): changed A/E probes and five new review/audit probes passed; unchanged probes were not needlessly repeated. A [final changed-probe refresh](evidence/coordinator_replay_refresh/RECEIPT.json) reran only F’s reviewer script after its last metadata-field correction. Across the three phases, all eleven primary/review executables have passing receipts bound to their delivered hashes. Each receipt pins commands and source/script hashes. Replay folders retain only the JSON outputs of listed commands. Packaging removed redundant copied dependencies; original execution receipts remain unchanged, and the top-level evidence directory retains those dependencies.

To repeat the coordinator's initial phase from this checkout:

```sh
rtk proxy python reports/research/20261023_astra_ultra_round2/probes/coordinator_replay.py
```

The `--refresh` phase reruns only scripts changed after the saved initial and post-review receipts. The `--final` phase compares current primary-probe hashes with the initial receipt, reruns changed probes and runs the five review/audit probes. These commands overwrite the respective local replay receipt folders; the committed historical receipts remain the evidence of this session. The fixed F replay reads the saved fixture and does not repeat discovery. See F's dossier for the separately recorded discovery procedure.

[Preservation](evidence/PRESERVATION.json) verifies all 236 pre-existing dirty/untracked entries in the original checkout and all 81 parent-inventory artifacts plus its inventory. [Artifact validation](evidence/ARTIFACT_VALIDATION.json) records syntax/link/receipt checks. [INVENTORY.json](INVENTORY.json) hashes delivered files, excluding itself and Python caches. Review records may name pre-formatting snapshot hashes; delivered bytes are authoritative in the final inventory.

Research publication is on `research/astra-ultra-continuation-20261023`, separate from main. No claim in this package upgrades software, manufactured or saved-count evidence into patient efficacy.
