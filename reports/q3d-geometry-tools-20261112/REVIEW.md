# Independent software review

Evidence: **SOFTWARE_ONLY**. Three independent read-only review scopes covered
geometry/header preparation and loader guards, the pinned topology/ERF probe
contract, and GPU-profile prerequisites/instrumentation. The coordinator fixed
the findings before the final full-suite run. No review ran patient training,
the new ERF probe, or a GPU profile.

| Finding | Resolution |
| --- | --- |
| A native-unit conversion could produce forbidden unit spacing | Apply the shared geometry validator after conversion to millimetres; regression covers the converted unit placeholder. |
| A repaired report could retain matching spacing but an inconsistent header affine | Shared `validate_audit_geometry` checks coded-form selection, native zooms, unit conversion, and the full XYZ-to-ZYX affine, including orientation and translation. Both TRAIN proposal and DEV profile validation use it. |
| Original CINE and prepared-array hashes were compared across formats | Remove that invalid equality. Original NIfTI bytes and prepared `.npy` bytes have separate hashes; declared mapping provenance and the unresolved `NOT_VERIFIED` voxel bridge remain explicit. The reviewer retracted the original equality suggestion. |
| Output close failure could leave a partial repair file | Cleanup now covers write and close failures after exclusive creation; an existing path is never removed. This is exception rollback, not a crash-durability guarantee. |
| Canonical probe identity was only implied by pinned source | Explicitly validate CPU, 4 threads, ResEncL architecture, checkpoint and registry hashes. Label nnssl revision as declared metadata; record field-validated inspection and the exact two pinned nnU-Net source files. |
| Real profile API could accept fake test dependencies | Separate the explicit SOFTWARE_ONLY harness from the public real-profile API; synthetic results cannot emit GPU_PROFILE evidence. |
| Profile report write failure could skip output cleanup | Nested cleanup runs even if report publication fails; no retry or architecture/crop changes. |
| Largest-patient selection and timing projections overstated coverage | Force only the first eligible L/U selections, retain the remaining sampler behavior, verify exactly 20 optimizer updates and T-a phase, and label timing extrapolations PARTIAL_PROJECTION. |
| Config and canonical probe use different architecture identifiers | Validate `nnfoundation_resenc_l` and `ResEncL` against their own pinned contracts, with canonical receipt acceptance and rejection tests. |

The final independent profile reviewer reported no additional actionable
contract bugs after these fixes. The geometry reviewer also confirmed normal
and protocol-B engine guards execute before loader construction; caller-supplied
objects cannot retroactively undo external array reads, as documented.

Final verification: `593 passed, 2 skipped`, `check_repo` PASS, and all 13
synthetic smoke recipes completed. The two skipped tests are opt-in nnssl
registry/checkpoint integration tests whose environment variables were unset.
Full original-header audit and patient-array equivalence are distinct from
these software checks.

The current Phase A header conflicts remain a real stop condition. New physical
geometry, teacher phase/stride lock, corrected-GT physiology, PLAN v2, GPU timing
and training are **NOT_RUN**. Registering new validated config pins is pending
Phase B; the profile command intentionally cannot use the historical real recipe
as if this registration had happened.
