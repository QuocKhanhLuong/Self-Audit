# Current source audit

Fetched main: `826a11ca9c37fd024c9d7d83df594f8b051ec9ca` (merge of CUTS/DFC work).
Isolated branch: `feat/no-gt-baseline-20260918`, worktree `Self-Audit-nogt`.
The original active checkout and clean event worktree were not switched or edited.
Event HEAD really exists locally: `9568217ffbdfc2abf4956eabad8e7c48c455627c`.
Its prior reports are at `Self-Audit-event/reports/astra_event_self_audit_20260917T175414Z/`.
The event verdict applies to the tested fixed region/edge proxy, not all reference-free
signals. Its CPU64/120-update runs do not establish convergence or universal failure.

## Separate systems

Canonical `src/self_audit/`: supervised segmentation and transition auditing.
REW is supervised proposal/search/critic refinement, not the new no-GT method.
Event `src/self_audit_event/` in its separate worktree permits supervised annotation
GT while its auditor/trigger are reference-free. Candidate C is historical restitution.
Maskfree `src/self_audit_maskfree/` has image-only evidence partitions, anatomical
naming, observation fitting/scoring and two distilled students. It is not event audit.
New `src/self_audit_nogt/` is isolated and imports only the unchanged maskfree ontology
and contracts. No canonical/event/Candidate-C loss is reused.

## Implemented current maskfree path

| Stage | Actual symbol/line at base HEAD | Scope / interpretation |
|---|---|---|
| Raw discovery | `data/discovery.py:discover_dataset:1122` | Enumerates image headers, native geometry and frames/slices; optional preprocessed input paths need provenance checks. No assumed GT-free history merely from present loader. |
| Data support | `data/dataset.py:_build_training_unit_from_record:368` | Builds fixed fit/select/verify partition and `[3,H,W]` context; only fitting view goes to producer/candidates. |
| Normalize / resize | `data/dataset.py:_fit_statistics:148`, `_resize_role:178` | Percentiles/mean/std from fitting pixels; zero/resample role supports separately. New control instead uses full image, so leaf timing is not a scientifically matched quality comparison. |
| Representation | `models.py:Producer:156`, `_DenseTrunk:125` | Width16 GroupNorm encoder/decoder, features plus reconstruction; not a foundation backbone. |
| Producer objective | `losses.py:producer_loss:230` | Three neural forwards (original/transformed/masked), backward per batch; additional detached feature forward at trainer:1114. |
| Candidate bank | `trainer.py:_generate_bank:1321`, `hypotheses.py:generate_bank:460` | Four hypotheses per unit; features detached, CPU candidate construction. Recomputed on each visited unit/epoch. |
| Ontology | `ontology.py:resolve_roles:558` | Called by grouping/initializer/boundary/split candidates; semantic-alternative branch can reuse naming alternatives. |
| Audit | `auditor.py:audit_bank:677`, `audit_banks:767`; `observation.py:ObservationModel:926` | Fit nuisance appearance model on fit evidence, score select evidence. Bulk API amortizes fit_many/score_many over B×4; not four new producer forwards. |
| Two students | `trainer.py:_train_batch:1213`, `models.py:Student:190`, `losses.py:student_loss:394` | Independent identical-start students learn initial vs selected detached soft targets and validity. This is already distillation/amortization. |
| Updates | `trainer.py:_optimizer_step:1263` | Producer and active students stepped at accumulation-group boundary; invalid student support can suppress its update. |
| Finalization | `trainer.py:_finalize:1664` | Freeze checkpoints, revisit unit-level banks/audits, repeat annotation and export/verification passes. Training-step timing alone omits this cost. |

`workers/luna_train_batch.md` supplies the narrower call trace. Astra independently
read the caller, loss signature, model constructors, bank entry and finalization. The
new bounded profile measures component calls on real images and counts actual neural
forwards; its measurements and exclusions are reported in RESULTS. Workers/IPC and
whole finalization cost are UNKNOWN here, not inferred from host CPU percent.

## Ontology: what it actually guarantees

`ontology.py:128–153` fixes border share .10, max three non-BG groups, enclosure .90,
and primary prior penalty zero. R2 (`_merge_excess`:385) merges excess groups by size
and adjacency. R3 (`_holes`:305, `_enclosure_table`:408) uses complement components
that do not touch the border. A unique strongest enclosure names MYO and LV.
No enclosure leads to R5 draft intensity naming and semantic_unresolved. R9
(`resolve_roles`:728–758) retains draft labels with validity0 rather than silently BG.
All-background input has no valid semantic pixels. Missing anatomical classes are legal.
`_prior_penalty`:517–555 records diagnostic violations but returns 0 by design.

None of these static facts alone proves a bug. We run the actual resolver on a closed
ring and a four-connected gap with the same image, then on real C0 partitions. Counters
separate R2 merges, R3 eligibility, abstention, valid per-class coverage and rule reasons.
A topological discontinuity is not automatically the root cause of MRI failures.

## Reuse and removal in this control

Reuse native NIfTI geometry conventions, patient separation, frozen-output evaluation,
the explicit ontology/validity contract and its diagnostics. Replace online representation
+ candidate-bank search with a deterministic image-only anchor and a separately tested
single scratch CNN. Do not retain a second student, audit/score module or history merely
to preserve project naming. Existing students already establish the amortization idea;
the new control isolates whether any useful named target survives this simplification.

Potential GT entry points excluded from the new method: paired supervised datasets,
supervised initialization, benchmark Hungarian matching, reference-based crops, optional
reference evaluator outputs fed back into selection, and unlabeled-looking preprocessed
arrays with unknown provenance. The historical frame-copy concern remains explicitly
unresolved in SUPERVISION_LEDGER even though current mask opens are blocked.
