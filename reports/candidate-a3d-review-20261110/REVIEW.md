# Independent Candidate A-3D source review

Review scope: frozen Candidate A-3D source in this checkout, with the final
software receipts already copied under this directory. This is a source and
protocol audit, not a claim about teacher quality or segmentation efficacy.

## Verdict

No new current-source blocker was found after the final freeze. The software
contract is **PASS_SYNTHETIC_ONLY**. The authoritative checks report 117 tests,
`check_repo`, the focused Ruff check, staged diff validation, 88 synthetic
ablation jobs, A-3D student-only CLI inference, and the official MedNeXt CPU
adapter check as passing. The 32-file source inventory is
`f9ad328fd27add333ea296edeb482e0279e752fc85faa334e312c43503573c67`.

One protocol distinction remains important before patient reporting: training/query
targets exclude the outer FOV from anatomy, while the current evaluation
surface metric treats the outer FOV as a surface. The documentation now names
this explicitly (including the voxel-surface, anisotropic-EDT and 1 mm metric
details); it is a resolved documentation item, not an open implementation
blocker, and the two quantities must not be presented as the same boundary
definition.

## A3D-01..12 checklist

| ID | Status | Source evidence | Independent assessment |
|---|---|---|---|
| A3D-01 | PASS (software) | `self_audit/candidate_a3d/data.py:15-96`; `teacher.py:585-615`; `model.py:218-223`; `tests/test_candidate_a3d_native_contract.py:93-106` | Data, teacher and student use native `[B,1,D,H,W]` / `[B,D,H,W]` contracts. No patient-volume evidence. |
| A3D-02 | PASS (software); quality UNKNOWN | `teacher.py:257-355,502-643`; `scripts/check_mednext_adapter.py:31-70` | Original MedNeXt region route is retained; boundary/SDF path is separate and consumes high/deep features. The CPU receipt shows exact region parity and finite checkpointed backward, with random initialization only. |
| A3D-03 | PASS (synthetic) | `bank.py:218-241`; `model.py:225-241`; `tests/test_candidate_a3d_model.py:53-65` | REGION pools its declared footprint and BOUNDARY samples its declared voxel. Locality is fixture-tested; no real anatomy validation. |
| A3D-04 | PASS (synthetic); convention documented | `bank.py:256-290,346-464`; `tests/test_candidate_a3d_native_contract.py:24-63`; `docs/CANDIDATE_A3D.md:166-176` | Anisotropic spacing, continuous internal voxel-face SDF, absent `+1`, and full-occupied `-1` are implemented and tested. Evaluation uses a separately documented voxel-surface convention. |
| A3D-05 | PASS (synthetic) | `self_audit/candidate_a3d/codec.py:117-223`; `self_audit/candidate_a/k1.py:388-425,526-633`; `tests/test_candidate_a3d_bank_codec.py:72-88` | Student values are quantized before code/trial use; IDs, R0, geometry, checksum and packet bytes are charged. Production K1 requires affine geometry. |
| A3D-06 | PASS (synthetic) | `model.py:322-357`; `evaluation.py:391-403`; `tests/test_candidate_a3d_model.py:35-49` | Decoder API receives code and public geometry only; serialized/dequantized round-trip is checked. |
| A3D-07 | PASS (synthetic) | `self_audit/candidate_a/k1.py:388-444,539-633`; `tests/test_candidate_a3d_k1.py:96-132` | Utility uses the same decoder/state, quantized student value, incremental packet cost, detached utility target, restricted CE and STOP. |
| A3D-08 | PASS (software) | `engine.py:177-235,291-339`; `evaluation.py:34-66,356-418`; `metrics.py:99-110` | Validation is preflighted and disjoint; U labels are not opened; student loss checks teacher gradients; centralized validation rows preserve pooled counts and separate macro-patient fields. |
| A3D-09 | PASS (synthetic) | `self_audit/candidate_a/io.py:31-35`; `self_audit/candidate_a/io.py:90-110`; `scripts/candidate_a3d_infer.py:99-127`; `tests/test_candidate_a3d_pipeline.py:77-84` | Export contains student only; inference performs no teacher/GT access and checks packet round-trip parity. |
| A3D-10 | PASS (planning/software); efficacy UNKNOWN | `configs/ablations/candidate_a3d.json:3-24`; `scripts/candidate_a_ablations.py:124-155`; `reports/candidate-a3d-review-20261110/evidence/MATRIX_CHECKS.json` | Arms share the declared schedule/seed/data/bank/bit contract, with changes explicitly named. The 88 completed jobs are synthetic software evidence, not a fair patient result. |
| A3D-11 | PARTIAL | `self_audit/candidate_a3d/profile.py:11-53`; `evaluation.py:373-420`; `docs/CANDIDATE_A3D.md:108-115` | Wall-clock rollout, packet serialization/roundtrip/decode phases and total timing plus packet bytes/query counts are recorded. The docs now accurately state that `rollout_ms` is combined and that encoder/value-head/policy timing is not separately instrumented; component-level speed claims remain unsupported. |
| A3D-12 | PASS (evidence boundary) | `teacher.py:702-713`; `evaluation.py:324-326,425-428`; `docs/CANDIDATE_A3D.md:141-158`; `reports/candidate-a3d-review-20261110/RECEIPT.json` | Reports distinguish synthetic software checks, random-initialized adapter checks and student export from real efficacy. No SOTA/96-Dice/novelty claim is justified. |

## Boundary metric versus target convention

The target convention is explicit in `bank.py:256-290,346-464` and
`docs/CANDIDATE_A3D.md:29-41`: only internal voxel interfaces are anatomical;
FOV planes are excluded; absent and full-occupied cases have explicit sentinels.

The evaluation metric is different. `evaluation.py:82-94` computes six-
connected binary surfaces with `ndimage.binary_erosion(..., border_value=0)`,
so foreground touching the volume edge receives an outer-FOV surface. It then
uses anisotropic voxel-centre EDT, a fixed 1 mm voxel-counted (not
area-weighted) surface-Dice tolerance, ASSD and HD95. A full foreground volume
therefore returns metric status `ok`, while the bank returns
`full_occupied_no_boundary`. `docs/CANDIDATE_A3D.md:166-176` now documents this
as a separate evaluation estimand. It is not an implementation failure, but
the two values remain non-interchangeable.

## Resolved findings from the earlier audit

- Engine now calls `bank.sdf_targets(...)` (`engine.py:291-292`), and the bank
  owns the shared doubled-grid continuous-face construction. The old fallback
  voxel-centre EDT mismatch is no longer present.
- MedNeXt loading now requires the exact reviewed SHA and checks tracked and
  untracked source state (`teacher.py:358-375`), import origin and foreign
  cached submodules (`teacher.py:377-405`). The official CPU receipt confirms
  the actual pinned source path, native region parity and checkpointed backward.
- External weight metadata is hash/cohort/mapping validated by
  `teacher.py:151-195` and `engine.py:155-174,225-234`. This is metadata
  validation, not empirical proof that output channels have the intended
  biological semantics; provenance explicitly records that trust boundary at
  `teacher.py:711`.
- Geometry crop/context rejection is enforced in the data and cache adapters
  (`data.py:33-43`; `geometry.py:118-135`). Physical spacing is passed to the
  teacher boundary path and student value head (`teacher.py:525-540`;
  `model.py:243-263`).
- Pooled validation counts and macro-patient metrics are no longer overwritten:
  `metrics.py:99-110` is the single validation-row implementation used by the
  engines.
- The FOV/target distinction is now explicit in `docs/CANDIDATE_A3D.md:166-176`:
  evaluation uses six-connected voxel surfaces with outside-FOV background and
  anisotropic voxel-centre EDT, while the target bank uses continuous internal
  voxel faces. The former P1 documentation finding is resolved; this remains a
  protocol distinction to preserve in result tables.

## Remaining limitations

- No pretrained MedNeXt weights, patient training, GPU validation, real ACDC
  result, real M&Ms result, or RefineFormer result is present in the receipt.
- The official adapter check is `PASS_RANDOM_INITIALIZATION_SOFTWARE_ONLY`;
  it verifies architecture/forward/checkpoint mechanics, not SOTA or Dice.
- `codec.py:57-69` permits a convenience identity affine when a direct codec
  caller omits `affine_zyx`. Production K1/CLI paths require and pass the
  declared affine, so this is an API footgun rather than a current pipeline
  failure; a future strict public codec API should reject omission.
- The teacher deliberately pads internally and crops outputs
  (`teacher.py:593-604`) to make MedNeXt's four down/upsampling stages valid;
  the documentation now says “no data-level padding” and declares the internal
  model padding (`docs/CANDIDATE_A3D.md:23-25`).

## Evidence boundary

The final receipt is `reports/candidate-a3d-review-20261110/RECEIPT.json`.
It records 117 passing tests, 88 synthetic matrix jobs, student-only CLI
inference, source and manifest hashes, and the official CPU adapter receipt.
Those artifacts establish reproducible software behavior only. They do not
establish pretrained-teacher quality, patient Dice/F1/precision/recall,
clinical validity, SOTA, 96 Dice, novelty, or a preferred ablation.
