# Independent teacher-adapter review

Review date: 2026-10-24. Scope: `self_audit/cinema_teacher.py`, its focused tests,
and `scripts/check_cinema_teacher.py`. Reviewer is independent of implementation.
Current adapter decision: **APPROVED FOR BOUNDED ENGINEERING SMOKE** after fix verification.
The original findings below are retained as the review history; all six adapter
findings are now closed in code. The builder findings are also closed below; both components are approved for the
coordinator's bounded engineering run, subject to their recorded provenance limits.
No weight file, server state, or patient dataset was accessed by this review.

## Contract checks

Correct source facts in the initial implementation: pinned GitHub/HF revisions,
config and weight SHA256; full safetensors `strict=True`; four BG/RV/MYO/LV outputs;
XYZ model ordering; 32-channel pre-head decoder tap; frozen evaluation inference;
image-only full-FOV clipping at 0.95/99.5 percentiles; upstream-like uint8
quantization; end padding; 50%-overlap probability and feature averaging; explicit
full-FOV versus historical GT-crop normalization deviation; affine-based native
restoration preserving orthogonal oblique directions and flips.

## Initial findings requiring closure

1. **P1: imported source is not bound to the verified checkout.**
   `_import_pinned_get_model` inserts a path but accepts a preexisting `cinema`
   module in `sys.modules`, or an already-present path in a lower search position.
   An independently manufactured process with a foreign preloaded
   `cinema.segmentation.convunetr` returned its foreign `get_model` from the
   supposedly pinned loader. Dependencies were mocked only to isolate Python's
   module-cache behavior. Require all executed CineMA modules to resolve within
   the verified source root and reject foreign preloads.
2. **P1: nonfinite predictions can pass parity.** A synthetic decoder emitting
   positive infinity and a positive Conv3d head produce matching infinite logits;
   `torch.allclose` returns true. The adapter then returns NaN probabilities and
   infinite features with `feature_logit_parity_checked=True`. This was reproduced
   independently. Require finiteness before parity/softmax and after native
   restoration; reject nonfinite values instead of assigning valid lineage.
3. **P1: synthetic seam marker mismatch.** The adapter sets `_synthetic`, whereas
   the builder's observed boundary reads `getattr(adapter, "synthetic", False)`.
   Independent check confirms the public check is false for a synthetic adapter.
   Expose an explicit immutable/read-only property and enforce real provenance
   at every patient bank/cache API, not only the CLI.
4. **P2: hook-on/off parity is not established.** The initial code checks
   `head(captured_feature)` against the hooked forward, but has no unhooked native
   comparison. Record numerical errors, tolerances, counts, and finiteness for
   both comparisons using the same real model/input/dtype. A mock cannot prove
   real-checkpoint parity.
5. **P2: full-volume memory is unbounded and duplicated.** A `[512,512,16]`
   float32 32-channel accumulator is 512 MiB; out-of-place fusion adds another
   512 MiB before native outputs and temporary stacks. At `[768,768,32]` the two
   feature lattices alone occupy 4.5 GiB. Require preallocation resource checks,
   in-place normalization and/or streaming output, plus recorded workspace
   estimates for real input geometry. A small manufactured input does not measure
   full-FOV resource use.
6. **P2: extrapolation/coverage policy is implicit.** The adapter uses ceil grid
   sizes and nearest boundary extension in both resampling directions. These are
   legitimate choices if frozen and recorded, but differ from upstream round and
   zero defaults. Record them and valid physical coverage. Native target indices
   outside center range are not automatically invalid: voxel support extends by
   half a voxel. Distinguish extra target centers outside the original physical
   FOV from valid native edge interpolation. Correct the comment claiming first
   center alignment; the implemented shift aligns leading voxel edges.

## Independent checks executed

- `rtk python -m pytest -q tests/test_cinema_teacher.py`: **6 passed** initially.
- Foreign preloaded module accepted: reproduced in isolated Python process.
- Synthetic public marker false while private marker true: reproduced.
- Infinite feature/logit path returned nonfinite prediction with parity true:
  reproduced using a manufactured network.
- Affine-grid calculation for native XYZ spacing `[1,1,5]`, shape `[4,4,3]`:
  teacher shape `[4,4,2]`; target z centers map to native z `[0.5,2.5]`; native
  centers map to target z `[-0.25,0.25,0.75]`. This identifies the boundary
  extension behavior; it is not evidence of a coordinate transpose.

Real-checkpoint strict loading, real-checkpoint parity, target-GPU allocation,
real TRAIN image inversion, and patient efficacy remain outside this reviewer’s
executed evidence. Coordinator receipts must be assessed separately.

## Adapter closure verification

The revised adapter rejects foreign preloaded CineMA modules, exposes the public
read-only synthetic marker, rejects float32 overflow and nonfinite decoder/head
outputs, re-enforces frozen evaluation, and compares an unhooked native forward
against the hooked forward. It records numeric parity errors. Resource estimates
now run before physical resampling, use overflow-safe Python integer products,
and exclude model-internal activations explicitly; normalization is in-place and
native channels are filled without a full list/stack duplicate.

Geometry uses an explicit nearest-edge, half-voxel-cell support policy. Inverse
interpolation divides both probabilities and features by resampled support
weights. Partially supported voxels remain usable with their support metadata;
no-support voxels abstain. The constant-one decoder repro now returns `[1,1,1]`
for the anisotropic native endpoint sequence that previously returned `[1,1,0.5]`.
Coordinate-ramp checks on orthogonal oblique and flipped affines found interior
agreement to approximately 1e-6; edge discrepancies come from the declared
nearest extension, not XYZ confusion. Shears are explicitly rejected.

Independent rerun: **14 adapter tests passed**. Separate isolated repros confirmed
foreign-module rejection, nonfinite-output rejection, public synthetic marking,
zero hook-on/off and head-feature deltas for the deterministic mock, preserved
constant features, and a huge-spacing allocation request rejected before
resampling. These are CPU/manufactured software checks. They authorize proceeding
to the coordinator's bounded real-checkpoint strict-load smoke; they do not
establish real-checkpoint parity or patient quality.

## Builder findings and closure

The current builder was separately found to pair generic class boundaries across
an intervening third class: a `0|2|1` slab at 1 mm produced `(0,1)` endpoint pairs
at 2 mm. It must require an actual shared c/d interface. Teacher-mode output must
also validate the native-to-prepared voxel/affine bridge before assigning the
prepared image hash to an A1 cache, and A5 must reject synthetic student
checkpoints. These findings were sent to the builder owner for correction.

Adapter source SHA256 at closure: `7f0743d298ebfefa7b8fd347356b54792297106189a5cfe97839985baf93ef52`.

### Final builder verification

All identified builder findings are closed in the reviewed source. Candidate
segments now contain exactly one c-to-d transition, cannot cross a third class
or unsupported voxel, and retain the 3 mm / 0.8 confidence / at-most-8-per-group
limits. Independent `0|2|1` and low-confidence-middle `0|1|0|1` repros produce
no invalid outer `(0,1)` pair. Empty groups remain explicitly invalid, and final
selection uses patient round robin.

Both teacher and A5 paths validate the prepared TRAIN image against the full
native cine after the canonical image-only 0.5/99.5 percentile transform and
XYZT-to-TZYX permutation, with atol=2e-6 and rtol=0. Native image files and affines
are bound by hashes; source TRAIN membership, label role, evaluation exclusion,
duplicate image content, and coverage declarations are checked again on load.
Spatial units are recorded and explicitly converted from meter/mm/micron to mm;
unknown units fail closed. Public adapter and per-prediction lineage must match
the exact CineMA source, config, weights and class map.

A5 uses the prepared student input array and the frozen student's predictions
and final features. Before loading weights, its checkpoint manifest must be
hash-verified, non-synthetic, explicit A0 without external cache/bank ancestry,
and exactly match the source TRAIN data hash, L/U patient IDs and label budget;
audit reads must stay inside L. The loaded model must contain no memory module.
A real A5 bank therefore still requires a fresh matched A0 checkpoint; an
operator profile with manufactured inputs does not satisfy that requirement.

Default bank construction retains no dense A1 maps. Optional A1 collection now
streams probability/support arrays to per-patient NPY memmaps. Final indexing
requires the exact patient set and all timepoints once each, checks native shapes
and boolean support, and hashes the payloads. Partial coverage cannot become a
full A1 cache. The integration owner independently reports support payload
verification and crop-aligned runtime gating; that consumer is reviewed by the
integration reviewer rather than certified by this adapter/builder report.

Final independent focused command:
`rtk python -m pytest -q tests/test_memory_bank_builder.py tests/test_cinema_teacher.py`
returned **40 passed** (26 builder + 14 adapter). One intermediate refactor run
had five failures from an undefined `model` in a metadata-only A0 checker; the
owner corrected it, and the final run above is green. No failed result was
upgraded into a success.

Decision: **APPROVED FOR BOUNDED ENGINEERING RUN**. No remaining blocker was
identified in the reviewed teacher-adapter/builder scope. Full real A1 coverage,
real A5 bank construction, patient efficacy, and unseen-ACDC generalization remain
unestablished. The upstream ACDC patient-overlap status remains UNKNOWN.

Final reviewed `self_audit/cinema_teacher.py` SHA256: `7f0743d298ebfefa7b8fd347356b54792297106189a5cfe97839985baf93ef52`.

Final reviewed `self_audit/memory_bank.py` SHA256: `3cf3bbe146ccc14a268cffbf948aeaf0f11cc80933c61b1ae4f0afc0e828bbac`.

Final reviewed `scripts/prepare_teacher_images.py` SHA256: `d3968c9ece9ab1ee1639957dbb31c12adf1fa48d0680de4cbca6dcdebddc11d5`.

Final reviewed `scripts/build_teacher_memory.py` SHA256: `a2ad8eee4f5e84c63a99e6eda8fd76c2116ae1ac10e0191f30e4f13e0c9f0ea1`.

Final reviewed `tests/test_cinema_teacher.py` SHA256: `b364dd59d8d7fb631f5f815be2d1ba9c00ebfb8b57492159c2ea16706bb6e62a`.

Final reviewed `tests/test_memory_bank_builder.py` SHA256: `aa4326ab163b6a4c869219f137fe8e4323c39fee40018332cb906209e042a632`.

## Publication snapshot reconciliation — 2026-10-24

Independently recomputed current file digests and matched all six files against
the final reviewed digests above. **No source or test drift was found.** The
adapter's lines 343–345 already use `math.prod` for all three allocation-count
products; that overflow fix is included in the reviewed adapter digest, not a
post-review change.

Current executed-source digest: `7f5856216d59bc787633f8a703fa362cf83582427b830ab1ce09c15aeb55e8a9`.

| Reviewed file | Current SHA256 | Comparison |
| --- | --- | --- |
| `self_audit/cinema_teacher.py` | `7f0743d298ebfefa7b8fd347356b54792297106189a5cfe97839985baf93ef52` | MATCH |
| `self_audit/memory_bank.py` | `3cf3bbe146ccc14a268cffbf948aeaf0f11cc80933c61b1ae4f0afc0e828bbac` | MATCH |
| `scripts/prepare_teacher_images.py` | `d3968c9ece9ab1ee1639957dbb31c12adf1fa48d0680de4cbca6dcdebddc11d5` | MATCH |
| `scripts/build_teacher_memory.py` | `a2ad8eee4f5e84c63a99e6eda8fd76c2116ae1ac10e0191f30e4f13e0c9f0ea1` | MATCH |
| `tests/test_cinema_teacher.py` | `b364dd59d8d7fb631f5f815be2d1ba9c00ebfb8b57492159c2ea16706bb6e62a` | MATCH |
| `tests/test_memory_bank_builder.py` | `aa4326ab163b6a4c869219f137fe8e4323c39fee40018332cb906209e042a632` | MATCH |

Acceptance remains **APPROVED FOR BOUNDED ENGINEERING RUN** for this exact
snapshot. No source files were edited. No tests were rerun because there was no
actual delta; the final independent 40-pass focused result remains applicable
to these byte-identical source/test files. Real pinned-weight load/parity and the
bounded three-patient, one-timepoint runtime receipts are separate coordinator
evidence and were not re-executed or promoted into efficacy evidence by this
hash reconciliation.
