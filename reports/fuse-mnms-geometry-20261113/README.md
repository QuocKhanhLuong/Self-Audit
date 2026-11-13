# M&Ms open345 geometry adjudication

Evidence: DATA_HEADER / SOFTWARE_ONLY. No target model was evaluated and no
VAL/TEST target was opened. Only the locked target L8 masks were inspected
after the immutable cohort lock was written. Track A runs in its existing pinned checkout.
This report supplements, and does not rewrite, the earlier blocked raw audit.

The organizer's [download page](https://www.ub.edu/mnms/download.php) describes
345 publicly downloadable cases from Spanish and German centres; 30 Canadian
cases require a separate collaboration. Local archive partitions contain
175 TRAIN, 34 VAL and 136 TEST. Their IDs match the supplied 345-row metadata
CSV exactly. This is the open release, not the full original challenge cohort.
TEST image headers can be read inside Testing-003.zip without unpacking its
8.97 GB contents or opening mask members.

The organizer's [data page](https://www.ub.edu/mnms/data.html) specifies source
values 1=LV, 2=MYO, 3=RV. Canonical values BG/RV/MYO/LV therefore use
0→0, 1→3, 2→2, 3→1. No numeric labels were inferred from image appearance.

[Campello et al., Table III](https://diposit.ub.edu/server/api/core/bitstreams/f2f7dc43-becc-4554-910d-360ed7fd339c/content)
reports acquisition resolution and slice dimensions in mm. The raw files all
leave spatial units unset. The named policy records mm as a dataset-level
interpretation grounded in that acquisition documentation; it does not claim
that original DICOM metadata was recovered. Source document hashes are in
SOURCES.json; the policy also binds the full image/archive source inventory.

The selected sform agrees with pixdim in scale. A polar projection restores
orthogonal directions while preserving pixdim, translation and handedness.
This is an explicit bounded metadata approximation. It is NOT described as
proven float32 rounding. Both raw and effective affine are retained. Limits
are Gram error ≤1e-3 and maximum native-corner displacement ≤0.05 mm. Every
result must satisfy the existing canonical geometry validator. Invalid
VAL/TEST still blocks. Original headers, voxel arrays, labels, ED/ES indices
and split membership remain unchanged; output is a new geometry sidecar.

## Result

The effective geometry gate is **GEOMETRY_READY: 345/345**, with no exclusions
and no blocked IDs. Raw header findings are retained separately.

| Partition | Header count | Effective geometry pass | Excluded |
|---|---:|---:|---:|
| TRAIN | 175 | 175 | 0 |
| VAL | 34 | 34 | 0 |
| TEST | 136 | 136 | 0 |

Maximum corner displacement is 0.02751569 mm.
C8J7L5 is 0.02249006 mm. The original strict 1e-4 orthogonality predicate
also flags C8O0P2 in TEST; both satisfy the new bounded metadata policy.
None of these numbers measures model efficacy. Missing spatial-unit tags
remain visible in all 345 raw records.

All 25 image-only TRAIN records have blank index ED/ES and a 0/0 metadata
sentinel. They retain effective phases null and phase_status UNANNOTATED.
No phase is inferred and no U cine frame is excluded using this sentinel.

Target L8 was locked with the existing sha256(seed:patient_id) selector,
seed42, on the 150 labeled TRAIN candidates before any target mask read:
A2N8V0, A9E3G9, B9E0Q1, C1K8P5, D3F9H9, E3T0Z2, E5F5V7, J6P5T8.
Selected-ID SHA256: `3c86e7c6cad5193443a473d81f512727a7ffdb1ab2fd4037f4007d5212653af7`.
The immutable lock file SHA256 is
`ab239da264645bf82e246b8db6bb55dc5f3314260ef2dcccadc6c9e49bd579f4`.
All 16 selected ED/ES mask checks pass shape/raw-affine/label-domain/class-remap
checks. Outside-L8 mask reads: 0. VAL/TEST masks remain unopened.

Files: POLICY.json pins the input inventory and unit evidence; adjudicated/
contains all raw/effective affine rows, CSV and geometry_manifest.json; l8/
contains the cohort lock and permitted mask-map checks. EXECUTION.json binds
the exact locally/remotely matching code SHA used for this CPU data audit.
Snapshot acquisition aborts before producing a receipt on corrupt/unreadable
source headers. This intentional fail-closed behavior cannot silently exclude
VAL/TEST patients. SHA hashing streams compressed raw file/archive bytes;
no image voxel arrays or TEST GT members were opened/materialized by the
header audit.

## Remaining scope

The two reported geometry blockers are resolved by the explicit amendment.
The runtime B/B0 training manifests, normalized image arrays, B/B0 config and
engine extension, zero-shot evaluation, target training and target TEST remain
NOT_RUN. geometry_manifest.json is a metadata sidecar, not a runtime training
manifest. This task does not claim B/B0 launched or ready to launch. Track A
continues unchanged in its pinned checkout.

Software verification output is attached separately.

Final software checks: check_repo PASS; full pytest (default OMP environment)
`915 passed, 2 skipped, 3 subtests passed in 243.80s (0:04:03)`; synthetic FUSE
T→J→VAL3→export smoke PASS with all three cold-load modes bitwise equal.
These are SOFTWARE_ONLY. The final v2 adjudication summary pins geometry
and per-patient report SHA; the v2 L8 run also checks their agreement before
mask access. Symlinks to another patient are rejected before opening a mask.
The new v2 outputs preserve the original target L8 lock byte-for-byte. Earlier
intermediate receipts remain on the server, unmodified. No A/A3D/Q3D source
was changed. At 13:04:32 UTC, A was RUNNING at T epoch125/500, exposure2000.

Independent read-only review found no remaining findings after the v2 receipt
refresh; see REVIEW.md.
