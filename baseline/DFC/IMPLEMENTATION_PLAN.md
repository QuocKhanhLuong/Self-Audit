# DFC Cardiac Baseline Implementation Plan

Status: design document only. No wrapper, environment, adapter, tests, or benchmark has been implemented or executed by creating this plan.

Basis: [DFC Cardiac Benchmark Audit](../DFC_CARDIAC_BENCHMARK_AUDIT.md), dated 2026-09-17. Repository identity was checked again for this plan: branch `master`, clean initial worktree, HEAD `181318ad40dbfb5c0add8580a05c30e5e3a7ad58`, matching the audited official DFC source. Origin is the Azios1010 fork. This plan is the only file added.

DFC reference: `kanezaki/pytorch-unsupervised-segmentation-tip@181318ad40dbfb5c0add8580a05c30e5e3a7ad58`. FreeMask reference policy: `QuocKhanhLuong/Self-Audit@96c32b10fc7b8e09b48822e10ae9eb6cc149e253`, restricted to `src/self_audit_maskfree/` and its associated mask-free configuration, as inspected in the audit. No new FreeMask audit is claimed. The older supervised `src/self_audit/` pipeline is outside scope.

`FROZEN` below means a required design decision. `UNRESOLVED` means evidence, an external artifact, or a specification is still missing. `DEFERRED` identifies a real-data/server gate whose required data or execution resources are unavailable; it is not a local implementation failure. Proposed future paths are not existing or delivered files. Gates block only their stated dependent acceptance scope: unresolved real-data/server gates do not block fixture-based local P0.2-P0.7 after P0.1a PASS. They remain mandatory for scientific readiness.

## 1. Scope and scientific objective

Adapt DFC into a reproducible cardiac MRI baseline on ACDC and M&Ms while preserving its scientific identity: per-image, transductive, test-time optimization with iterative self-label clustering.

Track A evaluates each held-out image by constructing a fresh CNN, optimizing only on that image, exporting an anonymous partition, and releasing learned state. There is no dataset-level encoder training, dev checkpoint selection, pretrained encoder, or shared learned initialization. Optimizing image intensities at test time is permitted and must be disclosed and timed; using its GT is prohibited.

The benchmark shell supplies common data accounting, semantic adaptation, stacking, prediction freezing, and isolated evaluation. These are benchmark services, not new DFC contributions. The comparison must retain the distinction between DFC's per-image optimization and FreeMask's representation, candidate-partition, predictive-evidence, and challenge/selection machinery.

This plan specifies P0 trustworthy raw partitions, P1 the primary semantic benchmark, and P2 optional sensitivities. Creating this document does not authorize or claim execution of those phases.

## 2. Frozen design decisions

| Decision | FROZEN contract |
|---|---|
| Primary profile | `DFC-Direct-2D-Default-MinL3` on both datasets |
| Input | Central MRI slice `[1,H,W]`; one-sample batch `[1,1,H,W]` |
| Model | Fresh `MyNet(input_dim=1)` for every sample; 101,800 trainable parameters |
| Width and depth | `nChannel=100`, `nConv=2`; two 3x3 feature blocks plus final 1x1 convolution |
| Default optimization | `maxIter=1000`, `minLabels=3`, `lr=0.1`, both loss weights `1` |
| Optimizer | SGD, momentum `0.9`, weight decay `0`, dampening `0`, Nesterov disabled; no scheduler |
| BN | Official placement and defaults; training mode throughout optimization and final forward |
| Stop | Count pre-update argmax labels; update; then check that count against `minLabels` |
| Final partition | Fresh post-update headless forward; retain original anonymous channel IDs |
| Seeds | `benchmark_seed=42`, versioned SHA-256 sample-specific derivation; `split_seed=42` |
| Data | One frozen manifest shared exactly with FreeMask; no independent DFC split |
| Precision policy | Explicit float32 baseline, no AMP; deterministic backend policy validated before freeze |
| Semantic processing | External shared deterministic `cardiac_adapter_v1`, then stacking and isolated evaluation |
| Exclusions | GT, scribbles, reference training, pretrained checkpoints, FreeMask audit/selection logic in primary generation |
| Runtime | Real-data preflight and projected full wall time with margin <=72 hours before full launch |

MinL3 is the official default. `minLabels=4` does not force four clusters; cluster cardinality is not semantic anatomy cardinality. Neither task cardinality nor a Dice result can change the primary profile.

The intended grid is `224x224`, subject to verification that this is the actual frozen common benchmark grid. A mismatch blocks scientific data-contract acceptance; DFC must not silently resample to an independently chosen grid. Local implementation may use a synthetic common-grid fixture after P0.1a PASS; real grid verification is DEFERRED to P0.1b/server validation when data are unavailable.

## 3. DFC core vs benchmark-shell boundary

```text
Frozen shared image manifest
  -> image-only central MRI slice and declared preprocessing
  -> deterministic sample-specific seed
  -> fresh MyNet(input_dim=1) and SGD
  -> direct DFC optimization and official stopping
  -> fresh post-update train-mode forward
  -> anonymous integer partition [H,W] and raw provenance
  -> raw artifact freeze

External shared benchmark shell, P1:
  anonymous partition + permitted image/geometry inputs
  -> cardiac_adapter_v1
  -> BG / RV / MYO / LV / VOID + validity
  -> semantic slice assembly
  -> semantic prediction/config freeze
  -> isolated GT evaluator
```

Core computation accepts an image tensor, immutable profile, seed, and explicit device; it returns anonymous labels and optimization diagnostics. Identifiers and image geometry are carried by the wrapper and never become anatomical supervision.

The core has no adapter dependency, semantic class assignment, GT reader, best-cluster selection, or oracle permutation. It does not emit calibrated anatomical probabilities. The adapter is not described as part of the DFC method or a DFC contribution.

FreeMask's auditor, predictive evidence, candidate challenges, `O_fit`, `O_select`, and `O_verify` must not enter any DFC generation call graph. Common image discovery and geometry policy may be shared independently of those mechanisms.

## 4. Shared patient/data protocol

P0 separates knowledge of the required contract from possession of the real datasets. The following block is the source of truth for local/server dependency scope throughout this plan:

```text
Local development contract
--------------------------
P0.1a protocol/schema: can PASS without real data
P0.2-P0.7: may be implemented/tested with fixtures after P0.1a PASS
No scientific claims from fixture data

Server/scientific contract
--------------------------
P0.1b real manifests: required
P0.8 real preflight: required
server parity: required
<=72h gate: required

Scientific P0 ready:
only after all server/scientific gates pass
```

**P0.1a — protocol/schema verification.** Verify the shared manifest schema and data-contract interface against the authoritative `QuocKhanhLuong/Self-Audit` FreeMask implementation at `96c32b10fc7b8e09b48822e10ae9eb6cc149e253`, specifically its `src/self_audit_maskfree/` discovery, split, dataset, geometry, and firewall behavior, together with this plan. This verification needs the pinned source, not real ACDC/M&Ms images. Record the source revision and a field/behavior mapping covering patient-level split semantics, `split_seed=42`, canonical sample identity, frame/slice/context meaning, geometry fields, image-only lineage, GT firewall, shared-grid interface, and scientific-versus-mock manifest distinction.

P0.1a PASS means the wrapper knows exactly which manifest/data contract it must consume. It does not mean any real scientific manifest or cohort exists. Synthetic fixtures, fake patient trees, schema-valid mock manifests, and deterministic MRI-like arrays may exercise this contract; their provenance must explicitly identify them as fixtures. After P0.1a PASS, local P0.2-P0.7 may proceed while P0.1b is DEFERRED.

**P0.1b — real shared manifest materialization.** This is a scientific/server gate requiring accessible real image-only roots for both ACDC and M&Ms. Resolve the authoritative artifacts as follows:

1. Locate an existing valid FreeMask scientific manifest and its generation receipt. Verify its source policy SHA, `split_seed=42`, content hashes, patient-disjoint splits, image-only lineage, and geometry/grid contract. Reuse the immutable artifact exactly if valid.
2. If no valid artifact exists, build exactly one shared benchmark manifest per dataset using the pinned FreeMask discovery/split policy above and `split_seed=42`. Both FreeMask and DFC must explicitly consume the same artifact/hash for that dataset. No independent DFC split generator is allowed.
3. If an existing manifest lacks required shared fields, preserve it as source provenance and resolve one common, versioned replacement or companion data contract consumed by both methods. Do not silently rewrite it or introduce a DFC-only scientific inventory.
4. Reject annotation-derived selection for the strict primary. A legacy ED/ES-only or mask-cropped source cannot silently become an image-only cohort. Its use would require a separately declared protocol, outside this frozen primary.

Absent real local image roots, P0.1b is DEFERRED, not FAIL and not a blocker for local P0.2-P0.7. Missing data never permits a mock manifest to be promoted to a scientific manifest. When real validation is attempted and provenance, content, or shared identity is invalid, report FAIL rather than disguising that validation failure as a deferral.

Required per-record or unambiguously referenced manifest fields, specified in P0.1a and validated on real artifacts in P0.1b:

| Group | Fields and validation |
|---|---|
| Identity | Canonical `sample_id`, `dataset`, `patient_id`, study/volume identity; unique sample IDs |
| Cohort | Train/dev/test membership; all frames/slices of a patient remain in one split |
| Location | Frame index, central slice index, depth/frame axes, declared context indices |
| Sources | Image-only source identity/path, content hash, decoder/provenance policy, deduplication lineage |
| Geometry | Native/stored shape, affine availability, orientation, spacing and units validity, inverse transform |
| Grid | Whole field of view, target grid, resize policy and version; no mask-derived crop |
| Lineage | Discovery/split policy SHA, split seed, schema version, manifest ID and content digest |

Execution metadata must distinguish `development / fixture mode` from `scientific_run=true`, with explicit `mock/fixture` designation. Bind this designation into manifest provenance or a hash-bound companion receipt if adding fields would change an existing immutable FreeMask manifest. Fixtures use separate artifact namespaces and retain their synthetic lineage in exports and freezes. A valid mock schema or checksum alone is never evidence of a real cohort.

**Scientific runtime guard:** before a scientific sample runs, require a real non-mock manifest, valid manifest hash, existing real image roots, resolvable and matching source-image hashes, verified shared FreeMask/DFC manifest identity, and an explicit `mock/fixture=false` designation. Missing, contradictory, or unverifiable evidence rejects scientific execution; there is no fallback to fixture data. Flipping a flag cannot override synthetic source provenance. Both the raw-generation and freeze/launch entry points must enforce this distinction; fixture output cannot resume into or satisfy a scientific run.

No GT paths, annotation sidecars, segmentation labels, mask statistics, diagnostic grouping, or semantic scores enter the generation manifest. Discovery must follow the pinned policy, including image-only frame enumeration and its dataset-specific official-cohort rules, rather than implementing a fresh 70/15/15 approximation.

The audited policy enumerates each acquired target slice and all acquired frames where available. A 'central slice' is the target of a context stack, not a selection of only middle ventricular slices. Context indices are `[max(z-1,0), z, min(z+1,Z-1)]` within the same frame and patient.

For Track A, the primary generation inventory is the shared held-out test records for both datasets, including all declared target frames/slices rather than selecting only records with GT. Train/dev membership remains available for provenance, image-only engineering preflight, and future Track B. No global parameters are learned from train records. Evaluator reference availability may define a separately reported scoring subset after freeze, but must not change generation membership.

UNRESOLVED: authoritative real manifest paths/hashes, actual data roots and image-only lineage, exact record counts, real common grid/geometry contract, and confirmation that FreeMask consumes the same artifacts. These block scientific data acceptance at P0.1b, not local protocol verification or fixture implementation. Real materialization/validation stays DEFERRED while the required roots are unavailable.

## 5. Primary and secondary DFC profiles

The primary resolved profile must contain all fields below; scientific fields cannot be overridden under the primary profile ID.

| Field | Primary value |
|---|---|
| `profile_id` | `DFC-Direct-2D-Default-MinL3` |
| Input context | Central slice only, `[1,H,W]` |
| `input_dim` | `1` |
| `nChannel` | `100` |
| `nConv` | `2` |
| `maxIter` | `1000` |
| `minLabels` | `3` |
| `lr` | `0.1` |
| `stepsize_sim` | `1` |
| `stepsize_con` | `1` |
| Optimizer / momentum / weight decay | SGD / `0.9` / `0` |
| Dampening / Nesterov | `0` / `False` |
| Scheduler / clipping / AMP | None / none / disabled |
| Output policy | Official `visualize=0` semantics; no GUI |
| BN mode | Train, including final forward |

Secondary definitions are fixed relative to this profile:

| Profile | Only scientific changes | Purpose |
|---|---|---|
| `DFC-Direct-2D-MinL4` | `minLabels=4` | Task-cardinality sensitivity |
| `DFC-Direct-2.5D-MinL3` | Three-slice context, `input_dim=3`, declared stack statistics | Input-context sensitivity |
| `DFC-Direct-2.5D-MinL4` | Both preceding changes | Optional combined sensitivity |

2.5D retains Conv2d, BN, argmax, CE, and in-plane continuity. It has 103,600 parameters and predicts only the central slice. It is not 3D DFC. Keep all 100 output channels; never reduce the head to four anatomical classes.

No profile selection or replacement of the primary is permitted using GT, Dice, or a favorable seed. Secondary results have separate experiment and artifact identities.

## 6. Reproducibility and per-sample seed contract

Freeze `seed_derivation_version = "dfc-sample-seed-v1"` with the following byte-level specification for future implementation:

1. `sample_id` is the exact canonical, immutable string from the shared manifest, unique across both datasets. If the upstream ID is not globally unique, define the dataset-qualified canonical ID in the common manifest contract, not using local paths or row order.
2. Serialize the JSON array `["dfc-sample-seed-v1", 42, sample_id]` as UTF-8, with ASCII escaping enabled, no optional whitespace, no trailing newline, and the elements in the displayed order.
3. Compute SHA-256 over those bytes. Interpret the first eight digest bytes as an unsigned big-endian integer; mask to the lower 63 bits. This is `sample_seed`.
4. Record the derivation version, payload identity, digest, benchmark seed, sample seed, and sample ID. Verify test vectors and scan the frozen inventory for seed collisions; a collision requires resolution before freeze, never an order-dependent reseed.

For each sample, seed Python `random` and PyTorch CPU with `sample_seed`; seed all relevant CUDA generators with it. Seed NumPy's legacy RNG with the two ordered 32-bit words `[sample_seed & 0xffffffff, sample_seed >> 32]`; any independent NumPy generator must be created explicitly from the same declared seed policy. There are no stochastic augmentations.

The scientific model is initialized on CPU as in the original script, after resetting CPU PyTorch RNG, then transferred to its worker device. No unrelated torch-random operation may occur between that reset and model construction. Test fixture creation and visualization must use separate RNG streams.

Do not use Python `hash()`, manifest index, process order, worker ID, retry number, or absolute file path. Reuse the same sample seed across matched profile comparisons; input-channel changes still alter parameter shapes and subsequent initialization draw positions, so identical weights across 2D/2.5D are not promised.

Candidate backend policy: float32, deterministic algorithms required, cuDNN benchmarking disabled, TF32 disabled for the reference baseline, and any required deterministic CUDA environment settings established before CUDA initialization. Freeze their exact supported settings with the environment after validation; unsupported deterministic execution fails instead of silently falling back.

Determinism claims are scoped to the recorded validated environment/hardware. Cross-GPU bitwise equivalence is UNRESOLVED until tested. Scheduler placement is recorded; a hardware change cannot silently be treated as an identical reproducibility context.

## 7. Per-image optimization protocol

Local P0.4 implements and validates the same protocol using schema-valid fixture manifests and deterministic synthetic images after local P0.2/P0.3. All provenance is marked development/fixture; no real cohort or scientific performance claim follows. Scientific execution additionally requires the section 4 runtime guard and server qualification.

For every sample, including retries:

1. Validate manifest membership, source hash, image shape, finite input, immutable primary config, and device assignment. Load and preprocess without GT.
2. Set sample RNG state and instantiate a fresh CPU `MyNet(input_dim=1)`. Transfer it to the explicit device and call `train()`.
3. Instantiate fresh SGD with the primary values. No weights, momentum, BN running statistics, or optimizer state can be loaded from another sample.
4. Optimize a batch of one. Each iteration zeroes gradients, forwards the image, and flattens final BN responses to `[HW,100]`.
5. Form per-pixel channel argmax pseudo-labels, count unique IDs, calculate similarity and spatial continuity, backpropagate, update, and check the pre-update count as defined in section 8.
6. Run the official final forward, export the anonymous map and diagnostics, then release model/optimizer/graph references before processing another sample.

The feature blocks remain `Conv3x3 -> ReLU -> BN`, repeated twice for `nConv=2`, followed by `Conv1x1 -> BN`. All convolution biases and framework BN defaults must match the pinned source in the validated runtime. Targets are integer argmax indices with no gradient; all 100 response channels remain allocated.

For final BN responses `R[H,W,C]`, the protected objective is:

```text
L_sim = CrossEntropyLoss(reduction="mean")(R.reshape(H*W,C), argmax_channels)
L_row = mean(abs(R[1:,:,:] - R[:-1,:,:]))
L_col = mean(abs(R[:,1:,:] - R[:,:-1,:]))
L_total = 1 * L_sim + 1 * (L_row + L_col)
```

Each directional mean uses its own element count. Continuity operates on real-valued post-BN responses; there is no softmax, discrete-label smoothing, or through-plane term. Preserve official arithmetic and reduction semantics in the wrapper, even if logging is reorganized.

Workers may reuse only non-learned resources such as immutable image caches and parsed config. Avoid concurrent samples sharing process-global RNG via threads: use independent worker processes or equivalent isolated RNG execution proven by tests. Multiple images must not be batched through one shared model or BN.

On nonfinite loss, OOM, unreadable input, or another exception, emit an explicit failed attempt. Never reuse partially updated state, alter the seed/profile, or drop the sample. Operational retries restart the entire image and preserve all attempts. The finite retry limit is frozen before preflight/launch; its exact value is UNRESOLVED.

## 8. Stopping and final-output semantics

The required logical order is:

```text
forward
-> argmax pseudo-labels
-> count unique labels before the update
-> similarity loss and two continuity losses
-> backward
-> optimizer.step()
-> check previously measured nLabels <= minLabels
```

The original source calculates continuity tensors before argmax; those independent calculations have no parameter update between them. The wrapper must preserve the same tensor values, graph, and loss reductions while enforcing the above assignment/update/stop relationship. Reference parity governs any rearrangement.

`maxIter=1000` means at most 1,000 optimizer updates. `minLabels=3` is a threshold, not exact-K, a lower bound on final labels, or a count of connected regions. Even an initial count below threshold incurs one update before stopping. If a pre-update count moves from five to three between iterations, the iteration observing three still updates before exiting.

After the last update, perform one fresh forward in training mode and argmax to obtain the saved partition. No final `eval()` call is allowed. This final pass updates BN buffers just as the official headless path does. Default implementation should retain ordinary gradient mode until an explicit `no_grad()` parity test confirms identical logits, partition, and BN state; omission of the unused graph must not alter state semantics.

For a successful run with `N` updates, expect `N+1` model forwards, including the final pass. Record both `pre_stop_active_ids/count` from the last optimization forward and `final_active_ids/count` from the post-update forward. They may differ in either direction.

Use `stop_reason=label_threshold` when the official check fires, including on the last allowed update; otherwise use `max_iterations`. Exceptions have failure status and a separate reason, not a fabricated converged partition. Do not select an earlier best-looking iteration or force a particular final label count.

## 9. Cardiac preprocessing adaptation

The original DFC direct path (`demo.py:68-70`) reads OpenCV BGR `uint8` and
divides by `255`; it has no percentile, mean/std, MRI, crop, or geometry
normalizer. That byte-image rule is retained only in the source parity harness
and is not silently applied to float NIfTI MRI.

For Self-Audit ACDC v3, the adapter therefore uses the checked-in source
normalization (`scripts/preprocess_acdc.py:19-25` and
`src/self_audit/data/common.py:188-204,529-546`): decode the full image-only
frame, clip at 0.5/99.5 percentiles, apply one volume-wise population z-score,
then select the central plane for direct 2D DFC. The endpoint context remains
manifest provenance and is not used as an additional DFC statistic. There is
no DFC-specific re-normalization, no GT/ROI support, and no per-slice or
dataset-global fit. Record the source normalization version, context indices,
grid transform, and source identity. Constant volumes use the source epsilon
floor and remain in accounting.

The audited FreeMask full-input path uses whole-field area-based resizing. P0.1a verifies the pinned spatial-policy interface and behavior for implementation; local P0.3 may implement and test the 2D loader, optional 2.5D loader, normalization, geometry, context, and strict schema validation against mock manifests and a synthetic common-grid fixture. Real shared-grid/geometry verification is DEFERRED to P0.1b/server validation when images are unavailable. Scientific P0.3 passes only with validated real shared manifests, real source images, and their real common grid/geometry contract. No method-specific crop or invented affine is allowed. Reject a scientific mismatch rather than substituting an unrecorded transform.

FreeMask's internal fitting/selection/verification preprocessing is not
imported. Shared cohort/FOV/grid does not imply byte-identical method inputs:
DFC consumes one source-normalized central plane and preserves the source stack
indices only as metadata. No masks, ROIs, GT-derived statistics, or withheld-
observation roles enter DFC normalization.

## 10. Raw output and provenance contract

Use lossless `raw_cluster_map.npy` containing a C-contiguous little-endian int32 `[H,W]` array with original channel IDs in `[0,99]`. Load with object/pickle support disabled. Keep a JSON metadata record beside it. Color images, if ever produced, are optional downstream previews and never the scientific artifact.

The raw schema must include:

| Group | Required fields |
|---|---|
| Identity | `dataset`, `patient_id`, `split`, `frame`, `slice`, `sample_id`, study/volume ID, manifest ID/hash |
| Execution scope | Development/fixture or `scientific_run=true`, explicit mock/fixture designation, hash-bound source-lineage and qualification receipt identities |
| Partition | `raw_cluster_map` relative artifact path, `shape`, `dtype`, array-content SHA-256, file SHA-256 |
| RNG | `benchmark_seed`, `sample_seed`, `seed_derivation_version`, seed payload/digest |
| Configuration | Complete resolved scientific config, profile ID, canonical config hash; separate operational config |
| Code/environment | Official DFC SHA, wrapper SHA, wrapper source digest, environment ID/lock hash, precision/backend settings |
| Optimization | Optimizer update count, forward count, stop reason; optional per-iteration image-only diagnostics |
| Labels | Pre-stop active IDs/count, final active IDs/count; distinguish missing fields on failure |
| Final semantics | Final-forward policy, gradient-mode policy, `BN_mode=train` |
| Input | Source image identity/hash, manifest record hash, preprocessed tensor checksum |
| Geometry | Native/stored dimensions, axes, affine/spacing validity, source/target grid, context and transform metadata |
| Runtime | Total elapsed time, load/optimization/final-forward/export time, peak allocated/reserved VRAM, GPU identity/driver |
| Accounting | Success/failure, structured error, attempt ID, retry history and terminal status |

Use a versioned canonical JSON encoding for scientific hashes: sorted keys, UTF-8, fixed separators, finite JSON numbers, and no timestamps or worker paths. Config hashes cover all behavior-affecting parameters, normalization, seed policy, and final-forward policy. Environment and input identity are separately bound into the run identity. Never omit a behavior-changing backend option as merely operational.

Distinguish reproducible scientific payload hashes from complete receipt hashes: elapsed time and timestamps legitimately change, but partition, input, seed, and scientific configuration must match on a repeat run. The array-content digest includes an unambiguous shape/dtype header plus contiguous label bytes; define golden serialization fixtures before freeze.

Write to isolated sample/attempt paths with an atomic completion step. No fixed `output.png`, silent overwrite, or concurrent writers to one artifact. A resume skips only an artifact whose manifest, source, config, environment compatibility, and checksums validate. Successful retries must retain the failed-attempt ledger.

Raw freeze binds the exact expected inventory, per-sample status, raw files, metadata, config, code/environment identity, and hashes. Verify every referenced byte, not merely the presence of files. A one-byte mutation must invalidate verification. Completeness means every expected sample has an explicit terminal record, not that failures disappeared. Failure artifacts have no fabricated raw map. All-success scientific completion and complete accounting are distinct states.

P0.6 may PASS locally for int32 export, canonical hashes, attempt ledgers, mutation rejection, and resume checks using fixtures. These receipts must say local/fixture PASS; they neither certify a real cohort nor constitute scientific acceptance. Reject fixture-to-scientific resume/freeze promotion, even when numerical arrays and schema otherwise validate.

## 11. Shared semantic adapter boundary

P0 defines this interface only; no adapter implementation or GT metrics belong in P0.

Conceptual shared interface:

```text
inputs:
  immutable anonymous partition [H,W]
  permitted image-only central image/context, explicitly specified
  image geometry and shared adapter config/version
outputs:
  semantic map [H,W] with BG/RV/MYO/LV/VOID
  validity/abstention map [H,W]
  deterministic diagnostics and adapter provenance
```

The adapter must be deterministic, non-learned, GT-free, method-independent, and shared across anonymous baselines. It must not branch on DFC profile identity or numerical raw channel IDs. Permuting cluster IDs must leave both semantics and validity unchanged.

P1 must freeze the allowed image/context inputs, anatomical rules, tie handling, numerical semantic encoding, VOID encoding, and whether mappings only name/merge clusters or also split them. These are UNRESOLVED and block semantic benchmarking. No threshold may be fitted to GT. Prefer a naming/merging contract; any boundary-changing operation must be explicit, shared, and reported as adapter processing rather than a DFC capability.

A three-cluster DFC partition may not separate four semantic roles. Preserve that limitation and abstain where the frozen adapter requires it; do not silently reconstruct anatomy to repair the primary. Adapter-only diagnostics and raw cluster counts should accompany semantic results.

Do not reuse FreeMask's auditor, predictive-evidence scores, candidate generation/challenges, observation roles, or selected predictions to supply semantics for DFC. Location/ownership of the shared adapter package and evidence that it is reused by the other anonymous baselines are UNRESOLVED.

## 12. Slice-to-volume assembly

After semantic adaptation, group by dataset, patient, acquisition/volume, and frame; order by manifest central `z` index; stack maps and validity into `[Z,H,W]`. Use the term semantic slice assembly or stacking.

DFC has no inter-slice reasoning. 2.5D provides neighbor input context but does not add inter-slice consistency constraints. No fusion or 3D consistency claim is permitted. Do not introduce voting, smoothing, connected-component cleanup across slices, or neighbor-based label correction in the primary.

Assembly validates every expected slice index, shape, frame identity, and transform. Missing or failed slices remain explicit in completeness and validity; they cannot become background by default or disappear from volume denominators. If a policy represents missing output as all-VOID, record the originating failure distinctly and freeze its metric treatment.

Use the shared inverse-grid contract and nearest-neighbor interpolation for categorical maps, with separately specified validity handling. Physical-space metrics require valid geometry and units. For geometry-less containers report stored-grid results; do not fabricate a native affine. Exact semantic encoding, validity-resampling rules, and partial-volume evaluation policy are UNRESOLVED until P1 specification freeze.

## 13. GT firewall

Generation and adaptation must run with reference-mask storage physically unavailable or access-denied. The runner API, config schema, and manifest accept no GT path, label tensor, scribble, mask crop, mask statistics, semantic validation score, or Dice feedback. Reject unknown scientific config keys instead of forwarding arbitrary legacy CLI arguments.

The primary CLI must have no reachable scribble switch, reference-directory option, checkpoint initializer, or `demo_ref.py` execution path. Official scribble mode is weakly supervised and remains only in the untouched provenance script. Reference-set training is image-only transfer but is scientifically different and is excluded from primary.

Use allowlisted image-only manifest entries and content provenance in addition to path-name checks: a filename filter alone cannot prove that an innocuously named file is not a mask. Annotation sidecars and mask-derived preprocessing sources fail primary data acceptance.

GT access begins only in a separate evaluator process after raw partitions, semantic predictions, and configuration are frozen with verified hashes. The evaluator receives read-only frozen predictions and separately mounted/reference-configured GT, writes metrics elsewhere, and cannot mutate predictions or trigger generation retries, seed changes, adapter changes, or checkpoint selection.

Firewall tests must cover execution with masks absent, denied forbidden reads, rejection of forbidden parameters, and invariance to evaluator-side GT mutation. Synthetic test labels are allowed only in evaluator tests and never supplied to generation. P0 runs no GT metrics.

P0.7 may PASS locally on fake patient trees and schema-valid mock manifests for GT firewall behavior, scribble/reference exclusion, processing-order independence, and failed-sample accounting. This is local implementation evidence only. Tests must also prove `scientific_run=true` rejects mock/fixture manifests, missing real roots, mismatched source hashes, unverified shared manifest identity, and absent/contradictory execution-scope metadata. The guard fails closed and never generates substitute synthetic inputs.

## 14. Environment and compatibility strategy

Create a dedicated candidate environment named `dfc-cardiac` during future P0. Do not inherit a CUTS environment.

Candidate direction is Python 3.10, NumPy 1.26.4, and one modern PyTorch/CUDA build compatible with RTX 4070 12GB, RTX 4080 16GB, and RTX 5070 Ti 16GB. Exact PyTorch, CUDA, cuDNN, driver minimum, OpenCV/MRI I/O, and test-package pins are **UNRESOLVED**. The audit identified Blackwell compatibility as a constraint for 5070 Ti; do not assume the older CUDA 12.1 candidate works across all three targets.

Use a single validated build across GPUs if practical. If that cannot be demonstrated, record separate environments and validation receipts; do not claim cross-device bitwise equivalence. No package is installed or upgraded by this plan.

Qualification has separate local and server/scientific scopes. After P0.1a PASS, P0.2 candidate-environment work and P0.5 reference parity may complete locally using deterministic fixtures without P0.1b. A local PASS is restricted to that candidate environment; it is not scientific-server parity.

Qualification sequence:

1. Build an isolated candidate environment and record all versions and GPU/runtime metadata.
2. Execute the pinned upstream reference harness and proposed wrapper in the same candidate runtime with controlled seeds and fixture inputs.
3. Pass architecture, initialization, one-step, loss, stop, final-forward, raw-partition, and independence tests locally on deterministic fixtures; record local candidate environment, tolerance policy, and local PASS/FAIL receipts. Local implementation parity does not require real MRI data.
4. Separately qualify the actual server/scientific environment against the pinned reference on each GPU type that will be used. After P0.1b, validate real loading and a real DFC smoke run, then execute P0.8 real-data preflight. These server gates remain required even after local parity PASS.
5. Freeze the scientifically accepted environment lock/export, backend policy, source hashes, tolerances, and server receipts. Keep local qualification records separately identified. Any subsequent environment change invalidates the relevant scoped qualification and runtime projection.

Historical PyTorch reconstruction is not mandatory. Comparisons establish preservation of pinned source behavior in the qualified runtime; do not claim untested bitwise equality to an unspecified 2020 runtime.

Preserve original scripts. Handle legacy `Variable`, `.data`, and `L1Loss(size_average=True)` through behavior-equivalent wrapper usage verified by parity. `np.int` is confined to the excluded scribble path. The production wrapper need not import unused torchvision or GUI OpenCV; the reference harness must account for original imports without importing/executing the full demo in production. Every compatibility adjustment requires a reason, source location, equivalence test, and recorded receipt.

No AMP, compilation, fused optimizer, architecture optimization, or lower-precision variant is silently introduced during environment work. Such variants belong to P2 and need separate identities and parity evidence.

## 15. Runtime and 72-hour feasibility gate

Audit estimates are analytical: 101,800 parameters for 2D, 103,600 for 2.5D, about 10 GFLOPs per 224x224 forward, and about 19 MiB for one float32 response tensor. One sample is expected to fit in 4070 12GB, 4080 16GB, or 5070 Ti 16GB memory, but VRAM safety and seconds/sample are **UNVERIFIED until measured**. Do not hard-code worker concurrency from these estimates.

P0.8 is real-data only and is blocked until P0.1b completes for ACDC and M&Ms, the actual server environment passes parity, and real loader/DFC smoke validation passes. With no accessible real roots, report P0.8 DEFERRED while local P0.2-P0.7 may still complete. Synthetic timing can test instrumentation but can never satisfy P0.8, validate safe scientific concurrency, or establish a scientific <=72h projection.

P0.8 must run a real-data preflight of 50-100 fixed real slices on the actual target grid and target GPU/environment under the full primary MinL3 configuration, including `maxIter=1000` and ordinary stopping. Freeze the pilot IDs before measurements, using manifest metadata and image-only geometry/context strata across both datasets. Prefer 100 total where available and represent both datasets, boundary/interior positions, and source-size strata. Do not select slices by GT, known Dice, favorable convergence, or prior speed. Freeze the actual pilot allocation before execution.

Record per sample and aggregate by dataset/device/concurrency:

- Mean, p50, p95, and maximum seconds per slice; end-to-end and stage timings.
- Optimizer updates per slice and stop-reason distribution.
- Load, optimization, final-forward, and export time; startup costs separately.
- Peak allocated and reserved VRAM, worker count, GPU model/driver, and competing GPU workload.
- Failures, OOMs, retries, and any operational stall; never discard failed timing observations.

Synchronize CUDA around measured GPU phases; use wall-clock end-to-end throughput as the authority for concurrency. Reset per-sample allocator peak statistics while recording cached reserved memory and process-level constraints. Measure a repeated deterministic subset to verify that the selected concurrency does not alter outputs on the same runtime/device.

Test increasing independent process counts only during preflight, with a separate model/optimizer/BN/RNG stream per sample. Freeze the highest validated safe configuration with measured throughput and memory headroom; the fastest nominal concurrency is not automatically the safest. Requalify if hardware, environment, source grid, or worker policy changes.

Projection must use the exact inventory and actual device schedule. For each assigned device `g`, use its measured concurrent wall seconds per completed sample `t_g` and assigned count `N_g`; estimate concurrent makespan as `max_g(N_g * t_g)`, including stratification where needed. Do not divide single-worker latency by worker count without a measured concurrency trial.

Define:

```text
T_base = measured inventory makespan + startup/I/O overhead not already included
         + planned adapter/assembly/freeze/evaluator overhead
         + declared retry/failure-handling allowance
T_budget = 1.25 * T_base
PASS only if T_budget <= 72 hours
```

The 25% minimum operating margin is frozen here; increase it before launch if pilot variability or workload evidence requires, never lower it to manufacture a pass. Publish a p95-based stress projection alongside the mean-based estimate. If the pilot is unrepresentative, concurrency unstable, or any included overhead cannot be bounded, the gate remains UNRESOLVED rather than passing on an optimistic number.

P0 supplies the raw-generation projection. P1 must add measured or defensibly bounded adapter/evaluation costs and refresh the end-to-end gate before scientific launch. The primary budget covers both datasets in the declared primary run. Include all samples/profiles/seeds in the projection if they are part of the same requested launch; optional P2 work cannot be hidden in the primary budget or used to shrink its inventory.

If the gate fails: **DO NOT launch the full benchmark.** Allowed responses are independent workers/multiple GPUs, verified I/O improvements, and headless execution followed by remeasurement. Do not lower `maxIter`, alter `minLabels`/loss weights, drop samples, or choose a fast seed to force a pass. If adequate resources remain unavailable, report the blocked runtime gate.

UNRESOLVED: real inventory counts, device availability, real timings, peak VRAM, safe concurrency, retry cap, final overhead measurements, and resulting <=72h receipt. These block scientific readiness/launch, not fixture-based implementation. Missing real data leaves P0.8 DEFERRED; a measured budget violation is FAIL. No runtime feasibility claim is currently validated.

## 16. Proposed module/file architecture

The inspected checkout contains only `demo.py`, `demo_ref.py`, `README.md`, `LICENSE`, and demo-image folders (`BBC`, `BSD500`, `PASCAL_VOC_2012`). There is no existing package, dependency lock, cardiac loader, or evaluator to extend.

Keep the wrapper small; use the following future structure rather than a training framework:

```text
demo.py                              # untouched upstream reference
demo_ref.py                          # untouched upstream reference
src/cardiac_benchmark/
  __init__.py
  manifest.py                        # shared artifact validation/policy bridge
  dataset.py                         # MRI decoding, context, normalization, geometry
  dfc_runner.py                      # narrow MyNet mirror, losses, per-image lifecycle
  output.py                          # raw array/metadata and attempt ledger
  provenance.py                      # seeds, canonical hashes, environment/config identity
  freeze.py                          # inventory binding, hash verification, raw/semantic freeze
scripts_cardiac/
  build_shared_manifest.py            # invoke pinned shared policy only if needed
  run_dfc.py                          # explicit device/profile/inventory entry point
  preflight.py                        # fixed pilot, concurrency measurements, budget receipt
  freeze_predictions.py              # raw in P0; semantic extension in P1
config/cardiac/
  dfc_direct_2d_minl3.yaml             # P0 primary
  dfc_direct_2d_minl4.yaml             # P2 sensitivity
  dfc_direct_25d_minl3.yaml            # P2 sensitivity
tests/cardiac/
  test_reference_parity.py
  test_data_contract.py
  test_seed_independence.py
  test_firewall.py
  test_artifacts.py
  test_runtime_gate.py
  test_adapter_assembly.py            # P1
cardiac_evaluation/
  evaluate.py                        # P1, separate process and GT dependency boundary
environment/
  dfc-cardiac.*                       # future validated environment specification/lock
```

P1 may add one thin shell module/entry point for the external shared adapter and assembly; its concrete package location is UNRESOLVED. Do not create a DFC-specific anatomical implementation and later call it shared. Add optional combined-profile config only when P2 is selected. Tests and configuration should remain explicit; no plugin system, model registry, generalized trainer, or distributed training layer is needed.

Because `demo.py` executes argparse/loading/training at import time and `MyNet` reads global `args`, production must not import that script directly. Preferred approach: a minimal attributed, structurally identical model definition inside `dfc_runner.py`, parameterized by immutable config, plus a small direct loop. Preserve the original source untouched and verify the mirror against the pinned class/loop in a test-only reference harness. This is a compatibility boundary, not a redesign. Retain the upstream MIT notice for copied code.

The test harness may extract only the pinned model definition into an isolated namespace and run a separately captured reference loop, and should also exercise the original headless script on a deterministic image fixture with test-only output interception. Avoid validating a wrapper against itself. Single-channel fixtures test the constructor-supported adaptation; RGB fixtures test the actual official loader/headless path.

All paths above are planned only. This task creates none of them.

## 17. Existing files/semantics to preserve

| Existing location | Preserve |
|---|---|
| `demo.py::MyNet.__init__`, lines 43-54 | Convolution kernels/stride/padding/bias, width/depth, BN construction; original constructor already accepts `input_dim` |
| `demo.py::MyNet.forward`, lines 56-66 | Conv -> ReLU -> BN feature blocks; final Conv -> BN |
| `demo.py`, lines 98-106 | CE and two separately mean-reduced L1 continuity losses |
| `demo.py`, lines 114-145 | SGD semantics; zero-grad/forward/argmax/self-label/loss/backward/update behavior |
| `demo.py`, lines 123-131 | Continuous response differences, argmax over channels, unique-label count |
| `demo.py`, lines 149-151 | Post-update check of the previously measured count |
| `demo.py`, lines 154-158 | Fresh headless final forward and argmax in training mode |
| `demo_ref.py` | Entire script retained as provenance; not invoked by primary |
| `README.md`, `LICENSE`, demo assets | Retain unchanged as upstream documentation/license/fixtures |

The primary network has two 3x3 convolutions and one 1x1 convolution; `nConv=2` is not two total convolutions. No change to channel count, BN placement, CE target meaning, loss normalization, optimizer momentum, or exact-K constraint is allowed under the primary profile.

Protect originals using baseline source hashes in parity/provenance tests. If a future compatibility issue truly requires changing upstream code, it needs an explicit separately reviewed patch and parity receipt; this plan does not presume that such an edit is necessary.

## 18. Existing components to wrap/replace

| Existing location | Problem | Replacement boundary | Scientific impact |
|---|---|---|---|
| `demo.py:69-73`, OpenCV input | BGR uint8 `/255`; no MRI container/geometry | `dataset.py` image-only MRI tensor loader | Declared cardiac preprocessing and single-channel adaptation |
| `demo.py:108-109,123`, `im.shape` | Shape derived from color image object | Explicit tensor `H,W` and checked geometry | None if loss dimensions/reductions match |
| `demo.py:132-136`, GUI | `imshow`/wait delay and headless incompatibility | Headless runner; optional external previews | Operational; preserve official headless output semantics |
| `demo.py:115,133-134,159-161`, palette | Random colors, only 100 entries, not raw cluster IDs | Lossless int32 array and separate metadata | Preserves partition identity rather than lossy presentation |
| `demo.py:161`, `output.png` | Fixed name and overwrite/concurrency collisions | Sample/profile/attempt-addressed atomic output | Operational; supports complete accounting |
| `demo.py:15,71-72,94-95`, `.cuda()` | Implicit availability/device | Explicit device per worker and recorded backend | Neutral only within validated runtime/hardware scope |
| `demo.py:17-40` and top-level execution | Unsafe import; legacy global argparse | Strict wrapper config and isolated runner API | Neutral subject to parity; no arbitrary legacy flags |
| `demo.py:93,115`, unseeded RNG | Initialization/output randomness uncontrolled | Versioned per-sample seeding in `provenance.py` | Declared reproducibility policy, no GT seed selection |
| `demo.py:147,161`, logging/output | No provenance, timings, geometry, or freeze | `output.py`, `provenance.py`, `freeze.py` | Operational auditability; no model feedback |
| `demo.py:75-90,139-140`, scribbles | Human annotations and annotation-derived threshold | Unreachable in primary API/CLI | Enforces no-label primary |
| `demo_ref.py`, reference transfer | Shared training/state and different defaults | Separate optional P2 executable/profile | Distinct scientific experiment; excluded from primary |

## 19. P0 implementation tasks

**P0 — trustworthy raw DFC partitions.** No semantic adapter, GT metrics, or reference mode is implemented or executed in this phase.

| Task | Deliverable | Dependency / completion evidence |
|---|---|---|
| P0.1a Protocol/schema verification | Pinned FreeMask-to-wrapper schema/behavior mapping, mock/scientific distinction, synthetic contract fixtures | Can PASS locally without real images; verifies section 4 interface, split seed/semantics, identity, context, geometry and lineage rules |
| P0.1b Real shared manifest materialization | Real ACDC and M&Ms artifacts reused exactly or one shared newly materialized manifest per dataset; shared hashes and source/grid receipts | Requires real roots and P0.1a; PASS/DEFERRED/FAIL; absence of local data is DEFERRED and does not block local P0.2-P0.7 |
| P0.2 Environment candidate | Dedicated local candidate environment and provisional version inventory | P0.1a PASS; local PASS/FAIL independent of real datasets; scientific environment qualified separately |
| P0.3 Loader and primary config | Central-slice/optional stack loader, normalization, geometry/context/schema checks, strict MinL3 config | Local: P0.1a, synthetic arrays/mock manifests/common-grid fixture. Scientific: P0.1b real images, real grid/geometry and real loader validation |
| P0.4 Seed and runner | Fresh model/optimizer/BN per fixture sample, sample-specific seed, official MinL3 loop/update-before-stop/train-mode final pass, anonymous output | Local P0.2/P0.3; synthetic fixtures suffice; no scribble/reference path or scientific claims |
| P0.5 Reference parity | Independent pinned-DFC fixture harness: architecture, initialization, losses, one-step/state, stopping, final-forward and raw partition parity | Local P0.4; local candidate-environment PASS allowed; separate server/scientific environment parity mandatory before scientific execution |
| P0.6 Raw artifacts | Int32 export, canonical hashes/provenance, attempts, freeze/mutation rejection and resume identity checks | Local P0.4; fixture-based local PASS allowed; preserve fixture scope and reject scientific promotion |
| P0.7 Firewall and independence | Masks-unavailable fixture runs, scientific-mode rejection tests, scribble/reference exclusion, order/worker/retry independence and failed-sample accounting | Local P0.3-P0.6; fixture-based local PASS allowed; no shared learned state |
| P0.8 Real-data preflight | Fixed 50-100 real-slice receipt on actual target grid/GPU/environment, runtime/VRAM/concurrency and <=72h projection | P0.1b PASS, local P0.2-P0.7 PASS, server parity, real loader validation and real DFC smoke; synthetic timing cannot pass |
| P0.9 Scoped P0 acceptance | Separate local-completion and scientific-readiness reports with evidence hashes | Local completion: P0.1a and local P0.2-P0.7 PASS. Scientific readiness: additionally all real-data/server gates and section 15 runtime gate PASS |

Report scoped statuses using this template; attach receipt identities for any PASS and reasons for FAIL/DEFERRED. This template does not claim tests have been executed:

```text
P0.1a protocol/schema              PASS/FAIL locally
P0.1b real manifest                PASS/DEFERRED/FAIL

P0.2 local implementation          PASS/FAIL
P0.3 local implementation          PASS/FAIL
P0.4 local implementation          PASS/FAIL
P0.5 local parity                  PASS/FAIL
P0.6 local artifacts/freeze        PASS/FAIL
P0.7 local firewall/independence   PASS/FAIL

P0.8 real-data preflight           DEFERRED/PASS/FAIL

Scientific P0 ready                YES/NO
```

Keep server parity, real loader validation, real DFC smoke, and the budget receipt separately visible alongside this summary. `Scientific P0 ready=YES` requires all server/scientific gates to pass; local completion alone reports NO. An unexecuted local test must be explicitly reported as not yet executed rather than assigned a fabricated PASS. Local completion is allowed with P0.1b and P0.8 DEFERRED; lack of real data is not local implementation failure.

Optional 2.5D loading support may be implemented and fixture-tested in P0 if it shares the same small loader. Its scientific sensitivity run remains P2 and cannot delay or replace the 2D primary gate. No full scientific benchmark launches simply because raw preflight passed; P1 adapter/freeze/evaluation and end-to-end runtime requirements must also pass.

## 20. P1 implementation tasks

**P1 — primary semantic cardiac benchmark.** Scientific progression requires Scientific P0 ready, not merely local P0 implementation completion. P1's adapter, evaluator and full end-to-end launch gates remain additional requirements; a local PASS never authorizes a scientific benchmark.

| Task | Deliverable | Acceptance dependency |
|---|---|---|
| P1.1 Shared adapter specification | Frozen non-learned, GT-free, method-independent `cardiac_adapter_v1` input/output and rule contract | Resolve adapter ownership/inputs/encoding/VOID/splitting; no GT tuning |
| P1.2 Shared adapter implementation | One shared implementation plus provenance, deterministic tie rules and semantic/validity outputs | Permutation-invariance and determinism tests pass |
| P1.3 Semantic assembly/freeze | Ordered per-frame volumes, validity/failure accounting, immutable raw-to-semantic lineage | Geometry round-trip/completeness and mutation tests pass |
| P1.4 Isolated evaluator | Separate GT process, read-only predictions, patient-level metrics and coverage reports | Evaluator cannot modify artifacts; freeze validation required |
| P1.5 Full launch readiness | Updated end-to-end <=72h receipt, frozen evaluation/adapter/config specifications | All Definition-of-Done gates and P1 tests pass before production generation |
| P1.6 Primary execution/report | ACDC and M&Ms results for `DFC-Direct-2D-Default-MinL3` only | Exact manifest coverage; frozen semantic predictions before GT access |

Specify per-class Dice for RV/MYO/LV with patient-level aggregation as a primary reporting target; patient is the sampling unit, not pooled slices. Freeze the exact frame-to-patient aggregation, empty-class convention, uncertainty/CI policy, evaluation grid, VOID treatment, and failed-patient handling before opening scientific GT. These metric details are UNRESOLVED and must agree across compared methods.

Report total expected/succeeded/failed samples and patients, semantic coverage and VOID fractions, plus explicit scoring-subset reasons. Coverage-restricted Dice must never be the sole primary score or make abstention look like accuracy. No GT-based cluster matching, best-permutation oracle, profile selection, seed selection, or adapter threshold optimization is allowed.

Use synthetic evaluator fixtures to validate metrics before scientific GT access. Report runtime as test-time image optimization plus shell overhead, clearly separated from FreeMask's training/inference costs.

## 21. P2 secondary tasks

P2 is optional and cannot substitute for failed P0/P1 gates or replace the frozen primary result.

| Work | Required separation |
|---|---|
| 2D MinL4 | Same manifest, seeds, core and adapter; only threshold differs; cardinality sensitivity |
| 2.5D MinL3 | Same central inventory; clipped neighbors and common stack normalization; context sensitivity |
| 2.5D MinL4 | Optional combined sensitivity, distinct profile ID |
| Reference-image mode | Separate reference-transfer protocol, reference-set split/provenance, defaults and BN semantics; no primary-state reuse |
| Multi-seed robustness | Preregister additional benchmark seeds and aggregate all results; never choose the best seed by Dice |
| CommonStudent pseudo-label utility | Track B only; student training labels restricted to shared training manifest, with separate downstream training protocol |
| Performance variants | Explicit identities and parity evidence; no silent precision, algorithm, batch/BN, or stopping changes |

All optional runs retain complete accounting and their own launch-budget receipts. If a reference variant is pursued, account for the audited differences: continuity weight 5, reference batch training, no `minLabels` stopping, and train-mode target BN without target optimizer updates. Reference selection must be image-only and cannot include held-out test patients as undeclared training references.

## 22. Tests and acceptance gates

No test in this table has passed merely because it appears in this document. Fixture inputs, expected reference hashes, runtime environment, and comparison policy must be frozen before acceptance execution. Start with exact equality on deterministic same-runtime reference/wrapper paths; if float tolerances are necessary, document numerical reasons and preregister absolute/relative bounds before testing, never widen them to accommodate a failure. Integer labels, counters, ID sets, hashes, and stop reasons require exact equality. Numerical tolerance values remain UNRESOLVED pending the candidate-runtime reference harness.

Scope every receipt as local/fixture or server/scientific. P0.1a and data-agnostic P0.2-P0.7 gates may PASS locally using synthetic fixtures, fake patient trees, schema-valid mock manifests and deterministic MRI-like arrays. This includes architecture/initialization/loss/one-step/stopping/final-forward/raw-partition parity in the local candidate runtime, plus artifacts, freeze/resume, firewall, exclusion and independence tests. It does not imply real-data or server PASS. Real source/grid verification, server-environment parity, real loader/smoke validation and P0.8 remain separate mandatory scientific gates.

| Gate | PASS criterion | FAIL criterion | Phase |
|---|---|---|---|
| P0.1a protocol/schema | Pinned FreeMask schema/behavior mapping covers split seed 42, patient splits, sample/frame/slice/context identity, geometry/grid interface, lineage/firewall and mock/scientific distinction; no real images needed | Missing/inconsistent contract, wrong reference SHA/policy or inability to distinguish fixtures | P0.1a local |
| P0.1b real shared manifests | Real ACDC and M&Ms roots and source hashes resolve; exact shared FreeMask/DFC artifact hashes and real cohort/grid/geometry/lineage validate | Invalid real lineage/hash/split/grid or divergent consumer artifact; absent required roots yields DEFERRED, not local FAIL | P0.1b server/scientific |
| Scientific runtime guard | `scientific_run=true` rejects mock/fixture/missing-designation manifests, absent roots, bad source/manifest hashes or unverified shared identity; never falls back | Any fixture promoted to science, silent fallback, or unchecked prerequisite | Local negative tests in P0.7; actual validation before scientific execution |
| Architecture parity | Same module order, conv settings/biases, BN placement/defaults and counts: 101,800 for 1ch; 103,600 for 3ch | Any unapproved layer/setting/count difference | P0 |
| Initialization parity | Same deterministic CPU seed/runtime yields equal initial parameters and BN buffers against pinned reference | RNG draw/order, device initialization, or state mismatch | P0 |
| One-step optimization parity | Same input/state yields matching losses, all parameter gradients, post-step weights, momentum and BN state within preregistered tolerance | Any unexplained mismatch or missing state comparison | P0 |
| Loss parity | CE, row mean-L1, column mean-L1 and weighted total match reference and non-square analytic fixtures | Wrong axis, summed instead of mean reduction, different representation or scaling | P0 |
| Stop parity | Below-threshold start, exact equality, 5->3 transition and iteration-cap fixtures preserve update/check order and update counts | Pre-update exit, exact-K enforcement, wrong stop reason or off-by-one updates | P0 |
| Final-forward parity | Fresh post-update train-mode pass gives exact reference headless map; BN buffers and N+1 forward count match | Stale map, eval-mode BN, omitted final pass, incorrect buffer state | P0 |
| Optional no-grad parity | Logits/map and final BN state identical to ordinary gradient-mode final pass | Any output/state difference; then retain reference gradient mode | P0, if enabled |
| Seed contract | Fixed byte-level test vectors match; seed unaffected by row order/worker/retry; distinct audited IDs handled as specified | Python hash/index/order dependence, unrecorded generator or collision left unresolved | P0 |
| Determinism | Same sample/config/environment gives same update count and exact final raw partition hash on repeat | Unexplained differences | P0 |
| Per-image independence | Sample alone, after other samples, reversed order, and after a failed sample yields the same result; same-runtime worker assignment invariant | Reused weights/momentum/BN or global RNG interference | P0 |
| Loader/normalization | Central-only 2D statistics, fixed stack rule, boundary neighbors, constant/non-square fixtures and transforms match spec | GT/ROI statistics, hidden neighbor use in 2D, wrong context/frame/grid | P0 |
| Scientific loader and real smoke | P0.1b artifacts validate on the server; actual images/grid/geometry load correctly and a fresh per-image DFC smoke completes with recorded real provenance | Fixture substituted for real validation, unresolved grid or failed real loading/runner | Server/scientific P0.3/P0.4, before P0.8 |
| Server/scientific parity | Required pinned-reference parity suite passes in the actual target environment/device with separate receipt | Local-only receipt reused as server qualification, or any parity failure | Server/scientific P0.5, before real execution |
| GT firewall | Generation succeeds with masks physically unavailable; forbidden inputs/reads rejected; evaluator-side GT changes cannot affect generation | Mask access, GT argument, score feedback, or hidden annotation dependency | P0/P1 |
| Scribble exclusion | Primary API/CLI cannot activate/load scribbles | Reachable legacy scribble branch or accepted scribble flag | P0 |
| Reference-mode exclusion | Fresh model/optimizer/BN per sample; no checkpoint/ref-directory initialization path | Any transferred learned state or reference training in primary | P0 |
| Raw export reproducibility | Valid lossless integer map, exact shape/IDs/hash; deterministic scientific payload repeats | Color-only export, relabeling without contract, nondeterministic payload | P0 |
| Raw artifact freeze | One-byte mutation in any bound raw file/config/receipt rejected; expected-inventory gaps detected | Trusting filenames/existence, missing hash checks or incomplete coverage accepted as success | P0 |
| Failure/resume accounting | Every expected ID has explicit terminal state; all attempts retained; resume verifies identity/hashes | Missing/dropped/replaced samples, silent overwrite, changed-seed retry | P0/P1 |
| Runtime logging | Synchronized end-to-end/stage timing, updates, stop reasons, device and peak allocated/reserved VRAM recorded | Asynchronous timing, omitted failures or missing memory metrics | P0 |
| Runtime launch gate | P0.1b PASS and 50-100 fixed real slices under full MinL3 on actual target grid/GPU/environment; measured runtime/VRAM/safe concurrency; 25%-minimum-margin complete projection <=72h | Synthetic timing offered as evidence, projection >72h, omitted scope/overhead or invalid measurements; missing real prerequisites yields DEFERRED, never PASS | P0.8 server/scientific projection; P1 full-launch gate |
| Cluster-ID invariance | Arbitrary bijective relabelings leave adapter semantic and validity maps exactly unchanged, including ties | Numeric ID used as anatomical evidence or tie breaker | P1 |
| Adapter determinism/firewall | Same permitted inputs give same outputs; no GT, learning, method-specific branch or FreeMask audit dependency | GT tuning, hidden method identity, unstable rules or borrowed selection machinery | P1 |
| Assembly/geometry | Manifest ordering and inverse transforms pass round-trip fixtures; missing slices explicit | Axis/frame swaps, invented geometry, silent missing-slice background | P1 |
| Semantic freeze/evaluator isolation | Frozen config/raw/semantic hashes verified before GT; evaluator read-only; mutation rejected | Predictions change after GT, incomplete lineage or writable prediction access | P1 |
| Metrics/coverage | Frozen patient-level aggregation and VOID/failure/empty-class policy; complete cohort denominators | Slice-pooled primary, hidden exclusions, oracle mapping or report-only favorable subset | P1 |

Use both controlled label-count traces for stop-order unit tests and actual deterministic image fixtures for end-to-end partition parity. A synthetic trace alone cannot certify optimizer behavior. A good Dice result cannot override any failed parity, firewall, accounting, or runtime gate.

## 23. Expected experiment matrix

Primary results table, fixed before evaluation:

| Dataset | Profile | Seed policy | Role |
|---|---|---|---|
| ACDC | `DFC-Direct-2D-Default-MinL3` | `42 -> sample`, `dfc-sample-seed-v1` | Primary |
| M&Ms | `DFC-Direct-2D-Default-MinL3` | `42 -> sample`, `dfc-sample-seed-v1` | Primary |

Secondary table, reported separately:

| Dataset | Profile | Seed policy | Role |
|---|---|---|---|
| ACDC / M&Ms | `DFC-Direct-2D-MinL4` | Same per-sample derivation from 42 | Task-cardinality sensitivity |
| ACDC / M&Ms | `DFC-Direct-2.5D-MinL3` | Same per-sample derivation from 42 | Input-context sensitivity |
| ACDC / M&Ms | `DFC-Direct-2.5D-MinL4` | Same per-sample derivation from 42 | Optional combined sensitivity |

Every row binds manifest, profile/config, environment, source, adapter, raw/semantic freeze, and evaluator versions. No averaging, best-of-profile result, or secondary substitution enters the primary table. Sample counts and metrics are UNRESOLVED until authoritative manifest and frozen predictions exist.

## 24. Risks and unresolved decisions

| Severity | Risk | Required mitigation / blocking gate |
|---|---|---|
| CRITICAL | GT-derived preprocessing or annotation-selected cohort | Image-only lineage, allowlisted manifest, masks-unavailable execution |
| CRITICAL | Scribble accidentally reachable | No primary flag/API/call path; rejection test |
| CRITICAL | GT-based `minLabels` selection | Immutable primary MinL3 and separate preregistered sensitivity |
| CRITICAL | GT-based seed selection | Frozen hash policy; all preregistered seeds reported |
| CRITICAL | GT-based cluster mapping | Deterministic GT-free shared adapter; no oracle permutation |
| CRITICAL | Mock/fixture results presented as scientific cohorts or runtime evidence | Hash-bound execution scope and source lineage; scientific-mode guard; P0.8 accepts real-data timing only |
| HIGH | State shared between images | Fresh CPU model, optimizer/momentum, BN; order/failure/worker tests |
| HIGH | Wrong final-forward semantics | Post-update fresh map and exact reference parity |
| HIGH | Eval-mode BN | Train-mode assertion and BN buffer/forward-count tests |
| HIGH | Different cohort from FreeMask | Exact authoritative manifest digest and inventory identity |
| HIGH | Silent failed-image dropping | Complete attempt/terminal ledger, freeze completeness and denominator checks |
| HIGH | Cluster IDs treated as semantics | Semantic/validity permutation invariance |
| HIGH | Undisclosed transductive compute | Per-sample timing and explicit test-time optimization reporting |
| MEDIUM | Environment nondeterminism | Validated backend policy, locks, deterministic fixtures per used device |
| MEDIUM | Unstable seed derivation | Canonical bytes, golden vectors; no Python hash or row indices |
| MEDIUM | Different preprocessing context | Explicit central-only vs common-stack statistics and context disclosure |
| MEDIUM | Geometry mistakes | Common grid contract, axis/affine validation and round-trip tests |
| MEDIUM | Optimistic runtime estimate | Real pilot, measured concurrency, 25% minimum margin and blocked launch |
| MEDIUM | Missing local datasets incorrectly blocks data-agnostic implementation | P0.1a local PASS unlocks P0.2-P0.7; P0.1b/P0.8 can stay DEFERRED without local failure |
| HIGH | Local parity mistaken for scientific-server qualification | Separate scoped receipts and required server parity/real loader/smoke gates |

Open items and the evidence needed to resolve them:

| ID | Status | Item | Required resolution before |
|---|---|---|---|
| U1 | UNRESOLVED; P0.1b DEFERRED when roots unavailable | Existing real scientific manifests, roots, image-only lineage and inventory counts for ACDC and M&Ms | Scientific P0.1b acceptance only; P0.1a and local P0.2-P0.7 remain possible without real images |
| U2 | UNRESOLVED; real verification DEFERRED when roots unavailable | Pinned schema/grid/decoder interface plus real FreeMask consumer binding and common grid/geometry realization | Verify protocol interface in local P0.1a and test synthetic common grid in local P0.3; validate real artifact/grid/geometry at P0.1b and scientific P0.3 |
| U3 | UNRESOLVED | Local candidate build qualification and separate actual-server PyTorch/CUDA/driver/GPU compatibility | Local environment/parity can PASS independently; server evidence required for scientific environment freeze |
| U4 | UNRESOLVED | Numerical parity tolerances, golden fixture hashes and separately scoped local/server reproducibility evidence | Local fixture parity may PASS first; server parity remains mandatory; no post-failure tolerance relaxation |
| U5 | UNRESOLVED | `cardiac_adapter_v1` owner/location, permitted image inputs, anatomical/tie/split rules, semantic/VOID encoding | P1 implementation/spec freeze; shared GT-free contract |
| U6 | UNRESOLVED | Metric aggregation, empty-class/VOID/failure policy, scoring eligibility, uncertainty and geometry-dependent metrics | Evaluator specification before scientific GT access |
| U7 | UNRESOLVED; P0.8 DEFERRED without real/server prerequisites | Hardware allocation, real pilot time/VRAM/concurrency, retry cap and end-to-end overhead | Scientific readiness/production launch; real <=72h receipt with 25% minimum margin; synthetic timings cannot close this item |
| U8 | UNRESOLVED | Reference-transfer cohort and optional seed/performance/Track B protocols | Only the corresponding P2 work; never blocks defining the primary |

Frozen primary design, hash derivation, normalization recipe, core/shell boundary, and original update/stop/final-forward semantics are not open for GT-based reconsideration. Missing evidence remains UNRESOLVED; mark unavailable real-data/server gate execution DEFERRED with a reason. This document update claims no new local or server PASS, runtime result, adapter completion, or scientific readiness. U1/U7 and real-data portions of U2 do not block local P0.2-P0.7 once P0.1a passes.

## 25. Definition of Done

**Local P0 implementation complete** may be reached with P0.1a PASS and P0.2-P0.7 local PASS while P0.1b and P0.8 remain DEFERRED. This means the data-agnostic code path is implemented and fixture-validated: loaders/schema/geometry, seeds, independent model/optimizer/BN state, official loop and final-forward behavior, pinned-reference parity, raw artifacts/freeze/resume, GT firewall and accounting all have scoped local receipts. It does not mean real ACDC/M&Ms cohorts exist, real grid validation passed, or Scientific P0 is ready. No scientific cohort, metric or runtime claim may be made from fixtures.

**Scientific P0 ready** requires local completion plus P0.1b PASS for real manifests and accessible image-only roots for both ACDC and M&Ms, verified shared FreeMask/DFC identity, real common grid/data contract, server-environment parity, real loader validation, real DFC smoke, the real runtime/VRAM/concurrency pilot, and the section 15 <=72h gate with the unchanged 25% minimum margin. The scientific runtime guard must reject mocks and unresolved source/manifest identity. Report `Scientific P0 ready=NO` while any server/scientific prerequisite is DEFERRED, FAIL, or unverified. Local PASS cannot substitute for these gates.

**Primary scientific benchmark launch readiness** additionally requires the existing P1 adapter/semantic/evaluator gates and final end-to-end budget refresh. Preserve the following complete scientific checklist; none of these real requirements is waived by local completion:

The primary benchmark is ready for scientific execution only after all of the following are evidenced:

1. DFC and FreeMask use the exact same frozen patient/sample manifest and shared data contract, with `split_seed=42`.
2. Primary profile is `DFC-Direct-2D-Default-MinL3` on both ACDC and M&Ms, with all specified optimization defaults.
3. `benchmark_seed=42` uses the stable versioned per-sample SHA-256 derivation and validated serialization/test vectors.
4. Every image starts with fresh model weights, optimizer/momentum, and BN state; independence tests pass.
5. Primary generation has no GT, scribble, reference-training, or pretrained-state path.
6. Architecture, initialization, losses, and one-step optimization match the pinned reference under the qualified runtime.
7. Stopping matches official pre-update counting and post-update threshold checking, including all edge-case fixtures.
8. Headless output is a fresh post-update train-mode forward; exact final partition and BN-state parity pass.
9. Lossless anonymous raw partitions, scientific payload hashes, provenance, complete accounting, and mutation-rejecting freeze are reproducible.
10. The GT firewall passes with masks physically unavailable, and the later evaluator cannot feed information back into generation.
11. A fixed 50-100 real-slice preflight under the full primary configuration has passed and reported the required timing/memory/failure statistics.
12. The full declared run, including both datasets, frozen safe concurrency, shell overhead, retries and operating margin, projects to <=72 hours. Otherwise launch remains blocked.
13. The environment and backend settings are frozen only after parity/determinism tests, with qualification for each used device.
14. DFC core ends at anonymous `[H,W]` integer partitions; semantics remain outside the method.
15. Before the P1 scientific launch, the shared adapter spec/implementation is frozen and deterministic, GT-free, method-independent, and cluster-ID invariant.
16. Semantic stacking, VOID/failure accounting, patient-level evaluation policy and immutable prediction/evaluator boundary have passed their P1 tests.
17. All production-relevant UNRESOLVED items U1-U7 are closed by evidence; P2 has not replaced or weakened a primary gate.

Scientific execution completion is a later state: every expected sample has a traceable terminal outcome, frozen raw and semantic artifacts are verified, patient-level results and coverage/failures are reported for the fixed primary, and actual wall time is disclosed. An implementation plan, successful smoke test, or favorable segmentation metric alone is not that completion state.
