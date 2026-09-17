# CUTS Cardiac Baseline Implementation Plan

## 1. Scope and scientific objective

Status: implementation plan only, prepared 2026-09-16. No P0 implementation or training is authorized by this document's creation.

The primary question is whether CUTS's image-only reconstruction/contrastive representation and per-image anonymous partitions support cardiac segmentation through a shared, non-learned benchmark interface. This is not a redesign of CUTS into FreeMask. Native partition quality, adapter failures, and optional pseudo-label utility must remain distinguishable.

Source baseline inspected for this plan:

| Repository | Source identity | Evidence and limitation |
| --- | --- | --- |
| Current CUTS checkout | `27751160b8dc3304f57d3725e7e1028a576f6849`, branch `main`, initially clean | Local `origin` is `https://github.com/Azios1010/CUTS.git`; do not claim that the fork is byte-identical to the latest KrishnaswamyLab upstream. |
| FreeMask reference | `QuocKhanhLuong/Self-Audit`, `96c32b10fc7b8e09b48822e10ae9eb6cc149e253`, initially clean | Inspected local snapshot at `C:/Users/ADMIN/AppData/Local/Temp/self-audit-readonly`. Relevant implementation is **only** `src/self_audit_maskfree/`, not the older supervised pipeline. |

CUTS links below are relative to this repository and refer to the pinned checkout. FreeMask paths are relative to the reference repository at the pinned SHA. Code/config verification is not verification of an executed scientific run.

Evidence anchors:

- [src/model/CUTS_model.py](src/model/CUTS_model.py), `CUTSEncoder.__init__/forward` and `PatchRecon` (lines 10-195): four same-resolution convolution blocks, sampled patch reconstruction, inference-only latent return.
- [src/main.py](src/main.py), `train` (18-114), `test` (117-177), CLI (197-200): image-only training losses, minimum dev-loss checkpoint, legacy label-carrying export and automatic post-training test.
- [src/data_utils/prepare_dataset.py](src/data_utils/prepare_dataset.py), `prepare_dataset`; [src/data_utils/split.py](src/data_utils/split.py), `split_dataset`: item-level random splitting and full-dataset legacy test route, unsuitable for this patient-held-out benchmark.
- [src/scripts_analysis/helper_generate_kmeans.py](src/scripts_analysis/helper_generate_kmeans.py), `phate_clustering` (46-60): primary numerical clustering kernel; `generate_kmeans` (17-43): separate GT-assisted conversion that must not be called.
- FreeMask [`data/discovery.py`](https://github.com/QuocKhanhLuong/Self-Audit/blob/96c32b10fc7b8e09b48822e10ae9eb6cc149e253/src/self_audit_maskfree/data/discovery.py), `assign_splits` (252-335), `discover_dataset` (1122 onward): patient split policy, not evidence of final patient membership.
- FreeMask [`data/geometry.py`](https://github.com/QuocKhanhLuong/Self-Audit/blob/96c32b10fc7b8e09b48822e10ae9eb6cc149e253/src/self_audit_maskfree/data/geometry.py), `read_slice_stack` (228-264): `[z-1,z,z+1]`, clipped slice indices at boundaries.
- FreeMask `configs/maskfree_acdc_150.yaml` and `configs/maskfree_mnms_150.yaml`: templates specify 224 pixels, seed 42, depth axis 2, independent dataset runs, and explicitly unverified image-root placeholders. They are not production resolved configs.

## 2. Frozen design decisions

1. **Shared patient split:** `FreeMask train/dev/test patient IDs = CUTS train/dev/test patient IDs`, with the exact same frozen patient/sample manifest for each dataset. Accept either (A) an existing verified scientific FreeMask manifest, reused exactly, or (B) if none exists, exactly one new shared benchmark manifest produced using pinned FreeMask discovery/splitting with `split_seed=42`, frozen before either method's scientific benchmark. Case A must also verify the frozen split-seed contract. Manifest verification/establishment remains **UNRESOLVED** until P0.1 passes; a historical production run is not required. Independently generated CUTS/FreeMask cohorts fail even with identical nominal ratios or seeds. Never create a CUTS-specific 70/15/15 or legacy 7:3 slice split.
2. **Separate input profiles:** CUTS-2D is method-faithful, central slice `[1,H,W]`; CUTS-2.5D is input-matched, neighboring slices `[3,H,W]`. Report separately; the latter is not original 2D CUTS.
3. **Cluster-ID invariance:** adapter decisions cannot depend on raw cluster values or their order. Permuting cluster IDs must leave the semantic map and validity exactly unchanged. Geometry-based canonical ordering is mandatory.
4. **Precise leakage terminology:** CUTS Stage-1 objective is image-only reconstruction plus contrastive learning. The original pipeline is GT-accessible in data/export plumbing and has GT-assisted post-processing/oracle evaluation. Do not describe Stage-1 objective as supervised or GT-consuming.
5. **Sampler compatibility:** preserve official `PatchSampler` behavior, including per-call reseeding, CPU SSIM, sampling/fallback behavior and call order. Persistent RNG, cached CPU images, and equivalent vectorization belong to a named secondary hardened variant, never a silent primary fix.
6. ACDC and M&Ms train independently from random initialization. No cross-dataset checkpoints, initialization, resume, normalization fitting, or pooled latent clustering.
7. Train patients alone receive gradient updates; dev patients only supply forward image-only validation loss; test patients never participate in Stage 1 or checkpoint selection.
8. Primary core output is per-image PHATE + K-means **K=10**, named `CUTS-K10`. K is a partition granularity, not the number of anatomical classes.
9. The deterministic cardiac adapter, evaluation, and optional CommonStudent are outside CUTS core. Diffusion and CommonStudent are excluded from P0.
10. Freeze checkpoints, configs, raw partitions and semantic predictions, and record hashes before reference-mask evaluation. No GT-guided method, K, adapter, seed, or checkpoint selection.
11. **Primary seed policy:** `GLOBAL / BENCHMARK SEED = 42` for model initialization, training RNG, deterministic loader/worker seed derivation, benchmark run identity and the FreeMask comparison. Freeze `split_seed=42` once; model/training/clustering reruns never regenerate membership. PHATE/K-means keeps official `clustering_seed=1`, with `clustering_retry_seed=2` exactly once on exception, never a seed search.
12. **Minimal adapter:** `cardiac_adapter_v1` must be the minimum sufficient deterministic semantic resolver, not a competing segmentation method. High VOID/unresolved rates are acceptable. At first primary reference-GT access, and certainly once any primary GT metric has been observed, v1 is permanently frozen. Every subsequent semantic-rule change requires `cardiac_adapter_v2`, new preregistered predictions, a new prediction freeze and a new evaluation run.

### Primary RNG contract

This table is authoritative if other prose is ambiguous. RNG roles must remain separate in configs and provenance.

| Role / field | Frozen primary value or policy |
| --- | --- |
| Shared manifest `split_seed` | `42`; freeze membership once for both methods |
| `benchmark_seed` / run identity | `42`; GLOBAL / BENCHMARK SEED |
| `model_init_seed` and training RNG seed | `42` |
| DataLoader / worker `loader_seed` policy | Deterministic derivation based on `42`; record the exact derivation |
| CUTS PatchSampler seed | `42` through the primary training config; preserve existing constructor/call and per-call reseeding behavior |
| FreeMask comparison seed | `42` |
| PHATE/K-means `clustering_seed` | `1`, preserving official behavior, not performance-selected |
| `clustering_retry_seed` | `2`, exactly once and only after the seed-1 attempt raises an exception; if it also fails, record failure |
| Additional model seeds | Optional P2 robustness only, preregistered before GT evaluation; never required for P0/P1 |

## 3. CUTS core vs benchmark-shell boundary

Primary architecture:

```text
                   TRAIN PATIENTS
                         |
                  CUTS Stage 1 <--- dev forward image-only loss
                         |          (checkpoint selection only)
                 frozen checkpoint
                         |
                     test image
                         |
                  dense latent map
                         |
                PHATE + K-means K10
                         |
                anonymous partition       <-- CUTS core ends
                         |
             shared cardiac_adapter_v1     <-- benchmark interface
                         |
               BG / RV / MYO / LV / VOID
                         |
                  prediction freeze
                         |
                isolated GT evaluator      <-- evaluation environment
```

CUTS core is `cardiac MRI -> CUTS Stage-1 encoder -> dense latent -> PHATE/K10 -> anonymous partition`. Cardiac loading is dataset plumbing; semantic interpretation is not a learned CUTS head and must not be presented as a CUTS contribution.

Four separately identified layers:

| Layer | Responsibility | Forbidden additions |
| --- | --- | --- |
| CUTS Core | Existing encoder, PatchRecon, objectives, sampler and primary clustering | FreeMask producer, auditor, predictive challenge, pooling, semantic network |
| Shared Benchmark Adapter | Geometry-aware deterministic partition-to-semantics interface | GT, Dice/IoU selection, learned confidence, method-specific tuning |
| Evaluation | Read frozen predictions and separately mounted reference masks | Rewriting predictions, feeding scores back into generation |
| Optional Downstream Experiments | Same CommonStudent for frozen train pseudo-labels from each method | Calling student results native/direct CUTS results |

## 4. Shared patient/data protocol

### Verified policy versus unresolved cohort

FreeMask `assign_splits` deterministically ranks remaining patients using dataset, split seed and patient ID; this benchmark fixes `split_seed=42`. Its current branches preserve declared M&Ms train/dev/test folders **when an official dev cohort exists**; otherwise explicit test membership is preserved and remaining patients are assigned by the fallback policy. With an official test cohort, remaining train/dev proportions are renormalized; without one, the fallback is 70/15/15. ACDC's `training` folder alone is not a fixed training assignment. Therefore a ratio alone cannot reproduce the cohort, and this policy is not permission to create an independent CUTS split.

`splits/acdc_patient_split_seed42.json` in the reference checkout describes a 0.8 train fraction over `preprocessed_data/ACDC`; no reference to that filename was found in `src/self_audit_maskfree/`. It is not accepted as FreeMask cohort evidence. Inspected `reports/maskfree150/local_baseline/resolved_config.json` and `performance_cpu/after_ordinary/resolved_config.json` point to synthetic/corrected fixtures at 128 pixels. They do not verify production ACDC/M&Ms cohorts.

**P0.1 accepts either route, separately for each dataset:**

- **Case A — verified existing production manifest:** verify actual patient membership, source SHA, configuration, frame/sample inventory and split provenance, including `split_seed=42`. CUTS must reuse that exact manifest; do not regenerate it from its ratio or seed. An existing artifact with a different or unverifiable split seed is not compliant evidence and requires explicit protocol resolution, not silently rewriting its provenance or membership.
- **Case B — no valid production manifest exists:** create exactly one new shared benchmark manifest using the pinned FreeMask discovery/splitting policy on inventoried images with `split_seed=42`. Freeze it before either method's scientific benchmark. Both FreeMask and CUTS must consume that exact manifest. This is shared benchmark preparation, not an independent CUTS split, and does not require a historical FreeMask production run.

**UNRESOLVED:** verify Case A or establish Case B during P0.1 before downstream data-path implementation/scientific runs. Templates, ratios and supervised Self-Audit split artifacts alone do not pass. **FAIL:** CUTS and FreeMask independently generate cohorts, even if both use 70/15/15 and seed 42.

### P0.1 local-development interpretation

P0.1 is deliberately split when clinical image roots are unavailable locally:

- **P0.1a — protocol/schema verification:** PASS locally when the shared image-only manifest schema/importer, pinned FreeMask discovery contract, split seed 42, patient-disjoint validation, mock-fixture rejection guard and `scientific_run=true` guard are tested. Fixtures and fake patient trees are implementation evidence only, never scientific cohorts.
- **P0.1b — real shared-manifest materialization:** DEFERRED until the training server exposes ACDC/M&Ms image roots. On that server, execute Case A or Case B exactly once per dataset and freeze the resulting real manifest before either scientific method run.

P0.1a permits P0.2–P0.8 data-agnostic local development only. It does not permit clinical training, scientific checkpointing, comparison results, or an overall scientific P0-complete claim. Every scientific executable must reject a mock manifest and require a frozen real manifest hash and existing image roots.

```text
shared patient/sample manifest
       +-- split_seed = 42
       +-- immutable train/dev/test membership consumed by BOTH methods

CUTS benchmark/model/training seed = 42
PHATE/K-means seed = 1 (exception-only retry = 2)
```

Changing model, training or clustering seeds in later preregistered experiments must never regenerate or alter the shared cohort, patient IDs, selected samples or split membership.

### Manifest contract

Each generation record must include dataset namespace, patient ID, split, acquisition/source ID, frame index/axis, slice index/depth axis, context indices, image-only relative path, image checksum, native shape, affine/orientation/spacing, transform/inverse-transform metadata, stable sample ID and source-cohort provenance. Include split algorithm/version and explicit `split_seed=42`, distinct from model and clustering RNG roles. Both methods consume the same frozen patient/sample manifest and its checksum. No reference path, mask field, foreground hint or GT-derived crop belongs in this schema.

Enforce patient-set equality and sample/frame-set parity with FreeMask, not merely equal patient counts. Resolve canonical patient identity across repeat scans, ED/ES files, cine files and sites; reject uncertain IDs, conflicting folders, cross-split duplicate acquisitions and unresolved aliases. Record image-only deduplication decisions. All time points and slices from a patient remain together.

Train on the **train patients only**, not all available images. Dev patients do not receive updates. The frozen encoder may process test inputs; PHATE/K10 may fit independently on each test image. These are declared per-image inference operations, not cohort-global representation training or leakage. No test image can influence another patient's fitted representation, checkpoint or hyperparameters.

### Frames, geometry and preprocessing

- Match FreeMask's actual frozen source records: all selected cine frames if present, or the same ED/ES-only limitation if only those sources were used. Do not select frames using masks or silently add cine data to one method. Evaluation-only reference availability may restrict scored phases after predictions are frozen; it cannot restrict training membership retrospectively.
- Do not read `Info.cfg`, annotation inventories or supervised-preprocessing metadata to decide generation cohort/crops. Any image-only frame policy must be shared and frozen. Record unavailable source geometry rather than inventing it.
- Match whole-image field of view, orientation handling and target grid across primary comparisons. The template grid is 224x224; actual production grid, frame inventory and native geometry are **UNRESOLVED** until the manifest/config gate. No GT-centered cropping.
- Implement one explicit deterministic image-only normalization contract. FreeMask `data/dataset.py::_fit_statistics` uses 0.5/99.5 percentiles and clipped mean/std over fitting pixels across context; `load_full_input` is a deployment path, not its predictive training observation path. Do not copy its role masks or predictive challenge into CUTS.
- Before loader implementation, freeze CUTS's exact normalization support and numeric range, with SSIM compatibility tested. CUTS-2D must derive statistics from the central slice only; using hidden neighbors would compromise that profile. CUTS-2.5D may use the supplied stack. Neither fits dataset-global statistics on dev/test. Deterministic per-input statistics at inference are permitted and explicitly distinct from cohort-level fitting.
- **UNRESOLVED:** which frozen native FreeMask prediction endpoint and observation support define Track A. Input-matched means matched context, cohort and grid; it does not claim identical objectives or FreeMask's internal evidence-withholding protocol.

## 5. CUTS-2D and CUTS-2.5D profiles

| Property | CUTS-2D: method-faithful | CUTS-2.5D: input-matched |
| --- | --- | --- |
| Input | Central slice `[1,H,W]` | Same-frame `[z-1,z,z+1]`, `[3,H,W]` |
| Edges | Not applicable | Clip z indices, replicating edge slices, as `read_slice_stack` does |
| Encoder | Existing `CUTSEncoder(in_channels=1)` | Existing `CUTSEncoder(in_channels=3)` |
| Patch reconstruction | Reconstruct central-channel image patches | Reconstruct all three input channels, retaining official objective |
| Output | One central-slice latent map/partition | One central-slice latent map/partition, not three segmentations |
| Adapter intensity input | Central MRI slice | Central MRI slice only |
| Reporting | Closest cardiac adaptation to original medical 2D regime | Explicit input-context adaptation |

For both, `k=16` gives latent `[B,128,H,W]`; patch tensors are `[B,8,C,5,5]` and anchor/positive tensors `[B,8,128]`. No 3D convolutions, pooling, interpolation inside the encoder, pretrained weights or cross-profile checkpoint reuse. Keep independently trained checkpoints and separately keyed artifacts for each dataset/profile/benchmark_seed. Use identical central sample membership in the two profiles.

## 6. Stage-1 training protocol

Implement a thin cardiac training entry point using the existing model, loss and scheduler modules. Reproduce `src/main.py::train` numerical behavior while replacing only dataset/identity plumbing, checkpoint metadata and unsafe automatic test dispatch.

Primary compatibility configuration takes the non-seed training recipe from [config/brain_tumor_seed1.yaml](config/brain_tumor_seed1.yaml), explicitly setting the cardiac training config's `random_seed=42` (`benchmark_seed=42`, `model_init_seed=42`): `num_kernels=16`, `sampled_patches_per_image=8`, `patch_size=5`, AdamW `lr=1e-3`, `weight_decay=1e-4`, 200 epochs, physical batch 16 if preflight permits, FP32. The historical filename does not prescribe the cardiac training seed. MSE and existing `NTXentLoss(temperature=0.1)` mix as `0.001*contrastive + 0.999*reconstruction`. Use existing warmup/cosine scheduler: 10 warmup epochs, initial LR `1e-3*learning_rate`, same scheduler-step order. Do not copy FreeMask's 150 epochs or scheduler simply to make configurations look alike.

- Preserve four 5x5 convolutions, BatchNorm/LeakyReLU, replicate padding, PatchRecon, sample counts, loss reductions and within-image contrastive negatives.
- Preserve official `PatchSampler.sample` reseeding and CPU SSIM. Pass seed 42 through the existing constructor/call path and preserve its per-call reseeding exactly; benchmark seed 42 is not permission to rewrite the sampler. Preserve threshold 0.5, maximum 20 attempts and fallback semantics. Persistent RNG or cleanup remains P2 only.
- Preserve train shuffling and official small-dataset extension semantics (`ExtendedDataset`, minimum five physical batches) if triggered; record both unique images and repeated exposures. Do not apply extension to dev/test.
- Train uses only train manifest records. Dev uses `eval()` and `no_grad()` with the same reconstruction/contrastive loss, weighted by sample count as in the original. BatchNorm statistics must not update on dev.
- Select the strictly lowest finite dev total loss, matching the original strict `<` rule; ties retain the earlier checkpoint. Record epoch, component losses, loader order/config and checkpoint hash. No semantic metric or labeled validation.
- Test loader is unavailable to this executable. No automatic `main.py::test`, no post-epoch test forward, no test normalization fitting. Resume must validate dataset/profile/manifest/config identity; preserve optimizer/scheduler/RNG state in a separate run-state artifact.
- Primary P0/P1 uses one benchmark/model/training seed, **42**, aligned with FreeMask. Derive deterministic DataLoader/worker seeds from 42 and record the derivation. The shared manifest's `split_seed=42` is frozen independently; reruns never regenerate it. No multiple-model-seed requirement exists for P0 completion or the primary P1 matrix. Additional model seeds are optional P2 robustness only, preregistered before GT evaluation; do not add seeds retrospectively to choose favorable runs after seeing seed-42 GT results. Never publish only the best test seed. PHATE/K-means uses its separate seed-1/exception-retry-2 contract, not 42.
- If physical batch 16 fails preflight, resolve and record a smaller batch before real runs. It changes BatchNorm/training trajectory and is a reported training-profile deviation, not exact equivalence. No silent mid-run OOM fallback or LR rescaling.

A tiny synthetic compatibility test must compare the wrapper's one-step losses, gradients, parameter updates, scheduler state and sampler outputs with the original training algebra. This is not full training.

## 7. Raw clustering protocol

Primary `CUTS-K10`, for each image independently:

1. Load the frozen dataset/profile checkpoint, `eval()`, inference-only mode, no gradients. Export latent `[128,H,W]` without label arrays or unnecessary dense reconstructed patches.
2. Flatten with the original pixel/channel order to `[H*W,128]`. No new whitening, feature normalization, pooling, subsampling or dataset-global fit.
3. Match `helper_generate_kmeans.py::phate_clustering`: `phate.PHATE(n_components=3, knn=100, n_landmark=500, t=2, verbose=False, random_state=actual_clustering_seed_used, n_jobs=workers)`; `fit_transform(latent)`; `phate.cluster.kmeans(operator, n_clusters=10, random_state=actual_clustering_seed_used)`. Both use `clustering_seed=1` initially, or `clustering_retry_seed=2` only for the single exception-triggered retry.
4. Reshape cluster IDs to `[H,W]`; export raw integers, image/sample identity, latent hash, actual clustering seed, PHATE/dependency versions, thread count and timing. Do not export a GT-derived `seg_kmeans`.

The official helper defaults to `clustering_seed=1`, separate from `benchmark_seed=42` and model/training seed 42. Preserve seed 1 because it is official CUTS behavior, not because of benchmark performance. Only if the seed-1 clustering attempt raises an exception, retry exactly once with `clustering_retry_seed=2`; if seed 2 also fails, record clustering failure. Do not try additional seeds or retry a successful result based on quality. Log attempted seeds, exception details, `actual_clustering_seed_used`, retry occurrence and outcome. The retry is a failure fallback, never an alternative selected run. Freeze `n_jobs` and BLAS thread counts; begin with one worker for reproducibility, recording this runtime setting.

The pure numerical function resides in a module with oracle-related imports. Prefer a small isolated kernel in the new clustering module reproducing only these calls, with differential tests against the original function in a development fixture. This deliberate minimal duplication avoids importing the original `generate_kmeans`/CLI and its label-based shape, Dice and conversion path. Do not duplicate the encoder or losses.

K=10 is the verified official setting, not ten cardiac classes. Many clusters may map to one anatomy; one cluster may be unresolved. K=4 is a separately named, preregistered P2 sensitivity analysis, not a GT-selected substitute. Reproducibility is exact within the pinned hardware/software/thread environment; do not promise cross-version bitwise equivalence. Store both original raw-array hashes and deterministic artifact hashes. Do not relabel clusters just to conceal instability in the raw-partition gate.

## 8. Diffusion policy

| Status | Policy |
| --- | --- |
| PRIMARY | PHATE + K-means K=10 only |
| SECONDARY, later P2 | One preregistered deterministic no-GT diffusion hierarchy/selection policy |
| ORACLE DIAGNOSTIC ONLY | GT-best diffusion level, with explicit access/selection ledger |

The current analysis path is not a validated image-only cardiac final-level selector. Existing hierarchy/persistence heuristics must not be equated with such a rule. GT-assisted selections in `src/scripts_analysis/run_metrics.py` cannot enter generation. P0/P1 must not depend on diffusion, its packages, or a search over levels. A later no-GT policy needs its own specification, failure behavior, immutable config and tests before test metrics are opened. Never select K or diffusion level using test Dice.

## 9. Deterministic cardiac semantic adapter

`cardiac_adapter_v1` is a **shared benchmark interface**, not part of CUTS. Implement in P1 only. It must later accept DFC/PiCIE/STEGO partition exports without method-specific branches or tuning. FreeMask retains its native anatomy-resolution method; do not copy its predictive evidence/auditor here.

**Frozen complexity rule:** `cardiac_adapter_v1` must be the minimum sufficient deterministic semantic resolver. Its purpose is not to become a competing segmentation method. It may use only connected components, border/exterior relation, adjacency, enclosure/topology, simple geometry, central-image intensity statistics, valid header orientation/spacing and deterministic many-to-one merging. Do not accumulate increasingly complex heuristics in response to GT performance. High VOID/unresolved rates are acceptable scientific results.

Contract:

```text
Input:  raw integer partition [H,W], central image [H,W], optional image geometry
Output: uint8 semantics [H,W]: BG=0, RV=1, MYO=2, LV=3, VOID=255
        bool validity [H,W] = (semantics != 255)
        trace: component evidence, decisions, rejections, class-resolution status,
               adapter version/config hash, input hashes, geometry validity
```

No GT, Dice/IoU, Hungarian assignment, learned classifier, learned confidence, predictive challenge or optimization against benchmark scores. Failure/abstention is a first-class output. No method name or raw cluster ID is available to scoring/tie-breaking functions.

### Minimal algorithm to implement

1. **Components and graph.** Split every cluster into 4-connected components; use the complementary 8-connected convention for hole/background topology. Compute area, centroid, bounding box, perimeter, border contact, adjacency length, holes/enclosure and central-image intensity summaries. Use physical spacing where available; otherwise use explicitly normalized pixel units. Sort canonical component records by geometry and, as a final complete tie-break, lexicographic sorted pixel coordinates. Store raw IDs only as trace metadata.
2. **Conservative background.** Identify components with supported border dominance using frozen border-contact and exterior-connectivity rules. Border contact alone is insufficient to call a structure background. A component spanning exterior and a candidate internal anatomy is ambiguous. Do not declare all non-cardiac candidates BG by elimination.
3. **Cavity/ring candidates.** On the remaining region graph, enumerate compact cavity components and directly adjacent unions of compatible ring components. A candidate LV/MYO pair requires observed enclosure: the ring separates the cavity from exterior on the specified digital grid, with image-only cavity/ring intensity and geometry checks. All ring pixels must come from observed components; no dilation, synthetic gap closure or contour completion. Limit candidate unions using a frozen graph-search bound and canonical order; on bound overflow abstain and record it.
4. **LV/MYO assignment.** Accept only a uniquely supported cavity/ring hypothesis passing all frozen tests and ambiguity margins. Canonical ordering makes enumeration deterministic; it does not justify choosing an anatomically ambiguous hypothesis. Assign observed cavity components LV and observed ring components MYO.
5. **RV assignment.** From remaining cavity-like compact components/compatible unions, require adjacency to the accepted MYO complex, outside-LV position and frozen intensity/geometry criteria. Validated image-header orientation may constrain position, never array-left/right assumptions. If orientation is needed but unknown, or multiple candidates remain plausible, RV is unresolved.
6. **Many-to-one merges.** Merge only existing, mutually compatible adjacent components consistent with the accepted anatomical hypothesis. Raw cluster equality is not evidence for merging disconnected anatomy. Do not split a connected mixed component using intensity thresholds or a new segmentation algorithm in v1. Disconnected pieces of one raw cluster may receive different semantics after component decomposition.
7. **Abstain and trace.** All unaccepted pixels remain VOID. Set validity from the final semantic map, not from a learned confidence. Report unresolved classes separately from background and from a claim that an anatomy is truly absent. Never reinterpret unresolved as proven anatomical absence.

Before P1 coding, freeze a small adapter specification/config containing exact border-dominance definition, size/compactness/intensity bounds, adjacency tests, topology convention, merge/search bound, ambiguity margin, orientation fallback and floating-point comparison rules. **UNRESOLVED:** numerical constants and their image-only justification. Choose once on synthetic fixtures and designated train/dev images without masks; do not tune separately for CUTS, datasets, or future methods. If geometry-unit conversions are needed, use one shared rule. Once frozen, any rule change creates v2 and cannot silently replace v1.

**Permanent post-GT freeze:** at first primary reference-GT access, and certainly once any primary GT metric has been observed, `cardiac_adapter_v1` is permanently immutable. Any subsequent semantic-rule change, however minor, requires **`cardiac_adapter_v2` + new preregistered predictions + new prediction freeze + new evaluation run**, with prior GT exposure disclosed and original v1 results retained. Preregistration after exposure does not erase that exposure or make v2 the original primary result. `observe Dice -> change topology/intensity threshold -> rerun same v1` is forbidden. Record first GT-access/evaluation time and adapter code/config hashes to enforce this boundary.

| Failure or ambiguity | Required v1 behavior |
| --- | --- |
| One anatomy spans several clusters | Merge compatible observed components only; retain individual evidence and union trace. |
| One connected component mixes two anatomies | VOID for that component; if essential to a cavity/ring hypothesis, reject dependent assignments too. |
| RV cannot be resolved | RV status unresolved; unassigned candidates VOID, not forced RV or BG. |
| MYO ring is incomplete | Do not close it. Reject the enclosure-based pair; retain only assignments independently supported by the fixed rules. |
| LV/MYO topology absent, apical/basal ambiguity | Leave LV/MYO unresolved rather than using largest/brightest-region shortcuts. |
| Extra surrounding tissue clusters | BG only under the same explicit background rule; otherwise VOID. |
| Equal anatomical hypotheses | Abstain; geometry ordering is for reproducibility, not manufactured certainty. |
| Clustering failure | Export a failure record and all-VOID semantic prediction, retaining the sample in coverage/failure accounting. |

Permutation requirement: `adapter(permuted_cluster_ids) == adapter(original_cluster_ids)` exactly for semantic/validity arrays, class states and canonical decision trace, except explicitly recorded raw-ID metadata. Test arbitrary bijections, including sparse/non-contiguous labels and reversed label enumeration.

## 10. GT isolation and prediction freeze

Generation environment is not the reference evaluation environment. Mask-free means inaccessible masks, not merely an ignored `label` tensor or `no_label` flag.

- Generation mounts only allowlisted image sources, manifests, configs and output storage. No reference-mask tree, reference config or evaluator credentials. If images/masks are co-located, prepare an image-only mount from verified source inventory; filename filtering alone is not the firewall. Resolve symlinks/junctions and reject paths outside the allowlist.
- Image loader, training, latent export, clustering and adapter accept no GT-path argument or annotation-bearing schema. Do not import supervised datasets, `prepare_dataset`, `OutputSaver` or analysis entry points in the new route.
- Permit existing pure image utilities needed by CUTS, including the sampler's `utils.metrics::range_aware_ssim`; a module's historical name is not GT access. Forbid GT-assisted call paths, not legitimate image SSIM. Instrument actual file opens as well as import/call-graph checks.
- Use a two-stage seal: checkpoint/config freeze before held-out generation; complete prediction-bundle freeze before any GT scoring. In P0 a raw-only bundle may be sealed for testing, but cannot satisfy final semantic freeze or trigger scientific GT evaluation.
- Final bundle includes manifest/sample-list hash, source-image hashes, source/deviation identity, environment lock, resolved configs, checkpoint hash, latent/partition hashes, semantic and validity hashes, adapter trace/config, failures and schema versions. Every expected sample must appear, even if it failed.
- Persist explicit RNG roles in every run: `split_seed=42`, `benchmark_seed=42`, `model_init_seed=42`, `training_rng_seed=42`, exact `loader_seed` derivation based on 42, PatchSampler seed 42 and its official constructor/per-call reseeding behavior, `clustering_seed=1`, `clustering_retry_seed=2`, per-image `actual_clustering_seed_used`, whether retry occurred, attempted seeds and success/failure. On total clustering failure, record both failed attempts and no successful seed. A generic `seed: 42` alone is insufficient. Preserve the manifest hash across all model/clustering reruns.
- Canonicalize serialization, dtypes, byte order and metadata ordering. Hash array payloads and final files with SHA-256; avoid nondeterministic archive timestamps. Store a sealed index hash in an independent immutable registry/read-only release, so modifying both a file and its local hash list does not bypass verification.
- Evaluator receives the sealed bundle plus separately mounted GT and evaluation-only label convention mapping. It checks registry/index and every artifact before opening reference data, rejects mutation/missing/unexpected samples, and writes metrics to a separate directory. Predictions remain read-only.
- Disable optional reference reporting during training for this benchmark, including FreeMask's separately configured epoch reference evaluator. Image-only dev logging is allowed. No reference scores until final prediction/config freeze across the declared primary run set.
- Record the first primary reference-GT access. At that point v1 is permanently frozen, including all topology/intensity thresholds; observing any primary GT metric cannot reopen it. Every semantic-rule change requires v2, new preregistered predictions, a new seal and a new evaluation run, never overwriting the v1 release.

## 11. Proposed module/file architecture

The repository already has `src/model`, `src/data_utils`, `src/utils`, `src/scripts_analysis` and `config`. Add a narrow package and scripts; do not reorganize existing CUTS or create another model/loss framework. Paths below are proposed, not files created by this planning task.

```text
src/
  model/CUTS_model.py                 # existing, reused unchanged
  data_utils/patch_sampler.py         # existing, reused unchanged
  utils/{losses,scheduler}.py         # existing, reused unchanged
  cardiac_benchmark/
    __init__.py                      # no evaluator/legacy analysis side effects
    manifest.py                      # shared manifest import, identity/split validation
    dataset.py                       # image-only ACDC/M&Ms, geometry + profile loading
    train_stage1.py                  # narrow wrapper over official training algebra
    export_latents.py                # frozen inference=True, label-free export
    cluster_kmeans.py                # isolated official PHATE/K10 numerical kernel
    cardiac_adapter.py               # shared cardiac_adapter_v1; P1, outside CUTS core
    provenance.py                   # schemas, canonical records and source identity
    freeze.py                        # immutable bundle/index validation
scripts_cardiac/
  build_manifest.py                  # Case A import; Case B one shared FreeMask-policy build
  train_cuts.py
  export_latents.py
  generate_partitions.py
  generate_semantics.py               # P1
  freeze_predictions.py
  preflight.py
cardiac_evaluation/
  evaluate.py                        # separate executable/environment; P0 skeleton, P1 metrics
config/cardiac/
  acdc_2d.yaml, acdc_25d.yaml
  mnms_2d.yaml, mnms_25d.yaml
  cardiac_adapter_v1.yaml             # P1 immutable shared rules
benchmark/
  manifests/                         # immutable image-only manifests + provenance
  environments/                      # exact locks/build specs + deviation ledger
  protocols/                         # resolved benchmark/evaluation contracts
tests/cardiac/                        # synthetic, firewall, parity, freeze and geometry tests
```

Large artifacts live outside source control under `dataset/profile/benchmark_seed/run_id/{checkpoints,latents,partitions,semantics,freeze}`; explicit RNG-role fields remain mandatory inside provenance, not inferred from this directory. The manifest entry point never creates a CUTS-only split: Case B produces the single shared artifact for both methods via pinned FreeMask policy. Evaluation outputs have a separate root. In P2 add a `benchmark/downstream/` entry-point/config layer only if no shared CommonStudent implementation already exists; prefer consuming the same external student runner for every method.

## 12. Existing files to preserve

| Protected file/function | Primary behavior that must remain intact |
| --- | --- |
| `src/model/CUTS_model.py::CUTSEncoder` | Four 5x5 convolutions with channels `C,16,32,64,128`, same spatial resolution, replicate padding, BatchNorm/LeakyReLU; no pooling/downsampling. Only supported `in_channels` differs by named profile. |
| `src/model/CUTS_model.py::PatchRecon` | Existing two-linear-layer reconstruction core and channel-dependent patch output. |
| `src/data_utils/patch_sampler.py::PatchSampler`, `compute_ssim` | Official RNG reseeding, CPU SSIM, sample/fallback behavior and parameters, including quirks. |
| `src/utils/losses.py::NTXentLoss` | Temperature, normalization, within-image positive/negative construction and reductions. |
| `src/utils/scheduler.py::LinearWarmupCosineAnnealingLR` | Official schedule implementation; same invocation/order. |
| `src/main.py::train` numerical recipe | AdamW, MSE, mixture coefficient, scheduler and dev-loss selection; wrapper changes plumbing, not scientific algebra. |
| `src/scripts_analysis/helper_generate_kmeans.py::phate_clustering` | PHATE parameters, per-image fitting, K=10 and locked library semantics, mirrored in isolated kernel. |

Do not edit these source files for the primary path if wrappers suffice. If an unavoidable runtime compatibility patch is discovered, first document the exact failure, minimal patch, source diff and parity evidence; do not silently modernize behavior. Architectural/objective/sampler redesign is a separate scientific variant, not a compatibility patch.

## 13. Existing files to wrap/replace

“Replace” below means bypass in the new benchmark executable, not overwrite the legacy file.

| Existing path/function | Current behavior/problem | Proposed boundary replacement | Scientific impact |
| --- | --- | --- | --- |
| `src/data_utils/prepare_dataset.py::prepare_dataset`, `split.py::split_dataset` | Legacy dataset dispatch, random item split, full-dataset test | `manifest.py` + `dataset.py`, fixed patient cohort and no labels | Necessary patient-held-out protocol change; objective unchanged. |
| `src/datasets/*.py` | Dataset-specific loaders; no verified cardiac shared-manifest contract | One cardiac image-only dataset over both dataset namespaces | New domain/input preprocessing, explicitly specified. |
| `src/main.py::train` and CLI | Existing training recipe; CLI automatically calls `test` | `train_stage1.py` reuses core modules, validates train/dev identities, never dispatches test | Isolation and provenance change. |
| `src/main.py::test` | CPU, label-carrying loader, sampled/full reconstruction export | `export_latents.py`, eval/inference-only latent return | Same latent computation; avoid unnecessary reconstruction. Validate parity. |
| `src/utils/output_saver.py::OutputSaver` | Label-bearing/sample-index archives; no cardiac native-geometry contract | `provenance.py` exports schema-validated label-free arrays and geometry | Artifact/interface change, not representation change. |
| `helper_generate_kmeans.py::generate_kmeans` and `generate_kmeans.py` | Label-derived shape, `label_hint_seg`, Dice and label-bearing files | `cluster_kmeans.py`, shape from latent/image, raw partition only | Remove oracle post-processing; preserve raw numerical clustering. |
| `src/utils/segmentation.py::label_hint_seg`, `point_hint_seg` | GT-derived foreground/point conversion | Do not call; P1 shared deterministic adapter | New external benchmark interface, never attributed to CUTS core. |
| `src/utils/metrics.py::guided_relabel`, `src/utils/diffusion.py::cluster_indices_from_mask`, `src/scripts_analysis/run_metrics.py` | GT relabeling/merging and GT-assisted diffusion evaluation | Separate isolated evaluator; optional oracle runner in P2 | No oracle-assisted primary semantic prediction. |
| `src/utils/parse.py::parse_settings` | Legacy paths/settings assumptions | Explicit pathlib-based cardiac config resolution, allowlisted schema | Runtime/plumbing only; no new model defaults hidden in parser. |
| `src/utils/seed.py::seed_everything` | Existing RNG/backend settings | Reuse/record primary settings; audit deterministic behavior in pinned environment | Any needed backend-flag change must be explicit and tested. |

Forbidden in primary generation: `label_hint_seg`, `point_hint_seg`, `guided_relabel`, GT-assisted cluster merging and GT-best diffusion level selection. Leaving legacy files in the repository is acceptable; reaching them from the primary executable is not.

## 14. P0 implementation tasks

P0 proves the pipeline is clean, reproducible, method-faithful and runnable with one engineering/scientific primary benchmark/model seed, **42**. Multiple model seeds are not required. Keep `split_seed=42` fixed and PHATE seeds 1/exception-only 2 separate. Do not implement diffusion, the semantic adapter, or CommonStudent here.

| Order | Deliverable | Explicit PASS / FAIL |
| --- | --- | --- |
| P0.1a | Local protocol/schema verification | PASS locally: image-only schema/importer mirrors the pinned FreeMask discovery/split/context contract, fixes `split_seed=42`, detects patient overlap/forbidden annotation fields, and rejects mock fixtures for `scientific_run=true`. Fixtures never count as cohorts. |
| P0.1b | Real shared-manifest materialization on the training server | PASS A: verify/reuse an existing scientific FreeMask manifest. PASS B: create exactly one shared manifest via pinned FreeMask discovery/splitting with `split_seed=42`, freeze it before either scientific benchmark, and require both methods to consume it. DEFERRED locally without real roots. FAIL: independently generated CUTS/FreeMask cohorts, even with the same ratio/seed, or unverified/unfrozen artifacts. |
| P0.2 | Build compatibility environment and exact lock; source/deviation ledger | PASS: imports and synthetic model/loss/PHATE smoke tests succeed reproducibly. FAIL: unspecified versions, changed math or unsupported hardware without documented resolution. |
| P0.3 | Manifest importer/validator and image-only loader for both profiles | PASS: no overlap, identical central sample lists, geometry round-trip and edge-stack tests, enforced mask-inaccessible process. FAIL: uncertain identities, annotation-derived preprocessing or hidden neighbor input in 2D. |
| P0.4 | Stage-1 wrapper with train/dev-only enforcement | PASS: synthetic one-step parity and runtime sample-access log prove train-only updates, dev-only selection, no test access or dev BN updates. FAIL: test/GT access or unrecorded core deviation. |
| P0.5 | Frozen, label-free latent export | PASS: inference-only latent agrees with original model's latent return; stable IDs and native metadata preserved. FAIL: labels/placeholder label arrays or missing inverse geometry. |
| P0.6 | Raw per-image PHATE/K10 export | PASS: isolated kernel matches official pure function under lock; repeated-process raw hashes identical and retry/failure records complete. FAIL: seed search, hidden normalization or silent failures. |
| P0.7 | Provenance, raw-bundle sealing and isolated evaluator skeleton | PASS: evaluator rejects mutated/unfinished/missing artifacts and accepts a synthetic sealed bundle without generation access to GT. FAIL: writable predictions or trusted mutable local index only. |
| P0.8 | Bounded image-only preflight and integrated smoke pipeline | PASS: measured resource budgets and explicit resolved batch/grid before runs; all P0 tests pass. FAIL: OOM hidden by changing architecture, or unresolved blocking gates. |

Local P0.2–P0.8 completion authorizes no scientific run while P0.1b is deferred. Only P0.1b plus all P0 gates authorizes later approved scientific Stage-1 runs; neither authorizes GT evaluation of incomplete semantics. Preserve synthetic parity fixtures and logs as publication audit artifacts.

## 15. P1 implementation tasks

1. Freeze the method-independent adapter specification and evaluation contract using only synthetic/train/dev image-only evidence. PASS: one versioned config and rationale, no method-specific thresholds; FAIL: test/GT feedback or unresolved topology/tie rules.
2. Implement `cardiac_adapter_v1` and semantic exporter. PASS: synthetic topology, split/mixed anatomy, VOID, missing-class and permutation tests all pass; FAIL: raw-ID dependence, forced four-class labeling or synthetic ring repair.
3. Complete isolated Track A evaluator: RV/MYO/LV Dice, macro Dice, HD95 and ASSD in millimeters when native geometry is valid, coverage, unresolved rate, sample failures and patient-level aggregation. PASS: metric fixtures and VOID/empty-class rules pass; FAIL: silently excluding hard samples or calculating millimeter distances from unknown spacing.
4. Run ACDC independently for both profiles with `benchmark_seed=model_init_seed=42`, then M&Ms independently for both profiles under the same seed-42 policy, matching FreeMask's comparison seed. Keep `clustering_seed=1` and exception-only retry seed 2. Additional preregistered model seeds belong to optional P2 robustness, not primary P1 completion. Dataset ordering is operational, not weight transfer. PASS: provenance validates the explicit RNG roles, shared frozen manifest and no cross-dataset/profile checkpoint use; FAIL: cohort/config drift or retrospective favorable-seed selection.
5. Freeze the complete primary seed-42 prediction set before GT evaluation. PASS: sealed checkpoint/config/raw/semantic/validity artifacts for every expected sample and immutable v1 at first primary GT access; FAIL: selecting runs or changing any v1 semantic rule after access/metrics. Such changes require v2, new preregistered predictions, new prediction freeze and new evaluation run.
6. Release publication-ready result tables and reproducibility artifacts. PASS: profile-separated patient-level results, uncertainty, coverage/failures, runtime and complete command/config/source identities; FAIL: oracle rows mixed into primary tables or unreproducible prediction provenance.

## 16. P2 secondary tasks

Each secondary task has an independent go/no-go gate; none is required to claim P0/P1 compatibility.

| Task | PASS condition; otherwise FAIL and do not promote to primary |
| --- | --- |
| Optional seed sensitivity / multi-seed robustness | Preregister additional model/training seeds before GT evaluation, with fixed shared manifest and `split_seed=42`; report every specified run separately from the seed-42 primary. Keep official clustering seeds 1/exception-only 2. No seed search or retrospective addition of seeds after seeing seed-42 GT results to select favorable runs. |
| Track B: pseudo-label utility | Freeze train-patient CUTS and FreeMask pseudo-labels; use the same CommonStudent architecture, initialization, optimizer, scheduler, augmentation, seeds, train patients, updates and held-out evaluation. Shared VOID/ignore rule; no method-specific filtering/selection. |
| CUTS-K4 | Preregister K=4/config before relevant evaluation; reuse checkpoint/latents and adapter v1 without retuning; report separately from K10. |
| No-GT diffusion | Freeze deterministic hierarchy/level rule and abstention behavior using no GT; pass reproducibility/firewall tests. |
| Oracle partition diagnostic | Only after primary freeze: isolated raw-partition-to-GT optimal mapping or GT-best diffusion; explicitly report mapping objective, many-to-one versus one-to-one constraints and reference access. Never feed mapping back. |
| AMP | Validate finite gradients, loss/latent drift and stability without GT selection; label numerical/performance variant and retain FP32 compatibility reference. |
| GPU latent export | Compare eval latents and resulting partitions against reference within declared tolerances; differing raw hashes create a distinct provenance profile. |
| Hardened PatchSampler | Named secondary variant only after primary compatibility result; persistent RNG changes training trajectory. Benchmark cached/vectorized SSIM separately and test numerical/sampling equivalence claims. |
| Runtime/VRAM optimization | Compare measured performance and resource cost against P0 preflight; record all changes; no pooling/downsampling/model redesign. |

Track B is separate from the primary architecture:

```text
CUTS train-patient semantic predictions   -> freeze -> CommonStudent -> held-out evaluation
FreeMask train-patient semantic predictions -> freeze -> same CommonStudent -> same evaluation
```

This measures pseudo-label utility, not native CUTS architecture. Use the same student initialization per paired seed, no reference masks, and the same checkpoint policy. Freeze student update budget independently of pseudo-label coverage; log effective valid supervision and all-VOID batches. Do not silently let coverage differences change training budgets or omit patients. Exact CommonStudent implementation and checkpoint policy are **UNRESOLVED**, to be frozen before P2.

## 17. Tests and acceptance gates

All gates fail closed. Synthetic fixtures can contain test labels solely inside test/evaluation fixtures; production generation must not access reference masks.

| Gate | Test and PASS criterion | Phase |
| --- | --- | --- |
| Shared split / P0.1a | Local PASS: schema validates patient-disjoint records, fixed seed 42 and pinned contract; scientific guard rejects mock fixtures, absent roots and missing manifest hashes. | P0 local development |
| Shared split / P0.1b routes | Scientific PASS A: verified existing scientific manifest reused exactly. Scientific PASS B: one new shared pinned-FreeMask-policy manifest with `split_seed=42`, frozen before either scientific benchmark. Both methods consume identical patient/sample membership and manifest hash; pairwise split intersections empty. Independently generated cohorts fail even with the same ratio/seed. | Training server |
| RNG-role and cohort immutability | `split_seed=42`, `benchmark_seed=model_init_seed=training_rng_seed=42`, loader derivation from 42, sampler seed/behavior recorded; PHATE roles separate. Changing model/training/clustering seeds in a test cannot alter manifest hash or membership. No multi-model-seed requirement. | P0/P1 |
| Alias/duplicate protection | Repeat scans, renamed files and ED/ES/cine aliases cannot cross splits; uncertain identity rejected | P0 |
| Stage-1 test isolation | Instrument sample IDs/file opens; no test patient in train/dev loaders, optimizer steps, checkpoint selection or cohort normalization fitting | P0 |
| Dev isolation | Dev forwards produce no gradients, optimizer steps or BN running-stat mutations | P0 |
| GT firewall | No reference mount; annotation-bearing schemas rejected; attempted reference open fails; generation integration test completes without GT installed | P0 |
| No hidden oracle | Static dependency/call review plus runtime tripwire makes all forbidden hint/relabel/GT-merge functions raise; primary pipeline still succeeds | P0/P1 |
| Architecture preservation | Protected-source hashes and model structure verify four 5x5 same-resolution blocks, no pooling; shape and one-step numerical parity fixtures pass | P0 |
| Sampler preservation | Repeated calls, batch ordering, threshold/fallback fixtures match original outputs/RNG side effects under pinned versions | P0 |
| Raw partition | Same checkpoint, image, config, clustering seed and locked environment in fresh processes produce identical raw-array/file hashes | P0 |
| Clustering retry | Success at seed 1 makes no retry; injected first-attempt exception invokes seed 2 exactly once; two exceptions record failure with no third attempt. Provenance records actual seed, retry and outcomes; no quality-driven fallback. | P0 |
| Dataset/profile independence | ACDC checkpoint cannot initialize/resume M&Ms, nor can a 2D run resume 2.5D; mismatch rejected before load | P0 |
| Geometry | Asymmetric phantom with affine, flips, spacing and frame/slice indices survives export/inversion; semantics/validity use label-safe resampling | P0/P1 |
| Adapter determinism | Repeated runs and shuffled component enumeration give exactly identical semantics/validity/canonical trace | P1 |
| Adapter permutation | Random cluster-ID bijections, sparse IDs and swapped numeric ordering give exactly identical semantic/validity arrays; raw-ID trace metadata exempt only | P1 |
| Adapter GT independence | Mask tree absent; changing a separately held test-reference fixture cannot affect output; adapter schema has no GT input | P1 |
| Adapter failure handling | Mixed components, missing RV, incomplete ring, no topology, ambiguous cavities and extra tissue satisfy documented VOID rules | P1 |
| Adapter complexity / permanent v1 freeze | Only declared generic image-only signals; high VOID accepted. After first primary GT access or metric observation, any semantic-rule/hash change under v1 is rejected, even a minor threshold edit. Changes require v2, new preregistered predictions, new seal and separate evaluation; retain v1. | P1 |
| Freeze | Change one byte in a frozen prediction: evaluator rejects it; changing local hash index also fails independent seal verification | P0/P1 |
| Evaluation immutability | Prediction hashes identical before/after evaluator; evaluator output cannot be consumed by generation/checkpoint selection | P1 |
| Complete accounting | Every expected sample is predicted or marked failed/all-VOID; no dropping low-coverage images; patient aggregation includes failures | P1 |
| Student parity | Paired method runs share student config/init/update count/seed/cohort; only frozen pseudo-label content differs | P2 |

Track A metric contract, to be frozen before GT:

- Use semantic one-vs-class masks over the **whole evaluation grid** for primary Dice; VOID is not an ignored region. A GT foreground pixel predicted VOID remains a false negative. Do not remap VOID to BG or optimize foreground mappings with GT. An all-VOID prediction on present anatomy scores zero Dice.
- Report valid-only Dice, if desired, as explicitly coverage-conditioned secondary information, never the primary score. Report valid-pixel fraction, foreground coverage when calculable in evaluation, VOID fraction, class-unresolved rate and pipeline failures.
- Reconstruct per-frame volumes in native geometry before distance metrics when possible. Specify nearest-neighbor semantics/validity inversion and preserve unsupported pixels as VOID. Report stored-grid-only cases separately; missing physical geometry cannot silently yield mm metrics.
- Proposed empty-class rule: both GT/prediction empty -> class Dice/HD95/ASSD not applicable with counts; GT present/prediction empty -> Dice 0, distances infinite/failure. False-positive anatomy with empty GT -> Dice 0 and explicit surface-distance failure. Never silently nan-drop failure cases. Report finite-distance summaries alongside failure rate, not as an all-case mean.
- Average scored phases within patient first, then patients equally; macro Dice across RV/MYO/LV with the frozen absent-class rule. Report per-class denominators and patient-level bootstrap uncertainty for the seed-42 primary (record the evaluator's bootstrap RNG separately). Seed variability is optional P2 only, not estimable from the single primary model seed. No slice-weighted domination by long cine studies.
- Evaluation-only native-label conversion to `0/1/2/3` must be verified independently for ACDC and M&Ms; never assume numeric label conventions match.

## 18. Environment and reproducibility

[README.md](README.md), Dependencies (313 onward), reports Python 3.9.13, PyTorch 1.12.1, torchvision 0.13.1, torchaudio 0.12.1 and CUDA toolkit 11.3. PHATE and multiple other packages were installed without exact pins. This is a starting compatibility target, **not** a complete reproducible lock or proof it runs on the intended GPU.

1. Build an isolated compatibility environment, preferably on a recorded Linux/container target where the old stack can execute. Test native Windows only if it is the intended execution platform. Do not alter the current global environment or modernize model code first.
2. Resolve exact Python, PyTorch/CUDA/cuDNN, NumPy, SciPy, scikit-learn, scikit-image/SSIM, PHATE and dependencies, nibabel, YAML and any transitively required legacy imports. Pin wheels/conda builds/source commits with hashes; record GPU model/driver, OS, BLAS, thread settings and image/build digest.
3. **UNRESOLVED:** runnable complete historical dependency combination and hardware compatibility. Unpinned historical packages cannot be guessed from README. Test candidate locks on synthetic parity fixtures, then freeze one before scientific runs. PHATE/sklearn defaults must be captured by the lock, not quietly replaced by contemporary defaults.
4. Do not require diffusion packages merely for future experiments unless existing imports genuinely require them. FreeMask and isolated evaluation may have separate locks; do not force FreeMask's newer requirements into CUTS.
5. Record every deviation in a ledger: source location, observed failure, category, old/new behavior, rationale, parity test and affected run IDs. Preserve source snapshot and code hashes with each release.

Deviation categories:

| Category | Examples | Treatment |
| --- | --- | --- |
| Scientific/protocol change | 2.5D input, patient-held-out split, smaller BatchNorm batch, persistent sampler RNG | Explicit named profile/protocol; never claim exact training equivalence |
| Runtime compatibility change | Necessary import/path/API repair that preserves numerical behavior | Minimal patch only after failure reproduced; parity evidence and ledger |
| Performance-only candidate | Loader prefetch/caching, eliminating unused reconstruction on export | Accept as performance-only only after equivalence checks; AMP/backend/device changes can alter numerics and require separate validation |

Freeze explicit RNG roles from section 2: `split_seed=42`, `benchmark_seed=42`, model initialization/training seed 42, deterministic loader/worker derivation from 42 and unchanged PatchSampler calls with seed 42; preserve `clustering_seed=1` and exception-only `clustering_retry_seed=2`. Record all actual clustering attempts. Freeze RNG states for resume, loader order, sampler behavior, thread counts and backend flags. Repeatability tests rerun the same primary model seed 42, not multiple scientific model seeds. The existing seed helper's settings do not themselves prove bitwise determinism. If locked-environment repeatability fails, stop and resolve/document the cause rather than weakening the hash gate without disclosure.

## 19. Compute/preflight plan

Run a bounded image-only preflight before any full scientific run, after P0 cohort/environment gates. Start with synthetic tensors, then a small fixed train-only sample list. No test images, reference masks or full training needed for this check.

At each resolved primary grid and both input profiles, measure warmup separately from steady state, synchronize GPU timing, and record:

- Peak allocated/reserved GPU VRAM for physical batches 1, 2, 4, 8, 16 as feasible; available headroom, forward/backward/optimizer and dev step time.
- Stage-1 images/second and epoch estimate using actual manifest size and extension behavior; profile CPU SSIM and host/device transfers as well as convolutions.
- Frozen latent-export images/second, CPU/GPU memory, image I/O time and reconstruction-free parity.
- PHATE fit and K-means wall time separately, peak process CPU RAM, thread count, failures/retries and distribution across fixed train samples.
- Actual latent, raw partition, geometry/trace and final prediction bytes per image; estimate dataset/seed storage and checkpoint retention needs before runs.

Planning scale, not measured capacity: at 224x224 and latent width 128, a single FP32 latent contains 6,422,528 values, approximately 24.5 MiB. Full-resolution intermediate activations/autograd/workspaces and BatchNorm dominate training memory despite modest parameter count. PHATE receives 50,176 points per image; do not assume a 500-landmark setting makes every step constant-memory. Record actual process peaks before choosing parallel image workers.

Safe optimization order after the compatibility path works: image-only loader/I/O improvements and reusable frozen latent artifacts; smaller preregistered physical batch if necessary; validated GPU inference; validated AMP; then named hardened sampler variants. Gradient accumulation is not exact large-batch equivalence because BatchNorm statistics, optimizer frequency and sampler reseeding interact. CUTS negatives are within-image, not an across-batch negative bank. Any accumulation policy requires a documented secondary configuration.

No pooling, latent subsampling or encoder redesign to meet memory limits. A resolution sensitivity experiment is secondary and must apply the same grid/FOV policy to comparison methods. Preflight may stop a run for inadequate resources; it must not silently downgrade the scientific configuration. Detailed optimization comparisons belong to P2; initial resource measurement is mandatory in P0.

## 20. Expected experiment matrix

Primary Track A: all rows use train-only Stage 1, dev image-only selection, the exact same frozen patient/sample manifest as FreeMask (`split_seed=42`, accepted via P0.1 Case A or B), independent initialization with `benchmark_seed=model_init_seed=42`, and FreeMask comparison seed 42. Only one model seed is required for the primary benchmark.

| Dataset | Profile | Benchmark seed | PHATE seed | Clustering | Adapter | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| ACDC | CUTS-2D | 42 | 1 | PHATE + K10 | v1 | Method-faithful |
| ACDC | CUTS-2.5D | 42 | 1 | PHATE + K10 | v1 | Input-matched |
| M&Ms | CUTS-2D | 42 | 1 | PHATE + K10 | v1 | Method-faithful |
| M&Ms | CUTS-2.5D | 42 | 1 | PHATE + K10 | v1 | Input-matched |

Official fallback only: `PHATE/K-means seed 1 -> only on exception: seed 2 exactly once -> if that also fails: record clustering failure`. Successful seed-1 results are never retried. Seed 2 is not an alternative selected run and is never reported as an additional primary experiment; per-image provenance records its use.

Primary comparisons use FreeMask's explicitly verified native direct/pseudo-label generation endpoint, not the older supervised pipeline and not a substituted downstream student. Publish adapter coverage/failure alongside CUTS semantics so representation and interface limitations remain visible.

Secondary matrix, separate tables/claims:

| Variant | Required separation |
| --- | --- |
| Optional model-seed sensitivity | P2 only; additional seeds preregistered before GT evaluation, same immutable split/sample manifest; report all runs, no retrospective favorable-seed selection |
| CUTS-K4 | Preregistered granularity sensitivity; same v1, no per-K retuning |
| Fixed no-GT diffusion | Separate hierarchy/level-selection protocol, later only |
| Oracle mapping / GT-best diffusion | Oracle diagnostic, GT-assisted, never primary |
| Hardened PatchSampler | New trajectory/engineering variant after compatibility result |
| CommonStudent Track B | Frozen train pseudo-label utility; identical student across CUTS and FreeMask |
| AMP/GPU export/resolution variants | Explicit numerical or input changes with parity/resource evidence |

## 21. Risks and unresolved decisions

| Item | Status / required resolution | Blocking point |
| --- | --- | --- |
| Shared ACDC/M&Ms manifests and actual patient/sample IDs | **UNRESOLVED:** P0.1 may verify/reuse an existing scientific manifest (Case A), or, if none is valid, create one new shared manifest via pinned FreeMask policy with `split_seed=42` and freeze before both methods' scientific runs (Case B). Both methods must consume it exactly. Historical production artifacts are not mandatory; independent cohorts fail even with the same ratio/seed. Templates/legacy splits alone are insufficient. | P0.1, before downstream data-path implementation/scientific runs |
| Actual image roots, aliases, frame inventory, source preprocessing and native geometry | **UNRESOLVED.** Inventory images only and reconcile with the shared manifests; reject unverifiable GT-derived preprocessing. | Loader specification |
| Actual grid and normalization contract | **UNRESOLVED.** 224 is template evidence only; freeze support/range and demonstrate sampler SSIM compatibility without changing sampler. | Loader implementation/preflight |
| FreeMask native evaluation endpoint/observation support | **UNRESOLVED.** Pin artifact schema and generation route; distinguish its internal predictive protocol from context matching. | Comparative result protocol |
| Exact dependency lock and target GPU feasibility | **UNRESOLVED.** Historical README pins are partial; require runnable synthetic parity tests and measured resource budget. | Scientific Stage-1 runs |
| Physical batch and compute allocation | **UNRESOLVED.** Proposed B16; resolve before runs without GT. Primary seeds are resolved: split/benchmark/model/training 42, PHATE 1 with exception-only retry 2; additional model seeds are optional preregistered P2 only. | Run-config freeze |
| Adapter constants and conservative-rule adequacy | **UNRESOLVED.** Freeze minimum-sufficient shared v1 on synthetic/image-only evidence; high VOID is acceptable. Permanently immutable at first primary GT access/metric observation; any semantic-rule change requires v2, new preregistered predictions, freeze and evaluation. | P1 implementation |
| Basal/apical slices, incomplete myocardium, disease and M&Ms appearance shift | Expected limitation; do not force topology/brightness assumptions or tune separate per-dataset adapters. Report failure distribution. | Interpretation, not license to tune |
| PHATE nondeterminism, defaults, CPU RAM and retry behavior | Pin dependencies/threads, preserve one retry and test repeatability. Do not search seeds or drop failures. | Raw partition gate |
| Evaluator label conventions/empty-class and geometry rules | **UNRESOLVED until reviewed.** Freeze proposed metric contract and dataset label conversions independently before scoring. | P1 evaluator release |
| Shared CommonStudent implementation/budget/selection | **UNRESOLVED**, not a P0/P1 dependency. | P2 Track B |

No claim of completed training, runtime compatibility, resolved production cohorts or adapter quality is made by this plan. A failed conservative adapter is a measurable benchmark limitation, not permission to consult GT during generation.

## 22. Definition of Done

The primary CUTS cardiac baseline is complete only when all ten conditions hold:

1. CUTS and FreeMask consume the exact same immutable patient/sample manifest with `split_seed=42`, accepted either by verifying/reusing an existing scientific manifest (Case A) or by creating one shared pinned-FreeMask-policy manifest before either scientific benchmark (Case B). Independently generated cohorts never satisfy this condition, even with the same ratio/seed.
2. ACDC and M&Ms are independent runs, without checkpoint/data-statistics transfer.
3. CUTS Stage 1 has no test exposure, including updates, dev selection and cohort normalization fitting.
4. Reference GT is inaccessible throughout prediction generation.
5. CUTS core behavior remains method-faithful within the explicitly reported 2D/2.5D input profiles and documented compatibility deviations; official sampler is preserved.
6. K10 raw partitions are reproducible under the exact locked environment with `clustering_seed=1`, exception-only retry seed 2 exactly once and failure thereafter; actual attempts and outcomes are recorded, without seed search.
7. Shared adapter v1 is the minimum sufficient deterministic, non-learned, GT-free, cluster-ID-invariant semantic resolver, with honest VOID/failure handling. First primary GT access/metric observation permanently freezes v1; all later semantic-rule changes require v2, new preregistered predictions, a new freeze and a new evaluation run.
8. All primary predictions/configs/checkpoints are frozen and independently hash-sealed before GT evaluation; the evaluator never modifies them.
9. CUTS-2D and CUTS-2.5D results are separately reported for both datasets, with patient-level metrics, coverage and unresolved/failure rates.
10. Direct semantic results can be reproduced from manifests, source/image hashes, environment lock, config, checkpoint and explicit RNG-role provenance, including geometry and adapter traces. Primary benchmark/model/training and FreeMask comparison seed is 42; loader seeds derive from 42; official PatchSampler behavior is preserved under seed 42. Split seed 42 is independent of model reruns. Additional model seeds are optional preregistered P2 only, never a primary completion requirement.

P0.1a plus local P0.2–P0.8 completion is not scientific P0 completion or publication readiness. P0.1b must materialize real frozen ACDC/M&Ms manifests on the server first. P2 is optional and cannot repair or replace a failed primary acceptance gate without a new declared protocol.

Implementation order summary:

```text
P0 — required before any scientific run:
     verify/reuse or establish one shared seed-42 manifest and data contract -> lock environment ->
     image-only 2D/2.5D loaders -> train/dev-only CUTS -> clean latent/PHATE-K10 export ->
     provenance/freeze/evaluator skeleton -> firewall/parity/reproducibility/preflight gates.

P1 — required before publication runs:
     freeze and implement shared adapter v1 -> permutation/VOID/geometry tests ->
     isolated Track A evaluator -> independent dataset/profile runs at benchmark seed 42 ->
     freeze complete prediction set -> patient-level evaluation and release artifacts.

P2 — secondary analyses and engineering/performance improvements:
     preregistered seed sensitivity, CommonStudent utility, K4, fixed no-GT diffusion, isolated oracle
     diagnostics, validated AMP/GPU export, hardened sampler and detailed profiling.
```
