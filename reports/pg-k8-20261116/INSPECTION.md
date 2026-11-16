# Candidate PG / semi-K8 source inspection

**Inspection date:** 2026-11-16
**Scope:** source/spec/config/report read-through before any Candidate PG implementation.
**Write ownership:** this file only; no old-track source, config, run, server, or checkpoint was changed.
**Evidence boundary:** this is a source and receipt inspection. It is not a PG result, GPU run, patient efficacy claim, or proof that the proposed MedNeXt/PG/KD design is valid.

## 1. Read set and governing contract

The complete FUSE contract was read first (`docs/CANDIDATE_FUSE_SPEC.md:1-611`). It locks native full-volume operation, seed 42, TRAIN80/VAL18/TEST50 identity, disjoint frame keys for L/U, no crop/resampling, teacher-only z-score, the FUSE student/fusion objective, epoch-wise VAL selection, the 0.01 deploy margin, and one three-mode TEST transaction (`docs/CANDIDATE_FUSE_SPEC.md:25-72`, `81-157`). The public interfaces and runtime geometry are explicitly specified in `docs/CANDIDATE_FUSE_SPEC.md:159-299`; the semi-K8 amendment and its L8/U/mask rules are in `docs/CANDIDATE_FUSE_SPEC.md:331-393`; the A-aug augmentation/RNG amendment is in `docs/CANDIDATE_FUSE_SPEC.md:526-611`.

Files inspected directly:

- FUSE: `self_audit/candidate_fuse/student.py`, `fusion.py`, `engine.py`, `augmentation.py`, `data.py`, `export.py`, `config.py`; `evaluation.py` was also read because it owns the metric and TEST transaction APIs.
- Q3D dependencies: `self_audit/candidate_q3d/alignment.py`, `losses.py`, and the relevant `_sdf_targets`/teacher-loss sections of `engine.py`; `teacher_resenc.py` was checked to confirm the live output keys.
- PG request contract: `reports/pg-k8-20261116/REQUEST.md` was read as the current v2 amendment (code/local checks only; real S1/S2/S3/A1-A4 results remain NOT_RUN).
- K8/config/receipts: `configs/candidate_fuse/semi_k8.json`, `scripts/fuse_build_semi_manifest.py`, `splits/acdc_patient_split_seed42.json`, `reports/fuse-semi-20261113/{recipe.A.json,EPOCH_BUDGET_LOCK.json,LAUNCH_RECEIPT.json}`, and `reports/fuse-semi-20261113/track-A/{README.md,RESULT.json,RUN_STATE.json,TEST_ONCE/TEST_TRANSACTION.json}`.
- Determinism/diagnostic receipts: `reports/fuse-acdc-k8-aug-20261114/{README.md,PHASE0.md,PHASE0_RECEIPT.json,PHASE0_L8.json}`.

The source tree has no `.codegraph/`; Semble was used before source navigation. The focused Semble query returned the FUSE export/fusion, Q3D `_sdf_targets`, FUSE engine Track-A gate, Q3D loss, and FUSE data implementation locations, after which the files were read at those source locations.

## 2. Reusable FUSE APIs and exact behavior

### 2.1 Data, frame identity, and geometry

| API/contract | Source-grounded behavior | PG consequence |
|---|---|---|
| `Frame` | Frozen dataclass fields are `image`, `target`, `geometry`, `patient_id`, `frame_index` (`candidate_fuse/data.py:57-64`). | PG data adapters should return this exact shape rather than inventing a second frame object. |
| `FuseData(manifest_path, expected_sha256=None, *, role, synthetic=False)` | Checks role, manifest SHA, v1/v2 schema, `synthetic`, class map, split disjointness, and fixed 80/18/50 sizes for real data (`data.py:66-176`). Runtime geometry is canonicalized from exactly `shape`, `spacing_zyx`, `affine_zyx`, `context="full_volume"`; manifest provenance is retained outside runtime geometry (`data.py:49-55`, `docs/CANDIDATE_FUSE_SPEC.md:161-165`). | PG must consume the same native geometry and SHA-bound identity. No crop, resample, or inferred shape. |
| `labeled_keys()` / `unlabeled_keys()` | Evaluation roles expose only their ED/ES labeled frames (`data.py:298-303`); TRAIN L is ED/ES only and TRAIN U excludes annotated ED/ES (`data.py:305-310`). | PG never opens a TRAIN U mask; VAL/TEST target reads stay evaluation-only. |
| `load(key, *, labeled)` | TRAIN labeled loads require ED/ES and read the mask; TRAIN U loads reject ED/ES and return `target=None`; development/test reject image-only access and allow only their labeled ED/ES (`data.py:312-340`). | Keep this fail-closed access gate; do not add a PG path that can load U/VAL/TEST masks during training. |
| `FuseSampler(data, *, seed=42)` | TRAIN-only; separate seeded streams for T frame permutations, J patient permutations, L stream, and U frame choice. `next_labeled_batch(1|2)`, `next_pair()`, `state_dict()`, `load_state_dict()` are checkpointable (`data.py:343-443`). | PG training must preserve cursor/RNG checkpoint semantics if it reuses the engine. |
| `open_fuse_data(...)` | Dispatches v1/v2 to `FuseData` and v3 to `MNMSFuseData`, retaining the existing API (`data.py:483-495`). | PG ACDC K8 can use v2; do not silently route to M&Ms v3. |

Runtime tensors are image float32 `[1,1,D,H,W]`, target long `[1,D,H,W]`, labels 0..3 (`docs/CANDIDATE_FUSE_SPEC.md:165-166`; `data.py:312-340`). `FuseData` checks image arrays as finite normalized `[T,D,H,W]`, exact declared shape, and masks as native shape with labels 0..3 (`data.py:252-288`).

### 2.2 Semi-K8 lock and reports

The semi builder is metadata-only: it reads JSON and NumPy image headers, never mask/NIfTI payloads or copies arrays (`scripts/fuse_build_semi_manifest.py:1-8`). It pins the expected K8 source SHA `60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433`, requires seed 42, label budget 8, and the reviewed K8 schema/class map (`scripts/fuse_build_semi_manifest.py:22-27`, `60-78`). It checks that the selected L/U IDs partition TRAIN80, that each selected L image/mask SHA agrees with the full-label source, and that selected masks are exactly source ED/ES (`scripts/fuse_build_semi_manifest.py:146-186`). The output keeps masks only for L8 and strips mask references from every U record; the labeled contract contains sorted IDs, an ID-list SHA, source manifest path/SHA, and the receipt records zero mask/NIfTI reads (`scripts/fuse_build_semi_manifest.py:188-216`, `218-268`).

The real semi recipe receipt uses external, hash-pinned manifests and the FUSE v2 schema: train SHA `ef7562a251e5c30622ab00dfa46a5b62e0c4552cc0aac06b24caa3e63ff23363`, VAL SHA `e3421779441afcab8f8feadbd293bedad6c00877730eb3c7151fa6a15667e88b`, TEST SHA `84f24ca7a50e4516548cc16037b6159eaed2eb1ea07d94287e8305894a951297`; teacher batch 2, 500 T epochs, 100 J epochs (`reports/fuse-semi-20261113/recipe.A.json:1-18`, `166-179`). `EPOCH_BUDGET_LOCK.json` records L8 = 8 patients × 16 ED/ES frames, 8,000 T exposures, 4,000 physical-batch-2 optimizer steps, 400 warmup exposures, and 8,000 J pairs; selection is VAL-only and TEST is one locked three-mode transaction (`reports/fuse-semi-20261113/EPOCH_BUDGET_LOCK.json:1-32`).

The ACDC L8 patient IDs observed in the locked Phase0 report are `patient007`, `patient010`, `patient011`, `patient024`, `patient026`, `patient041`, `patient077`, `patient099` (`reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json:19-184`). This is a receipt-derived identity check, not permission to regenerate a manifest. The A-aug report confirms the same L8/U manifests, seed42, native geometry, physical batch2, T500/8,000 exposures/4,000 steps, J100/8,000 pairs, and unchanged VAL selection/deploy margin (`reports/fuse-acdc-k8-aug-20261114/README.md:12-19`).

**Config hygiene note:** `configs/candidate_fuse/semi_k8.json` contains `teacher_batch_size` twice (lines 172 and 174); JSON loading keeps the final value `2`, and `validate_config` only sees the deduplicated mapping. The real receipt `recipe.A.json` has one key (`:166-179`). PG should emit a canonical config without duplicate keys and should bind its config SHA before any run.

### 2.3 Student, fusion, phases, and dropout

`FuseStudent(config, *, teacher_channels=None, teacher_phases=None, include_fusion=True, include_mimic=True)` validates five EPA-compatible channel widths, four classes, and constructs stem, stages, decoder, region logits head, optional `GatedFusion` modules, and optional S2/S3/S4 mimic heads (`candidate_fuse/student.py:50-78`). The live phase tuple is derived from the actual stem/down-conv path (`student.py:79-85`), not guessed constants. `forward(image, geometry, *, teacher_features=None, keep_mask=None)` validates normalized finite input and native shape, requires five teacher features plus binary `[B,1,1,1,1]` keep masks when supplied, fuses aligned detached teacher features, and returns `logits`, five `features`, `gate_means`, and `mimic_loss` (`student.py:94-138`). `infer_logits(image, geometry)` is the pure student route (`student.py:158-159`).

`GatedFusion(student_channels, teacher_channels)` is a 1×1 teacher projector plus a gate initialized to weight zero and bias -2; it returns `student + keep_mask * sigmoid(gate([student,projected])) * projected`, with teacher detached (`candidate_fuse/fusion.py:12-36`). `align_feature(...)` calls Q3D `align_teacher_to_student4`, then only high-side replicate-pads an internal student grid (`fusion.py:39-51`). The alignment helper samples native target centers, clamps endpoint coordinates, and preserves declared phase (`candidate_q3d/alignment.py:178-224`).

Teacher dropout is `p=0.5*update/max(total_updates-1,1)` and `sample_keep_mask` draws one binary mask per sample using an optional saved generator (`student.py:29-41`). In a dropped sample, training-only mimic averages `1-cosine` over S2/S3/S4 valid native centers; padding is excluded and targets detach (`student.py:124-135`). The spec additionally requires m=0 to be bitwise equal to the pure student path and no unused fusion calculation (`docs/CANDIDATE_FUSE_SPEC.md:103-115`); this is a PG regression gate if the subclass changes the forward path.

### 2.4 Augmentation and deterministic state

`StrongInplaneAugment(seed=42, *, probabilities=None)` owns a private CPU generator seeded `run_seed+2003`, validates float32 `[1,1,D,H,W]`/long `[1,D,H,W]`, and strictly serializes generator state plus probability map (`candidate_fuse/augmentation.py:21-52`, `54-67`; `docs/CANDIDATE_FUSE_SPEC.md:582-597`). `spatial`, `intensity`, `teacher_view`, and `joint_views` implement the locked ordered operations and return unchanged metadata/shape (`augmentation.py:94-138`, `140-186`). Engine checkpoints the augmentor state under `rng.strong_augmentation` when enabled (`engine.py:283-307`). `prepare_teacher_batch` applies independent views per frame while requiring same-patient native shape/geometry for a true stacked batch (`engine.py:310-340`).

### 2.5 Objective and epoch/checkpoint orchestration

`joint_loss(student, teacher, labeled, unlabeled, *, update, total_updates, config, generator=None, augmentation=None)` requires a frozen/eval teacher, applies shared teacher/student spatial views, samples separate L/U keep masks, obtains detached teacher region logits/features, computes `CE+foreground Dice + lambda_u*SoftCE + mimic_weight*(mimic_L+mimic_U)`, and returns scalar loss plus per-term/gate/dropout stats (`candidate_fuse/engine.py:119-170`). It does not add an SDF student term; this agrees with the FUSE spec (`docs/CANDIDATE_FUSE_SPEC.md:117-127`).

`run_training(config,out,*,device="cpu",stage="all",resume=False,max_updates=None,teacher_artifact=None,profile=False)` is the canonical engine entry (`engine.py:722-733`; spec `:235-264`). It validates real manifest paths/SHA, applies Track-A identity/L8 checks, constructs teacher/student, and fixes T/J budgets from manifest counts and config (`engine.py:760-795`). T evaluates VAL every epoch, selects earliest-tie best teacher, writes `teacher.pt` plus metadata, freezes/cleans the teacher before J, and J evaluates teacher/student/fusion every epoch, selects the best fusion VAL snapshot, and writes `joint_best.pt` (`engine.py:880-975`, `1026-1104`). Resume stores identity, optimizer/sampler/RNG state, committed metric-log receipt, and archives uncommitted suffixes (`engine.py:260-307`, `420-427`, `514-527`, `1062-1089`).

### 2.6 Export, metric, and TEST APIs

`export_model(student, teacher, config, path, *, mode, metadata=None)` supports exactly `teacher`, `student`, and `fusion`; writes weights-only payload, absolute path/SHA/mode receipt, sidecar, config/data identity, parameter count, and latency metadata (`candidate_fuse/export.py:56-115`). Student export constructs a student without fusion/mimic; fusion export includes frozen teacher plus fusion student; `load_export(path, *, device="cpu", expected_sha256=None)` verifies sidecar bytes before weights-only deserialization and reconstructs strict architecture (`export.py:164-192`). `verify_cold_load(path, volumes, *, original_model, device="cpu")` requires exactly three volumes and checks original, reloaded, and fresh-interpreter outputs bitwise (`export.py:230-299`).

`evaluate_modes(teacher, student, data, *, device="cpu", modes=("teacher","student","fusion"))` evaluates only labeled ED/ES frames, preserves model modes and RNG, computes patient-macro per-class Dice/HD95 in native spacing, and returns `fg_dice`, `per_class`, `patient_count`, and `per_patient` (`candidate_fuse/evaluation.py:80-173`). `choose_deployment(metrics, *, margin=.01)` is fixed to `student >= fusion - .01` (`evaluation.py:176-188`).

`evaluate_export(...)` is VAL/development-only and explicitly rejects single-mode TEST (`evaluation.py:191-258`). `evaluate_exports(bundle,data_manifest,out,device,role,expected_sha256)` accepts exactly three FUSE receipts, pins every artifact/sidecar/manifest SHA before opening role data, creates an atomic `TEST_TRANSACTION.json`, refuses an existing ledger/rerun, evaluates all three modes, and commits or locks failure (`evaluation.py:275-451`). This matches the historical Track-A receipt: one committed TEST attempt, three export SHAs, TEST manifest SHA, and no rerun (`reports/fuse-semi-20261113/track-A/TEST_ONCE/TEST_TRANSACTION.json:1-18`; `RUN_STATE.json:9-21`, `50-57`).

## 3. Q3D interfaces PG may reuse, and the head conflict

`GridPhase` is `((jz,jy,jx),(oz,oy,ox))`; `grid_phase` derives step/offset from the actual Conv3d/AvgPool3d chain and rejects unsupported modules (`candidate_q3d/alignment.py:15-46`, `49-91`). `align_teacher_to_student4` computes target lattice shape as `ceil(native/step)`, samples the teacher at native target centers, clamps endpoint coordinates, and returns detached values (`alignment.py:143-224`). `pool_native_cells`/`pool_teacher_targets` implement centered native-cell pooling with valid support and explicit even-stride boundary weights (`alignment.py:340-421`).

The Q3D SDF target helper is `_sdf_targets(labels, geometry, *, radius_mm)`: it constructs `QueryBank3D(BankConfig3D(radius_mm=...))` and calls `bank.sdf_targets(labels, geometry["spacing_zyx"])` (`candidate_q3d/engine.py:588-592`). FUSE `teacher_loss` calls this helper with radius 5 mm, then calls Q3D `q3d_teacher_loss(..., lambda_distance=1.0, radius_mm=5.0, near_mm=2.5, alpha=.5)` (`candidate_fuse/teacher.py:213-221`).

`q3d_teacher_loss(outputs, labels, sdf_targets, ...)` requires the output key set **exactly** `{"region_logits","boundary_logits","sdf"}` and computes separate supervised region and boundary CE+foreground-Dice terms plus balanced near/far SDF (`candidate_q3d/losses.py:94-126`). The balanced SDF function expects `[B,4,D,H,W]`, physical-radius-normalized targets in `[-1,1]`, and optional unknown-label masking (`losses.py:32-91`). Q3D engine stage T computes `_sdf_targets` before the teacher forward and passes all three outputs to `q3d_teacher_loss` (`candidate_q3d/engine.py:1287-1307`); the live ResEnc teacher actually emits all three keys (`candidate_q3d/teacher_resenc.py:425-461`).

**Resolved compatibility decision:** the PG request keeps the public teacher output as `region_logits,sdf,features5` but explicitly retains a compatibility `boundary_logits` head so the unchanged Q3D `q3d_teacher_loss` contract remains usable (`reports/pg-k8-20261116/REQUEST.md:43-45`). The current FUSE/Q3D declaration is `heads: ["region", "boundary_sdf"]` (`configs/candidate_fuse/semi_k8.json:89-92`; `configs/candidate_q3d/v1.json:266-273`); PG must document this boundary compatibility head and must not silently pass only region/SDF to `q3d_teacher_loss`.

## 4. Candidate PG design boundary (request-resolved proposal; no code yet)

The requested PG track is isolated under `self_audit/candidate_pg/` with its own config namespace under `configs/candidate_pg/`; no files under `candidate_fuse`, `candidate_q3d`, `candidate_a`, or `candidate_a3d` may be modified. The exact v2 request requires `configs/candidate_pg/pg_k8_base.json` plus arm files containing only switch dictionaries, schema `self_audit.candidate_pg.v2`, and four identical resolved arms: A1 (PG off/CBKD off), A2 (PG on/CBKD off), A3 (PG off/CBKD on), A4 (PG on/CBKD on) (`reports/pg-k8-20261116/REQUEST.md:15-17`, `9-13`).

The intended student base is a `FuseStudent` subclass without changing FUSE itself; its widths are `[16,32,64,128,256]` and it adds FP-only pre-decode skip dropout, a train-only code branch, and post-EPA stage3 (128 channels) output (`reports/pg-k8-20261116/REQUEST.md:23-25`, `55-58`). Q3D v1 happens to declare `[16,32,64,128,256]` for its own UNETR++-lite student (`configs/candidate_q3d/v1.json:169-209`), but that does not establish PG architecture equivalence.

The PG request fixes one labeled frame plus two patient-balanced U frames per update, U partner from a different patient, and exact K8/native/no-VAL-or-TEST-training identity (`reports/pg-k8-20261116/REQUEST.md:9-13`, `19-21`). It fixes an anisotropic MedNeXt-L teacher with channels `[32,64,128,256,512]`, declared feature phases, region and SDF heads (plus the compatibility boundary head above), random initialization, 8,000 updates, and VAL18 teacher-region-FG selection (`reports/pg-k8-20261116/REQUEST.md:27-28`, `43-45`). S1 CineMA initialization is a separate TRAIN-image-only stage with its own official-source/config/SHA and encoder-only student checkpoint; it must not use ACDC/M&Ms finetuned checkpoints (`reports/pg-k8-20261116/REQUEST.md:25`, `39-41`).

The new anisotropic MedNeXt-L encoder/blocks, optional PG pathway, and optional frozen-codebook auxiliary KD must be defined and tested inside `candidate_pg`. Reuse may be limited to stable contracts (native `Frame`/geometry, phase/alignment primitives, FUSE metrics/export patterns) with explicit adapter boundaries. If PG emits a new auxiliary output, keep the canonical segmentation `logits` path compatible with `FuseStudent.forward` until a separate PG API is frozen. Frozen-codebook KD must make teacher/codebook tensors detached and record whether the auxiliary branch was enabled; the request fixes a full-C frozen codebook, student stage3 tap, assignment/commit/qtask terms, and `lambda_cb=.5` (`reports/pg-k8-20261116/REQUEST.md:30`, `34`, `62-65`). No codebook/tokenizer/physiology component is part of FUSE by default (`docs/CANDIDATE_FUSE_SPEC.md:9-14`).

Required pre-implementation checks for PG:

1. Preserve the resolved boundary compatibility head and write it into the PG teacher wrapper/config.
2. Implement the request's exact PG/CBKD switches and coefficients; no per-arm tuning or hidden defaults (`reports/pg-k8-20261116/REQUEST.md:32-36`, `60`).
3. Define anisotropic MedNeXt-L phase chains and expected `GridPhase` values from live Conv3d modules; do not copy Q3D offsets by assumption (`reports/pg-k8-20261116/REQUEST.md:27`, `43-45`).
4. Add PG-only config schema and SHA identity; preserve FUSE v1/v2/v3 config validation unchanged.
5. Define a PG export payload/sidecar contract before writing any PG code (`reports/pg-k8-20261116/REQUEST.md:37`, `72`).

## 5. Output-contract conflicts to resolve before PG implementation

### 5.1 TEST “per epoch” versus FUSE one-shot TEST

The historical/spec FUSE rule is one locked three-mode TEST evaluation after VAL selection (`docs/CANDIDATE_FUSE_SPEC.md:141-157`, `261-284`; `candidate_fuse/evaluation.py:275-451`). The PG v2 request explicitly authorizes per-epoch monitoring while retaining a final one-shot transaction (`reports/pg-k8-20261116/REQUEST.md:67-73`). Treat this as a **PG-only override**, never a change to `evaluate_exports` or historical FUSE receipts:

- Keep FUSE TEST once-only and ledger-compatible.
- Define a PG ledger with explicit `epoch`/`checkpoint`/artifact SHA on every row, immutable VAL selection fields, and a final status that cannot be mistaken for a one-shot FUSE transaction.
- Evaluate TRAIN16/VAL18/TEST50 every 80 updates (one PG epoch), with TEST values descriptive only; TEST cannot select/checkpoint/budget PG even when observed every epoch.
- The final `TEST_ONCE` transaction locks all four selected arm export hashes; it is not a claim that TEST was previously unseen (`reports/pg-k8-20261116/REQUEST.md:67-73`).
- Use a new PG run/output directory and new schema, with no reuse or mutation of the existing Track-A `TEST_TRANSACTION.json`.

### 5.2 Fixed epoch columns versus `best_val`

The current engine emits variable event rows (`train_update`, `validation`, `validation_selected`) with `stage`, `epoch`, and `update` (`candidate_fuse/engine.py:228-230`, `947-968`, `1081-1104`). Selection is a separate best snapshot (`joint_best.pt`) keyed by fusion VAL FG and earliest ties (`engine.py:1073-1080`; spec `:141-147`). The PG v2 request resolves the report shape: fixed ordered columns followed by `best_val` as the final appended column (`reports/pg-k8-20261116/REQUEST.md:67-71`). Therefore:

- one unique row per `(epoch, split, path)` with this ordered prefix: `stage, arm, epoch, update, split, path, loss_total, loss_sup, loss_pg, loss_cbkd, fg_dice, rv_dice, myo_dice, lv_dice, fg_hd95, rv_hd95, myo_hd95, lv_hd95, fg_assd, rv_assd, myo_assd, lv_assd, lr, seconds`;
- append `best_val` as the final column, marking the selected VAL snapshot consistently after selection (not a replacement for the fixed epoch columns);
- no replacement of missing/undefined values with zero, and no overwrite of prior epoch rows;
- tie policy must be stated (FUSE uses earliest tie), and TEST columns must be labeled descriptive/post-selection when present.

This output schema is resolved in the request and remains unimplemented until the PG worker writes it; all four arms must use identical rows/columns and TEST changes must not affect VAL-only selection (`reports/pg-k8-20261116/REQUEST.md:69-73`).

## 6. Deterministic seed/TF32 policy to carry into PG

The FUSE engine sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, enables deterministic algorithms (CPU strict; CUDA warn-only for unsupported kernels), sets cuDNN deterministic and benchmark inverse, and sets both CUDA matmul and cuDNN TF32 to `not deterministic` (`candidate_fuse/engine.py:733-753`). With the locked `deterministic=true`, TF32 is therefore disabled. It seeds Python, NumPy, Torch, and CUDA with root seed 42 (`engine.py:758-759`). The FUSE sampler uses seed 42; strong augmentation uses a private CPU generator seed `42+2003`; joint keep-mask sampling uses a checkpointed Torch generator seeded 42 (`data.py:346-352`, `augmentation.py:24-27`, `engine.py:773`, `student.py:35-41`). Evaluation snapshots/restores Python/NumPy/Torch/CUDA RNG (`candidate_fuse/evaluation.py:35-45`).

The final historical Phase0 diagnostic explicitly matched deterministic/TF32 flags; the earlier default-TF32 diagnostic is retained separately and is not exact historical inference (`reports/fuse-acdc-k8-aug-20261114/PHASE0.md:1-12`, `PHASE0_RECEIPT.json:1-14`). PG should record the same flags and seed tuple in its run receipt. Any PG experiment that intentionally enables TF32 must use a PG-only config identity and label its determinism/equivalence status; it must not overwrite or reinterpret FUSE receipts.

## 7. Inspection result

**INSPECTION COMPLETE; no implementation was performed.** The reusable source APIs, K8 lock and receipts, TRAIN-only mask contract, native geometry, grid phases, export/metric/TEST behavior, and deterministic seed/TF32 policy are recorded above. The previously noted conflicts are resolved by `reports/pg-k8-20261116/REQUEST.md`: retain a compatibility boundary head for unchanged Q3D loss, allow PG-only per-epoch descriptive TEST monitoring plus final four-export TEST_ONCE, and append `best_val` after the fixed ordered epoch columns.
