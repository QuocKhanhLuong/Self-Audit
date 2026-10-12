# Current-main model, data, loss, training, inference and diffusion audit

Audit date: **2026-10-02**. Status: **IMPLEMENTED source inspected; software execution NOT RUN by this worker; new real-data metrics NOT MEASURED; RTX 4080 Super measurements NOT RUN**.

## 1. Source of truth and conclusion

The authoritative snapshot for this report is **`c31825a7f90df47f9f9382a9c9595e43f7f71236`**, the refreshed `origin/main` (merge PR #29). The open workspace remains **`234be0a9cc9d557d1263025f163ef28792da4423`**, an older Astra audit branch. The initially supplied `0605e3b` was superseded by the coordinator's completed fetch. Every main-source reference below is an immutable GitHub link at `c31825a`; sources were actually read with `git show <SHA>:<path>`. The coordinator also made `/private/tmp/self-audit-main-20261002` available at that SHA. This worker made no source edits, deletions, Git writes or training runs.

**Current main contains several distinct systems. The user's confirmed fully no-GT constraint makes the runnable v3 pseudo-label pipeline and Maskfree150 relevant candidates; the README's canonical Self-Audit training recipe is mask-supervised and is not an eligible no-GT teacher. No inspected evidence establishes a current no-GT Dice >= 0.90 result or RTX 4080 Super speed.**

| Namespace / route | What main actually contains | Meaning for this task |
|---|---|---|
| `src/self_audit`, `scripts/train_self_audit.py` | ConvNeXt-Tiny/FPN, recurrent Dynamic Window annotator, learned counterfactual auditor, strict unified trainer and evaluation | Active supervised reference, shared code dependency; cannot be relabeled no-GT |
| `src/self_audit_pseudolabel`, `run_full_pipeline_v3.py` | Runnable image-only full-cine teacher, handwritten anatomical evidence, frozen pseudo-label artifacts, optional small student | Main no-GT v3 research path, with unresolved semantic quality |
| `src/self_audit_maskfree`, `train_maskfree.py` | Separate image-only producer/hypothesis/auditor/two-student timeline | Separate active no-GT implementation, not v3's trainer or teacher |
| `baseline/CUTS`, `src/shared_benchmark` | Image-only representation/clustering and separate frozen semantic evaluation protocols | Active comparison; distinguish PHATE/K-means runner from diffusion-condensation route |
| `baseline/DSS_US`, `baseline/SGSCN` | Active milestone protocol/evaluator work with explicit blocked/reference-only profiles | Preserve status; availability of code is not an executable or measured baseline |
| `baseline/DFC`, `baseline/STEGO`, `baseline/PICIE` | Explicitly out of the current milestone; historical contracts/tests remain | Historical scope does not by itself authorize deletion |

The first three routes are verified below. The milestone explicitly retains Self-Audit core/Maskfree150, shared benchmark, CUTS, DSS-US and SGSCN, and marks DFC/STEGO/PiCIE legacy. DSS-US producers remain `BLOCKED_PROTOCOL`; SGSCN paper profiles remain blocked and official-code reference profiles are `REFERENCE_READY`; native datasets/adapters also have blocked statuses. See [active milestone scope][scope].

**Important delta from the old checkout:** main adds `data_v3.py`, `trainer_v3.py`, `losses_v3.py`, `evidence.py`, `evolution.py`, `consistency.py`, `freeze.py`, `pipeline_v3.py`, `adaptive.py` and four v3 CLI scripts. The prior audit's “no loader/trainer/evaluator/CLI” finding is now historical. Main still does not establish anatomical accuracy. The current [v3 runbook][runbook] explicitly says its fixes concern integrity, not a new accuracy result.

## 2. Current no-GT v3 runtime call path

```text
scripts/run_full_pipeline_v3.py::main
  -> subprocess scripts/train_pseudolabel_v3.py
     -> pipeline_v3.teacher_main
        -> discover full-cine images + read explicit patient splits
        -> CineSliceDataset(train) + within-patient sampler
        -> ProgressiveTeacherTrainer.train_batch
           -> CinePseudoTeacher(prev, cur, nxt)           [one teacher encoding]
           -> build_region_evidence + acceptance gates
           -> teacher_loss(raw neural logits, raw evidence)
           -> AdamW step + raw-evidence prototype-bank update
        -> infer_batch on requested train/val exports
        -> motion-aligned consistency rejection
        -> native pseudo NPZs + teacher.pt + FROZEN.json
  -> subprocess scripts/evaluate_pseudolabel_frozen.py
     -> verify all frozen artifacts BEFORE reference-mask access
     -> independently score frozen teacher labels
  -> if --train-student:
     scripts/train_student_v3.py -> pipeline_v3.student_main
        -> FrozenPseudoDataset (TRAIN entries only, source-image hash match)
        -> AdaptiveAnnotationStudent(cur spatial triplet)
        -> valid-only hard pseudo-label CE + 0.25 * A0 CE
        -> student checkpoint bound to manifest ID and trained profile
```

This is a **verified static call path**, not a claim that this worker executed the pipeline. Its orchestration is [run_full_pipeline_v3.py:64–109][pipeline-cli]; teacher/student construction and export are [pipeline_v3.py][pipeline]; training/inference internals are [ProgressiveTeacherTrainer][teacher-trainer].

### Data: temporal outer context, spatial inner channels

`data_v3.py` discovers native `[X,Y,Z,T]` full-cine NIfTI and rejects reference-like paths before loading image data. ACDC reads only ED/ES fields from `Info.cfg`; that phase metadata is an acknowledged input, not a manual segmentation mask. M&Ms requires unambiguous 4D images and an explicit patient split. Missing requested cine patients fail instead of silently shrinking the cohort. [Discovery, geometry and split code][cine-data].

`load_cine` transposes to `[T,Z,Y,X]`, clips the **entire cine** at 1/99 percentiles, then applies a cine-wide z-score. It preserves native acquisition axes and native in-plane size: **the current v3 loader does not implement a fixed 224 or 256 canvas**. The old B1 224 proposal and the supervised 256 recipe are separate experiment identities. Each sample has:

- `prev = [z-1,z,z+1]` at `max(t-1,0)`;
- `cur = [z-1,z,z+1]` at `t`;
- `nxt = [z-1,z,z+1]` at `min(t+1,T-1)`;
- each tensor is `[3,H,W]`, with replicated spatial boundaries; time is clamped, not cyclically wrapped.

Thus `prev/cur/nxt` are **three temporal frames**, each containing **three spatial slices**. `MotionBranch` uses only each frame's center channel; appearance uses the current three-slice input. `PatientBatchSampler` keeps each physical batch within one patient/native shape. [data_v3.py:85–160][cine-data]; [MotionBranch:34–63][teacher-model]. Feeding one canonical spatial triplet as if its channels were temporal neighbors would still be invalid, although main now has the correct dedicated cine loader.

### Teacher and semantic identity

The teacher is a small randomly initialized appearance encoder plus pairwise registration branch. Appearance returns quarter-scale features and an ordinary reconstruction. Registration predicts previous/next displacement fields, warps center-slice images, and creates a motion descriptor. Fused features receive a soft assignment to 12 anonymous prototypes; pooled feature, intensity, motion and area statistics feed a four-class semantic MLP. [CinePseudoTeacher and component definitions][teacher-model]; [resolved default config][v3-config].

The named class anchor comes from **handwritten image-only priors**, not from anonymous prototypes alone:

- border contact and relatively low flow magnitude add background evidence;
- boundaries and ring enclosure add MYO evidence;
- enclosed regions add LV evidence;
- ring adjacency and a usable affine-derived patient-left direction distinguish RV/LV proposals.

The implementation operates on hard prototype regions at the low-resolution feature grid and uses CPU NumPy/SciPy morphology. `trainer_v3.build_evidence` deliberately supplies displacement fields, not arbitrary learned feature magnitude, to the motion prior. Scores are heuristic evidence, **not calibrated correctness probabilities**. No manual masks, scribbles, pretrained segmentation teacher or external semantic model is loaded by this route. [build_region_evidence][evidence]; [trainer_v3.py:27–55][teacher-trainer].

Prototype history can corroborate a currently supported class but cannot create or rename an unsupported seed. Raw-evidence thresholds are probability >= 0.70, top-two margin >= 0.20 and at least four hard-assigned region pixels; bank updates additionally require >= 0.80 confidence. Dense decoding requires supported region mass >= 0.5, final probability/margin checks and agreement between neural-plus-evidence prediction and evidence. With no external evidence, all labels are exactly `UNKNOWN=255` even if the neural head is spuriously confident. [trainer acceptance][teacher-trainer], [PrototypeBank][prototype-bank], [decode_evidence:105–145][teacher-model].

### Teacher and student losses

Default teacher objective:

```text
L_teacher = reconstruction
          + 0.1 * anonymous_prototype_information
          + semantic_seed_CE
          + motion_photometric
          + 0.05 * motion_smoothness
```

Reconstruction is Smooth-L1 on the current center image, **not masked reconstruction**. The prototype term is mean assignment entropy minus marginal entropy. Registration compares warped previous/next images with current and penalizes spatial flow differences. Seed CE trains raw semantic probabilities against detached raw heuristic class targets on accepted regions; it does not add its target evidence to the logits being scored. [losses_v3.py:6–40][v3-loss]; [config][v3-config]. These objectives can learn appearance/motion without establishing semantic correctness.

The student uses a separate width-32 encoder with two stride-2 reductions, a 1x1 A0 head, bilinear upsampling, and optional shared `AnnotationExpert`. There is no learned skip decoder in `DeploymentEncoder`. Pseudo-label loss masks UNKNOWN **before** CE, checks accepted classes are 0..3 and returns a differentiable zero for entirely invalid data. The CLI skips invalid batches before AdamW so weight decay/momentum cannot create fake updates. [system_v3.py:154–195][teacher-model]; [pipeline_v3.py:216–258][pipeline]. Thin-boundary quality is a hypothesis to test, not an observed failure in this audit.

### Export, freeze, evaluation and deployment

Export requires all T*Z slices for every declared exported patient. Temporal consistency warps neighbor probabilities and validity with predicted flow, then only removes validity when agreement is insufficient. It does **not average probabilities, promote invalid pixels, diffuse labels or wrap the final frame to the first**. Cross-slice voting is explicitly disabled because correspondence is unimplemented. [pipeline `_flush`][pipeline]; [consistency_gate][consistency].

Schema-3 freeze records per-file content hashes, sample identity, source/config/split metadata, image hashes and available teacher/checkpoint artifacts. Consumers validate artifact bytes before reference access or student training; this protects accidental corruption, not malicious re-signing. Student also rechecks source image hashes and re-verifies the freeze after training. [freeze.py][freeze]; [FrozenPseudoDataset][pipeline].

`AdaptiveAnnotationStudent.forward` is the deployment model API. Compact/balanced/accurate execute **0/1/2 outer refinement turns**, which mean **0/1/3 internal Dynamic Window passes**, because turn 0 uses depth 1 and turn 1 uses depth 2. The encoder runs once; compact returns A0 unchanged. It uses feature-only conditioning, without the supervised learned auditor, counterfactual gate or Candidate-C solver. [student definitions][teacher-model], [AnnotationExpert:351–452][expert], [pass-count regression][pass-test].

`AdaptiveRuntime` additionally chooses one profile from average A0 entropy with a caller-supplied cap and thresholds 0.25/0.55, reusing encoded features. It is not an actual VRAM/latency autotuner, and its one scalar entropy chooses one profile for the input batch. Training CLI uses a fixed profile; it does not automatically train every deployment profile. There is no dedicated trained-student native-volume evaluator or inference CLI among the inspected `src/` and `scripts/` consumers of `AdaptiveAnnotationStudent`/`AdaptiveRuntime`. The frozen evaluator scores **teacher pseudo-labels**, not student predictions. [adaptive.py][adaptive]; [student checkpoint schema][pipeline]; [teacher evaluator][pseudo-eval].

## 3. Mask-supervised canonical Self-Audit, kept distinct

The README's canonical command is `python scripts/train_self_audit.py --config configs/self_audit_full.yaml`. It enforces the official environment when run as a script, loads strict unified configuration, builds `UnifiedTrainer`, optionally resumes and calls `train`. [CLI main][supervised-cli].

```text
train_self_audit.main -> load_unified_config -> UnifiedTrainer
  -> _utils.build_model_from_config -> SelfAuditNet
     -> ConvNeXt-Tiny features -> LightweightFPN (96 channels)
     -> initial head A0 -> recurrent AnnotationExpert
     -> CounterfactualAuditor (local FIX/UNCHANGED/REGRESS + delta_q)
  -> build_patient_dataset -> ACDC/M&Ms VolumeSliceDataset
     -> image AND mask -> spatial [z-1,z,z+1], center-slice mask
  -> interval objective -> backward/optimizer -> validation -> bound selected-best
  -> post-training calibration + diagnostics
```

The default YAML uses a pretrained ConvNeXt-Tiny encoder, four classes, 8-point windows, three outer turns, `window_mode: current`, 256 in-plane size and paired 3D data. It is not a Mamba backbone despite the README title. Candidate C is an explicit alternate mode; its presence in configuration does not activate it. [configuration][supervised-config]; [SelfAuditNet initialization][supervised-model].

`VolumeSliceDataset` explicitly opens `record.mask_path` in initialization/loading, returns `batch['mask']`, clips each 3D volume at 0.5/99.5 percentiles and z-scores it, constructs spatial z-triplets and resizes images/masks in plane. This is **mask supervision**, unlike the v3 full-cine loader. [common.py:188–244,475–605][supervised-data].

| Interval in default YAML | Runtime objective | Supervision |
|---|---|---|
| epochs `[0,100)` | `forward_annotation` -> `phase_a_loss`, weighted A0..A3 Dice+CE, weights 0.5/0.7/0.8/1.0 normalized by sum | Manual center masks |
| `[100,120)` | `_auditor_batch`, frozen annotator, adjacent/synthetic transitions, local CE + signed ranking/regression | GT correctness transitions and GT delta-Dice |
| `[120,130)` | `compute_joint_losses`, retained final annotation under threshold gate, all modules trainable | Manual masks and attempted-transition targets |

The unified dispatcher reads `batch['mask']`; local audit targets compare previous/candidate labels to ground truth, and global targets are their foreground Dice difference. This invalidates any claim that the supervised auditor is intrinsically no-GT merely because inference itself omits GT. [UnifiedTrainer.compute_batch_loss][unified-loss], [phase_a_loss][phase-a], [annotation loss][annotation-loss], [transition targets][targets], [audit loss][audit-loss].

Inference computes encoder/FPN once, proposes a recurrent candidate only for active rows, audits detached previous/candidate probabilities, accepts if `delta_q > tau`, retains the old state on rejection and stops that row. Alternate `initial_only` and `always_accept_refinement` are comparisons; `oracle_accept` is explicitly GT-based analysis. `infer_patient_volume` builds spatial batches, stacks slice predictions and rejects `oracle_accept`. [SelfAuditNet.infer][supervised-model]; [infer_patient_volume][volume-infer]. GT-free inference does not erase GT-based training.

The selected-best mechanism is still active: only a gated interval at/after the configured minimum epoch and complete validation can select its configured Dice; immutable snapshots and committed references back the public best checkpoint. Bounded/incomplete runs skip completed-pipeline calibration. [UnifiedTrainer selection][selection]. These are good supervised provenance controls but are **not a valid no-GT checkpoint-selection policy** because they select with validation GT.

## 4. Separate active Maskfree150 flow

`train_maskfree.py` constructs `MaskfreeTrainer` and runs preflight or `run()`, with explicit completed/partial/failure outcomes. It remains an official environment consumer in the milestone. [Maskfree CLI][maskfree-cli]; [scope][scope].

Its `_train_batch` uses fitting context/support from `ImageOnlyDataset`, trains a randomly initialized representation producer on image-only losses, extracts detached features, generates hypothesis banks, audits them using fitting/selection observations, then trains **two separately stored students** from initial versus audited pseudo-probabilities and validity. Targets do not backpropagate into the producer. Producer/hypothesis execution and audit placement have separate costs; the inspected implementation transfers a feature batch to CPU for deterministic candidate generation. [MaskfreeTrainer:1068–1261][maskfree-trainer].

Producer and students are small dense CNNs; this is not `CinePseudoTeacher` and not the ConvNeXt/FPN canonical network. Producer losses include masked reconstruction/representation objectives; student loss is validity-weighted **soft** CE, different from v3's hard-label CE. The configuration rejects `teacher_checkpoint`, pretrained weights, reference/mask roots, Dice selection and threshold tuning fields. [models.py][maskfree-model], [losses.py][maskfree-loss], [config.py:70–85][maskfree-config]. Preserve this route as a separate scientific identity. Its completion or preflight receipt cannot substitute for v3 semantic quality or the requested 4080 Super deployment measurement.

## 5. What “diffusion” means here

**The identified diffusion is graph/manifold feature processing and clustering in CUTS. It is not a DDPM/score-based generative model and does not denoise Gaussian noise into segmentation masks. It also does not propagate manual class labels.** The inspected v3 and canonical Self-Audit runtime paths have no diffusion call; a literal all-occurrence search for `diffusion` in main `src/`, `scripts/`, `configs/`, `tests/`, `docs.md` and `docs/` returned none. Relevant implementations are under `baseline/CUTS`.

| Route | Verified calls | Why it exists / output |
|---|---|---|
| Current shared scientific CUTS runner | `run_cuts_scientific.run.generate` -> latent encoder -> `cluster_latent` -> `phate_kmeans` -> `phate.PHATE(..., t=2).fit_transform` -> `phate.cluster.kmeans(..., n_clusters=10)` | Turn learned dense features into anonymous regions under a fixed PHATE/K=10 contract; optional frozen semantic adapter afterwards |
| Separate CUTS diffusion-condensation route | `generate_diffusion` -> `diffusion_condensation_msphate` -> normalized latent vectors -> `Multiscale_PHATE.fit` -> `NxTs` labels at multiple hierarchy levels | Obtain multiscale anonymous partitions instead of fixing a single cluster count |
| Persistent partition export | `persistent_partition` -> `_upstream_persistent_structures_safe` -> dense anonymous IDs | Select persistent regions with a fixed GT-free rule, preserving int64 IDs and artifact provenance |
| v3 temporal consistency | estimated flow + validity-aware class agreement | Rejection only; not diffusion |

The scientific runner's exact PHATE parameters are components=3, knn=100, landmarks=500, t=2 and K=10, with fixed clustering seed/retry receipts. **Do not describe that runner as executing multiscale persistent condensation**: it imports `cluster_kmeans`, not `generate_diffusion`. [run_cuts_scientific.py:149–248][cuts-runner]; [cluster_kmeans.py:15–64][cuts-cluster]. The detailed PHATE graph algorithm executes in the external dependency; this source audit verifies the called API/configuration, not a reimplementation of its internals.

The multiscale route passes `[H*W,C]` feature vectors to Multiscale-PHATE (or the explicit CATCH alternative), obtains cluster labels over scales and removes the finest all-distinct level. The persistent exporter applies an area/persistence rule and outputs anonymous partitions, never named RV/MYO/LV semantics. [generate_diffusion][diffusion], [diffusion_condensation_msphate][condensation], [persistent exporter][persistent]. It exists because a representation encoder needs a region/partition extraction procedure; it is not required to deploy the v3 student. Its cost must be measured as offline generation if that comparator is used, not silently included/excluded when comparing deployment speed.

The original-style ACDC CUTS route is explicitly **label-backed**, includes slice-level random splitting and whole-dataset export, and must not be called a patient-held-out image-only baseline. Its separate image-only config avoids GT reads but still has a different split/export contract. Upstream `diffusion-best` selects a hierarchy level with GT and even persistent multi-class scoring uses GT-guided relabeling: those are oracle diagnostics, ineligible as no-GT runtime output. Retain the image-only/frozen shared benchmark path and provenance boundaries. [ACDC_ORIGINAL_STYLE_PROTOCOL.md:1–73][cuts-protocol].

## 6. Metric provenance and concrete failure gates

### What current metrics actually score

1. **v3 frozen pseudo-label evaluator:** verifies freeze first, then reads independent references. For each patient it sums TP/predicted/GT class counts across all supplied reference frames and z slices, computes each of RV/MYO/LV Dice, averages present classes, then averages patient foreground means. ACDC default references are declared ED and ES; M&Ms requires explicit phase and label mapping. It is **patient-balanced, phase-pooled native-grid teacher pseudo-label Dice**, not a slice mean, not separately averaged ED/ES, not student Dice and not full-cine accuracy without full-cine reference labels. UNKNOWN is absent from predicted classes but still leaves GT anatomy in the denominator; both-empty classes are excluded. `known_fraction` is all known pixels, including background, so it cannot certify foreground coverage or seed precision. [evaluate_pseudolabel_frozen.py:16–83][pseudo-eval].
2. **Supervised training selection:** the primary slice proxy excludes both-empty foreground classes and is distinct from native volume Dice. Its legacy audit *target* still awards 1.0 to both-empty classes; this deliberate target contract must not be confused with reporting semantics. [slice_proxy_dice][metrics], [targets:88–109][targets], [selection][selection].
3. **Native supervised evaluator:** requires true native GT, not an inverse-resized preprocessed mask, and suppresses unsupported physical surface-metric claims. [evaluate_volume_native][native-metrics].
4. **Historical scores and tests:** the main runbook warns that old C0/C1 freezes belong to a previous recipe and that synthetic NIfTI tests do not measure ACDC/M&Ms quality. This worker did not bind any real-data score to a current teacher/student checkpoint and untouched cohort. **Current no-GT Dice >=0.90, external robustness, teacher precision/coverage and 4080 Super latency therefore remain UNKNOWN.** [runbook:3–5,65–67,90–106,122–129][runbook].

### Existing gates versus missing gates

| Gate | Existing implementation | Required action / fail condition |
|---|---|---|
| No manual supervision | v3 image discovery blocks mask/scribble names; no supervised checkpoint load in its producer; Maskfree config rejects supervised fields | Keep GT mounts/paths available only to independent scoring; reject any supervised-teacher, manual scribble or GT-derived mapping fallback |
| Temporal/spatial identity | Native XYZT and separate temporal frames with spatial triplets; cross-slice voting disabled | Fail on ED/ES-only input substituted for full cine, ambiguous patients, mismatched affine or unregistered slice voting |
| Integrity and cohort isolation | Disjoint patients; all declared export slices; SHA-bound artifact bytes and image identity | Any missing case, changed artifact/input or split overlap is a hard failure, never silently dropped |
| Semantic seed availability | `NO_FOREGROUND_SEEDS` only checks total accepted foreground >0; valid labels checked 0..3; `NO_VALID_UPDATES` rejects empty training | **Missing:** per-class RV/MYO/LV support floor, per-patient support diagnostics and independent semantic quality certification; one wrong foreground class can pass current gate |
| Teacher readiness | Generator writes `NOT_EVALUATED`; evaluator writes `NOT_AUTOMATICALLY_CERTIFIED` | Full pipeline logs scores and runs requested student with no quality threshold. Do not interpret `complete` as `TEACHER_READY`; absence of an external reviewed quality receipt must block accuracy/deployment claims |
| Strict no-GT selection | Training uses image-only losses and configured schedules | Do not adjust seeds, evidence thresholds, class naming, hyperparameters or checkpoints using reference Dice; any development-feedback use must be disclosed, and final untouched evaluation retained |
| Final held-out cohort | Evaluator accepts `--split test`, but standard teacher CLI exports only train/val | **Gap:** ordinary pipeline cannot produce a test freeze as configured; evaluator labels every non-train split `held_out_development`. Final-test export/evaluation identity needs an explicit validated extension, not a relabeled validation report |
| Deployment-quality claim | Student checkpoint stores manifest and `profile_trained`; no dedicated current student scoring CLI identified | Require independently frozen student predictions, profile/checkpoint identity, same patient cohort and native metric contract before treating teacher scores as deployment scores |
| Dice >=0.90 | No verified current receipt here | Predeclare patient-level native foreground macro Dice >=0.90; report RV/MYO/LV separately, failures/UNKNOWN, phase aggregation, uncertainty and untouched cohort. This is a proposed acceptance target, not a measurement |
| Fast/lightweight on requested hardware | Profile APIs exist; no measured target-hardware result in this audit | Measure **RTX 4080 Super 16GB**, fixed precision/batch/native shapes, synchronized warmed p50/p95 slice and volume latency, peak memory, parameters and full I/O accounting for the same scored checkpoint; CPU/random-weight timings cannot satisfy this gate |
| Refinement value | Three compute profiles; entropy controller | Keep Dynamic Window only if it improves the quality/cost tradeoff over compact A0 and a matched small CNN refiner at the same data/checkpoint protocol; profile names are not evidence |

Teacher readiness/code claims above are directly supported by [pipeline_v3.py:175–184,216–258][pipeline], [full pipeline:100–109][pipeline-cli] and [evaluator:77–83][pseudo-eval]. The additional scientific/hardware gates are **PROPOSED**, not implemented by this report. A strict no-GT experiment may be rejected after independent evaluation; it must not quietly turn evaluation labels into a training or selection channel.

## 7. Required retention and regression surface

This is a retention map, **not deletion authorization**. No code was deleted. Main's active milestone explicitly preserves historical benchmark contracts. Supervised task ineligibility does not imply a module is dead code.

| Keep | Verified reason |
|---|---|
| All current `src/self_audit_pseudolabel/*.py`, `configs/pseudolabel_v3.json`, the four v3 CLI scripts, split manifests, current v3 runbooks | Complete teacher/data/loss/freeze/student/evaluation dependency chain described above |
| `src/self_audit/models/annotation_expert.py`, `dynamic_window.py`, and package import dependencies | v3 student imports `AnnotationExpert`; it reuses canonical Dynamic Window. Deleting `src/self_audit` breaks v3 |
| `src/self_audit/__init__.py`, `models/__init__.py`, `self_audit_net.py`, encoder/FPN/head/auditor dependencies | Eager package exports import the full model namespace even when a caller requests only the expert; extraction would require a separately tested refactor |
| Canonical data/loss/training/evaluation/provenance/config/checkpoint files and current supervised scripts, until explicitly retired as a separate feature | They implement a live, documented supervised reference and regression contracts |
| `scripts/train_self_audit_legacy.py`, `training/train_annotation.py`, `train_auditor.py`, `finetune_joint.py` while canonical consumers remain | Canonical CLI imports legacy calibration helpers; UnifiedTrainer imports phase loss/validation/joint helpers. These files are live dependencies despite historical names |
| Entire `src/self_audit_maskfree`, its config/entrypoints/evaluation contracts | Separate active no-GT pipeline; deleting it because it predates v3 would remove an active milestone consumer |
| `src/shared_benchmark`, active environment contract, CUTS, DSS-US/SGSCN protocol locks and frozen evidence | Active comparison and no-GT evaluation controls; keep historical dependencies needed by shared regression tests |
| Old reports/frozen receipts | Evidence/provenance, not necessarily executable code; preserve recipe/SHA labels rather than silently upgrading old results |

Eager import evidence: [self_audit/__init__.py][core-init], [models/__init__.py][models-init]. Legacy dependency evidence: [train_self_audit.py:27–45][legacy-imports], [unified_trainer.py:99–120][phase-imports].

Minimum regression surfaces to retain and run after any later cleanup, on the **combined final source tree**:

- `tests/test_pseudolabel_v3_regressions.py`: UNKNOWN loss/gradients, no-evidence abstention, noncircular seed targets, once-only encoding, native axes, reference-read firewall, patient splits, shape-safe sampling, frozen tampering, temporal alignment/no wrap, background-only student stop and 0/1/3 attention passes.
- `tests/test_pseudolabel_v3_complete.py`, `test_pseudolabel_v3_integration.py`, `test_pseudolabel_full_pipeline.py`, `test_pseudolabel_system_v3.py`, `test_pseudolabel_api_compat.py`, `test_self_audit_pseudolabel_v3.py`: model/prior/profile/API/CLI coverage. Do not repeat the old CI omission of `test_pseudolabel_*.py`.
- Shared model tests: `test_self_audit_core.py`, `test_self_audit_audit.py`, `test_self_audit_hardening.py`, `test_self_audit_regressions.py`, entropy and Candidate-C tests if their namespace remains; package import smoke must construct the actual v3 student.
- Canonical training retained: unified config/trainer/lifecycle, checkpoint binding/commit/resume, calibration lineage, native metrics/volume, external M&Ms and split/geometry tests.
- Maskfree retained: `test_maskfree_*` contract, firewall, training/runtime, export and independent evaluation coverage; do not replace CUDA proof with CPU/injected passes.
- CUTS retained: `baseline/CUTS/tests/test_original_style_contract.py`, `baseline/CUTS/tests`, `tests/shared_benchmark`, `tests/native_baselines` and active native baseline suites; keep environment contract tests as the milestone requires.

The v3 test functions and assertions were inspected at the pinned SHA; test execution is **NOT RUN by this worker**, because this task owns only the report and prohibits training. The coordinator separately reported 47 v3 tests passing; that is coordinator-owned evidence and should be backed by its command/environment receipt in the combined report, not counted here as independent verification. No source patch, model run, CUDA test or accuracy measurement is implied by this source audit.

## 8. Inspection receipt and boundaries

- Verified Git SHA/checkout mismatch, clean initial audit-worktree status and absence of `.codegraph/`.
- Used Semble first for model/trainer/diffusion/metric locations. A remote Semble clone timed out after 60 seconds; discovery continued against the local exact-main checkout and all decisive source was read from pinned Git blobs.
- Read current source and AST symbol locations rather than treating old audit reports or README claims as the implementation.
- Used literal all-occurrence Git searches only for dependency/diffusion/consumer accounting after discovery.
- Report validation: all 45 immutable source-reference paths and cited line bounds were checked against the pinned Git blobs; all reference labels resolve and the report has no trailing whitespace.
- This file is the sole owned modification. No data, references, model checkpoints or source files were modified; no training, tests or benchmark was launched by this worker.
- Historical memory supplied a lead about the older v3 temporal contract; current-main code independently verified and superseded its scaffold-only conclusion. It is not evidence of current performance.

## Immutable source references

[scope]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/reports/ACTIVE_MILESTONE_SCOPE.md
[runbook]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/docs/pseudolabel_v3_review_runbook.md
[pipeline-cli]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/scripts/run_full_pipeline_v3.py#L64-L109
[pipeline]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/pipeline_v3.py#L78-L258
[cine-data]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/data_v3.py#L26-L160
[teacher-model]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/system_v3.py#L26-L195
[teacher-trainer]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/trainer_v3.py#L10-L65
[evidence]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/evidence.py#L39-L88
[prototype-bank]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/evolution.py#L8-L34
[v3-config]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/configs/pseudolabel_v3.json#L1-L9
[v3-loss]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/losses_v3.py#L6-L40
[consistency]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/consistency.py#L6-L35
[freeze]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/freeze.py#L23-L76
[adaptive]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_pseudolabel/adaptive.py#L7-L33
[expert]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/models/annotation_expert.py#L295-L462
[pass-test]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/tests/test_pseudolabel_v3_regressions.py#L219-L231
[pseudo-eval]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/scripts/evaluate_pseudolabel_frozen.py#L9-L90
[supervised-cli]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/scripts/train_self_audit.py#L187-L271
[supervised-config]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/configs/self_audit_full.yaml#L12-L169
[supervised-model]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/models/self_audit_net.py#L855-L1196
[supervised-data]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/data/common.py#L188-L605
[unified-loss]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/training/unified_trainer.py#L1166-L1308
[phase-a]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/training/train_annotation.py#L97-L118
[annotation-loss]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/losses/annotation.py#L10-L42
[targets]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/audit/targets.py#L70-L122
[audit-loss]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/losses/audit.py#L12-L100
[volume-infer]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/evaluation/volume_inference.py#L496-L571
[selection]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/training/unified_trainer.py#L2767-L3072
[metrics]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/evaluation/metrics.py#L122-L265
[native-metrics]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/evaluation/volume_inference.py#L731-L887
[maskfree-cli]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/scripts/train_maskfree.py#L130-L197
[maskfree-trainer]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_maskfree/trainer.py#L1068-L1261
[maskfree-model]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_maskfree/models.py#L1-L45
[maskfree-loss]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_maskfree/losses.py#L230-L490
[maskfree-config]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit_maskfree/config.py#L70-L85
[cuts-runner]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/scripts/run_cuts_scientific.py#L149-L248
[cuts-cluster]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/baseline/CUTS/src/cardiac_benchmark/cluster_kmeans.py#L15-L64
[diffusion]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/baseline/CUTS/src/scripts_analysis/generate_diffusion.py#L28-L46
[condensation]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/baseline/CUTS/src/utils/diffusion_condensation.py#L12-L67
[persistent]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/baseline/CUTS/src/scripts_analysis/export_diffusion_persistent.py#L39-L139
[cuts-protocol]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/baseline/CUTS/ACDC_ORIGINAL_STYLE_PROTOCOL.md#L1-L79
[core-init]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/__init__.py#L1-L5
[models-init]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/models/__init__.py#L1-L43
[legacy-imports]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/scripts/train_self_audit.py#L27-L45
[phase-imports]: https://github.com/QuocKhanhLuong/Self-Audit/blob/c31825a7f90df47f9f9382a9c9595e43f7f71236/src/self_audit/training/unified_trainer.py#L99-L120
