# Candidate FUSE — native supervised/cine teacher–student fusion

2026-11-13. This is the implementation contract for `candidate_fuse`.
The original full-label run uses **all TRAIN80 labels; not a few-label result**.
The semi-supervised and transfer extensions are locked in the amendment below.
val means the historical development/DEV18 split; internal role remains `development`.
Implementation verified with focused tests, CPU smoke and one independent Claude review (SOFTWARE_ONLY). Patient/GPU evidence is recorded separately in the launch report.

## 1. Scope and latest user amendments

New modules/configs belong to candidate_fuse. Candidate A, A3D and Q3D source
trees and Q3D configs are frozen and may only be imported. Existing datasets,
splits, voxel arrays, geometry receipts and run history are preserved.
No tokenizer, codebook, physiology or patient weighting belongs to FUSE.

User amendment: prioritize implementation and real launch, use parallel Luna
Fast workers and exactly one independent Claude reviewer after integration.
Run focused formula/data/gradient/phase tests, check_repo and one complete CPU
synthetic smoke; the historical full suite is NOT_RUN unless needed for a
specific regression. Train by epochs and provide a live server log window.
CPU checks and review precede server migration; only the coordinator changes
server services. Existing Q3D stays running during development. FUSE gets a new
checkout and new run directory; nothing is deleted from the old run.

## 2. Data and epochs

Seed 42. Keep the exact TRAIN80 / VAL18 / TEST50 patient IDs. The three patient
sets must be mutually disjoint. L and U deliberately use the SAME 80 TRAIN
patients but disjoint frame keys: L is the two original ED/ES masks per patient
(160 frames), U is every other cine frame, image only. ED/ES exclusion uses
the annotated frame indices; do not read non-ED/ES targets. No VAL/TEST target
enters training. TEST is not consulted for model/budget/deploy selection.

Use existing normalized native image arrays [0,1] and original mask arrays;
new metadata manifests attach the already repaired `acdc_pixdim_native_v1`
geometry, without crop/resampling or edits to old manifests. Missing any TRAIN
ED/ES mask is an error, not permission to synthesize targets. Manifest builder
may reconstruct missing TRAIN mask arrays directly from original NIfTI labels,
with exact axis transpose and original-to-array hashes recorded; no interpolation.
Report source SHA, generated SHA, class mapping BG/RV/MYO/LV and split identity.

T epoch = a seeded permutation of all 160 L frame keys, exactly once each.
T is 50 epochs = 8,000 labeled-frame exposures (8,000 optimizer steps at default batch1; fewer steps with the explicit batch2 amendment below). T-a is first 400 exposures (5%),
including half of epoch 3; T schedule boundaries remain exact in frame-exposure units.
J epoch = a seeded permutation of the 80 TRAIN patients, one randomly selected
non-ED/ES frame each. Each J update pairs one L frame and one U frame, one
optimizer step. L uses a separate shuffled 160-frame stream spanning two J
epochs. All RNGs/permutation cursors are checkpointed. Default J is 100 epochs
(8,000 pairs); coordinator records final epoch budget from GPU profile before
launch and never changes it after viewing VAL/TEST scores.

## 3. Teacher

Reuse Q3DResEncTeacher + strict nnFoundation loader. Real encoder widths are
[32,64,128,256,320,320], strides
[(1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2),(1,2,2)].
Weights revision f443c09319988e833b3d9589bf95909e4443d7b2; SHA256
ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd.
Preserve pinned nnssl/DNA versions, provenance and existing zero phase correction
receipt 36f44d7f85cd3872168b01339353fb17787099d309e167ea5d64ffbc1721b72b.
Strict encoder preload excludes public decoder/head keys. Decoder/heads are
new for TRAIN80. Subsequent FUSE trained checkpoints strict-load all heads and
encoder, including validation of shared encoder aliases. No old few-label
fine-tuned teacher is reused as the new T result.

Teacher-only whole-volume z-score and high-side padding are unchanged.
T loss reuses q3d_teacher_loss (region/boundary CE+Dice and near/far SDF), on L
only. T optimizer AdamW wd=1e-4. T-a freezes encoder, heads LR1e-3. T-b
encoder LR1e-4, decoder/heads LR1e-3 with existing Q3D warmup/cosine semantics;
deduplicate shared parameter identities. Select T by best VAL18 patient-macro
foreground Dice, evaluated at every epoch (ties retain earliest). Freeze all
teacher parameters, eval and no_grad throughout J and evaluation.

Wrapper returns stage0..4 encoder features from ONE actual encoder pass.
It can capture encoder output with a scoped hook without modifying Q3D source.
Teacher feature phases are derived from live main-path conv modules, preserving
the existing ResEnc residual/ERF phase convention. F0..S4 teacher widths are
[32,64,128,256,320]; steps [(1,1,1),(1,2,2),(1,4,4),(2,8,8),(2,16,16)],
offsets zero. These constants are checked against the live modules, not guessed.

## 4. Student and fusion

Reuse Q3D low-level UNETR++-lite conv/EPA blocks in a new code-free network.
Widths F0,S1,S2,S3,S4 = [32,64,128,256,512]. Native z stride stays 1.
F0 stem maps input to C0, then fusion then residual blocks. Each S level:
down-conv, fusion, then that level's conv/EPA blocks. S1/S2 use residual conv;
S3/S4 each use two EPA blocks and uniform masked supports. Decoder follows
the existing dense upsample/reduce + additive skip + residual-conv path and
four-class region head. No A_S/C0/W_q/g or SDF student head.

At each level l, P_l is Conv3d1x1 teacher channels to student channels;
G_l = sigmoid(Conv3d1x1([F_S,l, P_l(F_T,l)])). Gate weights initialize zero,
bias=-2, hence G initially sigmoid(-2)=0.1192029 at every cell.
F_l = F_S,l + m * G_l * P_l(F_T,l), before the level's blocks.
Use the existing phase sampler `align_teacher_to_student4` with the actual
target phase for EACH level (the helper name does not restrict target stride).
F0–S2 coincide on the native lattice; S3/S4 interpolate along z. High-side pad
can differ in extent: first sample the native target lattice, then extend only
the internal padded student grid as needed, preserving native values/phase.
No crop or resample of source volumes. Ramp/probe compares native cell centers
and handles border endpoint behavior explicitly.

m is one binary keep decision per sample, shared across all five levels.
p_drop(t)=0.5*t/max(J_updates-1,1), t=completed updates before current update.
Train draws m from Bernoulli(1-p_drop) using a saved generator. Forced student
mode uses m=0 and does not need any teacher; forced fusion uses m=1. m=0 must
be bitwise identical to the pure student route. Do not calculate unused fusion
contributions on the pure path. Teacher tensors are always detached.

Training-only H_l at S2/S3/S4 maps student features to teacher channels using
1x1 conv. For each frame with m=0, d = mean over three levels of the native
valid-cell mean of 1-cos(H_l(F_S,l), aligned F_T,l). For m=1, d=graph-preserving
zero. Targets detach. A batch averages zeros and eligible samples over B.
Mimic may use pre-fusion features; when m=0 they are exactly the unfused student
features. Padding cells are excluded. Export removes H_l.

## 5. J loss, selection, evaluation and deploy

L_J = CE_Dice(logits_L, y_L) + lambda_u(t)*SoftCE(logits_U, sg(softmax(T_U)))
      + 0.1*(d_L+d_U).
CE_Dice is the existing sum CE + foreground soft Dice; no SDF on student.
SoftCE=-mean_native_voxels sum_classes p_T*log_softmax(logits_U), temperature1,
no hard pseudo-mask, confidence threshold or class/patient reweighting.
lambda_u(t)=min(1,t/max(1,ceil(0.1*J_updates))). One pair gives one optimizer
step; do not divide the supervised/unsupervised terms by two.
Student AdamW lr3e-4 wd1e-4, 5% update warmup then cosine, grad clip1, bf16
autocast on GPU, fp32 parameters. Synthetic CPU uses fp32 and reduced widths.
Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization. CPU verification uses
strict deterministic algorithms. CUDA training uses deterministic algorithms
with warnings for unsupported trilinear/adaptive-pool backward kernels; retain
the warnings and report GPU backward determinism as best effort. Profiling uses
the same setting; this does not change the network, objective or data.
Teacher/student receive the SAME spatially transformed image for fusion. The
main recipe uses shared Q3D spatial flips/spacing-safe 90-degree rotations and
shared optional intensity augmentation; no independent spatial views.
L and U may come from different patients and have different native shapes and
spacing. Transform each frame independently, sharing that transformed frame
between its teacher and student. Draw one independent keep mask per frame;
never require L and U to have matching shapes or geometry.

Evaluate after every epoch and at final update. Log teacher, pure student and
fusion FG/RV/MYO/LV Dice and mean G for all five levels; preserve RNG and modes.
Choose ONE J snapshot by fusion VAL FG Dice (ties earliest), then compare its
student and fusion outputs. Teacher uses the selected T snapshot throughout.
Metrics are native-volume patient-macro ED/ES, using the existing Dice/HD95 mm
primitives. Mean three foreground class Dice defines FG. Record class/sample
absence handling consistently across all modes. No test-based decisions.

Locked rule: if student_VAL_FG >= fusion_VAL_FG - 0.01, deploy student;
otherwise deploy fusion. Export student has no teacher/fusion projectors/gates/
H heads; fusion export includes frozen teacher + P/G + student, no H heads.
Cold-load in a fresh interpreter with same dtype/threads/determinism must be
bitwise identical on three synthetic volumes, and checked on VAL volumes when
real artifacts exist. Record artifact/config/data SHA, mode, parameters and
measured latency. Evaluate TEST50 ONCE as one locked evaluation transaction
covering the three modes; save per-patient results and a ledger. No rerun after
seeing scores, no TEST-driven choice. Missing TEST is reported explicitly.

## 6. Locked implementation interfaces

Geometry is canonical dict with `shape` (D,H,W), `spacing_zyx` in mm,
`affine_zyx` 4x4, `context="full_volume"` — exactly these four runtime keys.
Retain provenance fields in the manifest/source records, outside runtime
geometry, because reused canonical_geometry deliberately rejects extra keys.
Images float32 [1,1,D,H,W], targets long [1,D,H,W], labels0..3.
GridPhase = ((jz,jy,jx),(oz,oy,ox)), as Q3D alignment.

### data.py (W1)
`Frame` dataclass: image, target (Tensor|None), geometry, patient_id:str,
frame_index:int. `FrameKey=tuple[str,int]`.
`FuseData(manifest_path, expected_sha256=None, *, role, synthetic=False)`:
properties patients(tuple), manifest_sha256, manifest(dict);
`labeled_keys()->list[FrameKey]`, `unlabeled_keys()->list[FrameKey]`,
`load(key, *, labeled:bool)->Frame`. role train/development/test checked.
Labeled reads are allowed for each role's own evaluation/training caller;
FuseSampler and training entrypoints accept only role=train. Do not globally
disable development/test label loading needed by the evaluation interface.
`FuseSampler(data, *, seed=42)`: next_labeled()->FrameKey;
next_pair()->tuple[FrameKey,FrameKey]; state_dict/load_state_dict;
teacher_epoch_size=len(Lkeys), joint_epoch_size=len(patients).
`write_synthetic_manifests(out, *, seed=42)->dict`: keys train/development/test,
each {path,sha256}. At least 2 TRAIN patients, 1 val, 1 test, >=3 cine frames.
Builder CLI `scripts/fuse_build_manifest.py`; produce new FUSE manifests from
fixed existing split and original mask inventory. All real manifests carry
per-record images/masks hashes and geometry. Worker documents exact schema.

### teacher.py (W2)
`build_teacher(config, *, synthetic=False)->tuple[FuseTeacher,dict]` delegates
Q3D build_q3d_teacher using config.model.teacher and seed.
`FuseTeacher` exposes `.model` Q3DResEncTeacher, `.feature_channels` first5,
`.feature_phases` first5, `.forward(image, geometry, *, return_features=True)`
return region_logits/boundary_logits/sdf and features tuple when requested.
`.freeze_for_joint()` permanently freezes parameters/eval for J.
`teacher_parameter_groups(teacher, config)->list[dict]` names encoder/heads,
LR/base_lr, deduplicated. `configure_teacher_step(teacher, optimizer, update,
total_updates, config)->dict` sets T-a/T-b and LR; `teacher_loss(outputs,target,
geometry)->Tensor`. Full artifact state load remains strict.

### student.py / fusion.py (W3)
`FuseStudent(config, *, teacher_channels=None, teacher_phases=None,
include_fusion=True, include_mimic=True)`; config is full canonical mapping.
`.forward(image, geometry, *, teacher_features=None, keep_mask=None)` -> dict
with logits, features(tuple5), gate_means(list5 floats), mimic_loss(Tensor).
With teacher_features absent force pure route and graph-zero mimic_loss.
With supplied features, use keep_mask [B,1,1,1,1] (required), align all levels,
return actual mean gates (zero/None for unused levels), compute mimic only in
training and for dropped samples. `.phases` tuple5 from live conv path.
`.infer_logits(image,geometry)->Tensor` pure student.
`teacher_dropout_probability(update,total_updates)->float`,
`sample_keep_mask(batch_size,update,total_updates,*,device,generator=None)`.
`GatedFusion(student_channels,teacher_channels)` and phase helper are public
within fusion.py; expose tests without depending on engine.

### config.py / registrations (W4-config)
`load_config(path)->dict`, `normalize_config(mapping)->dict`,
`config_sha256(mapping)->str`, `default_config(*,synthetic=False)->dict`.
Top keys: schema="self_audit.candidate_fuse.v1", method="candidate_fuse",
seed=42, data, model, training, evaluation, runtime.
data: train_manifest/train_manifest_sha256, val_manifest/val_manifest_sha256,
test_manifest/test_manifest_sha256. model.teacher is copied from pinned Q3D
recipe; model.student.channels fixed real widths. training fields teacher_epochs
(50), joint_epochs(100), teacher_lr_encoder(1e-4), teacher_lr_head(1e-3),
student_lr(3e-4), weight_decay(1e-4), warmup_fraction(.05), gradient_clip_norm(1),
unlabeled_ramp_fraction(.1), mimic_weight(.1), max_teacher_dropout(.5).
evaluation: every_epochs=1, deploy_margin=.01, test_once=true.
runtime: synthetic(bool), amp_dtype="bfloat16" for GPU/fp32 CPU,
deterministic=true, num_threads=1 for synthetic default.
Synthetic reduced channels [4,8,12,16,32], same network structure and contracts,
teacher_epochs=1,joint_epochs=1. Manifest fields can be null only in template/
synthetic-before-materialization; real execution requires actual pinned files.
Do not add q3d config keys to FUSE or mutate old track definitions.
train.py adds --fuse-stage {T,J,all}, --fuse-resume, --fuse-max-updates,
--fuse-teacher-artifact; evaluate.py dispatches export + modes on FUSE.

### engine.py (W4-engine)
`run_training(config,out,*,device="cpu",stage="all",resume=False,
max_updates=None,teacher_artifact=None,profile=False)->dict`.
Stage all does T then J, selects snapshots, VAL3/deploy/export. Stage J imports
the selected teacher artifact with SHA/config/data validation. Explicit
max_updates bounds profiles/smokes; real full training uses epochs.
`joint_loss(student,teacher,labeled,unlabeled,*,update,total_updates,config,
generator=None)->tuple[Tensor,dict]` uses Frame and returns exact §5 objective.
Teacher eval/no_grad, teacher frozen before this function.
`out/STATE.json`: schema,status,stage,epoch,update,updated_utc,
checkpoint_path,checkpoint_sha256. Status RUNNING/PAUSED/COMPLETED/FAILED.
`out/metrics.jsonl`: train_update or validation events with stage,epoch,update,
losses or modes (teacher/student/fusion), gate_means; print each epoch summary.
Atomic resume.pt + hash; teacher.pt selected T; joint_best.pt selected J;
checkpoint stores optimizer/RNG/sampler/data identity/config and selected state.
Resume must not overwrite existing progress or replay committed metric rows.
Real training saves resume state every 20 optimizer updates and after each
epoch's validation/selection; per-update metrics remain live. Synthetic runs
may checkpoint each update. Checkpoints record the committed metrics offset;
explicit resume archives any uncommitted suffix before restoring that prefix.
Result fields include status, stage, teacher_artifact (path), joint_checkpoint
(path), exports (student/fusion/teacher -> export receipt with path/sha256/mode),
deployment (with mode), teacher_has_no_grads_after_J (measured parameter check),
export_clean (measured selected-artifact inventory). Completed J writes all
three comparison artifacts under exports/, but only the locked selected mode
is the deployment artifact. Teacher-only export contains the selected T only.
Engine never reads TEST; campaign locks the three SHA-bound artifacts before
the one three-mode TEST evaluation transaction and records per-mode completion.
Engine writes teacher artifact even for bounded profile T (marked PROFILE,
not acceptable as production T); J profile may use it only with profile=True.

### evaluation.py / export.py (W4-eval)
`evaluate_modes(teacher,student,data,*,device="cpu",modes=("teacher","student",
"fusion"))->dict[mode,metrics]`, metrics fg_dice/per_class{RV,MYO,LV},
patient_count, optional hd95; preserves model modes and RNG.
`choose_deployment(metrics,*,margin=.01)->dict` fixed comparison.
`export_model(student,teacher,config,path,*,mode,metadata=None)->dict` saves
weights_only-compatible artifact + SHA metadata. `load_export(path,*,device="cpu")`
returns inference nn.Module with forward(image,geometry)->logits Tensor.
Modes are student/fusion/teacher, with teacher used only for the comparison
artifact. Every export receipt contains absolute path, sha256 and mode.
`verify_cold_load(path,volumes,*,device="cpu")->dict` volumes sequence of
(image,geometry), same dtype/thread/determinism; new subprocess bitwise equality.
`evaluate_export(path,manifest,manifest_sha256,*,role,device="cpu",out=None)`
entrypoint-compatible. TEST ledger enforced by campaign/evaluation, not a
training dataset access. Independent import with no teacher for student export.
`evaluate_exports(bundle,data_manifest,out,*,device="cpu",role="test",
expected_sha256=None)` consumes the engine's three export receipts as one
transaction: lock artifact hashes before loading TEST targets, record completion
per mode, and reject an already completed transaction.

### operations / smoke (W5)
scripts/fuse_campaign.py orchestrates T/J/VAL/deploy/export/one TEST transaction
using canonical train/evaluate entrypoints, writes RUN_STATE.json, logs errors.
scripts/fuse_profile.py profiles actual native L/U including largest volume,
20 T +20 J updates (profile-only teacher), writes measured peak VRAM/time/dtype
and recommends/records J epoch budget before real launch. No guessed ETA.
scripts/fuse_watchdog.py observes every420s, no duplicate writers; default
automatic retry limit0 for OOM/deterministic failures; explicit recovery only
after cause fixed, bounded total retries<=3. Healthy flow advances automatically.
scripts/fuse_switch.py provides checkpoint-boundary snapshot/receipt helper,
never executes server changes by import/test. Only coordinator invokes switch.
scripts/smoke_fuse.py materializes synthetic data, T/J, VAL3, export/coldload;
outputs SOFTWARE_ONLY receipt in explicit /tmp output, never reports/ by default.
New scripts/fuse_watch_metrics.py may show epoch loss + VAL3 + gates live.

## 7. Acceptance and server execution

Focused tests cover split/frame/SHA guards; frozen teacher gradient; strict
encoder/head load; m0 bitwise pure; initial gate; phase probes F0..S4 incl odd
shapes/borders; dropout schedule; mimic only m0; CE/Dice/softCE coefficients;
deploy threshold; resume and export coldload. Check_repo and one end-to-end CPU
smoke must pass. One Claude reviews integrated logic/math; assigned Luna fixes
findings. No worker edits another worker's files or server processes.

After checks: merge/push/verify main. At next committed checkpoint boundary,
coordinator disables old alarm, stops watchdog then queue, verifies no writers,
archives original receipts/logs/checkpoint by hash and marks unrun jobs SUPERSEDED
in a separate SWITCH_RECEIPT (never deletes history). New server checkout only.
Run base /root/mount_data/runs/fuse-YYYYMMDD, profile, lock epoch budget, launch
T→J→VAL3→deploy/export→TEST3 once. Supervisor, watchdog and a visible training
window remain active. Formula/data errors must be fixed, never hidden; unknown
results stay NOT_RUN. Report SOFTWARE_ONLY/GPU/patient evidence separately.

### [AMEND 2026-11-13] TEST transaction entrypoint

The campaign calls the canonical evaluation module `evaluate_exports` in its own process for the single locked three-mode TEST transaction. The standalone `evaluate.py` FUSE CLI remains VAL-only; TEST cannot be opened through a single-mode CLI invocation. This intentional exception preserves one ledger and one attempt for the three frozen exports.

### [AMEND 2026-11-13] User-requested larger physical teacher batch

`training.teacher_batch_size` is opt-in 1 (compatibility default) or 2. At 2, T groups remaining ED/ES frames of the same patient within the current shuffled epoch, retaining a singleton if the other frame was consumed before resume. Both share one sampled spatial augmentation, identical native geometry, and one true stacked tensor forward. No crop/resampling/padding across patients. Each frame has its own original teacher-loss reduction; their losses are averaged for one optimizer step. Batch2 therefore changes optimizer update composition and stochastic augmentation correlation. It is an explicit training amendment, not numerically equivalent to two batch1 optimizer steps.

T retains 50 epochs and 8,000 labeled-frame exposures; historical `update` in T now denotes examples_seen, with `optimizer_steps` logged separately. T-a (first400 exposures), T-b LR schedule, epoch validation, budget and 20-frame checkpoint boundaries remain on exposure units; a batch never crosses them. B1 is unchanged. Existing progress/optimizer/RNG/sampler is retained by an explicit checkpoint/config identity migration, archived and SHA-receipted; no silent hash bypass. Original checkpoint assumed one optimizer step per consumed frame. J remains 100 epochs and 8,000 paired optimizer updates, oneL+oneU per step.

Hardware selection: largest native TRAIN patient057 shape (8,512,428), bf16 T-b batch2 fits (16.62GiB allocated) with 2.1348s/frame versus batch1 2.1641s/frame in the short probe. This is a small measured difference, not a validated campaign speedup. Full-fusion J batch2 exceeds 24GB GPU capacity (OOM); J batch1 is retained. Model, losses, split, labels, geometry and epoch counts are unchanged; no TEST outcome is used for this choice.

### [AMEND 2026-11-13] Semi K8 and source-to-target transfer

The full-label artifacts and all frozen historical tracks remain unchanged.
Track A uses exactly the L8 in the existing hash-pinned ACDC K8 pixdim manifest.
L contains only those eight patients' ED/ES frames. U contains ALL cine frames
of the other 72 TRAIN patients, plus non-ED/ES frames of L8. U records contain
image paths/hashes only. TRAIN construction, validation, profiling and training
must never open a mask outside L8; VAL/TEST masks remain evaluation-only.
No new selector is run for A. Existing TRAIN80/VAL18/TEST50 IDs, image arrays,
geometry and labels are unchanged.

User budget clarification: T preserves 8,000 labeled-frame exposures, hence
500 epochs of 16 L8 frames, with physical teacher batch2 (4,000 optimizer steps
when all batches pair ED/ES). T-a remains the first400 exposures; T-b, losses,
augmentation, teacher/student architecture and phase remain unchanged. J keeps
100 epochs of80 TRAIN patients, 8,000 L/U optimizer steps. Evaluate every epoch;
select on VAL only and apply the same fixed 0.01 FG deploy margin. Report actual
optimizer steps and exposures separately. The 500-epoch budget is fixed before
opening Track A TEST; full-label TEST is historical comparison evidence only.

Compatibility interface: original config/data v1 and their hashes are unchanged.
Semi config schema `self_audit.candidate_fuse.v2` adds exactly one root field
`experiment={"track":"A", "dataset":"ACDC"}` initially. Its other keys and
scalars match v1, except teacher_epochs=500. Synthetic epochs remain1.
Semi TRAIN manifest schema `self_audit.candidate_fuse.data.v2` retains the v1
record/geometry/split structure and adds `dataset` and `labeled_contract`.
`labeled_contract` contains `patient_ids` (sorted, unique), `patient_ids_sha256`
(SHA256 of compact JSON of that sorted list), `source_manifest` and
`source_manifest_sha256`. Verify source bytes and its L IDs; real A requires8.
TRAIN patients equal U patients; L is a subset. For L patients, exclude exactly
their annotated ED/ES from U; for others, allow every cine frame. Evaluation
v1 manifests can be reused byte-for-byte. Existing FuseData/FuseSampler public
signatures remain unchanged; labeled_keys iterates L only and the J epoch still
visits all U patients once. Builder outputs metadata only and reuses arrays;
it filters mask references before any FuseData validation or mask I/O.

Track B preparation audits every original M&Ms image header (pixdim, affine,
units and geometry consistency), using official TRAIN/VAL/TEST membership.
No ACDC pixdim waiver extends to M&Ms. Invalid TRAIN patients may be excluded
with an explicit receipt. Any invalid VAL/TEST patient blocks B and B0; report
IDs and continue A. Missing official split or required data also blocks rather
than inventing a split. Class mapping must be explicit and verified against
dataset evidence and permitted L8 masks. Select target L8 with the repository
seed42 selector and lock its IDs and manifest SHA before training. Target U has
all frames of other TRAIN patients plus non-ED/ES of target L8, image only.

After B preparation passes, B imports the SHA-verified selected ACDC80 teacher,
reports zero-shot target VAL, then fine-tunes only target L8 for2,000 optimizer
updates with T-b LR groups and target VAL selection. Frozen adapted teacher
drives the unchanged J objective. Source teacher, adapted teacher, student and
fusion are evaluated within one locked target TEST transaction. B uses source
ACDC80 labels plus8 target patients. Optional B0 uses nnFoundation initialization
and target L8 only, sharing the exact B split and L8 SHA. B/B0 execution is NOT_RUN
until the geometry/split gate passes and their explicit config extension is
implemented and checked; no placeholder may launch them.

Save per-patient metrics and paired two-sided Wilcoxon plus percentile bootstrap
95% CIs (10,000 patient resamples, seed42), fusion-minus-teacher and
fusion-minus-student, and B-minus-B0 if available. Label these analyses as
performed after TEST opening; never use them to select checkpoints or recipes.
Each track opens its TEST exactly once. OOM stops without crop/resampling.
For this extension run check_repo, the full pytest suite, and the synthetic
end-to-end smoke. CPU evidence is SOFTWARE_ONLY; patient metrics are separate.

### [AMEND 2026-11-13] M&Ms open-release geometry adjudication

The user requested resolution of the M&Ms geometry blockers. This explicit
opt-in policy is `mnms_open345_mm_polar_v1`; strict raw-header audit remains
unchanged. It applies only to the SHA-bound public 345-case release, whose
archive partitions are TRAIN175/VAL34/TEST136. This is the open subset, not
the complete original challenge cohort. Original NIfTI files, voxel arrays,
patient IDs, ED/ES indices and archive partition membership are immutable.
The organizer's data page specifies source labels 1=LV, 2=MYO, 3=RV; canonical
BG/RV/MYO/LV mapping is {0:0,1:3,2:2,3:1}.

The acquisition paper Table III explicitly reports spatial resolution and
slice dimensions in mm. Its pinned source is
https://diposit.ub.edu/server/api/core/bitstreams/f2f7dc43-becc-4554-910d-360ed7fd339c/content
with SHA256 `c3584856b3933cdeb440d3d2215f1bcdc6d90a7bbe5803bee615e7a823b4c7fd`;
an arbitrary caller-supplied document cannot authorize unknown units.
For these source-bound headers with unknown units,
record an explicit dataset-level mm interpretation, never claim that the raw
NIfTI declared mm or that original DICOM geometry was recovered. Preserve raw
findings and both original and effective affine in a separate metadata receipt.
No unknown-unit override applies to other data. Known meter/micron units are
converted explicitly, including affine translation, rather than relabelled.

A valid coded sform or qform is required; no identity/fallback geometry. Require
finite positive pixdim, finite nonsingular affine with homogeneous final row,
affine-column norms matching pixdim (rtol=1e-5, atol=1e-5), and normalized
Gram error <=1e-3. Use the polar factor U@Vt of column-normalized affine
(SVD) multiplied by original pixdim; preserve translation and handedness.
Accept only if the maximum displacement of all eight native image corners
is <=0.05 mm. These are metadata approximation bounds, not assertions that
all deviations are float32 rounding. Canonical z/y/x geometry must then pass
the existing repository validator. Material shear, invalid units/provenance,
missing header, or any bound failure remains a blocker for VAL/TEST.

The policy is locked before any target model result or target mask is opened.
Geometry evidence may read image headers inside the TEST ZIP without extracting
voxel data. Bind each row to source file/archive SHA and header SHA, preserve
all raw failures, and write new receipts only. A geometry pass alone does not
claim a prepared training manifest, target L8 lock, mask-map sample validation,
B/B0 implementation, TEST evaluation, or efficacy; those remain NOT_RUN until
separately completed. Never modify Track A's pinned checkout or running jobs.

### [AMEND 2026-11-13] Track B execution and implementation interface

The user explicitly chose 100 target J epochs: TRAIN175 gives 17,500 paired
optimizer steps. This is a larger J update budget than ACDC's 8,000 steps;
report both, without claiming equal compute. B T_tgt is exactly 2,000 optimizer
steps, with physical batch2, no T-a freeze, and the existing T-b LR groups.
Thus 4,000 labeled-frame exposures span 250 L8 epochs. Log exposures and
optimizer steps separately; validate after each complete L8 epoch. B0 remains
optional and is not automatically queued by this implementation.

Configuration schema `self_audit.candidate_fuse.v3` is B only. Its experiment
object is `{track: B, dataset: MNMS, source_teacher: {path, sha256,
metadata_sha256, identity}}`; identity has source config/train/val manifest
SHA256 fields. The source artifact must be SELECTED, match its sidecar and
the complete declared source identity, and strict-load every teacher parameter
(encoder, decoder, region/boundary heads). Real B pins the selected ACDC80
artifact SHA `bf2de8287a3fb6a5df7c490b0e081d39e58911a99cfa081243015a2c6ca894ec`.
All v1/v2 normalization and hashes remain unchanged. B training has the same
scalar keys, teacher_epochs=250, joint_epochs=100, teacher_batch_size=2;
the exact optimizer-step contract is additionally enforced by the engine.
Synthetic B uses one epoch and a local selected synthetic source fixture.

Data schema `self_audit.candidate_fuse.data.v3` represents native M&Ms NIfTI
sources, with a frozen manifest per role (train/development/test), exact
TRAIN175/VAL34/TEST136 membership, the existing geometry receipt and target
L8 lock pinned by SHA. `open_fuse_data(path, expected_sha256, *, role,
synthetic=False)` returns the existing FuseData API: manifest, manifest_sha256,
patients, labeled_keys(), unlabeled_keys(), load(key, labeled=...). Existing
v1/v2 callers and bytes retain their old behavior. The v3 reader lives in
`mnms_data.py`; metadata-only builder is `scripts/fuse_prepare_mnms.py`.
No new normalized cine arrays are materialized on disk: read native NIfTI,
transpose XYZ/T to T/ZYX and use the existing normalize_cine transform with
a bounded in-memory image cache. Cache settings affect storage only. Native
geometry is taken from the adjudicated SHA-bound receipt, never guessed.
TRAIN mask records/reads are limited to the locked L8. Other TRAIN patients'
ED/ES can be known or unknown and every frame remains U. TEST ZIP image/mask
voxel data is opened only inside the locked TEST transaction, not preparation.
Manifest preparation may use archive inventory/header receipts and CRCs.
All source container/member identities must be checked before voxel use.
For a TEST member without a prior full-member digest, member_sha256 is null;
the pinned complete archive SHA, exact member name, byte size and CRC bind
its bytes. header_sha256 is a separate image-header field, never substituted
for the full-member hash. Preparation must not open TEST member payloads to
fill missing hashes. Cache validated container hashes by unchanged file stat;
fail if the source changes. The image cache defaults to 32 GiB and can be
bounded through FUSE_MNMS_CACHE_BYTES without changing data or model identity.

Before T_tgt, preserve a frozen source export at
`exports/teacher_zero_shot.pt` (payload mode teacher), its standard sidecar,
and target VAL result `ZERO_SHOT_VAL.json`. Resume must reuse and verify these
artifacts rather than substituting the adapted teacher. Regular teacher export
means T_tgt. The unchanged three-mode J VAL selection and deployment rule use
adapted teacher/student/fusion only. B's RESULT exports add teacher_zero_shot;
the campaign/evaluate_exports API accepts exactly four comparison labels for
B: teacher_zero_shot, teacher, student, fusion. All four checkpoints, SHA and
manifest identity are locked before one TEST ledger opens; evaluation remains
one transaction, and failure never automatically reopens TEST. Source zero-shot
is report-only. v1/v2 retain exactly three comparison labels.

Operations: `scripts/fuse_transfer_queue.py` waits for A's committed campaign
completion and one-shot TEST result, verifies the pinned B config/source/data,
then runs bounded largest-native-volume T/J GPU profiles and the B campaign.
No active A process, checkout, run file or alarm is changed by preparation.
The queue uses its own lock, state, logs, checkout and run directory; it never
retries OOM, TEST or a failed profile, and has no unbounded automatic retries.
Require disk headroom for atomic checkpoint replacement before launch. No
old data/run deletion is authorized. A missing resource is reported explicitly,
not hidden behind a successful plan or silently reduced model/geometry.

Ownership during implementation: W1 data.py/mnms_data.py + manifest builder
and data tests; W2 config.py/engine.py/teacher.py + B configs and transfer tests;
W3 evaluation.py + four-mode transaction tests; W4 campaign/transfer queue/
profile/smoke scripts and operation tests. Coordinator owns this contract,
integration report, launch pins and server execution. Workers communicate
interface changes before editing another owner's files. Frozen historical
tracks and existing A config/manifest identities remain immutable.

### [AMEND 2026-11-13] User-authorized historical Q3D storage migration

The user authorized moving only the stopped historical run
`/root/mount_data/runs/q3d-resenc-pixdim-20261112` to their private Drive,
verifying SHA256, and then releasing its server copy. This is the sole exception
to the earlier no-old-run-deletion clause. Preserve datasets, full-label FUSE,
Track A, source teacher, and all active runs. Save the source inventory and
archive/deletion receipts outside the historical directory. Require completed
uploads, full remote-byte SHA256 verification for every ordered archive part,
an unchanged source inventory, and no old campaign/watchdog writer before
removal. A partial upload or failed verification never authorizes deletion.

### [AMEND 2026-11-14] A-aug: locked stronger in-plane augmentation

A-aug changes only augmentation relative to Track A at main `8fe7ace` and
`reports/fuse-semi-20261113/recipe.A.json` (raw SHA256
`d7af9576751eb7409093a8d7a5685d65c50baf088b168b289c83cf4b4d1bf8a8`).
Keep its exact L8/U/manifests, seed42, architecture, T 500 epochs/8,000 frame
exposures/physical batch2, J 100 epochs/8,000 paired updates, optimizer and
selection/deploy rule. Do not run or modify B/B0. Use a new server run directory.
The old A TEST50 was already opened: this experiment is explicitly **TEST
opening 2, after TEST was opened**, descriptive only; never use it for selection.
The primary comparison is student A-aug minus student A on VAL18, paired
two-sided Wilcoxon and 10,000 patient bootstrap percentile 95% CI, seed42.
Before training, report old teacher/student/fusion metrics on all16 L8 frames
without augmentation next to old VAL18; this diagnosis is not a training gate.

New `candidate_fuse.augmentation` code owns this opt-in policy. Existing v1/v2/v3
recipes without `augmentation` keep identical normalization, hashes and behavior.
Only the v2 A config may add top-level `augmentation`, exactly
`{"policy":"inplane_strong_v1"}`. The numerical policy is fixed in source and
this contract; unit tests may override probabilities on the augmentor directly.
Its enabled recipe must be identical to old A after removing this one key.

For each frame independently, sample one set of y/x transform draws, shared
across all z slices. Apply the transforms sequentially, including their
interpolation passes. Scale is visual magnification: s>1 zooms in (the inverse
sampling map divides coordinates by s). In order: arbitrary rotation [-180,180] degrees (p=.5),
isotropic y/x scaling [.7,1.4] (p=.5), 2D elastic deformation (p=.2), y and x
flips (p=.5 each), then the existing spacing-safe quarter turn (p=.5; k=1..3).
For a rectangular frame, a quarter turn samples onto the original centered
H/W canvas with zero padding, rather than swapping output shape. No operation
mixes z slices. The fixed output native shape, spacing and affine metadata are
unchanged. These are training view deformations, not dataset resampling.
Use bilinear image and nearest categorical mask interpolation, outside0/BG.
Elastic alpha U[0,200] pixels and sigma U[9,13] pixels are the nnU-Net v1 2D
defaults at commit `db16c6cef5fdd5a180159184e46b58bcca670446`:
https://github.com/MIC-DKFZ/nnUNet/blob/db16c6cef5fdd5a180159184e46b58bcca670446/nnunet/training/data_augmentation/default_data_augmentation.py
Smooth independent uniform[-1,1] 2D displacement fields with Gaussian sigma,
then multiply by alpha, matching batchgenerators' convention.

After spatial transformation: gamma [.7,1.5] p=.3, brightness multiplier
[.75,1.25] p=.15, contrast [.75,1.25] p=.15, zero-mean Gaussian noise with
standard deviation U[0,.1] p=.1, in-plane Gaussian blur sigma U[.5,1] p=.2,
in-plane low-resolution simulation scale U[.5,1] p=.25 (nearest downsample,
bilinear upsample to original H/W). Intensity parameters are per-frame;
noise is voxelwise. Finish with clip[0,1], the input domain of both wrappers;
teacher's unchanged internal full-volume z-score follows this clipping.

T uses strong spatial+intensity input and matching spatial mask; its two ED/ES
frames receive independent per-frame draws and retain their same native batch
shape. J L and U each use one spatial view shared by teacher and student;
only student receives the added strong intensity transforms. Teacher logits
and fusion/mimic features come from the spatial-only view. L target follows
that exact spatial transform. Evaluation, including Phase0, has no augment.
All probabilities zero must return bitwise-identical inputs. Assert finite
normalized images, categorical labels0..3 and unchanged z/shape/metadata.

Interface before parallel implementation:
`StrongInplaneAugment(seed=42, *, probabilities=None)` exposes
`spatial(image, target, geometry) -> (image, target, geometry)`,
`intensity(image) -> image`, `teacher_view(image,target,geometry)` returning
the spatial+intensity image/target/geometry triple, and
`joint_views(image,target,geometry)` returning
`(teacher_image,student_image,target,geometry)`.
Inputs are [1,1,D,H,W] float32[0,1], optional target [1,D,H,W] long.
Its private CPU torch.Generator is seeded run_seed+2003; all random decisions,
fields and noise use this generator, never global/sampler/dropout generators.
`state_dict()/load_state_dict()` strictly serialize/restore this RNG. Optional
probability keys are `rotate`, `scale`, `elastic`, `flip_y`, `flip_x`, `rot90`,
`gamma`, `brightness`, `contrast`, `noise`, `blur`, `low_resolution`.
Engine constructs one augmentor per run, checkpoints it as
`rng.strong_augmentation` (required when enabled), and passes it explicitly to
`joint_loss(..., augmentation=None)`; disabled calls retain old behavior.
Shared `prepare_teacher_batch(frames, device, *, py_rng, augmentation=None)`
returns image,target,geometry and is used by engine and the new profile script.

Save and inspect12 real L8 overlays. Profile20 actual optimizer updates each
of T(batch2) and J, including the largest native volume, into scratch output.
Record wall time/update, CPU augmentation time, peak VRAM, dtype and GPU model;
compare against the old A profile with explicit unit/batch caveats. If >50%
slower, explain the bottleneck before full launch. OOM or mask misalignment
stops; no automatic crop/resample or post-VAL weakening. Full pytest,
check_repo and synthetic T/J/eval/export/cold-load smoke precede launch.
One independent Claude reviewer checks logic and formulas. Run new watchdog
with zero automatic training/TEST retries and a visible live log. Publish
results promptly after TEST commits, including old/new VAL/TEST per-class
Dice+HD95, Phase0, paired VAL statistics and profile/total timing.
