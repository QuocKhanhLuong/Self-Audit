# Candidate FUSE — native supervised/cine teacher–student fusion

2026-11-13. This is the implementation contract for `candidate_fuse`.
Uses labels of **all TRAIN80 patients; not a few-label result**.
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
T is 50 epochs = 8,000 optimizer updates. T-a is first 400 updates (5%),
including half of epoch 3; schedule boundaries remain exact in update units.
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
