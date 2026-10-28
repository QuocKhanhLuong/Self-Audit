# Compact implementation coordination contract — 2026-10-28

User prompt is authoritative. Read external design:
/Users/alvinluong/.local/share/openresearch/files/self-audit/fewshot-design-20261028/FEWSHOT_ACCURACY_SPEED_NOVELTY_VI.md
Base cf0fe5a, worktree /Users/alvinluong/orca/workspaces/Self-Audit/compact-kd-audit-20261028.
This is code-only: CPU manufactured tests and bounded smoke, no full patient jobs,
downloads or active job changes. Workers do not commit/push; coordinator integrates and publishes the separate branch.
You are not alone. Edit only owned files, never revert others. Use Semble before
semantic discovery and rtk prefix shell. Python /Users/alvinluong/miniforge3/bin/python.

## Interfaces to agree across workers

Mode schema `self_audit.compact.v1`, handled from canonical train.py -> engine.train
dispatch, implementation in self_audit/compact_training.py. Historical defaults unchanged.
No student EMA, prototype, feature caching, teacher memory or test-time adaptation.
Methods `supervised`, `plain_kd`, `scalar_guard`, `joint_guard`.
Patient CompactStudent: 2.5D residual UNet, configurable widths, GroupNorm, native
geometry, 3-channel context -> 4-class logits, fully learned encoder/decoder/head.
Default widths [16,32,64,128], random Kaiming init recorded distinct from DINO.

Worker model owns compact_model.py, compact_artifacts.py and corresponding tests.
Expose build_compact_model(architecture:dict), CompactStudent.module_parameters()
groups encoder/decoder/head. model(x,valid_mask=None) returns B4HW logits.
Artifact functions save_compact_checkpoint(model,optimizer,config,data,out,*,step,
teacher_identity=None,extra=None), load_compact_checkpoint(path,expected_sha,device),
export_compact_student(checkpoint,expected_sha,out), load_compact_export(path,expected_sha,device).
Loads return (model,metadata). Checkpoint saves return manifest SHA string. Data
supplies lids,uids,manifest,path. Export must work with only package+torch/numpy+
export assets, no source weights/teacher/cache/support/DINO. Strict config/state,
file hashes and separate runtime metadata. No path dereferences of training assets.

Worker teacher/data owns compact_teacher.py, compact_data.py, minimal opt-in edits
data.py and teacher/cache/preflight CLI + tests. Expose CompactTrainingData compatible
TrainingData methods, opt-in K1 reused audit patient, no duplicate fake patients.
Legacy loader K>=2 behavior unchanged. Teacher lineage preflight must count union
all labeled IDs (teacher, student, warm-start, audit, auxiliary), require <=K and
declared exact IDs. Frozen full EMA run quality identity/raw-vs-EMA state explicit;
unknown lineage blocks real use. Cache validates teacher + TRAIN image hashes and
native axes/validity, streams/memmaps with bounded handle cache, image-only U.
Expose FixedTeacherCache(path,sha,data,teacher_identity) and sample_batch(metadata,
device=...) -> (probabilities B4HW, valid BHW); expose validate_teacher_identity(...)
with documented signature. Worker decides teacher preflight/build CLI but coordinate
schema with trainer before completion. Default teacher state raw student from full
EMA historical baseline (report evaluates student, not teacher EMA). Known historical
pins available reports/baseline_recovery/HISTORICAL_EVIDENCE.json. Do not silently
ignore source-closure mismatch. New canonical teacher can be verified by exact
source/hash/recipe/physical artifacts. Historical bridge must be explicit and safe.

Worker audit owns compact_control.py and tests. Expose build_class_candidates(model,
x,targets,valid,*,lr,temperature=1.,coefficient=1.,confidence=0.,max_norm=None),
return dictionary directions (list named-param displacement dicts), source_ids,
denominator and telemetry. Soft CE T^2 * coefficient normalized by COMMON valid
teacher-confident pixels, group by teacher argmax up to4; sum exactly plain SGD.
Expose apply_compact_update(model,candidates,*,method,audit_batches=None,config=None)
-> receipt. audit_batches is repeatable callable yielding mappings x,y,valid,
frame (int),patient(str); stream whole labeled frames. Post-supervised reference.
Functional trials with cloned buffers, restore modes/RNG, no live state mutations
until direct commit. No optimizer called. Hard multiclass Dice per frame then patient
class averages; absent frames have separate FP fixed-valid-count checks. eps0,
absentFPbudget0, numeric tolerance separately declared; default max_trials8.
Finite deterministic search acceptable, honest no convex-optimality claim. Scalar
common alpha vector; joint vector finite candidates; M1 identical. Same ridge,norm,
trial cap, inputs. Log all trial alphas/objectives/margins, status no_candidate,
zero_displacement,rejected_all,accepted_nonzero; whole-network actual forwards.

Worker inference owns compact_inference.py, scripts/infer_compact_student.py,
scripts/benchmark_compact_student.py, scripts/evaluate_compact_student.py and tests.
Use compact export loader and existing evaluation.confusion_matrix/class_metrics.
Native manifest-driven input, streamed uint8 mask files, full TZYX shape + affine/
axis/class metadata. Real affine metadata required (do not invent); explicit image
normalization contract; per-cine .5/99.5 percentile only if raw mode, prepared mode
validate [0,1] without renormalizing. Freeze all outputs before opening references.
Metric EXACT historical: per-frame class Dice, available frames per patient/class,
available patients per class, macro4 class means, FG separate. Absent class None.
Exported model only quality. Benchmark baseline and compact same device/cases/
precision/batch/warmup/IO; isolated subprocess per model to measure peak process
RAM (no inference workers or include them). CUDA sync, allocated+reserved, median,
p95, model-only and E2E including IO/preprocess/transfer/write, load separately,
param count/weights-only bytes; no full probability cache. EMA reference means
one exact full EMA run checkpoint with explicitly selected raw/teacher state.
No unverified equivalent source/state. Missing real assets stays NOT_RUN.

Coordinator owns compact_training.py + train.py/engine.py integration, config/docs
and integration tests unless explicitly reassigned. Every worker supplies tests
and short report of actual commands/results and unresolved integration contract.
No quality/speed/novelty claim from synthetic results. No new sweep/full run.


## Integration disposition

All five responsibility areas have been implemented and delivered. The trainer
ownership was explicitly reassigned to a worker. AGY supplied the initial model,
teacher/data, control, trainer and inference work; after recorded quota failures,
Luna completed trainer and inference repairs under the same Orca tasks. The
coordinator independently reviewed and repaired cache identity, complete audit
input hashing, zero coefficients, geometry, bounded memory and evidence gates.
The final acceptance record is reports/compact_delivery/VALIDATION.json. Earlier
worker test summaries are intermediate observations, not final receipts.
