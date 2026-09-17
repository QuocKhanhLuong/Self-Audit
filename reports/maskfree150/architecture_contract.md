# Maskfree150 architecture contract v1

Authority: complete `self_audit_maskfree_150_orca_prompt.md`, read before dispatch.
Coordinator is sole architect/integrator/final reviewer, requested Astra XHigh.
Current source: a3737081fe10c1e4c38ba3e2fbeaddc79f1c4e65, main, verified upstream.
LATEST USER CHANGE: CODE ONLY. RTX 4070 is occupied; do not contact it, use it,
or launch training there. Implement and focused CPU-check on this existing main.
No new worktree. Production output defaults remain under
`/home/linhdang/workspace/quockhanh_workspace/SpecMamba`. Full real-data/GPU
execution and batch8 verification are NOT STARTED, deferred by user.
Code/review evidence lives in this checkout. No unrelated edits, staging or push
by workers. Each worker is not alone: respect ownership; never revert others.

## Supervision and information firewall

2026-09-15 user amendment: enable validation Dice every epoch after freezing
both compared students' development predictions. ACDC development membership
comes from the existing patient-ID training split; M&Ms preserves its official
validation image folder. Reference masks/metadata remain confined to a separate
post-freeze evaluation process. Scalars are report-only, with no threshold,
loss, scheduler or checkpoint selection feedback. This overrides the original
final-only reference timing below; O_verify and common-bank final freeze rules
remain unchanged. Production configs enable `epoch_validation` by default and
may forward an opaque `epoch_reference_config` path to the isolated process.

New package `src/self_audit_maskfree` MUST NOT import `self_audit` or
`self_audit_candidate_c` or their utilities. No pretrained weights, GT teacher,
manual mask loading, reference-derived crop, ED/ES selection, split, tuning,
checkpoint selection. Never open annotation metadata to get image membership.
Only isolated `self_audit_maskfree.evaluation.reference` may read masks,
and only after validating a complete immutable prediction freeze manifest.
No evaluator import from training. Missing reference evaluation is unavailable.

Study patient partitions and observation masks are fixed before learning and
hashed. Producer/normalization/hypotheses/ontology/nuisance fitting receive
ONLY FittingView. Auditor receives ScoringView(role=select); verify view is
constructed only by final verification after ALL predictions are frozen.
Student training inputs are the fit-view context, pseudo-labels detached.
Full-input deployment inference is a distinct manifest and receives no claim
of withheld-intensity prediction. Selection images are adaptive evidence.
Final-test patients never participate in producer/student fitting.

## Frozen shared API (contracts.py belongs to coordinator)

Use the dataclasses in `src/self_audit_maskfree/contracts.py`; do not duplicate
them. Numeric tensors: image/context float32; masks bool; labels long;
probabilities float32. No batched axis inside FittingView/ScoringView/Hypothesis.
FittingView image [1,H,W], context [3,H,W], support [H,W]. Every withheld
pixel AND its guard band is physically zero in ALL three channels. All masks
share native pixel coordinates at this stage. A unit is a center slice from a
volume, or a center slice at a fitting cine time. Metadata is whitelisted.
ScoringView exposes ONLY values of its role, zero elsewhere, normalized by
the same stored fit statistics. FittingView has no reference to original image.

### W3 data API

`discover_dataset(root: str|Path, dataset: str, *, seed=42,
  protocol='auto', depth_axis=2) -> dict` returns JSON manifest (schema_version,
dataset, records, manifest_id, resolved_protocol, readiness, limitations).
Each record contains study_id, patient_id, unit source path, split
(`train`,`dev`,`test`), shape, dtype, native geometry status, frame metadata,
cohort provenance. Preserve official image folder test membership, otherwise
image-ID hashed 70/15/15 split at patient level; no synthetic fresh-test claim.
Prefer raw images if present; detect duplicates, never count 4D cine plus its
ED/ES exports twice. Historical preprocessed-only roots disclose cohort bias.
NPY lacks native affine: stored-grid export only, never native-grid claim.
Native NIfTI retains affine/header and exact inverse axis transform.
`ImageOnlyDataset(manifest, split='train', image_size=128, seed=42)` implements
`__len__`, `__getitem__ -> TrainingUnit`, `unit_ids`; no verify access.
`load_verification_unit(manifest, unit_id, freeze_manifest) -> ScoringView`
requires a validated freeze receipt and must match the training partition_id.
Expose `load_full_input(manifest, unit_id) -> (Tensor[3,H,W], record)` ONLY
for deployment export, never training. W3 may provide small helper functions.
W3 is responsible for documenting any extra exact read/export APIs in its report.

Spatial profile: reproducible blocks 8px, 20% select, 20% verify, remaining
fit; 2px guard band around withheld blocks removed from fit. Same XY role
mask across Z context prevents neighbor leakage. Build splits on native grid,
then zero, then resize independently (nearest masks; masked interpolation
that cannot mix role values). Use fit-only percentiles 0.5/99.5 and mean/std
with floor 1e-6. No image-informed crop: whole FOV. Frozen sampling unit list.
All raw reading is data-layer-only; emit fitting/selection capabilities.

Cine protocol may only resolve when real multi-frame images, reliable times
and geometry, at least 8 frames allow fit/select/verify. Fit frames and nuisance
motion must never consume select/verify images. If valid temporal transport
implementation is unavailable, resolve and disclose spatial_predictive on each
image volume (weaker experiment), NEVER claim cine_predictive. This explicit
fallback and reason are persisted. No manufactured cine from ED/ES.

### W4 models/losses API

`Producer(feature_dim=16, width=16)` forward([B,3,H,W]) returns dict with
`features` [B,16,H,W], `reconstruction` [B,1,H,W]. GroupNorm CNN, random init,
<=5M params, no external checkpoints/downloads. `Student(width=16)` returns
logits [B,4,H,W]. `make_models(seed=42, width=16, feature_dim=16)` returns
dict producer, student_no_audit, student_audited. Students exact cloned initial
state, no shared storage. Expose measured parameter counts.
`producer_loss(model, context, fit_support, *, generator=None)` returns
(differentiable scalar, dict scalar tensors). It performs known invertible
flip/90-degree paired correspondences, bounded sampled contrastive positives
and negatives from same patient's spatially separated patches (false-negative
limitation), local randomly masked fitting reconstruction, geometric equivariance.
At most 128 samples per image, no HW squared allocation. Targets only O_fit.
`student_loss(logits, probabilities, validity)` returns (scalar, metrics).
Shapes B4HW/BHW; weighted soft CE, denominator sum(validity), differentiable
zero for all-invalid batch, explicit skipped flag. Same loss in both arms.
SSL collapse feature variance metrics. Producer loss never consumes student.

### W1 observation API

`ObservationModel(max_iterations=5, variance_floor=0.05, beta=0.01)`;
`fit(hypothesis: Hypothesis, fitting_view: FittingView)->FittedHypothesis`;
`score(fitted: FittedHypothesis, scoring_view: ScoringView)->EvidenceScore`.
Only full draft partition enters likelihood (validity cannot hide bad score).
Fixed 4 anatomical region components, optional exactly 2 nuisance BG modes
ONLY if capacity matched across every bank member. Use deterministic constrained
Gaussian/Student-t appearance fit, same capacity/steps/init recipe for candidates.
Empty regions use fit-global distribution, explicit count. Fixed variance floor
in fit-normalized intensity squared. Restrict bias to at most affine XY plane,
ridge >=0.1; no free pixel means or unrestricted decoder/target skips. W1 MUST
record resolved exact equation/capacity/fit budget in its report.
Score = mean NLL nats per valid observed pixel + beta*complexity + prior.
complexity = neighbor disagreement fraction (+ fixed component penalty if used).
Report NLL sum/count, normalized NLL, complexity/prior separately. Missing support
=> unavailable with reason and None, never zero. Enforce study/unit/partition IDs.
Fit gradients and states local to candidate; all data detached for nuisance fit.
No refit during score. If cine supported later, only explicit fit-time motion
parameters predict withheld times; otherwise data resolves spatial as above.

### W2 anatomy/bank/audit API

`resolve_roles(partition, fitting_view)->Hypothesis`: concrete image-only rules,
never GT cluster matching. Use orientation, enclosure/adjacency, continuity,
role alternatives; fixed named order BG=0 RV=1 MYO=2 LV=3. Missing geometry or
ambiguous role sets semantic_unresolved, alternatives, validity downweighted/0.
Do not require all classes in all slices. Do not call arbitrary clusters anatomy.
`generate_bank(fitting_view, features=None, *, seed=42)->list[Hypothesis]`:
four TOTAL candidates including incumbent from bounded grouping and an image
anatomical initializer; others distinct bounded boundary / split-merge / semantic
alternatives. Use fit intensities/features only, nearest-fit extension for hidden
locations, never O_select intensity. Return IDs and generation costs. Ensure
at least semantic alternatives preserved when partition score ties.
`audit_bank(bank, fitting_view, selection_view, model, *, rounds=2,
  improvement_threshold=0.001)->AuditResult`: streamed fit 5 iterations each,
four total candidates (two rounds over bounded candidate slots, not 4*2 hidden
budget); cache same incumbent fit; choose strict improvement >=0.001 nats/pixel;
incumbent retained for ties/no improvement. No early starvation after rejection.
Regional margins over distinct alternatives; confidence never correctness.
Pseudo validity intersects semantic validity and region evidence/reproducibility,
without relabeling uncertain pixels BG. Empty challenger search inconclusive.
Trace candidate distinctness/costs/acceptance/prior violations; full masks retained.

### W5 config/trainer/runtime API

`MaskfreeConfig` strict dataclass fields: dataset (acdc|mnms), data_root, output_dir,
total_epochs=150, seed=42, batch_size=8, accumulation_steps=1, image_size=128,
lr=0.001, weight_decay=0.0001, warmup_epochs=5, width=16, feature_dim=16,
protocol='auto', depth_axis=2, device='cuda', num_workers=0, amp=True,
wandb_mode='offline', wandb_project='self-audit-maskfree', run_id optional,
max_steps optional (smoke only), max_epochs optional (interruption test only),
resume optional, allow_cpu=False. Support unknown-key rejection and positive
range validation. `load_config(path)->MaskfreeConfig`; `to_dict()` method.
No arbitrary legacy overrides, pretrained options or reference paths.
`MaskfreeTrainer(config).run()` used by `scripts/train_maskfree.py --config X`.
CLI additionally `--resume`, `--max-steps`, `--max-epochs`, `--allow-cpu` through
same validated dataclass. `--preflight` performs actual bounded fit/score/backward,
checkpoint and tiny export, records physical/effective batch; never marks full
completion. Full launcher MUST require successful matching GPU/batch gate receipt.

One epoch visits shuffled seeded unit IDs once, including final short batch.
Each batch producer SSL step, independent bank creation from detached producer,
both students use SAME bank/initialization/augmentation/update budgets. No student
feedback. Label ramp .05+.95*min(1,(epoch+1)/20). Every epoch audits from zero.
Three optimizers single warmup cosine global schedule, recorded component steps.
Accumulate correctly including final partial group; no silent OOM adjustment.
Track label lineage per epoch/unit, natural coverage AND E6 matched-coverage
analysis (W6) using same arms; do not imply extra matched-coverage learner.
Exact resume at batch boundary: all model/optimizer/scaler/RNG states, epoch,
sampler permutation and cursor, counters, histories, dataset/config/source and
observation partition hashes. Config/source mismatch fail closed except explicit
operational resume/max_steps/max_epochs; no old checkpoint overwrite on failure.
Atomic writes; immutable run ID/versions; no reuse unrelated output directory.
Final valid state last_completed_epoch=149, epochs_completed=150. Partial smoke
or missing final jobs is NOT COMPLETED. last.pt and producer_final.pt,
student_no_audit_final.pt, student_audited_final.pt; saved identity snapshots.
GPU stats null on CPU; all compute/I/O passes including final export recorded.
W&B one run per dataset, offline if credentials unavailable, 3 namespaces.
Finalization: freeze banks/predictions/checkpoint hashes, export all compared
E0-E5/controls and student arms, validate freeze, THEN image-only verification,
never use verify for best checkpoint or refitting. Call W6 and W7 APIs below.

### W6 experiments/evaluation API

`run_bank_experiments(audit_result, fitting_view, selection_view, model,
  *, seed=42)->dict` returns rows E0-E5 and six negative controls + paired search
comparison using same number/cost random edits. E1 cuts_inspired_control;
E2 uses fit-only confidence/consistency metadata; E3 fit score; E4 and E5 may
tie exactly on common bank. E0 simple intensity grouping with same ontology.
Return selected candidate IDs AND Hypothesis objects in `predictions` mapping
for exporter; serializable rows in `rows`; exact CUTS reproduction pending.
`matched_coverage(validity_a, validity_b)` returns two masks of equal supported
count using stable ordering; no GT. Coverage curves 25/50/75/100 explicit.
`verify_frozen_bank(fitted, verify_view, model, freeze_manifest)->dict`
checks all freeze files hashes and never refits, then scores O_verify; include
selection-verification gaps, perturbation sensitivity, annotation stability,
degenerate challenge outcomes, availability reasons. One-time verification
receipt, no tuning API. Reference module only standalone process after freeze.
Metrics rows required dataset/split/protocol/epoch/checkpoint/population/count/
availability/unit/contract_version. W6 shared `metric_row` helper may be used.
Reference CLI explicit frozen manifest/reference config/output; no training import.
Dice/IoU 3D classwise patient macro FG, both-empty excluded/count, one-empty
error; geometry-valid surfaces or unavailable; paired patient bootstrap;
evidence-quality correlation/ranking with tie support policies; true edit harm;
fixed named map no oracle permutation. JSON/CSV/MD tables.

### W7 exports/launch API

`export_prediction(output_dir, *, record, prediction_name, labels,
  probabilities, validity, alternatives, checkpoint_id, version)->dict`:
2D units assembled by record native slice IDs into full volumes at finalization.
Explicit native reverse shape/axis interpolation for probabilities, nearest labels,
validity weighted support; output 4HW per-unit then 4ZHW volumes. Native-grid
NIfTI only when inverse geometry verified; NPY stored-grid explicitly labelled
and native export unavailable. No silent volume resize without transform.
`freeze_predictions(output_dir, entries, checkpoint_paths, *, dataset,
  protocol, epoch=149)->dict` creates manifest with file SHA256, entire compared
method set, observation/manifest lineage; rejects overwrite/divergent identities.
`validate_freeze(manifest)->None` verifies physical files and checkpoint hashes.
Exports label/prob/validity/alternative sidecars. W7 coordinate unit assembly API
with W5 by message, not editing W5 files. Freeze must enumerate all comparisons.

Launch scripts strict set -euo pipefail, workspace containment, read-only unique
resolved configs via W5 schema, independent ACDC then M&Ms runs. Defaults preserve
batch8 accumulation1 and fixed image_size128, no fallback/OOM retry. Probe actual
GPU identity, choose verified RTX4070 UUID (or explicitly user configured rental
GPU), refuse unsupported GPU silently selected. Explicit preflight fallback config
4x2 or 2x4 and record contrastive batch limitation before full run. Check complete
receipt before second dataset. No existing jobs touched. All artifacts under
runtime workspace/runs/maskfree150/<dataset>/<run_id>. Strict --help documented.

## Shared validation and reporting

Only 1-3 focused checks per owner (no repository-wide suite), then root combined
critical integration gate. CPU synthetic fixtures are software evidence only.
No real data exists in this local checkout; native metadata/cine remains UNKNOWN.
W8 owns firewall tests/red_team.md only and independently inspects production
code; asks owners to fix, does not change their code. Root performs final review.
Each worker returns report Wn_delivery.md with paths, actual commands/output,
requested/resolved provider/model if observed, PASS/REVISE/BLOCK limitations.
No novelty/accuracy/correctness guarantees. Semantic failure and pipeline runtime
completion are distinct; zero useful coverage marks label-generation objective
failed even when legal SSL ran. Missing metrics = None/available=false/reason.

## Provider and execution facts at dispatch

Orca runtime 3a1eeb25-5acb-4652-bf9a-f004079f0bbf; run run_d19ed61b546b.
Claude quota API status ok: session used 12%, weekly 2% (snapshot only).
AGY binary exists but Antigravity quota API unavailable, Gemini OAuth disabled;
W3/W6/W7/W8 reassigned to Opus. Model provenance comes from actual launch/
transcript; an opus request is not proof of resolved provider model.
SSH probe before user cancellation failed sandbox connection Operation not
permitted; privileged retry was interrupted. Do not claim remote auth failure
or inspected GPU/data. User subsequently forbade use of occupied 4070 for now.


## Integration amendments (2026-09-15; root decisions)

These clarify v1 after review; they do not claim real-data validation.

- Enumerate all native Z slices and all available T frames. This implementation
  resolves spatial prediction only. A frame is not temporal evidence. Patient
  identity controls splits; acquisition plus frame controls `volume_id`; Z adds
  `unit_id`. Reject mixed native grids within one study when their role maps
  cannot be guaranteed consistent. Deduplicate only proven image-identical
  frame exports with matching geometry. Physical millimetres require known units.
- Primary appearance capacity is one Gaussian per named region, identical across
  candidates. Primary prior penalty is zero; ontology rule violations remain
  diagnostics. Fit snapshots are immutable and score supports must be disjoint
  from recorded fitting supports.
- Regional evidence must score distinct candidate fits on the same observed
  region. No challenger or no observed region gives zero evidence margin.
  Same-bank label agreement is not independent repeated annotation stability.
  Current one-hot candidates have constant confidence; E2 is consequently a
  smoothness/consistency control, not a learned confidence selector.
- Auditing runs explicitly on detached CPU inputs/features. GPU timing is
  synchronized and stage times are inclusive, so they must not be summed.
- Apply label ramp to the normalized student loss. Skip an arm optimizer when
  the accumulation group has no valid labels (including AdamW decay), record the
  skip, and continue producer learning. Validate all component gradients before
  any optimizer mutates parameters. Weight a short accumulation group by its
  actual number of examples; never silently change physical batch.
- Exact resume includes partial-epoch accumulators and committed log offsets,
  runtime/device identity, source/input fingerprints, and deterministic RNG.
  Checkpoints are taken at accumulation boundaries, at least every epoch and
  every 50 optimizer groups. A short software timeline can finalize artifacts
  but cannot claim a completed 150-epoch experiment.
- Persist per-unit nuisance fits before freezing; do not retain all banks in
  RAM. `run_bank_experiments` additionally returns `fitted_predictions` and
  `control_fitted`; reuse selector fits from the common bank. Both students get
  their own fit-only appearance fit before verification. All these states enter
  the same hash manifest as the predictive exports and model checkpoints.
- A run-local foreign-evidence cache is seeded with actual other-study scores
  before selector comparisons. Use declared generator-slot correspondence across
  studies, not coincidental candidate ID equality. One-study controls remain
  explicitly unavailable.
- Freeze the complete predictive set first; then export and freeze both full-input
  student deployment predictions; then verify. Verification rankings are only
  diagnostics. Perturbations and degenerate challenges use the candidate already
  selected on O_select. No reference data or O_verify selects a checkpoint.
- Native exports assemble by volume/frame identity, never overwrite multiple
  acquisitions under one patient. Reference aggregates volume results within
  patient before patient-level summaries or bootstrap.
- Software exceptions in mandatory finalization fail the run. Useful foreground
  coverage is reported separately from runtime completion and appearance NLL.
- Opus session quota was exhausted during integration. User-authorized Luna
  workers finish isolated corrections; real returned IDs and unresolved model
  identifiers are recorded in `worker_registry.json`. Root remains architect,
  integrator and final reviewer. The busy RTX 4070 is not used.
- Final annotation stability is a separate pre-freeze repeat: fixed normalized
  Gaussian noise sigma 0.05 on O_fit only, same bank-generation seed, producer
  inference and bounded audit repeated without changing the primary labels.
  E1/E5 and both student agreements include common-valid denominators; repeat
  predictions/fits and cost counts are frozen with the primary nuisance state.
  This is perturbation repeatability, not accuracy or independent confirmation.
- Full verification uses the original manifest object inside a process-local
  read-only session. Full corpus hashes are checked on entry and exit. Each unit
  has an atomic durable result keyed by the actual prediction freeze ID and its
  observation partition; resume consumes that result without rescoring.
- Default data paths are unverified raw-image placeholders, not the supervised
  preprocessing directory. Set actual native image roots after inventory. No
  mask-independent preprocessing history is assumed from an image-only filename.
