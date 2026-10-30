# Compact student: fixed-teacher transfer with a finite TRAIN guard

Status at implementation: software delivery, not a verified patient model. The
historical full-EMA-run baseline is a reference, not a result of the compact model.
This mode is `self_audit.compact.v1` through canonical `train.py` and
`self_audit.engine`. The default DINO/EMA, window and automatic-head recipes retain
their original meaning. This mode does not use teacher memory, prototype retrieval,
test-time adaptation, ensembling or a compact-student EMA.

## Locked experiment and acceptance

The first comparison uses the exact historical ACDC TRAIN L8/U72 manifest SHA
`e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`.
L is patient007, patient010, patient011, patient024, patient026, patient041,
patient077 and patient099. All supervised, support and audit label access belongs
to these eight patients. U records contain images only. VAL/TEST masks never make
targets, select support/crops or govern a training transaction.

The target is **Macro4 >= 0.87**, using the exact historical reduction: compute
confusion within a reference frame volume; average defined frame Dice within each
patient/class; average defined patients within each class; then average the four
class means BG/RV/MYO/LV. Report each class and foreground macro separately. If GT
for a class is absent in a frame, its Dice is undefined, even if it has false
positives. Ignore255 and padding contribute no confusion. Do not substitute pooled
voxel Dice or foreground87%. The already exposed TEST50 is a development recovery
reference, not a new confirmatory holdout. No post-hoc noninferiority margin exists.

The historical baseline report is Macro4 .8729522993309857 and foreground
.831862063046926. Its four class means each have 50 present patients, so the extra
patient-first summary in the historical report happens to agree. The fixed-window
report is Macro4 .8780314989716015 and foreground .8385200133019562; it is a
different recipe. `reports/compact_delivery/SOURCE_INVENTORY.json` records these
identities and the hashes of all six supplied design snapshots. All design assets
were available; the physical patient checkpoint payload was not verified here.

The exact baseline inference state is **raw student** from method `ema_full`,
step10000, including 1000 full-model supervised warmup steps. Its source is
`0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`, manifest SHA
`03c1730f88941ada898bed7fcf1b9f13cdd8f08bcce4390539fec665c82e97de`, payload SHA
`b36bcafe9613325d85ee70dabd6a36fef7c17429882967831a151c0cce6d8437`.
Calling the training method EMA does not mean the published inference used the
payload's teacher state. A different selected state needs its own quality evidence.

The scalar transfer-utility study uses a **different recovered B8 teacher**,
recorded in `results/compact_acdc_l8_20261029/experiment_20261028/`:
manifest SHA `ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa`,
payload SHA `0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6`,
inference state **raw student**, with the same ACDC L8 lineage. These are archived
identity claims until the physical payload, quality receipt, source closure and
cache are verified on the execution server. Do not substitute the historical
`b36b...` payload or its quality result for this recovered teacher. The existing
historical example commands below retain their original teacher meaning.

Final GO requires evaluation of the exported-and-reloaded compact student plus a
matched-device/workload benchmark showing lower end-to-end seconds/volume, peak
GPU allocated/reserved memory, peak inference-process host RAM and deployed
weights-only bytes. Report p95, model-only timing and throughput too. Missing
patient measurements means NOT_RUN/BLOCKED, not GO. If plain KD passes deployment
gates while audit adds no benefit, report deployment success and unproven audit
contribution. Joint superiority requires empirical evidence over scalar and the
matched plain update rule. Synthetic witnesses establish mechanics only.

## Model and optimization

The patient-capable compact model is a residual 2.5D U-Net with small encoder and
decoder, skip connections, configurable widths (initially16/32/64/128), GroupNorm
for small batches and a four-class head. Three adjacent Z slices with repeated edge
slices form the input context. Native grids are preserved. Random initialization
is explicit and differs from the teacher's SSL DINO initialization. The compact
arms share seed, initial weights, architecture, schedule and L/U sampling.

Supervised CE + present-foreground Dice trains encoder, decoder and head. The
warm-start witness records optimizer ownership, nonzero gradients and actual
parameter deltas for these groups. Teacher parameters are frozen in eval mode.
There is no parameter EMA across different architectures and no compact EMA gate
on fixed-teacher targets. Distillation is output soft CE with declared temperature,
T-squared scaling, coefficient, teacher confidence and a common accepted-pixel
denominator. All target transformations must align with the student's spatial
crop, padding, flip, axes, support validity and canonical class order.

Each transaction has this state timeline:

1. Commit the supervised AdamW update, defining theta_ref.
2. Obtain fixed teacher targets on TRAIN U, independent of student confidence.
3. Compute at most four teacher-argmax class displacement candidates at theta_ref.
4. Propose scalar or vector coefficients through bounded deterministic search.
5. Run actual whole-network functional trials on the fixed TRAIN audit unit.
6. Commit the selected additive SGD displacement once, or retain theta_ref.

The common-denominator class displacements sum to the matched plain-KD SGD
displacement. No momentum or second AdamW transformation follows an accepted U
update. Scalar and joint share candidates, audit data, normalization, regularizer,
norm policy and maximum trial evaluations. M=1 reduces to the same policy. Finite
search is not a claim of globally optimal joint allocation. Whole-network response
cannot be inferred from an affine-head feature cache.

Default audit uses complete labeled reference frames of one budgeted TRAIN
patient, streamed by slice batches. Reference and trials use identical labels,
images and validity. Present foreground classes must not lose hard argmax Dice;
absent-class FP has a separate fixed-valid-voxel denominator and zero increase
budget. Epsilon=0; numeric tolerance is separate and logged. Never relax a guard
because it rejects all trials. The default cap is eight trial evaluations, plus
the reference evaluation; all proposal/solver/audit time is charged. Log no
candidate, zero displacement, rejected-all and accepted nonzero distinctly.

Rejected proposals preserve post-supervised parameters, buffers, optimizer state,
RNG and teacher. The guarantee is only about the extra U displacement on the
declared current TRAIN measure. It is not a test Dice, supervised-step, cumulative
trajectory or unseen-patient guarantee.

## Opt-in scalar transfer-utility selectors

The [implementation contract](CPTU_IMPLEMENTATION_CONTRACT.md) and
[pilot runbook](TRANSFER_UTILITY_RUNBOOK.md) describe this additive extension.
`selector=original` is the default. The original finite search minimizes
`1 - foreground_macro_Dice + ridge * sum(alpha**2) / 2` over feasible trials.
The scalar grid is `[1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]`, with
`alpha = s * ones(m)` and penalty `ridge * m * s**2 / 2`. Its historical search
does not include zero as an objective competitor; zero is retained when there
is no accepted nonzero displacement. This behavior is preserved.

`larger_tie` first resolves that original search. Among feasible trials whose
foreground Dice equals the original choice within the declared Dice tolerance,
it selects the largest scalar. The default Dice tie tolerance is exactly zero.
The guard, ridge and ordering of distinct Dice values are unchanged.

`cross_patient_utility` retains the same proposal, grid and hard guard. It draws
one image-only TRAIN-U probe batch from a patient absent from the proposal batch
using an independent, checkpointed RNG. After the supervised update, it freezes
the probe inputs, aligned fixed-teacher targets and confidence/validity mask.
For each feasible trial it measures `gain = KD(theta_ref) - KD(theta_trial)`
on that same probe. Zero has gain zero and reuses the reference evaluation.
A nonzero trial must improve gain beyond `utility_gain_tolerance`; gain ties
within `utility_tie_tolerance` prefer smaller displacement. Both tolerances
default to `1e-7`. No eligible probe, no accepted pixels or nonfinite utility
means zero, with an explicit reason. These tolerances do not change the Dice
guard epsilon or absent-FP budget (both zero) or its separate numeric tolerance.

Raw teacher confidence is applied before temperature softening, exactly as in
proposal KD. Each probe loss uses the same fixed denominator and targets for
the reference and all trials. No VAL labels, extra teacher update, optimizer
step or affine-feature approximation enters selection. Probe patient separation
holds within a transaction; it is not a previously unseen patient cohort or a
generalization guarantee. Positive teacher KD gain is not clinical correctness.
The optional `same_batch_utility` control reuses the proposal batch explicitly.

The archived compact recipe at `d93c4de` samples a batch of slices from one
patient, applies image-geometry random crops with edge padding, and excludes
padding through cached validity. It does **not** call the native EMA trainer's
flip or strong-intensity augmentation helpers. The scalar pilot preserves this
actual crop-only recipe for both proposal and probe. Labeled slice sampling is
foreground-aware with probability .5 inside budgeted L; U frame/slice sampling
is uniform and image-only. Inputs are already normalized TZYX arrays in [0,1].

Exact resume records global RNGs, the proposal and probe generators, optimizer,
coverage and protocol identities. A legacy compact checkpoint remains loadable
for export, but cannot claim exact training resume without those states. Probe
assets and selector state are training-only: deployed exports remain independent
of teacher, support and probe data.

## Label lineage and expansion

Count the union of labeled patients used by teacher, any segmentation warm-start,
student, audit and auxiliary supervision. Unknown lineage blocks a dependent real
run. TeacherL8 cannot be used to claim totalK1/2/4. K1 explicitly reuses support
within the same patient and does not claim patient-disjoint audit. No K sweep runs
automatically. Small-K configs require a teacher with matching total lineage.

Joint ACDC+M&Ms is a separate locked protocol. Namespace IDs, keep duplicate checks
and existing quarantine, declare K_acdc/K_mnm and the union with teacher labels.
Default domain sampling is explicit50/50. Canonical labels are BG0/RV1/MYO2/LV3;
raw M&Ms mapping is0->0,1->3,2->2,3->1 exactly once, with metadata proving whether
prepared masks are already canonical. Do not reuse ACDC-L8 teacher plus four MnM
labels and claim totalL8. Complete the ACDC milestone before any joint run.

## Deployment and accounting

The deployed artifact contains architecture, preprocessing/class metadata and
student tensors only. It reconstructs from that architecture and its own weights;
teacher/support/cache/bank/original pretrained files are not runtime requirements.
The runtime is Python/PyTorch/NumPy with declared versions; ONNX is optional and
not implied. Changing precision/compilation requires reevaluation of that exact
artifact/config. Inference is image -> compact student -> mask only.

Manifest-driven inference writes uint8 masks on native TZYX grids with verified
axes and affine metadata; it does not retain dataset-wide probability volumes.
All prediction outputs are frozen before reference masks are scored. Prepared
inputs must already be normalized; raw preprocessing is explicit image-only per-cine
.5/99.5 percentile normalization. Do not invent missing physical geometry.

Reference and compact benchmarks share device/environment, cases, precision,
batch size, warmup, synchronization and IO/cache policy. Separate model-load/cold
start from steady workload; include preprocessing, transfers, forward and writing
masks in E2E time. Peak inference memory excludes training-only optimizer/teacher
state; account for any child workers. Keep memory bounded as case count grows.
Report teacher training, cache generation, student training and proposal/audit
costs separately, both total and marginal; an unknown teacher cost is unknown,
not zero. Equal-update and equal-wall-time comparisons are distinct budgets and
do not authorize extending a failed run or automatically launching every arm.

## Execution commands and verification


The commands below are a handoff for the operator; no patient command was run in
this delivery. Use a fresh output directory for every run. Install the repository
requirements in your chosen environment; the tested software environment was
Python3.11.16, PyTorch2.14.0 CPU, NumPy and psutil. CUDA/PRO4000 is NOT_RUN.
The deploy loader itself needs Python, PyTorch and NumPy; raw NIfTI IO also needs
nibabel and the benchmark needs psutil. No DINO assets are needed by the compact
export. The historical reference still requires its pinned local DINO source and
initialization file to reconstruct the original architecture; those are included
in reference cold-load time, never required by compact inference.

Set these paths to existing operator-owned artifacts. Keep the historical TRAIN
manifest bytes unchanged; if its relative data paths are unavailable, restore the
declared layout rather than silently changing the split hash.

```bash
export TRAIN_MANIFEST=/workspace/data/acdc/train_l8.json
export TRAIN_SHA=e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4
export TEACHER_CKPT=/root/self-audit-pilot-artifacts/baseline/final
export HISTORICAL_SOURCE=/workspace/Self-Audit-history-0f54455b
export RUN_ROOT=/workspace/compact-acdc-l8
mkdir -p "$RUN_ROOT"
git worktree add --detach "$HISTORICAL_SOURCE" 0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca
python scripts/preflight_compact_teacher.py --use-historical-baseline \
  --historical-source "$HISTORICAL_SOURCE" \
  --student-manifest "$TRAIN_MANIFEST" --student-manifest-sha256 "$TRAIN_SHA"
python scripts/build_compact_teacher_cache.py --data-manifest "$TRAIN_MANIFEST" \
  --data-sha256 "$TRAIN_SHA" --use-historical-baseline \
  --checkpoint-dir "$TEACHER_CKPT" --historical-source "$HISTORICAL_SOURCE" \
  --output-dir "$RUN_ROOT/teacher-cache" --device cuda --batch-size 8 --precision float32
```

The preflight command checks metadata/source, and explicitly does **not** certify
physical teacher weights. The cache builder verifies checkpoint manifest, payload,
source closure, selected state and label lineage before using that teacher. It
writes probability files incrementally and pins each one in `CACHE_MANIFEST.json`.
Input mmaps retain at most8 handles; probability mmaps retain at most16. These
limits bound retained files, with a live volume/frame allowed to finish before its
mapping is released. Peak memory still depends on the largest native case.

For a different canonical full-EMA teacher, provide `--teacher-identity` and its
`--teacher-identity-sha256` with `--checkpoint-dir`. Its identity must pin source,
checkpoint, payload, selected state, exact label union and a measured quality JSON
via `quality_receipt: {path, sha256}`. That quality JSON must contain matching
`checkpoint_manifest_sha256`, `inference_state`, and `class_first_macro4`.
K1/2/4 templates require matching-budget teachers and explicit preselected TRAIN
manifests; they do not select labels or launch a sweep.

Bind the selected template to the verified cache. The full template defaults to
plain KD, 10000 total updates including1000 warmup, Lbatch8/Ubatch8, crop160.
This is a declared compact recipe, not a claim that random U-Net initialization
has the same convergence as pretrained DINO. First inspect a short supervised fit
on budgeted L and a bounded profile; choose the checkpoint rule before full runs.
Default quality reporting uses the final fixed-step export, with no TEST-based
checkpoint selection. Do not increase steps/K after observing a failure.

```bash
export CONFIG_TEMPLATE=configs/compact/acdc_l8_full.json
python - <<'PY'
import hashlib, json, os
from pathlib import Path
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
root = Path(os.environ['RUN_ROOT'])
c = json.loads(Path(os.environ['CONFIG_TEMPLATE']).read_text())
m = json.loads(Path(os.environ['TRAIN_MANIFEST']).read_text())
assert sha(os.environ['TRAIN_MANIFEST']) == os.environ['TRAIN_SHA']
assert c['data']['manifest_sha256'] == os.environ['TRAIN_SHA']
assert set(c['data']['labeled_ids']) == {r['id'] for r in m['labeled']}
cache_path = root/'teacher-cache/CACHE_MANIFEST.json'
cache = json.loads(cache_path.read_text())
assert cache['train_manifest_sha256'] == os.environ['TRAIN_SHA']
assert cache['teacher_identity']['verified'] is True
c['teacher'].update(cache_path=str(cache_path.resolve()), cache_sha256=sha(cache_path),
                    teacher_identity=cache['teacher_identity'])
target = root/'locked.json'
with target.open('x') as f: json.dump(c, f, indent=2); f.write('\n')
# A separate cost probe deliberately uses a short, explicitly different warmup.
profile = json.loads(json.dumps(c))
profile.update(steps=4, warmup_steps=2)
with (root/'profile.json').open('x') as f: json.dump(profile, f, indent=2); f.write('\n')
print('locked config SHA256:', sha(target))
PY
python train.py --config "$RUN_ROOT/profile.json" --data "$TRAIN_MANIFEST" \
  --data-sha256 "$TRAIN_SHA" --out "$RUN_ROOT/profile" --device cuda
# Operator opt-in only, after the small-L fit/profile and teacher checks:
python train.py --config "$RUN_ROOT/locked.json" --data "$TRAIN_MANIFEST" \
  --data-sha256 "$TRAIN_SHA" --out "$RUN_ROOT/plain-kd" --device cuda --allow-full-run
```

For a supervised fit use the same bound config with `method="supervised"` and a
declared short budget. The full scope is encoder+decoder+head; `TRAIN_LOG.jsonl`
records actual per-group deltas and ownership, not just requires_grad. To compare
scalar/joint, explicitly copy the locked config and change only `method` to
`scalar_guard` or `joint_guard`. Same initial-state and per-step data-schedule
hashes must match. Keep the same steps/batches/L/U/temperature/denominator/trial
cap. The supervisor-only arm still samples U images to preserve schedules but
does not optimize on them. For an equal-time comparison, predeclare
`equal_wall_time={"enabled":true,"budget_seconds":B,"max_steps":N}` in every arm,
with N no larger than its locked steps. The runner finishes the current atomic
transaction and reports any resulting time overshoot; it never extends N.

`REPORT.json` and `TRAIN_LOG.jsonl` record nonzero accepts, rejection reasons,
actual audit replays/slices, module deltas, synchronization, timings, update counts
and training GPU peaks. Teacher generation telemetry is retained separately from
cache lookup cost. Supply an evidence-backed `teacher.source_training_cost` if
known; otherwise it stays NOT_KNOWN. Compact resume is currently rejected by the
CLI; a saved checkpoint is exportable but is not advertised as an exact-resume
contract. Existing native/automatic resume modes are unchanged.

Export and use the same artifact for all downstream commands:

```bash
export CHECKPOINT_SHA=$(python -c 'import hashlib,os; from pathlib import Path; print(hashlib.sha256((Path(os.environ["RUN_ROOT"])/"plain-kd/final/MANIFEST.json").read_bytes()).hexdigest())')
python scripts/export_compact_student.py --checkpoint "$RUN_ROOT/plain-kd/final" \
  --checkpoint-sha256 "$CHECKPOINT_SHA" --out "$RUN_ROOT/export"
export EXPORT_SHA=$(python -c 'import hashlib,os; from pathlib import Path; print(hashlib.sha256((Path(os.environ["RUN_ROOT"])/"export/MANIFEST.json").read_bytes()).hexdigest())')
```

Evaluation inputs use native TZYX, canonical classes and a verified physical
affine. If the historical prepared NPY manifest lacks affine, attach geometry
from a hash-pinned image-only raw-NIfTI mapping. A sources JSON has schema
`self_audit.compact_geometry_sources.v1` and
`patients: {"patientID": {"path":"cine.nii.gz", "sha256":"..."}}`, covering
exactly the evaluation patients. The bridge verifies normalized voxel
correspondence and millimeter geometry, keeps image/mask/split identities, and
never opens masks. Missing raw images/geometry remains BLOCKED.

```bash
python scripts/attach_compact_geometry.py --manifest /workspace/data/acdc/test50.json \
  --manifest-sha256 22a9ea63821f266bd59e810bcd8f9a6958e6f92cc363e4a360b5ab9d4eb1e25c \
  --sources /workspace/data/acdc/raw-geometry-sources.json --sources-sha256 RAW_SOURCES_SHA \
  --out "$RUN_ROOT/test50-native.json"
export EVAL_MANIFEST="$RUN_ROOT/test50-native.json"
export EVAL_SHA=$(python -c 'import hashlib,os; from pathlib import Path; print(hashlib.sha256(Path(os.environ["EVAL_MANIFEST"]).read_bytes()).hexdigest())')
python scripts/infer_compact_student.py --export "$RUN_ROOT/export" --export-sha256 "$EXPORT_SHA" \
  --manifest "$EVAL_MANIFEST" --manifest-sha256 "$EVAL_SHA" \
  --out /workspace/compact-predictions --device cuda --batch-size 8
python scripts/evaluate_compact_student.py --export "$RUN_ROOT/export" --export-sha256 "$EXPORT_SHA" \
  --manifest "$EVAL_MANIFEST" --manifest-sha256 "$EVAL_SHA" \
  --train-manifest "$TRAIN_MANIFEST" --train-manifest-sha256 "$TRAIN_SHA" \
  --out /workspace/compact-evaluation --device cuda --batch-size 8
python scripts/benchmark_compact_student.py --compact-export "$RUN_ROOT/export" \
  --compact-export-sha256 "$EXPORT_SHA" --manifest "$EVAL_MANIFEST" --manifest-sha256 "$EVAL_SHA" \
  --baseline-checkpoint "$TEACHER_CKPT" \
  --baseline-checkpoint-sha256 03c1730f88941ada898bed7fcf1b9f13cdd8f08bcce4390539fec665c82e97de \
  --baseline-state student --historical-source "$HISTORICAL_SOURCE" \
  --out "$RUN_ROOT/benchmark.json" --device cuda --batch-size 8 --warmup 2 --repeats 5
export QUALITY_SHA=$(python -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("/workspace/compact-evaluation/METRICS.json").read_bytes()).hexdigest())')
export BENCHMARK_SHA=$(python -c 'import hashlib,os; from pathlib import Path; print(hashlib.sha256((Path(os.environ["RUN_ROOT"])/"benchmark.json").read_bytes()).hexdigest())')
python scripts/assess_compact_deployment.py --export "$RUN_ROOT/export" --export-sha256 "$EXPORT_SHA" \
  --quality /workspace/compact-evaluation/METRICS.json --quality-sha256 "$QUALITY_SHA" \
  --benchmark "$RUN_ROOT/benchmark.json" --benchmark-sha256 "$BENCHMARK_SHA" \
  --out "$RUN_ROOT/ACCEPTANCE.json"
```

The evaluation output must be outside input artifact/data directories. Masks are
native uint8 NPY volumes with affine/axes in the frozen inventory. The reference
benchmark verifies and extracts one selected state before starting its measured
inference subprocess, so optimizer tensors are absent from measured runtime.
One benchmark volume means one complete native4D cine; both models run all frames
and slices, with float32, no TTA, warm OS filesystem cache and actual NPY writes.
Report values for a different volume definition separately. The assessor checks
the locked historical split (or its geometry-enriched child), complete frame
coverage and the same exported weights, and exits2 on BLOCKED or NO_GO.

Joint4+4 is an explicitly non-launchable protocol template until its manifest,
budget union, teacher lineage and ACDC gate are locked. The inherited shared
loader already validates namespaces, mapping receipts, quarantine, duplicates
and explicit50/50 domain scheduling. Compact evaluation reports per dataset;
neither joint training nor external-domain generalization is claimed here.

For software-only reproduction:

```bash
python scripts/check_repo.py
python -m pytest -q
python scripts/smoke_compact_pipeline.py --out /tmp/compact-software-smoke --device cpu
```

See `reports/compact_delivery/DELIVERY.md`, `VALIDATION.json`, and
`SOFTWARE_SMOKE.json` for actual root-reviewed receipts. They do not establish
Macro4≥.87, patient inference speed, target-GPU memory savings, audit benefit or
joint superiority. Missing teacher/patient/GPU evidence leaves deployment BLOCKED.
