# Fully automatic head pilot runbook

Software pilot only. Patient efficacy, novelty and PRO4000 24GB measurements are
**NOT RUN**. See [contract](AUTOMATIC_PILOT_CONTRACT.md) and
[review](../reports/automatic_pilot/INTEGRATION_REVIEW.md). The only trainer remains
`train.py`, with opt-in schema `self_audit.automatic.v1`.

## Contract and matrix

Deploy raw NativeStudent. Freeze the full feature path during head adaptation.
The guard protects only the extra U displacement relative to the ordinary
supervised reference theta_sup. Rejected trials perform no live writes. After
the final student state, EMA runs exactly once, including on rejection. No
audited direction passes through AdamW. No certificate extends to EMA or query.

The runner selects all declared TRAIN-L patients as fixed complete frame support.
Support and prior checkpoint target labels count in the declared budget; K=1
reuse is training data. No U/query/VAL/TEST mask enters learner or guard. Arms
share one SHA-pinned support-only warm-start, schedules and normalization; each
owns its teacher trajectory.

| Arm | Extra step |
|---|---|
| scalar_brier / joint_brier | common scalar / joint Brier proposer and finite Brier guard |
| scalar_dice / joint_dice | same plus finite hard-Dice and absent-FP guard |
| frozen | no student update |
| support_only | supervised head only |
| plain_head_u | unguarded pseudo-label head update |
| extra_supervised | second supervised update on the same scheduled support slice |
| smaller_lr | configured pseudo LR 0.0002, applied once; not a tuned winner |
| full_ema | trainable decoder/window/head continuation; encoder follows NativeStudent's frozen-backbone contract |

Scalar uses alpha=s*1, a_c=sum_ij A_cij, b'_c=sum_i b_ci, ridge_scalar=m*ridge.
Brier weights balance positives/negatives in a whole frame, average frames within
patient, then patients. Multiclass logits argmax produces volume confusion before
frame/patient Dice reductions. Primary endpoints are patient macro per RV/MYO/LV.
Absent Dice is undefined; absent FP uses a fixed observed-voxel denominator.
Ignore 255/padding has no mass. Cached screening is followed by actual finite
logit predicates; numerical allclose alone cannot protect argmax ties.

## Inputs and target hardware

Use existing PyTorch/NumPy/SciPy and repository dependencies, pinned source and
pre-existing authorized DINO source/weights. No downloads or license acceptance.
Images are normalized TZHW arrays. Training uses a hashed self_audit.data.v2
manifest. A model override JSON contains:

```json
{
  "backbone": "dinov3_convnext_tiny",
  "source_repo": "/existing/dinov3-checkout",
  "source_commit": "<exact commit>",
  "weights": "/existing/weights.pth",
  "weights_sha256": "<SHA-256>"
}
```

PRO4000 24GB is the target, not measured hardware evidence. Record actual device,
driver, PyTorch/CUDA versions and available VRAM before running. Defaults bound
native support to 10 million voxels, padded cache to 2 GiB and estimated controller
workspace to 4 GiB. Estimates do not guarantee total VRAM usage. Start with
bounded profiling and inspect measured allocated/reserved peaks.

## Commands

Use new output directories outside input/checkpoint directories. Replace all
placeholder hashes/paths with existing artifacts.

```bash
python scripts/run_automatic_pilot.py --validate-configs

# Manufactured nonempty-candidate smoke; no efficacy interpretation.
python scripts/run_automatic_pilot.py --mode smoke --arms all --steps 2 \
  --pseudo-threshold 0 --out /tmp/pilot-smoke-new

# Creates ONLY the shared support-only warm-start.
python scripts/run_automatic_pilot.py --mode warm_start --steps 8 \
  --data /inputs/train.json --data-sha256 <train_sha> \
  --model-config /inputs/model.json --device cuda --out /runs/warm-start-new

# Bounded equal-update profile.
python scripts/run_automatic_pilot.py --mode profile --steps 8 --arms all \
  --data /inputs/train.json --data-sha256 <train_sha> --device cuda \
  --warm-start-checkpoint /runs/warm-start-new/warm_start_support_only/final \
  --warm-start-sha256 <warm_manifest_sha> --out /runs/equal-update-new

# Bounded equal-time profile; reaching the safety cap first is INCOMPLETE.
python scripts/run_automatic_pilot.py --mode profile --comparison equal_time \
  --time-budget 30 --max-steps 32 --steps 8 --arms all \
  --data /inputs/train.json --data-sha256 <train_sha> --device cuda \
  --warm-start-checkpoint /runs/warm-start-new/warm_start_support_only/final \
  --warm-start-sha256 <warm_manifest_sha> --out /runs/equal-time-new

# Full native raw-student export. Omit --inference-only for offline scoring.
python scripts/evaluate_automatic.py \
  --checkpoint /runs/equal-time-new/joint_dice/final \
  --checkpoint-sha256 <adapted_manifest_sha> \
  --data /inputs/query.json --data-sha256 <query_sha> \
  --device cuda --out /runs/query-joint-new --inference-only
```

No long patient run is authorized in this delivery. Longer runs retain an explicit
--allow-full-run switch for separate authorization. Resume through train.py using
--resume and --resume-sha256 (MANIFEST hash), a new output directory, and identical
config/data; only target steps and save interval may change. Optimizers, local
NumPy generator, Torch/Python RNG, counters, support identity and teacher restore.
Local exact-resume evidence is CPU/manufactured, not CUDA parity.

## Costs and offline decision

REPORT.json measures synchronized step p50/p95, throughput, transaction acceptance
separately from candidate counts, actual nonzero displacements, support margins,
VRAM and timing components. CPU VRAM is NOT_MEASURED. Empty proposals do not imply
100% acceptance.

Marginal phase cost includes load, support extraction, adaptation and checkpoint
write. Cold phase cost sums warm-start, adaptation and full native export once.
Phase clocks exclude interpreter/import startup; matrix subprocess wall times
are also reported. Missing export or unmeasured external warm-start leaves cold
total INCOMPLETE. Undefined values never become zero.

Use fresh trajectories for compute comparisons. Numerical resume restores exact
state, but checkpoint cumulative clocks stop before writing that checkpoint.
Resumed timing therefore omits that prior write boundary: cold cost remains
INCOMPLETE and offline matching rejects resumed timing as equal-compute evidence.

Equal-time includes startup/extraction, stops at transaction boundaries and reports
overshoot. Offline matching additionally requires positive updates and actual
phase times within a fixed 5% spread. Equal-update alone does not establish equal
compute. Full predictions and recipe are hashed before reference scoring;
prediction and scoring coverage are separate. Development VAL is not confirmation.

After scoring at least joint_dice, scalar_dice, smaller_lr and extra_supervised,
prepare a pinned evaluation index (paths relative to index or absolute):

```json
{
  "schema": "self_audit.automatic_evaluation_index.v1",
  "matrix_report": {"path": "/runs/equal-time-new/summary.json", "sha256": "<sha>"},
  "evaluations": {
    "joint_dice": {"directory": "/runs/query-joint", "metrics_sha256": "<METRICS sha>"},
    "scalar_dice": {"directory": "/runs/query-scalar", "metrics_sha256": "<sha>"},
    "smaller_lr": {"directory": "/runs/query-smaller", "metrics_sha256": "<sha>"},
    "extra_supervised": {"directory": "/runs/query-extra", "metrics_sha256": "<sha>"}
  }
}
```

```bash
python scripts/run_automatic_pilot.py --mode evaluate \
  --evaluation-index /runs/evaluations.json --evaluation-index-sha256 <index_sha> \
  --report-path /runs/decision-new.json
python scripts/check_repo.py
python -m pytest -q
```

Collation checks pinned matrix/report/checkpoint/prediction chains, shared
initialization/configuration, lineage, query cohort and complete coverage.
Missing primary classes or unmatched compute are INCONCLUSIVE. Research GO
requires joint_dice > scalar_dice, joint_dice > smaller_lr, and joint_dice >
extra_supervised on independent real queries at matched compute. These point
contrasts are a pilot rule, not statistical or clinical certification. Synthetic
runs always return NO_GO_SOFTWARE_ONLY.

Coordinator receipts and source/manufactured-weight hashes are under
reports/automatic_pilot/. Patient efficacy, GPU performance and novelty remain
unmeasured.
