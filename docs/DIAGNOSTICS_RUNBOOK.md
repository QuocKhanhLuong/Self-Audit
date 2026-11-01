# Context and gradient bridge diagnostics

This is an offline diagnostic interface. It never updates the student or teacher,
starts a training job, selects a patient-specific correction using labels, or
changes the historical training recipes. The implementation task runs only
manufactured fixtures. Patient results must be produced separately by the user.

The complete input brief and bridge-v2 review are preserved under
[`diagnostics_design/`](diagnostics_design/CODEX_PROMPT_SELF_AUDIT_DIAGNOSTICS.md),
with their SHA256 values in `diagnostics_design/SOURCE_RECEIPT.json`.

## Identities and decisions to lock

The historical profile requires the physical raw-student B8 payload
`0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6`
and wide plain-KD export
`32a35c1fc3b83d57d083de4ed17ab80bda6ba38a8ebc50133e5aede17ab00380`.
Receipt JSON is evidence about an artifact, not its weights. B52, EMA state,
the reproduced B8, and compact checkpoints from other schedules cannot replace
this pair. The exposed historical VAL18 wide reference is Macro4
0.777413392930114 and foreground 0.7060258837373868. It is a reproduction
check after prediction freeze, not a new result of this implementation.

Required primary rho, sensitivity grid, numerical/gradient tolerances, practical
absolute foreground-Dice improvement and statistical decision rules must be
filled from TRAIN choices before locking B. Unfilled scientific fields are
intentional blockers in a template, not defaults to tune on VAL. All fallback
patients stay in the cohort. Invalid numerics or a disconnected bridge block
scientific interpretation; they are not evidence against the hypothesis.

## Execution boundaries

Use a new output round directory. The required sequence is:

1. Preflight verifies checkpoint bytes, source closure, lineage, manifest and
   image identities without opening reference mask payloads.
2. Lock A, then produce both teacher and compact context predictions.
3. Fit or load a hash-verified adapter P using TRAIN images only. Fit has a
   separately declared fixed budget and objective and never runs during predict.
4. Lock B, then produce all prespecified D0–D5 predictions/logit shards.
5. Score A, score B, then render the report. All output shards and both locks
   are verified before first GT access.

The round records `protocol-A.lock.json`, `protocol-B.lock.json`,
`predictions-A.frozen.json`, `predictions-B.frozen.json` and
`VAL_EXPOSURE.json`. These are append-once records with content hashes. Source,
input, weight or prediction changes require a new round. Do not edit a lock to
make a check pass. Labels may be opened only by a scoring command.

If a prerequisite makes B impossible, explicitly record the concrete B blocker
when scoring A alone. That round is then exposed and cannot acquire a late B
lock. A negative A result does not constitute a B blocker: B still runs under
its own fixed protocol.

## Interpretation

Context uses spatial neighboring Z slices within the same time frame, with edge
replication and no separate normalization for lambda views. Lambda one is an
identity path and must match original inference. Prediction changes, rescue,
harm and false positives are distinct outputs; an empty opportunity denominator
is undefined rather than a successful zero-percent estimate.

The gradient tap is the compact encoder bottleneck. Native student skips are
detached and fixed. The adapter maps the bottleneck into a declared teacher
suffix beginning before `refine[0]`; teacher raw-detail is zero in the primary
bridge. This suffix is not the entire teacher decoder. Its Jacobian includes
the adapter and alignment, not only pretrained teacher layers. Every arm is
decoded by the native student decoder for its reported segmentation.

The D3 negative control shuffles D2's scalar coordinates (channels and spatial
sites) within valid feature support and matches its displacement norm and
fallback. A singleton scalar is an explicitly logged identity control, not an
extra fallback. D4 and D5 exchange output cotangents. Cotangents
are obtained by autograd from the exact shared KD objective with one application
of temperature/normalization. Near-zero gradients preserve D0 for that arm;
nonfinite values have a distinct invalid status.

Historical scoring averages eligible frame Dice within patient/class, then
patients within class, then classes. Absent GT excludes that frame/class Dice
but does not hide false positives. Ignore 255 and native padding rules remain
explicit. Paired patient bootstrap and leave-one-patient-out recompute this
class-first metric with the same cohort and paired resamples for both primary
contrasts.

`GO_TO_R_PILOT` requires D2 to beat both D0 and D1 on native foreground Dice
under the locked practical, uncertainty and leave-one-out criteria. Native KD
need not favor D2. D4/D5 explain the mechanism and are not extra GO conditions.
`NO_GO`, `INCONCLUSIVE` and `BLOCKED` remain separate. No gate starts a job.

Any later R study would compare R0 = supervised + KD, R1 = the same R plus a
detached self-gradient target, and R2 = the same R plus a detached bridge-gradient
target, with matched graph, initialization and budget. No R trainer is included
in this delivery. A fixed-checkpoint negative result does not prove that all
co-adaptation schemes are impossible.

## Server commands

Run from this branch's repository root in an environment satisfying
`requirements.txt`. Use a private checkout; do not update a checkout used by an
active training process. These commands are provided for later execution and
were **not** run against patient data during implementation.

Prepare two config copies outside the source checkout. Resolve the historical
payload directories, source closures and manifests without changing their
expected hashes. Use **absolute paths** in both configs: A resolves relative
paths from the repository working directory, while B resolves them from its
config directory. Set the same device and parity rules. A is native FP32 with
batch one; B's chunk batch is explicitly locked and does not change the cohort.

```bash
mkdir -p ../diagnostics-config
cp configs/context_diagnostic_historical_val18.json ../diagnostics-config/context.json
cp configs/gradient_bridge_historical_val18.template.json ../diagnostics-config/bridge.json
```

Before proceeding, fill all required `null` fields in `bridge.json`: primary rho, the finite prespecified rho
list (including primary), norm epsilon, minimum absolute foreground Dice delta,
bootstrap resample count/seed, confidence level and minimum defined fraction.
The historical KD values are pinned from the wide export's manifest: temperature
1, coefficient 1, confidence 0.7, unweighted loss and valid-confident-pixel count
denominator. Historical mode rejects changes to these values. Null confidence
is **not** an instruction to disable masking; `class_weights: null` alone means
the unweighted current loss. Do not choose scientific fields after seeing VAL outcomes.
The checked-in fit-P budget is a new declared diagnostic choice, not a historical
training recipe: deterministic patient-balanced ridge regression, bilinear
alignment followed by a 1x1 affine projector, bounded TRAIN sources/pixels. Fit
records the actual selected patients, sources and pixel counts; P is never fit
during prediction.

After resolving and reviewing the files, use one new output directory outside
the checkout. All invocations fail closed on changed source/config/assets.

```bash
DIAG_A_CONFIG="$(realpath ../diagnostics-config/context.json)"
DIAG_B_CONFIG="$(realpath ../diagnostics-config/bridge.json)"
DIAG_ROUND="$(realpath ..)/diagnostics-round-$(date -u +%Y%m%dT%H%M%SZ)"
DIAG_A_SHA="$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$DIAG_A_CONFIG")"
DIAG_B_SHA="$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$DIAG_B_CONFIG")"

python scripts/run_context_diagnostic.py preflight --config "$DIAG_A_CONFIG" --config-sha256 "$DIAG_A_SHA" --round-dir "$DIAG_ROUND"
python scripts/run_context_diagnostic.py lock --stage A --config "$DIAG_A_CONFIG" --config-sha256 "$DIAG_A_SHA" --round-dir "$DIAG_ROUND"
python scripts/run_context_diagnostic.py predict --stage A --config "$DIAG_A_CONFIG" --config-sha256 "$DIAG_A_SHA" --round-dir "$DIAG_ROUND"

python scripts/run_gradient_bridge_diagnostic.py preflight --config "$DIAG_B_CONFIG" --config-sha256 "$DIAG_B_SHA" --round-dir "$DIAG_ROUND"
python scripts/run_gradient_bridge_diagnostic.py fit-p --config "$DIAG_B_CONFIG" --config-sha256 "$DIAG_B_SHA" --round-dir "$DIAG_ROUND"
python scripts/run_gradient_bridge_diagnostic.py lock-b --config "$DIAG_B_CONFIG" --config-sha256 "$DIAG_B_SHA" --round-dir "$DIAG_ROUND"
python scripts/run_gradient_bridge_diagnostic.py predict --config "$DIAG_B_CONFIG" --config-sha256 "$DIAG_B_SHA" --round-dir "$DIAG_ROUND"

python scripts/run_context_diagnostic.py score --stage A --config "$DIAG_A_CONFIG" --config-sha256 "$DIAG_A_SHA" --round-dir "$DIAG_ROUND"
python scripts/run_gradient_bridge_diagnostic.py score --config "$DIAG_B_CONFIG" --config-sha256 "$DIAG_B_SHA" --round-dir "$DIAG_ROUND"
python scripts/run_gradient_bridge_diagnostic.py report --round-dir "$DIAG_ROUND"
```

Execute these as successive successful commands (`set -e` in a shell runner).
Stop at a failed stage; do not score partial or modified predictions. A genuine
missing B prerequisite permits the separate A-only command below. Replace the
reason with the actual evidence; a negative context result is not a prerequisite
failure.

```bash
python scripts/run_context_diagnostic.py score --stage A \
  --config "$DIAG_A_CONFIG" --config-sha256 "$DIAG_A_SHA" \
  --round-dir "$DIAG_ROUND" \
  --b-blocked-reason 'B BLOCKED: exact prerequisite and evidence here'
```

`A/` contains context frame/patient/class counts, Dice and transition tables.
`B/` contains bounded logit shards, gradient diagnostics, frame/patient/class
metrics and the locked primary gate. Root `REPORT.json` and `REPORT.md` link both
stages. Preserve locks, freezes, exposure ledger, fit-P receipt and projector
alongside outputs. A reproduction mismatch remains a reported integrity limit;
it does not authorize changing the reference or tuning a later protocol.

## Local software verification

```bash
python scripts/run_context_diagnostic.py --help
python scripts/run_gradient_bridge_diagnostic.py --help
python scripts/run_gradient_bridge_diagnostic.py fixture-smoke --round-dir /tmp/self-audit-diagnostic-smoke-unique
python scripts/check_repo.py
python -m pytest -q
```

Use fixture smoke only on a new temporary output directory. Fixture weights and
arrays are generated locally and explicitly marked synthetic; they never stand
in for a historical payload in the patient profile. Exact executed commands,
test counts and output hashes are recorded in the delivery verification report.
The combined `fixture-smoke` command launches all ten actual CLI stages in order,
using two synthetic TRAIN and two synthetic development patients with three
distinct Z slices. Its sibling `_cli_smoke.json` records each argv, exit code,
stdout/stderr, elapsed time and fixture hashes. Choose a new path on every run.
