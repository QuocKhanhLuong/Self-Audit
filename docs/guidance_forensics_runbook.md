# Run the stable-guidance forensics tools

This guide accompanies experimental branch `QuocKhanhLuong/guidance-forensics-20261014`, based on `e4d3269ce244d3d4ca0afdb1c92615f049116911`. It targets run `stable_guidance_acdc_4070_20261013_222918` without retraining or changing its artifacts. It does not authorize another full run.

The implementation and manufactured CPU checks are complete: 120 tests passed. Real-run attribution is still blocked by SSH authentication to machine `4070`; the Downloads review ZIP does not contain the checkpoint or banks. See [run forensics](../reports/guidance_forensics_20261014/RUN_FORENSICS.md), [failure attribution](../reports/guidance_forensics_20261014/FAILURE_ATTRIBUTION.md), and [repair decision](../reports/guidance_forensics_20261014/REPAIR_DECISION.md).

## 1. Use a separate source checkout

On the machine that holds the original job, start in an existing Self-Audit repository and create a **new** worktree. Do not edit or switch the checkout used by an active process.

```bash
git fetch origin
git worktree add --detach ../Self-Audit-guidance-forensics \
  origin/QuocKhanhLuong/guidance-forensics-20261014
cd ../Self-Audit-guidance-forensics
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Use the completed run's Python environment. The read-only tools need Python, PyTorch, NumPy, SciPy and nibabel; tests additionally need pytest. They do not need to download the frozen teacher. Check imports without updating the original environment:

```bash
python -c 'import torch, numpy, scipy, nibabel; print(torch.__version__, numpy.__version__, scipy.__version__, nibabel.__version__)'
python scripts/diagnose_stable_guidance.py --help
```

Set actual paths; the values below are placeholders, not discovered locations. `SELF_AUDIT_JOB` may name the launcher job containing `run/`, or the run directory containing `RECIPE.json`. Choose a diagnostic root outside the original job, bank, student and frozen prediction folders.

```bash
export SELF_AUDIT_JOB=/absolute/path/to/stable_guidance_acdc_4070_20261013_222918
export SELF_AUDIT_DIAG=/absolute/path/to/separate/forensics_20261014
```

Each command below creates its own output directory. Existing destinations are refused, so choose a fresh destination for a repeated check.

## 2. Run the mathematical checks first

```bash
python scripts/diagnose_stable_guidance.py probes \
  --out "$SELF_AUDIT_DIAG/math_checks"
```

Expected exit: `0`; status `MANUFACTURED_MATHEMATICAL_CHECKS_NOT_PATIENT_EVIDENCE`. Output: `DIAGNOSTICS.json`. It demonstrates valid oblique geometry and unchanged coarse loss with changed native argmax. It does not use patients or prove anatomical improvement.

## 3. Inspect the original checkpoint on a bounded TRAIN batch

```bash
python scripts/diagnose_stable_guidance.py audit \
  --run "$SELF_AUDIT_JOB" \
  --max-patients 1 --max-batches 1 \
  --out "$SELF_AUDIT_DIAG/train_one_batch"
```

This uses CPU, performs no optimizer step and never reads target masks. The expected checkpoint digest is pinned to:

```
516e569c9141e2d8edb00b598df5cf12c8a9a4741cf1ae0cca8cbf23d6d03353
```

Do not override it to bypass a mismatch. Required files include RECIPE, effective split, preflight, bank/student freezes and payloads, student history/report, both prediction manifests and their pair lock. The tool verifies bank/student payload hashes before inspecting the bounded batch: **the forward budget is small, but hashing a full bank still requires disk I/O.**

The JSON reports:

- Run-local configuration and available provenance; checkpoint and bank bindings.
- R0/Q/hard-label support, class permissions, entropy, weighted soft-target mass and overlapping veto counts.
- CE, target entropy and KL using the actual loss convention; feature/semantic gradient norms and cosine on the shared encoder.
- Common-grid teacher fidelity, native argmax and gated support, and disagreement with the predetermined coarse-to-native probability control.
- Original training history, recorded semantic-update counter, orientation metadata/header availability, and the exact T/Z batch indices.

The batch is the first eligible block in fixed lexicographic T/Z order, falling back to an inactive block when none is eligible. Its activity is not an unbiased population estimate. Teacher agreement is fidelity, not anatomical accuracy. Native comparisons cover the selected batches, not the entire held-out cohort.

Expected exits: `0` for a completed bounded audit; `2` for `BLOCKED`, with missing files or the exact error in JSON. Individual header checks can also be BLOCKED inside an otherwise completed checkpoint audit; inspect nested statuses.

After the effective cohort is verified, a whole-TRAIN support and activation-schedule review can use the same command with a **fresh** output path and `--max-patients 80 --max-batches 1`. This increases I/O and CPU forwards. It replays the fixed-bank batch-eligibility schedule and compares it with the recorded update counter; it does not reconstruct historical gradients or every epoch's CE/KL.

## 4. Recompute exposed-VAL metrics from frozen counts

Point to the **actual recovered evaluation directory** containing `FROZEN.json` and `METRICS.json`; verify its location rather than assuming it was renamed to `evaluation_val`.

```bash
python scripts/diagnose_stable_guidance.py metrics \
  --evaluation /absolute/path/to/run/evaluation_val_verified_grid \
  --out "$SELF_AUDIT_DIAG/val_counts"
```

Outputs: `DIAGNOSTICS.json` and `DIAGNOSTICS.csv`. Reduction is frame mean within patient → patient mean within class → RV/MYO/LV mean. Undefined denominators remain null. Overall and aligned/verified-recovery strata are retained; patients are not dropped to improve scores. This mode rejects TEST evaluations and reads no reference arrays.

Count verification does **not** reexecute original frame/cine voxel-equality proofs. It also does not generate VAL teacher outputs, fit thresholds, verify every TEST prediction payload, or recover source vote histories from R0. These are separate outstanding checks.

## 5. Geometry repair is opt-in and requires a new bank

The existing `prepare` stage accepts:

```text
--orientation-policy cardinal_v1    # unchanged default
--orientation-policy world_left_v2  # continuous coded world geometry
```

Only after the original-artifact diagnosis justifies a geometry comparison, run `prepare` into a fresh bank directory with `world_left_v2`. Use the original effective split, seed, resolved config, image root, tile budget, and exact frozen SSL teacher/checkpoint/config from RECIPE. Do not copy current checked-in defaults into the experiment. With relocated artifacts, verify identity before changing path references.

For a matched comparison, prepare both arms on the same predeclared TRAIN patients and frozen teacher, changing only the orientation policy. Version both new banks and preserve the original failed bank. This stage invokes the teacher and is **not part of the read-only commands above**.

The new bank manifest records `orientation_policy`; each image records the pixel-center transform, continuous left covector, sform/qform provenance and abstention reason. Correct geometry does not certify that a candidate is RV. The full-run wrapper/preflight has not silently acquired this policy: use the explicit prepare option.

## 6. Training diagnostics and remaining execution boundary

The existing train stage accepts `--diagnostics` to add CE/H/KL, active fraction and per-class mass to each epoch's `training.json`, without changing its objective. Its trainer is still **epoch-based**, not a 32-step pilot runner. Do not translate the proposed fixed-step pilot into an arbitrary epoch count or launch another 50-epoch run.

No native-resolution objective/export repair or graph replacement has been selected. The coarse-to-native projection remains diagnostic only. Real pilot and full-run decisions remain governed by [REPAIR_DECISION.md](../reports/guidance_forensics_20261014/REPAIR_DECISION.md).

## 7. Regression check and safe evidence handoff

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_self_audit_stable_guidance.py \
  tests/test_self_audit_guidance_full.py \
  tests/test_guidance_final_metrics.py \
  tests/test_guidance_wandb_runner.py \
  tests/test_guidance_forensics.py
```

The recorded local result is 120 passed, zero failures/errors/skips. Tests include manufactured tiny optimizer steps; they are not an ACDC pilot. See the checked-in [verification receipt](../reports/guidance_forensics_20261014/VERIFICATION.json).

Share reviewed scalar diagnostics, confusion counts, hashes, resolved config and epoch history. Inspect diagnostic JSON before publication: copied run recipes can contain local paths. Do not upload patient NPZ/NIfTI arrays, masks, weights or credentials. Publication of this experimental branch does not merge it into `main` or change the NO-GO full-run decision.
