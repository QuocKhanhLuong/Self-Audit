# Self-Audit v3 checkup repair receipt

Base: `e513bc0a4d34ed3a39de2b3fc3c10a362d9216ed`, 05-Oct-2026 checkup.
Scope: the main `self_audit_pseudolabel` path and its existing student dependency.
SGSCN, CUTS, supervised references, and their recipes are unchanged.

This changes the seed and routing recipes and introduces stricter artifact contracts.
Regenerate teacher freezes and student checkpoints. Legacy generic schema-3 artifacts
remain integrity-readable, but cannot enter role-specific training/evaluation consumers.
No confidence threshold was lowered, and no mask, scribble, or mask-trained teacher
was added to training. Independent reference evaluation remains after complete freeze.

| Finding | Implemented correction | Regression evidence |
|---|---|---|
| F01 | Wall and cavity must be connected, internal, adjacent, and share one substantially filled hole; cavity cannot itself have holes. Exterior regions receive no MYO vote. Enclosure contribution is bounded. | Ring class identity and validity; nested, disconnected, and border cases. |
| F02 | Every physical pair contributes symmetrically by per-class maximum. Projection ties abstain from orientation votes; no region-ID anchor. | Every channel permutation on six topology fixtures, including multiple cavities. |
| F03 | Raw-logit cross entropy, detached seed target, accepted-region mask. | Opposite saturated logits at +/-100 give CE 200 and nonzero gradients with correct signs. |
| F04 | Persistent trained-profile cap, inference guard, checkpoint loader, strict runtime cap. | Compact/balanced state/checkpoint round trips; excessive profile rejection; turn/iteration gradient coverage. |
| F05 | Count actual supervised class pixels and patient/frame identities only after successful optimizer updates. Refuse to save when actual foreground is zero. | Foreground exists after batch cap but earlier updates are BG-only; declared vs observed patients remain distinct. |
| F06 | Reject common reference directories/tokens before headers; M&Ms uses a recognized image-name allowlist and rejects unknown layouts. | GT/segmentation/reference names and unknown layout cannot reach `nib.load`. |
| F07 | Strict integer/nonnegative/upper-bound index validation, ED/ES declarations, no fractional coercion. | Fraction, bool, string, negative, out-of-range, and valid one-based ACDC conversion. |
| F08 | Runtime thresholds flow from checkpoint config; CLI epoch omission uses configured default. | Nondefault thresholds change routing; configured epoch default and explicit CLI override. |
| F09 | Explicit train/val/test evaluation kinds and checkpoint-only test export. | Each split's metadata; immutable teacher and student test-export paths. |

## Design risks and remaining evidence limits

| Risk | Handling and limit |
|---|---|
| R01 | RV still abstains without cardinal orientation under default confidence thresholds. Tests fix this policy; export reports class support and orientation per patient. No RV coverage claim. |
| R02 | Default one-neighbor disagreement remains accepted; two valid disagreeing neighbors reject the center. Matrix regression documents this exact policy. No warp/threshold change. |
| R03 | Missing classes are explicitly reported for frozen exports and actual student updates. No arbitrary class minimum or GT-tuned quality threshold was introduced; positive support never certifies quality. |
| R04 | Teacher/student role schemas require artifacts, locked split consistency, native shape/metadata, complete export cohorts, and recomputed pixel statistics. Generic hash validity is insufficient for consumers. Hashes do not prove authenticity against deliberate re-signing. |
| R05 | Student checkpoint pins preprocessing, model config, source files plus AnnotationExpert/DynamicWindow dependencies, environment versions, source manifest, observed coverage, and trained profile. Loader has no silent override. Checkpoint SHA is recorded in receipt and prediction freeze. |
| R06 | New `infer_student_v3.py` and `export_teacher_v3.py` export native XYZT volumes and complete frozen per-slice predictions without optimizer/reference access. Runner with `--train-student` adds student native freeze and independent validation. Final test history remains `NOT_CERTIFIED_BY_CODE`. |
| R07 | ED+ES required for every evaluated patient; result has frame/class/phase rows, phase-macro estimates, phase-pooled patient metrics, reference hashes, frame mapping, and explicit estimand. UNKNOWN remains penalized against reference anatomy. |
| R08 | Consecutive patient batches reduce cache churn; export uses temporary disk-backed arrays and at most three temporal slices per gate. Cache/load/export/scratch/peak-RSS instrumentation added. Full-cine normalization, mapped pages, disk capacity, real-patient throughput, and target-GPU memory remain unmeasured. |
| R09 | Evaluator requires matching spatial units; unknown units require explicit, recorded both-unknown policy. No implicit rescaling. |
| R10 | Adaptive routing is per example using maximum pixel entropy; a small uncertain area cannot be averaged away. Encode-once retained. This is a conservative compute policy, not calibrated anatomy correctness; cost/quality needs measurement. |

## Verification

Final local integrated run:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_self_audit_*.py tests/test_pseudolabel_*.py
python -m compileall -q src scripts tests
git diff --check
```

- **194 passed, 0 failed, 0 skipped**, 6.79 seconds; one existing tensor-to-scalar
  warning in `tests/test_self_audit_core.py`.
- Compile and whitespace checks passed.
- Includes **60 new checkup cases** plus existing regressions, native NIfTI integration,
  synthetic bounded optimizer updates, student/test export, and independent scoring.
- Local stack: macOS arm64, Python 3.11.16, PyTorch 2.14.0, NumPy 2.4.6,
  SciPy 1.17.1, nibabel 5.4.2. This is a portability receipt.
- Canonical Linux CI status is recorded in the publication receipt; no local canonical
  environment or GPU result is implied.
All local model updates in tests use generated arrays/NIfTI fixtures, including explicit
synthetic pseudo-labels. They establish software behavior only. Real-data training,
ACDC/M&Ms accuracy, clinical quality, and RTX 4080 Super latency/VRAM were **NOT RUN**.

Local validation uses the available macOS Python/PyTorch stack, not the pinned Linux
CPU environment. The repository CI uses the hashed canonical lock and its environment
validator. These two receipts must not be conflated.

See [the runbook](pseudolabel_v3_review_runbook.md) for inference/evaluation commands,
reference protocol, regeneration requirements, and resource limitations.
