# Review Report

## Summary Judgment

**Ready for the stated Phase A+B CPU/software scaffolding scope.** The final isolated verifier reports `PASS_CPU_SOFTWARE_ONLY` with 88 tests, zero failures/errors/skips; all six reference operator modules remain byte-identical; and no baseline/ablation files were modified. The float32 projection limitation is preserved and explicitly separated as a reference deviation; the supported guarded path checks actual finite-step losses. No scientific or medical readiness claim is supported.

## Critical Issues

None found in the reviewed operator source that invalidate the manufactured test suite or the stated scope limits.

## Major Issues

The following historical findings retain their original evidence for auditability. M2 and M3 are **superseded and resolved** by the final integration/docs/tests below. M1 remains an intentional reference limitation and is not a blocker for the supported guarded-step path.

### M1 — Projected direction can violate a guard after the documented dtype cast (P2)

- **Finding:** `reports/layer_level_upgrade_20261015/reference/layer_upgrade/guarded.py:47-50` computes the active-set solution in float64, then casts it to `candidate.dtype`. The receipt is still labelled `KKT_VERIFIED` and reports a guard inner product from the cast result.
- **Evidence:** Run from the worktree root:

  ```bash
  rtk python - <<'PY'
  import sys, torch
  sys.path.insert(0, "experiments/layer_guided_student_v1")
  from layer_upgrade.guarded import project_direction
  torch.manual_seed(0)
  for k in range(4):
      s = torch.randint(-4, 8, (1,)).item()
      d = (torch.randn(20) * 10**s).float()
      g = torch.randn(4, 20).float()
      out, receipt = project_direction(d, g)
      print(k, s, float(d.norm()), float((g.double() @ out.double()).min()),
            receipt["status"], receipt["min_normalized_inner_product"])
  PY
  ```

  With `torch.manual_seed(0)`, the fourth generated float32 case has `||candidate||=4,965,879`; the returned status is `KKT_VERIFIED`, while `min(g @ result) = -0.1379108540713787` and `receipt["min_normalized_inner_product"] = -0.03282279148697853`.
- **Why it matters:** The value actually returned to a float32 caller is not feasible by a material margin. This does not invalidate the supported `guarded_sgd_step` measured-loss guarantee, because that wrapper reevaluates the fixed guard losses and objective and can reject or backtrack. Any Phase B integration that treats `KKT_VERIFIED` or the returned direction as a first-order guard certificate is unsound.
- **Recommended fix:** Preserve the reference unchanged and record this exact counterexample in `REFERENCE_DEVIATIONS.md`. Until a separately versioned fix exists, integration must rely on the actual finite-step acceptance and must not advertise the projection receipt as post-cast feasibility. A future fix should recheck feasibility after casting and use a documented fallback or higher-precision return type.

### M2 — Geometry receipt does not constrain the transport transform (P1 for semantic use) — SUPERSEDED

- **Finding:** `integration/geometry.py:65-105` stores `affine_xyzt`, but `_overlap`/`transport_density` at lines 108-136 use only `native_hw`, `teacher_hw`, and a hard-coded `full_native_fov_pixel_edges` declaration. There is no teacher affine, orientation, origin, or source-grid registration check.
- **Evidence:** Two `GridGeometry` objects with identical dimensions and image digest but native affines `diag(1,1,1,1)` and `diag(-1,1,1,1)` produce byte-identical `transport_density` for the same source tensor (`same_transport_despite_different_affine True`, `maxdiff 0.0`). A mirrored or differently cropped source can therefore receive the same transport receipt while semantic locations are wrong.
- **Why it matters:** The current code can be safe only under an external assertion that every source artifact is already registered to the same full-FOV pixel-edge grid. The receipt alone does not establish that. Applying named mass or teacher features across orientation/crop differences would create silent semantic misregistration while all 84 synthetic tests pass.
- **Recommended fix:** Keep the operator/reference unchanged. Before real semantic use, require a source-side geometry/registration receipt (including source affine and transform hash) and fail closed on mismatch; alternatively restrict the API to a visibly verified same-image grid and record that verification. Add a mirrored/cropped-source regression test.

### M3 — Historical-bank adapter has an undocumented `src` import dependency (P2) — SUPERSEDED

- **Finding:** `integration/bank.py:8` imports `self_audit_guidance.io`, but the current experiment test/preflight environment only places the experiment directory on `PYTHONPATH`.
- **Evidence:** From `experiments/layer_guided_student_v1/`, `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -c 'from integration import bank'` raises `ModuleNotFoundError: No module named 'self_audit_guidance'`. The import succeeds only after adding `../../src` to `PYTHONPATH`.
- **Why it matters:** The pytest `conftest.py` currently masks this by inserting `src`, but a direct adapter import or external caller using only the experiment directory fails. The preflight CLI does not exercise the bank adapter, so the 84-test PASS does not establish standalone importability.
- **Recommended fix:** Document/package the required `src` path explicitly and add a direct import check. The preflight must fail clearly on a missing source dependency before any bank read, and the final receipt should include the exact environment used.

## Minor Issues

### m1 — Non-finite normalization constants are accepted

- **Finding:** `reports/layer_level_upgrade_20261015/reference/layer_upgrade/layers.py:44-49` claims finite means/stds but checks only sequence lengths and `min(std) <= 0`.
- **Evidence:** `CenterDifferenceInput(mean=(nan, .456, .406))` and `CenterDifferenceInput(std=(nan, .224, .225))` both construct successfully; a finite `[0,1]` input then produces non-finite output.
- **Why it matters:** A malformed configuration can enter the encoder despite the input finite-value guard. Defaults are finite, so this is configuration validation rather than a current default-path failure.
- **Recommended fix:** Keep the reference unchanged for exact reproduction; make any new integration loader reject non-finite mean/std values before constructing the operator, or record a versioned validation fix separately.

## Missing Evidence

- The final software scope now has independent differential, loader-failure, guard rollback, masked-view, geometry edge, bank mutation, registration, and subprocess-entrypoint tests; these are recorded in the 88-case verifier receipt.
- The approval and bank teacher fields are syntactically hash-checked but trusted as externally reviewed receipts; this authentication/trust-model boundary remains deferred and must not be described as cryptographic provenance.
- No patient training, ACDC evaluation, GPU run, Dice result, or generalization claim is supported by the current CPU/operator evidence.

## Reproducibility Checklist

- Config-driven experiments: **unknown**
- Fixed seeds: **pass** for the supplied manufactured tests and smoke scripts
- Dataset split documented: **unknown**; no dataset was used in this review
- Preprocessing documented: **pass** for the operator contract; integration wiring pending
- Baselines documented: **unknown**; baseline/ablation was intentionally outside this patch
- Metrics documented: **pass** for operator receipts; no medical metric evidence
- Environment documented: **pass** for the isolated verifier receipt; CPU only
- Checkpoint/eval procedure documented: **unknown** for Phase B

## Reviewer-Style Critique

- The numerical projection receipt overstates feasibility for ordinary float32 callers unless the finite-step wrapper is the only consumer.
- The geometry integration now fails closed on source/destination registration, affine, image, axis, manifest, and producer mismatches; actual semantic correctness of a self-authored receipt remains outside the trust model.
- A random ConvNeXt smoke test establishes tensor and gradient plumbing, not DINOv3 checkpoint compatibility or medical segmentation performance.
- The image-energy descent proof is conditional on frozen, semantically meaningful unaries/anchors; proxy energy reduction cannot support an anatomy claim.
- The guarded-loss guarantee is fixed-witness and fixed-proxy only; it does not establish validation or external-domain non-harm.

## Recommendation

Keep M1 recorded as a reference deviation and use only actual finite-step acceptance for guarded updates. The next authorized action is the documented approved-backbone preflight with real reviewed local artifacts, followed by a matched control pilot. Do not alter the baseline ablation.

## Verification Receipts

```text
rtk python scripts/materialize_layer_upgrade_reference.py --out /private/tmp/self-audit-layer-reference-review-20261015 --run-checks
PASS: reference bytes verified; pytest/full_model/probes exit 0

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider -c pytest.ini --rootdir=. --confcutdir=. tests
84 passed in 2.10s

rtk python - <<'PY'
import sys, torch
sys.path.insert(0, ".")
from integration.geometry import GridGeometry, transport_density
def geometry(a):
    return GridGeometry((47, 53), (7, 9), "1" * 64,
        tuple(tuple(float(a[i][j]) for j in range(4)) for i in range(4)), "unknown")
i = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
m = [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
x = torch.arange(63, dtype=torch.double).reshape(1, 1, 7, 9)
a, b = transport_density(x, geometry(i)), transport_density(x, geometry(m))
print(torch.equal(a, b), float((a - b).abs().max()))
PY
# output: True 0.0

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -c 'from integration import bank'
# FAIL: ModuleNotFoundError: No module named 'self_audit_guidance'
```

The commands above are retained as historical pre-fix receipts. The final
disposition and fresh 88-case verifier receipt below supersede them.

## Final Disposition — Phase A+B re-review (2026-10-15)

**Verdict: PASS for the stated CPU/software scaffolding scope; no implementation blocker found.** The latest suite passes independently with **88 passed in 3.70s**, and the fresh verifier reports `PASS_CPU_SOFTWARE_ONLY` with zero failures/errors/skips. `git diff --check` is clean. The six reference operator files remain byte-identical to the reviewed reference, and no baseline/ablation files were changed.

The former geometry finding M2 is addressed for this integration: `TeacherRegistration` binds native/teacher dimensions, image digest, affine, source manifest digest, producer digests, `TZYXC`, and the declared pixel-edge mapping; `GridGeometry` rejects source/target mismatch; and `coarse_evidence` requires the registration digest. The mirror, image, axis, and registration-mutation tests pass. The former import finding M3 is addressed operationally by `tests/test_entrypoints.py`, which runs a subprocess with the explicit experiment-plus-repository-`src` path and verifies the missing-approval preflight fails before creating output.

### Former P2 — evidence receipt manifest binding — RESOLVED

`integration/geometry.py:197-203` now requires `source["manifest_sha256"]` to equal `geometry.registration.source_manifest_sha256`, in addition to requiring the registration digest. The explicit wrong-manifest regression in `tests/test_integration.py:108-111` passes, so the prior provenance mismatch is closed.

M1 remains a documented reference deviation: ordinary float32 direct projection can report `KKT_VERIFIED` after a material post-cast constraint violation. The supported integration calls `guarded_sgd_step`, measures the actual finite-step losses, and explicitly records that projection alone is not a certificate. The approval and producer receipts remain self-authored trust inputs rather than cryptographic authentication; this is deferred by contract. No patient training, approved DINOv3 runtime, GPU run, anatomy-quality result, or medical metric is evidenced here.

Verification commands for this disposition:

```text
rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider -c pytest.ini --rootdir=. --confcutdir=. tests
88 passed in 3.70s

rtk env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:../../src /Users/alvinluong/miniforge3/bin/python verify_port.py --out /private/tmp/layer-port-review-final-20261015
PASS_CPU_SOFTWARE_ONLY; tests=88; failures=0; errors=0; skipped=0

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../../src python -c 'from integration import bank; print("BANK_IMPORT_OK")'
BANK_IMPORT_OK

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../../src python -c 'from integration import bank; print("subprocess import ok")'
subprocess import ok

rtk git diff --check
clean
```
