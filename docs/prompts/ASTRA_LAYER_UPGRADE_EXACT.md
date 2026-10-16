# Astra — reproduce the layer-level upgrade exactly, then integrate explicitly

Repository: `QuocKhanhLuong/Self-Audit`
Reference branch: `feat/layer-level-upgrade-20261015`
Publication base: `4d58f3aebe4f205b58b3184c8971b1d838109d82`
Reference directory: `reports/layer_level_upgrade_20261015/reference/`
Reference manifest SHA-256: `3f442aa35872bea5db00eb8880c002b1049a1e53da2a88accfbf8e6e5cec2576`

## Task

Reimplement the supplied layer/function-level prototype with the SAME mathematical operations, architecture, tensor contracts, initialization, detached paths, numerical guards and failure behavior. Then provide opt-in integration scaffolding for Self-Audit. Do not replace this task with another survey, a different backbone recommendation or an untested full training run.

The user wants the implementation to match the reference. Phase A is exact reproduction; Phase B is integration. A research improvement, optimizer substitution, new supervision rule, or changed tolerance is a separate version, not an exact reproduction.

## 1. Resolve the correct source before writing

Fetch remote refs, record current HEAD and the reference-branch SHA. Do not assume current main is the reference branch. Use a fresh worktree based on that branch. Never switch, clean, reset or overwrite a checkout used by another process.

Read, in order:
1. `reports/layer_level_upgrade_20261015/README.md`.
2. `docs/research/layer_level_upgrade_20261015/EXACT_SPEC.md`.
3. Reference `MANIFEST.json`, `THEORY.md`, `README.md`, `ASTRA_INTEGRATION.md`.
4. Every `reference/layer_upgrade/*.py`, both probe scripts and `reference/tests/test_operators.py.txt`.
5. The pinned historical student/bank contracts ONLY as integration context, not as the architecture to reproduce.

The reference README/manifest say no Git push occurred: those statements describe the original pre-publication archive. They are preserved verbatim, not the current delivery status. Reference tests use `.py.txt` solely to avoid repository-wide pytest collection. Restore that suffix only in the isolated copy.

Run the supplied verifier with a NEW directory outside the checkout:

```bash
python scripts/materialize_layer_upgrade_reference.py \
  --out /tmp/self-audit-layer-reference-unique --run-checks
```

Replace the example destination if it already exists; never overwrite it. Do not run probes in the immutable reference directory because they write JSON results. All 19 reference files must match their pinned bytes before any imported code is run. Do not weaken hash checks to pass.

## 2. Phase A — exact implementation

Put the new implementation in `experiments/layer_guided_student_v1/`, keeping the reference read-only. Preserve the `layer_upgrade` module layout and exported public names initially. Copying verified module bytes is the lowest-risk starting point; reorganizing modules is allowed only after differential equivalence tests pass and an explicit symbol map is documented.

Implement these exact pieces:

- `CenterDifferenceInput`: shared image-only [0,1] MRI inputs; center slice repeated and RGB-normalized; bias-free 2-to-3 3x3 difference convolution, zero-initialized. Do not replace the pretrained stem or add a second zero gate.
- `OfficialConvNeXtStages`: raw DINOv3 ConvNeXt native stages, not resized intermediate features; no implicit checkpoint download or random fallback.
- `LayerGuidedStudent`: four stage channels 96/192/384/768; decoder channels 96/128/192/256; exactly two `ResidualUnit`s per level, additive lateral fusion, GroupNorm(8), GELU, 64-channel feature projector from stage index 2, native 32-channel detail path. Preserve all convolution/bias choices from the source.
- Explicit bottom/right replicate padding to multiples of 32. Preserve native crop and the validity mask. Coarse stride is 4. Keep every returned tensor/key.
- `block_mean`, `block_lift`, `zero_mean_detail`, `compose_logits`: use disjoint cell averaging and repetition, not bilinear upsampling in this decomposition. Preserve `l = U a + (I-UD)[b*tanh(v)]`; `b=1.0` is a recorded buffer, not a tuned validated constant. Detail head weights and bias start at zero.
- Evidence, soft-CE, admissible-set and masked-feature functions: preserve detached targets/weights, support masks, no implicit geometry resize, original denominators and zero-support behavior. Uniform storage is never semantic permission.
- `project_direction`: selected-parameter QP, normalized nonzero guard rows, float64 active-set/KKT reference solve with <=8 guards and disclosed zero fallback. Preserve its sign convention.
- `guarded_sgd_step`: actual additive trial update; reevaluate frozen witness losses AND objective; halve step on failure; restore selected parameters on rejection or exception. No hidden momentum/decay and no `.grad` accumulation.
- `FrozenImageEnergy`, `image_grid_graph`, `refine_distribution`: frozen summed energy, correct undirected edge accounting, float64 detached log-space mirror updates, `L=2*lambda*dmax+beta*wmax`, default eta 0.5/L (1 if L=0), positive-simplex support and explicit failure checks. Return a proposal/receipt, not automatic semantic authorization.

Do not implement EMA, temporal attention, learned flow, a neural auditor, a new Q/Z policy, label sharpening, or a different gradient optimizer in Phase A. The previous broader strong-student proposal does not override this concrete prototype.

## 3. Prove implementation equivalence rather than resemblance

Run the unmodified 47 reference tests against the port. Add differential tests with the reference imported under a separate package identity; do not compare two aliases of the same implementation. Use identical copied state_dict tensors, inputs and deterministic mode; compare outputs, gradients and side effects.

Cover:
- Repeated-center identity and nonzero difference-weight gradients at zero initialization.
- DU=I, DH=0, padded-grid coarse identity, zero DIRECT coarse-to-detail gradient, nonzero native-to-detail gradient, and the softmax/pooling counterexample.
- Shapes 32x32, 47x53, 64x96, 65x97; every output key, mask and crop.
- Evidence/set zero-information cases and CE/forward-KL gradient equivalence.
- Feasible, opposed, rank-deficient and zero guards; finite-step nonincrease; rollback on rejection/exception; selected versus nonselected parameters.
- Image gradient versus autograd, analytic descent slack, invalid step rejection, no-information uniform case, and BOTH correct- and wrong-unary counterexamples.
- Full-size RANDOM torchvision ConvNeXt-Tiny smoke: 32,890,558 total parameters, 27,818,592 encoder and 5,071,966 other parameters under the reference configuration. This is NOT a required parameter count for every future official DINO object.

Matching semantics is required; cross-device bitwise equality, wall time, JUnit timestamps and nondeterministic incidental log fields are not claimed. Start CPU float64 operator checks before GPU checks. Do not relax existing test tolerances, drop a failing test, silently clamp a probability, or change the energy reduction to obtain a pass.

If the source and a mathematical claim disagree, document the exact counterexample in `REFERENCE_DEVIATIONS.md`; preserve the reference and propose a separately versioned fix. Do not claim a faithful port of a modified operator.

## 4. Phase B — integration, without algorithm drift

Add an explicit architecture ID such as `layer_guided_convnext_tiny_v1`, a new recipe and a checkpoint constructor/loader. Reject the old 0.35M GuidanceStudent checkpoint for this architecture. Specify checkpoint schema, state_dict keys, preprocessing, geometry transforms, model/source/backbone hashes, teacher provenance, permitted supervision, and inference input/output contracts.

Use only the locally approved official self-supervised DINOv3 weights after checking provenance and the expected digest. A missing checkpoint is BLOCKED_PRETRAINED, not permission to call the random smoke model pretrained. Do not bundle weights or credentials.

The historical bank is z-scored; the adapter requires [0,1]. Implement and record an image-only per-cine transformation shared by neighboring slices, plus native/padded/coarse coordinate maps. Never normalize with target-cohort aggregate statistics or use reference masks, GT ROIs, or label-selected phases.

Coarse teacher guidance must act on `coarse_logits` (or registered resampling AFTER exact D), not arbitrary resized native logits while retaining the direct-gradient claim. Mask padded positions; transport source class mass and support together; preserve UNKNOWN as distinct from BG. Do not fabricate high-resolution teacher evidence.

Keep the protected optimizer substep isolated. Ordinary AdamW elsewhere must not secretly invalidate a whole-step guarantee. An AdamW displacement guard requires a separate implementation, model AND optimizer-state rollback and additional proof/tests. Do not describe the reference guarded additive step as AdamW-compatible by default.

Use the image refiner as a detached proposal primitive. Default semantic admission remains OFF until the source of unaries, fixed image evidence, authorization/weighting and view transforms are explicitly specified. Proxy-energy decrease, EMA agreement and confidence are not proof of anatomical correctness.

A masked-feature loss requires the student to actually see the aligned masked view. No online teacher tensor may enter the student inference path as a hidden shortcut.

## 5. Execution boundary and deliverables

Run software/operator tests and synthetic forward/backward. Do not launch patient training, dataset downloads, a 50-epoch run, a paid job, a push or a main merge unless separately authorized. Missing patient artifacts must not block delivery of the exact implementation/tests. Do not claim improved Dice or generalization from operator tests.

Deliver code, `FUNCTION_MAP.md`, `PARITY_REPORT.json`, `REFERENCE_DEVIATIONS.md`, an explicit integration config/loader, `RUNBOOK.md`, and a Vietnamese `DECISION_BRIEF.md`. Include exact commands, source/ref hashes, test counts and failures, actual device/runtime scope, missing checks and changed files. Preserve historical run evidence.

The first question answered by this task is: "Does the new implementation reproduce the reference functions and their assumptions?" Only a subsequent approved matched experiment can answer whether the design improves anatomy or generalization.
