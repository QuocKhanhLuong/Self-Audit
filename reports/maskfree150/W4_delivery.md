# W4 delivery — producer, students and losses

Worker: Opus through Orca. Initial implementation `task_9765ba229880` /
`ctx_8b47980d1335`; bounded root-review revision `task_20f367adf925` /
`ctx_51ecc55aebb0` (section 7 records what the revision changed).
Requested provider/model: Claude Opus. Resolved model identity as exposed to this
session: `claude-opus-5` (self-reported by the runtime preamble; this is not an
independently verified provider receipt).

Contract read in full before editing: `self_audit_maskfree_150_orca_prompt.md`,
`reports/maskfree150/architecture_contract.md`, `reports/maskfree150/W4_assignment.md`.
Coordinator follow-up `msg_a6b95f2e85c2` applied: the canonical import path is
`self_audit_maskfree` with `PYTHONPATH=.:src`, never `src.self_audit_maskfree`.

**Acceptance verdict: PASS** after the root-review revision, with the limitations
listed in section 5 and the revision record in section 7. Everything
here is CPU synthetic software evidence. No GPU was contacted, no real dataset
exists in this checkout, and no commit or push was made.

## 1. Files

| Path | Status | Lines |
| --- | --- | --- |
| `src/self_audit_maskfree/models.py` | new, owned by W4 | 263 |
| `src/self_audit_maskfree/losses.py` | new, owned by W4 | 500 |
| `tests/test_maskfree_models.py` | new, owned by W4 | 290 |
| `reports/maskfree150/W4_delivery.md` | new, owned by W4 | this file |

No other file was created, edited, staged or reverted. `contracts.py`,
`observation.py` and every other owner's file are untouched.

```
$ git status --porcelain src/self_audit_maskfree tests/test_maskfree_models.py
?? src/self_audit_maskfree/
?? tests/test_maskfree_models.py
```

(`src/self_audit_maskfree/` is reported as a single untracked directory because
the package itself is untracked on this branch; only the two W4 files inside it
were added by this task.)

## 2. Implemented API, exactly as frozen in the contract

### `models.py`

* `Producer(feature_dim=16, width=16)` — GroupNorm encoder/decoder, random
  initialization. `forward(context: [B,3,H,W]) -> {"features": [B,feature_dim,H,W],
  "reconstruction": [B,1,H,W]}`.
* `Student(width=16)` — `forward(context: [B,3,H,W]) -> logits [B,4,H,W]` in the
  frozen named order `0=BG, 1=RV, 2=MYO, 3=LV`.
* `make_models(seed=42, width=16, feature_dim=16) -> dict` with keys `producer`,
  `student_no_audit`, `student_audited`, plus `parameter_counts`, `version`,
  `seed`, `width`, `feature_dim`.
* `count_parameters(model, *, trainable_only=True) -> int`.
* `MAX_PRODUCER_PARAMETERS = 5_000_000`, `NUM_CLASSES = 4`, `CONTEXT_CHANNELS = 3`.

### `losses.py`

* `producer_loss(model, context, fit_support, *, generator=None) -> (Tensor, dict[str, Tensor])`.
* `student_loss(logits, probabilities, validity) -> (Tensor, dict[str, Tensor])`.
* Constants exposed for the trainer's provenance record: `MAX_SAMPLES_PER_IMAGE=128`,
  `CONTRASTIVE_TEMPERATURE=0.07`, `MIN_NEGATIVE_SEPARATION=8.0`,
  `RECONSTRUCTION_BLOCKS=4`, `WEIGHT_CONTRASTIVE=WEIGHT_RECONSTRUCTION=WEIGHT_EQUIVARIANCE=1.0`,
  `PROBABILITY_SUM_TOLERANCE=1e-3`.

**Metric dict convention (for W5).** Every value in both metric dicts is a
detached zero-dimensional `torch.Tensor`, including the boolean `skipped` flag.
This makes both `if metrics["skipped"]:` and `metrics["skipped"].item()` correct,
so the trainer cannot be broken by an assumption about the flag's type.

Producer metric keys: `loss`, `contrastive`, `reconstruction`, `equivariance`,
`contrastive_images`, `contrastive_samples`, `reconstruction_support`,
`equivariance_support`, `feature_std_mean`, `feature_std_min`,
`feature_cosine_mean`, `hidden_intensity_elements`, `transform_rotations`,
`transform_flip`, `skipped`.

`hidden_intensity_elements` counts context elements outside `fit_support` that
were non-zero on entry and had to be cleared. It is zero for a legal input; any
other value is a data-layer contract violation that W5 must surface in its epoch
report rather than absorb.

Student metric keys: `loss`, `valid_support`, `valid_fraction`, `pixels`,
`target_entropy`, `skipped`.

## 3. Method as implemented

### 3.1 Producer architecture

Two-level GroupNorm encoder/decoder: `enc0(3→w) → enc1(w→2w) → enc2(2w→4w)`,
decoder `dec1(4w+2w→2w) → dec0(2w+w→w)`, then a 1×1 feature head and a 1×1
reconstruction head. Each block is two 3×3 convolutions with GroupNorm and ReLU.
Group counts are the largest of `(8, 4, 2, 1)` that divides the channel width
**and leaves at least `MIN_CHANNELS_PER_GROUP = 2` channels per group**, so
normalization is always well formed. A one-channel group normalizes over `H*W`
elements alone, which at `1×1` has exactly zero variance and emits a constant
zero; the constraint is what makes the small-image promise true at the narrowest
supported width. Width 16, the contracted default, is unaffected: every layer
already used 8 groups of at least 2 channels, so parameter counts and numerics
are unchanged by this fix.

Down-sampling uses `adaptive_avg_pool2d` to `max(1, size // 2)` and up-sampling
interpolates back to the stored skip size. That is the **small-image GroupNorm
support** guard: odd, non-square and `1×1` inputs all work, because no path ever
requires an even size or a kernel larger than the input, and GroupNorm never
normalizes over the batch axis so a single pixel still has `C/G ≥ 2` elements per
group. Verified at `(5,7)`, `(1,1)` and `(3,16)` at width 16, and at `1×1` with
batch 1 at width 4.

Honest scope of that guarantee: at `1×1` a two-channel group can only record
which of its two channels is larger, so the normalized output is quantized to a
sign pattern. The network is input-dependent rather than constant — 32 random
`1×1` inputs at width 4 produce 13 distinct outputs — but it is not expressive at
a single pixel. The claim is "well defined and not degenerate", not "unimpaired".

### 3.2 Producer objectives, all restricted to `O_fit`

Before any forward pass, `producer_loss` physically zeroes all three context
channels outside `fit_support`. The data layer is already required to deliver
withheld pixels and their guard band as zeros, so this is a no-op for a legal
input; for an illegal one it makes the documented immunity real instead of
merely claimed. Without it the producer would forward hidden intensities and
every feature — hence every arm below — would depend on observations the fitting
side may not see, no matter how carefully the targets were restricted.

1. **Bounded dense contrastive correspondence.** One dihedral operation
   (0/90/180/270 degrees, with or without a horizontal flip) is sampled per step
   and applied to the context; features of the transformed view are mapped back
   through the *exact* inverse, so the same anatomical pixel in the two views is
   a known positive with no learned matcher and no mask. At most **128 locations
   per image** are sampled, only from `fit_support`, giving a similarity matrix of
   at most `128×128` per image. No `HW×HW` similarity is ever allocated.
2. **Local masked reconstruction.** Four small square blocks (side
   `max(1, min(8, min(H,W)//4))`) are physically zeroed in all three context
   channels; the reconstruction head predicts the centre channel and is scored
   only on `block ∧ fit_support`. The target is read from the fitting context
   itself, so no selection or verification intensity can enter.
3. **Geometric equivariance.** Mean squared error between features of the
   untransformed context and inverse-mapped features of the transformed context,
   over `fit_support` only.

Weights are 1.0/1.0/1.0, fixed provisional constants, not tuned against anything.

**Collapse indicators** (SSL feature variance): `feature_std_mean` and
`feature_std_min` are per-channel standard deviations computed over fit pixels
only; `feature_cosine_mean` is the mean off-diagonal cosine similarity among the
sampled anchors. All three are detached and are diagnostics, not objectives.

### 3.3 Negative-sampling rule, stated explicitly

Negatives are other sampled locations **from the same image of the same patient**,
at least 8 pixels away from the anchor in native pixel coordinates. Anchors with
no admissible negative are dropped row-wise (their columns remain available to
the surviving anchors), which avoids a degenerate all-positive softmax row.

### 3.4 Students

Two `Student` instances are built under one seeded RNG scope and then
`student_audited` is loaded from a deep copy of `student_no_audit`'s state dict.
`make_models` itself checks that the two arms share no parameter storage and
raises if they do. `student_loss` is a single function used by both arms, so the
arms differ only in the pseudo-labels they consume.

`student_loss` computes validity-weighted soft cross entropy with denominator
`validity.sum()` — the valid support, not the pixel count. Both `probabilities`
and `validity` are `detach()`ed inside the function, so no gradient can reach the
producer, the hypothesis bank, or the other arm even if a caller forgets.

### 3.5 Numerical guards

* Non-finite `context`, `probabilities` or `validity` raise `ValueError`;
  non-finite `logits` or a non-finite computed loss raise `FloatingPointError`.
  This is fail-closed on purpose: a silent `nan` would poison a 150-epoch run.
* Masked similarity entries use a finite `-1e4` rather than `-inf`, so `softmax`
  cannot produce `nan` gradients.
* Every denominator is a counted support clamped away from zero; an arm with no
  usable support contributes an exact differentiable zero derived from the
  feature tensor, never `0/0`.
* Pseudo-labels are validated, never repaired. `student_loss` rejects
  probabilities or validity outside `[0,1]` and, on valid support, rejects class
  vectors that do not sum to one within `PROBABILITY_SUM_TOLERANCE = 1e-3`.
  Pixels with zero validity are unconstrained, so a bank may leave them at zero,
  and the legal all-invalid unit is still a skip rather than an error.
  Renormalizing instead would manufacture a target no hypothesis produced, and
  an all-zero class vector on valid support would contribute exactly zero to the
  numerator while still counting in the denominator — a silent dilution of a
  real gradient. Clamping now appears only in denominators and in the log of the
  reported target entropy.

## 4. Focused checks — exact commands and output

Three focused checks were run. No broad repository suite was executed.

### Check 1 — the owned test file

```
$ cd /Users/alvinluong/Self-Audit && PYTHONPATH=.:src \
    /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_maskfree_models.py -v
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
collected 6 items

tests/test_maskfree_models.py::test_shapes_parameter_budget_and_small_images PASSED [ 16%]
tests/test_maskfree_models.py::test_students_start_identical_without_shared_storage PASSED [ 33%]
tests/test_maskfree_models.py::test_producer_loss_is_finite_bounded_and_fit_only PASSED [ 50%]
tests/test_maskfree_models.py::test_student_loss_validity_weighting_and_all_invalid PASSED [ 66%]
tests/test_maskfree_models.py::test_no_cross_arm_or_producer_gradients PASSED [ 83%]
tests/test_maskfree_models.py::test_revision_hidden_intensity_immunity_targets_and_narrow_width PASSED [100%]

============================== 6 passed in 0.62s ===============================
```

What each test actually asserts:

1. contracted output shapes; `parameter_counts` equals a fresh measurement and is
   within budget; both students have equal counts; forward passes succeed at
   `(5,7)`, `(1,1)` and `(3,16)`.
2. byte-identical initial state dicts; disjoint `data_ptr()` sets; mutating one
   arm leaves the other unchanged; seed 42 reproduces, seed 43 differs.
3. producer loss finite and differentiable; all three arms finite; sampled count
   bounded by `128 × batch`; backward yields finite, non-zero producer gradients;
   `_sample_support_locations` returns only support pixels; an all-withheld unit
   returns `skipped=True`, exactly `0.0`, and produces zero gradients.
4. student loss valid-support denominator (`128.0` for a `2×8×8` all-valid batch);
   halving every validity weight leaves the normalized loss unchanged while
   `valid_support` halves to `64.0`; an all-invalid batch returns exactly `0.0`
   with `skipped=True` and zero gradients; `nan` logits and `nan` probabilities
   both raise.
5. a student backward touches that arm only — the other arm's and the producer's
   parameters keep `grad is None`, even though the pseudo-labels were built from
   the producer's live graph on purpose.
6. the combined revision check (section 7): hidden-intensity immunity under a
   fixed generator seed, pseudo-label rejection cases, and the width-4 GroupNorm
   guarantee at `1×1` with batch 1.

### Check 2 — measured parameter counts and a batch-8 forward/loss at 128×128

```
$ PYTHONPATH=.:src python -c "<torch.manual_seed(0); measure counts, run both losses>"
counts {"producer": 118865, "student_no_audit": 118644,
        "student_audited": 118644, "total_trainable": 356153}
producer_loss 8.400172233581543
{'loss': 8.4002, 'contrastive': 7.1517, 'reconstruction': 1.0503,
 'equivariance': 0.1982, 'contrastive_images': 8.0, 'contrastive_samples': 1024.0,
 'reconstruction_support': 1469.0, 'equivariance_support': 1671168.0,
 'feature_std_mean': 0.3143, 'feature_std_min': 0.2378,
 'feature_cosine_mean': 0.5444, 'hidden_intensity_elements': 0.0,
 'transform_rotations': 3.0, 'transform_flip': 0.0, 'skipped': 0.0}
student_loss 1.4263014793395996 {'loss': 1.4263, 'valid_support': 65511.0781,
 'valid_fraction': 0.4998, 'pixels': 131072.0, 'target_entropy': 1.1146,
 'skipped': 0.0}
```

This is the post-revision rerun, with the fixture RNG now seeded so the numbers
are reproducible; the pre-revision run quoted slightly different values only
because its synthetic batch was drawn from an unseeded global RNG. Parameter
counts are identical before and after the revision, as expected: the `_groups`
fix changes nothing at width 16.

**Measured parameter counts (width 16, feature_dim 16):**

| Component | Trainable parameters |
| --- | --- |
| `producer` | 118,865 |
| `student_no_audit` | 118,644 |
| `student_audited` | 118,644 |
| total | 356,153 |

The producer is 0.119M parameters, 2.4% of the 5M budget. `contrastive_samples`
is 1024 for batch 8, i.e. exactly the `128 × 8` bound; `equivariance_support`
counts scored *elements* (`8 × 16 channels × 102 × 128` fit pixels), not pixels.

### Check 3 — optimization sanity and collapse watch, 30 steps, CPU synthetic

Run during the initial dispatch and **not repeated** in the revision, per the
revision instruction. The revision does not change this path for a legal input:
the fixture was already zero outside its support, so the masking added at
function entry is a no-op on it.

```
$ PYTHONPATH=.:src python -c "<30 Adam steps on a smoothed synthetic phantom>"
step0  [6.2754, 6.0071, 0.3313]     # total, contrastive, feature_std_mean
step29 [2.0519, 2.0123, 0.4328]
decreasing: True std alive: True
```

The objective is optimizable and feature standard deviation grows rather than
collapsing over the run. This is a software sanity check on a synthetic phantom;
it says nothing about anatomy or label quality.

### Supervision firewall grep

```
$ grep -nE "self_audit(_candidate_c)?[^_]|import self_audit|pretrained|\
load_state_dict\(torch\.load|hub\.|download" \
    src/self_audit_maskfree/models.py src/self_audit_maskfree/losses.py \
  | grep -v "self_audit_maskfree"
clean: no supervised imports, no checkpoint loads, no downloads
```

Both files import only `torch` and the package's own `contracts` / `models`.
There is no `pretrained` flag, no `torch.load`, no `torch.hub`, no network call,
and no import of `self_audit` or `self_audit_candidate_c`.

## 5. Documented limitations, as required by the assignment

### 5.1 Same-study negatives are false negatives

The negative rule ("other sampled locations in the same image, at least 8 pixels
from the anchor") deliberately avoids the common and false assumption that every
other patient is a true anatomical negative. It is still wrong in the other
direction, and this matters:

* two background patches on opposite sides of the field of view are the same
  tissue class;
* the septal and lateral myocardial walls are both MYO and are typically more
  than 8 pixels apart;
* at 128×128 with a heart occupying a small central fraction of the FOV, the
  large majority of sampled pairs are background-background.

Those pairs are **false negatives**, and the InfoNCE term actively pushes them
apart. The consequence is that the learned features should be read as *locally
discriminative*, not as a class-pure embedding, and any downstream grouping over
them inherits that bias. The 8-pixel threshold trades false negatives (too small)
against trivial negatives (too large); it is a fixed provisional constant chosen
a priori, not tuned, and no mask or reference was used to select it.

### 5.2 Physical batch, accumulation and the contrastive arm

Negatives are drawn **within one image**, never across the batch. The *negative
set* is therefore invariant to physical batch size: microbatch 4 × accumulation 2
draws exactly the same per-image negatives as physical batch 8. That is a real
difference from cross-batch InfoNCE, where shrinking the physical batch shrinks
the negative pool and silently changes the objective.

What is **not** identical under accumulation: `producer_loss` returns the mean
over the images it was given. Summing two microbatch means and halving equals the
batch-8 mean only when both microbatches contribute the same number of images to
each arm. Images with empty fit support, or with fewer than two sampled
locations, are skipped and counted in `contrastive_images`, so unequal skipping
makes the accumulated gradient differ slightly from the single-batch gradient.
The honest statement for the trainer's log is: *the negative distribution is
unchanged by the accumulation split; only the per-image weighting of skipped
images can differ.* Physical and effective batch must still be recorded
separately, and no numerical-identity claim should be made.

### 5.3 Other limitations

* All evidence here is CPU synthetic. There is no real ACDC or M&Ms data in this
  checkout, no GPU was contacted, and full execution remains **NOT STARTED**.
* The producer is 0.119M parameters, far below the 5M ceiling. This is what
  `width=16` yields under the contract; whether that capacity is sufficient for
  real cardiac features is unmeasured.
* The equivariance arm is a consistency objective with gradients through both
  branches. It admits a constant-feature solution in principle; the contrastive
  arm opposes that and the variance metrics are there to detect it, but no
  guarantee is claimed and the 30-step check is not evidence about 150 epochs.
* `producer_loss` runs three forward passes per step (untransformed, transformed,
  masked). The trainer should budget for that when recording compute.
* The dihedral transform is sampled once per batch, not per image, so all images
  in a batch share one augmentation. This is intentional for cheap exact
  inversion and it is a real reduction in augmentation diversity.
* `student_loss` and `producer_loss` fail closed on `nan`, and after the
  revision `student_loss` also fails closed on out-of-range or unnormalized
  pseudo-labels. A blown-up run, or a bank that emits a degenerate all-zero
  class vector on a pixel it still calls valid, will raise rather than log a
  number; W5 and W2 should decide where that is caught. The contract-conformant
  way to express "no label here" is validity zero, not an all-zero distribution.
* The `1×1` GroupNorm guarantee is "well defined and input dependent", not
  "expressive": at a single pixel a two-channel group only records which channel
  is larger, so the network's output is quantized to a sign pattern.
* `make_models` seeds and restores the global CPU RNG. It does not seed CUDA
  RNGs, so a CUDA-side initialization difference would not be caught here.

## 6. Acceptance

**PASS**, after the three root-review defects in section 7 were fixed and
regression-checked. The contracted `Producer`, `Student`, `make_models`, `producer_loss`
and `student_loss` are implemented with the exact signatures and tensor shapes
from `architecture_contract.md`; parameter counts are measured and within budget;
the contrastive arm is bounded at 128 samples per image with no `HW×HW`
allocation; students start identical with disjoint storage; all-invalid batches
return an explicit differentiable zero with a skip flag; and no gradient crosses
between the arms or into the producer. `producer_loss` is now demonstrably a
function of the fit observations alone, `student_loss` rejects invalid
pseudo-labels instead of repairing them, and no GroupNorm group holds a single
channel. Six tests in one focused command plus the seeded batch-8 measurement
were run for this revision and all passed; the 30-step sanity run from the
initial dispatch was not repeated. The limitations above are methodological, not unimplemented work.

No dependency was unavailable, so nothing is BLOCKED. No commit or push was made.

## 7. Root-review revision record

Bounded revision under `task_20f367adf925` / `ctx_51ecc55aebb0`. Three defects
were reported by root review of the source; all three were real, all three are
fixed, and the scope stayed inside the four W4-owned files. No commit, no push,
no edit to any other owner's file.

### 7.1 `producer_loss` forwarded context outside `fit_support`

**Defect.** The docstring claimed the objective could not see withheld
intensities, and the *targets* were indeed restricted to `fit_support`, but the
model was called on the raw `context`. Because every arm is a function of the
resulting features, a non-zero pixel outside the support would have changed the
loss and its gradient. The claim was therefore stronger than the code.

**Fix.** `producer_loss` now zeroes all three context channels outside
`fit_support` before the first forward pass, and reports how many non-zero
elements it had to clear as `hidden_intensity_elements`. Masking was chosen over
rejecting so that a data-layer bug degrades to a logged contract violation
instead of killing a 150-epoch run; the count makes it visible either way.

**Regression check.** With a fixed generator seed, the loss and every producer
gradient are compared between a legal context and the same context with every
pixel outside the support overwritten by `torch.randn` under seed 99. The loss
values are bitwise equal, all gradient tensors are bitwise equal,
`hidden_intensity_elements` is `0.0` for the legal input and non-zero for the
polluted one. This check fails against the pre-revision code.

### 7.2 `student_loss` repaired invalid pseudo-labels

**Defect.** The function clamped negative probabilities to zero, clamped
validity into `[0,1]`, and renormalized the class axis by a sum clamped at
`1e-6`. An all-zero class vector on a pixel with full validity therefore became
an all-zero target, contributed exactly zero to the numerator, and still counted
in the denominator — a silent dilution of the real gradient. More generally the
function was fabricating targets no hypothesis had produced.

**Fix.** Validation replaces repair. Probabilities and validity outside `[0,1]`
are rejected; on valid support the class axis must already sum to one within
`PROBABILITY_SUM_TOLERANCE = 1e-3`; non-finite inputs are still rejected. No
renormalization remains. Two legal behaviours are explicitly preserved: pixels
with zero validity are unconstrained, so a bank may leave them at zero, and an
entirely invalid unit still returns a differentiable exact zero with
`skipped=True` rather than raising.

**Regression check.** Five rejection cases (all-zero class vectors on valid
support, a uniformly unnormalized distribution, a negative probability, validity
above one, validity below zero), plus a mixed unit where a zero-validity pixel
carries an all-zero class vector and the loss is finite with
`valid_support = 15.0` of 16 pixels, plus the preserved all-invalid skip.

### 7.3 GroupNorm could place one channel in a group

**Defect.** `_groups` only required the group count to divide the channel count.
At `width=4` that gave four groups of one channel. A one-channel group
normalizes over `H*W` elements alone, so at `1×1` its variance is exactly zero
and GroupNorm emits a constant zero — the narrow network collapses to a constant
function, contradicting the small-image promise in the module docstring.

**Fix.** `_groups` now also requires `channels // groups >= MIN_CHANNELS_PER_GROUP
= 2`. Width 16 is unaffected — every layer already used 8 groups of at least two
channels — so parameter counts, initialization and numerics at the contracted
default are unchanged.

**Regression check.** `_groups(4) == 2`, `_groups(8) == 4`, `_groups(16) == 8`;
every `GroupNorm` in `Producer(width=4)` and `Student(width=4)` has at least two
channels per group; and a width-4 producer at `1×1` with batch 1 produces 13
distinct outputs over 32 random inputs, i.e. it is not a constant function. The
last assertion fails against the pre-revision code, which returned one output.

### 7.4 Scope and environment

All three regression checks live in one added test,
`test_revision_hidden_intensity_immunity_targets_and_narrow_width`, run together
with the five existing tests as a single focused command. The 30-step
optimization sanity run was not repeated, per the revision instruction.

Canonical imports are unchanged (`self_audit_maskfree` under `PYTHONPATH=.:src`).
The target runtime is Python 3.10 with PyTorch 2.4.1; **the checks here ran on
Python 3.11.16 with PyTorch 2.14.0**, which is the only interpreter available in
this checkout, so 3.10/2.4.1 compatibility is argued rather than executed: the
sources use no syntax newer than 3.10 and no torch API newer than 2.4 (the newest
constructs used are `torch.div(..., rounding_mode=)`, `Generator.device`,
`F.adaptive_avg_pool2d` and builtin generic annotations under
`from __future__ import annotations`), and all three files byte-compile. That is
a reasoned argument, not a passing test on the target runtime, and the
coordinator's integration gate should confirm it there.
