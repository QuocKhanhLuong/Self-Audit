# Reference limits and integration differences

No reference operator is modified. Differential tests deliberately retain the
following behavior. These are limits on the mathematical interpretation, not
failures to reproduce the source.

## Float32 projection can lose feasibility after casting

`project_direction` solves the active-set QP in float64, then returns the
direction in the caller's dtype. Its `KKT_VERIFIED` receipt describes the
float64 solution; it does not certify the returned float32 vector.

Reproduction: set `torch.manual_seed(0)`; generate four successive directions
with `scale=torch.randint(-4,8,(1,)).item()`,
`d=(torch.randn(20)*10**scale).float()` and `g=torch.randn(4,20).float()`.
The fourth direction has norm about 4,965,879. Both reference and port return
`KKT_VERIFIED`, while `min(g.double() @ projected.double())` is approximately
`-0.1379108541` (minimum normalized dot approximately `-0.03282279`).
See `tests/test_reference_deviations.py`.

The integration exposes only the actual finite-step acceptance receipt as a
protected-substep result. `guarded_sgd_step` reevaluates the objective and all
frozen witness losses at the actual selected-parameter values, backtracks and
restores them on rejection/exception. It retains the reference tolerance, so
the claim is nonincrease within that tolerance for those closures. A separately
versioned projection could validate the cast result and fall back to zero; that
change is proposed, not implemented in this exact port.

## Custom nonfinite normalization parameters pass the reference constructor

The original `CenterDifferenceInput` length/positivity checks accept custom
NaN means or standard deviations; finite images then produce nonfinite output.
Defaults are finite. The exact operator and its counterexample remain intact.
The integration fixes the recipe to the reference RGB mean/std and bound 1.0,
and rejects changed/nonfinite checkpoint buffers. General finite-value checking
for custom normalization would belong to a separately versioned operator.

## Conditional claims remain conditional

- `D U = I` and `D H = 0` concern the padded logit grid and the direct detail
  path. Shared decoder/encoder parameters can still change both paths. Pooling
  probabilities is not equivalent to applying softmax to pooled logits.
- Guard receipts apply to one additive substep and the supplied frozen TRAIN
  closures. They do not certify subsequent AdamW steps, future samples, source
  correctness, or validation/test performance. Witness metadata is a caller
  contract; it cannot prove which samples a Python closure actually loads.
- Mirror descent certifies a fixed proxy energy. Both favorable and incorrect
  unaries reduce that energy. A wrong unary can move the distribution toward
  the wrong class. Semantic admission stays off.
- Uniform storage at zero support is not background evidence or permission to
  use a named loss. UNKNOWN remains unsupported rather than being relabeled BG.

## New integration choices, outside the reproduced operators

- Historical z-score images receive one min/max transform per full cine, shared
  by all frames and neighboring spatial slices. No cohort statistics or masks
  are fitted. Constant cines become zero. This choice has software checks only.
- Registered teacher values are treated as densities on pixel cells spanning
  the full native field of view. Positive overlap transports mass and support
  with the same map. Fully padded cells have zero support; partial coarse cells
  retain fractional support. Feature learning excludes partially padded cells.
  This conservative cell interpretation is explicit; it does not recreate
  high-resolution teacher evidence or validate anatomical registration.
- Source and destination image hashes, grids, affine and mapping must match a
  source-side registration receipt. The historical adapter accepts only pinned
  producer code and a caller-pinned frozen manifest. A new producer or transform
  requires a separately reviewed registration contract.
- Provenance receipts are externally reviewed inputs, not signed attestations.
  A caller can fabricate a mutually consistent manifest. Hashes verify bytes
  relative to the supplied trusted digest; they do not prove pretraining ancestry,
  class correctness or truth of metadata.
- The conventional-head control preserves parameter count but computes and
  discards the split composition before replacing it. Equal FLOPs or timing are
  not claimed; measure and disclose that overhead in a later matched experiment.

Official DINOv3 runtime, real bank compatibility, patient training, GPU peak
memory and anatomical quality remain unmeasured. Mock loader fixtures and random
torchvision smoke results are labeled as such throughout the receipts.
