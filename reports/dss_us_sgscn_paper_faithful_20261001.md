# DSS-US / SGSCN PAPER_FAITHFUL_REIMPLEMENTATION profiles (2026-10-01)

Profile classes are now separate everywhere (configs, index hashes, raw-run provenance,
readiness):

| Class | Meaning |
|---|---|
| `PAPER_FAITHFUL_REIMPLEMENTATION` | implements exactly what the paper states; every value the paper leaves open is a required field (no defaults, no inherited official values); not called exact/original reproduction |
| `OFFICIAL_REFERENCE` (SGSCN `*_official_reference`) | unchanged official-code arithmetic; non-paper |
| original paper-reproduction profiles (`step*_*.yaml`, `*_paper.yaml`) | unchanged; BLOCKED_PROTOCOL |

A required field is filled only with `scripts/instantiate_native_profile.py`, which writes a
new, separately hashed profile (values recorded as `USER_SUPPLIED` with a source and the
base hash) and never edits the base profile. `load_lock` blocks any producer whose
`paper_unspecified`, `implementation_conventions` or `required_data` field is null.

## SGSCN (`ph2_paper_faithful`, `sysu_us_paper_faithful`)

Implemented directly from the paper (`baseline/SGSCN/src/sgscn/paper_faithful.py`):

- 3 conv layers, 3x3, stride 1, pad 1, 100 filters, each with ReLU and batch normalisation (section 3.2).
- S_hat zero mean / unit variance, labels = argmax (section 2.2).
- L_ce: standard softmax cross-entropy summed over pixels (section 2.3 "as in a standard CNN"; printed Eq. 1 `ln S_hat_{n-1}` is a typo and undefined on a zero-mean map).
- L_ss: Eq. 2 literally, k = 1..W-1, l = 1..H-1, channel L1 norms, summed.
- L_cc: Eqs. 3-4 over pixel coordinates, squared distance to the density centre (Eq. 4 `(C^k - C^l)` read as the centre `(C^k, C^l)`).
- L = L_ce + L_ss + L_cc, unweighted (section 2.6, Fig. 1); no spatial x5.
- SGD momentum 0.9; lr 0.1 PH2 / 0.05 SYSU-US; one network per image.

PAPER_UNSPECIFIED (required): `layer_order` {conv_relu_bn, conv_bn_relu};
`output_normalization` {final_batchnorm, per_channel_standardization, global_standardization};
`context_density` {literal_normalized_map, channel_softmax, epsilon_shift = OFFICIAL_CODE_FALLBACK};
`input_encoding` {rgb_unit_interval, bgr_unit_interval = OFFICIAL_CODE_FALLBACK, grayscale_unit_interval};
SYSU-US `cohort_inventory` (must equal the run's image-inventory hash).
Conditional: `standardization_epsilon` (standardisation options), `context_epsilon` (epsilon_shift; OFFICIAL_CODE_FALLBACK).

IMPLEMENTATION_CONVENTION: `stopping_rule` = stop when the active label count is unchanged and
the relative loss change is within `relative_loss_tolerance` for `stability_patience`
consecutive iterations ("until stable", section 2.6); required: `max_iterations`,
`stability_patience`, `relative_loss_tolerance`, `final_forward_mode`; fixed and declared:
`initialization` = framework default, `weight_decay` = 0.

Paper vs official reference: architecture (3x3 x3 vs 3x3 x2 + 1x1 without ReLU), loss
reduction (sums vs means), spatial weight (1 vs 5), stopping (stability convention vs
maxIter 50 / minLabels 3), input encoding and context density (required choice vs BGR/255
and epsilon 1e-3). Learning rate and momentum agree. Verified by tests.

## DSS-US (11 `*_paper_faithful` profiles)

Implemented directly from the paper (`baseline/DSS_US/src/dss_us/paper_faithful.py`, composing
the existing independent kernels): DINO ViT-S/8 last-layer keys, patch size 8; Eq. 1 positive
Gram affinity; Eqs. 2-3 SSD / MI patch affinities on 8x8 patches; Eq. 4 positional KNN affinity
on the unit grid; Eq. 5 linear combination; Eq. 6 normalised Laplacian; eigenvectors ->
K-means with 15 segments; nearest upscale; CRF for Step I (Table 1); Step II bounding-box crop
features, optional mask/position embeddings linearly combined (Ours step2), K-means.
Row components follow the paper: DSS baseline / +preprocessing / +affinities / combined; Step II
with DSS step2 or Ours step2.

PAPER_UNSPECIFIED (required, by row): feature L2 normalisation, DINO input policy, eigenvector
count and trivial-eigenvector handling, embedding row normalisation, upscale method, CRF backend
and parameters, CAMUS image conversion; preprocessing method and parameters (proc rows); C_ssd,
C_mi, C_pos, delta_ssd, delta_mi, MI bins and intensity range, patch intensity scale, positional
KNN k and symmetrisation (affinity rows); Step II cluster count, phi for crops, Step II CRF use,
DSS step2 definition or phi_mask / phi_position / C_mask / C_position (Step II rows).
IMPLEMENTATION_CONVENTION (required): K-means seed, n_init, max_iter, tolerance, algorithm.
REQUIRED_DATA: CAMUS cohort inventory, DINO checkpoint SHA-256.

Explicit gate: the end-to-end CAMUS runner is not wired for paper-faithful profiles (algorithm
functions only), and self-audit-canonical v1 has no dense-CRF backend; adding one is an
environment change. Step I Track B (per-image remapped Dice) is EVALUATOR_READY.

## Executability

- Executable now: SGSCN official-reference producers (unchanged, non-paper).
- Executable after the user supplies values via `instantiate_native_profile.py`: SGSCN
  `ph2_paper_faithful` (8 values: 4 paper-unspecified + 4 conventions, plus conditional epsilons)
  and `sysu_us_paper_faithful` (9, including the cohort inventory). Verified end to end on a
  synthetic image with a user-instantiated profile. Paper Track B stays blocked (overlap measure,
  ties, HM/XOR).
- Not executable even with user values: all 11 DSS-US paper-faithful profiles (runner not wired;
  no CRF backend in the canonical environment), and all original paper-reproduction profiles.
