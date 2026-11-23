# Checks before writing up S0-slicessm

## 1. Metric parity (val10, val10-selected checkpoint)

- native FG Dice (patient macro): 0.9070438171
- SSL4MIS-compatible mean Dice (medpy): 0.9070438171
- independent numpy Dice (per class, ED/ES per patient): 0.9070438171
- max |native - compat| = 0.00e+00; max |independent - compat| = 0.00e+00
- Jaccard: compat 0.8355039224, independent numpy 0.8355039224
- HD95 differs by construction: native 3.271 mm, compat 1.143 voxels (LV compat/native ratio 0.44, between 1/spacing and 1).
- Why Dice is identical: `evaluation._frame_metrics` defines `fg_dice` as the mean of the three per-class Dice values of a frame, and the compat evaluator takes the mean over the same three classes per volume; averaging the two frames of a patient and then patients equals the average over all volumes when every patient has exactly two frames. It is an algebraic identity, not a shared code path (the HD95 values prove the two evaluators ran). If a patient lacked one frame or a class were empty, the two would differ.

## 2. Paired per-patient comparison with T-M3HL (student - teacher; bootstrap 10,000, seed 42; stored per-volume results)

### vs official-path T-M3HL (h5, SSL4MIS min-max input, voxel metrics)

| metric | split | n | mean diff | 95% bootstrap CI | Wilcoxon p |
|---|---|---|---|---|---|
| Dice | test20 | 20 | +0.0209 | [+0.0076, +0.0354] | 0.005 |
| Dice | val10 | 10 | +0.0185 | [+0.0035, +0.0372] | 0.064 |
| Jaccard | test20 | 20 | +0.0313 | [+0.0118, +0.0517] | 0.004 |
| Jaccard | val10 | 10 | +0.0252 | [+0.0045, +0.0500] | 0.064 |
| HD95 (voxels) | test20 | 20 | -1.7476 | [-3.5015, -0.2998] | 0.019 |
| HD95 (voxels) | val10 | 10 | -0.4296 | [-0.7718, -0.1590] | 0.006 |
| ASD (voxels) | test20 | 20 | -0.7525 | [-1.4256, -0.2418] | 0.000 |
| ASD (voxels) | val10 | 10 | -0.2354 | [-0.4010, -0.1051] | 0.002 |

Verdict (CI on test20 excludes 0 and val10 has the same sign): supported for dice, hd95. n = 20 test patients (10 val), one seed.

### vs 3D-wrapper T-M3HL (our normalisation, same evaluator as the student)

| metric | split | n | mean diff | 95% bootstrap CI | Wilcoxon p |
|---|---|---|---|---|---|
| Dice | test20 | 20 | +0.0346 | [+0.0190, +0.0515] | 0.000 |
| Dice | val10 | 10 | +0.0277 | [+0.0139, +0.0467] | 0.004 |
| Jaccard | test20 | 20 | +0.0521 | [+0.0294, +0.0759] | 0.000 |
| Jaccard | val10 | 10 | +0.0399 | [+0.0218, +0.0638] | 0.004 |
| HD95 (voxels) | test20 | 20 | -3.0613 | [-6.4984, -0.4496] | 0.009 |
| HD95 (voxels) | val10 | 10 | -0.4783 | [-0.8123, -0.2192] | 0.004 |
| ASD (voxels) | test20 | 20 | -0.8908 | [-1.7329, -0.2592] | 0.000 |
| ASD (voxels) | val10 | 10 | -0.2152 | [-0.3711, -0.1028] | 0.002 |

Verdict (CI on test20 excludes 0 and val10 has the same sign): supported for dice, hd95. n = 20 test patients (10 val), one seed.

## 3. SliceAlign after training

Zero-initialised parameters (init = 0): `offset.2.weight` norm 0.9450, `offset.2.bias` norm 0.1344, `fuse.weight` norm 11.3469, `fuse.bias` norm 1.2523

Predicted offsets between adjacent slices on val10 (stage-3 cells, 1 cell = 8 native px; cap 4 cells): mean |offset| 5.6568 cells (45.254 px), p95 5.6569 cells, max 5.6569 cells; fraction of positions above 0.05 cell: 1.000.

Bypass test at inference (fusion weights zeroed, so SliceAlign is exactly the identity), val10:

| | trained | SliceAlign bypassed |
|---|---|---|
| FG Dice | 0.9070 | 0.8717 |
| HD95 mm | 3.27 | 7.46 |

This only measures reliance of the trained network on the module; it is not a retrained ablation (the rest of the network adapted to it). A real claim needs a `align = false` arm trained with the same recipe.


## Reading (interpretation, kept apart from the measurements)

- **Section 3 is a negative finding for the B2 module as an alignment mechanism.** The offsets are not close to zero and not informative: every position sits at the cap (|dy| = |dx| = 4 cells, magnitude 4 sqrt 2 = 5.657 cells = about 45 native px), so the `tanh` offset head is saturated and warps the neighbouring slices by a constant 32 px diagonal shift in every volume. SliceAlign therefore acts as a learned fusion of constant-shifted neighbour features, not as slice registration. The network relies on it (bypass costs 3.5 Dice points and 4.2 mm HD95), but that dependence does not show alignment helps; a retrained `align = false` arm with the same recipe is the only way to attribute anything to it. The shift-recovery auxiliary (injected shifts of up to 8 px) did not keep the offsets in a usable range in the final model.
- Sections 1 and 2 support the headline: Dice and the other metrics are computed correctly, and S0-slicessm beats T-M3HL on paired per-patient comparisons against both the official-path and the wrapper teacher, with test20 CIs excluding 0 and the same sign on val10. The official-path teacher is the stricter comparison (+0.021 Dice, CI [+0.008, +0.035]; HD95 -1.75 voxels). One seed, 20 test patients; student training uses the same data as T-M3HL's U set, so this is a comparison of two systems, not a significance test across seeds.
