# SliceSSM: results for the paper (status 2026-11-24 19:30 UTC)

This file is the hand-off for writing the paper. Every number below comes from a JSON file in `results/`. The files are
copied verbatim from the server at `/root/mount_data/runs/<run>/<arm>/{FINAL,TEST20}_RESULT.json`. The paired statistics
are in `stats_paired.json`, which was recomputed from the per-patient records in those files. Implementation details,
size and cost are in `README.md`.

One job had not finished at the time of writing: LA 2018 at 5%. It is listed in
the last section. Do not report numbers for them until their `TEST20_RESULT.json` exists.

## 1. Method in one paragraph

SliceSSM is a 2.53 M-parameter student (deployed) for anisotropic short-axis cardiac MRI. It has three parts on top of an
in-plane encoder/decoder (kernels 1×3×3, no z down-sampling; the CineMA-distilled S1 initialisation):

- **PhysZSSM.** A bidirectional selective state-space scan along z for every in-plane location, with 1 block at stage 3
  and 2 blocks at stage 4. The discretisation step is scaled by the physical slice spacing:
  `Δ = softplus(dt(x)) · s_z / 10 mm`. Padded slices get Δ = 0.
- **SliceAlign.** At stage 3, predicts in-plane offsets between adjacent slices, warps both neighbours onto each slice and
  fuses them residually. Zero-initialised, so it starts as the identity.
- **Shift-recovery auxiliary.** Known per-slice in-plane shifts are injected into the input, and the predicted
  inter-slice offsets must change by exactly the injected relative shift. The clean-view offsets are detached. This
  supervises SliceAlign without labels. Weight 0.1.

Training is UniMatch-style semi-supervised self-training with no teacher, identical for every arm:
- pseudo-labels from the weak view with τ = 0.95;
- two strong views with full-depth CutMix;
- a feature-perturbation stream (dropout3d on the skips);
- stream weights 0.25 / 0.25 / 0.5;
- 800 labelled-only warm-up updates, then λ_u ramped over 10% of training;
- 150 epochs.

## 2. Protocol

| Dataset | Labels | L / U / val / test (patients) | Notes |
|---|---|---|---|
| ACDC | 10% | 7 / 63 / 10 / 20 | SSL4MIS split: the first 136 slices of the pinned `train_slices.list`; ED + ES |
| ACDC | 5% | 3 / 67 / 10 / 20 | first 68 slices; the removed labelled cases join U as images only |
| M&Ms-1 | 10% | 15 / 135 / 34 / 136 | official challenge split (Training-Labeled / Validation / Testing); ED + ES; classes remapped to ACDC order |
| M&Ms-1 | 5% | 7 / 143 / 34 / 136 | strict subset of the 10% labelled set |
| LA 2018 | 10% / 5% | 7 / 63 / 10 / 20 and 3 / 67 / 10 / 20 | UA-MT h5; our val10 kept, so 1 labelled and 10 unlabelled cases fewer than BCP/UA-MT, which use no validation set |

**Model selection.** The checkpoint with the best foreground Dice on the validation set (val10 or val34) was selected
before test was opened. Test was opened exactly once per run, by `w2s_open_test20.py`, which writes `TEST20_RESULT.json`
once and refuses to rewrite it. Test played no role in any choice.

**Metrics.**
- Dice, Jaccard, HD95 and ASD, averaged over RV, MYO and LV (LA: one class).
- "mm" columns: the repository evaluator. ASD there is the symmetric ASSD.
- "vox" columns: SSL4MIS/medpy conventions, with one-directional ASD in voxels, the convention most SSL papers report.
- Dice and Jaccard are in %.
- "@ep150": the last checkpoint, given for robustness. It was not used for selection.

**Seeds.** Seed 42 unless stated. Seed 43 replicates change only the seed of the semi-supervised stage S0. The S1
initialisation is shared.

## 3. Main result: ACDC 10% (test20)

| Model | Params (deploy) | val10 | Dice | Jaccard | HD95 mm | ASD mm | HD95 vox | ASD vox |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SliceSSM (seed 42)** | 2.53 M | 90.70 | **90.56** | **83.18** | **3.37** | **0.75** | **1.19** | **0.28** |
| SliceSSM (seed 43) | 2.53 M | 90.90 | 90.56 | 83.27 | 3.93 | 0.90 | 1.19 | 0.31 |
| SliceSSM mean of 2 seeds | | 90.80 | 90.56 | 83.23 | 3.65 | 0.83 | 1.19 | 0.30 |
| SliceSSM + KD from M3HL teacher | 2.53 M | 90.69 | 90.29 | 82.84 | 3.91 | 0.89 | 1.21 | 0.29 |
| AnisoStudent (z-transformer), same recipe | 2.72 M | 90.14 | 89.58 | 81.78 | 6.20 | 1.41 | 2.69 | 0.79 |
| UNETR++-lite student, same recipe | 3.07 M (13.3 M train) | 87.30 | 88.14 | 79.33 | 4.33 | 1.04 | 1.27 | 0.37 |
| Earlier W2S pipeline, best arm: S2-arb M3HL, 150 epochs | 3.07 M | 89.17 | 89.36 | — | 4.28 | — | — | — |
| Earlier W2S pipeline: S1-kd M3HL | 3.07 M | 89.13 | 89.27 | — | 4.43 | — | — | — |

Per class for SliceSSM seed 42 (Dice / Jaccard / HD95 mm / ASD mm):
- RV 90.20 / 82.68 / 4.34 / 0.98
- MYO 88.08 / 78.90 / 2.70 / 0.57
- LV 93.40 / 87.96 / 3.06 / 0.70

Paired per-patient tests on test20, seed 42, n = 20. Values are SliceSSM minus the other model, with a bootstrap 95% CI
and the Wilcoxon p-value:

| SliceSSM vs | ΔDice (points) | p | ΔHD95 mm | p |
|---|---:|---:|---:|---:|
| UNETR++-lite | +2.42 [1.62, 3.24] | <0.001 | −0.97 [−1.66, −0.31] | 0.016 |
| AnisoStudent | +0.98 [−0.10, 2.30] | 0.15 | −2.83 [−8.62, 0.33] | 0.43 |
| SliceSSM + KD from M3HL | +0.27 [−0.35, 0.89] | 0.39 | −0.54 [−1.06, −0.06] | 0.036 |

Published reference points at ACDC 10%:
- UniMatch v1 (CVPR 2023): Dice 89.90 at 10%, 88.90 at 5%.
- BCP (CVPR 2023): at 10%, Dice 88.84, Jaccard 80.62, HD95 3.98, ASD 1.17; at 5%, 87.59 / 78.67 / 1.90 / 0.67.
  The HD95 and ASD are in SSL4MIS voxel units.

These come from the Baselines tab of the project sheet and must be checked against the papers. Our figures for ABD
(about 89.8), OFD-Net (about 90.0) and IPNet (about 90.6) were noted from memory and are **not verified**. Look them up
before citing them.

## 4. Component ablation: ACDC 10%, two seeds

Arms change one thing each, starting from the full model:

| Arm | PhysZSSM | Δ ∝ spacing | SliceAlign | Shift-aux |
|---|:-:|:-:|:-:|:-:|
| Full | ✓ | ✓ | ✓ | ✓ |
| A1 | ✓ | ✗ | ✓ | ✓ |
| A2 | ✓ | ✓ | ✗ | ✗ |
| A3 | ✓ | ✓ | ✓ | ✗ |
| A4 | ✗ | — | ✓ | ✓ |
| A5 | ✗ | — | ✗ | ✗ |

Test20 results; selection on val10. A1 has only seed 42.

| Arm | val10 s42 / s43 | Dice s42 / s43 | **Dice mean** | HD95 mm s42 / s43 | **HD95 mean** | ASD mm mean | Dice @ep150 mean |
|---|---|---|---:|---|---:|---:|---:|
| Full | 90.70 / 90.90 | 90.56 / 90.56 | 90.56 | 3.37 / 3.93 | **3.65** | **0.83** | 90.46 |
| A1 | 90.54 / – | 90.12 / – | 90.12 (1 seed) | 3.90 / – | 3.90 | 0.89 | 90.13 |
| A2 | 90.69 / 90.71 | 90.32 / 90.23 | 90.28 | 3.54 / 5.21 | 4.38 | 0.93 | 90.27 |
| A3 | 90.68 / 90.84 | 90.64 / 90.12 | 90.38 | 4.24 / 4.64 | 4.44 | 0.99 | 90.25 |
| A4 | 90.15 / 89.95 | 91.18 / 91.03 | **91.11** | 5.28 / 6.83 | 6.06 | 1.15 | 90.78 |
| A5 | 89.71 / 90.03 | 90.28 / 89.52 | 89.90 | 8.26 / 9.25 | 8.76 | 1.79 | 90.09 |

Paired tests pooled over both seeds (40 pairs = 2 seeds × 20 patients):

| Comparison | Component | ΔDice (points) | p | ΔHD95 mm | p |
|---|---|---:|---:|---:|---:|
| A4 − A5 | SliceAlign + shift-aux, without z-SSM | **+1.20 [0.54, 1.96]** | **<0.001** | **−2.70 [−5.85, −0.32]** | **0.011** |
| Full − A4 | PhysZSSM, on top of the line above | −0.54 [−1.09, 0.05] | 0.019 | −2.41 [−6.50, 0.49] | 0.20 |
| Full − A2 | SliceAlign + shift-aux, with z-SSM | +0.28 [−0.15, 0.78] | 0.32 | −0.73 [−2.43, 0.35] | 0.98 |
| Full − A3 | shift-aux alone | +0.18 [−0.28, 0.64] | 0.27 | −0.79 [−1.82, 0.00] | 0.32 |
| Full − A5 | all three | +0.66 [−0.30, 1.74] | 0.48 | −5.10 [−11.24, −0.54] | 0.30 |

Seed 42 only (20 pairs): Full − A1, the physical step, gives ΔDice +0.44 [−0.19, 1.03] (p 0.18) and ΔHD95 −0.53 mm
[−1.05, −0.02] (p 0.09).

**What the ablation supports. Write the claims to match this.**
1. **Aligned slice fusion with self-supervised shift recovery is the component with a significant effect on both Dice
   and boundary error:** +1.2 Dice, −2.7 mm HD95, both p ≤ 0.011, measured without the z-SSM.
2. **PhysZSSM trades Dice for robustness.**
   - It lowers Dice by about 0.5 points (p = 0.019).
   - It lowers mean HD95 by 2.4 mm and ASD by 0.3 mm. This is not significant, because the gain comes from removing a
     few gross 3D failures (basal and apical slices) rather than from improving every case: A4 has the better HD95 in
     23 of 40 pairs, yet the worse mean.
   - Across the two seeds, the full model has the lowest mean HD95 and ASD of all arms.
3. **When PhysZSSM is present, SliceAlign and shift-aux add little** (+0.2–0.3 Dice, about −0.75 mm HD95, not
   significant). The two mechanisms overlap.
4. **The physical step helps slightly in one seed** (−0.5 mm HD95, p ≈ 0.09).
5. **Validation and test disagree on Full versus A4.** val10 ranks Full above A4 in both seeds. test20 ranks A4 higher
   on Dice. With n = 10, val10 cannot separate differences of about 0.5 Dice. The model was chosen on validation, so
   Full is the reported model. Do not switch to A4 on the basis of test.
6. **Voxel versus mm metrics.** Voxel-unit HD95 and ASD count an error of one 5–10 mm slice the same as one ~1.5 mm
   pixel, so they under-weight exactly the through-plane errors this architecture targets. Report mm as primary and
   voxels for comparability.

### 4b. Follow-up: token-gated PhysZSSM (negative result)

Motivated by item 2 above, a gate was added to every z-SSM residual:
`x + sigmoid(W·norm(x) + b) ⊙ Δ`, per token and channel, with W = 0 and b = 2 at initialisation. This adds 99.9 K
parameters (2.65 M in total) and reuses the full model's S1 artifact. Everything else in the recipe is identical.
This arm was designed and evaluated **after** test20 results of the ablation had been seen. The decision rule was fixed
before its own test was opened: keep it only if its 2-seed mean test Dice beats the full model and its mean HD95 is at
most 0.5 mm worse.

| | val10 s42 / s43 | Dice s42 / s43 | Dice mean | HD95 mm s42 / s43 | HD95 mean | ASD mm mean | Dice @ep150 s42 / s43 |
|---|---|---|---:|---|---:|---:|---|
| Full | 90.70 / 90.90 | 90.56 / 90.56 | **90.56** | 3.37 / 3.93 | **3.65** | 0.83 | 90.66 / 90.26 |
| Gated (G) | 90.87 / 91.05 | 89.81 / 90.46 | 90.14 | 4.16 / 3.77 | 3.97 | 0.91 | 90.06 / 90.29 |

**Outcome: rejected. The full model is kept.**
- The gated variant has the highest validation Dice of all arms in both seeds, but a lower test Dice.
- On validation it plateaus from about epoch 90, while the full model is still improving. Mean val Dice over epochs
  121–150: G 89.87 / 90.22 versus Full 90.47 / 90.66.
- Its selected epochs (82 and 72) are early validation spikes.
- This is a second instance of val10 (n = 10) failing to rank close variants. Validation does not decrease late in
  training for any arm, so this is not classical overfitting.
- Raw results: `results/slicessm-gated-20261124/`.

## 5. Width ablation (S / M / L): ACDC 10%, seed 42

Same recipe. Each width has its own CineMA S1 distillation.

| Size | Widths | Params (train) | val10 | Dice | Jaccard | HD95 mm | ASD mm | HD95 vox | ASD vox | S0 time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S | M × 2/3 | 1.21 M | 89.92 | 89.60 | 81.82 | 5.44 | 1.32 | 1.96 | 0.59 | 1.74 h |
| **M** | 24-48-96-160-192 | 2.55 M | 90.70 | **90.56** | **83.18** | **3.37** | **0.75** | **1.19** | **0.28** | – |
| L | M × 1.5 | 5.69 M | 90.77 | 90.40 | 83.06 | 5.25 | 1.27 | 2.08 | 0.64 | 2.08 h |

M versus S: +0.96 Dice (p 0.076), −2.07 mm HD95 (p 0.035). M versus L: +0.16 Dice (p 1.0), −1.88 mm HD95 (p 0.62).
Scaling up to 5.7 M parameters does not help, so the 3 M budget is not limiting.

## 6. Label ratio and datasets: full model, seed 42

| Dataset | Labels | Test | val | Dice | Jaccard | HD95 mm | ASD mm | HD95 vox | ASD vox |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ACDC | 10% | test20 | 90.70 | 90.56 | 83.18 | 3.37 | 0.75 | 1.19 | 0.28 |
| ACDC | 5% | test20 | 87.00 | 87.65 | 78.76 | 5.74 | 1.35 | 1.88 | 0.55 |
| M&Ms-1 | 10% | test136 | 83.90 | 82.96 | 72.32 | 11.98 | 2.69 | 7.71 | 2.37 |
| M&Ms-1 | 5% | test136 | 80.06 | 77.63 | 65.70 | 16.20 | 3.93 | 7.40 | 2.23 |
| LA 2018 | 10% (L7) | test20 | 90.93 | 90.91 | 83.68 | 4.59 | 1.23 | 7.34 | 2.34 |

LA has a single foreground class, stored in slot 3; the row reports the slot-3 metrics. Do not use the
three-class `ssl4mis_compat.mean` in the LA JSON: it averages in two empty classes and reads 30.3. LA voxels are
0.625 mm isotropic (declared), so a voxel-unit HD95 is the mm value divided by 0.625. LA uses one labelled case and ten
unlabelled cases fewer than the BCP/UA-MT 8-labelled setting, because our val10 is held out. Published LA 10% numbers
from those papers are therefore not directly comparable. Look them up and state this difference.

Per class (Dice / HD95 mm):
- ACDC 5%: RV 86.11 / 8.88, MYO 85.17 / 4.28, LV 91.67 / 4.05.
- M&Ms 10%: RV 80.56 / 19.54, MYO 80.41 / 7.29, LV 87.90 / 9.10.
- M&Ms 5%: RV 72.07 / 21.79, MYO 77.52 / 12.14, LV 83.31 / 14.66.

M&Ms uses the official challenge split. Its test set contains vendors C and D, which are absent or rare in the labelled
training set, so this is semi-supervised learning under vendor shift. No published semi-supervised method reports on
this exact split that we know of, so baselines must be run on the same split for a fair comparison. They have not been
run yet. Most of the error is on RV, especially basal slices.

## 7. Cost

| | Value |
|---|---|
| Deployed parameters | 2,531,734 (2,553,001 with train-only heads) |
| Speed and memory | 0.118 s/iter, 1.48 GB peak (RTX 5000 Ada, bf16, 1×1×10×256×224, measured with other jobs running). AnisoStudent: 0.069 s/iter. |
| Semi-supervised stage (S0), ACDC 10% | about 1.5–2 h for 150 epochs |
| Semi-supervised stage (S0), M&Ms 10% | about 9.4 h |
| Shared CineMA S1 distillation | about 3 h per dataset |

## 8. Not finished at the time of writing (fill in later; do not estimate)

- **LA 2018, 5% (L3 / U67).** Running from about 19:25 UTC on 2026-11-24, about 7 h. (LA 10% is done; see section 6.
  The gated variant is done; see section 4b.)
- **Single seed only:** A1 and the S / L widths.
- **Baselines on M&Ms and LA with the same splits:** not run.

## 9. File map

| What | Where (`results/`) |
|---|---|
| Full model, seeds 42 / 43 | `slicessm-screen-20261122/S0slicessm`, `slicessm-ablation-s43-20261124/S0slicessms43` |
| Ablations, seed 42 / 43 | `slicessm-ablation-20261123/S0ssA{1..5}`, `slicessm-ablation-s43-20261124/S0ssA{2..5}s43` |
| SliceSSM + KD from M3HL | `slicessm-screen-20261122/S1kdM3ss` |
| Width S / L | `slicessm-sml-20261123/S0slicessm{S,L}` |
| ACDC 5%, M&Ms 10% / 5%, LA 10% | `slicessm-acdc-5pct-20261123`, `slicessm-mnms1-20261123`, `slicessm-mnms1-5pct-20261123`, `slicessm-la2018-20261123` |
| AnisoStudent, UNETR++-lite | `aniso-screen-20261122/S0aniso`, `aniso-screen-20261122/S0unet150` |
| Earlier W2S arms (M3HL teacher) | `m3hl-teacher-20261121/*`, `w2s-ssl4mis-20261120/*` (FINAL_RESULT only) |
| Paired tests | `stats_paired.json` |
| Code | `self_audit/candidate_pg/slicessm_student.py`; configs via `scripts/w2s_make_configs_ssl4mis.py` (arms `S0slicessm`, `S0ssA1..5`, `S0ssG`) |
