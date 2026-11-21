# Parity and significance: student vs frozen T-M3HL (same evaluators)

Student: `S2arbM3` val10-selected epoch 79. Differences are student - teacher, per patient (ED/ES averaged), n = 20.
n = 20 patients, one seed: seed variance is not estimated. 'Supported' needs a bootstrap CI that excludes 0 and the same sign on val10.

## val10

| metric | mean diff | 95% bootstrap CI | Wilcoxon p |
|---|---|---|---|
| compat_dice | +0.0061 | [-0.0157, +0.0345] | 0.770 |
| compat_hd95_vox | -0.2026 | [-0.6306, +0.1873] | 0.322 |
| native_fg_dice | +0.0061 | [-0.0157, +0.0345] | 0.770 |
| native_fg_hd95_mm | -0.4209 | [-1.6769, +0.3993] | 0.770 |

Means: teacher compat Dice 0.8794, student 0.8855; native FG Dice teacher 0.8794, student 0.8855.

## test20

| metric | mean diff | 95% bootstrap CI | Wilcoxon p |
|---|---|---|---|
| compat_dice | +0.0167 | [+0.0021, +0.0323] | 0.064 |
| compat_hd95_vox | -2.6536 | [-5.4570, -0.4252] | 0.007 |
| native_fg_dice | +0.0167 | [+0.0021, +0.0323] | 0.064 |
| native_fg_hd95_mm | -4.9920 | [-10.1312, -1.0259] | 0.016 |

Means: teacher compat Dice 0.8710, student 0.8877; native FG Dice teacher 0.8710, student 0.8877.

## Verdict (test20 CI excludes 0 and val10 has the same sign)

- compat_dice: **supported**
- native_fg_dice: **supported**
- compat_hd95_vox: **supported**
- native_fg_hd95_mm: **supported**

## Caveat that changes how to read this (measured, not inferred)

The teacher here is scored through the 3D wrapper, i.e. on our percentile-normalised inputs with per-volume min-max applied afterwards. The same
T-M3HL checkpoint scored with the official h5 path (SSL4MIS min-max on raw intensity) gives test20 Dice **0.8847** (compat, voxel HD95 2.94), against
**0.871** through the wrapper. The student (0.8877 test20) is compared above with the *wrapper* teacher, so most of the +0.017 gap is the
normalisation handicap of the wrapper, not a student gain: against the official-path teacher the gap is about +0.003. Treat the supported verdict
as "supported against the wrapper teacher", and the claim "student beats M3HL" as not established. The same wrapper teacher produces the pseudo-labels
the student trains on, so the handicap is also a real property of the training signal. Evaluating the student against the official-path teacher with
paired per-patient statistics needs the official per-case table (`T_M3HL_OFFICIAL.json`, per-case Dice/HD95 in voxels) and the student's compat
per-patient table; this is a follow-up, not done here.
