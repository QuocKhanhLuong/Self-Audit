# Q3D pixdim amendment — 2026-11-12

The user approved `acdc_pixdim_native_v1`, rechecking, and subsequent training. [Original Drive audit](../q3d-original-drive-audit-20261112/README.md): all 150 CINEs match the existing server files. Original NIfTI files and arrays are unchanged. Unknown units have explicit ACDC mm declarations. Reconstructed native affine scale is not independent verification of scanner-world pose.

## Manifest repair

All six corrected views preserve every non-geometry JSON value, including record order, IDs, image/mask hashes and labels. [Amendment and exact SHA list](geometry/MANIFEST_AMENDMENT.json); [prior restoration lineage](geometry/RESTORATION_LINEAGE.json). Earlier restoration had already relocated paths, reordered K8 U and removed fewshot metadata; those changes are not attributed to this geometry amendment.

- K5 TRAIN: `305b88f403f62798f4daa725e2b16035e13b0b35bcefb2a56ff60c555750a3c1`.
- K8 TRAIN: `60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433`.
- K10 TRAIN: `482f0d60b587fe6e377958a021717114bebb79fcc48ee39d8209e5ef6c67620a`.
- DEV18 in all three directories: `c83031b6deaf275f830401f8da9ba9a0e30530620633106a3a62cc6d492b1f7b`.

TRAIN80 reference is a byte-identical alias of corrected K10 TRAIN; it preserves 10 L + 70 U, not 80 labeled patients. New manifests are siblings of their sources so relative array paths remain valid. L5 ⊂ L8 ⊂ L10 reproduces the original seed42 selector. External hash-pinned historical/source manifests restore selector evidence without editing restored manifest metadata.

**A-3D caveat:** historical A-3D K8 runs used the old geometry; SDF/mm-dependent results may be affected. No baseline rerun or baseline source/config edit was made.

## Geometry and phase

Median TRAIN ZYX spacing=(10,1.5625,1.5625) mm, median z=10 mm, anisotropy=6.4, median shape=(9,256,216). Pinned nnU-Net topology/min-feature-map4 yields strides `[(1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2),(1,2,2)]`. Kernels remain 3³, padding multiple=(2,32,32); no crop/resampling.

Canonical CPU probe: same 16 seeds42–57, logical shape(33,97,97), replicate pad, stage3 cell(8,6,6), signed mean256channels, threshold2voxels. Signed mean offset ZYX=(0.6737596988677979,1.7588436603546143,1.2107672691345215): PASS on every axis, so step=(2,8,8), offset=(0,0,0), zero correction. Max individual3.921062469482422 remains diagnostic only.

[Immutable receipt](erf_probe.json), SHA `36f44d7f85cd3872168b01339353fb17787099d309e167ea5d64ffbc1721b72b`; [separate lock](GEOMETRY_LOCK.json). ERF is SOFTWARE_ONLY random-input architecture evidence.

## L8 GT and OPEN review

[L8 diagnostics](L8_GEOMETRY_DIAGNOSTICS.json) read only 16 supplied L8 GT frames. All eight volume/EF guards pass, LV ED=95.0439453125–309.7265625 mL. This is a unit/guard check, not a clinical normality claim; U/DEV/TEST masks were not read.

**[OPEN] MYO:** 245/29,664 phase-correct S4 cells win MYO (0.825916936%), below the unchanged2% review trigger. Independent review completed before P2: 245 cells do not make four codes mathematically impossible, but do not prove learned teacher-MYO code coverage. Q anatomy CE averages cells and remains vulnerable to rare-class dilution; codeKD balances active classes. Retain design and weights; no threshold is relaxed. Actual Q must still meet ≥4 codes per foreground class, perplexity≥32 and dead-code≤25%, or stop. GT support is not a substitute for the teacher/codebook gate.

## Execution boundary

[Software receipt](SOFTWARE.md) records the earlier605-test pass and all-arm smoke. Updated pins/selector integration will have separate final verification. GPU profile and patient T/Q/P/J are NOT_RUN at this lock. Authorized order: K8V1 → K5V1 → K10V1 → K8V1-u/C1/C0. T-only20-update profile precedes production K8T/Q/P; J20-update profile requires genuine T/Q/P artifacts and precedes productionJ. No fake codebook or real gate override is permitted.
