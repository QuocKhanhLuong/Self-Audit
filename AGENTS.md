# Repository source of truth

Read `ACTIVE_PIPELINE.json` and `docs/METHOD.md` before changing the pipeline.
`train.py` is the only trainer and dispatches the explicitly declared Candidate
A or Candidate A-3D track. Keep the two schemas, geometry contracts and
checkpoint source inventories separate.

Both tracks use a dense labeled TRAIN patient budget. All teacher targets,
measurement targets and K1 utility labels count inside that budget. U images are
currently unused. No U/VAL/TEST mask may influence training, target generation,
label-subset selection, checkpoint choice or stopping. Development metrics are
read-only. A teacher output is frozen for each complete student rollout and the
student loss must not update teacher parameters.

The decoder is a code-only boundary: serialized IDs, finite quantized values,
query coordinates and declared public geometry are its complete inputs. Do not
introduce a feature concatenation, teacher-feature skip, raw-image bypass or
unbounded float vector and call it Candidate A/A-3D. K1 must trial quantized
student values with one decoder state, account for incremental bytes and allow
STOP from step zero.

Track A uses `[B,3,H,W]` and 2D in-plane physical distances. Track A-3D uses
`[B,1,D,H,W]`, native full-volume frames, `shape_zyx`, runtime `spacing_zyx` and
`affine_zyx`; its signed distance is anisotropic 3D. The reference A-3D path
does not crop or resample. A crop, unknown unit, unsupported axis or missing
geometry must fail closed. A-3D is not a renamed Conv2d model.

ACDC and original M&Ms class mapping, ED/ES, source units, affine and patient
splits are explicit in the preparation manifests. Preserve the existing native
2D/2.5D track as its own comparison. Use the same declared labels, seeds, model
capacity, budget and protocol within each ablation factor.

Before delivery run `python scripts/check_repo.py` and `python -m pytest -q`,
plus the relevant synthetic smoke. Report exact test output, source and weight
hashes, artifact paths, and missing real-data/GPU evidence. Never claim a target
Dice/SOTA/novelty result from a fixture or an unverified checkpoint. Do not run
full patient training, download/accept licenses, alter active jobs, delete local
datasets, or use `git clean -fdx` around user data.

The old joint_guard/EMA, frozen-Q/CineMA and pre-cleanup trees are historical.
The pre-cleanup reference is `f01add12524cf73bb15675ecfeadc1014e420210`; frozen-Q
history is `archive/pre-semisupervised-20261020` at
`924265f7df604d3dca84dda49578b1a0cf28d821`. Do not mix their checkpoints into
either current track.
