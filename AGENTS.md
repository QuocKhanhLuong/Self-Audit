# Repository source of truth

Read `ACTIVE_PIPELINE.json` and `docs/METHOD.md` before changing the pipeline.
`train.py` is the only trainer and dispatches the explicitly declared Candidate
A or Candidate A-3D track. Keep the two schemas, geometry contracts and
checkpoint source inventories separate.

Both tracks use a dense labeled TRAIN patient budget for segmentation,
measurement and K1 utility targets. The default recipes keep image-only U
unused. The optional A-3D semi-supervised profile may read image-only TRAIN U
records for detached EMA consistency only: no U mask, U GT-derived SDF target
or K1 utility label is allowed. EMA SDF consistency is permitted, and no
U/VAL/TEST mask may influence training, label-subset
selection, checkpoint choice or stopping. Development metrics are read-only. A
teacher output is frozen for each complete labeled student rollout and the
student loss must not update teacher parameters; the semi profile's EMA target
is a separate detached target used only on U consistency.

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

The A-3D semi-supervised profile is opt-in via
`configs/candidate_a3d/semi_supervised.json`. It keeps the labeled loss and K1
rollout unchanged. Each logical update pairs one labeled-frame step with one
image-only U-patient visit; the U cycle visits every U patient once per cycle
and chooses one native cine frame cyclically for that visit, while labeled
frames repeat through balanced cycles. `steps_per_epoch: 0` resolves to
`max(number_of_labeled_frames, number_of_TRAIN_patients)` (80 on the ACDC
protocol), subject to covering every labeled frame and U patient, and that
logical-update count is held fixed across few-shot `k`.
For original M&Ms, the plan records its own resolved labeled-patient count;
the ACDC value is not silently reused.
Weak/strong labeled and U views use intensity-only gamma/voxel-noise
perturbations, preserving native geometry. Report logical sample updates and
optimizer updates separately. Every arm, including `teacher_free`, retains its
student EMA helper; `teacher_free`/`teacher_frozen` remove the dual-branch
teacher U objective by factor. EMA state is stage-local and reset at a new
pretrain/warm-up/joint stage; interrupted resume is not implied.

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
